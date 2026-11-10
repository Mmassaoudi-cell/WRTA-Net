"""Data handling and metrics for the learning experiments."""
from __future__ import annotations
import os
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
W = 12
RESID_SRC = os.environ.get("RESID_SRC", "pmu")      # front-end of the learned detectors: "pmu" (default, final protocol) or "hyb"
CLASSES = ["Normal", "MDDA", "LT-RDCA", "ST-RDCA"]


class Pool:
    """One scenario pool of one grid. Whitened residual R=(z - zhat_HYB)/sigma_nominal (physics front-end)."""

    def __init__(self, case: str, tag: str):
        d = np.load(os.path.join(DATA, f"{case}_{tag}.npz"))
        g = np.load(os.path.join(DATA, f"{case}_grid.npz"))
        self.case, self.tag = case, tag
        self.sigma = g["sigma"].astype(np.float32)
        self.thr = float(g["thr"])
        self.kind, self.loc = g["kind"], g["loc"]
        self.f, self.t = g["f"], g["t"]
        self.nb = int(g["nb"]); self.nl = int(g["nl"])
        self.z = d["z"]; self.zhyb = d["zhyb"]; self.zpmu = d["zpmu"]; self.lnr = d["lnr"]; self.chi = d["chi"]
        self.mask = d["mask"]; self.cls = d["cls"]
        self.tflag = d["tflag"] if "tflag" in d.files else np.zeros(d["z"].shape, np.float32)
        self.meta = {k[2:]: d[k] for k in d.files if k.startswith("m_")}
        self.N, self.T, self.m = self.z.shape
        self.Rh = ((self.z - self.zhyb) / self.sigma).astype(np.float32)          # hybrid-SE residual (source pipeline: WLS + HYB-SE)
        self.Rp = ((self.z - self.zpmu) / self.sigma).astype(np.float32)          # PMU-only LSE residual (non-iterative front-end)
        self.R = self.Rp if RESID_SRC == "pmu" else self.Rh                       # residual used by all learned models
        self.vuln = g["vuln"]
        self.L = clone_feature(self.z / self.sigma, scale=robust_scale(self.R))

    def raw(self, mu=None, sd=None):
        """No-physics variant: standardised raw RTU window (stats from train)."""
        if mu is None:
            mu = self.z.reshape(-1, self.m).mean(0); sd = self.z.reshape(-1, self.m).std(0) + 1e-6
        return ((self.z - mu) / sd).astype(np.float32), mu, sd

    def idx_pairs(self):
        n, t = np.meshgrid(np.arange(self.N), np.arange(self.T), indexing="ij")
        return n.ravel(), t.ravel()


def clone_feature(R, P=W - 1, cap=12.0, scale=None):
    """Clone-fingerprint (lag-agreement) channel: min over lags p<=P of |r(t)-r(t-p)| / robust scale, capped.
    Replayed payloads repeat earlier RTU values; exact repeats make the minimum collapse to ~0."""
    N, T, m = R.shape
    s = np.maximum((robust_scale(R) if scale is None else scale)[..., None], 0.3)
    L = np.full((N, T, m), cap, np.float32)
    for tt in range(1, T):
        best = np.full((N, m), np.inf, np.float32)
        for p in range(1, min(P, tt) + 1):
            best = np.minimum(best, np.abs(R[:, tt] - R[:, tt - p]))
        L[:, tt] = np.minimum(best / s[:, tt], cap)
    return L


def pad_series(X: np.ndarray, w: int = W) -> np.ndarray:
    """(N,T,m) -> (N,T+w-1,m) replicate-padded at the start so that window(t)=Xp[:,t:t+w]."""
    first = np.repeat(X[:, :1], w - 1, axis=1)
    return np.concatenate([first, X], axis=1)


def windows(Xp: np.ndarray, n, t, w: int = W):
    """Gather windows for sample pairs -> (B,w,m)."""
    off = np.arange(w)
    return Xp[n[:, None], t[:, None] + off[None, :]]


def robust_scale(R_t):
    """Robust noise-scale estimate per time step from whitened residuals (median absolute value / 0.6745)."""
    return np.median(np.abs(R_t), axis=-1) / 0.6745


# ----------------------------------------------------------------------------------------------- metrics
def task_a_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], zero_division=0)
    out = dict(macro_f1=float(f1.mean()), weighted_f1=float(f1_score(y, p, average="weighted", zero_division=0)),
               bal_acc=float(balanced_accuracy_score(y, p)), acc=float((y == p).mean()))
    for k, c in enumerate(CLASSES):
        out[f"P_{c}"], out[f"R_{c}"], out[f"F1_{c}"] = float(pr[k]), float(rc[k]), float(f1[k])
    out["minority_recall"] = float(rc[3])                         # ST-RDCA = rarest class
    out["attack_recall_cls"] = float(rc[1:].mean())
    # window-level detection derived from the class prediction (attack present or not)
    yb, pb = (y > 0), (p > 0)
    tp = (yb & pb).sum(); fp = (~yb & pb).sum(); fn = (yb & ~pb).sum(); tn = (~yb & ~pb).sum()
    out["det_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1)); out["det_fpr"] = float(fp / max(fp + tn, 1)); out["det_fnr"] = float(fn / max(fn + tp, 1))
    return out, confusion_matrix(y, p, labels=[0, 1, 2, 3])


def task_b_metrics(mask, flag=None, score=None, thr=None, auc=True):
    """Measurement-level attacked-vs-clean metrics. mask: (...) bool, flag: bool decisions or score + thr."""
    y = mask.ravel().astype(bool)
    if flag is None:
        flag = score.ravel() >= thr
    else:
        flag = flag.ravel().astype(bool)
    tp = int((y & flag).sum()); fp = int((~y & flag).sum()); fn = int((y & ~flag).sum()); tn = int((~y & ~flag).sum())
    f1_att = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_cln = 2 * tn / max(2 * tn + fp + fn, 1)
    out = dict(TP=tp, FP=fp, FN=fn, TN=tn, tpr=tp / max(tp + fn, 1), tnr=tn / max(tn + fp, 1), fpr=fp / max(fp + tn, 1),
               fnr=fn / max(fn + tp, 1), prec=tp / max(tp + fp, 1), f1_att=f1_att, macro_f1_b=(f1_att + f1_cln) / 2,
               acc_b=(tp + tn) / len(y), bal_acc_b=0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
    if auc and score is not None:
        s = score.ravel()
        if len(s) > 3_000_000:                       # deterministic thinning for speed (keep all positives)
            rs = np.random.default_rng(0)
            keep = y | (rs.random(len(y)) < 3_000_000 / len(y) * 1.0)
            w = np.where(y, 1.0, len(y) / max(keep.sum(), 1))
            ys, ss = y[keep], s[keep]
        else:
            ys, ss = y, s
        out["pr_auc"] = float(average_precision_score(ys, ss)) if ys.any() else float("nan")
        out["roc_auc"] = float(roc_auc_score(ys, ss)) if ys.any() and (~ys).any() else float("nan")
    return out


def f1_best_threshold(mask, score):
    """Validation-only threshold choice: maximise attacked-class F1 (negatives thinned with importance weights)."""
    y = mask.ravel().astype(bool); s = score.ravel()
    wneg = 1.0
    if len(s) > 2_000_000:
        pk = 2_000_000 / len(s)
        rs = np.random.default_rng(1); keep = (rs.random(len(s)) < pk) | y
        y, s = y[keep], s[keep]; wneg = 1.0 / pk
    order = np.argsort(-s)
    ys = y[order]; ss = s[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys) * wneg
    f1 = 2 * tp / np.maximum(tp + fp + ys.sum(), 1)
    return float(ss[int(np.argmax(f1))])


def save_scen(path, pool, pa, pm, thr):
    """Per-scenario sufficient statistics used for scenario-level bootstrap and operational (alarm) metrics.
    pa: (N*T,) class predictions or None; pm: (N,T,m) meter scores."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flag = pm >= thr
    y = pool.mask
    cnt = np.stack([(flag & y).sum((1, 2)), (flag & ~y).sum((1, 2)), (~flag & y).sum((1, 2)), (~flag & ~y).sum((1, 2))], 1).astype(np.int64)
    d = dict(cnt=cnt, nflag=flag.sum(2).astype(np.int16), nhit=(flag & y).sum(2).astype(np.int16), cls=pool.cls.astype(np.int8),
             t_a=np.asarray(pool.meta["t_a"]), kind=np.asarray(pool.meta["kind"]))
    if pa is not None:
        d["pa"] = np.asarray(pa).reshape(pool.N, pool.T).astype(np.int8)
    np.savez_compressed(path, **d)
