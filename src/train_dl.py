"""Generic trainer/evaluator for the neural models (baselines + proposed candidates), v2 protocol.

Input tensor (N,T,m,5): channel 0 = whitened residual, 1 = clone feature, 2 = Stage-1 vulnerability bit, 3 = log robust noise scale,
4 = topology-processor flag.  Every learned model receives all five channels (equal information)."""
from __future__ import annotations
import time, json, os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nets
from mldata import save_scen, Pool, W, pad_series, task_a_metrics, task_b_metrics, f1_best_threshold, CLASSES, robust_scale

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCH = 5
POOL_ARRAYS = ("z", "zhyb", "zpmu", "lnr", "chi", "mask", "cls", "R", "Rh", "Rp", "L", "tflag")


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_input(pool: Pool, src="pmu", stats=None):
    st = None
    if src == "hyb":
        X = pool.Rh
    elif src == "pmu":
        X = pool.Rp
    elif src == "raw":
        X, mu, sd = pool.raw(*(stats if stats else (None, None)))
        st = (mu, sd)
    else:
        raise ValueError(src)
    N, T, m = X.shape
    isv = np.zeros(m, np.float32); isv[pool.vuln] = 1
    logs = np.log(robust_scale(pool.Rp if src != "hyb" else pool.Rh) + 1e-3).astype(np.float32)          # (N,T)
    out = np.stack([X, pool.L, np.broadcast_to(isv, (N, T, m)), np.broadcast_to(logs[..., None], (N, T, m)), pool.tflag], -1)
    return out.astype(np.float32), st


class Adapter(nn.Module):
    """Feeds the (B,W,m,5) input to a base model according to its input mode."""
    def __init__(self, base, mode, nch):
        super().__init__()
        self.base, self.mode, self.nch = base, mode, nch

    def forward(self, x, ctx=None):
        if self.mode == "stack":
            return self.base(x)
        planes = [x[..., c] for c in range(self.nch)]
        return self.base(torch.cat(planes, 1 if self.mode == "time" else 2))


def vbit_of(tr):
    isv = np.zeros(tr.m, int); isv[tr.vuln] = 1
    return isv


def make_model(name, tr: Pool, cfg: dict):
    m = tr.m
    A = nets.build_adj(tr.kind, tr.loc, tr.f, tr.t)
    k = NCH if cfg.get("fair", True) else 2
    w = cfg.get("width")
    if name == "MLP": return Adapter(nets.MLP(m, W, h=w or 512, cin=m * k), "time", k)
    if name == "TCN": return Adapter(nets.TCN(m, W, h=w or 128, cin=m * k), "chan", k)
    if name == "GRU": return Adapter(nets.GRUNet(m, W, h=w or 128, cin=m * k), "chan", k)
    if name == "Transformer": return Adapter(nets.TemporalTransformer(m, W, d=w or 128, cin=m * k), "chan", k)
    if name == "FT-Transformer": return Adapter(nets.FTTransformer(m, W, d=w or 48, win=W * k), "time", k)
    if name == "GCN": return Adapter(nets.GraphBaseline(m, W, A, "gcn", d=w or 64, win=W * k), "time", k)
    if name == "GAT": return Adapter(nets.GraphBaseline(m, W, A, "gat", d=w or 64, win=W * k), "time", k)
    if name == "Mamba": return Adapter(nets.MambaLite(m, W, d=w or 96, cin=m * k), "chan", k)
    if name.startswith("WR"):
        v = dict(temporal=cfg.get("temporal", "ssm"), graph=cfg.get("graph", True), film=cfg.get("film", True),
                 d=cfg.get("d", 32), layers=cfg.get("layers", 2), glayers=cfg.get("glayers", 2), attn=cfg.get("attn", False), heads=cfg.get("heads", 4),
                 lag=cfg.get("lag", True), attn_bias=cfg.get("bias", True), prior=cfg.get("prior", True), prior_drop=cfg.get("prior_drop", 0.0))
        return Adapter(nets.WRNet(m, W, A, tr.kind, vbit_of(tr), **v), "stack", NCH)
    raise ValueError(name)


def subsample_pool(pool, frac, seed):
    """Random subset of training scenarios (low-data experiments)."""
    rs = np.random.default_rng(1234 + seed)
    k = max(8, int(round(frac * pool.N)))
    idx = np.sort(rs.choice(pool.N, k, replace=False))
    p = copy.copy(pool)
    for a in POOL_ARRAYS:
        setattr(p, a, getattr(pool, a)[idx])
    p.meta = {k_: v[idx] for k_, v in pool.meta.items()}
    p.N = k
    return p


class Gpu:
    """Holds a pool on the GPU as padded windows source."""
    def __init__(self, pool: Pool, X):
        self.Xp = torch.tensor(pad_series(X), device=DEV)
        self.cls = torch.tensor(pool.cls, device=DEV)
        self.mask = torch.tensor(pool.mask, device=DEV)
        self.N, self.T, self.m = pool.N, pool.T, pool.m

    def batch(self, n, t):
        off = torch.arange(W, device=DEV)
        return self.Xp[n[:, None], t[:, None] + off[None]]


@torch.no_grad()
def predict(model, g: Gpu, bs=256):
    model.eval()
    n, t = torch.meshgrid(torch.arange(g.N, device=DEV), torch.arange(g.T, device=DEV), indexing="ij")
    n, t = n.reshape(-1), t.reshape(-1)
    lc, lm = [], []
    for i in range(0, len(n), bs):
        c, mm = model(g.batch(n[i:i + bs], t[i:i + bs]))
        lc.append(c.float().cpu()); lm.append(torch.sigmoid(mm.float()).cpu())
    return torch.cat(lc).numpy(), torch.cat(lm).numpy().reshape(g.N, g.T, g.m)


def evaluate(model, g: Gpu, pool: Pool, thr_b=0.5, auc=True):
    lc, pm = predict(model, g)
    y = pool.cls.reshape(-1)
    ma, cm = task_a_metrics(y, lc.argmax(1))
    mb = task_b_metrics(pool.mask, score=pm, thr=thr_b, auc=auc)
    return ma, mb, cm, pm, lc


def feature_attack(model, x, mk, eps, steps=1, cw=None, pw=None, y=None, lam=1.0, norm="linf", rand_eps=False, rand_start=False):
    """Feature-space white-box attacker on the compromised meters (selected by mk): their clone feature is resampled from the clean-meter
    distribution (the attacker re-noises the replayed data) and their whitened residuals are shifted along the loss gradient.
    norm='linf': |delta|<=eps per entry; norm='l2': ||delta||_2 <= eps*sqrt(#entries).  rand_eps draws the budget per batch (training use)."""
    sel = mk[:, None, :].expand_as(x[..., 0])                                  # (B,W,m)
    if rand_eps:
        eps = float(np.random.uniform(0.25, 1.0) * eps)
    x_adv = x.clone()
    l = x[..., 1]
    clean_vals = l[(~mk)[:, None, :].expand_as(l)]
    if clean_vals.numel() > 0:
        samp = clean_vals[torch.randint(0, clean_vals.numel(), l.shape, device=l.device)]
        x_adv[..., 1] = torch.where(sel, samp, l)
    delta = (torch.empty_like(x[..., 0]).uniform_(-eps, eps) * sel) if rand_start else torch.zeros_like(x[..., 0])
    step = 2.5 * eps / max(steps, 1)
    n_ent = sel.sum((1, 2)).clamp(min=1).float()
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        xi = x_adv.clone(); xi[..., 0] = x[..., 0] + delta
        c, mm = model(xi)
        loss = F.cross_entropy(c, y, weight=cw) + lam * F.binary_cross_entropy_with_logits(mm, mk.float(), pos_weight=pw)
        g, = torch.autograd.grad(loss, delta)
        if norm == "linf":
            delta = (delta.detach() + step * g.sign() * sel).clamp(-eps, eps) * sel
        else:
            gn = (g * sel).flatten(1).norm(dim=1).clamp(min=1e-12)[:, None, None]
            delta = delta.detach() + step * (g * sel) / gn * n_ent.sqrt()[:, None, None] / 1.0
            dn = delta.flatten(1).norm(dim=1)[:, None, None]
            lim = eps * n_ent.sqrt()[:, None, None]
            delta = delta * torch.clamp(lim / dn.clamp(min=1e-12), max=1.0) * sel
    x_adv[..., 0] = x[..., 0] + delta.detach()
    return x_adv.detach()


def eval_pools(model, thr, pools, src, st, scen=None):
    out = {}
    for tag, pool in pools.items():
        X, _ = get_input(pool, src, st)
        ma, mb, cm, pm_, lc_ = evaluate(model, Gpu(pool, X), pool, thr, auc=False)
        if scen:
            save_scen(f"{scen}_{tag}.npz", pool, lc_.argmax(1), pm_, thr)
        out[tag] = dict(A_macroF1=ma["macro_f1"], A_balAcc=ma["bal_acc"], B_F1att=mb["f1_att"], B_TPR=mb["tpr"], B_FPR=mb["fpr"])
    return out


def train_one(name, case, seed, cfg=None, tr=None, va=None, te=None, src="pmu", epochs=None, verbose=False,
              eval_test=True, keep_model=False, save=None, train_frac=None, extra=None, extra_test=None, scen=None):
    cfg = cfg or {}
    seed_all(seed)
    tr = tr or Pool(case, "train"); va = va or Pool(case, "val")
    if train_frac is not None and train_frac < 1.0:
        tr = subsample_pool(tr, train_frac, seed)
    Xtr, st = get_input(tr, src); Xva, _ = get_input(va, src, st)
    gtr, gva = Gpu(tr, Xtr), Gpu(va, Xva)
    model = make_model(name, tr, cfg).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    epochs = epochs or cfg.get("epochs", 20)
    bs = cfg.get("bs", 128)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-3), weight_decay=cfg.get("wd", 1e-2))
    steps_per = (tr.N * tr.T + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 2e-3), total_steps=epochs * steps_per, pct_start=0.15)
    cnt = np.bincount(tr.cls.ravel(), minlength=4).astype(np.float64)
    cw = torch.tensor((cnt.sum() / (4 * cnt)) ** 0.5, dtype=torch.float32, device=DEV)
    pos = tr.mask.mean()
    pw = torch.tensor(float(np.clip(np.sqrt((1 - pos) / max(pos, 1e-6)), 1, 10)), device=DEV)
    lam = cfg.get("lam", 1.0)
    adv_eps, adv_p = cfg.get("adv_eps", 0.0), cfg.get("adv_p", 0.5)
    best, best_state, hist = -1, None, []
    n_all = torch.arange(tr.N, device=DEV)[:, None].expand(tr.N, tr.T).reshape(-1)
    t_all = torch.arange(tr.T, device=DEV)[None].expand(tr.N, tr.T).reshape(-1)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(n_all), device=DEV)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            n, t = n_all[idx], t_all[idx]
            x = gtr.batch(n, t)
            y = gtr.cls[n, t]; mk_b = gtr.mask[n, t]
            if adv_eps > 0 and np.random.rand() < adv_p and mk_b.any():
                x = feature_attack(model, x, mk_b, adv_eps, 1, cw, pw, y, lam, rand_eps=True, rand_start=True)
            c, mm = model(x)
            mk = mk_b.float()
            loss = F.cross_entropy(c, y, weight=cw) + lam * F.binary_cross_entropy_with_logits(mm, mk, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        lc, pm = predict(model, gva)
        ma, _ = task_a_metrics(va.cls.reshape(-1), lc.argmax(1))
        mb = task_b_metrics(va.mask, score=pm, thr=0.5, auc=False)
        score = ma["macro_f1"] + mb["f1_att"]
        hist.append(dict(ep=ep, loss=tot / len(perm), val_macro_f1=ma["macro_f1"], val_f1_att=mb["f1_att"]))
        if verbose:
            print(f"  ep{ep:02d} loss {tot/len(perm):.4f} valA {ma['macro_f1']:.4f} valB {mb['f1_att']:.4f}", flush=True)
        if score > best:
            best, best_state, best_ep = score, copy.deepcopy(model.state_dict()), ep
        if cfg.get("patience") and ep - best_ep >= cfg["patience"]:
            break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    res = dict(model=name, case=case, seed=seed, src=src, params=n_par, train_time=train_time, best_ep=best_ep, epochs_run=len(hist))
    _, pm_va = predict(model, gva)
    thr = f1_best_threshold(va.mask, pm_va)
    ma, mb, cm, _, _ = evaluate(model, gva, va, thr)
    res.update({f"val_{k}": v for k, v in ma.items()}); res.update({f"val_b_{k}": v for k, v in mb.items()}); res["thr_b"] = thr
    if extra:
        res["extra_val"] = eval_pools(model, thr, extra, src, st)
    if eval_test:
        te = te or Pool(case, "test")
        Xte, _ = get_input(te, src, st)
        gte = Gpu(te, Xte)
        ma, mb, cm, pm, lc = evaluate(model, gte, te, thr)
        res.update({f"test_{k}": v for k, v in ma.items()}); res.update({f"test_b_{k}": v for k, v in mb.items()})
        res["test_cm"] = cm.tolist()
        if scen:
            save_scen(scen + "_test.npz", te, lc.argmax(1), pm, thr)
    if extra_test:
        res["extra_test"] = eval_pools(model, thr, extra_test, src, st, scen)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        torch.save(dict(state=model.state_dict(), name=name, cfg=cfg, case=case, seed=seed, src=src, thr_b=thr, stats=st), save)
    if keep_model:
        return res, model, hist
    return res
