"""Tabular (tree / linear) baselines + WR-Boost candidate + rule-based baselines (Source HYB-SE threshold, LNR BDD)."""
from __future__ import annotations
import time, os
import numpy as np
import torch
from mldata import save_scen, Pool, W, pad_series, task_a_metrics, task_b_metrics, f1_best_threshold, robust_scale
import nets

KINDS = 5


def window_tensor(R, Wn=W):
    """(N,T,m) -> (N,T,Wn,m) windows (replicate-padded)."""
    Rp = pad_series(R, Wn)
    N, T, m = R.shape
    idx = np.arange(T)[:, None] + np.arange(Wn)[None]
    return Rp[:, idx]                                    # (N,T,Wn,m)


# ------------------------------------------------------------------------------------------ Task A features
def feats_task_a(pool: Pool, X=None, rich=False):
    R = pool.R if X is None else X
    Wd = window_tensor(R)                               # (N,T,W,m)
    N, T, _, m = Wd.shape
    cur = Wd[:, :, -1]
    a = np.abs(Wd)
    if not rich:
        f = np.concatenate([cur, a.mean(2), a.max(2), pool.L, pool.tflag], -1)
        s = robust_scale(cur)[..., None]
        return np.concatenate([f, s], -1).reshape(N * T, -1).astype(np.float32)
    kinds = pool.kind
    cols = []
    s_t = robust_scale(cur)
    for k in range(KINDS):
        sel = np.where(kinds == k)[0]
        ak = a[..., sel]
        z = ak / np.maximum(s_t[:, :, None, None], 0.3)           # scale-normalised
        cols += [ak[:, :, -1].mean(-1), ak[:, :, -1].max(-1), (z[:, :, -1] > 3).mean(-1), (z[:, :, -1] > 6).mean(-1),
                 ak.mean((2, 3)), ak.max((2, 3)), (z > 3).mean((2, 3)),
                 np.abs(Wd[..., sel].mean(-1)).max(2),                         # temporal peak of cross-sectional mean
                 Wd[:, :, -1][..., sel].mean(-1), Wd[:, :, -1][..., sel].std(-1)]
        # temporal profile of exceedance fraction (last 6 steps)
        ex = (z > 3).mean(-1)                                      # (N,T,W)
        cols += [ex[:, :, i] for i in range(W - 6, W)]
    isv = np.zeros(pool.m, bool); isv[pool.vuln] = True
    Lw = window_tensor(pool.L)
    for sel in (np.where(isv)[0], np.where(~isv)[0]):
        z = a[..., sel] / np.maximum(s_t[:, :, None, None], 0.3)
        cols += [a[:, :, -1][..., sel].mean(-1), (z[:, :, -1] > 3).mean(-1), (z > 3).mean((2, 3)),
                 (Lw[:, :, -1][..., sel] < 0.3).mean(-1), (Lw[..., sel] < 0.3).mean((2, 3)), Lw[:, :, -1][..., sel].mean(-1)]
    for k in range(KINDS):
        sel = np.where(kinds == k)[0]
        cols += [(Lw[:, :, -1][..., sel] < 0.3).mean(-1), (Lw[..., sel] < 0.3).mean((2, 3))]
    cols += [s_t, np.log(s_t + 1e-3), (np.abs(cur) > 3).mean(-1), pool.chi[:, :, None][..., 0] / pool.m, pool.tflag.mean(-1)]
    F = np.stack(cols, -1)
    return F.reshape(N * T, -1).astype(np.float32)


# ------------------------------------------------------------------------------------------ Task B features
def feats_task_b(pool: Pool, X=None, rows=None, graph=False, A=None):
    """Per-(n,t,i) feature rows. rows = (n_idx,t_idx,i_idx) subset or None for all."""
    R = pool.R if X is None else X
    Wd = window_tensor(R)                               # (N,T,W,m)
    N, T, _, m = Wd.shape
    cur = Wd[:, :, -1]
    s_t = robust_scale(cur)
    frac = (np.abs(cur) > 3).mean(-1)
    kinds = pool.kind
    if rows is None:
        n, t, i = [a.ravel() for a in np.meshgrid(np.arange(N), np.arange(T), np.arange(m), indexing="ij")]
    else:
        n, t, i = rows
    w = Wd[n, t, :, i]                                  # (rows, W)
    aw = np.abs(w)
    Lw = window_tensor(pool.L)[n, t, :, i]              # clone-fingerprint window
    isv = np.zeros(m, np.float32); isv[pool.vuln] = 1
    cols = [w, Lw, aw.max(1, keepdims=True), aw.mean(1, keepdims=True), kinds[i][:, None].astype(np.float32), isv[i][:, None], pool.tflag[n, t, i][:, None],
            s_t[n, t][:, None], frac[n, t][:, None], (aw[:, -1] / np.maximum(s_t[n, t], 0.3))[:, None]]
    if graph:
        An = A / A.sum(1, keepdims=True)
        absWd = np.abs(Wd)
        nb_cur = absWd[:, :, -1] @ An.T                 # neighbour mean |r(t)|
        nb_win = absWd.mean(2) @ An.T                   # neighbour window-mean
        nb_max = np.stack([np.max(np.where(A[j][None, None] > 0, absWd[:, :, -1], 0), -1) for j in range(m)], -1) if m <= 200 else nb_cur
        kind_mean = np.stack([absWd[:, :, -1][..., kinds == k].mean(-1) for k in range(KINDS)], -1)   # (N,T,5)
        d1 = (w[:, -1] - w[:, -2])[:, None]
        nbL = (window_tensor(pool.L)[:, :, -1] < 0.3).astype(np.float32) @ An.T          # neighbour clone-flag density
        cols += [nb_cur[n, t, i][:, None], nb_win[n, t, i][:, None], nb_max[n, t, i][:, None], kind_mean[n, t],
                 d1, nbL[n, t, i][:, None], (nb_cur[n, t, i] / np.maximum(s_t[n, t], 0.3))[:, None]]
    return np.concatenate(cols, 1).astype(np.float32), (n, t, i)


def sample_rows(pool: Pool, rng, cap_pos=300_000, cap_neg=600_000):
    mk = pool.mask
    pos = np.argwhere(mk)
    neg_n = min(cap_neg, mk.size - len(pos))
    if len(pos) > cap_pos:
        pos = pos[rng.choice(len(pos), cap_pos, replace=False)]
    flat = rng.choice(mk.size, size=min(mk.size, neg_n * 2), replace=False)
    ng = np.stack(np.unravel_index(flat, mk.shape), 1)
    ng = ng[~mk[ng[:, 0], ng[:, 1], ng[:, 2]]][:neg_n]
    allr = np.concatenate([pos, ng])
    return allr[:, 0], allr[:, 1], allr[:, 2]


# ------------------------------------------------------------------------------------------ model factory
def make_tab(name, seed, task, params=None):
    p = dict(params or {})
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=p.get("C", 1.0), max_iter=300, class_weight="balanced", random_state=seed))
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=p.get("n", 80), max_depth=p.get("depth", 16), min_samples_leaf=p.get("leaf", 3), n_jobs=-1,
                                      class_weight="balanced_subsample", random_state=seed)
    if name == "ET":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=p.get("n", 80), max_depth=p.get("depth", 18), min_samples_leaf=p.get("leaf", 3), n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=seed)
    if name == "XGB":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=p.get("n", 300), max_depth=p.get("depth", 6), learning_rate=p.get("lr", 0.1), subsample=0.8,
                                 colsample_bytree=p.get("col", 0.8), tree_method="hist", device="cuda", random_state=seed, verbosity=0)
    if name in ("LGBM", "WR-Boost"):
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=p.get("n", 300), num_leaves=p.get("leaves", 63), learning_rate=p.get("lr", 0.08), subsample=0.8,
                                  subsample_freq=1, colsample_bytree=p.get("col", 0.8), random_state=seed, verbose=-1, n_jobs=8)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=p.get("n", 400), depth=p.get("depth", 6), learning_rate=p.get("lr", 0.1), random_seed=seed, verbose=0,
                                  task_type="GPU", devices="0")
    raise ValueError(name)


def fit_eval_tab(name, case, seed, params=None, tr=None, va=None, te=None, eval_test=True, save=None, train_frac=None, extra=None, extra_test=None, scen=None):
    tr = tr or Pool(case, "train"); va = va or Pool(case, "val")
    if train_frac is not None and train_frac < 1.0:
        from train_dl import subsample_pool
        tr = subsample_pool(tr, train_frac, seed)
    rich = name == "WR-Boost"
    rng = np.random.default_rng(seed)
    res = dict(model=name, case=case, seed=seed, src="hyb")
    t0 = time.time()
    # ---- Task A
    Xa_tr = feats_task_a(tr, rich=rich); ya = tr.cls.reshape(-1)
    clf = make_tab(name, seed, "A", params)
    if name == "XGB":
        clf.fit(Xa_tr, ya, sample_weight=_cw(ya))
    elif name in ("CatBoost",):
        clf.fit(Xa_tr, ya, sample_weight=_cw(ya))
    elif name in ("LGBM", "WR-Boost"):
        clf.fit(Xa_tr, ya, sample_weight=_cw(ya))
    else:
        clf.fit(Xa_tr, ya)
    # ---- Task B
    A = nets.build_adj(tr.kind, tr.loc, tr.f, tr.t)
    graph = rich
    rows = sample_rows(tr, rng)
    Xb_tr, _ = feats_task_b(tr, rows=rows, graph=graph, A=A); yb = tr.mask[rows]
    clb = make_tab(name, seed, "B", params)
    if name in ("XGB", "CatBoost", "LGBM", "WR-Boost"):
        pos_w = float(np.clip(np.sqrt((1 - yb.mean()) / yb.mean()), 1, 10))
        clb.fit(Xb_tr, yb.astype(int), sample_weight=np.where(yb, pos_w, 1.0))
    else:
        clb.fit(Xb_tr, yb.astype(int))
    res["train_time"] = time.time() - t0
    res["params"] = _n_params(clf) + _n_params(clb)

    def score_pool(pool):
        pa = clf.predict(feats_task_a(pool, rich=rich)).astype(int)
        # chunked Task-B scoring
        pm = np.zeros((pool.N, pool.T, pool.m), np.float32)
        for s in range(0, pool.N, 25):
            sub = _slice(pool, s, min(s + 25, pool.N))
            Xb, (n, t, i) = feats_task_b(sub, graph=graph, A=A)
            pm[s:s + sub.N] = clb.predict_proba(Xb)[:, 1].reshape(sub.N, sub.T, sub.m)
        return pa, pm

    pa_va, pm_va = score_pool(va)
    thr = f1_best_threshold(va.mask, pm_va)
    ma, _ = task_a_metrics(va.cls.reshape(-1), pa_va); mb = task_b_metrics(va.mask, score=pm_va, thr=thr)
    res.update({f"val_{k}": v for k, v in ma.items()}); res.update({f"val_b_{k}": v for k, v in mb.items()}); res["thr_b"] = thr
    def _extra(pools):
        out = {}
        for tag, pool in pools.items():
            pa_, pm_ = score_pool(pool)
            ma_, _ = task_a_metrics(pool.cls.reshape(-1), pa_); mb_ = task_b_metrics(pool.mask, score=pm_, thr=thr, auc=False)
            if scen:
                save_scen(f"{scen}_{tag}.npz", pool, pa_, pm_, thr)
            out[tag] = dict(A_macroF1=ma_["macro_f1"], A_balAcc=ma_["bal_acc"], B_F1att=mb_["f1_att"], B_TPR=mb_["tpr"], B_FPR=mb_["fpr"])
        return out
    if extra:
        res["extra_val"] = _extra(extra)
    if extra_test:
        res["extra_test"] = _extra(extra_test)
    if save and name not in ("RF", "ET"):
        import joblib
        os.makedirs(os.path.dirname(save), exist_ok=True)
        joblib.dump(dict(clf=clf, clb=clb, thr=thr, rich=rich, graph=graph, A=A, name=name), save)
    if eval_test:
        te = te or Pool(case, "test")
        pa, pm = score_pool(te)
        ma, cm = task_a_metrics(te.cls.reshape(-1), pa); mb = task_b_metrics(te.mask, score=pm, thr=thr)
        res.update({f"test_{k}": v for k, v in ma.items()}); res.update({f"test_b_{k}": v for k, v in mb.items()}); res["test_cm"] = cm.tolist()
        if scen:
            save_scen(scen + "_test.npz", te, pa, pm, thr)
    return res


class _Sub:
    pass


def _slice(pool: Pool, a, b):
    s = _Sub()
    for k in ("sigma", "thr", "kind", "loc", "f", "t", "nb", "nl", "m", "T", "case", "tag", "vuln"):
        setattr(s, k, getattr(pool, k))
    s.tflag = pool.tflag[a:b]; s.L = pool.L[a:b]; s.R = pool.R[a:b]; s.Rh = pool.Rh[a:b]; s.Rp = pool.Rp[a:b]; s.z = pool.z[a:b]; s.chi = pool.chi[a:b]; s.mask = pool.mask[a:b]; s.cls = pool.cls[a:b]; s.N = b - a
    return s


def _cw(y):
    cnt = np.bincount(y, minlength=4).astype(float)
    w = (cnt.sum() / (4 * np.maximum(cnt, 1))) ** 0.5
    return w[y]


def _n_params(m):
    try:
        if hasattr(m, "estimators_"):
            return int(sum(e.tree_.node_count for e in m.estimators_))
        if hasattr(m, "booster_"):
            return int(m.booster_.num_trees() * m.get_params()["num_leaves"])
        if hasattr(m, "get_booster"):
            return int(len(m.get_booster().get_dump()) * 2 ** 6)
        if hasattr(m, "tree_count_"):
            return int(m.tree_count_ * 2 ** 6)
        if hasattr(m, "steps"):
            return int(m.steps[-1][1].coef_.size)
    except Exception:
        pass
    return -1


# ------------------------------------------------------------------------------------------ rule-based baselines (no training)
def rule_eval(name, pool: Pool, thr_scale=1.0, noise_aware=False):
    """SOURCE: flag |z - zhat_HYB| > delta (fixed, from the paper).  LNR: largest-normalised-residual BDD (|r_N|>3).
    SOURCE-ADAPT: delta = 4.5*sigma_i*s_t (uses the *true* noise scale of the scenario: oracle, not counted in wins)."""
    if name == "SOURCE":
        score = np.abs(pool.z - pool.zhyb) / pool.thr
        flag = score > 1.0
    elif name == "LNR":
        score = pool.lnr / 3.0
        flag = pool.lnr > 3.0
    elif name == "SOURCE-ADAPT":
        ns = pool.meta["noise_scale"][:, None, None]
        score = np.abs(pool.Rh) / (3.0 * ns)
        flag = score > 1.0
    else:
        raise ValueError(name)
    mb = task_b_metrics(pool.mask, flag=flag, score=score)
    return mb


if __name__ == "__main__":
    import sys
    case = sys.argv[1]; va = Pool(case, "val")
    for nm in ["SOURCE", "LNR", "SOURCE-ADAPT"]:
        r = rule_eval(nm, va)
        print(nm, {k: round(v, 4) for k, v in r.items() if k in ("tpr", "fpr", "f1_att", "pr_auc", "roc_auc")})
