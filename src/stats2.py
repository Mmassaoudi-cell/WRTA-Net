"""Scenario-level cluster bootstrap and operational (alarm) metrics from the per-scenario statistics in results2/scen.
python stats2.py --prop PROP --models A B C ... [--reps 0,1,2]"""
import os, sys, json, argparse, glob
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results2")
CASES = ["case14", "case39", "case118"]
NB = 2000


def path(name, case, seed, tag="test", rep=0):
    d = f"{name}_rep{rep}" if rep else name
    return os.path.join(RES, "scen", d, f"{case}_s{seed}_{tag}.npz")


def conf_per_scen(pa, cls):
    N = len(cls)
    C = np.zeros((N, 4, 4), np.int64)
    for k in range(4):
        for j in range(4):
            C[:, k, j] = ((cls == k) & (pa == j)).sum(1)
    return C.reshape(N, 16)


def macro_f1_from_conf(c16):
    c = c16.reshape(-1, 4, 4).astype(float)
    tp = np.einsum("bii->bi", c); fp = c.sum(1) - tp; fn = c.sum(2) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return f1.mean(1)


def f1_att(c4):
    tp, fp, fn = c4[..., 0], c4[..., 1], c4[..., 2]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def load_model(name, case, seeds, tag="test", rep=0):
    """returns list per seed of dict(cnt, conf or None)"""
    out = []
    for s in seeds:
        p = path(name, case, s, tag, rep)
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        out.append(dict(cnt=d["cnt"], conf=conf_per_scen(d["pa"], d["cls"]) if "pa" in d.files else None))
    return out


def boot_metrics(mods, idx):
    """seed-averaged bootstrap distribution of (A macro-F1, B attacked-meter F1); idx: (B,N)."""
    A = [],
    a_list, b_list = [], []
    for d in mods:
        b_list.append(f1_att(d["cnt"][idx].sum(1)))
        if d["conf"] is not None:
            a_list.append(macro_f1_from_conf(d["conf"][idx].sum(1)))
    return (np.mean(a_list, 0) if a_list else None), np.mean(b_list, 0)


def point(mods):
    a = [macro_f1_from_conf(d["conf"].sum(0, keepdims=True))[0] for d in mods if d["conf"] is not None]
    b = [f1_att(d["cnt"].sum(0, keepdims=True))[0] for d in mods]
    return (np.mean(a) if a else np.nan), np.mean(b)


def compare(prop, models, seeds, reps, tag="test"):
    rng = np.random.default_rng(0)
    rows = []
    store = {}
    for case in CASES:
        N = None
        for rep in reps:
            m0 = load_model(prop, case, seeds, tag, rep)
            if m0:
                N = len(m0[0]["cnt"]); break
        for mdl in models:
            diffs = {"A": [], "B": []}
            pts = {"A": [], "B": []}
            for rep in reps:
                P = load_model(prop, case, seeds, tag, rep); Q = load_model(mdl, case, seeds if mdl not in ("SOURCE", "LNR", "CUSUM", "SOURCE-ADAPT") else [0], tag, rep)
                if not P or not Q:
                    continue
                n = len(P[0]["cnt"])
                idx = rng.integers(0, n, size=(NB, n))
                pa, pb = boot_metrics(P, idx); qa, qb = boot_metrics(Q, idx)
                if pa is not None and qa is not None:
                    diffs["A"].append(pa - qa)
                diffs["B"].append(pb - qb)
                pa0, pb0 = point(P); qa0, qb0 = point(Q)
                pts["A"].append(pa0 - qa0); pts["B"].append(pb0 - qb0)
            for k in ("A", "B"):
                if not diffs[k]:
                    continue
                d = np.mean(diffs[k], 0)                      # average over replicates (replicates are fixed effects)
                store.setdefault((mdl, k), []).append((d, float(np.mean(pts[k]))))
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                rows.append(dict(benchmark=mdl, case=case, metric=k, n_rep=len(diffs[k]), diff=float(np.mean(pts[k])), ci_lo=float(lo), ci_hi=float(hi), p_boot=max(p, 1.0 / NB),
                                 per_rep=[float(x) for x in pts[k]]))
    for (mdl, k), lst in store.items():                  # system-averaged difference (independent resamples per system)
        if len(lst) == len(CASES):
            d = np.mean([x[0] for x in lst], 0)
            lo, hi = np.percentile(d, [2.5, 97.5]); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            rows.append(dict(benchmark=mdl, case="avg", metric=k, n_rep=len(reps), diff=float(np.mean([x[1] for x in lst])), ci_lo=float(lo), ci_hi=float(hi),
                             p_boot=max(p, 1.0 / NB), per_rep=[]))
    df = pd.DataFrame(rows)
    # Holm over all comparisons
    sub = df[df.case != "avg"]
    order = sub.sort_values("p_boot").index.tolist(); m = len(order); run = 0.0; ph = {}
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * df.loc[i, "p_boot"])); ph[i] = run
    for i in df.index[df.case == "avg"]:
        ph[i] = min(1.0, df.loc[i, "p_boot"] * 2)
    df["p_holm"] = df.index.map(ph)
    def verdict(r):
        if r.case == "avg":
            return "superior" if r.ci_lo > 0 else ("inferior" if r.ci_hi < 0 else "tie")
        if r.ci_lo > 0 and r.p_holm < 0.05: return "superior"
        if r.ci_hi < 0 and r.p_holm < 0.05: return "inferior"
        return "tie"
    df["verdict"] = df.apply(verdict, axis=1)
    return df


# ------------------------------------------------------------------------------------------------ alarm metrics
def alarm_metrics(name, case, seeds, k_min=3, persist=2, tag="test", rep=0):
    res = []
    for s in seeds:
        p = path(name, case, s, tag, rep)
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        nflag, nhit, cls, t_a = d["nflag"].astype(int), d["nhit"].astype(int), d["cls"], d["t_a"]
        N, T = nflag.shape
        hit = nflag >= k_min
        al = np.zeros_like(hit)
        for t in range(persist - 1, T):
            al[:, t] = hit[:, t - persist + 1:t + 1].all(1)
        attacked = (cls > 0).any(1)
        det, delay = [], []
        for n in np.where(attacked)[0]:
            act = np.where(cls[n] > 0)[0]
            ok = al[n, act] & (nhit[n, act] >= 1)
            det.append(ok.any())
            if ok.any():
                delay.append(act[np.argmax(ok)] - act[0])
        normal_scans = (cls == 0)
        edges = al & ~np.concatenate([np.zeros((N, 1), bool), al[:, :-1]], 1)
        fa = (edges & normal_scans).sum()
        hours = normal_scans.sum() / 3600.0
        res.append(dict(det_rate=float(np.mean(det)), delay=float(np.mean(delay)) if delay else np.nan, fa_per_hour=float(fa / hours)))
    return pd.DataFrame(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", required=True); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--reps", default="0")
    ap.add_argument("--tag", default="test"); ap.add_argument("--out", default="bootstrap_pairs.csv")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]; reps = [int(x) for x in a.reps.split(",")]
    df = compare(a.prop, a.models, seeds, reps, a.tag)
    df.to_csv(os.path.join(RES, a.out), index=False)
    print(df.round(4).to_string(index=False))
