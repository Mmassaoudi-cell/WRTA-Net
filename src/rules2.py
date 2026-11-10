"""Rule-based benchmarks evaluated with per-scenario statistics (v2): SOURCE (fixed HYB-SE threshold), LNR, SOURCE-ADAPT (true noise scale),
CUSUM (per-meter CUSUM on the scale-normalised PMU-LSE residual, parameters fitted on the validation pool).
python rules2.py --cases case14 case39 case118 --test-tags test,rob_pmu10,..."""
import os, sys, json, argparse, itertools
import numpy as np
from mldata import Pool, task_b_metrics, save_scen, robust_scale

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results2")


def cusum_scores(pool, k):
    """Per-meter one-sided CUSUM of |r|/s_t - k  (returns the running statistic, shape (N,T,m))."""
    s = np.maximum(robust_scale(pool.Rp), 0.3)[..., None]
    x = np.abs(pool.Rp) / s - k
    S = np.zeros_like(x)
    for t in range(pool.T):
        S[:, t] = np.maximum(0.0, (S[:, t - 1] if t else 0.0) + x[:, t])
    return S


def score(name, pool, par=None):
    """returns (score, thr)"""
    if name == "SOURCE":
        return np.abs(pool.z - pool.zhyb) / pool.thr, 1.0
    if name == "LNR":
        return pool.lnr / 3.0, 1.0
    if name == "SOURCE-ADAPT":
        return np.abs(pool.Rh) / (3.0 * pool.meta["noise_scale"][:, None, None]), 1.0
    if name == "CUSUM":
        return cusum_scores(pool, par["k"]) / par["h"], 1.0
    raise ValueError(name)


def fit_cusum(val):
    best, bp = -1, None
    for k, h in itertools.product([0.5, 1.0, 1.5, 2.0, 3.0], [0.5, 1, 2, 3, 4, 6, 8, 12, 16]):
        S = cusum_scores(val, k)
        f1 = task_b_metrics(val.mask, flag=S >= h, auc=False)["f1_att"]
        if f1 > best:
            best, bp = f1, dict(k=k, h=float(h))
    return bp, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["case14", "case39", "case118"])
    ap.add_argument("--test-tags", default="test")
    ap.add_argument("--reps", default="0")
    ap.add_argument("--val-tag", default="val_a")
    ap.add_argument("--out", default="rules.jsonl")
    a = ap.parse_args()
    tags = [t for t in a.test_tags.split(",") if t]
    for case in a.cases:
        for rep in [int(r) for r in a.reps.split(",")]:
            pre = f"r{rep}_" if rep else ""
            val = Pool(case, pre + a.val_tag)
            par, vf1 = fit_cusum(val)
            for name in ["SOURCE", "LNR", "SOURCE-ADAPT", "CUSUM"]:
                for tag in tags:
                    tt = pre + tag if tag == "test" else tag
                    if rep and tag != "test":
                        continue
                    pool = Pool(case, tt)
                    sc, thr = score(name, pool, par)
                    mb = task_b_metrics(pool.mask, score=sc, thr=thr)
                    save_scen(os.path.join(RES, "scen", name if rep == 0 else f"{name}_rep{rep}", f"{case}_s0_{tag}.npz"), pool, None, sc, thr)
                    row = dict(name=name, case=case, seed=rep, rep=rep, pool=tag, cusum=par if name == "CUSUM" else None, **{f"b_{k}": v for k, v in mb.items()})
                    with open(os.path.join(RES, a.out), "a") as f:
                        f.write(json.dumps(row) + "\n")
            print(case, rep, "CUSUM fit", par, round(vf1, 3), flush=True)
