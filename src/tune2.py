"""Validation-only hyper-parameter search (Optuna TPE) under the v2 protocol: the objective is the mean validation score over two systems
(IEEE-14 and NE-39) on the augmented validation pools, with the same trial budget for every learned model.
python tune2.py --model FT-Transformer --trials 8 --epochs 10"""
import os, sys, json, argparse, time
import numpy as np, optuna
from mldata import Pool
import bench, train_dl, tab
from tune import space, suggest, WIDTH

TUNED = os.path.join(bench.RES, "tuned")
os.makedirs(TUNED, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
DEF = dict(lr=2e-3, wd=1e-2, bs=128, d=24, lam=4.0, layers=3, glayers=2)
DEF_W = dict(MLP=512, TCN=128, GRU=128, Transformer=128, **{"FT-Transformer": 48}, GCN=64, GAT=64, Mamba=96)
DEF_T = dict(XGB=dict(n=300, depth=6, lr=0.1, col=0.8), LGBM=dict(n=300, leaves=63, lr=0.08, col=0.8), CatBoost=dict(n=400, depth=6, lr=0.1),
             RF=dict(n=80, depth=16, leaf=3), ET=dict(n=80, depth=18, leaf=3), LR=dict(C=1.0), **{"C5-WR-Boost": dict(n=300, leaves=63, lr=0.08, col=0.8)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--cases", nargs="+", default=["case14", "case39"])
    ap.add_argument("--trials", type=int, default=8); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pools = {c: (Pool(c, "train_a"), Pool(c, "val_a")) for c in a.cases}
    base, base_cfg = bench.CANDIDATES.get(a.model, (a.model, {}))
    base_cfg = dict(base_cfg)
    sp = space(a.model if a.model != "CatBoost" else "CatBoost", pools[a.cases[0]][0])

    def objective(trial):
        p = suggest(trial, sp)
        cfg = dict(base_cfg); cfg.update(p)
        sc = []
        for c, (tr, va) in pools.items():
            if base in tab.make_tab.__globals__.get("TABULAR_NAMES", ["LR", "RF", "ET", "XGB", "LGBM", "CatBoost", "WR-Boost"]):
                r = tab.fit_eval_tab(base, c, a.seed, params=cfg, tr=tr, va=va, eval_test=False)
            else:
                r = train_dl.train_one(base, c, a.seed, cfg, tr=tr, va=va, epochs=a.epochs, eval_test=False)
            sc.append(r["val_macro_f1"] + r["val_b_f1_att"])
        return float(np.mean(sc))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    if a.model in DEF_T:
        d0 = DEF_T[a.model]
    elif a.model in DEF_W:
        d0 = {k: DEF[k] for k in sp if k in DEF}; d0["width"] = DEF_W[a.model]
    else:
        d0 = {k: DEF[k] for k in sp if k in DEF}
    study.enqueue_trial(d0)
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials)
    out = dict(model=a.model, cases=a.cases, trials=a.trials, epochs=a.epochs, best_value=study.best_value, best_params=study.best_params,
               all=[dict(params=t.params, value=t.value) for t in study.trials], seconds=time.time() - t0)
    json.dump(out, open(os.path.join(TUNED, f"{a.model}.json"), "w"), indent=1)
    print(a.model, "best", round(study.best_value, 4), study.best_params, f"{time.time()-t0:.0f}s")
