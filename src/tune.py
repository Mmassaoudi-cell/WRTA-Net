"""Validation-only hyper-parameter search (Optuna TPE) for baselines and top candidates.
python tune.py --model FT-Transformer --case case39 --trials 8 --epochs 10"""
import os, sys, json, argparse, time
import numpy as np, optuna
from mldata import Pool
import bench, train_dl, tab

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TUNED = os.path.join(ROOT, "results", "tuned")
os.makedirs(TUNED, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

WIDTH = {"MLP": [256, 512, 1024], "TCN": [64, 128, 256], "GRU": [64, 128, 256], "Transformer": [64, 128, 192],
         "FT-Transformer": [32, 48, 64, 96], "GCN": [32, 64, 128], "GAT": [32, 64, 128], "Mamba": [64, 96, 128]}


def space(model, tr):
    if model in WIDTH:
        return dict(lr=("logf", 5e-4, 6e-3), wd=("logf", 1e-4, 1e-1), bs=("cat", [64, 128, 256]), width=("cat", WIDTH[model]))
    if (model.startswith("C") and model != "CatBoost") or model.startswith("V"):          # proposed candidates
        s = dict(lr=("logf", 5e-4, 6e-3), wd=("logf", 1e-4, 1e-1), d=("cat", [24, 32, 48]), lam=("cat", [0.5, 1.0, 2.0, 4.0]),
                 bs=("cat", [64, 128]), layers=("cat", [1, 2, 3]), glayers=("cat", [1, 2]))
        if model == "C5-WR-Boost":
            return dict(n=("int", 150, 500), leaves=("int", 15, 255), lr=("logf", 0.03, 0.3), col=("float", 0.5, 1.0))
        return s
    return dict(XGB=dict(n=("int", 150, 600), depth=("int", 4, 10), lr=("logf", 0.03, 0.3), col=("float", 0.5, 1.0)),
                LGBM=dict(n=("int", 150, 600), leaves=("int", 15, 255), lr=("logf", 0.03, 0.3), col=("float", 0.5, 1.0)),
                CatBoost=dict(n=("int", 200, 700), depth=("int", 4, 10), lr=("logf", 0.03, 0.3)),
                RF=dict(n=("int", 40, 120), depth=("int", 10, 24), leaf=("int", 1, 10)),
                ET=dict(n=("int", 40, 120), depth=("int", 10, 24), leaf=("int", 1, 10)),
                LR=dict(C=("logf", 1e-3, 10.0)))[model]


def suggest(trial, sp):
    out = {}
    for k, v in sp.items():
        if v[0] == "logf": out[k] = trial.suggest_float(k, v[1], v[2], log=True)
        elif v[0] == "float": out[k] = trial.suggest_float(k, v[1], v[2])
        elif v[0] == "int": out[k] = trial.suggest_int(k, v[1], v[2])
        else: out[k] = trial.suggest_categorical(k, v[1])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--case", default="case39")
    ap.add_argument("--trials", type=int, default=8); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tr, va = Pool(a.case, "train"), Pool(a.case, "val")
    sp = space(a.model, tr)
    base_cfg = dict(bench.CANDIDATES.get(a.model, (a.model, {}))[1])
    base = bench.CANDIDATES[a.model][0] if a.model in bench.CANDIDATES else a.model

    def objective(trial):
        p = suggest(trial, sp)
        cfg = dict(base_cfg); cfg.update(p)
        if base in tab.make_tab.__globals__.get("TABULAR_NAMES", ["LR", "RF", "ET", "XGB", "LGBM", "CatBoost", "WR-Boost"]):
            r = tab.fit_eval_tab(base, a.case, a.seed, params=cfg, tr=tr, va=va, eval_test=False)
        else:
            r = train_dl.train_one(base, a.case, a.seed, cfg, tr=tr, va=va, epochs=a.epochs, eval_test=False)
        return r["val_macro_f1"] + r["val_b_f1_att"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    DEF = dict(lr=2e-3, wd=1e-2, bs=128, d=32, lam=1.0, layers=2, glayers=2)
    DEF_W = dict(MLP=512, TCN=128, GRU=128, Transformer=128, **{"FT-Transformer": 48}, GCN=64, GAT=64, Mamba=96)
    DEF_T = dict(XGB=dict(n=300, depth=6, lr=0.1, col=0.8), LGBM=dict(n=300, leaves=63, lr=0.08, col=0.8), CatBoost=dict(n=400, depth=6, lr=0.1),
                 RF=dict(n=80, depth=16, leaf=3), ET=dict(n=80, depth=18, leaf=3), LR=dict(C=1.0), **{"C5-WR-Boost": dict(n=300, leaves=63, lr=0.08, col=0.8)})
    if a.model in DEF_T:
        d0 = DEF_T[a.model]
    else:
        d0 = {k: DEF[k] for k in sp if k in DEF}
        if a.model in DEF_W: d0["width"] = DEF_W[a.model]
    study.enqueue_trial(d0)      # first trial = default configuration: the search can never be worse than untuned
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials)
    best = dict(base_cfg); best.update(study.best_params)
    out = dict(model=a.model, case=a.case, trials=a.trials, epochs=a.epochs, best_value=study.best_value, best_params=study.best_params,
               all=[dict(params=t.params, value=t.value) for t in study.trials], seconds=time.time() - t0)
    json.dump(out, open(os.path.join(TUNED, f"{a.model}.json"), "w"), indent=1)
    print(a.model, "best", round(study.best_value, 4), study.best_params, f"{time.time()-t0:.0f}s")
