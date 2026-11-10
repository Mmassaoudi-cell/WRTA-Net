"""Experiment driver (v2) with resume.  Results appended as JSON lines.
python bench.py --models M ... --seeds ... --out FILE [--test] [--rep K] [--extra-val a,b] [--extra-test a,b] [--cfg JSON] [--tag SUFFIX]
Protocol tags: train_a / val_a (augmented robust-training mixture), test (nominal) and shifted pools rob_* / val_*.
Replicate K in {1,2} uses r{K}_train_a / r{K}_val_a / r{K}_test."""
import os, sys, json, argparse, time, traceback
import numpy as np
import torch
from mldata import Pool
import train_dl
import tab

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results2")

_P = json.load(open(os.path.join(ROOT, "results", "tuned", "C4-WR-TGA.json")))["best_params"]       # v1-tuned hyper-parameters (starting point)
BASE_WR = dict(lr=_P["lr"], wd=_P["wd"], d=_P["d"], lam=_P["lam"], bs=_P["bs"], layers=_P["layers"], glayers=_P["glayers"], temporal="ssm")

CANDIDATES = {
    "V1": ("WR", dict(BASE_WR, attn=True, graph=True, bias=True)),                                  # previous design
    "V2a": ("WR", dict(BASE_WR, attn=True, graph=False, bias=True)),                                # no local graph mixing
    "V2b": ("WR", dict(BASE_WR, attn=True, graph=False, bias=False)),                               # attention without graph bias
    "V2c": ("WR", dict(BASE_WR, attn=True, graph=False, bias=True, prior_drop=0.3)),                # + vulnerability-prior dropout
    "V2d": ("WR", dict(BASE_WR, attn=True, graph=False, bias=True, prior_drop=0.3, adv_eps=1.0)),   # + feature-space adversarial augmentation
    "V2e": ("WR", dict(BASE_WR, attn=True, graph=False, bias=False, prior_drop=0.3, adv_eps=1.0)),
    "C5-WR-Boost": ("WR-Boost", {}),
}
DEEP = ["MLP", "TCN", "GRU", "Transformer", "FT-Transformer", "GCN", "GAT", "Mamba"]
TABULAR = ["LR", "RF", "ET", "XGB", "LGBM", "CatBoost"]
RULES = ["SOURCE", "LNR", "SOURCE-ADAPT"]


def done_keys(path):
    ks = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); ks.add((r["name"], r["case"], r["seed"], r.get("rep") or 0))
            except Exception:
                pass
    return ks


def load(case, tag, cache):
    return cache.setdefault((case, tag), Pool(case, tag))


def run_one(name, case, seed, ov, epochs, tags, cache, save, train_frac, extra_val, extra_test, scen_name):
    tr, va = load(case, tags["train"], cache), load(case, tags["val"], cache)
    te = load(case, tags["test"], cache) if tags["test"] else None
    ev = {t: load(case, t, cache) for t in extra_val}
    et = {t: load(case, t, cache) for t in extra_test}
    base, cfg = CANDIDATES[name] if name in CANDIDATES else (name, {})
    cfg = dict(cfg)
    tj = os.path.join(RES, "tuned", f"{name}.json")
    if os.path.exists(tj) and not (ov or {}).get("untuned"):
        cfg.update(json.load(open(tj))["best_params"])
    cfg.update({k: v for k, v in (ov or {}).items() if k != "untuned"})
    scen = os.path.join(RES, "scen", scen_name, f"{case}_s{seed}") if scen_name else None
    if base in ("WR-Boost",) or base in TABULAR:
        sv = os.path.join(RES, "models", f"{scen_name or name}_{case}_s{seed}.pkl") if save else None
        return tab.fit_eval_tab(base, case, seed, params=cfg, tr=tr, va=va, te=te, eval_test=bool(te), save=sv, train_frac=train_frac,
                                extra=ev or None, extra_test=et or None, scen=scen if te else None)
    sv = os.path.join(RES, "models", f"{scen_name or name}_{case}_s{seed}.pt") if save else None
    return train_dl.train_one(base, case, seed, cfg, tr=tr, va=va, te=te, epochs=epochs, eval_test=bool(te), src=cfg.get("src", "pmu"), save=sv,
                              train_frac=train_frac, extra=ev or None, extra_test=et or None, scen=scen if te else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["case14", "case39", "case118"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--train-tag", default="train_a"); ap.add_argument("--val-tag", default="val_a"); ap.add_argument("--test-tag", default="test")
    ap.add_argument("--extra-val", default=""); ap.add_argument("--extra-test", default="")
    ap.add_argument("--cfg", default=None, help="json dict override")
    ap.add_argument("--save", action="store_true"); ap.add_argument("--tag", default=None, help="name suffix for variant rows")
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--scen", action="store_true", help="store per-scenario statistics for bootstrap / alarm metrics")
    a = ap.parse_args()
    out = os.path.join(RES, a.out)
    os.makedirs(RES, exist_ok=True)
    tags = dict(train=a.train_tag, val=a.val_tag, test=(a.test_tag if a.test else None))
    if a.rep:
        tags = dict(train=f"r{a.rep}_{a.train_tag}", val=f"r{a.rep}_{a.val_tag}", test=(f"r{a.rep}_{a.test_tag}" if a.test else None))
    extra_val = [t for t in a.extra_val.split(",") if t]; extra_test = [t for t in a.extra_test.split(",") if t]
    jobs = [(m, c, s) for c in a.cases for m in a.models for s in a.seeds]
    done = done_keys(out)
    cache = {}
    ov = json.loads(a.cfg) if a.cfg else None
    for (m, c, s) in jobs:
        nm = (m + a.tag) if a.tag else m
        if (nm, c, s, a.rep or 0) in done:
            continue
        t0 = time.time()
        try:
            r = run_one(m, c, s, ov, a.epochs, tags, cache, a.save, a.frac, extra_val, extra_test, ((nm + (f"_rep{a.rep}" if a.rep else "")) if a.scen else None))
            r.update(name=nm, base_model=m, frac=a.frac, rep=a.rep, wall=time.time() - t0, epochs_cfg=a.epochs, tags=tags)
            if ov: r["override"] = ov
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[{nm} {c} s{s}] {time.time()-t0:.0f}s valA={r.get('val_macro_f1', float('nan')):.3f} valB={r.get('val_b_f1_att', float('nan')):.3f}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAILED {nm} {c} {s}", flush=True)
