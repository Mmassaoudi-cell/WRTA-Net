"""Generate scenario pools.  usage: python gen_data.py case14 train 600 [--kind K --noise S --stealth X --tag NAME]"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import sys, argparse, time
import numpy as np
from multiprocessing import Pool
from powersys import Grid, vulnerable_set
from scenarios import make_scenario, make_events, CLASSES

POOL_SEED = dict(train=10_000, val=20_000, test=30_000, ood=40_000, rob=50_000)
EV_N = dict(train=40, val=15, test=15, ood=15, rob=15)
SEEDOFF = int(os.environ.get("GEN_SEEDOFF", 0))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

_G = None; _V = None; _EV = None; _KW = None


def _init(case, pool, kw):
    global _G, _V, _EV, _KW
    _G = Grid(case)
    _V, _ = vulnerable_set(_G)
    _EV = make_events(_G, EV_N[pool], POOL_SEED[pool] + SEEDOFF + 7)
    _KW = kw


def _work(args):
    i, seed = args
    s = make_scenario(_G, _V, _EV, seed, **_KW)
    s["i"] = i
    return s


def save_grid(case):
    g = Grid(case)
    v, parts = vulnerable_set(g)
    np.savez(os.path.join(DATA, f"{case}_grid.npz"), sigma=g.sigma, kind=g.m_kind, loc=g.m_loc, f=g.f, t=g.t, vuln=v,
             thr=g.cfg["thr"], nb=g.nb, nl=g.nl, pmu_bus=g.pmu_bus, **{f"meas_{k}": a for k, a in g.meas.items()})


def generate(case, pool, n, tag=None, nproc=20, **kw):
    os.makedirs(DATA, exist_ok=True)
    tag = tag or pool
    t0 = time.time()
    base = POOL_SEED[pool] + SEEDOFF
    jobs = [(i, base + 17 * i + 3) for i in range(n)]
    res = []
    with Pool(nproc, initializer=_init, initargs=(case, pool, kw)) as p:
        for s in p.imap(_work, jobs, chunksize=4):
            res.append(s)
    res.sort(key=lambda s: s["i"])
    out = {k: np.stack([s[k] for s in res]) for k in ["z", "zhyb", "zpmu", "lnr", "chi", "mask", "cls", "ztrue", "Vtrue", "Vwls", "Vhyb", "tflag"]}
    meta = res[0]["meta"].keys()
    for k in meta:
        vals = [s["meta"][k] for s in res]
        out["m_" + k] = np.array([CLASSES.index(v) if k == "kind" else v for v in vals])
    np.savez_compressed(os.path.join(DATA, f"{case}_{tag}.npz"), **out)
    print(f"{case} {tag}: {n} scenarios in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("pool"); ap.add_argument("n", type=int)
    ap.add_argument("--kind", default=None); ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--stealth", type=int, default=None); ap.add_argument("--tag", default=None)
    ap.add_argument("--nproc", type=int, default=20); ap.add_argument("--offprior", type=float, default=None); ap.add_argument("--outage", action="store_true"); ap.add_argument("--aug", action="store_true"); ap.add_argument("--pmuscale", type=float, default=None); ap.add_argument("--topoknown", action="store_true")
    a = ap.parse_args()
    save_grid(a.case)
    kw = {}
    if a.kind: kw["kind"] = a.kind
    if a.noise is not None: kw["noise_scale"] = a.noise
    if a.stealth is not None: kw["stealth"] = a.stealth
    if a.offprior is not None: kw["off_prior_p"] = a.offprior
    if a.outage: kw["outage"] = True
    if a.aug: kw["aug"] = True
    if a.pmuscale is not None: kw["pmu_scale"] = a.pmuscale
    if a.topoknown: kw["topo_known"] = True
    generate(a.case, a.pool, a.n, a.tag, a.nproc, **kw)
