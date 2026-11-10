"""Leave-one-system-out generalisation: train on two grids, test on the held-out grid (zero-shot) and after few-shot adaptation.
The meter-shared network has no system-specific parameters (graph and meter kinds are swapped in), so it transfers directly.
python crosssys.py --model C4-WR-TGA --seeds 0 1 2"""
import os, sys, json, argparse, time, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mldata import Pool, W, task_a_metrics, task_b_metrics
import train_dl, nets, bench, tab
from train_dl import DEV, Gpu, get_input, predict, seed_all, subsample_pool

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results2")
CASES = ["case14", "case39", "case118"]


class Sys_:
    def __init__(self, case, tag):
        self.pool = Pool(case, tag)
        self.g = Gpu(self.pool, get_input(self.pool, "pmu")[0])
        isv = np.zeros(self.pool.m, int); isv[self.pool.vuln] = 1
        self.A = nets.build_adj(self.pool.kind, self.pool.loc, self.pool.f, self.pool.t)
        self.kinds = self.pool.kind; self.vbit = isv


def fit(model, systems, epochs, lr, bs=128, lam=4.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * max(s.pool.N * s.pool.T for s in systems) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    cw = []; pws = []
    for s in systems:
        cnt = np.bincount(s.pool.cls.ravel(), minlength=4).astype(float)
        cw.append(torch.tensor((cnt.sum() / (4 * cnt)) ** 0.5, dtype=torch.float32, device=DEV))
        pos = s.pool.mask.mean(); pws.append(torch.tensor(float(np.clip(np.sqrt((1 - pos) / pos), 1, 10)), device=DEV))
    model.train()
    for step in range(steps):
        k = step % len(systems); s = systems[k]
        model.base.set_system(s.A, s.kinds, s.vbit)
        n = torch.randint(0, s.pool.N, (bs,), device=DEV); t = torch.randint(0, s.pool.T, (bs,), device=DEV)
        c, mm = model(s.g.batch(n, t))
        loss = F.cross_entropy(c, s.g.cls[n, t], weight=cw[k]) + lam * F.binary_cross_entropy_with_logits(mm, s.g.mask[n, t].float(), pos_weight=pws[k])
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
    model.eval()


def evaluate(model, s, thr=0.5):
    model.base.set_system(s.A, s.kinds, s.vbit)
    lc, pm = predict(model, s.g)
    ma, _ = task_a_metrics(s.pool.cls.reshape(-1), lc.argmax(1))
    mb = task_b_metrics(s.pool.mask, score=pm, thr=thr)
    return dict(A_macroF1=ma["macro_f1"], A_balAcc=ma["bal_acc"], A_STrecall=ma["minority_recall"], B_F1att=mb["f1_att"], B_PRAUC=mb["pr_auc"], B_TPR=mb["tpr"], B_FPR=mb["fpr"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="V2c"); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--out", default="crosssys.jsonl")
    ap.add_argument("--targets", nargs="+", default=CASES)
    a = ap.parse_args()
    base, cfg = bench.CANDIDATES[a.model]
    cfg = dict(cfg); tj = os.path.join(RES, "tuned", f"{a.model}.json")
    if os.path.exists(tj): cfg.update(json.load(open(tj))["best_params"])
    for seed in a.seeds:
        for tgt in a.targets:
            srcs = [c for c in CASES if c != tgt]
            t0 = time.time(); seed_all(seed)
            tr_sys = [Sys_(c, "train_a") for c in srcs]
            te = Sys_(tgt, "test")
            model = train_dl.make_model(base, Pool(srcs[0], "train_a"), cfg).to(DEV)
            fit(model, tr_sys, a.epochs, cfg.get("lr", 2e-3), lam=cfg.get("lam", 4.0))
            row = dict(model=a.model, target=tgt, seed=seed, sources="+".join(srcs), mode="zero-shot", frac=0.0, **evaluate(model, te))
            open(os.path.join(RES, a.out), "a").write(json.dumps(row) + "\n")
            tgt_train = Pool(tgt, "train_a")
            for frac in (0.05, 0.10, 0.25):
                m2 = copy.deepcopy(model)
                sub = subsample_pool(tgt_train, frac, seed)
                s2 = Sys_.__new__(Sys_); s2.pool = sub; s2.g = Gpu(sub, get_input(sub, "pmu")[0]); s2.A, s2.kinds, s2.vbit = te.A, te.kinds, te.vbit
                fit(m2, [s2], 12, cfg.get("lr", 2e-3) / 4, lam=cfg.get("lam", 4.0))
                row = dict(model=a.model, target=tgt, seed=seed, sources="+".join(srcs), mode="few-shot", frac=frac, **evaluate(m2, te))
                open(os.path.join(RES, a.out), "a").write(json.dumps(row) + "\n")
            print(f"{a.model} target {tgt} seed {seed} {time.time()-t0:.0f}s zero-shot A={row['A_macroF1']:.3f}", flush=True)
