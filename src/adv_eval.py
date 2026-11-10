"""White-box feature-space attacker evaluation of saved deep models.
The attacker controls the replayed meters: it re-noises the replayed data (the clone feature of the compromised meters is set to the clean level) and shifts their
whitened residuals by up to eps along the loss gradient (PGD, 5 steps).
python adv_eval.py --models NAME ... --cases case14 --seeds 0 1 2 --pool test --eps 0 0.5 1 2"""
import os, json, argparse
import numpy as np, torch
import torch.nn.functional as F
from mldata import Pool, task_a_metrics, task_b_metrics
import train_dl
from train_dl import DEV, Gpu, get_input, make_model, feature_attack

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results2")


def load_ckpt(path, tr_cache, case, train_tag):
    ck = torch.load(path, weights_only=False)
    tr = tr_cache.setdefault((case, train_tag), Pool(case, train_tag))
    model = make_model(ck["name"], tr, ck["cfg"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    return model, ck, tr


def attack_eval(model, ck, pool, eps, steps=5, bs=128, norm="linf"):
    X, _ = get_input(pool, ck["src"], ck["stats"])
    g = Gpu(pool, X)
    cnt = np.bincount(pool.cls.ravel(), minlength=4).astype(np.float64)
    cw = torch.tensor((cnt.sum() / (4 * cnt)) ** 0.5, dtype=torch.float32, device=DEV)
    pos = pool.mask.mean(); pw = torch.tensor(float(np.clip(np.sqrt((1 - pos) / pos), 1, 10)), device=DEV)
    n, t = torch.meshgrid(torch.arange(g.N, device=DEV), torch.arange(g.T, device=DEV), indexing="ij")
    n, t = n.reshape(-1), t.reshape(-1)
    lc, pm = [], []
    for i in range(0, len(n), bs):
        x = g.batch(n[i:i + bs], t[i:i + bs]); y = g.cls[n[i:i + bs], t[i:i + bs]]; mk = g.mask[n[i:i + bs], t[i:i + bs]]
        if eps > 0 and mk.any():
            x = feature_attack(model, x, mk, eps, steps, cw, pw, y, 4.0, norm=norm, rand_start=True)
        with torch.no_grad():
            c, mm = model(x)
        lc.append(c.cpu()); pm.append(torch.sigmoid(mm).cpu())
    lc = torch.cat(lc).numpy(); pm = torch.cat(pm).numpy().reshape(g.N, g.T, g.m)
    ma, _ = task_a_metrics(pool.cls.reshape(-1), lc.argmax(1))
    mb = task_b_metrics(pool.mask, score=pm, thr=ck["thr_b"], auc=False)
    return dict(A_macroF1=ma["macro_f1"], A_balAcc=ma["bal_acc"], B_F1att=mb["f1_att"], B_TPR=mb["tpr"], B_FPR=mb["fpr"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--cases", nargs="+", default=["case14", "case39", "case118"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--pool", default="test")
    ap.add_argument("--train-tag", default="train_a"); ap.add_argument("--eps", nargs="+", type=float, default=[0, 0.5, 1, 2])
    ap.add_argument("--nscen", type=int, default=0); ap.add_argument("--out", default="adv.jsonl"); ap.add_argument("--norms", nargs="+", default=["linf", "l2"])
    a = ap.parse_args()
    cache = {}
    for case in a.cases:
        pool = Pool(case, a.pool)
        if a.nscen:
            from robust import take
            pool = take(pool, np.arange(a.nscen))
        for name in a.models:
            for s in a.seeds:
                p = os.path.join(RES, "models", f"{name}_{case}_s{s}.pt")
                if not os.path.exists(p):
                    continue
                model, ck, _ = load_ckpt(p, cache, case, a.train_tag)
                for norm in a.norms:
                    for eps in a.eps:
                        if eps == 0 and norm != a.norms[0]:
                            continue
                        r = attack_eval(model, ck, pool, eps, norm=norm)
                        with open(os.path.join(RES, a.out), "a") as f:
                            f.write(json.dumps(dict(name=name, case=case, seed=s, pool=a.pool, eps=eps, norm=norm, **r)) + "\n")
                print(name, case, s, "done", flush=True)
