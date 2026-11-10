"""Zero-shot leave-one-system-out comparator: LightGBM on system-agnostic physics features (WR-Boost features)."""
import os, json, sys, numpy as np
import lightgbm as lgb
from mldata import Pool, task_a_metrics, task_b_metrics
import tab, nets
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CASES = ["case14", "case39", "case118"]
out = os.path.join(ROOT, "results2", "crosssys.jsonl")
for seed in [0, 1, 2]:
    for tgt in CASES:
        srcs = [c for c in CASES if c != tgt]
        rng = np.random.default_rng(seed)
        XA, yA, XB, yB = [], [], [], []
        for c in srcs:
            tr = Pool(c, "train_a"); A = nets.build_adj(tr.kind, tr.loc, tr.f, tr.t)
            XA.append(tab.feats_task_a(tr, rich=True)); yA.append(tr.cls.reshape(-1))
            rows = tab.sample_rows(tr, rng, cap_pos=150_000, cap_neg=300_000)
            xb, _ = tab.feats_task_b(tr, rows=rows, graph=True, A=A); XB.append(xb); yB.append(tr.mask[rows])
        XA, yA, XB, yB = np.concatenate(XA), np.concatenate(yA), np.concatenate(XB), np.concatenate(yB)
        cnt = np.bincount(yA, minlength=4).astype(float); w = ((cnt.sum() / (4 * cnt)) ** 0.5)[yA]
        ca = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=6).fit(XA, yA, sample_weight=w)
        pw = float(np.clip(np.sqrt((1 - yB.mean()) / yB.mean()), 1, 10))
        cb = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=6).fit(XB, yB.astype(int), sample_weight=np.where(yB, pw, 1.0))
        te = Pool(tgt, "test"); A = nets.build_adj(te.kind, te.loc, te.f, te.t)
        pa = ca.predict(tab.feats_task_a(te, rich=True)).astype(int)
        pm = np.zeros((te.N, te.T, te.m), np.float32)
        for s in range(0, te.N, 25):
            sub = tab._slice(te, s, min(s + 25, te.N)); xb, _ = tab.feats_task_b(sub, graph=True, A=A)
            pm[s:s + sub.N] = cb.predict_proba(xb)[:, 1].reshape(sub.N, sub.T, sub.m)
        ma, _ = task_a_metrics(te.cls.reshape(-1), pa); mb = task_b_metrics(te.mask, score=pm, thr=0.5)
        row = dict(model="WR-Boost", target=tgt, seed=seed, sources="+".join(srcs), mode="zero-shot", frac=0.0, A_macroF1=ma["macro_f1"], A_balAcc=ma["bal_acc"],
                   A_STrecall=ma["minority_recall"], B_F1att=mb["f1_att"], B_PRAUC=mb["pr_auc"], B_TPR=mb["tpr"], B_FPR=mb["fpr"])
        open(out, "a").write(json.dumps(row) + "\n"); print(tgt, seed, round(ma["macro_f1"], 3), round(mb["f1_att"], 3), flush=True)
