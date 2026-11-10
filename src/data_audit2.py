"""v2 data audit: manifest of every scenario of every pool, exact-duplicate and near-duplicate checks across pools, class balance."""
import os, hashlib
import numpy as np, pandas as pd, torch
from mldata import Pool, CLASSES

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
POOLS = ["train", "val", "test", "train_a", "val_a", "r1_train_a", "r1_val_a", "r1_test", "r2_train_a", "r2_val_a", "r2_test"] + \
        [f"val_{t}" for t in ("pmu10", "adapt", "off", "topo", "topof")] + [f"rob_{t}" for t in ("n3", "n4", "off", "adapt", "topo", "topof", "pmu3", "pmu10", "pmu30", "pmu100")]
rows, lines = [], []
for case in ["case14", "case39", "case118"]:
    pools = {p: Pool(case, p) for p in POOLS if os.path.exists(os.path.join(ROOT, "data", f"{case}_{p}.npz"))}
    hs = {p: {hashlib.md5(d.z[i].tobytes()).hexdigest() for i in range(d.N)} for p, d in pools.items()}
    hr = {p: {hashlib.md5(d.z[i, t].tobytes()).hexdigest() for i in range(d.N) for t in range(d.T)} for p, d in pools.items()}
    names = list(pools)
    dup_s = sum(len(hs[a] & hs[b]) for i, a in enumerate(names) for b in names[i + 1:])
    dup_r = sum(len(hr[a] & hr[b]) for i, a in enumerate(names) for b in names[i + 1:])
    for p, d in pools.items():
        for i in range(d.N):
            rows.append(dict(case=case, pool=p, scenario_id=i, kind=CLASSES[d.meta["kind"][i]], noise_scale=d.meta["noise_scale"][i], stealth=d.meta["stealth"][i],
                             t_a=d.meta["t_a"][i], L=d.meta["L"][i], n_compromised=d.meta["n_comp"][i], event_id=d.meta["ev"][i], n_bad_data=d.meta["n_bad"][i],
                             off_prior=d.meta.get("off_prior", np.zeros(d.N))[i], pmu_scale=d.meta.get("pmu_scale", np.ones(d.N))[i], topology_error=d.meta.get("topo", np.zeros(d.N))[i]))
    def emb(d): return torch.tensor(d.R[:, :12].reshape(d.N, -1)).float().cuda()
    tr = emb(pools["train_a"]); dtt = torch.cdist(tr, tr); dtt.fill_diagonal_(float("inf")); nn_tt = dtt.min(1).values
    near = {}
    for p in ["val_a", "test", "r1_test", "r2_test"]:
        if p in pools:
            dv = torch.cdist(emb(pools[p]), tr).min(1).values
            near[p] = (float(dv.median()), int((dv < nn_tt.min()).sum()))
    cls_cnt = {p: np.bincount(pools[p].cls.ravel(), minlength=4) / pools[p].cls.size for p in ["train_a", "val_a", "test"]}
    lines.append(f"### {case}\n- pools audited: {len(pools)}; exact duplicate scenarios across all pool pairs: {dup_s}; exact duplicate (scenario, scan) RTU vectors: {dup_r}\n"
                 f"- median nearest-train_a distance: " + ", ".join(f"{p} {v[0]:.1f}" for p, v in near.items()) + f"; median train_a->train_a nearest neighbour {float(nn_tt.median()):.1f}; "
                 "scenarios closer to a train_a scenario than the closest train_a pair: " + ", ".join(f"{p} {v[1]}" for p, v in near.items()) + "\n"
                 "- scan class frequencies (Normal, MDDA, LT-RDCA, ST-RDCA): " + "; ".join(f"{p}: " + ", ".join(f"{x:.3f}" for x in c) for p, c in cls_cnt.items()) + "\n"
                 "- attacked-meter fraction: " + ", ".join(f"{p} {pools[p].mask.mean():.4f}" for p in ["train_a", "val_a", "test"]) + "\n")
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "DATA_SPLIT_MANIFEST.csv"), index=False)
open(os.path.join(ROOT, "results2", "data_audit_stats.md"), "w").write("\n".join(lines))
print("\n".join(lines))
