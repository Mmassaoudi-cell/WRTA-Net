import os, json, time, joblib, numpy as np
from mldata import Pool
import tab
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results2")
for case in ["case14", "case39", "case118"]:
    pool = Pool(case, "test"); sub = tab._slice(pool, 0, 10)
    for name in ["LGBM", "XGB", "CatBoost", "C5-WR-Boost"]:
        d = joblib.load(os.path.join(RES, "models", f"{name}_{case}_s0.pkl"))
        def run():
            d["clf"].predict(tab.feats_task_a(sub, rich=d["rich"]))
            Xb, _ = tab.feats_task_b(sub, graph=d["graph"], A=d["A"]); d["clb"].predict_proba(Xb)
        run(); t0 = time.perf_counter(); run(); dt = (time.perf_counter() - t0) / (sub.N * sub.T)
        row = dict(name=name, case=case, m=pool.m, lat_scan_cpu_ms=1e3 * dt, note="feature build + both heads, amortised over 240 scans")
        open(os.path.join(RES, "efficiency_tab.jsonl"), "a").write(json.dumps(row) + "\n"); print(row)
