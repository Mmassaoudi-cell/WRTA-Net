"""Per-snapshot cost of the physics pipeline: source (RTU WLS + HYB-SE + reconstruction) vs PMU-LSE front-end used by the learned models."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time, numpy as np, pandas as pd
from powersys import Grid
rows = []
for case in ["case14", "case39", "case118"]:
    g = Grid(case); rng = np.random.default_rng(0)
    z0, z = g.measure(g.V0, rng); ph = g.pmu_measure(g.V0, rng)
    def t(fn, n=30):
        fn(); t0 = time.perf_counter()
        for _ in range(n): fn()
        return 1e3 * (time.perf_counter() - t0) / n
    out = g.wls(z, return_all=True); Vw, G = out[1] * np.exp(1j * out[2]), out[4]; nit = out[5]
    t_wls = t(lambda: g.wls(z))
    t_hyb = t(lambda: g.hybrid_se(Vw, G, ph))
    t_rec = t(lambda: g.full_meas(np.abs(Vw), np.angle(Vw)))
    t_pmu = t(lambda: g.pmu_lse(ph))
    rows.append(dict(case=case, m=g.m_rtu, nb=g.nb, wls_iters=nit, t_wls_ms=t_wls, t_hybse_ms=t_hyb, t_recon_ms=t_rec,
                     t_source_total_ms=t_wls + t_hyb + t_rec, t_pmu_lse_recon_ms=t_pmu + t_rec))
df = pd.DataFrame(rows)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
df.to_csv(os.path.join(ROOT, "results", "efficiency_physics.csv"), index=False)
print(df.round(3).to_string())
