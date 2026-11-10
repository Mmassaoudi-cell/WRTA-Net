"""Time-series scenario generator: load dynamics, replay attacks (MDDA / LT-RDCA / ST-RDCA), bad data,
RTU WLS + hybrid SE + PMU LSE residual extraction.  One scenario -> dict of arrays."""
from __future__ import annotations
import numpy as np
from powersys import Grid, vulnerable_set

T = 24                      # seconds (1 Hz RTU scan)
CLASSES = ["Normal", "MDDA", "LT-RDCA", "ST-RDCA"]
NOISE_SCALES = [0.5, 1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------------------------- load dynamics
def load_profile(g: Grid, rng, T_len, big=False, kind=None):
    """Return load factors (T_len, nb) applied to both Pd and Qd."""
    has_load = np.where(g.Pd0 > 1e-6)[0]
    F = np.ones((T_len, g.nb))
    # slow random walk
    F *= np.cumprod(1 + 0.002 * rng.standard_normal((T_len, 1)), axis=0)
    amp = rng.uniform(0.3, 0.6) if big else rng.uniform(0.10, 0.30)
    frac = rng.uniform(0.4, 0.8) if big else rng.uniform(0.25, 0.6)
    k = max(2, int(frac * len(has_load)))
    sel = rng.choice(has_load, k, replace=False)
    sign = rng.choice([-1, 1])
    t = np.arange(T_len)
    t_d = rng.integers(1, 4) if big else rng.integers(2, 12)
    ramp_len = rng.uniform(1, 3) if big else rng.uniform(2, 6)
    ramp = np.clip((t - t_d) / ramp_len, 0, 1)
    osc_a = rng.uniform(0.05, 0.15) if big else rng.uniform(0.0, 0.05)
    osc = osc_a * np.sin(2 * np.pi * rng.uniform(0.1, 0.35) * t + rng.uniform(0, 6.28)) * (t >= t_d)
    prof = sign * amp * ramp + osc
    F[:, sel] *= (1 + prof)[:, None]
    return F


def solve_series(g: Grid, F, fault=None):
    """AC PF along a load-factor series. fault = list of (t_start, t_end, bus, y_fault)."""
    T_len = F.shape[0]
    V = np.zeros((T_len, g.nb), complex)
    Vprev = g.V0
    Pg_scale_base = g.Pd0.sum()
    for t in range(T_len):
        Pd, Qd = g.Pd0 * F[t], g.Qd0 * F[t]
        ratio = Pd.sum() / Pg_scale_base
        Pg = g.Pg0 * ratio
        Y = g.Ybus
        if fault is not None:
            for (ts, te, b, yf) in fault:
                if ts <= t < te:
                    Y = Y.copy(); Y[b, b] += yf
        Vt, ok, _ = g.runpf(Pd, Qd, Pg, V0=Vprev, Ybus=Y)
        if not ok:
            Vt, ok, _ = g.runpf(Pd, Qd, Pg, V0=None, Ybus=Y)
        if not ok:
            raise RuntimeError("PF diverged")
        V[t] = Vt
        Vprev = Vt if (fault is None or not any(ts <= t < te for ts, te, *_ in fault)) else g.V0
    return V


# ---------------------------------------------------------------------------------------------- event library
def make_events(g: Grid, n_each: int, seed: int):
    """Historical, unrelated event records replayed by the attacker.
    Returns dict(kind -> list of (Le, m) noisy measurement sequences)."""
    rng = np.random.default_rng(seed)
    out = {"dist": [], "fault": []}
    pq = g.pq
    while len(out["dist"]) < n_each:
        Le = int(rng.integers(8, 13))
        F = load_profile(g, rng, Le, big=True)
        try:
            V = solve_series(g, F)
        except RuntimeError:
            continue
        out["dist"].append(np.array([g.measure(v, rng)[1] for v in V]))
    while len(out["fault"]) < n_each:
        kf = int(rng.integers(2, 4))
        Le = kf + 4
        b = int(rng.choice(pq))
        yf = rng.uniform(3, 12) * np.exp(-1j * 1.3)
        F = np.ones((Le, g.nb))
        try:
            V = solve_series(g, F, fault=[(2, 2 + kf, b, yf)])
        except RuntimeError:
            continue
        out["fault"].append(np.array([g.measure(v, rng)[1] for v in V]))
    return out


# ---------------------------------------------------------------------------------------------- one scenario
def make_scenario(g: Grid, vuln_idx, events, seed: int, kind: str | None = None, noise_scale: float | None = None,
                  stealth: int | None = None, bad_data_p=0.2, T_len=T, wls_stage=True, F=None, t_a=None, L=None, rho=None,
                  gap=None, ev_idx=None, off_prior_p=0.2, outage=False, pmu_scale=1.0, topo_known=False, aug=False,
                  renoise_sigma=0.7):
    rng = np.random.default_rng(seed)
    if kind is None:
        kind = CLASSES[int(rng.integers(0, 4))]
    if noise_scale is None:
        noise_scale = float(rng.choice(NOISE_SCALES))
    if aug:                                              # augmented training/validation mixture (robust training distribution)
        pmu_scale = float(rng.choice([1.0, 3.0, 10.0, 30.0], p=[0.4, 0.2, 0.25, 0.15]))
        outage = bool(rng.random() < 0.12)
        topo_known = bool(rng.random() < 0.5)
        off_prior_p = 0.4
        if stealth is None:
            stealth = int(rng.choice([0, 1, 2], p=[0.3, 0.3, 0.4]))
        renoise_sigma = float(rng.uniform(0.3, 1.2))
    if stealth is None:
        stealth = int(rng.integers(0, 2))              # 0: exact clone, 1: re-noised replay, 2: adaptive (smoothed + jittered + variance-matched)
    m = g.m_rtu
    if F is None:
        F = load_profile(g, rng, T_len)
    V = solve_series(g, F)
    gt = [g] * T_len
    ge = [g] * T_len                                     # grid model used by the (PMU-only) estimator
    t_o = T_len
    tripped = -1
    if outage:                                           # N-1 topology error: true grid loses a line at t_o; estimator keeps the nominal model unless topo_known
        t_o = int(rng.integers(2, 6))
        for _ in range(50):
            l = int(rng.integers(0, g.nl)); g_out = g.outaged(l)
            try:
                V_out = solve_series(g_out, F)
                if np.abs(V_out).min() > 0.8:
                    break
            except Exception:
                continue
        else:
            raise RuntimeError("no feasible outage")
        V = np.concatenate([V[:t_o], V_out[t_o:]]); gt = [g] * t_o + [g_out] * (T_len - t_o)
        tripped = l
    Zc = np.zeros((T_len, m)); Zn = np.zeros((T_len, m)); PH = []
    for t in range(T_len):
        Zc[t], Zn[t] = gt[t].measure(V[t], rng, noise_scale)
        PH.append(gt[t].pmu_measure(V[t], rng, pmu_scale))
    Zo = Zn.copy()
    mask = np.zeros((T_len, m), bool)
    cls = np.zeros(T_len, int)
    t_a = int(rng.integers(5, 9)) if t_a is None else t_a
    meta = dict(kind=kind, noise_scale=noise_scale, stealth=stealth, t_a=t_a, L=0, n_comp=0, ev=-1, off_prior=0,
                pmu_scale=pmu_scale, topo=int(outage), renoise_sigma=renoise_sigma)
    if kind != "Normal":
        cid = CLASSES.index(kind)
        if kind == "MDDA":
            L_ = int(rng.integers(8, 17)); gap = int(rng.choice([2, 3])) if gap is None else gap
            rec_t = np.arange(0, t_a, gap)
            tau = np.arange(t_a)
            rec = np.stack([np.interp(tau, rec_t, Zn[rec_t, i]) for i in range(m)], axis=1) if len(rec_t) > 1 else np.repeat(Zn[:1], t_a, 0)
            seq = rec
        elif kind == "LT-RDCA":
            L_ = int(rng.integers(10, 17)); ei = int(rng.integers(0, len(events["dist"]))) if ev_idx is None else ev_idx; seq = events["dist"][ei]; meta["ev"] = ei
        else:
            L_ = int(rng.integers(2, 5)); ei = int(rng.integers(0, len(events["fault"]))) if ev_idx is None else ev_idx; seq = events["fault"][ei]; meta["ev"] = ei
            seq = seq[2:]                                  # replay starts at fault inception
        L = min(L_ if L is None else L, T_len - t_a)
        rho = rng.uniform(0.5, 1.0) if rho is None else rho
        off_prior = bool(rng.random() < off_prior_p)
        pool_idx = np.arange(m) if off_prior else vuln_idx          # attacker unaware of the PTDF ranking -> any meter
        if off_prior:
            comp = rng.choice(pool_idx, max(1, int(round(rho * len(vuln_idx)))), replace=False)
        else:
            comp = vuln_idx[rng.random(len(vuln_idx)) < rho] if rho < 1 else vuln_idx
        meta["off_prior"] = int(off_prior)
        if len(comp) == 0:
            comp = vuln_idx[:1]
        if stealth == 2:                                 # adaptive attacker: 3-tap smoothing of the record, then fresh noise of matched variance
            seq_s = (np.roll(seq, 1, axis=0) + seq + np.roll(seq, -1, axis=0)) / 3.0
        for j in range(L):
            t = t_a + j
            if stealth == 2:
                rep = seq_s[(j + int(rng.integers(-1, 2))) % len(seq_s)]       # random playback jitter breaks periodicity
            else:
                rep = seq[j % len(seq)]
            val = rep[comp].copy()
            if stealth == 1:
                val = val + rng.standard_normal(len(comp)) * g.sigma[comp] * noise_scale * renoise_sigma
            elif stealth == 2:
                val = val + rng.standard_normal(len(comp)) * g.sigma[comp] * noise_scale * np.sqrt(2.0 / 3.0)
            Zo[t, comp] = val
            mask[t, comp] = True
            cls[t] = cid
        meta.update(L=L, n_comp=len(comp))
    # bad data (natural, non-attack): gross errors on un-compromised meters
    nbad = 0
    if rng.random() < bad_data_p:
        nbad = int(rng.integers(1, 3))
        for _ in range(nbad):
            i = int(rng.integers(0, m)); t = int(rng.integers(0, T_len))
            if not mask[t, i]:
                Zo[t, i] += rng.choice([-1, 1]) * rng.uniform(6, 12) * g.sigma[i] * noise_scale
    meta["n_bad"] = nbad
    # estimation stage ----------------------------------------------------------------
    Zh = np.zeros((T_len, m)); Zp = np.zeros((T_len, m)); LNR = np.zeros((T_len, m)); CHI = np.zeros(T_len)
    Vwls = np.zeros((T_len, g.nb), complex); Vhyb = np.zeros_like(Vwls)
    for t in range(T_len):
        x, Vm, Va, J, G, nit = g.wls(Zo[t], return_all=True)
        Vw = Vm * np.exp(1j * Va)
        Vwls[t] = Vw
        r = Zo[t] - g.full_meas(Vm, Va)
        w = g.sigma ** 2
        Ginv = np.linalg.inv(G)
        var_r = w - np.einsum("ij,jk,ik->i", J, Ginv, J)
        LNR[t] = np.abs(r) / np.sqrt(np.maximum(var_r, 1e-12))
        CHI[t] = np.sum((r / g.sigma) ** 2)
        Vh = g.hybrid_se(Vw, G, PH[t])
        Vhyb[t] = Vh
        Zh[t] = g.full_meas(np.abs(Vh), np.angle(Vh))
        Vp = ge[t].pmu_lse(PH[t])
        Zp[t] = ge[t].full_meas(np.abs(Vp), np.angle(Vp))
    tflag = np.zeros((T_len, m), np.float32)             # topology-processor flag: meters touching the end buses of a tripped line (if known)
    if outage and topo_known and tripped >= 0:
        ends = {int(g.f[tripped]), int(g.t[tripped])}
        for i in range(m):
            bs = {int(g.m_loc[i])} if g.m_kind[i] <= 2 else {int(g.f[g.m_loc[i]]), int(g.t[g.m_loc[i]])}
            if bs & ends:
                tflag[t_o:, i] = 1.0
    meta["topo_flag"] = int(outage and topo_known)
    out = dict(z=Zo.astype(np.float32), zhyb=Zh.astype(np.float32), zpmu=Zp.astype(np.float32),
               lnr=LNR.astype(np.float32), chi=CHI.astype(np.float32), mask=mask, cls=cls,
               ztrue=Zc.astype(np.float32), Vtrue=V.astype(np.complex64), Vwls=Vwls.astype(np.complex64),
               Vhyb=Vhyb.astype(np.complex64), tflag=tflag, meta=meta)
    return out
