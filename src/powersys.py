"""AC power-system core: grid container, Newton-Raphson PF, measurement model, WLS SE,
PMU linear SE and the hybrid (HYB-SE) estimator of De & Sodhi (2025), Eq. (6)-(8).

All quantities per-unit on 100 MVA.  Internal bus index = row index of the MATPOWER-format ppc
that pandapower builds from the standard case files.
"""
from __future__ import annotations
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from scipy.optimize import milp, LinearConstraint, Bounds

SIGMA = dict(V=0.006, P=0.01, Q=0.01, Pf=0.01, Qf=0.01)      # Tables II/IV of the source paper
import os
PMU_SIGMA_MAG = 1e-5
PMU_SIGMA_ANG = float(os.environ.get('PMU_ANG', np.radians(0.001)))   # 0.001 deg (reading of 'SD 0.001 (Ang)' in Tables II/IV)

CONFIG = {
    # counts of installed RTU meters: (V, P-inj = Q-inj, flow lines (Pf=Qf)), PMU buses (0-based set later)
    "case14": dict(nV=7, nInj=8, nFlow=16, pmu=[2, 7, 10, 13], thr=0.03,
                   vuln=(2, 3, 7, 7)),
    "case39": dict(nV=30, nInj=30, nFlow=32, pmu=[2, 6, 9, 10, 11, 14, 17, 19, 20, 22, 23, 26, 37, 38], thr=0.05,
                   vuln=(10, 10, 23, 23)),
    "case118": dict(nV=90, nInj=90, nFlow=100,
                    pmu=[2, 5, 10, 12, 15, 17, 21, 25, 29, 34, 37, 41, 45, 49, 52, 56, 62, 64, 72, 73, 75, 77, 80,
                         85, 87, 90, 94, 102, 105, 110, 114, 116],
                    thr=0.04, vuln=(22, 22, 81, 81)),
}


class Grid:
    def __init__(self, name: str, seed: int = 0):
        self.name = name
        net = getattr(pn, name)()
        pp.runpp(net)
        ppc = net._ppc
        self.baseMVA = float(ppc["baseMVA"])
        bus, br, gen = ppc["bus"], ppc["branch"], ppc["gen"]
        self.nb, self.nl = bus.shape[0], br.shape[0]
        assert np.allclose(bus[:, 0], np.arange(self.nb))
        it = ppc["internal"]
        self.Ybus = np.asarray(it["Ybus"].todense())
        self.Yf = np.asarray(it["Yf"].todense())
        self.Yt = np.asarray(it["Yt"].todense())
        self.f = br[:, 0].real.astype(int)
        self.t = br[:, 1].real.astype(int)
        self.br_x = br[:, 3].real
        self.br_tap = np.where(br[:, 8].real == 0, 1.0, br[:, 8].real)
        self.ref = int(np.where(bus[:, 1] == 3)[0][0])
        self.gen_bus = gen[:, 0].real.astype(int)
        self.Vg = np.ones(self.nb)
        self.Vg[self.gen_bus] = gen[:, 5].real
        self.pv = np.array(sorted(set(np.where(bus[:, 1] == 2)[0]) - {self.ref}))
        self.pq = np.array(sorted(set(range(self.nb)) - set(self.pv) - {self.ref}))
        self.Pd0 = bus[:, 2].real / self.baseMVA
        self.Qd0 = bus[:, 3].real / self.baseMVA
        self.Pg0 = np.zeros(self.nb)
        np.add.at(self.Pg0, self.gen_bus, gen[:, 1].real / self.baseMVA)
        self.Cf = np.zeros((self.nl, self.nb)); self.Cf[np.arange(self.nl), self.f] = 1
        self.Ct = np.zeros((self.nl, self.nb)); self.Ct[np.arange(self.nl), self.t] = 1
        self.nonref = np.array([i for i in range(self.nb) if i != self.ref])
        self.cfg = CONFIG[name]
        self.V0 = self.runpf(self.Pd0, self.Qd0, self.Pg0)[0]
        self._place(seed)

    # ------------------------------------------------------------------ power flow
    def runpf(self, Pd, Qd, Pg, V0=None, Ybus=None, tol=1e-9, maxit=30):
        Y = self.Ybus if Ybus is None else Ybus
        Sbus = (Pg - Pd) - 1j * Qd
        V = np.ones(self.nb, complex) * self.Vg if V0 is None else V0.copy()
        V[self.ref] = self.Vg[self.ref] * np.exp(1j * 0.0)
        pvpq = np.r_[self.pv, self.pq]
        for it in range(maxit):
            S = V * np.conj(Y @ V)
            mis = S - Sbus
            F = np.r_[mis[pvpq].real, mis[self.pq].imag]
            if np.max(np.abs(F)) < tol:
                return V, True, it
            dVa, dVm = dSbus_dV(Y, V)
            J = np.block([[dVa[np.ix_(pvpq, pvpq)].real, dVm[np.ix_(pvpq, self.pq)].real],
                          [dVa[np.ix_(self.pq, pvpq)].imag, dVm[np.ix_(self.pq, self.pq)].imag]])
            dx = np.linalg.solve(J, -F)
            Va, Vm = np.angle(V), np.abs(V)
            Va[pvpq] += dx[:len(pvpq)]
            Vm[self.pq] += dx[len(pvpq):]
            V = Vm * np.exp(1j * Va)
        return V, False, maxit

    # ------------------------------------------------------------------ meter placement
    def _place(self, seed):
        c = self.cfg
        rng = np.random.default_rng(1000 + seed)
        for _ in range(200):
            vb = np.sort(rng.choice(self.nb, c["nV"], replace=False))
            ib = np.sort(rng.choice(self.nb, c["nInj"], replace=False))
            fl = np.sort(rng.choice(self.nl, c["nFlow"], replace=False))
            self.meas = dict(V=vb, P=ib, Q=ib, Pf=fl, Qf=fl)
            if self.rtu_observable():
                break
        else:
            raise RuntimeError("no observable RTU placement found")
        self.m_rtu = c["nV"] + 2 * c["nInj"] + 2 * c["nFlow"]
        sig = np.r_[np.full(c["nV"], SIGMA["V"]), np.full(c["nInj"], SIGMA["P"]), np.full(c["nInj"], SIGMA["Q"]),
                    np.full(c["nFlow"], SIGMA["Pf"]), np.full(c["nFlow"], SIGMA["Qf"])]
        self.sigma = sig
        # measurement -> bus / line incidence (for graph construction): kind id, location id
        kind = np.r_[np.zeros(c["nV"]), np.ones(c["nInj"]), 2 * np.ones(c["nInj"]), 3 * np.ones(c["nFlow"]),
                     4 * np.ones(c["nFlow"])].astype(int)
        loc = np.r_[vb, ib, ib, fl, fl]
        self.m_kind, self.m_loc = kind, loc
        self._pmu_setup()

    def rtu_observable(self):
        V = self.V0
        J = self.meas_jac(np.abs(V), np.angle(V))
        return np.linalg.matrix_rank(J) == 2 * self.nb - 1

    # ------------------------------------------------------------------ measurement model (RTU, polar)
    def full_meas(self, Vm, Va):
        V = Vm * np.exp(1j * Va)
        S = V * np.conj(self.Ybus @ V)
        Sf = V[self.f] * np.conj(self.Yf @ V)
        m = self.meas
        return np.r_[Vm[m["V"]], S.real[m["P"]], S.imag[m["Q"]], Sf.real[m["Pf"]], Sf.imag[m["Qf"]]]

    def meas_jac(self, Vm, Va):
        V = Vm * np.exp(1j * Va)
        dSa, dSm = dSbus_dV(self.Ybus, V)
        dSfa, dSfm = dSbr_dV(self.Yf, self.f, self.Cf, V)
        m = self.meas
        nb = self.nb
        rows_a, rows_m = [], []
        Ia = np.zeros((len(m["V"]), nb)); Im = np.zeros((len(m["V"]), nb)); Im[np.arange(len(m["V"])), m["V"]] = 1
        rows_a += [Ia, dSa.real[m["P"]], dSa.imag[m["Q"]], dSfa.real[m["Pf"]], dSfa.imag[m["Qf"]]]
        rows_m += [Im, dSm.real[m["P"]], dSm.imag[m["Q"]], dSfm.real[m["Pf"]], dSfm.imag[m["Qf"]]]
        return np.hstack([np.vstack(rows_a)[:, self.nonref], np.vstack(rows_m)])

    def measure(self, V, rng, noise_scale=1.0):
        z0 = self.full_meas(np.abs(V), np.angle(V))
        return z0, z0 + rng.standard_normal(z0.shape) * self.sigma * noise_scale

    def wls(self, z, sigma=None, x0=None, tol=1e-5, maxit=15, return_all=False):
        sig = self.sigma if sigma is None else sigma
        w = 1.0 / sig ** 2
        nb = self.nb
        x = np.r_[np.zeros(nb - 1), np.ones(nb)] if x0 is None else x0.copy()
        for it in range(maxit):
            Va = np.zeros(nb); Va[self.nonref] = x[:nb - 1]; Vm = x[nb - 1:]
            h = self.full_meas(Vm, Va)
            J = self.meas_jac(Vm, Va)
            r = z - h
            Jw = J * w[:, None]
            G = J.T @ Jw
            dx = np.linalg.solve(G, Jw.T @ r)
            x = x + dx
            if np.max(np.abs(dx)) < tol:
                break
        Va = np.zeros(nb); Va[self.nonref] = x[:nb - 1]; Vm = x[nb - 1:]
        if return_all:
            return x, Vm, Va, J, G, it + 1
        return Vm * np.exp(1j * Va)

    # ------------------------------------------------------------------ PMU measurement model (rectangular, linear)
    def _pmu_setup(self):
        c = self.cfg
        self.pmu_bus = np.array(c["pmu"]) - 1
        nb = self.nb
        rows_v = [];  rows_i = []
        self.pmu_I_lines = []
        for b in self.pmu_bus:
            for l in range(self.nl):
                if self.f[l] == b:
                    self.pmu_I_lines.append((l, "f"))
                elif self.t[l] == b:
                    self.pmu_I_lines.append((l, "t"))
        Vsel = np.zeros((len(self.pmu_bus), nb), complex); Vsel[np.arange(len(self.pmu_bus)), self.pmu_bus] = 1
        Irow = np.array([self.Yf[l] if s == "f" else self.Yt[l] for l, s in self.pmu_I_lines])
        self.A_pmu = np.vstack([Vsel, Irow])                  # complex phasor = A_pmu @ V
        self.n_pmu_ph = self.A_pmu.shape[0]
        self.pmu_count = len(self.pmu_bus) + len(self.pmu_I_lines)
        # real-linear rect model:  [re; im] = [[Re A, -Im A],[Im A, Re A]] [Vre; Vim]
        Ar, Ai = self.A_pmu.real, self.A_pmu.imag
        self.H_pmu = np.block([[Ar, -Ai], [Ai, Ar]])          # rows: first all re, then all im
        self._pmu_weights(self.V0)
        # PMU-only LSE gain
        Wd = self.W_pmu_diag
        G = self.H_pmu.T @ (Wd[:, None] * self.H_pmu)
        self.pmu_observable = np.linalg.matrix_rank(self.H_pmu) == 2 * nb
        self.K_pmu = np.linalg.solve(G, (self.H_pmu * Wd[:, None]).T) if self.pmu_observable else None
        # build helper for reconstruction h(x) from rectangular state
        self.W_pmu_blocks = None

    def _pmu_weights(self, Vref):
        """Rectangular-coordinate PMU noise covariance from polar (mag, ang) SDs at the nominal phasors.
        Returned as per-measurement diagonal approximation of the inverse covariance (re/im decoupled)."""
        ph = self.A_pmu @ Vref
        mag, ang = np.abs(ph), np.angle(ph)
        sm, sa = PMU_SIGMA_MAG, PMU_SIGMA_ANG
        var_re = (np.cos(ang) * sm) ** 2 + (mag * np.sin(ang) * sa) ** 2
        var_im = (np.sin(ang) * sm) ** 2 + (mag * np.cos(ang) * sa) ** 2
        var_re = np.maximum(var_re, 1e-12); var_im = np.maximum(var_im, 1e-12)
        self.W_pmu_diag = np.r_[1 / var_re, 1 / var_im]

    def pmu_measure(self, V, rng, noise_scale=1.0):
        noise_scale = noise_scale * float(os.environ.get('PMU_NOISE_SCALE', 1.0))     # unmodelled PMU degradation (robustness test)
        ph = self.A_pmu @ V
        mag = np.abs(ph) + rng.standard_normal(ph.shape) * PMU_SIGMA_MAG * noise_scale
        ang = np.angle(ph) + rng.standard_normal(ph.shape) * PMU_SIGMA_ANG * noise_scale
        return mag * np.exp(1j * ang)

    def pmu_lse(self, ph):
        zc = np.r_[ph.real, ph.imag]
        xr = self.K_pmu @ zc
        return xr[:self.nb] + 1j * xr[self.nb:]

    def hybrid_se(self, V_rtu, G_pol, ph):
        """HYB-SE (Eq. 6-8): rectangular states from [RTU-WLS states ; PMU phasors] weighted by the covariances.
        V_rtu: complex WLS states, G_pol: WLS gain matrix (states [theta_nonref, Vm])."""
        nb = self.nb
        Vm, Va = np.abs(V_rtu), np.angle(V_rtu)
        # polar -> rectangular Jacobian for the 2nb-1 independent coordinates (drop Vim of reference bus)
        T = np.zeros((2 * nb, 2 * nb - 1))
        for k, b in enumerate(self.nonref):
            T[b, k] = -Vm[b] * np.sin(Va[b]); T[nb + b, k] = Vm[b] * np.cos(Va[b])
        for b in range(nb):
            T[b, nb - 1 + b] = np.cos(Va[b]); T[nb + b, nb - 1 + b] = np.sin(Va[b])
        Cpol = np.linalg.inv(G_pol)
        Crect = T @ Cpol @ T.T                                      # 2nb x 2nb (rank 2nb-1 -> drop ref Vim)
        keep = np.r_[np.arange(nb), nb + self.nonref]
        Cr = Crect[np.ix_(keep, keep)] + 1e-14 * np.eye(2 * nb - 1)
        Wr = np.linalg.inv(Cr)
        # stacked measurement model
        Hr = np.zeros((2 * nb - 1, 2 * nb)); Hr[np.arange(2 * nb - 1), keep] = 1
        zr = np.r_[V_rtu.real, V_rtu.imag[self.nonref]]
        zp = np.r_[ph.real, ph.imag]
        Wp = self.W_pmu_diag
        G = Hr.T @ Wr @ Hr + self.H_pmu.T @ (Wp[:, None] * self.H_pmu)
        b = Hr.T @ Wr @ zr + self.H_pmu.T @ (Wp * zp)
        x = np.linalg.solve(G, b)
        return x[:nb] + 1j * x[nb:]

    def outaged(self, l):
        """Copy of the grid with line l tripped (true system); estimators keep the nominal topology."""
        import copy
        g2 = copy.copy(self)
        f, t = self.f[l], self.t[l]
        Y = self.Ybus.copy()
        Y[f, f] -= self.Yf[l, f]; Y[f, t] -= self.Yf[l, t]; Y[t, f] -= self.Yt[l, f]; Y[t, t] -= self.Yt[l, t]
        g2.Ybus = Y
        g2.Yf = self.Yf.copy(); g2.Yf[l] = 0
        g2.Yt = self.Yt.copy(); g2.Yt[l] = 0
        A = self.A_pmu.copy()
        for k, (ll, s) in enumerate(self.pmu_I_lines):
            if ll == l:
                A[len(self.pmu_bus) + k] = 0
        g2.A_pmu = A
        return g2

    # ------------------------------------------------------------------ DC / PTDF utilities
    def ptdf(self):
        nb, nl = self.nb, self.nl
        A = self.Cf - self.Ct
        b = 1.0 / (self.br_x * self.br_tap)
        Bbus = A.T @ (b[:, None] * A)
        X = np.zeros((nb, nb))
        X[np.ix_(self.nonref, self.nonref)] = np.linalg.inv(Bbus[np.ix_(self.nonref, self.nonref)])
        return (b[:, None] * A) @ X          # nl x nb  (= Bx A X, Eq. 4)


# ====================================================================== derivatives (MATPOWER style)
def dSbus_dV(Y, V):
    Ibus = Y @ V
    dV = np.diag(V); dI = np.diag(Ibus); dVn = np.diag(V / np.abs(V))
    dSm = dV @ np.conj(Y @ dVn) + np.conj(dI) @ dVn
    dSa = 1j * dV @ np.conj(dI - Y @ dV)
    return dSa, dSm


def dSbr_dV(Yf, f, Cf, V):
    If = Yf @ V
    Vf = V[f]
    dV = np.diag(V); dVn = np.diag(V / np.abs(V)); dIf = np.diag(If); dVf = np.diag(Vf)
    dSa = 1j * (np.conj(dIf) @ Cf @ dV - dVf @ np.conj(Yf @ dV))
    dSm = dVf @ np.conj(Yf @ dVn) + np.conj(dIf) @ Cf @ dVn
    return dSa, dSm


# ====================================================================== Stage-1 vulnerability analysis
def vulnerable_set(g: Grid):
    """PTDF-based selection of the critical meters (Eq. 1-4 of the source paper).
    Returns index arrays into g.meas[...] (positions in the RTU vector)."""
    nv, ninj, nPf, nQf = g.cfg["vuln"]
    PT = g.ptdf()                                       # N-PTDF (nl x nb)
    nodal = np.abs(PT).sum(axis=0)                      # per-bus total |flow change| (Eq. 3-4)
    # BR-PTDF vulnerability index per line (Eq. 1): mean over (s,r) pairs of |PTDF_{s->r,l}|
    psi = np.array([np.mean(np.abs(PT[l][:, None] - PT[l][None, :])) for l in range(g.nl)])
    inst = g.meas
    # ILP of Eq. (2) read as: pick exactly nPf installed flow meters maximizing sum(psi)
    cand = inst["Pf"]
    k = min(nPf, len(cand))
    res = milp(c=-psi[cand], constraints=LinearConstraint(np.ones((1, len(cand))), k, k),
               integrality=np.ones(len(cand)), bounds=Bounds(0, 1))
    sel_lines = cand[np.round(res.x).astype(bool)]
    # N-PTDF top buses among installed injection meters -> P-injection and the same-index voltage meters
    ib = inst["P"]
    top_inj = ib[np.argsort(-nodal[ib])[:ninj]]
    vb = inst["V"]
    top_v = vb[np.argsort(-nodal[vb])[:nv]]
    m = {}
    off = 0
    idx_v = np.where(np.isin(vb, top_v))[0]
    idx_p = len(vb) + np.where(np.isin(ib, top_inj))[0]
    off_pf = len(vb) + 2 * len(ib)
    idx_pf = off_pf + np.where(np.isin(inst["Pf"], sel_lines))[0]
    idx_qf = off_pf + len(inst["Pf"]) + np.where(np.isin(inst["Qf"], sel_lines))[0]
    return np.sort(np.r_[idx_v, idx_p, idx_pf, idx_qf]), dict(V=idx_v, P=idx_p, Pf=idx_pf, Qf=idx_qf)


if __name__ == "__main__":
    import time
    for nm in ["case14", "case39", "case118"]:
        g = Grid(nm)
        print(nm, "nb", g.nb, "nl", g.nl, "m_rtu", g.m_rtu, "pmu phasors", g.pmu_count, "pmu observable", g.pmu_observable)
        net = getattr(pn, nm)(); pp.runpp(net)
        print("  PF check max |V-V_pp|:", np.max(np.abs(np.abs(g.V0) - net.res_bus.vm_pu.values)),
              " ang:", np.max(np.abs(np.degrees(np.angle(g.V0)) - net.res_bus.va_degree.values)))
        rng = np.random.default_rng(0)
        z0, z = g.measure(g.V0, rng)
        t0 = time.time(); Vw = g.wls(z); tw = time.time() - t0
        print("  WLS err", np.max(np.abs(Vw - g.V0)), "time %.1f ms" % (1e3 * tw))
        vs, parts = vulnerable_set(g)
        print("  vulnerable meters:", len(vs), {k: len(v) for k, v in parts.items()})
