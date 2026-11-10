import os, numpy as np
from powersys import Grid
from mldata import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = []
names = {"case14": "IEEE 14-bus", "case39": "NE 39-bus", "case118": "IEEE 118-bus"}
for c in ["case14", "case39", "case118"]:
    g = Grid(c); g0 = np.load(os.path.join(ROOT, "data", f"{c}_grid.npz"))
    tr, va, te = Pool(c, "train"), Pool(c, "val"), Pool(c, "test")
    cls = np.bincount(te.cls.ravel(), minlength=4) / te.cls.size
    rows.append((names[c], g.nb, g.nl, g.m_rtu, len(g.pmu_bus), g.pmu_count, len(g0["vuln"]), g.cfg["thr"], tr.N, va.N, te.N, 100 * te.mask.mean(), 100 * cls[3]))
L = [r"\begin{tabular}{lccc}", r"\toprule", r"Quantity & " + " & ".join(names.values()) + r" \\", r"\midrule"]
labels = ["Buses / branches", "RTU meters $m$", "PMUs / channels", "Vulnerable meters", "$\delta$ (p.u.)", "Scenarios tr/va/te", "Attacked (\%, test)", "ST-RDCA (\%, test)"]
vals = [
    [f"{r[1]} / {r[2]}" for r in rows], [f"{r[3]}" for r in rows], [f"{r[4]} / {r[5]}" for r in rows], [f"{r[6]}" for r in rows],
    [f"{r[7]:.2f}" for r in rows], [f"{r[8]} / {r[9]} / {r[10]}" for r in rows], [f"{r[11]:.1f}" for r in rows], [f"{r[12]:.1f}" for r in rows]]
for lab, v in zip(labels, vals):
    L.append(lab + " & " + " & ".join(v) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "paper", "tables", "systems.tex"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
