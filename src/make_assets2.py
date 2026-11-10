"""v2 manuscript assets: every table (LaTeX) and figure is generated from results2/*; numbers quoted in the text are written to paper/numbers.tex."""
import os, json, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stats2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, TAB, FIG = os.path.join(ROOT, "results2"), os.path.join(ROOT, "paper", "tables"), os.path.join(ROOT, "paper", "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.linewidth": 0.6, "mathtext.fontset": "stix"})
PROP = "V2c"
CASES = ["case14", "case39", "case118"]; CN = {"case14": "IEEE 14", "case39": "NE 39", "case118": "IEEE 118"}; TAGN = ["Fourteen", "ThirtyNine", "OneEighteen"]
DISP = {PROP: "Proposed", "SOURCE": "Source (HYB-SE)", "LNR": "LNR test", "CUSUM": "CUSUM", "SOURCE-ADAPT": "Source, noise-aware$^\\dagger$", "LR": "LR", "RF": "RF", "ET": "ET",
        "XGB": "XGBoost", "LGBM": "LightGBM", "CatBoost": "CatBoost", "MLP": "MLP", "TCN": "TCN", "GRU": "GRU", "Transformer": "Transformer", "FT-Transformer": "FT-Transformer",
        "GCN": "GCN", "GAT": "GAT", "Mamba": "Mamba-style SSM", "C5-WR-Boost": "WR-Boost"}
ORDER_RULE = ["SOURCE", "LNR", "CUSUM"]
ORDER_ML = ["LR", "RF", "ET", "XGB", "LGBM", "CatBoost", "MLP", "TCN", "GRU", "Transformer", "FT-Transformer", "GCN", "GAT", "Mamba", "C5-WR-Boost"]
LEARNED = ORDER_ML
NUM = {}
BS = chr(92)


def macro(k, v): NUM[k] = v


def f3(x): return f"{x:.3f}"


def rows(fn):
    p = os.path.join(RES, fn)
    return pd.DataFrame([json.loads(l) for l in open(p)]) if os.path.exists(p) else pd.DataFrame()


main = rows("final2_main.jsonl").drop_duplicates(["name", "case", "seed"], keep="last")
rules = rows("rules.jsonl")
EXTRA = ["rob_n3", "rob_n4", "rob_off", "rob_adapt", "rob_topo", "rob_topof", "rob_pmu3", "rob_pmu10", "rob_pmu30", "rob_pmu100"]


def metric_rows():
    """long table: name, case, seed, pool, A, B, TPR, FPR"""
    out = []
    for _, r in main.iterrows():
        out.append(dict(name=r["name"], case=r["case"], seed=r["seed"], pool="test", A=r["test_macro_f1"], B=r["test_b_f1_att"], TPR=r["test_b_tpr"], FPR=r["test_b_fpr"],
                        Abal=r["test_bal_acc"], ST=r["test_minority_recall"], PR=r.get("test_b_pr_auc", np.nan)))
        for tag, v in (r.get("extra_test") or {}).items():
            out.append(dict(name=r["name"], case=r["case"], seed=r["seed"], pool=tag, A=v["A_macroF1"], B=v["B_F1att"], TPR=v["B_TPR"], FPR=v["B_FPR"], Abal=v["A_balAcc"], ST=np.nan, PR=np.nan))
    for _, r in rules.iterrows():
        if r["rep"] == 0:
            out.append(dict(name=r["name"], case=r["case"], seed=0, pool=r["pool"], A=np.nan, B=r["b_f1_att"], TPR=r["b_tpr"], FPR=r["b_fpr"], Abal=np.nan, ST=np.nan, PR=r.get("b_pr_auc", np.nan)))
    return pd.DataFrame(out)


L = metric_rows()
T = L[L.pool == "test"]
agg = T.groupby(["name", "case"]).agg(A=("A", "mean"), As=("A", "std"), B=("B", "mean"), Bs=("B", "std"), TPR=("TPR", "mean"), FPR=("FPR", "mean"), n=("seed", "count")).reset_index()
Aidx = agg.set_index(["name", "case"])


def best2(vals):
    v = np.array(vals, float); order = [i for i in np.argsort(-np.nan_to_num(v, nan=-9)) if not np.isnan(v[i])]
    return (order[0] if order else None), (order[1] if len(order) > 1 else None)


# ============================================================================ main table
ok = lambda n: all((n, c) in Aidx.index for c in CASES)
names = [n for n in ORDER_RULE + ORDER_ML + [PROP] if ok(n)]
cells = {}
for key in ("A", "B"):
    for c in CASES:
        cells[(key, c)] = best2([Aidx.loc[(n, c), key] if (n, c) in Aidx.index else np.nan for n in names])
body = []
for i, n in enumerate(names):
    r = [DISP[n]]
    for key in ("A", "B"):
        for c in CASES:
            if (n, c) not in Aidx.index or np.isnan(Aidx.loc[(n, c), key]):
                r.append("--"); continue
            m, s = Aidx.loc[(n, c), key], Aidx.loc[(n, c), key + "s"]
            t = f"{m:.3f}" + (f"{BS}scriptsize$\\pm${s:.3f}" if s > 0 and not np.isnan(s) else "")
            b1, b2 = cells[(key, c)]
            if i == b1: t = BS + "textbf{" + t + "}"
            elif i == b2: t = BS + "underline{" + t + "}"
            r.append(t)
    r.append(f"{100*np.nanmean([Aidx.loc[(n, c), 'FPR'] for c in CASES if (n, c) in Aidx.index]):.2f}")
    body.append(" & ".join(r) + " " + BS + BS)
    if n == "CUSUM" or n == "C5-WR-Boost":
        body.append(BS + "midrule")
if ("SOURCE-ADAPT", "case14") in Aidx.index:
    pass
hdr = (BS + "begin{tabular}{lccccccc}" + "\n" + BS + "toprule\n& " + BS + "multicolumn{3}{c}{Typing: macro-F1} & " + BS + "multicolumn{3}{c}{Localisation: attacked-meter F1} & " + BS + BS + "\n" +
       BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}\nMethod & " + " & ".join(CN[c] for c in CASES) + " & " + " & ".join(CN[c] for c in CASES) + " & FPR (" + BS + "%) " + BS + BS + "\n" + BS + "midrule\n")
open(os.path.join(TAB, "main.tex"), "w", encoding="utf-8").write(hdr + "\n".join(body) + "\n" + BS + "bottomrule\n" + BS + "end{tabular}")

for c, tg in zip(CASES, TAGN):
    macro(f"PropA{tg}", f3(Aidx.loc[(PROP, c), "A"])); macro(f"PropB{tg}", f3(Aidx.loc[(PROP, c), "B"])); macro(f"PropTPR{tg}", f3(Aidx.loc[(PROP, c), "TPR"]))
    macro(f"PropFPR{tg}", f"{100*Aidx.loc[(PROP, c), 'FPR']:.2f}")
    macro(f"SrcB{tg}", f3(Aidx.loc[("SOURCE", c), "B"])); macro(f"SrcTPR{tg}", f3(Aidx.loc[("SOURCE", c), "TPR"]))
    macro(f"CusumB{tg}", f3(Aidx.loc[("CUSUM", c), "B"]))
macro("PropAavg", f3(np.mean([Aidx.loc[(PROP, c), "A"] for c in CASES]))); macro("PropBavg", f3(np.mean([Aidx.loc[(PROP, c), "B"] for c in CASES])))
macro("SrcBavg", f3(np.mean([Aidx.loc[("SOURCE", c), "B"] for c in CASES]))); macro("CusumBavg", f3(np.mean([Aidx.loc[("CUSUM", c), "B"] for c in CASES])))
sA = {n: np.mean([Aidx.loc[(n, c), "A"] for c in CASES]) for n in LEARNED if ok(n)}
sB = {n: np.mean([Aidx.loc[(n, c), "B"] for c in CASES]) for n in LEARNED + ORDER_RULE if ok(n)}
bA, bB = max(sA, key=sA.get), max(sB, key=sB.get)
macro("BestBenchA", DISP[bA]); macro("BestBenchAval", f3(sA[bA])); macro("BestBenchB", DISP[bB]); macro("BestBenchBval", f3(sB[bB]))
print("strongest A", bA, sA[bA], "strongest B", bB, sB[bB])

# ============================================================================ bootstrap comparisons (scenario-level) and win/tie/loss
bench = [n for n in ORDER_RULE + ORDER_ML if ok(n)]
if os.path.exists(os.path.join(RES, "scen")):
    boot = stats2.compare(PROP, bench, [0, 1, 2, 3, 4], [0])
    boot.to_csv(os.path.join(RES, "bootstrap_pairs.csv"), index=False)
    pc = boot[boot.case != "avg"]
    W = []
    for n, g in pc.groupby("benchmark"):
        W.append(dict(benchmark=n, comparisons=len(g), superior=int((g.verdict == "superior").sum()), tie=int((g.verdict == "tie").sum()), inferior=int((g.verdict == "inferior").sum()),
                      avgA=g[(g.metric == "A")]["diff"].mean(), avgB=g[g.metric == "B"]["diff"].mean()))
    W = pd.DataFrame(W).sort_values("benchmark")
    W.to_csv(os.path.join(ROOT, "BENCHMARK_WTL.csv"), index=False)
    av = boot[boot.case == "avg"].set_index(["benchmark", "metric"])
    for n, nk in [("FT-Transformer", "Ft"), ("C5-WR-Boost", "Boost"), ("GAT", "Gat"), ("LGBM", "Lgbm")]:
        for k in ("A", "B"):
            if (n, k) in av.index:
                a_ = av.loc[(n, k)]
                macro(f"Main{nk}{k}", f"{a_['diff']:+.3f}"); macro(f"Main{nk}{k}Lo", f"{a_['ci_lo']:+.3f}"); macro(f"Main{nk}{k}Hi", f"{a_['ci_hi']:+.3f}")
    lines = [BS + "begin{tabular}{lcccc}", BS + "toprule", "Benchmark & " + BS + "multicolumn{2}{c}{$\\Delta$ typing macro-F1} & " + BS + "multicolumn{2}{c}{$\\Delta$ localisation F1} " + BS + BS,
             BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){4-5}", " & mean [95" + BS + "% CI] & sup./tie/inf. & mean [95" + BS + "% CI] & sup./tie/inf. " + BS + BS, BS + "midrule"]
    for n in bench:
        row = [DISP[n]]
        for k in ("A", "B"):
            g = pc[(pc.benchmark == n) & (pc.metric == k)]
            if (n, k) in av.index:
                a = av.loc[(n, k)]
                row += [f"{a['diff']:+.3f} [{a['ci_lo']:+.3f}, {a['ci_hi']:+.3f}]", f"{int((g.verdict=='superior').sum())}/{int((g.verdict=='tie').sum())}/{int((g.verdict=='inferior').sum())}"]
            else:
                row += ["--", "--"]
        lines.append(" & ".join(row) + " " + BS + BS)
    tot = W[["superior", "tie", "inferior"]].sum()
    tA = pc[pc.metric == "A"].verdict.value_counts(); tB = pc[pc.metric == "B"].verdict.value_counts()
    fa = "/".join(str(int(tA.get(k, 0))) for k in ("superior", "tie", "inferior")); fb = "/".join(str(int(tB.get(k, 0))) for k in ("superior", "tie", "inferior"))
    lines += [BS + "midrule", f"Total & & {fa} & & {fb} " + BS + BS, BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "boot.tex"), "w", encoding="utf-8").write("\n".join(lines))
    macro("WTLsup", int(tot.superior)); macro("WTLtie", int(tot.tie)); macro("WTLinf", int(tot.inferior)); macro("NComparisons", int(tot.sum())); macro("NBench", len(bench))
    inf = pc[pc.verdict == "inferior"][["benchmark", "case", "metric", "diff"]]
    json.dump(inf.to_dict("records"), open(os.path.join(RES, "inferior_cases.json"), "w"), indent=1)
    print("WTL", tot.to_dict()); print(inf.to_string())

# ============================================================================ shifted pools
POOLS = [("test", "Nominal"), ("rob_n3", "Noise $3\\times$"), ("rob_n4", "Noise $4\\times$"), ("rob_off", "Off-prior attacker"), ("rob_adapt", "Adaptive replay"),
         ("rob_pmu3", "PMU noise $\\times3$"), ("rob_pmu10", "PMU noise $\\times10$"), ("rob_pmu30", "PMU noise $\\times30$"), ("rob_pmu100", "PMU noise $\\times100$"),
         ("rob_topo", "N-1, no flag"), ("rob_topof", "N-1, flag")]
SH = [PROP, "FT-Transformer", "GAT", "C5-WR-Boost", "LGBM", "XGB", "CUSUM", "SOURCE"]
SH = [n for n in SH if n in set(L.name)]
tab = L.groupby(["name", "pool", "case"]).agg(A=("A", "mean"), B=("B", "mean")).reset_index().groupby(["name", "pool"]).agg(A=("A", "mean"), B=("B", "mean"))
lines = [BS + "begin{tabular}{l" + "cc" * len(SH) + "}", BS + "toprule", "Condition & " + " & ".join(f"{BS}multicolumn{{2}}{{c}}{{{DISP[n]}}}" for n in SH) + " " + BS + BS,
         "".join(f"{BS}cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(SH))), " & " + " & ".join("A & B" for _ in SH) + " " + BS + BS, BS + "midrule"]
for tag, lab in POOLS:
    r = [lab]
    for n in SH:
        if (n, tag) in tab.index:
            a, b = tab.loc[(n, tag), "A"], tab.loc[(n, tag), "B"]
            r += ["--" if np.isnan(a) else f3(a), f3(b)]
        else:
            r += ["--", "--"]
    lines.append(" & ".join(r) + " " + BS + BS)
lines += [BS + "bottomrule", BS + "end{tabular}"]
open(os.path.join(TAB, "shift.tex"), "w", encoding="utf-8").write("\n".join(lines))
for tag, key in [("rob_n4", "NFour"), ("rob_off", "Off"), ("rob_adapt", "Adapt"), ("rob_pmu10", "PmuTen"), ("rob_pmu30", "PmuThirty"), ("rob_pmu100", "PmuHundred"), ("rob_topo", "Topo"), ("rob_topof", "TopoF")]:
    for n, nk in [(PROP, "Prop"), ("FT-Transformer", "Ft"), ("C5-WR-Boost", "Boost"), ("LGBM", "Lgbm"), ("CUSUM", "Cusum"), ("SOURCE", "Src")]:
        if (n, tag) in tab.index:
            macro(f"Sh{key}{nk}B", f3(tab.loc[(n, tag), "B"]))
            if not np.isnan(tab.loc[(n, tag), "A"]): macro(f"Sh{key}{nk}A", f3(tab.loc[(n, tag), "A"]))
macro("ShNomPropA", f3(tab.loc[(PROP, "test"), "A"])); macro("ShNomPropB", f3(tab.loc[(PROP, "test"), "B"]))

# PMU sweep figure
sc = [(1, "test"), (3, "rob_pmu3"), (10, "rob_pmu10"), (30, "rob_pmu30"), (100, "rob_pmu100")]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
cols = {PROP: "#1F4E79", "FT-Transformer": "#FC8D62", "C5-WR-Boost": "#66C2A5", "LGBM": "#8DA0CB", "CUSUM": "#999999", "SOURCE": "#E78AC3"}
for ax, key, ttl in zip(axes, ["A", "B"], ["Typing macro-F1", "Localisation F1"]):
    for n in [PROP, "FT-Transformer", "C5-WR-Boost", "LGBM", "CUSUM", "SOURCE"]:
        ys = [tab.loc[(n, t), key] if (n, t) in tab.index else np.nan for _, t in sc]
        if np.all(np.isnan(ys)): continue
        ax.plot([s for s, _ in sc], ys, marker="o", ms=3, lw=1.1, color=cols[n], label=DISP[n].replace("$^\\dagger$", ""))
    ax.set_xscale("log"); ax.set_xlabel("PMU noise multiplier"); ax.set_title(ttl, fontsize=8); ax.grid(lw=0.3, alpha=0.5)
axes[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_pmu.pdf")); fig.savefig(os.path.join(FIG, "fig_pmu.png"), dpi=300); plt.close(fig)

# ============================================================================ operational metrics
al_rows = []
for n in [PROP, "FT-Transformer", "GAT", "C5-WR-Boost", "LGBM", "XGB", "CatBoost", "CUSUM", "SOURCE"]:
    for c in CASES:
        d = stats2.alarm_metrics(n, c, [0, 1, 2, 3, 4] if n not in ("CUSUM", "SOURCE") else [0])
        if len(d):
            al_rows.append(dict(name=n, case=c, det=d.det_rate.mean(), delay=d.delay.mean(), fa=d.fa_per_hour.mean()))
if al_rows:
    al = pd.DataFrame(al_rows).groupby("name")[["det", "delay", "fa"]].mean()
    al.to_csv(os.path.join(RES, "alarm_summary.csv"))
    lines = [BS + "begin{tabular}{lccc}", BS + "toprule", "Method & Detection rate & Delay (scans) & False alarms/h " + BS + BS, BS + "midrule"]
    for n in [PROP, "FT-Transformer", "GAT", "C5-WR-Boost", "LGBM", "XGB", "CatBoost", "CUSUM", "SOURCE"]:
        if n in al.index:
            lines.append(f"{DISP[n]} & {al.loc[n,'det']:.3f} & {al.loc[n,'delay']:.2f} & {al.loc[n,'fa']:.1f} " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "alarm.tex"), "w", encoding="utf-8").write("\n".join(lines))
    for n, nk in [(PROP, "Prop"), ("C5-WR-Boost", "Boost"), ("FT-Transformer", "Ft"), ("SOURCE", "Src"), ("CUSUM", "Cusum")]:
        if n in al.index:
            macro(f"Det{nk}", f3(al.loc[n, "det"])); macro(f"Delay{nk}", f"{al.loc[n, 'delay']:.2f}"); macro(f"Fa{nk}", f"{al.loc[n, 'fa']:.1f}")

# ============================================================================ replicates
rp = rows("final2_rep.jsonl")
if len(rp):
    rp = rp.drop_duplicates(["name", "case", "seed", "rep"], keep="last")
    m0 = main[main.seed < 3].assign(rep=0)
    allr = pd.concat([m0[["name", "case", "seed", "rep", "test_macro_f1", "test_b_f1_att"]], rp[["name", "case", "seed", "rep", "test_macro_f1", "test_b_f1_att"]]])
    ms = [n for n in [PROP, "FT-Transformer", "GAT", "C5-WR-Boost", "LGBM"] if all(rp[(rp.name == n) & (rp.rep == r_)].case.nunique() == 3 for r_ in (1, 2))]
    lines = [BS + "begin{tabular}{l" + "ccc" * 2 + "}", BS + "toprule", "Method & " + BS + "multicolumn{3}{c}{Typing macro-F1} & " + BS + "multicolumn{3}{c}{Localisation F1} " + BS + BS,
             BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}", " & R0 & R1 & R2 & R0 & R1 & R2 " + BS + BS, BS + "midrule"]
    for n in ms:
        r = [DISP[n]]
        for col in ("test_macro_f1", "test_b_f1_att"):
            for rep in (0, 1, 2):
                v = allr[(allr.name == n) & (allr.rep == rep)].groupby("case")[col].mean().mean()
                r.append(f3(v))
        lines.append(" & ".join(r) + " " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "replicates.tex"), "w", encoding="utf-8").write("\n".join(lines))
    if os.path.exists(os.path.join(RES, "scen")):
        rb = stats2.compare(PROP, [n for n in ms if n != PROP], [0, 1, 2], [0, 1, 2])
        rb.to_csv(os.path.join(RES, "bootstrap_replicates.csv"), index=False)
        for n, nk in [("FT-Transformer", "Ft"), ("C5-WR-Boost", "Boost"), ("GAT", "Gat"), ("LGBM", "Lgbm")]:
            for k in ("A", "B"):
                g = rb[(rb.benchmark == n) & (rb.metric == k) & (rb.case == "avg")]
                if len(g):
                    macro(f"Rep{nk}{k}", f"{g.iloc[0]['diff']:+.3f}"); macro(f"Rep{nk}{k}Lo", f"{g.iloc[0]['ci_lo']:+.3f}"); macro(f"Rep{nk}{k}Hi", f"{g.iloc[0]['ci_hi']:+.3f}")

# ============================================================================ ablation (CIs vs the full model on the same seeds)
ab = rows("ablation2.jsonl")
VAR = [("_noAttn", "w/o attention"), ("_noBias", "w/o graph bias"), ("_graphMix", "with local graph mixing"), ("_tcnEnc", "TCN instead of SSM encoder"), ("_noTemporal", "w/o temporal encoder"),
       ("_noFiLM", "w/o FiLM"), ("_noClone", "w/o clone feature"), ("_noPrior", "w/o vulnerability bit"), ("_noPriorDrop", "w/o prior dropout"),
       ("_hybResid", "hybrid-SE residual"), ("_rawInput", "raw RTU input"), ("_nominalTrain", "nominal-only training")]
if len(ab) and os.path.exists(os.path.join(RES, "scen")):
    ab = ab.drop_duplicates(["name", "case", "seed"], keep="last")
    have = [t for t, _ in VAR if (PROP + t) in set(ab.name)]
    cmp_ = stats2.compare(PROP, [PROP + t for t in have], [0, 1], [0])
    cmp_.to_csv(os.path.join(RES, "bootstrap_ablation.csv"), index=False)
    lines = [BS + "begin{tabular}{lcc}", BS + "toprule", "Variant & " + "$\\Delta$ typing macro-F1 [95" + BS + "% CI] & $\\Delta$ localisation F1 [95" + BS + "% CI] " + BS + BS, BS + "midrule"]
    for t, lab in VAR:
        if t not in have: continue
        g = cmp_[(cmp_.benchmark == PROP + t) & (cmp_.case == "avg")].set_index("metric")
        if len(g) < 2: continue
        # diff in cmp_ is prop - variant: the ablation effect is variant - prop
        def fmt(k):
            r = g.loc[k]
            return f"{-r['diff']:+.3f} [{-r['ci_hi']:+.3f}, {-r['ci_lo']:+.3f}]"
        lines.append(f"{lab} & {fmt('A')} & {fmt('B')} " + BS + BS)
        macro("Abl" + t.strip("_") + "A", f"{-g.loc['A', 'diff']:+.3f}"); macro("Abl" + t.strip("_") + "B", f"{-g.loc['B', 'diff']:+.3f}")
        macro("Abl" + t.strip("_") + "Asig", "1" if (g.loc["A", "ci_lo"] > 0 or g.loc["A", "ci_hi"] < 0) else "0")
        macro("Abl" + t.strip("_") + "Bsig", "1" if (g.loc["B", "ci_lo"] > 0 or g.loc["B", "ci_hi"] < 0) else "0")
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "ablation.tex"), "w", encoding="utf-8").write("\n".join(lines))
    # ablation on shifted pools (mean over systems)
    SHP = ["test", "rob_adapt", "rob_off", "rob_pmu10", "rob_topo"]
    rows_ = []
    for _, r in ab.iterrows():
        rows_.append(dict(name=r["name"], case=r["case"], pool="test", B=r["test_b_f1_att"], A=r["test_macro_f1"]))
        for tag, v in (r.get("extra_test") or {}).items():
            rows_.append(dict(name=r["name"], case=r["case"], pool=tag, B=v["B_F1att"], A=v["A_macroF1"]))
    ar = pd.DataFrame(rows_).groupby(["name", "pool"]).agg(A=("A", "mean"), B=("B", "mean"))
    ar.to_csv(os.path.join(RES, "ablation_shift.csv"))

# ============================================================================ low data
lo = rows("lowdata2.jsonl")
if len(lo):
    lo["frac"] = lo["frac"].astype(float)
    fr = sorted(lo.frac.unique()) + [1.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for n, col, mk in [(PROP, "#1F4E79", "o"), ("FT-Transformer", "#FC8D62", "s"), ("LGBM", "#8DA0CB", "^"), ("C5-WR-Boost", "#66C2A5", "v")]:
        for ax, key, ttl in zip(axes, ["test_macro_f1", "test_b_f1_att"], ["Typing macro-F1", "Localisation F1"]):
            ys = []
            for f_ in fr:
                d = (main[(main.name == n) & (main.seed < 3)] if f_ == 1.0 else lo[(lo.base_model == n) & np.isclose(lo.frac, f_)])
                ys.append(d.groupby("case")[key].mean().mean() if len(d) else np.nan)
            ax.plot([100 * f_ for f_ in fr], ys, marker=mk, ms=3, lw=1.1, color=col, label=DISP[n])
            ax.set_xscale("log"); ax.set_title(ttl, fontsize=8); ax.set_xlabel("Training scenarios (%)"); ax.grid(lw=0.3, alpha=0.5)
            if n == PROP and f_ in (0.01, 0.05):
                pass
    axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_lowdata.pdf")); fig.savefig(os.path.join(FIG, "fig_lowdata.png"), dpi=300); plt.close(fig)
    for f_, k in [(0.01, "One"), (0.05, "Five")]:
        for n, nk in [(PROP, "Prop"), ("FT-Transformer", "Ft"), ("LGBM", "Lgbm"), ("C5-WR-Boost", "Boost")]:
            d = lo[(lo.base_model == n) & np.isclose(lo.frac, f_)]
            if len(d):
                macro(f"Low{k}{nk}A", f3(d.groupby("case").test_macro_f1.mean().mean())); macro(f"Low{k}{nk}B", f3(d.groupby("case").test_b_f1_att.mean().mean()))

# ============================================================================ white-box attacker
adv = rows("adv.jsonl")
if len(adv):
    g = adv.groupby(["name", "norm", "eps"]).agg(A=("A_macroF1", "mean"), B=("B_F1att", "mean")).reset_index()
    lines = [BS + "begin{tabular}{lcccccc}", BS + "toprule", "Model & " + BS + "multicolumn{3}{c}{" + "$L_\\infty$ budget" + "} & " + BS + "multicolumn{3}{c}{" + "$L_2$ budget" + "} " + BS + BS,
             BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}", " & 0.5 & 1 & 2 & 0.5 & 1 & 2 " + BS + BS, BS + "midrule"]
    for n in [PROP, "FT-Transformer", "GAT", "TCN"]:
        if n not in set(g.name): continue
        r = [DISP[n] + " (B)"]
        for norm in ("linf", "l2"):
            for e in (0.5, 1.0, 2.0):
                x = g[(g.name == n) & (g.norm == norm) & np.isclose(g.eps, e)]
                r.append(f3(x.B.iloc[0]) if len(x) else "--")
        lines.append(" & ".join(r) + " " + BS + BS)
        c0 = g[(g.name == n) & np.isclose(g.eps, 0)]
        if len(c0): macro("Adv0" + n.replace("-", "").replace("V2c", "Prop") + "B", f3(c0.B.iloc[0]))
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "adv.tex"), "w", encoding="utf-8").write("\n".join(lines))
    for n, nk in [(PROP, "Prop"), ("FT-Transformer", "Ft")]:
        for norm, kn in [("linf", "Linf"), ("l2", "Ltwo")]:
            x = g[(g.name == n) & (g.norm == norm) & np.isclose(g.eps, 1.0)]
            if len(x): macro(f"Adv{kn}{nk}B", f3(x.B.iloc[0])); macro(f"Adv{kn}{nk}A", f3(x.A.iloc[0]))

# ============================================================================ per-class analysis
CL = ["Normal", "MDDA", "LT-RDCA", "ST-RDCA"]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
for ax, c in zip(axes, CASES):
    d = main[(main.name == PROP) & (main.case == c)]
    cm = np.sum([np.array(x) for x in d["test_cm"]], axis=0).astype(float); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{100*cmn[i,j]:.1f}", ha="center", va="center", fontsize=6.5, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(CL, rotation=35, fontsize=6); ax.set_yticklabels(CL, fontsize=6); ax.set_title(CN[c], fontsize=8); ax.set_xlabel("Predicted", fontsize=7)
axes[0].set_ylabel("True (row-normalised, %)", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_confusion.pdf")); fig.savefig(os.path.join(FIG, "fig_confusion.png"), dpi=300); plt.close(fig)
lines = [BS + "begin{tabular}{llcccc}", BS + "toprule", "System & Method & Normal & MDDA & LT-RDCA & ST-RDCA " + BS + BS, BS + "midrule"]
cmp_models = [PROP, "FT-Transformer", "GAT"]
for c in CASES:
    for k, n in enumerate(cmp_models):
        d = main[(main.name == n) & (main.case == c)]
        v = [d[f"test_F1_{x}"].mean() for x in CL]
        lines.append((f"{BS}multirow{{{len(cmp_models)}}}{{*}}{{{CN[c]}}}" if k == 0 else "") + f" & {DISP[n]} & " + " & ".join(f3(x) for x in v) + " " + BS + BS)
    lines.append(BS + "midrule")
lines[-1] = BS + "bottomrule"; lines.append(BS + "end{tabular}")
open(os.path.join(TAB, "perclass.tex"), "w", encoding="utf-8").write(chr(10)
.join(lines))
for c, tg in zip(CASES, TAGN):
    d = main[(main.name == PROP) & (main.case == c)]
    macro(f"PropSTF{tg}", f3(d["test_F1_ST-RDCA"].mean())); macro(f"PropSTrec{tg}", f3(d["test_minority_recall"].mean()))

# ============================================================================ Fig. 3: main comparison bars
REP_M = ["SOURCE", "CUSUM", "LGBM", "C5-WR-Boost", "MLP", "TCN", "FT-Transformer", "GAT", PROP]
CMAP = {"SOURCE": "#E78AC3", "CUSUM": "#999999", "LGBM": "#8DA0CB", "C5-WR-Boost": "#66C2A5", "MLP": "#A6D854", "TCN": "#FFD92F", "FT-Transformer": "#FC8D62", "GAT": "#B3B3B3", PROP: "#1F4E79"}
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
for ax, key, ttl in zip(axes, ["A", "B"], ["Typing macro-F1", "Localisation F1"]):
    ms_ = [n for n in REP_M if ok(n) and not (key == "A" and n in ORDER_RULE)]
    w = 0.8 / len(ms_)
    for j, n in enumerate(ms_):
        ys = [Aidx.loc[(n, c), key] for c in CASES]; es = [Aidx.loc[(n, c), key + "s"] if not np.isnan(Aidx.loc[(n, c), key + "s"]) else 0 for c in CASES]
        ax.bar(np.arange(3) + (j - len(ms_) / 2 + 0.5) * w, ys, w, yerr=es, color=CMAP[n], label=DISP[n].replace("$^\\dagger$", ""), error_kw=dict(lw=0.5, capsize=1))
    ax.set_xticks(range(3)); ax.set_xticklabels([CN[c] for c in CASES]); ax.set_title(ttl, fontsize=8); ax.set_ylim(0.2, 1.0); ax.grid(axis="y", lw=0.3, alpha=0.5)
axes[0].legend(fontsize=5.5, ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_main.pdf")); fig.savefig(os.path.join(FIG, "fig3_main.png"), dpi=300); plt.close(fig)

# ============================================================================ training distribution (robust vs nominal) on nominal and shifted pools
if len(ab):
    arr = pd.read_csv(os.path.join(RES, "ablation_shift.csv")).set_index(["name", "pool"])
    SP = [("test", "Nominal"), ("rob_adapt", "Adaptive replay"), ("rob_off", "Off-prior attacker"), ("rob_pmu10", "PMU noise $\\times10$"), ("rob_topo", "N-1, no flag")]
    # robust-trained reference on the same seeds as the ablation (seeds 0-1) from the main run
    mm = main[(main.name == PROP) & (main.seed < 2)]
    ref = {}
    for _, r in mm.iterrows():
        ref.setdefault("test", []).append((r["case"], r["test_macro_f1"], r["test_b_f1_att"]))
        for tag, v in (r.get("extra_test") or {}).items():
            ref.setdefault(tag, []).append((r["case"], v["A_macroF1"], v["B_F1att"]))
    lines = [BS + "begin{tabular}{lcccc}", BS + "toprule", "Condition & " + BS + "multicolumn{2}{c}{Robust training} & " + BS + "multicolumn{2}{c}{Nominal-only training} " + BS + BS,
             BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){4-5}", " & A & B & A & B " + BS + BS, BS + "midrule"]
    for tag, lab in SP:
        if tag not in ref or (PROP + "_nominalTrain", tag) not in arr.index: continue
        ra = np.mean([x[1] for x in ref[tag]]); rb = np.mean([x[2] for x in ref[tag]])
        na, nb = arr.loc[(PROP + "_nominalTrain", tag), "A"], arr.loc[(PROP + "_nominalTrain", tag), "B"]
        lines.append(f"{lab} & {f3(ra)} & {f3(rb)} & {f3(na)} & {f3(nb)} " + BS + BS)
        macro("TrN" + {"test": "Nom", "rob_adapt": "Adapt", "rob_off": "Off", "rob_pmu10": "Pmu", "rob_topo": "Topo"}[tag] + "RobB", f3(rb))
        macro("TrN" + {"test": "Nom", "rob_adapt": "Adapt", "rob_off": "Off", "rob_pmu10": "Pmu", "rob_topo": "Topo"}[tag] + "NomB", f3(nb))
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "ablshift.tex"), "w", encoding="utf-8").write("\n".join(lines))

# ============================================================================ leave-one-system-out transfer
cs = rows("crosssys.jsonl")
if len(cs):
    cs = cs.drop_duplicates(["model", "target", "seed", "mode", "frac"], keep="last")
    cs["frac"] = cs["frac"].astype(float)
    def cell(sel, key):
        g = sel.groupby("target")[key].agg(["mean", "std"])
        return g
    lines = [BS + "begin{tabular}{lcccccc}", BS + "toprule", "Setting & " + BS + "multicolumn{3}{c}{Typing macro-F1} & " + BS + "multicolumn{3}{c}{Localisation F1} " + BS + BS,
             BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}", " & " + " & ".join(CN[c] for c in CASES) + " & " + " & ".join(CN[c] for c in CASES) + " " + BS + BS, BS + "midrule"]
    def add(label, sel):
        if not len(sel): return
        r = [label]
        for key in ("A_macroF1", "B_F1att"):
            g = cell(sel, key)
            for c in CASES:
                r.append(f"{g.loc[c,'mean']:.3f}{BS}scriptsize$\\pm${g.loc[c,'std']:.3f}" if c in g.index and not np.isnan(g.loc[c, 'std']) else (f3(g.loc[c, 'mean']) if c in g.index else "--"))
        lines.append(" & ".join(r) + " " + BS + BS)
    add("Proposed, zero-shot", cs[(cs.model == PROP) & (cs["mode"] == "zero-shot")])
    for f_ in (0.05, 0.1, 0.25):
        add(f"Proposed, fine-tuned on {int(100*f_)}" + BS + "% of target", cs[(cs.model == PROP) & (cs["mode"] == "few-shot") & np.isclose(cs.frac, f_)])
    add("WR-Boost features, zero-shot", cs[(cs.model == "WR-Boost") & (cs["mode"] == "zero-shot")])
    r = ["Proposed, trained on target"]
    for key in ("test_macro_f1", "test_b_f1_att"):
        for c in CASES:
            d = main[(main.name == PROP) & (main.case == c) & (main.seed < 3)][key]
            r.append(f"{d.mean():.3f}{BS}scriptsize$\\pm${d.std():.3f}")
    lines += [BS + "midrule", " & ".join(r) + " " + BS + BS, BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "crosssys.tex"), "w", encoding="utf-8").write("\n".join(lines))
    for c, tg in zip(CASES, TAGN):
        for lab, sel in [("Zero", cs[(cs.model == PROP) & (cs["mode"] == "zero-shot")]), ("ZeroBoost", cs[(cs.model == "WR-Boost") & (cs["mode"] == "zero-shot")]),
                         ("FewFive", cs[(cs.model == PROP) & (cs["mode"] == "few-shot") & np.isclose(cs.frac, 0.05)]), ("FewTwentyFive", cs[(cs.model == PROP) & (cs["mode"] == "few-shot") & np.isclose(cs.frac, 0.25)])]:
            s = sel[sel.target == c]
            if len(s): macro(f"{lab}A{tg}", f3(s.A_macroF1.mean())); macro(f"{lab}B{tg}", f3(s.B_F1att.mean()))

# ============================================================================ computational cost
ph = pd.read_csv(os.path.join(RES, "efficiency_physics.csv")).set_index("case")
em = rows("efficiency_models.jsonl"); et = rows("efficiency_tab.jsonl")
if len(ph):
    for c, tg in zip(CASES, TAGN):
        macro(f"TSrc{tg}", f"{ph.loc[c, 't_source_total_ms']:.2f}"); macro(f"TPmu{tg}", f"{ph.loc[c, 't_pmu_lse_recon_ms']:.3f}")
        macro(f"TRatio{tg}", f"{ph.loc[c, 't_source_total_ms'] / ph.loc[c, 't_pmu_lse_recon_ms']:.0f}")
if len(em) and len(ph):
    em = em.drop_duplicates(["name", "case"], keep="last").set_index(["name", "case"])
    for c, tg in zip(CASES, TAGN):
        if (PROP, c) in em.index:
            macro(f"PropGpuLat{tg}", f"{em.loc[(PROP, c), 'lat1_cuda_ms']:.1f}"); macro(f"PropCpuLat{tg}", f"{em.loc[(PROP, c), 'lat1_cpu_ms']:.1f}")
            macro(f"TPipe{tg}", f"{ph.loc[c, 't_pmu_lse_recon_ms'] + em.loc[(PROP, c), 'lat1_cuda_ms']:.1f}")
    macro("PropParamsFourteen", f"{em.loc[(PROP, 'case14'), 'params'] / 1000:.0f}")
    tt = main.groupby("name").train_time.mean()
    EM = [PROP, "TCN", "GRU", "Transformer", "FT-Transformer", "GCN", "GAT", "Mamba", "MLP"]
    lines = [BS + "begin{tabular}{lccccccc}", BS + "toprule", "Method & Params (k) & " + BS + "multicolumn{3}{c}{GPU latency (ms)} & CPU latency (ms) & Train (s) & Memory (MB) " + BS + BS,
             BS + "cmidrule(lr){3-5}", " & & " + " & ".join(CN[c] for c in CASES) + " & IEEE 118 & & " + BS + BS, BS + "midrule"]
    for n in EM:
        if not all((n, c) in em.index for c in CASES): continue
        r = [DISP[n], f"{em.loc[(n, 'case14'), 'params'] / 1000:.0f}" + ("" if em.loc[(n, 'case14'), 'params'] == em.loc[(n, 'case118'), 'params'] else f"--{em.loc[(n, 'case118'), 'params'] / 1000:.0f}")]
        r += [f"{em.loc[(n, c), 'lat1_cuda_ms']:.1f}" for c in CASES]
        r += [f"{em.loc[(n, 'case118'), 'lat1_cpu_ms']:.1f}", f"{tt.get(n, np.nan):.0f}", f"{em.loc[(n, 'case118'), 'peak_mem_MB']:.0f}"]
        lines.append(" & ".join(r) + " " + BS + BS)
    if len(et):
        et = et.drop_duplicates(["name", "case"], keep="last").set_index(["name", "case"])
        lines.append(BS + "midrule")
        for n in ["LGBM", "XGB", "CatBoost", "C5-WR-Boost"]:
            if all((n, c) in et.index for c in CASES):
                lines.append(f"{DISP[n]} & -- & -- & -- & -- & {et.loc[(n, 'case118'), 'lat_scan_cpu_ms']:.2f} & -- & -- " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(TAB, "efficiency.tex"), "w", encoding="utf-8").write("\n".join(lines))

# ============================================================================ numbers
with open(os.path.join(ROOT, "paper", "numbers.tex"), "w", encoding="utf-8") as f:
    for k, v in NUM.items():
        f.write(BS + "newcommand{" + BS + "num" + re.sub(r"[^A-Za-z]", "", k.translate(str.maketrans("0123456789", "ABCDEFGHIJ"))) + "}{" + str(v) + "}\n")
json.dump({k: str(v) for k, v in NUM.items()}, open(os.path.join(RES, "numbers.json"), "w"), indent=1)
print("assets written", len(NUM))
