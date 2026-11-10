"""Figure 1 (architecture) and Figure 2 (source pipeline vs proposed pipeline), drawn with matplotlib."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "paper", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "mathtext.fontset": "stix"})
C = dict(in_="#DAE8FC", phys="#D5E8D4", enc="#FFE6CC", sp="#E1D5E7", out="#F8CECC", grp="#F5F5F5", edge="#555555")


def box(ax, x, y, w, h, text, fc, fs=7.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=C["edge"], lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arrow(ax, p, q, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#333333", ls=ls, connectionstyle=f"arc3,rad={rad}"))


def group(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.1", fc=C["grp"], ec="#888888", lw=0.8, ls="--", zorder=0))
    ax.text(x + 0.12, y + h - 0.13, title, ha="left", va="top", fontsize=7.5, fontweight="bold")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================================ Fig. 1
fig, ax = plt.subplots(figsize=(7.16, 3.3)); ax.set_xlim(0, 15.5); ax.set_ylim(0, 7.3); ax.axis("off")
group(ax, 0.1, 3.3, 5.2, 3.2, "Physics front-end (non-iterative)")
box(ax, 0.3, 5.0, 1.9, 0.9, "PMU phasors\n$\\zeta(t)$", C["in_"])
box(ax, 0.3, 3.55, 1.9, 0.9, "RTU meters\n$\\mathbf{z}(t)$ (SCADA)", C["in_"])
box(ax, 2.6, 5.0, 2.5, 0.9, "PMU-LSE $\\hat{\\mathbf{x}}=\\mathbf{K}\\zeta$\n$\\hat{\\mathbf{z}}=h(\\hat{\\mathbf{x}})$", C["phys"])
box(ax, 2.6, 3.55, 2.5, 1.2, "residual $\\tilde r_i$, scale $\\hat s_t$\nclone $\\ell_i$, topology flag", C["phys"], fs=6.8)
arrow(ax, (2.2, 5.45), (2.6, 5.45)); arrow(ax, (3.85, 5.0), (3.85, 4.75)); arrow(ax, (2.2, 4.0), (2.6, 4.0))
group(ax, 5.55, 0.2, 6.1, 6.9, "Meter-shared network (size independent of $m$)")
box(ax, 5.8, 6.0, 5.6, 0.6, "PTDF bit $v_i$ (dropout) and meter kind $k_i$", C["sp"], fs=7, ls="--")
box(ax, 5.8, 4.5, 2.7, 1.2, "Meter-wise SSM encoder\n(S4D, $L$ layers)\nFiLM on $\\log\\hat s_t$", C["enc"])
box(ax, 8.8, 4.5, 2.6, 1.2, "meter states\n$\\mathbf{h}_i=[\\mathbf{h}_i(t);\\max_\\tau\\mathbf{h}_i(\\tau)]$", C["enc"])
arrow(ax, (8.5, 5.1), (8.8, 5.1)); arrow(ax, (7.15, 6.0), (7.15, 5.7), ls="--")
box(ax, 5.8, 2.95, 5.6, 1.15, "Pre-norm residual connections\nand feed-forward layers ($d=32$)", C["sp"])
box(ax, 5.8, 1.4, 5.6, 1.15, "Physics-biased multi-head attention\n$\\alpha_{ij}=\\mathrm{softmax}_j(\\mathbf{q}_i^\\top\\mathbf{k}_j/\\sqrt{d_h}+\\beta A_{ij})$", C["sp"])
box(ax, 5.8, 0.4, 5.6, 0.7, "global context $\\mathbf{c}=[\\mathrm{mean}_i\\mathbf{h}_i;\\max_i\\mathbf{h}_i;\\hat s_t,\\ldots]$", C["enc"], fs=7)
arrow(ax, (10.1, 4.5), (10.1, 4.1)); arrow(ax, (8.6, 2.95), (8.6, 2.55)); arrow(ax, (8.6, 1.4), (8.6, 1.1))
arrow(ax, (5.1, 4.15), (5.8, 5.05), rad=-0.1)
box(ax, 12.1, 3.7, 3.2, 1.6, "Meter head\n$p_i=\\sigma(o_i)\\geq\\tau$\nlocalises replayed meters", C["out"])
box(ax, 12.1, 0.9, 3.2, 1.6, "Class head\nNormal | MDDA |\nLT-RDCA | ST-RDCA", C["out"])
arrow(ax, (11.4, 2.0), (12.1, 4.2)); arrow(ax, (11.4, 0.75), (12.1, 1.4)); arrow(ax, (11.4, 2.0), (12.1, 1.9))
save(fig, "fig1_architecture")

# ============================================================================================ Fig. 2
fig, ax = plt.subplots(figsize=(7.16, 2.7)); ax.set_xlim(0, 15.5); ax.set_ylim(0, 5.6); ax.axis("off")
group(ax, 0.1, 2.95, 15.3, 2.55, "Fixed-threshold hybrid-SE pipeline (per snapshot)")
top = [("RTU WLS\n(Gauss-Newton,\niterative)", C["phys"]), ("HYB-SE\n(dense $2N\\times 2N$\ncovariance inverse)", C["phys"]), ("reconstruct $\\hat{\\mathbf{z}}$", C["phys"]),
       ("global threshold\n$|\\Lambda_i|>\\delta$", C["out"]), ("replace flagged,\nre-run WLS", C["phys"])]
for i, (t, fc) in enumerate(top):
    box(ax, 0.4 + i * 3.05, 3.15, 2.5, 1.5, t, fc)
    if i:
        arrow(ax, (0.4 + i * 3.05 - 0.55, 3.9), (0.4 + i * 3.05, 3.9))
group(ax, 0.1, 0.1, 15.3, 2.65, "Proposed pipeline (per snapshot)")
bot = [("PMU-LSE\n(one matrix product)", C["phys"]), ("whitened residual,\nscale $\\hat s_t$, clone $\\ell_i$", C["phys"]), ("meter-shared SSM\nencoder (12 s window)", C["enc"]),
       ("physics-biased\nmulti-head attention", C["sp"]), ("localisation $p_i$ and\nattack type", C["out"])]
for i, (t, fc) in enumerate(bot):
    box(ax, 0.4 + i * 3.05, 0.3, 2.5, 1.5, t, fc)
    if i:
        arrow(ax, (0.4 + i * 3.05 - 0.55, 1.05), (0.4 + i * 3.05, 1.05))
save(fig, "fig2_workflow")
print("figures written")
