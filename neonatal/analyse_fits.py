"""Figures and summary tables for the fits in fit_results.json (fit_mizuno.py)."""

import json
import numpy as np
import matplotlib.pyplot as plt
import fit_mizuno as fm

COL = {"M0": "#55554f", "MB": "#eb6834", "MC": "#1baf7a", "MCint": "#2a78d6",
       "MBimax": "#c98500"}
LS = {"M0": ":", "MB": "-", "MC": "--", "MCint": "-.", "MBimax": (0, (1, 1))}
INK, MUTED, GRID = "#1f1f1c", "#55554f", "#e3e3df"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, width=0.6)


def letter(ax, s, x=-0.12, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", color=INK)


res = json.load(open("fit_results.json"))
R = {(r["pid"], r["model"]): r for r in res}

# ------------------------------------------------------------- summary table
print(f"{'pid':<4}{'model':<7}{'k':>3}{'RMSE':>7}{'AICc':>8}{'dAICc':>7}{'BIC':>8}"
      "  parameters (estimate [approx. 95% CI factor])")
for pid in fm.PATIENTS:
    best = min(R[(pid, m)]["aicc"] for m in fm.MODELS)
    for m in fm.MODELS:
        r = R[(pid, m)]
        ps = "; ".join(f"{n}={10 ** v:.3g} (x{10 ** (1.96 * s):.1f})"
                       for n, v, s in zip(fm.MODELS[m]["names"], r["logx"], r["se_log10"]))
        print(f"{pid:<4}{m:<7}{r['k']:>3}{r['rmse']:7.2f}{r['aicc']:8.1f}"
              f"{r['aicc'] - best:7.1f}{r['bic']:8.1f}  {ps}")

# ------------------------------------------- cumulative withdrawal analysis
print("\nCumulative Finnegan score (score x day, birth to day 60): treated vs "
      "untreated counterfactual")
cum = {}
for pid in fm.PATIENTS:
    for m in fm.MODELS:
        x = R[(pid, m)]["logx"]
        a_t, t, y_t = fm.signed_area(m, x, pid)
        a_u, _, y_u = fm.signed_area(m, x, pid, zero_drug=True)
        cum[(pid, m)] = (a_t, a_u, t, y_t, y_u)
        print(f"  {pid} {m:<6} treated {a_t:7.1f}  untreated {a_u:7.1f}  "
              f"ratio {a_t / a_u:5.2f}")

# ------------------------------------------------------------------ figure
FIG_W = 174 / 25.4
fig, axs = plt.subplots(3, 2, figsize=(FIG_W, 6.6),
                        gridspec_kw={"height_ratios": [0.7, 1.4, 1.1]})
for j, pid in enumerate(fm.PATIENTS):
    tf, F, tc, cc = fm.load(pid)
    t, c = fm.conc_grid(tc, cc, fm.T_END[pid])
    ax = axs[0, j]
    ax.plot(t / 24, c, color=INK, lw=0.8)
    ax.set_ylabel("Buprenorphine (ng/mL)")
    ax.set_xlim(0, fm.T_END[pid] / 24)
    ax.text(0.98, 0.92, f"Infant {pid}", transform=ax.transAxes, ha="right",
            va="top", fontweight="bold")
    ax = axs[1, j]
    ax.plot(tf / 24, F, "o", ms=3.2, mfc="white", mec=INK, mew=0.8,
            label="Observed (digitized)", zorder=5)
    for m in fm.MAIN:
        y = fm.simulate(m, np.array(R[(pid, m)]["logx"]), t, c)[:, 0]
        ax.plot(t / 24, y, color=COL[m], ls=LS[m],
                label=f"{fm.MODELS[m]['label']} (RMSE {R[(pid, m)]['rmse']:.2f})")
    ax.axhline(8, color="#b5b5ae", lw=0.6, ls="--")
    ax.set_xlim(0, fm.T_END[pid] / 24)
    ax.set_ylim(0, 22)
    ax.set_ylabel("Finnegan score")
    ax.legend(frameon=False, fontsize=6.3, loc="upper right")
    ax = axs[2, j]
    for m in ["M0", "MB"]:
        a_t, a_u, tt, y_t, y_u = cum[(pid, m)]
        ax.plot(tt / 24, y_t, color=COL[m], ls=LS[m],
                label=f"{'Published' if m == 'M0' else 'B'} treated "
                      f"(cum. {a_t:.0f})")
        ax.plot(tt / 24, y_u, color=COL[m], ls=LS[m], lw=0.8, alpha=0.55,
                label=f"{'Published' if m == 'M0' else 'B'} untreated "
                      f"(cum. {a_u:.0f})")
    ax.set_xlim(0, 45)
    ax.set_xlabel("Time after birth (days)")
    ax.set_ylabel("Predicted score")
    ax.legend(frameon=False, fontsize=6.3, loc="upper right")
for ax, l in zip(axs.flat, "abcdef"):
    style(ax)
    letter(ax, l)
fig.tight_layout()
fig.savefig("Fig7.png", dpi=300)
fig.savefig("Fig7.pdf")
fig.savefig("Fig7.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("\nsaved Fig7")
