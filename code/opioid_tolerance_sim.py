"""
Simulating opioid tolerance and withdrawal with four PK/PD model structures.

Companion code for the manuscript "Tolerance is not dependence: a mathematical
analysis of pharmacodynamic models of opioid tolerance and withdrawal"
(submitted to J Pharmacokinet Pharmacodyn). Equation numbers refer to the manuscript.

Model A  Indirect response (Dayneka et al. 1993, model I)   no adaptation
Model B  Precursor-dependent indirect response (Sharma et al. 1998)
Model C  Nonlinear feedback / moderator (counter-adaptation)
Model D  Receptor down-regulation (occupancy-driven internalisation)

All models share the same PK and the same occupancy function, so differences
in tolerance and withdrawal arise from the adaptation structure alone.
R is a drug-inhibited drive, normalised so R/R0 = 1 at baseline:
    E = 1 - R/R0 (opioid effect, R < R0);  W = R/R0 - 1 (withdrawal, R > R0)

Time unit: hours. Concentrations in units of IC50. Parameters illustrative.
Run:  python opioid_tolerance_sim.py   (writes Fig1-Fig6 as .pdf, .tif and .png and prints the
      numbers behind the tables; about 2 min). Fig. 7 is made by neonatal/analyse_fits.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "mathtext.fontset": "custom",
    "mathtext.it": "Arial:italic", "mathtext.rm": "Arial",
})

# ---------------------------------------------------------------- parameters
TAU = 24.0                       # dosing interval (h)
KA = 1.0                         # absorption rate constant (1/h)
KE = np.log(2) / 24.0            # elimination rate constant (t1/2 = 24 h)
CSS = 3.0                        # average steady-state concentration / IC50
DOSE = CSS * KE * TAU            # oral dose (V = 1)

PAR = dict(
    imax=0.9,                    # maximal fractional inhibition
    kout=0.5,                    # response turnover (1/h), all models
    kp=0.03,                     # Model B precursor release (1/h)
    ktol=0.01,                   # Model C moderator turnover (1/h)
    kdeg=0.01,                   # Model D basal receptor turnover (1/h)
    kint=0.02,                   # Model D occupancy-driven internalisation (1/h)
)
NALOXONE_T12 = 1.1               # h
NALOXONE_A0 = 30.0               # peak antagonist concentration / K_A
N_DAYS = 21
TAPER_DAYS = 14
T_END = 50 * 24.0

MODELS = ["A", "B", "C", "D"]
LABELS = {"A": "A  Indirect response", "B": "B  Precursor pool",
          "C": "C  Feedback moderator", "D": "D  Receptor down-regulation"}
# Validated categorical palette (slots 1-4) + distinct line patterns so the
# figures remain readable in greyscale print.
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#c98500"}
STYLES = {"A": "-.", "B": "-", "C": "--", "D": ":"}
INK, MUTED, GRID = "#1f1f1c", "#55554f", "#e3e3df"
FIG_W = 174 / 25.4               # JPKPD full-width figure (in)


# ------------------------------------------------------------------------ PK
def oral_schedule(scenario, n_days=N_DAYS, t_end=T_END):
    times = list(np.arange(n_days) * TAU)
    amounts = [DOSE] * n_days
    if scenario == "taper":
        for k in range(1, TAPER_DAYS):
            times.append((n_days + k - 1) * TAU)
            amounts.append(DOSE * (TAPER_DAYS - k) / TAPER_DAYS)
    if scenario == "naloxone":          # maintenance continues throughout
        times = list(np.arange(int(t_end / TAU)) * TAU)
        amounts = [DOSE] * len(times)
    return np.array(times), np.array(amounts)


def oral_conc(times, amounts):
    """Eq. 1: one compartment, first-order absorption, superposition."""
    def c(t):
        dt = t - times
        on = dt >= 0
        dt = np.where(on, dt, 0.0)
        return float(np.sum(on * amounts * KA / (KA - KE)
                            * (np.exp(-KE * dt) - np.exp(-KA * dt))))
    return c


def infusion_conc(ke, t_stop, css=CSS):
    """Constant-rate infusion to css, stopped at t_stop."""
    def c(t):
        if t <= t_stop:
            return css * (1 - np.exp(-ke * t))
        return css * (1 - np.exp(-ke * t_stop)) * np.exp(-ke * (t - t_stop))
    return c


def naloxone(t_nal):
    def a(t):
        if t_nal is None or t < t_nal:
            return 0.0
        return NALOXONE_A0 * np.exp(-np.log(2) / NALOXONE_T12 * (t - t_nal))
    return a


def occupancy_fn(cfun, afun=lambda t: 0.0):
    """Eq. 2: competitive occupancy (IC50 = 1)."""
    def phi(t):
        c, a = cfun(t), afun(t)
        return c / ((1 + a) + c)
    return phi


# ---------------------------------------------------------------------- ODEs
# y = [R_A, R_B, P_B, R_C, M_C, R_D, rho_D]; all baselines = 1.
def rhs(t, y, phi_fn, p):
    RA, RB, PB, RC, MC, RD, rho = y
    phi = phi_fn(t)
    inh = p["imax"] * phi                                     # Eq. 3
    kout, kp, ktol = p["kout"], p["kp"], p["ktol"]
    return [
        kout * (1 - inh) - kout * RA,                         # Eq. 4
        kout * (1 - inh) * PB - kout * RB,                    # Eq. 6
        kp - kp * (1 - inh) * PB,                             # Eq. 5
        kout * MC * (1 - inh) - kout * RC,                    # Eq. 7
        ktol * (1 / RC - MC),                                 # Eq. 8
        kout * (1 - p["imax"] * rho * phi) - kout * RD,       # Eq. 10
        p["kdeg"] * (1 - rho) - p["kint"] * phi * rho,        # Eq. 9
    ]


def integrate(phi_fn, t_end, y0=None, p=PAR, dt=0.1):
    y0 = np.ones(7) if y0 is None else y0
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(phi_fn, p), t_eval=t_eval,
                    method="LSODA", max_step=0.25, rtol=1e-8, atol=1e-10)
    RA, RB, PB, RC, MC, RD, rho = sol.y
    return {"t": sol.t, "R": {"A": RA, "B": RB, "C": RC, "D": RD},
            "state": {"B": PB, "C": MC, "D": rho}}


def simulate_oral(scenario="abrupt", n_days=N_DAYS, t_end=T_END):
    times, amounts = oral_schedule(scenario, n_days, t_end)
    t_nal = n_days * TAU - 4.0 if scenario == "naloxone" else None
    cfun = oral_conc(times, amounts)
    sim = integrate(occupancy_fn(cfun, naloxone(t_nal)), t_end)
    sim["C"] = np.array([cfun(t) for t in sim["t"]])
    sim["t_ref"] = t_nal if t_nal is not None else n_days * TAU
    return sim


# ------------------------------------------------------------------- metrics
def withdrawal_metrics(t, R, t_ref, threshold=0.10, window=None):
    """Peak W, time to peak, duration of the continuous episode containing the
    peak with W > threshold, and AUC of W, from t_ref (optionally windowed)."""
    sel = (t >= t_ref) if window is None else (t >= t_ref) & (t < t_ref + window)
    W, ta = np.clip(R[sel] - 1, 0, None), t[sel]
    i = int(np.argmax(W))
    dur = 0.0
    if W[i] > threshold:
        lo = hi = i
        while lo > 0 and W[lo - 1] > threshold:
            lo -= 1
        while hi < W.size - 1 and W[hi + 1] > threshold:
            hi += 1
        dur = ta[hi] - ta[lo]
    return {"peak": W[i], "t_peak": ta[i] - t_ref, "dur": dur,
            "auc": np.trapezoid(W, ta)}


def peak_effect(t, R, day):
    win = (t >= day * TAU) & (t < (day + 1) * TAU)
    return float(np.max(1 - R[win]))


def areas(t, R):
    """ABEC (area below baseline) and ABRC (area above baseline), in h."""
    dev = R - 1
    return (np.trapezoid(np.clip(-dev, 0, None), t),
            np.trapezoid(np.clip(dev, 0, None), t))


def report(sims):
    rows = {}
    for s, sim in sims.items():
        t, R = sim["t"], sim["R"]
        ref = peak_effect(t, R["A"], N_DAYS - 1)
        print(f"\n=== scenario: {s}")
        print(f"{'model':<30}{'E d1':>7}{'E d21':>7}{'TI':>6}{'Wpk':>7}"
              f"{'tpk h':>8}{'dur h':>8}{'AUC W':>8}{'ABEC':>8}{'ABRC':>8}")
        for m in MODELS:
            e1, e21 = peak_effect(t, R[m], 0), peak_effect(t, R[m], N_DAYS - 1)
            wm = withdrawal_metrics(t, R[m], sim["t_ref"],
                                    window=24.0 if s == "naloxone" else None)
            abec, abrc = areas(t, R[m])
            rows[(s, m)] = dict(e1=e1, e21=e21, ti=1 - e21 / ref, **wm,
                                abec=abec, abrc=abrc)
            print(f"{LABELS[m]:<30}{e1:7.2f}{e21:7.2f}{1 - e21 / ref:6.2f}"
                  f"{wm['peak']:7.2f}{wm['t_peak']:8.1f}{wm['dur']:8.1f}"
                  f"{wm['auc']:8.1f}{abec:8.1f}{abrc:8.1f}")
    return rows


# ------------------------------------------------------ analytic predictions
def b_peak_rebound(I, kp=PAR["kp"], kout=PAR["kout"]):
    """Model B, instantaneous offset from steady state (Eq. 13)."""
    return I / (1 - I) * (kp / kout) ** (kp / (kout - kp))


def b_peak_time(kp=PAR["kp"], kout=PAR["kout"]):
    """Eq. 14."""
    return np.log(kout / kp) / (kout - kp)


def c_peak_rebound_bound(I):
    """Model C, upper bound attained as ktol/kout -> 0 (Eq. 19)."""
    return 1 / np.sqrt(1 - I) - 1


def c_eigenvalues(kout, ktol):
    """Eq. 17: identical at baseline and at every drug steady state."""
    tr, det = -(kout + ktol), 2 * kout * ktol
    disc = tr ** 2 - 4 * det
    return (tr + np.sqrt(disc + 0j)) / 2, (tr - np.sqrt(disc + 0j)) / 2


def steady_state(I, p=PAR):
    s = np.sqrt(1 - I)
    phi = I / p["imax"]
    rho = p["kdeg"] / (p["kdeg"] + p["kint"] * phi)
    return np.array([1 - I, 1.0, 1 / (1 - I), s, 1 / s,
                     1 - p["imax"] * rho * phi, rho])


def step_offset(I, p=PAR, t_end=400.0):
    """Start at the steady state for constant inhibition I, remove drug at t=0."""
    return integrate(lambda t: 0.0, t_end, y0=steady_state(I, p), p=p, dt=0.05)


# ------------------------------------------------------------------- figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, width=0.6)


def letter(ax, s, x=-0.1, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, name):
    fig.savefig(f"{name}.png", dpi=300)
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def fig1_schematic(name="Fig1"):
    fig, axs = plt.subplots(1, 4, figsize=(FIG_W, 2.0))

    def box(ax, xy, txt, w=0.24, h=0.17, fc="#f3f3f0"):
        ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=fc, ec=INK, lw=0.8))
        ax.text(*xy, txt, ha="center", va="center", fontsize=9)

    def arrow(ax, a, b, style="-|>", ls="-", color=INK, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8,
                                     lw=0.9, ls=ls, color=color,
                                     connectionstyle=f"arc3,rad={rad}"))

    def lab(ax, xy, s, **kw):
        ax.text(*xy, s, ha="center", va="center", fontsize=7.5, **kw)

    def inhibit(ax, x, txt):
        arrow(ax, (x, 0.83), (x, 0.49), style="-[", color=MUTED)
        lab(ax, (x, 0.91), txt, color=MUTED)

    ax = axs[0]                                           # A
    box(ax, (0.5, 0.42), "$R$")
    arrow(ax, (0.04, 0.42), (0.38, 0.42)); lab(ax, (0.2, 0.33), "$k_{in}$")
    arrow(ax, (0.62, 0.42), (0.96, 0.42)); lab(ax, (0.8, 0.33), "$k_{out}$")
    inhibit(ax, 0.2, "$I(C)$")

    ax = axs[1]                                           # B
    box(ax, (0.3, 0.42), "$P$"); box(ax, (0.74, 0.42), "$R$")
    arrow(ax, (0.0, 0.42), (0.18, 0.42)); lab(ax, (0.08, 0.33), "$k_0$")
    arrow(ax, (0.42, 0.42), (0.62, 0.42)); lab(ax, (0.52, 0.33), "$k_p$")
    arrow(ax, (0.86, 0.42), (1.0, 0.42)); lab(ax, (0.93, 0.33), "$k_{out}$")
    inhibit(ax, 0.52, "$I(C)$")

    ax = axs[2]                                           # C
    box(ax, (0.55, 0.42), "$R$"); box(ax, (0.55, 0.1), "$M$", fc="#e3f3ec")
    arrow(ax, (0.04, 0.42), (0.43, 0.42)); lab(ax, (0.08, 0.33), "$k_{in}$")
    arrow(ax, (0.67, 0.42), (0.98, 0.42)); lab(ax, (0.83, 0.5), "$k_{out}$")
    inhibit(ax, 0.2, "$I(C)$")
    arrow(ax, (0.62, 0.33), (0.62, 0.19), ls="--"); lab(ax, (0.76, 0.24), "$R_0/R$")
    arrow(ax, (0.43, 0.1), (0.3, 0.4), ls="--", rad=-0.35)
    lab(ax, (0.2, 0.15), "$\\times M$")

    ax = axs[3]                                           # D
    box(ax, (0.5, 0.42), "$R$"); box(ax, (0.5, 0.1), "$\\rho$", fc="#fbefd6")
    arrow(ax, (0.04, 0.42), (0.38, 0.42)); lab(ax, (0.2, 0.33), "$k_{in}$")
    arrow(ax, (0.62, 0.42), (0.96, 0.42)); lab(ax, (0.8, 0.33), "$k_{out}$")
    inhibit(ax, 0.2, "$I_{max}\\,\\rho\\,\\phi$")
    arrow(ax, (0.04, 0.1), (0.38, 0.1)); lab(ax, (0.2, 0.19), "$k_{deg}$")
    arrow(ax, (0.62, 0.1), (0.96, 0.1)); lab(ax, (0.8, 0.19), "$k_{deg}+k_{int}\\phi$")

    for ax, m in zip(axs, MODELS):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.0, 1.0, m, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.14)
    save(fig, name)


def fig2_abrupt(sim, name="Fig2"):
    d = sim["t"] / 24
    fig, axs = plt.subplots(3, 1, figsize=(FIG_W, 6.0), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1.6, 1.2]})
    ax = axs[0]
    ax.plot(d, sim["C"], color=INK, lw=1.2)
    ax.set_ylabel("$C$ / $IC_{50}$")
    ax = axs[1]
    ax.axhline(1, color=MUTED, lw=0.6)
    for m in MODELS:
        ax.plot(d, sim["R"][m], color=COLORS[m], ls=STYLES[m], label=LABELS[m])
    ax.text(21.4, 1.85, "withdrawal ($R>R_0$)", color=MUTED)
    ax.text(0.3, 0.1, "opioid effect ($R<R_0$)", color=MUTED)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("$R/R_0$")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax = axs[2]
    ax.axhline(1, color=MUTED, lw=0.6)
    names = {"B": "B  precursor $P/P_0$", "C": "C  moderator $M$",
             "D": "D  receptor density $\\rho$"}
    for m in ["B", "C", "D"]:
        ax.plot(d, sim["state"][m], color=COLORS[m], ls=STYLES[m], label=names[m])
    ax.set_ylabel("Adaptive state")
    ax.set_xlabel("Time (days)")
    ax.legend(frameon=False, loc="upper right")
    for ax, l in zip(axs, "abc"):
        style(ax); letter(ax, l, x=-0.07)
        ax.axvline(N_DAYS, color="#b5b5ae", lw=0.6, ls="--")
    fig.tight_layout()
    save(fig, name)


def fig3_scenarios(sims, name="Fig3"):
    fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 4.6), sharey="row")
    sc = {"abrupt": ("#2a78d6", "-", "abrupt stop"),
          "taper": ("#eb6834", "--", f"{TAPER_DAYS}-day linear taper")}
    for j, m in enumerate(["B", "C"]):
        ax = axs[0, j]
        for s, (col, ls, lab) in sc.items():
            sim = sims[s]
            ax.plot(sim["t"] / 24 - N_DAYS, np.clip(sim["R"][m] - 1, 0, None),
                    color=col, ls=ls, label=lab)
        ax.set_xlim(-1, 25)
        ax.set_xlabel("Days after last full dose")
        ax.text(0.98, 0.95, f"Model {m}", transform=ax.transAxes, ha="right",
                va="top", fontsize=8, fontweight="bold")
        ax = axs[1, j]
        sim = sims["naloxone"]
        ax.plot(sim["t"] - sim["t_ref"], np.clip(sim["R"][m] - 1, 0, None),
                color="#1baf7a", ls="-.", label="naloxone during maintenance")
        ax.set_xlim(-6, 24)
        ax.set_xlabel("Hours after naloxone (next dose at +4 h)")
        ax.text(0.98, 0.95, f"Model {m}", transform=ax.transAxes, ha="right",
                va="top", fontsize=8, fontweight="bold")
    for ax, l in zip(axs.flat, "abcd"):
        style(ax); letter(ax, l, x=-0.12)
    for i in range(2):
        axs[i, 0].set_ylabel("$W = R/R_0 - 1$")
        axs[i, 0].legend(frameon=False, loc="upper center" if i == 0 else "center right")
    fig.tight_layout()
    save(fig, name)


def fig4_theory(name="Fig4"):
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 2.8))
    ax = axs[0]
    I_line = np.linspace(0.05, 0.85, 100)
    ax.plot(I_line, b_peak_rebound(I_line), color=COLORS["B"], ls="-",
            label="B  analytic (Eq. 13)")
    ax.plot(I_line, c_peak_rebound_bound(I_line), color=COLORS["C"], ls="--",
            label="C  bound (Eq. 19)")
    I_pts = np.array([0.1, 0.3, 0.5, 0.675, 0.8])
    pb, pc = [], []
    for I in I_pts:
        s = step_offset(I)
        pb.append(np.max(s["R"]["B"]) - 1); pc.append(np.max(s["R"]["C"]) - 1)
    ax.plot(I_pts, pb, "o", ms=5, mfc="white", mec=COLORS["B"], mew=1.2,
            label="B  simulated")
    ax.plot(I_pts, pc, "s", ms=5, mfc="white", mec=COLORS["C"], mew=1.2,
            label="C  simulated")
    ax.set_xlabel("Steady-state inhibition $I$ before offset")
    ax.set_ylabel("Peak withdrawal $W_{max}$")
    ax.legend(frameon=False)
    ax = axs[1]
    I0 = 0.675
    for r, ls, col in [(50, "-", "#1baf7a"), (5, "--", "#2a78d6"),
                       (1, ":", "#eb6834")]:
        p = dict(PAR, ktol=PAR["kout"] / r)
        s = step_offset(I0, p=p, t_end=120.0)
        lam = c_eigenvalues(p["kout"], p["ktol"])
        kind = "oscillatory" if abs(lam[0].imag) > 0 else "monotone"
        ax.plot(s["t"], s["R"]["C"], color=col, ls=ls,
                label=f"$k_{{out}}/k_{{tol}}$ = {r} ({kind})")
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.set_xlabel("Hours after instantaneous offset")
    ax.set_ylabel("Model C  $R/R_0$")
    ax.legend(frameon=False)
    for ax, l in zip(axs, "ab"):
        style(ax); letter(ax, l, x=-0.14)
    fig.tight_layout()
    save(fig, name)
    return I_pts, pb, pc


def fig5_sweeps(name="Fig5"):
    days = (1, 2, 3, 5, 7, 10, 14, 21, 28, 42)
    dur_peak = {m: [] for m in "BC"}
    for n in days:
        sim = simulate_oral("abrupt", n_days=n, t_end=(n + 30) * TAU)
        for m in "BC":
            dur_peak[m].append(withdrawal_metrics(sim["t"], sim["R"][m],
                                                  n * TAU)["peak"])
    hls = (1, 2, 4, 8, 12, 24, 36, 48, 72, 96)
    hl_peak, hl_t = {m: [] for m in "BC"}, {m: [] for m in "BC"}
    t_stop = N_DAYS * TAU
    for h in hls:
        cfun = infusion_conc(np.log(2) / h, t_stop)
        sim = integrate(occupancy_fn(cfun), t_stop + 30 * TAU)
        for m in "BC":
            wm = withdrawal_metrics(sim["t"], sim["R"][m], t_stop)
            hl_peak[m].append(wm["peak"])
            hl_t[m].append(wm["t_peak"] / 24 if wm["peak"] > 1e-3 else np.nan)
    fig, axs = plt.subplots(1, 3, figsize=(FIG_W, 2.5))
    for m, mk in [("B", "o"), ("C", "s")]:
        kw = dict(color=COLORS[m], ls=STYLES[m], marker=mk, ms=4,
                  mfc="white", mew=1.1, label=LABELS[m])
        axs[0].plot(days, dur_peak[m], **kw)
        axs[1].plot(hls, hl_peak[m], **kw)
        axs[2].plot(hls, hl_t[m], **kw)
    axs[0].set_xlabel("Duration of maintenance (days)")
    axs[0].set_ylabel("Peak withdrawal $W_{max}$")
    axs[1].set_ylabel("Peak withdrawal $W_{max}$")
    axs[2].set_ylabel("Time to $W_{max}$ (days)")
    for ax in axs[1:]:
        ax.set_xscale("log")
        ax.set_xticks([1, 4, 12, 24, 48, 96])
        ax.set_xticklabels(["1", "4", "12", "24", "48", "96"])
        ax.set_xlabel("Elimination half-life (h)")
    axs[0].legend(frameon=False, loc="center right")
    axs[2].text(72, 0.4, "C: no\nrebound", ha="center", fontsize=7, color=MUTED)
    for ax, l in zip(axs, "abc"):
        style(ax); letter(ax, l, x=-0.2)
    fig.tight_layout()
    save(fig, name)
    return days, dur_peak, hls, hl_peak, hl_t


# --------------------------------------------------------------- robustness
# Single-model variants, state y = [R, X, rho], X = P (B) or M (C, Cint).
#   B     precursor pool, optional second precursor loss ks (Eq. 21)
#   C     proportional ("leaky") moderator, Eq. 8
#   Cint  integral moderator dM/dt = ktol (1 - R)  (Eq. 22)
# with_d adds receptor down-regulation (Eq. 9) to the drug term.
def rhs_variant(t, y, phi_fn, p, model, with_d):
    R, X, rho = y
    phi = phi_fn(t)
    inh = p["imax"] * (rho if with_d else 1.0) * phi
    drho = p["kdeg"] * (1 - rho) - p["kint"] * phi * rho if with_d else 0.0
    kout = p["kout"]
    if model == "B":
        kp, ks = p["kp"], p.get("ks", 0.0)
        dX = (kp + ks) - kp * (1 - inh) * X - ks * X
        dR = kout * ((1 - inh) * X - R)
    elif model == "C":
        dX = p["ktol"] * (1 / R - X)
        dR = kout * (X * (1 - inh) - R)
    else:  # Cint
        dX = p["ktol"] * (1 - R)
        dR = kout * (X * (1 - inh) - R)
    return [dR, dX, drho]


def run_variant(model, scenario="abrupt", with_d=False, hill=1.0, p=PAR):
    times, amounts = oral_schedule(scenario)
    cfun = oral_conc(times, amounts)
    phi_fn = lambda t: (lambda c: c ** hill / (1 + c ** hill))(cfun(t))
    t_eval = np.arange(0.0, T_END, 0.1)
    sol = solve_ivp(rhs_variant, (0.0, T_END), [1.0, 1.0, 1.0],
                    args=(phi_fn, p, model, with_d), t_eval=t_eval,
                    method="LSODA", max_step=0.25, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0]


def variant_metrics(model, with_d=False, hill=1.0, p=PAR):
    out = {}
    tA, RA = run_variant_reference(hill)
    ref = peak_effect(tA, RA, N_DAYS - 1)
    for s in ("abrupt", "taper"):
        t, R = run_variant(model, s, with_d, hill, p)
        wm = withdrawal_metrics(t, R, N_DAYS * TAU)
        abec, abrc = areas(t, R)
        out[s] = dict(ti=1 - peak_effect(t, R, N_DAYS - 1) / ref, **wm,
                      abec=abec, abrc=abrc)
    return out


_REF = {}
def run_variant_reference(hill):
    """Model A with the same occupancy function (for the tolerance index)."""
    if hill not in _REF:
        times, amounts = oral_schedule("abrupt")
        cfun = oral_conc(times, amounts)
        phi = lambda t: (lambda c: c ** hill / (1 + c ** hill))(cfun(t))
        sim = integrate(phi, T_END)
        _REF[hill] = (sim["t"], sim["R"]["A"])
    return _REF[hill]


VARIANTS = [
    ("B", "B (reference)", dict(model="B")),
    ("B", "B, Hill n = 0.5", dict(model="B", hill=0.5)),
    ("B", "B, Hill n = 2", dict(model="B", hill=2.0)),
    ("B", "B + second precursor loss (ks = kp/4)",
     dict(model="B", p=dict(PAR, ks=PAR["kp"] / 4))),
    ("B", "B + second precursor loss (ks = kp)",
     dict(model="B", p=dict(PAR, ks=PAR["kp"]))),
    ("B", "B + receptor down-regulation", dict(model="B", with_d=True)),
    ("C", "C (reference)", dict(model="C")),
    ("C", "C, Hill n = 0.5", dict(model="C", hill=0.5)),
    ("C", "C, Hill n = 2", dict(model="C", hill=2.0)),
    ("C", "C + receptor down-regulation", dict(model="C", with_d=True)),
    ("Cint", "C with integral moderator", dict(model="Cint")),
]


def robustness_report():
    print("\nRobustness (abrupt | taper):")
    print(f"{'variant':<42}{'TI':>6}{'Wpk':>7}{'AUC':>7}{'ABRC/ABEC':>10}"
          f" |{'Wpk':>7}{'AUC':>7}{'ABRC/ABEC':>10}")
    rows = {}
    for _, name, kw in VARIANTS:
        m = variant_metrics(**kw)
        rows[name] = m
        a, tp = m["abrupt"], m["taper"]
        print(f"{name:<42}{a['ti']:6.2f}{a['peak']:7.2f}{a['auc']:7.1f}"
              f"{a['abrc'] / a['abec']:10.3f} |{tp['peak']:7.2f}{tp['auc']:7.1f}"
              f"{tp['abrc'] / tp['abec']:10.3f}")
    print("\nIntegral moderator eigenvalues at baseline:",
          np.roots([1, PAR["kout"], PAR["kout"] * PAR["ktol"]]))
    return rows


def c_bound_ratio(I, eps):
    p = dict(PAR, ktol=PAR["kout"] * eps)
    s = step_offset(I, p=p, t_end=max(200.0, 5 / p["ktol"]))
    return (np.max(s["R"]["C"]) - 1) / c_peak_rebound_bound(I)


def fig6_robustness(name="Fig6"):
    fig, axs = plt.subplots(1, 3, figsize=(FIG_W, 2.6))
    ax = axs[0]
    eps = np.logspace(-3, np.log10(0.5), 12)
    print("\nEq. 19 tightness: ktol/kout, ratio simulated/bound (I=0.3, 0.675, 0.9*0.95)")
    for I, ls, mk, col in [(0.3, "--", "s", "#2a78d6"), (0.675, "-", "o", "#1baf7a"),
                           (0.855, ":", "^", "#eb6834")]:
        r = [c_bound_ratio(I, e) for e in eps]
        ax.plot(eps, r, color=col, ls=ls, marker=mk, ms=3.5, mfc="white",
                mew=1, label=f"$I$ = {I}")
        print(f"  I={I}: " + "  ".join(f"{e:.3g}:{x:.3f}" for e, x in zip(eps, r)))
    ax.set_xscale("log")
    ax.set_xlabel("$k_{tol}/k_{out}$")
    ax.set_ylabel("Simulated $W_{max}$ / bound (Eq. 19)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")

    ax = axs[1]
    for model, with_d, lab, col, ls in [
            ("B", False, "B", COLORS["B"], "-"),
            ("B", True, "B + receptor down-reg.", COLORS["B"], ":"),
            ("C", False, "C", COLORS["C"], "--"),
            ("C", True, "C + receptor down-reg.", COLORS["C"], ":"),
            ("Cint", False, "C, integral moderator", "#2a78d6", "-.")]:
        t, R = run_variant(model, "abrupt", with_d)
        ax.plot(t / 24 - N_DAYS, np.clip(R - 1, 0, None), color=col, ls=ls,
                label=lab)
    ax.set_xlim(-1, 20)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Days after last dose")
    ax.set_ylabel("$W = R/R_0 - 1$")
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)

    ax = axs[2]
    ks_rel = [0, 0.1, 0.25, 0.5, 1, 2]
    for s, col, ls, mk in [("abrupt", "#2a78d6", "-", "o"),
                           ("taper", "#eb6834", "--", "s")]:
        ratio = []
        for k in ks_rel:
            t, R = run_variant("B", s, p=dict(PAR, ks=PAR["kp"] * k))
            abec, abrc = areas(t, R)
            ratio.append(abrc / abec)
        ax.plot(ks_rel, ratio, color=col, ls=ls, marker=mk, ms=4, mfc="white",
                mew=1.1, label=s)
        print(f"  ABRC/ABEC vs ks/kp ({s}): " +
              "  ".join(f"{k}:{r:.3f}" for k, r in zip(ks_rel, ratio)))
    ax.set_xlabel("$k_s/k_p$ (second precursor loss)")
    ax.set_ylabel("ABRC / ABEC  (Model B)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    for ax, l in zip(axs, "abc"):
        style(ax); letter(ax, l, x=-0.22)
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    sims = {s: simulate_oral(s) for s in ["abrupt", "taper", "naloxone"]}
    report(sims)

    print("\nModel B analytic vs simulated (instantaneous offset from steady state)")
    print(f"  t_peak analytic {b_peak_time():.2f} h")
    I_pts, pb, pc = fig4_theory()
    for I, b, c in zip(I_pts, pb, pc):
        print(f"  I={I:.3f}  B sim {b:.4f} analytic {b_peak_rebound(I):.4f}  |"
              f"  C sim {c:.4f} bound {c_peak_rebound_bound(I):.4f}")
    print("\nModel C eigenvalues at baseline / any steady state (1/h):")
    for r in (50, 5, 1):
        lam = c_eigenvalues(PAR["kout"], PAR["kout"] / r)
        print(f"  kout/ktol={r:>3}: {lam[0]:.4f}, {lam[1]:.4f}")
    print(f"  oscillation window: {3 - 2 * np.sqrt(2):.3f} < kout/ktol "
          f"< {3 + 2 * np.sqrt(2):.3f}")

    robustness_report()
    fig6_robustness()

    fig1_schematic()
    fig2_abrupt(sims["abrupt"])
    fig3_scenarios(sims)
    days, dp, hls, hp, ht = fig5_sweeps()
    print("\nPeak W vs maintenance duration (B, C):")
    for n, b, c in zip(days, dp["B"], dp["C"]):
        print(f"  {n:>3} d  {b:.3f}  {c:.3f}")
    print("\nInfusion to Css, stop at day 21: half-life, peak W (B, C), t_peak d (B, C)")
    for h, b, c, tb, tc in zip(hls, hp["B"], hp["C"], ht["B"], ht["C"]):
        print(f"  {h:>3} h  {b:.3f}  {c:.3f}   {tb:.2f}  {tc:.2f}")
