"""
Fit alternative withdrawal models to neonatal opioid withdrawal data digitized
from Mizuno et al. (2021) Clin Pharmacokinet 60:249-259, Fig. 5 (two infants
treated with sublingual buprenorphine, 12-h mean Finnegan scores).

Drug input: the individual model-predicted blood concentration curve shown in
the same figure (cmean column), interpolated in time and zero outside the
plotted range. I(t) = C/(IC50 + C) (Imax = 1, as in the original model).

Models (time in h after birth, F = Finnegan score):
  M0    published structure: dF/dt = Kin (1 - I) exp(-Kremi t) - Kout F, F(0)=0
        (remission follows a postnatal-age clock, independent of treatment)
  MB    precursor pool (integral adaptation), in-utero state carried over:
        dP/dt = kp [1 - (1 - I) P], dR/dt = kout [(1 - I) P - R],
        P(0) = Pb, R(0) = 1, F = s max(R - 1, 0)
  MC    proportional moderator: dR/dt = kout [M (1 - I) - R],
        dM/dt = ktol (1/R - M), M(0) = Mb, R(0) = 1/Mb, F = s max(R - 1, 0)
  MCint integral moderator: dM/dt = ktol (1 - R), M(0) = Mb, R(0) = 1,
        F = s max(R - 1, 0)
  MBimax sensitivity: MB with I = Imax C/(IC50 + C) (partial agonism)

Additive-error least squares; global search by differential evolution on
log10 parameters, then local polish. Run: python fit_mizuno.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares

plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.3,
                     "legend.fontsize": 7, "mathtext.fontset": "custom",
                     "mathtext.it": "Arial:italic", "mathtext.rm": "Arial"})

DT = 0.25                       # RK4 step (h)
PATIENTS = ["A", "B"]
T_END = {"A": 28 * 24.0, "B": 24 * 24.0}


def read_csv(path):
    lines = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    cols = lines[0].split(",")
    data = np.array([[float(v) for v in l.split(",")] for l in lines[1:]])
    return {c: data[:, i] for i, c in enumerate(cols)}


def load(pid):
    f = read_csv(f"finnegan_patient{pid}.csv")
    c = read_csv(f"concentration_patient{pid}.csv")
    return f["day"] * 24, f["finnegan"], c["day"] * 24, c["cmean"]


def conc_grid(tc, cc, t_end):
    t = np.arange(0.0, t_end + DT, DT)
    c = np.interp(t, tc, cc, left=0.0, right=0.0)
    return t, np.clip(c, 0, None)


# ---------------------------------------------------------------- models
# Each model: (param names, log10 bounds, init(theta) -> y0, rhs(y, I, theta)
#              -> dy, output(y, theta) -> F). theta arrays have shape (k, npop).
def m0_init(th):
    return [np.zeros(th.shape[1])]


def m0_rhs(y, I, t, th):
    kin, kout, krem, _ = th
    return [kin * (1 - I) * np.exp(-krem * t) - kout * y[0]]


def mb_init(th):
    pb = th[4]
    return [pb, np.ones_like(pb)]


def mb_rhs(y, I, t, th):
    kp, kout = th[0], th[1]
    P, R = y
    return [kp * (1 - (1 - I) * P), kout * ((1 - I) * P - R)]


def mc_init(th):
    mb = th[4]
    return [mb, 1 / mb]


def mc_rhs(y, I, t, th):
    kout, ktol = th[0], th[1]
    M, R = y
    R = np.maximum(R, 1e-6)
    return [ktol * (1 / R - M), kout * (M * (1 - I) - R)]


def mci_init(th):
    mb = th[4]
    return [mb, np.ones_like(mb)]


def mci_rhs(y, I, t, th):
    kout, ktol = th[0], th[1]
    M, R = y
    return [ktol * (1 - R), kout * (M * (1 - I) - R)]


def mbi_rhs(y, I, t, th):
    """Model B with a partial-agonist ceiling: I -> Imax * I."""
    return mb_rhs(y, th[5] * I, t, th)


def withdrawal_out(y, th):
    return th[3] * np.maximum(y[-1] - 1, 0)


MODELS = {
    "M0": dict(label="Published (open-loop)",
               names=["Kin", "Kout", "Kremi", "IC50"],
               bounds=[(-2, 1), (-3, 0), (-4.5, -1.3), (-1.5, 1)],
               ic50=3, init=m0_init, rhs=m0_rhs, out=lambda y, th: y[0]),
    "MB": dict(label="B  Precursor pool (integral)",
               names=["kp", "kout", "IC50", "s", "Pb"],
               bounds=[(-3.5, -0.5), (-2, 0.5), (-1.5, 1), (-0.5, 2.5), (0.005, 1.7)],
               ic50=2, init=mb_init, rhs=mb_rhs, out=withdrawal_out),
    "MC": dict(label="C  Proportional moderator",
               names=["kout", "ktol", "IC50", "s", "Mb"],
               bounds=[(-2, 0.5), (-3.5, -0.5), (-1.5, 1), (-0.5, 2.5), (0.005, 1.3)],
               ic50=2, init=mc_init, rhs=mc_rhs, out=withdrawal_out),
    "MCint": dict(label="C  Integral moderator",
                  names=["kout", "ktol", "IC50", "s", "Mb"],
                  bounds=[(-2, 0.5), (-4, -0.5), (-1.5, 1), (-0.5, 2.5), (0.005, 1.7)],
                  ic50=2, init=mci_init, rhs=mci_rhs, out=withdrawal_out),
    "MBimax": dict(label="B  Precursor pool + Imax (sensitivity)",
                   names=["kp", "kout", "IC50", "s", "Pb", "Imax"],
                   bounds=[(-3.5, -0.5), (-2, 0.5), (-2, 2), (-0.5, 2.5), (0.005, 1.7),
                           (-1.5, 0)],
                   ic50=2, init=mb_init, rhs=mbi_rhs, out=withdrawal_out),
}
MAIN = ["M0", "MB", "MC", "MCint"]


def simulate(model, logth, t, c, zero_drug=False):
    """Vectorised RK4 over a population of parameter vectors (k, npop)."""
    m = MODELS[model]
    logth = np.asarray(logth, dtype=float)
    if logth.ndim == 1:
        logth = logth[:, None]
    th = 10.0 ** logth                               # shape (k, npop)
    ic50 = th[m["ic50"]]
    y = [np.array(v, dtype=float) for v in m["init"](th)]
    cd = np.zeros_like(c) if zero_drug else c
    I_all = cd[:, None] / (ic50[None, :] + cd[:, None])
    out = np.empty((t.size, th.shape[1]))
    out[0] = m["out"](y, th)
    f = m["rhs"]
    for i in range(t.size - 1):
        Ia, Ib = I_all[i], I_all[i + 1]
        Im = 0.5 * (Ia + Ib)
        ti = t[i]
        k1 = f(y, Ia, ti, th)
        k2 = f([a + 0.5 * DT * b for a, b in zip(y, k1)], Im, ti + DT / 2, th)
        k3 = f([a + 0.5 * DT * b for a, b in zip(y, k2)], Im, ti + DT / 2, th)
        k4 = f([a + DT * b for a, b in zip(y, k3)], Ib, ti + DT, th)
        y = [a + DT / 6 * (p + 2 * q + 2 * r + s) for a, p, q, r, s in zip(y, k1, k2, k3, k4)]
        out[i + 1] = m["out"](y, th)
    return out


def fit(model, pid):
    tf, F, tc, cc = load(pid)
    t, c = conc_grid(tc, cc, T_END[pid])
    idx = np.clip(np.round(tf / DT).astype(int), 0, t.size - 1)

    def sse(logth):
        pred = simulate(model, np.asarray(logth), t, c)[idx]
        res = pred - F[:, None]
        v = np.sum(res ** 2, axis=0)
        return np.where(np.isfinite(v), v, 1e12)

    b = MODELS[model]["bounds"]
    de = differential_evolution(sse, b, vectorized=True, popsize=20, maxiter=250,
                                tol=1e-8, seed=1, polish=False, updating="deferred")
    lo, hi = np.array(b).T
    ls = least_squares(lambda p: simulate(model, p, t, c)[idx, 0] - F,
                       np.clip(de.x, lo + 1e-9, hi - 1e-9), bounds=(lo, hi))
    x = ls.x if np.sum(ls.fun ** 2) <= de.fun else de.x
    rss = float(np.sum((simulate(model, x, t, c)[idx, 0] - F) ** 2))
    n, k = F.size, len(b)
    # approximate standard errors of log10 parameters from the Jacobian
    J = ls.jac
    sigma2 = rss / (n - k)
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return dict(model=model, pid=pid, logx=x.tolist(), se_log10=se.tolist(),
                rss=rss, n=n, k=k, rmse=float(np.sqrt(rss / n)),
                aic=float(n * np.log(rss / n) + 2 * k),
                aicc=float(n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)),
                bic=float(n * np.log(rss / n) + k * np.log(n)))


def signed_area(model, logx, pid, zero_drug=False):
    """Integral of (F-equivalent) withdrawal over the simulated course (score x day)."""
    tf, F, tc, cc = load(pid)
    t, c = conc_grid(tc, cc, 60 * 24.0)
    y = simulate(model, np.asarray(logx), t, c, zero_drug=zero_drug)[:, 0]
    return float(np.trapezoid(y, t) / 24), t, y


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]                       # optional subset of models
    try:
        results = json.load(open("fit_results.json")) if only else []
    except FileNotFoundError:
        results = []
    results = [r for r in results if r["model"] not in only]
    for pid in PATIENTS:
        for model in (only or MODELS):
            r = fit(model, pid)
            results.append(r)
            names = MODELS[model]["names"]
            ps = ", ".join(f"{nm}={10 ** v:.4g}" for nm, v in zip(names, r["logx"]))
            print(f"{pid} {model:<6} RMSE {r['rmse']:.2f}  AICc {r['aicc']:.1f}  "
                  f"BIC {r['bic']:.1f}  | {ps}")
    json.dump(results, open("fit_results.json", "w"), indent=1)
