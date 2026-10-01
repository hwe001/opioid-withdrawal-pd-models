"""
Moving-block bootstrap of the neonatal model comparison (fit_mizuno.py must
have been run first). Each replicate resamples blocks of consecutive 12-h
scores (block length 4 = 2 days, circular), adds digitization noise with the
SDs measured by digitization_error.py, rescales the concentration input by
the measured relative digitization SD, and refits every model locally from
its original optimum. Records AICc, the selected model and the treated /
untreated cumulative-score ratio.

Run: python bootstrap_fits.py [n_boot]   (parallel over infant x model)
"""

import json
import sys
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
import fit_mizuno as fm

BLOCK = 4
DIG = json.load(open("digitization_error.json"))


def boot_indices(n, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / BLOCK)))
    idx = np.concatenate([(s + np.arange(BLOCK)) % n for s in starts])[:n]
    return np.sort(idx)


def local_fit(model, x0, t, c, tobs, F):
    b = np.array(fm.MODELS[model]["bounds"])
    lo, hi = b[:, 0], b[:, 1]
    idx = np.clip(np.round(tobs / fm.DT).astype(int), 0, t.size - 1)
    h = 1e-4

    def resid(p):
        return fm.simulate(model, p, t, c)[idx, 0] - F

    def jac(p):
        P = np.column_stack([p] + [p + h * e for e in np.eye(p.size)])
        Y = fm.simulate(model, P, t, c)[idx]
        return (Y[:, 1:] - Y[:, [0]]) / h

    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    r = least_squares(resid, x0, jac=jac, bounds=(lo, hi), max_nfev=60)
    return r.x, float(np.sum(r.fun ** 2))


def cumulative(model, x, pid, scale):
    tf, F, tc, cc = fm.load(pid)
    t, c = fm.conc_grid(tc, cc * scale, 60 * 24.0)
    y = fm.simulate(model, np.column_stack([x, x]), t, c)
    yu = fm.simulate(model, x, t, c, zero_drug=True)[:, 0]
    return float(np.trapezoid(y[:, 0], t) / np.trapezoid(yu, t))


def run(args):
    pid, model, n_boot = args
    base = {r["model"]: r for r in json.load(open("fit_results.json")) if r["pid"] == pid}
    x_orig = np.array(base[model]["logx"])
    tf, F, tc, cc = fm.load(pid)
    rng = np.random.default_rng(2024 + (ord(pid) * 31) % 97)   # same resamples for every model
    out = []
    for b in range(n_boot):
        idx = boot_indices(F.size, rng)
        tobs = tf[idx] + rng.normal(0, DIG["sd_time_h"], idx.size)
        Fb = F[idx] + rng.normal(0, DIG["sd_score"], idx.size)
        scale = 1 + rng.normal(0, DIG["sd_conc_rel"])
        t, c = fm.conc_grid(tc, cc * scale, fm.T_END[pid])
        x, rss = local_fit(model, x_orig, t, c, tobs, Fb)
        n, k = Fb.size, x.size
        aicc = n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        out.append(dict(b=b, aicc=float(aicc), rss=rss, logx=x.tolist(),
                        ratio=cumulative(model, x, pid, scale)))
    return pid, model, out


if __name__ == "__main__":
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    jobs = [(p, m, n_boot) for p in fm.PATIENTS for m in fm.MAIN]
    with Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
    boot = {f"{p}|{m}": o for p, m, o in res}
    json.dump(boot, open("bootstrap_results.json", "w"))

    for p in fm.PATIENTS:
        A = np.array([[r["aicc"] for r in boot[f"{p}|{m}"]] for m in fm.MAIN])
        win = np.argmin(A, axis=0)
        print(f"\nInfant {p} ({n_boot} replicates)")
        for i, m in enumerate(fm.MAIN):
            d = A[i] - A[0]
            ratio = np.array([r["ratio"] for r in boot[f"{p}|{m}"]])
            print(f"  {m:<6} selected {np.mean(win == i) * 100:5.1f} %   "
                  f"dAICc vs M0 median {np.median(d):6.1f} [{np.percentile(d, 2.5):6.1f}, "
                  f"{np.percentile(d, 97.5):6.1f}]   treated/untreated ratio "
                  f"{np.median(ratio):.2f} [{np.percentile(ratio, 2.5):.2f}, "
                  f"{np.percentile(ratio, 97.5):.2f}]")
