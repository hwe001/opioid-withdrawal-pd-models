"""Compare digitizations at different resolutions (run digitize_mizuno.py first)."""
import json
import numpy as np
import fit_mizuno as fm

ref_dpi, others = 600, [450, 525, 675, 750, 900]
ref_f = json.load(open(f"dig{ref_dpi}_finnegan.json"))
ref_c = json.load(open(f"dig{ref_dpi}_conc.json"))
dt, df, dc = [], [], []
for d in others:
    f = json.load(open(f"dig{d}_finnegan.json"))
    c = json.load(open(f"dig{d}_conc.json"))
    for k in ("LA", "LB"):
        a, b = np.array(ref_f[k]), np.array(f[k])
        assert len(a) == len(b), (d, k)
        dt += list((b[:, 0] - a[:, 0]) * 24)
        df += list(b[:, 1] - a[:, 1])
    for k in ("UA", "UB"):
        ta, ca = np.array(ref_c[k]["t"]), np.array(ref_c[k]["cmean"])
        tb, cb = np.array(c[k]["t"]), np.array(c[k]["cmean"])
        grid = np.linspace(max(ta[0], tb[0]), min(ta[-1], tb[-1]), 2000)
        ia, ib = np.interp(grid, ta, ca), np.interp(grid, tb, cb)
        m = ia > 0.1
        dc += list((ib[m] - ia[m]) / ia[m])
dt, df, dc = map(np.array, (dt, df, dc))
print(f"score time:   SD {dt.std():.3f} h, max |diff| {abs(dt).max():.3f} h")
print(f"score value:  SD {df.std():.4f}, max |diff| {abs(df).max():.4f}")
print(f"concentration (relative): SD {dc.std() * 100:.2f} %, max |diff| {abs(dc).max() * 100:.2f} %")
# residual SD of the best fits, for comparison
res = json.load(open("fit_results.json"))
for r in res:
    if r["model"] == "M0":
        print(f"infant {r['pid']}: residual RMSE of best fit {r['rmse']:.2f}")
json.dump({"sd_time_h": float(dt.std()), "sd_score": float(df.std()),
           "sd_conc_rel": float(dc.std())}, open("digitization_error.json", "w"))
