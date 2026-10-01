"""
Digitize Fig. 5 of Mizuno et al. (2021) Clin Pharmacokinet 60:249-259.

Usage: python digitize_mizuno.py <path-to-pdf> <dpi> <out-prefix>
Requires pdftoppm (Poppler). Writes <out-prefix>_finnegan.json and
<out-prefix>_conc.json. Panel frames and ticks are located automatically, so
the procedure can be repeated at different resolutions to estimate
digitization error (see digitization_error.py).
"""

import json
import subprocess
import sys
import tempfile
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2

PAGE = 9                                   # PDF page with Fig. 5
X_VALUES = {"UA": [0, 7, 14, 21, 28], "LA": [0, 7, 14, 21, 28],
            "UB": [0, 7, 14, 21], "LB": [0, 7, 14, 21]}
Y_VALUES = {"UA": [1.5, 1.0, 0.5, 0.0], "LA": [20, 15, 10, 5, 0],
            "UB": [2.0, 1.5, 1.0, 0.5, 0.0], "LB": [20, 15, 10, 5, 0]}


def render(pdf, dpi):
    d = tempfile.mkdtemp()
    subprocess.run(["pdftoppm", "-f", str(PAGE), "-l", str(PAGE), "-r", str(dpi),
                    "-png", pdf, os.path.join(d, "p")], check=True)
    f = [x for x in os.listdir(d) if x.endswith(".png")][0]
    return np.asarray(Image.open(os.path.join(d, f)).convert("RGB")).astype(int)


def runs(idx):
    """Centres of runs of consecutive integers."""
    idx = list(idx)
    if not idx:
        return []
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v != p + 1:
            out.append((s + p) / 2)
            s = v
        p = v
    out.append((s + p) / 2)
    return out


def frames(gray, sc):
    """Return the four panel boxes (x0, x1, y0, y1) in reading order."""
    H, W = gray.shape
    rows = [r for r in runs(np.where(gray[:, int(0.18 * W):int(0.49 * W)].sum(1)
                                     > 0.9 * 0.31 * W)[0])]
    rows = [r for r in rows if 0.08 * H < r < 0.45 * H]
    assert len(rows) == 4, rows
    boxes = {}
    for (name_u, name_l), (xa, xb) in {("UA", "LA"): (0.1, 0.5),
                                       ("UB", "LB"): (0.5, 0.9)}.items():
        for name, (ya, yb) in {name_u: (rows[0], rows[1]),
                               name_l: (rows[2], rows[3])}.items():
            band = gray[int(ya) + 5:int(yb) - 5, int(xa * W):int(xb * W)]
            cols = runs(np.where(band.sum(0) > 0.9 * band.shape[0])[0])
            x0, x1 = cols[0] + xa * W, cols[-1] + xa * W
            boxes[name] = (int(round(x0)), int(round(x1)), int(ya), int(yb))
    return boxes


def calibration(gray, box, sc):
    x0, x1, y0, y1 = box
    a, b = int(22 * sc), int(6 * sc)
    yt = runs(np.where(gray[y0 - 5:y1 + 5, x0 - a:x0 - b].sum(1) >= (a - b) * 0.5)[0] + y0 - 5)
    xt = runs(np.where(gray[y1 + b:y1 + a, x0 - 5:x1 + 5].sum(0) >= (a - b) * 0.5)[0] + x0 - 5)
    return xt, yt


def digitize(pdf, dpi):
    sc = dpi / 600
    im = render(pdf, dpi)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    gray = (abs(R - G) < 15) & (abs(G - B) < 15) & (R < 140)
    blue = (B > 140) & (R < 130) & (G > 80) & (G < 190) & (B - R > 60)
    orange = (R > 200) & (G > 60) & (G < 150) & (B < 90)
    boxes = frames(gray, sc)
    fin, conc = {}, {}
    for name, box in boxes.items():
        x0, x1, y0, y1 = box
        xt, yt = calibration(gray, box, sc)
        assert len(xt) == len(X_VALUES[name]) and len(yt) == len(Y_VALUES[name]), (name, xt, yt)
        ax = np.polyfit(xt, X_VALUES[name], 1)
        ay = np.polyfit(yt, Y_VALUES[name], 1)
        m = blue[y0 + 3:y1 - 2, x0 + 3:x1 - 2]
        if name.startswith("L"):
            lab, n = ndimage.label(m)
            cents = np.array(ndimage.center_of_mass(m, lab, range(1, n + 1)))[:, ::-1] + [x0 + 3, y0 + 3]
            areas = ndimage.sum(m, lab, range(1, n + 1))
            keep = areas >= 5 * sc ** 2
            cents, areas = cents[keep], areas[keep]
            ys, xs = np.nonzero(m)
            pts = np.c_[xs + x0 + 3, ys + y0 + 3].astype(float)
            labs = fcluster(linkage(cents, "single"), t=16 * sc, criterion="distance")
            dot_area = 320 * sc ** 2
            dots = []
            for c in np.unique(labs):
                idx = labs == c
                A = areas[idx].sum()
                if A < 0.375 * dot_area:
                    continue
                cen = (cents[idx] * areas[idx, None]).sum(0) / A
                nd = max(1, int(round(A / dot_area)))
                if nd == 1:
                    dots.append(cen)
                else:
                    sel = np.linalg.norm(pts - cen, axis=1) < 30 * sc
                    c2, _ = kmeans2(pts[sel], nd, seed=1, minit="++")
                    dots += list(c2)
            dots = np.array(sorted(dots, key=lambda d: d[0]))
            fin[name] = [[float(np.polyval(ax, d[0])), float(np.polyval(ay, d[1]))]
                         for d in dots]
        else:
            t, cm = [], []
            for x in range(x0 + 4, x1 - 3):
                ys = np.nonzero(m[:, x - x0 - 3])[0]
                if ys.size:
                    t.append(float(np.polyval(ax, x)))
                    cm.append(float(np.polyval(ay, ys + y0 + 3).mean()))
            conc[name] = {"t": t, "cmean": cm}
    return fin, conc


if __name__ == "__main__":
    pdf, dpi, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    fin, conc = digitize(pdf, dpi)
    json.dump(fin, open(f"{prefix}_finnegan.json", "w"))
    json.dump(conc, open(f"{prefix}_conc.json", "w"))
    print({k: len(v) for k, v in fin.items()}, {k: len(v["t"]) for k, v in conc.items()})
