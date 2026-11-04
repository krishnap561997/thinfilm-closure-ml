#!/usr/bin/env python3
"""Track the initial-disturbance wave in a DNS run and report when it leaves the domain.

For each snapshot t0..t1 (step every) compute h(x) = int f dy per column and print
max h, its x location, and the max h over the outlet region x > x_out. The leading
transient shows up as a large max h that travels to x = LX; t_exit is the first time
after which max h stays below --h-big (default 1.5) over the whole domain.

    python scripts/mesh_study/find_transient_exit.py RUN_DIR --t1 650 [--csv out.csv]
"""
import argparse, csv, glob, os
import h5py
import numpy as np


def column_h(path, idx=None, shape=None):
    with h5py.File(path, "r") as s:
        t = float(np.asarray(s["Parameters/time"][...]).squeeze())
        if idx is None:
            cc = np.asarray(s["Geometry/Points"][...])[np.asarray(s["Topology"][...], dtype=np.int64)].mean(axis=1)
            x, y = np.round(cc[:, 0], 12), np.round(cc[:, 1], 12)
            ux, uy = np.unique(x), np.unique(y)
            idx = np.lexsort((x, y)); shape = (len(uy), len(ux))
            xs, ys = ux, uy
        else:
            xs = ys = None
        f = np.asarray(s["Cells/f"][...]).squeeze()[idx].reshape(shape)
    return t, f, idx, shape, xs, ys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--t1", type=float, default=650)
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--h-big", type=float, default=1.5)
    ap.add_argument("--x-out", type=float, default=700.0)
    ap.add_argument("--csv")
    a = ap.parse_args()

    rows, idx, shape, dy, xs = [], None, None, None, None
    t = a.t0
    while t <= a.t1 + 1e-9:
        p = os.path.join(a.run_dir, "output", f"snapshot_{t:06.0f}.h5")
        if os.path.exists(p):
            tt, f, idx, shape, xs_, ys_ = column_h(p, idx, shape)
            if xs_ is not None:
                xs = xs_; dy = np.diff(np.r_[0.0, 0.5 * (ys_[:-1] + ys_[1:]), ys_[-1] + 0.5 * (ys_[-1] - ys_[-2])])
            h = (f * dy[:, None]).sum(axis=0)
            i = int(np.argmax(h)); o = xs > a.x_out
            rows.append((tt, h.max(), xs[i], h[o].max(), h.mean()))
        t += a.every
    if not rows:
        raise SystemExit("no snapshots found")
    big = [r for r in rows if r[1] > a.h_big]
    print(f"{a.run_dir}\n{len(rows)} snapshots, nx={len(xs)}, ny={shape[0]}")
    print("   t   hmax   x(hmax)  hmax(x>%g)   hmean" % a.x_out)
    for r in rows[::max(1, len(rows) // 40)]:
        print(f"{r[0]:6.0f} {r[1]:6.3f} {r[2]:8.1f} {r[3]:9.3f} {r[4]:8.4f}")
    print("last t with hmax>%.2f: %s" % (a.h_big, big[-1][0] if big else "none"))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "hmax", "x_hmax", "hmax_outlet", "hmean"]); w.writerows(rows)


if __name__ == "__main__":
    main()
