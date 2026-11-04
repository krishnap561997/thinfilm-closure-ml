#!/usr/bin/env python3
"""Mesh sensitivity (level 14 vs 15): wave tracking, bulk stats, crest-aligned closure comparison.

Input: two closure files from extract_closure_terms.py on the same snapshot times.
Waves are found as crests of h(x,t) (smoothed), matched across runs at the same t by nearest crest
(offset << wavelength/2, same constant-frequency forcing), then every field is cut in
xi = x - x_crest and compared. Prints a short summary; writes csv/npz/figures to --out.

    python scripts/mesh_study/compare_mesh.py A.h5 B.h5 --labels L14 L15 --out DIR
"""
import argparse, csv, os
import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

FIELDS = ["h", "q", "delta_Mx_ML", "delta_Tw_ML", "delta_Px_ML", "delta_Dx_ML"]
CLOS = FIELDS[2:]


def load(path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][...] for k in FIELDS + ["t", "x"]}
    d["S_ML"] = -d["delta_Mx_ML"] + d["delta_Px_ML"] - d["delta_Tw_ML"] + d["delta_Dx_ML"]
    return d


def crests(x, h, xmin=60.0, xmax=760.0, sig=0.3, prom=0.5, sep=20.0):
    dx = x[1] - x[0]
    hs = gaussian_filter1d(h, sig / dx)
    p, _ = find_peaks(hs, prominence=prom, distance=int(sep / dx))
    out = []
    for i in p:  # parabolic refinement
        if 0 < i < len(x) - 1:
            a, b, c = hs[i - 1], hs[i], hs[i + 1]
            den = a - 2 * b + c
            xc = x[i] + (0.5 * dx * (a - c) / den if den != 0 else 0.0)
        else:
            xc = x[i]
        if xmin < xc < xmax:
            out.append((xc, hs[i]))
    return out


def track(d):
    """Per snapshot crest lists and per-crest celerity from the next snapshot."""
    x, t = d["x"], d["t"]
    cl = [crests(x, d["h"][:, i]) for i in range(len(t))]
    c0 = 1.8
    for _ in range(3):
        ds = []
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            nxt = np.array([c[0] for c in cl[i + 1]])
            for xc, _ in cl[i]:
                if len(nxt) == 0:
                    continue
                j = np.argmin(np.abs(nxt - (xc + c0 * dt)))
                if abs(nxt[j] - (xc + c0 * dt)) < 0.3 * c0 * dt:
                    ds.append((i, xc, (nxt[j] - xc) / dt, cl[i][0][1]))
        c0 = float(np.median([r[2] for r in ds]))
    return cl, np.array(ds)


def window(x, F, xc, half, dxi):
    xi = np.arange(-half, half + 1e-9, dxi)
    return xi, np.interp(xc + xi, x, F)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["L14", "L15"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=float, default=40.0, help="xi half-window")
    ap.add_argument("--smooth", type=float, default=1.0, help="common Gaussian width (x units) for the smoothed closure comparison")
    ap.add_argument("--match-tol", type=float, default=12.0, help="max crest x offset for matching waves")
    ap.add_argument("--x-max-match", type=float, default=700.0, help="ignore crests beyond this x (outlet influence)")
    ap.add_argument("--xlim-crest", type=float, nargs=2, default=[100.0, 700.0])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = [load(a.a), load(a.b)]; lab = a.labels
    t = D[0]["t"]; assert np.allclose(t, D[1]["t"])
    tr = [track(d) for d in D]

    print("== bulk (x in %g..%g), all snapshots" % tuple(a.xlim_crest))
    print("run   ncrest/snap  c_med   c_iqr          crest_h med   lambda_med  h_mean  h_rms'  h_max  h_min")
    bulk = {}
    for k, d in enumerate(D):
        cl, ds = tr[k]
        m = (d["x"] > a.xlim_crest[0]) & (d["x"] < a.xlim_crest[1])
        sel = ds[(ds[:, 1] > a.xlim_crest[0]) & (ds[:, 1] < a.xlim_crest[1])]
        lam = np.concatenate([np.diff([c[0] for c in c_]) for c_ in cl if len(c_) > 1])
        h = d["h"][m]
        bulk[lab[k]] = dict(c=np.median(sel[:, 2]), ch=np.median(sel[:, 3]), lam=np.median(lam))
        print(f"{lab[k]:5s} {np.mean([len(c) for c in cl]):6.1f}      {np.median(sel[:,2]):.4f}  [{np.percentile(sel[:,2],25):.4f},{np.percentile(sel[:,2],75):.4f}]"
              f"  {np.median(sel[:,3]):.3f}        {np.median(lam):6.2f}     {h.mean():.4f}  {h.std():.4f}  {h.max():.3f}  {h.min():.3f}")
    print("crest count per snapshot:", {lab[k]: [len(c) for c in tr[k][0]] for k in range(2)})

    # celerity vs x
    cx = {}
    for k in range(2):
        ds = tr[k][1]
        edges = np.arange(60, 761, 140)
        cx[lab[k]] = [np.median(ds[(ds[:, 1] >= e) & (ds[:, 1] < e + 140), 2]) if np.any((ds[:, 1] >= e) & (ds[:, 1] < e + 140)) else np.nan for e in edges[:-1]]
    print("celerity in x-bins of 140 starting at 60:", {l: np.round(v, 3).tolist() for l, v in cx.items()})

    # match by index from inlet
    dxi = 0.05
    xi = np.arange(-a.half, a.half + 1e-9, dxi)
    store = {s: {f: [[], []] for f in FIELDS + ["S_ML"]} for s in ("raw", "smooth")}
    pairs = []
    for i in range(len(t)):
        c = [[w for w in tr[k][0][i] if w[0] < a.x_max_match and w[0] - a.half > 5] for k in range(2)]
        pj = []
        for ja, (xa_, _) in enumerate(c[0]):  # nearest crest in B within MATCH_TOL (offset is far below lambda/2)
            if len(c[1]) == 0:
                break
            jb = int(np.argmin([abs(w[0] - xa_) for w in c[1]]))
            if abs(c[1][jb][0] - xa_) < a.match_tol:
                pj.append((ja, jb))
        for ja, jb in pj:
            pairs.append((i, ja, c[0][ja][0], c[1][jb][0], c[0][ja][1], c[1][jb][1]))
            cj = (c[0][ja][0], c[1][jb][0])
            for k in range(2):
                for f in FIELDS + ["S_ML"]:
                    F = D[k][f][:, i]
                    store["raw"][f][k].append(np.interp(cj[k] + xi, D[k]["x"], F))
                    Fs = gaussian_filter1d(F, a.smooth / (D[k]["x"][1] - D[k]["x"][0]))
                    store["smooth"][f][k].append(np.interp(cj[k] + xi, D[k]["x"], Fs))
    P = np.array(pairs)
    print("\n== matched waves:", len(P), "(nearest crest, same t)")
    dxc = P[:, 3] - P[:, 2]
    print(f"crest x difference ({lab[1]}-{lab[0]}): median {np.median(dxc):.2f}, range [{dxc.min():.2f},{dxc.max():.2f}]; amplitude diff median {np.median(P[:,5]-P[:,4]):.4f}")
    # phase drift: mean dx vs t
    tt = t[P[:, 0].astype(int)]
    drift = np.polyfit(tt, dxc, 1)
    print(f"d(dx_crest)/dt = {drift[0]:.4f}  (celerity difference {lab[1]}-{lab[0]} ~ this)")
    for i in range(len(t)):
        m = P[:, 0] == i
        if m.any():
            pass
    with open(os.path.join(a.out, "matched_waves.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["snap", "wave_idx", "x_" + lab[0], "x_" + lab[1], "hc_" + lab[0], "hc_" + lab[1]]); w.writerows(P)

    print("\n== crest-aligned closure comparison (xi in +-%g)" % a.half)
    print("field        mode    pk|%s| med  pk|%s| med  pk ratio  xi_pk diff(med)  relL2 diff med [iqr]   corr med" % tuple(lab))
    res = []
    for mode in ("raw", "smooth"):
        for f in FIELDS[2:] + ["S_ML"]:
            A, B = np.array(store[mode][f][0]), np.array(store[mode][f][1])
            pa, pb = np.abs(A).max(1), np.abs(B).max(1)
            xa, xb = xi[np.abs(A).argmax(1)], xi[np.abs(B).argmax(1)]
            rl = np.linalg.norm(B - A, axis=1) / np.linalg.norm(A, axis=1)
            cc = [np.corrcoef(A[n], B[n])[0, 1] for n in range(len(A))]
            res.append((f, mode, np.median(pa), np.median(pb), np.median(pb / pa), np.median(xb - xa), np.median(rl), np.percentile(rl, 25), np.percentile(rl, 75), np.median(cc)))
            print(f"{f:13s}{mode:7s} {np.median(pa):10.4g} {np.median(pb):10.4g} {np.median(pb/pa):8.3f} {np.median(xb-xa):10.3f}      {np.median(rl):.3f} [{np.percentile(rl,25):.3f},{np.percentile(rl,75):.3f}]   {np.median(cc):.3f}")
    for f in ("h", "q"):
        A, B = np.array(store["raw"][f][0]), np.array(store["raw"][f][1])
        rl = np.linalg.norm(B - A, axis=1) / np.linalg.norm(A, axis=1)
        print(f"{f:13s}raw     relL2 diff med {np.median(rl):.4f}  max {rl.max():.4f}")
    with open(os.path.join(a.out, "aligned_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["field", "mode", "pk_" + lab[0], "pk_" + lab[1], "pk_ratio_med", "xi_pk_diff_med", "relL2_med", "relL2_q25", "relL2_q75", "corr_med"]); w.writerows(res)
    np.savez_compressed(os.path.join(a.out, "aligned_profiles.npz"), xi=xi, **{f"{m}_{f}_{lab[k]}": np.array(store[m][f][k]) for m in store for f in store[m] for k in range(2)})

    # figures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for axx, f in zip(ax.ravel(), FIELDS + ["S_ML"]):
        for k, col in zip(range(2), ("C0", "C3")):
            M = np.array(store["raw"][f][k]); mu, sd = M.mean(0), M.std(0)
            axx.plot(xi, mu, col, lw=1, label=lab[k]); axx.fill_between(xi, mu - sd, mu + sd, color=col, alpha=0.2)
        axx.set_title(f + " (crest-aligned, raw)"); axx.axvline(0, c="k", lw=0.5)
    ax[0, 0].legend(); [axx.set_xlabel(r"$\xi=x-x_{crest}$") for axx in ax[-1]]
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "aligned_ensemble_raw.png"), dpi=110)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axx, f in zip(ax.ravel(), CLOS):
        for k, col in zip(range(2), ("C0", "C3")):
            M = np.array(store["smooth"][f][k]); mu, sd = M.mean(0), M.std(0)
            axx.plot(xi, mu, col, label=lab[k]); axx.fill_between(xi, mu - sd, mu + sd, color=col, alpha=0.2)
        axx.set_title(f + " (common Gaussian %.1f)" % a.smooth); axx.axvline(0, c="k", lw=0.5)
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "aligned_ensemble_smooth.png"), dpi=110)
    # single wave overlay (middle matched wave, last snapshot with >=3 waves)
    i0 = int(P[np.argmin(np.abs(P[:, 2] - 400)), 0]); m = np.where(P[:, 0] == i0)[0]
    j = m[len(m) // 2]
    fig, ax = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for axx, f in zip(ax.ravel(), FIELDS + ["S_ML"]):
        for k, col in zip(range(2), ("C0", "C3")):
            axx.plot(xi, store["raw"][f][k][j], col, lw=0.8, label=lab[k])
        axx.set_title(f"{f}: t={t[i0]:.0f}, x_c={P[j,2]:.0f}/{P[j,3]:.0f}")
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "single_wave_overlay.png"), dpi=110)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tt, dxc, "o"); ax.plot(tt, np.polyval(drift, tt), "k-"); ax.set_xlabel("t"); ax.set_ylabel(f"x_crest {lab[1]} - {lab[0]}")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "crest_offset.png"), dpi=110)
    print("figures + csv in", a.out)


if __name__ == "__main__":
    main()
