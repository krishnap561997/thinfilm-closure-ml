#!/usr/bin/env python3
"""ΔP_x extraction diagnostics on 2D DNS snapshots (ΔP_x plan, Step 1).

For each snapshot (current state t and previous state t - dt) this computes

  I_p = ∫_0^h ∂_x p dz   three ways
    a  current pipeline: 4th-order ∂_x p on every grid row, trapezoid to y = h
       (integrate_to_height), i.e. rows cut by the interface are included;
    b  Leibniz form on the liquid pressure only:
         I_p = ∂_x ∫_0^h p_l dz - p_l(h) h_x,
       where p_l is the cell pressure in fully liquid cells (y <= h - 1.5Δ) and
       p_l(h) is a linear least-squares extrapolation of the top 4 such cells;
    c  as a, but ∂_x p is integrated only up to the last cell whose whole
       5-point x-stencil is liquid, plus a linear-extrapolated top segment;

and the Lavalle P_x two ways
    raw  as in extract_closure_terms.closure_set: h h_xxx/We - cosβ/Fr² h h_x + q_xx/Re
         with RMSE-selected Savitzky-Golay derivatives (the selected (m, p) is logged);
    s<σ> Gaussian filter of width σ (h_N), analytic derivatives of the filtered h,
         exact curvature: h κ_x/We - cosβ/Fr² h h_x + q_xx/Re, κ = h_xx/(1+h_x²)^{3/2};
         the same filter is applied to I_p.

ΔP_x = -I_p - P_x,Lav for each (method, filter). Per snapshot it records:
  - spectral fractions of variance at wavelength < 1 h_N and < 0.25 h_N for each term;
  - std ratios std(ΔP_x)/std(I_p) (how much of I_p cancels against P_Lav);
  - agreement of methods a/b/c (correlation, relative L2 difference);
  - DNS-grid momentum-budget residual ||∂_t q - RHS|| / ||∂_t q|| with each I_p
    (RHS = -M_x - I_p - T_w + q_xx/Re + Λh/Re, averaged over both states), and the
    mass residual ||∂_t h + q_x|| / ||∂_t h||;
  - for t on the ClosureData grid, max |ΔP_x(a, raw) - stored delta_Px_ML| (sanity).

Analysis window: x0 < x < 0.95 L.  Output (in --out):
  <label>_metrics.csv  long format (label, t, metric, variant, value)
  <label>_fields.h5    x-by-time fields (h, q, I_p variants, ΔP_x variants, κ, κ_x)
                       for crest-aligned mesh comparison
Print: a short per-run summary (medians over snapshots).

    python dpx_diagnostics.py run --snapdir DIR --times 100:1900:200 --label CASE --out OUT \
        [--closure FILE] [--sigmas 0.25,0.5,1.0]
    python dpx_diagnostics.py summary OUT        # one table over all *_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_closure_terms import (  # noqa: E402
    best_savgol_filter, build_cartesian_cell_grid, compute_cell_centers, compute_px,
    compute_wall_gradient, depth_integral, first_derivative_x, get_param,
    integrate_to_height, scalar_to_matrix, scalar_value, sg_derivatives, vector_x_to_matrix,
    PX_GAP as GAP, PX_NFIT as NFIT, extrapolate_top, gauss_deriv, ip_liquid_leibniz,
    lavalle_px_smooth, liquid_top_index, trapz_to,
)

METHODS = ("a", "b", "c")
def ip_methods(x, ym, f, p, h, dy):
    """Return dict of I_p for methods a, b, c and p_l(h)."""
    ny, nx = p.shape
    px = compute_px(x, p)
    out = {"a": integrate_to_height(ym, px, h, 0.0)}

    # b: Leibniz on the liquid pressure (the v2 definition in extract_closure_terms)
    out["b"], pl_h = ip_liquid_leibniz(x, ym, p, h)

    # c: ∂_x p only where the whole 5-point stencil row is liquid
    hmin5 = h.copy()
    for s in (-2, -1, 1, 2):
        hmin5 = np.minimum(hmin5, np.roll(h, s))
    jc = liquid_top_index(hmin5 - GAP * dy, dy, ny)
    pxh = extrapolate_top(px, jc, h, dy)
    pxwall = 1.5 * px[0] - 0.5 * px[1]
    out["c"] = trapz_to(px, jc, h, pxwall, pxh, dy)

    # fraction of columns where the fit window reached below the film (thin substrate)
    out["_thin_frac"] = float(np.mean((h - GAP * dy) / dy - 0.5 < NFIT - 1))
    out["_pl_h"] = pl_h
    return out


# ---------------------------------------------------------------- one state
def state_fields(x, ym, s, suffix, idx, shape, dy):
    sf = "_minus" if suffix else ""
    f = scalar_to_matrix(s[f"Cells/f{sf}"][...], idx, shape)
    u = vector_x_to_matrix(s[f"Cells/u{sf}.x"][...], idx, shape)
    p = scalar_to_matrix(s[f"Cells/p{sf}"][...], idx, shape)
    xm = np.broadcast_to(x, shape)
    _, h = depth_integral(xm, ym, f, 0.0)
    _, q = depth_integral(xm, ym, u * f, 0.0)
    _, M = depth_integral(xm, ym, u * u * f, 0.0)
    _, dudy = compute_wall_gradient(xm, ym, u)
    ip = ip_methods(x, ym, f, p, h, dy)
    return {"h": h, "q": q, "M": M, "dudy": dudy, "ip": ip}


gauss = gauss_deriv


def lavalle_raw(x, h, q, P):
    hx, hxx, hxxx = sg_derivatives(x, h, 3, min_poly=3)
    qx, qxx = sg_derivatives(x, q, 2, min_poly=2)
    We, Fr2, cb = P["We"], P["Fr2"], np.cos(P["beta"])
    terms = {"cap": h * hxxx / We, "hyd": cb / Fr2 * h * hx, "visc": qxx / P["Re"]}
    return terms["cap"] - terms["hyd"] + terms["visc"], terms, {"hx": hx, "qx": qx, "qxx": qxx}


def lavalle_smooth(h, q, P, sig, dx):
    return lavalle_px_smooth(h, q, P["Re"], P["We"] / P["Re"], P["beta"], P["Lambda"], sig, dx)


# ---------------------------------------------------------------- metrics
def spectral_fracs(v, dx):
    v = v - v.mean()
    w = np.hanning(v.size)
    P = np.abs(np.fft.rfft(v * w)) ** 2
    k = 2 * np.pi * np.fft.rfftfreq(v.size, dx)
    tot = P[1:].sum()
    if tot <= 0:
        return np.nan, np.nan
    return P[k > 2 * np.pi].sum() / tot, P[k > 2 * np.pi / 0.25].sum() / tot


def rel_l2(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def parse_times(spec):
    if ":" in spec:
        a, b, c = map(float, spec.split(":"))
        return list(np.arange(a, b + 0.5 * c, c))
    return [float(v) for v in spec.split(",")]


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    snapdir = Path(args.snapdir)
    paths = []
    for t in parse_times(args.times):
        pth = snapdir / f"snapshot_{int(round(t)):06d}.h5"
        if pth.is_file():
            paths.append(pth)
        else:
            print(f"skip (missing): {pth.name}")
    if not paths:
        raise SystemExit("no snapshots found")

    with h5py.File(paths[0], "r") as s0:
        cc = compute_cell_centers(s0["Geometry/Points"][...], s0["Topology"][...])
        xm, ym, idx = build_cartesian_cell_grid(cc)
        Re = get_param(s0, "Re"); Ca = get_param(s0, "Ca"); beta = get_param(s0, "angle")
        Lam = get_param(s0, "Lambda", 3.0)
    P = {"Re": Re, "We": Re * Ca, "Fr2": Re * np.sin(beta) / Lam, "beta": beta, "Lambda": Lam}
    x = xm[0, :].copy(); dx = x[1] - x[0]; dy = ym[1, 0] - ym[0, 0]; L = x[-1] + 0.5 * dx
    mask = (x > args.x0) & (x < 0.95 * L)
    print(f"{args.label}: Re={Re:g} Ca={Ca:g} beta={np.degrees(beta):.1f} We={P['We']:g} "
          f"nx={x.size} ny={ym.shape[0]} dx={dx:.4g} dy={dy:.4g}  {len(paths)} snapshots")

    closure = None
    if args.closure and Path(args.closure).is_file():
        closure = h5py.File(args.closure, "r")
        ct = closure["t"][:].ravel()

    rows = []
    fields = defaultdict(list)
    times = []

    def rec(t, metric, variant, value):
        rows.append((args.label, t, metric, variant, float(value)))

    for pth in paths:
        with h5py.File(pth, "r") as s:
            t = scalar_value(s["Parameters/time"]); dtm = t - scalar_value(s["Parameters/time_minus"])
            cur = state_fields(x, ym, s, "", idx, xm.shape, dy)
            prv = state_fields(x, ym, s, "m", idx, xm.shape, dy)
        times.append(t)
        h, q = cur["h"], cur["q"]

        # SG selection actually used by the pipeline
        m3, p3, _, _ = best_savgol_filter(x, h, range(11, 42, 2), [3], metric="rmse", power=4)
        mq, pq, _, _ = best_savgol_filter(x, q, range(11, 42, 2), [2, 3], metric="rmse", power=4)
        rec(t, "sg_window_h", "m", m3); rec(t, "sg_window_q", "m", mq); rec(t, "sg_order_q", "p", pq)
        rec(t, "thin_frac", "b", cur["ip"]["_thin_frac"])

        # raw (pipeline) Lavalle and ΔP_x per method
        Plav, terms, der = lavalle_raw(x, h, q, P)
        dP_raw = {m: -cur["ip"][m] - Plav for m in METHODS}
        for name, v in [("-Ip_a", -cur["ip"]["a"]), ("-Ip_b", -cur["ip"]["b"]), ("-Ip_c", -cur["ip"]["c"]),
                        ("cap", terms["cap"]), ("hyd", terms["hyd"]), ("visc", terms["visc"])] + \
                       [(f"dPx_{m}_raw", dP_raw[m]) for m in METHODS]:
            f1, f4 = spectral_fracs(v[mask], dx)
            rec(t, "hk_frac_lt1", name, f1); rec(t, "hk_frac_lt0.25", name, f4)
            rec(t, "std", name, np.std(v[mask]))
        for m in METHODS:
            rec(t, "cancel_ratio", f"{m}_raw", np.std(dP_raw[m][mask]) / np.std(cur["ip"][m][mask]))
        for m1, m2 in (("a", "b"), ("a", "c"), ("b", "c")):
            A, B = cur["ip"][m1][mask], cur["ip"][m2][mask]
            rec(t, "corr_Ip", f"{m1}{m2}_raw", np.corrcoef(A, B)[0, 1])
            rec(t, "rel_l2_Ip", f"{m1}{m2}_raw", rel_l2(A, B))

        if closure is not None:
            j = np.where(np.isclose(ct, t, atol=1e-6))[0]
            if j.size:
                stored = closure["delta_Px_ML"][:, j[0]]
                rec(t, "max_abs_vs_stored", "a_raw", np.nanmax(np.abs(stored[mask] - dP_raw["a"][mask])))

        # budget residual on the DNS grid, per I_p method
        dtq = (q - prv["q"]) / dtm
        dth = (h - prv["h"]) / dtm
        def rhs_common(st):
            Mx = first_derivative_x(x, st["M"])
            qxx = lavalle_raw(x, st["h"], st["q"], P)[2]["qxx"]
            return -Mx - st["dudy"] / Re + qxx / Re + Lam * st["h"] / Re
        common = 0.5 * (rhs_common(cur) + rhs_common(prv))
        qx_avg = 0.5 * (first_derivative_x(x, q) + first_derivative_x(x, prv["q"]))
        for sig in [0.0] + sigmas:
            G = (lambda v: v) if sig == 0 else (lambda v, s_=sig: gauss(v, s_, dx))
            tag = "raw" if sig == 0 else f"s{sig:g}"
            rec(t, "mass_resid", tag,
                np.linalg.norm(G(dth + qx_avg)[mask]) / max(np.linalg.norm(G(dth)[mask]), 1e-300))
            for m in METHODS:
                Ip = 0.5 * (cur["ip"][m] + prv["ip"][m])
                R = G(dtq - (common - Ip))
                rec(t, "mom_resid", f"{m}_{tag}", np.linalg.norm(R[mask]) / np.linalg.norm(G(dtq)[mask]))

        # filtered targets with the exact-curvature Lavalle term
        fields["h"].append(h); fields["q"].append(q)
        for m in METHODS:
            fields[f"Ip_{m}"].append(cur["ip"][m]); fields[f"dPx_{m}_raw"].append(dP_raw[m])
        fields["p_l_h"].append(cur["ip"]["_pl_h"])
        for sig in sigmas:
            tag = f"s{sig:g}"
            Pl_s, kap, kap_x = lavalle_smooth(h, q, P, sig, dx)
            fields[f"kappa_{tag}"].append(kap); fields[f"kappa_x_{tag}"].append(kap_x)
            dP_s = {}
            for m in METHODS:
                dP_s[m] = -gauss(cur["ip"][m], sig, dx) - Pl_s
                fields[f"dPx_{m}_{tag}"].append(dP_s[m])
                f1, f4 = spectral_fracs(dP_s[m][mask], dx)
                rec(t, "hk_frac_lt1", f"dPx_{m}_{tag}", f1)
                rec(t, "std", f"dPx_{m}_{tag}", np.std(dP_s[m][mask]))
                rec(t, "cancel_ratio", f"{m}_{tag}",
                    np.std(dP_s[m][mask]) / np.std(gauss(cur["ip"][m], sig, dx)[mask]))
            for m1, m2 in (("a", "b"), ("a", "c"), ("b", "c")):
                rec(t, "rel_l2_dPx", f"{m1}{m2}_{tag}", rel_l2(dP_s[m1][mask], dP_s[m2][mask]))
        print(f"  t={t:g} done (SG m_h={m3}, m_q={mq})", flush=True)

    if closure is not None:
        closure.close()

    with open(out / f"{args.label}_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["label", "t", "metric", "variant", "value"]); w.writerows(rows)
    with h5py.File(out / f"{args.label}_fields.h5", "w") as fo:
        fo["x"] = x; fo["t"] = np.array(times)
        for k, v in fields.items():
            fo.create_dataset(k, data=np.stack(v, axis=1).astype(np.float32), compression="gzip")
        for k, v in P.items():
            fo.attrs[k] = v
        fo.attrs["sigmas"] = sigmas; fo.attrs["snapdir"] = str(snapdir)
    summarize([out / f"{args.label}_metrics.csv"])


# ---------------------------------------------------------------- summary
KEY = [
    ("sg_window_h", ["m"]),
    ("thin_frac", ["b"]),
    ("max_abs_vs_stored", ["a_raw"]),
    ("hk_frac_lt1", ["-Ip_a", "-Ip_b", "-Ip_c", "cap", "dPx_a_raw", "dPx_b_raw", "dPx_c_raw"]),
    ("hk_frac_lt0.25", ["-Ip_a", "-Ip_b", "cap", "dPx_a_raw", "dPx_b_raw"]),
    ("cancel_ratio", ["a_raw", "b_raw", "c_raw", "a_s0.5", "b_s0.5", "c_s0.5"]),
    ("rel_l2_Ip", ["ab_raw", "ac_raw", "bc_raw"]),
    ("rel_l2_dPx", ["ab_s0.25", "ab_s0.5", "ab_s1", "bc_s0.5"]),
    ("mass_resid", ["raw", "s0.5", "s1"]),
    ("mom_resid", ["a_raw", "b_raw", "c_raw", "a_s0.5", "b_s0.5", "c_s0.5", "a_s1", "b_s1", "c_s1"]),
]


def summarize(files):
    data = defaultdict(lambda: defaultdict(list))
    labels = []
    for fpath in files:
        with open(fpath) as fh:
            for r in csv.DictReader(fh):
                if r["label"] not in labels:
                    labels.append(r["label"])
                data[r["label"]][(r["metric"], r["variant"])].append(float(r["value"]))
    print("\nmedian over snapshots")
    print(f"{'metric':<16}{'variant':<12}" + "".join(f"{l[:20]:>22}" for l in labels))
    for metric, variants in KEY:
        for v in variants:
            vals = []
            for l in labels:
                a = data[l].get((metric, v))
                vals.append(f"{np.nanmedian(a):22.4g}" if a else f"{'-':>22}")
            print(f"{metric:<16}{v:<12}" + "".join(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--snapdir", required=True)
    r.add_argument("--times", required=True, help="a:b:step or comma list")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--closure", default=None, help="ClosureData file for the stored-field sanity check")
    r.add_argument("--sigmas", default="0.25,0.5,1.0")
    r.add_argument("--x0", type=float, default=100.0)
    s = sub.add_parser("summary")
    s.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        summarize(sorted(Path(a.out).glob("*_metrics.csv")))


if __name__ == "__main__":
    main()
