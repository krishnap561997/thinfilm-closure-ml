"""Short summary of linear_targets.csv (B.7, B.8)."""
import csv, os, sys
import numpy as np
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["THINFILM_STORE"], "analysis/orr_sommerfeld/linear_targets.csv")
R = [r for r in csv.DictReader(open(fn)) if not r.get("error")]
f = lambda r, k: float(r[k])
d = np.array([f(r, "delta_lin") for r in R])
bins = [(0, 5), (5, 15), (15, 30), (30, 70)]
def tab(title, keys):
    print(f"\n{title}  (median over cases in each delta_lin bin)")
    print(f"{'quantity':28s} " + " ".join(f"{f'd {a}-{b}':>11s}" for a, b in bins))
    for k in keys:
        v = np.array([f(r, k) for r in R])
        print(f"{k:28s} " + " ".join(f"{np.median(v[(d >= a) & (d < b)]):+11.4f}" for a, b in bins))
print(f"{len(R)} cases; n per bin: " + ", ".join(str(((d >= a) & (d < b)).sum()) for a, b in bins))
print(f"B.7 checks: max |momentum residual| {max(f(r, 'b7_kmax_resid') for r in R):.1e}, max |q' - c| {max(f(r, 'b7_kmax_qcheck') for r in R):.1e}")
tab("B.7 closure corrections per unit eta at the OS k_max (re = in phase with h, im = quadrature)",
    ["b7_kmax_k", "b7_kmax_dM_re", "b7_kmax_dM_im", "b7_kmax_dT_re", "b7_kmax_dT_im", "b7_kmax_dP_re", "b7_kmax_dP_im", "b7_kmax_dD_re"])
S_re = np.array([f(r, "b7_kmax_dM_re") + f(r, "b7_kmax_dT_re") for r in R]); S_im = np.array([f(r, "b7_kmax_dM_im") + f(r, "b7_kmax_dT_im") for r in R])
M_im = np.array([abs(f(r, "b7_kmax_dM_im")) for r in R])
print(f"{'S = dM+dT re':28s} " + " ".join(f"{np.median(S_re[(d >= a) & (d < b)]):+11.4f}" for a, b in bins))
print(f"{'S = dM+dT im':28s} " + " ".join(f"{np.median(S_im[(d >= a) & (d < b)]):+11.4f}" for a, b in bins))
print(f"{'|S| / |dM_im|':28s} " + " ".join(f"{np.median((np.hypot(S_re, S_im) / M_im)[(d >= a) & (d < b)]):11.3f}" for a, b in bins))
tab("B.8 max errors over the band: growth (rel. to max OS growth) and k_r (relative)",
    ["lav_err_ki", "abc_err_ki", "abcd_err_ki", "lav_err_kr", "abc_err_kr", "abcd_err_kr",
     "band_lav_err_ki", "band_abc_err_ki", "band_abcd_err_ki", "band_lav_err_kr", "band_abc_err_kr", "band_abcd_err_kr"])
tab("B.8 coefficients (D_aug = D_OS imposed; c follows from a, b)",
    ["abc_a", "abc_b", "abc_c", "abcd_a", "abcd_b", "abcd_c", "abcd_d", "band_abc_a", "band_abc_b", "band_abc_c",
     "band_abcd_a", "band_abcd_b", "band_abcd_c", "band_abcd_d"])
for k in ("band_abc_err_ki", "band_abcd_err_ki", "abcd_err_ki"):
    v = np.array([f(r, k) for r in R]); print(f"{k}: cases with error < 0.10: {(v < 0.1).sum()}/{len(v)}, max {v.max():.3f}")
