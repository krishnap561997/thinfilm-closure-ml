"""Short summary of case_table.csv (B.4): per-case table for cases with DNS data, plus aggregates."""
import csv, os, sys, math
import numpy as np
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["THINFILM_STORE"], "analysis/orr_sommerfeld/case_table.csv")
R = list(csv.DictReader(open(fn)))
f = lambda r, k: float(r[k]) if r.get(k) not in ("", None) else math.nan
def agg(name, v):
    v = np.array(v, float); v = v[np.isfinite(v)]
    print(f"  {name:38s} min {v.min():8.3g}  median {np.median(v):8.3g}  max {v.max():8.3g}  (n={len(v)})")
has = [r for r in R if r["status"] in ("complete", "partial")]
print(f"{len(R)} cases, {len(has)} with DNS data (complete/partial), errors: {sum(1 for r in R if r['error'])}")
print("\nCases with DNS data:")
print(f"{'case':26s} {'kmax':>6s} {'kmaxL':>6s} {'grow':>7s} {'growL':>7s} {'k_c':>6s} {'w_max':>6s} {'-ki':>7s} {'-kiL':>7s} {'f_c':>6s} {'f_mx':>6s} {'28/fc':>6s} {'d_lin':>6s} {'d_Lav':>6s}")
for r in sorted(has, key=lambda r: (f(r,'Re'), f(r,'Ca'), f(r,'beta_deg'))):
    print(f"{r['case']:26s} {f(r,'os_k_max'):6.3f} {f(r,'lav_k_max'):6.3f} {f(r,'os_growth_max'):7.4f} {f(r,'lav_growth_max'):7.4f} "
          f"{f(r,'os_k_c'):6.3f} {f(r,'os_omega_max'):6.3f} {f(r,'os_mki_max'):7.4f} {f(r,'lav_mki_max'):7.4f} "
          f"{f(r,'f_c_Hz'):6.1f} {f(r,'f_max_spatial_Hz'):6.1f} {f(r,'fcut_over_fc'):6.2f} {f(r,'delta_lin'):6.2f} {f(r,'lav_delta'):6.2f}")
for label, S in (("All 85 cases", R), ("Cases with DNS data", has)):
    print(f"\n{label}:")
    agg("28 Hz / f_c (OS)", [f(r,'fcut_over_fc') for r in S])
    agg("28 Hz / f_max spatial (OS)", [28/f(r,'f_max_spatial_Hz') for r in S])
    print(f"  forcing cut-off below OS most-amplified spatial f: {sum(28 < f(r,'f_max_spatial_Hz') for r in S)}/{len(S)}")
    print(f"  forcing cut-off below OS cut-off f_c:              {sum(28 < f(r,'f_c_Hz') for r in S)}/{len(S)}")
    agg("delta_lin (OS k_max Re)", [f(r,'delta_lin') for r in S])
    agg("k_max OS / Lavalle", [f(r,'os_k_max')/f(r,'lav_k_max') for r in S])
    agg("max temporal growth Lavalle / OS", [f(r,'lav_growth_max')/f(r,'os_growth_max') for r in S])
    agg("max spatial growth Lavalle / OS", [f(r,'lav_mki_max')/f(r,'os_mki_max') for r in S])
    agg("k_c OS / Lavalle", [f(r,'os_k_c')/f(r,'lav_k_c') for r in S])
    agg("k_c OS / Benney sqrt(3Ca(1.2Re-cot b))", [f(r,'os_k_c')/math.sqrt(3*f(r,'Ca')*(1.2*f(r,'Re')-1/math.tan(math.radians(f(r,'beta_deg'))))) for r in S])
