"""OS spatial dispersion targets for the L2 constraint of the structured closure (plan task C.1).

For each case: 20 frequencies over the forced band of B.8 (omega in (0.05 omega_c, min(2 omega_max,
0.95 omega_c))), the OS spatial wavenumbers k(omega) there, and D_OS = 1.2 Re - cot(beta).
The constraint is applied to the dispersion relation itself, not to fitted (a, b, c, d): in the ROM
the flux part of the closure is differentiated by the finite-volume scheme (unfiltered) while the
relaxation part sees the Gaussian-filtered gradients D_G(sigma), so the two parts of b, c, d carry
different factors at finite k (see losses.dispersion_residual).

Output: $THINFILM_STORE/codes/ml-models/structured_closure/os_dispersion.json
Usage: python os_dispersion.py [--cases ... | --all] [--ncpu N]
"""
import argparse, csv, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld"))
import os_solver as osv  # noqa: E402

STORE = os.environ["THINFILM_STORE"]
OUT = os.path.join(STORE, "codes/ml-models/structured_closure")
NW = 20


def one(row):
    t0 = time.time()
    Re, Ca, beta = float(row["Re"]), float(row["Ca"]), float(row["beta_deg"])
    cotb = 1 / math.tan(math.radians(beta))
    wc, wm = float(row["os_omega_c"]), float(row["os_omega_max"])
    om = np.linspace(0.05 * wc, min(2 * wm, 0.95 * wc), NW)
    out = dict(case=row["case"], Re=Re, Ca=Ca, beta_deg=beta, delta_lin=float(row["delta_lin"]),
               omega_c=wc, omega_max=wm, D_os=1.2 * Re - cotb, omega=om.tolist())
    try:
        K, _ = osv.spatial_curve(om, Re, Ca, cotb)
        K = np.asarray(K)
        out.update(k_re=K.real.tolist(), k_im=K.imag.tolist())
        lav = np.array([osv.lavalle_spatial(o, Re, Ca, cotb) for o in om])
        out["lav_err_ki"] = float(np.max(np.abs(lav.imag - K.imag)) / np.max(np.abs(K.imag)))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--all", action="store_true", help="every case of the OS case table (85), not only cache-v2 cases")
    ap.add_argument("--ncpu", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    A = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(STORE, "analysis/orr_sommerfeld/case_table.csv"))))
    if A.cases:
        rows = [r for r in rows if r["case"] in A.cases]
    elif not A.all:   # every case that has a cache-v2 file
        man = json.load(open(os.path.join(STORE, "codes/ml-models/cache_v2/manifest.json")))
        rows = [r for r in rows if r["case"] in man["cases"]]
    with Pool(max(1, A.ncpu)) as p:
        res = p.map(one, rows, chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, "os_dispersion.json")
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update({r["case"]: r for r in res})
    json.dump(old, open(fn, "w"), indent=1)
    bad = [r["case"] for r in res if "error" in r]
    print(f"{len(res)} cases -> {fn}; errors: {bad}")
    for r in res:
        if "error" not in r:
            print(f"  {r['case']:24s} omega {r['omega'][0]:.3f}-{r['omega'][-1]:.3f}  max -k_i {-min(r['k_im']):.4f}"
                  f"  Lavalle k_i err {r['lav_err_ki']:.3f}  {r['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
