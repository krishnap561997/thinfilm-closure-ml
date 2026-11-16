"""Cache v2 (functional-form-plan.md task 1.1; shared with structured-closure-os-plan.md Phase C).

From ClosureData/thinfilm_closure_data_<case>.h5 (fields (nx, nt), every 10 time units, full record),
writes $THINFILM_STORE/codes/ml-models/cache_v2/<case>.h5 with, as float32 arrays of shape (nt, nx):

  /x (nx), /t (nt)                      grid and kept snapshot times
  /h, /q, /int_u2_dy                    DNS depth-integrated fields (raw, unfiltered)
  /dF = int_u2_dy - q^2/h - h^5/5       momentum-flux defect (Lambda = 3)
  /delta_Mx_ML, /delta_Tw_ML            targets as extracted (Savitzky-Golay pipeline)
  /delta_Px_ML_v2                       if present in the closure file
  /s<sigma>/h_x, h_xx, h_xxx, q_x, q_xx, dF_x
        Gaussian-filter derivatives D_G(sigma), sigma in h_N for sigma in SIGMAS: scipy.ndimage.
        gaussian_filter1d(mode='nearest', truncate=4) along x, the operator of the ROM's
        gauss_deriv (rom/lavalle-cpp/src/ml/ml_closure.cpp).  h and q themselves stay unfiltered,
        as the ROM's ML inputs.
  attrs: Re, Ca, angle_rad, beta_deg, We, cotb, dx, split, source, source_mtime, dropped_t

No masks are applied: inlet/outlet exclusion and any flat-film selection are training-time choices
(note: v1 used flat_mask, which removed |h - 1| < 0.05; see docs/results/structured-closure.md A.1).

Snapshot hygiene (docs/decisions.md, 10 Nov 2026): snapshots overwritten by the unforced restarts of
10 Nov are dropped when present.  Re050 Ca0.01 b45 is used only if its closure file is newer than the
rerun (10 Nov); an older file would contain the unforced t > 1050 record and is dropped beyond 1050.

Check (per case and sigma): dF_x against delta_Mx_ML over the interior (5%-95% of x): NRMSE (rms of
the difference / std of delta_Mx_ML) and correlation.  Written to manifest.json and check.csv.

Usage: python build_cache_v2.py [--cases ...] [--ncpu N] [--out DIR]
"""
import argparse, csv, datetime, glob, json, math, os, subprocess, sys, time
from multiprocessing import Pool
import h5py, numpy as np
from scipy.ndimage import gaussian_filter1d

STORE = os.environ["THINFILM_STORE"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CD = os.path.join(STORE, "datasets/BasC/DNS/param_space/ClosureData")
OUT = os.path.join(STORE, "codes/ml-models/cache_v2")
SIGMAS = (0.25, 0.5, 1.0)
# snapshots overwritten by the unforced 10 Nov restarts (decisions.md); dropped wherever present
OVERWRITTEN = {
    "Re010_Ca0_005_theta45": [760], "Re010_Ca0_010_theta45": [1960], "Re010_Ca0_040_theta90": [1910, 1920],
    "Re010_Ca0_100_theta45": [1560, 1570], "Re020_Ca0_005_theta6_4": [1210], "Re040_Ca0_002_theta45": [1110],
    "Re040_Ca0_100_theta45": [1810, 1820], "Re040_Ca0_100_theta90": list(range(110, 290, 10)),
}
RE050_RERUN = datetime.datetime(2026, 11, 10).timestamp()


def sig_tag(s):
    return "s" + f"{s:g}".replace(".", "p")


def scalar(g, k):
    v = g[k][()] if k in g else g.attrs[k]
    return float(np.asarray(v).squeeze())


def build(case, out):
    t0 = time.time()
    src = os.path.join(CD, f"thinfilm_closure_data_{case}.h5")
    cfg = json.load(open(os.path.join(ROOT, "ml/functional_mlp/data_config.json")))
    split = "dev" if case in cfg["development_cases"] else "heldout" if case in cfg["heldout_cases"] else "extra"
    with h5py.File(src) as f:
        x, t = f["x"][:], f["t"][:]
        P = f["parameters"]
        Re, Ca, ang = scalar(P, "Re"), scalar(P, "Ca"), scalar(P, "angle")
        ang = ang if ang < 3.2 else math.radians(ang)
        keep = np.ones(len(t), bool)
        dropped = []
        for tb in OVERWRITTEN.get(case, []):
            hit = np.abs(t - tb) < 1e-6
            if hit.any():
                keep &= ~hit; dropped.append([float(tb), "overwritten by unforced 10 Nov restart"])
        if case == "Re050_Ca0_010_theta45" and os.path.getmtime(src) < RE050_RERUN:
            bad = t > 1050 + 1e-6
            keep &= ~bad; dropped += [[float(v), "Re050 unforced record t > 1050 (pre-rerun file)"] for v in t[bad]]
        it = np.where(keep)[0]
        rd = lambda k: f[k][:, it].T.astype(np.float64)          # (nt, nx)
        h, q, M = rd("h"), rd("q"), rd("int_u2_dy")
        dMx, dTw = rd("delta_Mx_ML"), rd("delta_Tw_ML")
        dPx = rd("delta_Px_ML_v2") if "delta_Px_ML_v2" in f else None
    t = t[it]
    dx = float(np.median(np.diff(x)))
    dF = M - q ** 2 / h - h ** 5 / 5.0
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"{case}.h5"); tmp = fn + ".tmp"
    comp = dict(compression="gzip", compression_opts=4, chunks=(1, len(x)))
    checks = []
    inner = (x > x[0] + 0.05 * (x[-1] - x[0])) & (x < x[0] + 0.95 * (x[-1] - x[0]))
    with h5py.File(tmp, "w") as g:
        g["x"] = x; g["t"] = t
        for k, v in (("h", h), ("q", q), ("int_u2_dy", M), ("dF", dF), ("delta_Mx_ML", dMx), ("delta_Tw_ML", dTw)) + \
                ((("delta_Px_ML_v2", dPx),) if dPx is not None else ()):
            g.create_dataset(k, data=v.astype(np.float32), **comp)
        for s in SIGMAS:
            gd = lambda v, m: gaussian_filter1d(v, s / dx, axis=1, order=m, mode="nearest") / dx ** m
            grp = g.create_group(sig_tag(s)); grp.attrs["sigma_hN"] = s
            fields = dict(h_x=gd(h, 1), h_xx=gd(h, 2), h_xxx=gd(h, 3), q_x=gd(q, 1), q_xx=gd(q, 2), dF_x=gd(dF, 1))
            for k, v in fields.items():
                grp.create_dataset(k, data=v.astype(np.float32), **comp)
            a, b = fields["dF_x"][:, inner], dMx[:, inner]
            checks.append(dict(case=case, sigma=s, nrmse=float(np.sqrt(np.mean((a - b) ** 2)) / b.std()),
                               corr=float(np.corrcoef(a.ravel(), b.ravel())[0, 1])))
        g.attrs.update(Re=Re, Ca=Ca, angle_rad=ang, beta_deg=math.degrees(ang), We=Re * Ca, cotb=1 / math.tan(ang),
                       dx=dx, split=split, source=src, source_mtime=os.path.getmtime(src),
                       dropped_t=json.dumps(dropped), sigmas=json.dumps(SIGMAS), layout="(nt, nx)")
    os.replace(tmp, fn)
    return dict(case=case, split=split, Re=Re, Ca=Ca, beta_deg=math.degrees(ang), nt=len(t), nx=len(x),
                t_first=float(t[0]), t_last=float(t[-1]), dropped=dropped, has_dPx_v2=dPx is not None,
                source=src, source_mtime=os.path.getmtime(src), file=fn, size_MB=os.path.getsize(fn) / 1e6,
                checks=checks, wall_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--ncpu", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()
    avail = sorted(os.path.basename(p)[len("thinfilm_closure_data_"):-3] for p in glob.glob(os.path.join(CD, "thinfilm_closure_data_*.h5")))
    cases = A.cases or avail
    missing = [c for c in cases if c not in avail]
    cases = [c for c in cases if c in avail]
    if A.ncpu > 1:
        with Pool(A.ncpu) as p:
            res = p.starmap(build, [(c, A.out) for c in cases], chunksize=1)
    else:
        res = [build(c, A.out) for c in cases]
    # merge into manifest (keeps entries of cases not rebuilt now)
    mfn = os.path.join(A.out, "manifest.json")
    man = json.load(open(mfn)) if os.path.exists(mfn) else {"cases": {}}
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    budget = {}
    bfn = os.path.join(CD, "budget_check/summary.csv")
    if os.path.exists(bfn):
        budget = {r["case"]: dict(mass_rel=float(r["mass_rel"]), dns_res_rel=float(r["dns_res_rel"]))
                  for r in csv.DictReader(open(bfn))}
    man.update(schema="cache_v2", sigmas=list(SIGMAS), layout="(nt, nx) float32, gzip 4",
               derivative="scipy gaussian_filter1d mode=nearest truncate=4 (= ROM gauss_deriv)",
               dF="int_u2_dy - q^2/h - h^5/5", built=datetime.datetime.now().isoformat(timespec="seconds"),
               git_commit=commit, split_source="ml/functional_mlp/data_config.json",
               missing_closure_files=sorted(set(man.get("missing_closure_files", [])) | set(missing)))
    for r in res:
        r["budget_check"] = budget.get(r["case"])
        man["cases"][r["case"]] = r
    man["missing_closure_files"] = sorted(c for c in man["missing_closure_files"] if c not in man["cases"])
    json.dump(man, open(mfn, "w"), indent=1)
    with open(os.path.join(A.out, "check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "split", "sigma", "nrmse", "corr"]); w.writeheader()
        for c, r in sorted(man["cases"].items()):
            for ch in r["checks"]:
                w.writerow(dict(case=c, split=r["split"], sigma=ch["sigma"], nrmse=f"{ch['nrmse']:.4f}", corr=f"{ch['corr']:.4f}"))
    print(f"built {len(res)} cases -> {A.out}" + (f"; no closure file for {missing}" if missing else ""))
    for r in res:
        ch = " ".join(f"s{c['sigma']:g}: {c['nrmse']:.3f}/{c['corr']:.3f}" for c in r["checks"])
        print(f"  {r['case']:24s} {r['split']:7s} nt {r['nt']:3d} t {r['t_first']:.0f}-{r['t_last']:.0f} "
              f"dropped {len(r['dropped'])} {r['size_MB']:.0f} MB {r['wall_s']:.0f}s | dF_x vs dMx nrmse/corr {ch}")


if __name__ == "__main__":
    main()
