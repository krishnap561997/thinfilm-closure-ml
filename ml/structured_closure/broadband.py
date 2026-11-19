"""Broadband held-out a posteriori test (plan Phase D, gate GD): wave statistics of the ROM against the DNS.

The ROM runs each held-out broadband DNS case from a flat film with the same Chang inlet parameters as the DNS
(F0 0.025, M 1000, f_cut 28 Hz), over the same domain (L 800) and time (t 0-2000, output every 10). The inlet
realisations differ (different random phases), so only statistics are compared, over t >= T0 (default 1000)
and 20 < x < 760.

make [--closure NAME ...] [--cases ...]
    writes $THINFILM_STORE/codes/ml-models/structured_closure/rom/broadband/<closure>/<case>/ (input.txt,
    closure.txt) and rom/broadband/runs.txt. Closures: lavalle, ml_sum_hhx (the 1x16 (h, h_x) sum MLP of
    ml-a-posteriori.md), and any structured_closure run name (V1: tables per case; flat MLP: one file).
    Run with run_broadband.sh.
analyse [--closure NAME ...] [--cases ...] [--t0 1000] [--csv out.csv]
    per case: h mean/std/skewness, main-crest height (prominence >= std h; mean, p90), mean wavelength (spectral centroid), and three distances to the
    DNS: envelope error (rms over x of std_ROM(x) - std_DNS(x), relative to mean std_DNS), PDF distance
    (Wasserstein-1 of h over std_DNS) and crest-height ratio.
"""
import argparse, json, math, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D  # noqa: E402

STORE = os.environ["THINFILM_STORE"]
OUT = os.path.join(D.BASE, "rom", "broadband")
PARAM = os.path.join(STORE, "datasets/BasC/DNS/param_space")
CLOSURE_DATA = os.path.join(PARAM, "ClosureData")
ML_SUM = os.path.join(STORE, "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/rom/runs/ml_sum_hhx/ml.txt")
X0, X1 = 20.0, 760.0
CASES = ["Re020_Ca0_010_theta45", "Re040_Ca0_040_theta45", "Re080_Ca0_005_theta6_4", "Re080_Ca0_010_theta90",
         "Re120_Ca0_010_theta45", "Re120_Ca0_020_theta90", "Re200_Ca0_010_theta6_4"]


def dns_params(case):
    kv = {}
    for line in open(os.path.join(PARAM, case, "params.in")):
        line = line.split("#")[0]
        if "=" in line:
            k, v = line.split("=", 1); kv[k.strip()] = v.strip()
    return kv


def input_txt(case):  # RHO_L, MU_L default to filmflow0.c (1072, 0.00673) when params.in omits them
    p = dns_params(case)
    return (f"fcut_dim  = {p.get('FCUT_DIM', '28.0')}\nMNOISE    = {p.get('MNOISE', '1000')}\n"
            f"F0_noise  = {p.get('F0_NOISE', '0.025')}\n\nt_end  = {p.get('T_END', '2000')}\ndt     = 0.005\n"
            f"t_save = {p.get('T_OUT', '10')}\n\nL = {p.get('LX', '800')}\nN = {int(round(float(p.get('LX', '800')) / 0.2))}\n\n"
            f"Re_input  = {float(p['RE']):g}\ntheta_deg = {float(p['ANGLE_DEG']):g}\nrho       = {p.get('RHO_L', '1072')}\n"
            f"mu        = {p.get('MU_L', '0.00673')}\nCa_input  = {float(p['CA']):g}\n")


def closure_file(name, case):
    """Config file for the ROM's third argument, or None (Lavalle)."""
    if name == "lavalle":
        return None
    if name == "ml_sum_hhx":
        return ML_SUM
    if name.endswith("_sg"):   # flat MLP + training-range clamp and thin-film taper (safeguard.py)
        from safeguard import config
        return config(name)
    cfg = json.load(open(os.path.join(D.BASE, "runs", name, "config.json")))
    if cfg.get("model", "v1") == "mlp":
        from export_mlp import export_mlp
        return export_mlp(name)
    from export_tables import export
    export(name, case)
    return os.path.join(D.BASE, "runs", name, "tables", f"sc_{case}.txt")


def make(closures, cases, listname="runs.txt"):
    dirs = []
    for name in closures:
        for case in cases:
            d = os.path.join(OUT, name, case); os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "input.txt"), "w").write(input_txt(case))
            cf = closure_file(name, case)
            if cf: shutil.copy(cf, os.path.join(d, "closure.txt"))
            elif os.path.exists(os.path.join(d, "closure.txt")): os.remove(os.path.join(d, "closure.txt"))
            dirs.append(d)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, listname), "w") as f:
        f.write("\n".join(dirs) + "\n")
    print(f"{len(dirs)} run directories -> {OUT}/{listname}")


# ------------------------------------------------------------------------------------------------ statistics
def load_dns(case, t0, xr):
    import h5py
    with h5py.File(os.path.join(CLOSURE_DATA, f"thinfilm_closure_data_{case}.h5")) as f:
        t, x, h = f["t"][:], f["x"][:], f["h"]
        k = np.where(t >= t0 - 1e-9)[0]
        H = np.asarray(h[:, k[0]:k[-1] + 1]).T  # (nt, nx)
    return t[k], np.array([np.interp(xr, x, row) for row in H])


def load_rom(d, t0):
    import h5py
    fn = os.path.join(d, "thinfilm_training_data.h5")
    if not os.path.exists(fn):
        return None
    with h5py.File(fn) as f:
        t, x = f["time"][:], f["x"][:]
        k = np.where(t >= t0 - 1e-9)[0]
        if len(k) == 0:
            return None
        return t[k], x, f["h"][k[0]:k[-1] + 1]


def stats(x, H):
    from scipy.signal import find_peaks
    m = (x > X0) & (x < X1); hm = H[:, m]; v = hm.ravel()
    s = v.std(); sk = np.mean((v - v.mean()) ** 3) / s ** 3
    crests = np.concatenate([row[find_peaks(row, prominence=max(s, 1e-6))[0]] for row in hm])  # main crests: prominence >= std(h)
    # mean wavelength: 1 / centroid wavenumber of the mean spatial spectrum over 300 < x < 760 (k 1/400 to 1/2)
    w = (x > 300) & (x < X1); dx = x[1] - x[0]; n = int(w.sum())
    P = np.mean(np.abs(np.fft.rfft((H[:, w] - H[:, w].mean(1, keepdims=True)) * np.hanning(n), axis=1)) ** 2, 0)
    k = np.fft.rfftfreq(n, dx); kb = (k > 1 / 400) & (k < 0.5)
    # envelope: std over time in x-bins of 20
    edges = np.arange(X0, X1 + 1e-9, 20.0)
    env = np.array([H[:, (x >= a) & (x < b)].std() for a, b in zip(edges[:-1], edges[1:])])
    return dict(mean=v.mean(), std=s, skew=sk, ncrest=len(crests),
                crest_mean=crests.mean() if len(crests) else np.nan,
                crest_p90=np.percentile(crests, 90) if len(crests) else np.nan,
                wavelength=P[kb].sum() / (k[kb] * P[kb]).sum(), env=env, values=v)


def wasserstein(a, b):
    qs = np.linspace(0.005, 0.995, 199)
    return np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs)))


def analyse(closures, cases, t0, csv):
    rows = []
    for case in cases:
        ref = None
        for name in closures:
            r = load_rom(os.path.join(OUT, name, case), t0)
            if r is None:
                print(f"{case:24s} {name:24s} missing or not past t0"); continue
            t, x, H = r
            if ref is None:
                td, HD = load_dns(case, t0, x); ref = stats(x, HD)
                rows.append(dict(case=case, closure="DNS", n_snap=len(td), **{k: v for k, v in ref.items() if k not in ("env", "values")},
                                 env_err=0.0, pdf_dist=0.0, crest_ratio=1.0, t_last=td[-1]))
            st = stats(x, H)
            rows.append(dict(case=case, closure=name, n_snap=len(t), **{k: v for k, v in st.items() if k not in ("env", "values")},
                             env_err=np.sqrt(np.mean((st["env"] - ref["env"]) ** 2)) / ref["env"].mean(),
                             pdf_dist=wasserstein(st["values"], ref["values"]) / ref["std"],
                             crest_ratio=st["crest_mean"] / ref["crest_mean"], t_last=t[-1]))
    cols = ["n_snap", "t_last", "std", "skew", "crest_mean", "crest_p90", "wavelength", "env_err", "pdf_dist", "crest_ratio"]
    print(f"{'case':24s} {'closure':22s} " + " ".join(f"{c:>10s}" for c in cols))
    for r in rows:
        print(f"{r['case']:24s} {r['closure']:22s} " + " ".join(f"{r[c]:10.3f}" if isinstance(r[c], float) else f"{r[c]:10d}" for c in cols))
    if csv:
        import csv as C
        with open(csv, "w", newline="") as f:
            w = C.DictWriter(f, fieldnames=["case", "closure", "n_snap", "t_last", "mean", "std", "skew", "ncrest", "crest_mean",
                                            "crest_p90", "wavelength", "env_err", "pdf_dist", "crest_ratio"])
            w.writeheader(); [w.writerow(r) for r in rows]
        print("wrote", csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "analyse"])
    ap.add_argument("--closure", nargs="+", default=["lavalle", "ml_sum_hhx"])
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--t0", type=float, default=1000.0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--list", default="runs.txt", help="make: run list under rom/broadband/ (LIST= for run_broadband.sh)")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.closure, a.cases, a.list)
    else:
        analyse(a.closure, a.cases, a.t0, a.csv)


if __name__ == "__main__":
    main()
