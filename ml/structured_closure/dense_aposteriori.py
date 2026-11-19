"""Short deterministic a posteriori runs on broadband DNS: the ROM from the DNS state with the DNS inlet, over the
10 time units of the dense restarts (profiles every 0.01) in $THINFILM_STORE/datasets/BasC/DNS/dense/.

forcing [--cases ...]   builds dense/<case>/<run>/rom/rom_forcing.h5 from the profiles (h, q conservatively
                        averaged onto the ROM grid; init and inlet as pipeline/oracle/make_rom_forcing.py; no
                        closure terms, so only `terms = none` runs are possible)
make [--closure ...]    writes $THINFILM_STORE/codes/ml-models/structured_closure/rom/dense/<closure>/<case>/ and
                        rom/dense/runs.txt (run with run_broadband.sh LIST=../dense/runs.txt, or run_dense.sh)
analyse [--closure ...] h error rms(h_ROM - h_DNS)/std(h_DNS) over 20 < x < 760 at elapsed 0.5, 1, 2, 5, 10, and
                        max h ROM/DNS; --csv writes the error every 0.1
"""
import argparse, glob, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "pipeline/extract_closures"))
import data as D  # noqa: E402
from broadband import closure_file, dns_params  # noqa: E402

STORE = os.environ["THINFILM_STORE"]
DENSE = os.path.join(STORE, "datasets/BasC/DNS/dense")
OUT = os.path.join(D.BASE, "rom", "dense")
CASES = ["Re020_Ca0_010_theta45", "Re120_Ca0_010_theta45", "Re040_Ca0_002_theta45"]
N, L = 4000, 800.0
TIMES = [0.5, 1, 2, 5, 10]


def run_dir(case):
    return sorted(glob.glob(os.path.join(DENSE, case, "run_t*_budget")))[0]


def forcing_path(case):
    return os.path.join(run_dir(case), "rom", "rom_forcing.h5")


def build_forcing(case):
    import h5py
    from check_momentum_budget import restrict
    files = sorted(glob.glob(os.path.join(run_dir(case), "profiles", "profile_*.h5")))
    out = forcing_path(case); os.makedirs(os.path.dirname(out), exist_ok=True)
    XR = (np.arange(N) + 0.5) * L / N
    with h5py.File(files[0]) as f:
        x = f["x"][:]; par = {k: float(f["parameters"][k][0]) for k in ("Re", "Ca", "angle")}
    nt = len(files)
    with h5py.File(out, "w") as o:
        for k, v in par.items(): o.attrs[k] = v
        o.attrs["L"] = L; o.attrs["source"] = os.path.dirname(files[0])
        o.create_dataset("x", data=XR); t = o.create_dataset("t", (nt,), "f8")
        H = o.create_dataset("dns/h", (nt, N), "f8", chunks=(1, N)); Q = o.create_dataset("dns/q", (nt, N), "f8", chunks=(1, N))
        inl = {k: o.create_dataset(f"inlet/{k}", (nt,), "f8") for k in ("h", "q", "h_x")}
        for j, fn in enumerate(files):
            with h5py.File(fn) as f:
                t[j] = float(f["parameters"]["time"][0])
                h, q = f["h"][:], f["q"][:]
            hR, qR, hxR = restrict(x, h, N, L), restrict(x, q, N, L), restrict(x, np.gradient(h, x), N, L)
            H[j], Q[j] = hR, qR
            inl["h"][j], inl["q"][j], inl["h_x"][j] = hR[0], qR[0], hxR[0]
            if j == 0:
                o.create_dataset("init/h", data=hR); o.create_dataset("init/q", data=qR); o.create_dataset("init/h_x", data=hxR)
        tv = t[:]
        if np.any(np.diff(tv) <= 0): raise SystemExit(f"{case}: profile times not increasing")
    print(f"{case}: {nt} profiles t {tv[0]:.2f}-{tv[-1]:.2f} -> {out}")


def make(closures, cases):
    dirs = []
    for name in closures:
        for case in cases:
            d = os.path.join(OUT, name, case); os.makedirs(d, exist_ok=True)
            p = dns_params(case)
            import h5py
            with h5py.File(forcing_path(case)) as f: dur = float(f["t"][-1] - f["t"][0]) - 0.005
            open(os.path.join(d, "input.txt"), "w").write(
                f"fcut_dim  = 20.0\nMNOISE    = 1000\nF0_noise  = 0.0\n\nt_end  = {dur:.3f}\ndt     = 0.005\nt_save = 0.1\n\n"
                f"L = {L}\nN = {N}\n\nRe_input  = {float(p['RE']):g}\ntheta_deg = {float(p['ANGLE_DEG']):g}\n"
                f"rho       = {p.get('RHO_L', '1072')}\nmu        = {p.get('MU_L', '0.00673')}\nCa_input  = {float(p['CA']):g}\n")
            open(os.path.join(d, "oracle.txt"), "w").write(f"forcing_file = {forcing_path(case)}\nterms = none\n")
            cf = closure_file(name, case)
            if cf: shutil.copy(cf, os.path.join(d, "closure.txt"))
            dirs.append(d)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "runs.txt"), "w").write("\n".join(dirs) + "\n")
    print(f"{len(dirs)} run directories -> {OUT}/runs.txt")


def analyse(closures, cases, csv):
    import h5py
    rows = []
    for case in cases:
        with h5py.File(forcing_path(case)) as f:
            tD, x, HD = f["t"][:], f["x"][:], f["dns/h"]
            m = (x > 20) & (x < 760)
            print(f"\n== {case} (t0 = {tD[0]:.2f})")
            print(f"{'closure':24s} " + " ".join(f"{'t=' + str(t):>13s}" for t in TIMES))
            for name in closures:
                fn = os.path.join(OUT, name, case, "thinfilm_training_data.h5")
                if not os.path.exists(fn): print(f"{name:24s} missing"); continue
                with h5py.File(fn) as r:
                    tR, HR = r["time"][:], r["h"]
                    line = f"{name:24s} "
                    for k in range(len(tR)):
                        j = int(np.argmin(abs(tD - tR[k])))
                        if abs(tD[j] - tR[k]) > 0.006: continue
                        hd, hr = HD[j][m], HR[k][m]
                        rows.append(dict(case=case, closure=name, elapsed=tR[k] - tD[0],
                                         err=np.sqrt(np.mean((hr - hd) ** 2)) / hd.std(), maxh_rom=hr.max(), maxh_dns=hd.max()))
                    for t in TIMES:
                        rr = [r_ for r_ in rows if r_["case"] == case and r_["closure"] == name]
                        if not rr: break
                        b = min(rr, key=lambda r_: abs(r_["elapsed"] - t))
                        line += f" {b['err']:5.3f} {b['maxh_rom']:.2f}/{b['maxh_dns']:.2f}"[:14].rjust(14)
                    print(line)
    if csv:
        import csv as C
        with open(csv, "w", newline="") as f:
            w = C.DictWriter(f, fieldnames=["case", "closure", "elapsed", "err", "maxh_rom", "maxh_dns"]); w.writeheader()
            [w.writerow(r) for r in rows]
        print("wrote", csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["forcing", "make", "analyse"])
    ap.add_argument("--closure", nargs="+", default=["lavalle", "ml_sum_hhx"])
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    if a.cmd == "forcing":
        [build_forcing(c) for c in a.cases]
    elif a.cmd == "make":
        make(a.closure, a.cases)
    else:
        analyse(a.closure, a.cases, a.csv)


if __name__ == "__main__":
    main()
