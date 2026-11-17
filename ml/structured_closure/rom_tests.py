"""ROM tests of trained structured closures (plan C.10) and the first a posteriori run (Phase D).

make RUN [RUN ...]   exports tables and writes run directories under
                     $THINFILM_STORE/codes/ml-models/structured_closure/rom/<RUN>/:
    uniform/            flat film h = q = 1, no inlet forcing, 100 time units: must stay uniform
    comb_<case>/        B.6 comb inlet (1e-5 per mode, period 100), Re 20/120/200 Ca0.01 b45
    oracle/             Re120 Ca0.01 b45 10 Hz, from the DNS state t = 1800, 100 time units,
                        DNS inlet (oracle.txt terms = none): as the ML runs of ml-a-posteriori.md
                     and appends them to rom/runs.txt (run with run_rom_tests.sh).
analyse RUN [RUN ...] prints the uniform drift, comb growth errors (ROM vs OS, ROM vs the model's own
                     linear dispersion) and the oracle h error against the DNS next to Lavalle and the
                     earlier ML runs.
"""
import math, os, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld"))
sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld/b6_rom"))
import data as D  # noqa: E402
from export_tables import export  # noqa: E402

STORE = os.environ["THINFILM_STORE"]
OUT = os.path.join(D.BASE, "rom")
ORACLE = os.path.join(STORE, "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/rom")
B6 = os.path.join(STORE, "analysis/orr_sommerfeld/b6_rom")
COMB = ["Re020_Ca0_010_theta45", "Re120_Ca0_010_theta45", "Re200_Ca0_010_theta45"]


def make(runs):
    dirs = []
    for run in runs:
        base = os.path.join(OUT, run)
        import json
        if json.load(open(os.path.join(D.BASE, "runs", run, "config.json"))).get("model", "v1") == "mlp":
            from export_mlp import export_mlp
            cf = export_mlp(run)
            sc = lambda c: cf
        else:
            tabs = {c: export(run, c) for c in COMB}
            sc = lambda c: os.path.join(D.BASE, "runs", run, "tables", f"sc_{c}.txt")
        # uniform film: oracle-case parameters, no forcing, flat start
        d = os.path.join(base, "uniform"); os.makedirs(d, exist_ok=True)
        txt = open(os.path.join(B6, "Re120_Ca0_010_theta45/lavalle/input.txt")).read()
        lines = [l for l in txt.splitlines() if not l.startswith(("F0_noise", "t_end", "t_save"))]
        open(os.path.join(d, "input.txt"), "w").write("\n".join(lines) + "\nF0_noise  = 0\nt_end  = 100\nt_save = 10\n")
        shutil.copy(sc("Re120_Ca0_010_theta45"), os.path.join(d, "sc.txt")); dirs.append(d)
        # comb runs: B.6 inputs
        for c in COMB:
            d = os.path.join(base, f"comb_{c}"); os.makedirs(d, exist_ok=True)
            for f in ("input.txt", "comb.txt"):
                shutil.copy(os.path.join(B6, c, "lavalle", f), d)
            shutil.copy(sc(c), os.path.join(d, "sc.txt")); dirs.append(d)
        # oracle a posteriori
        d = os.path.join(base, "oracle"); os.makedirs(d, exist_ok=True)
        for f in ("input.txt", "oracle.txt"):
            shutil.copy(os.path.join(ORACLE, "runs/ml_sum_hhx", f), d)
        shutil.copy(sc("Re120_Ca0_010_theta45"), os.path.join(d, "sc.txt")); dirs.append(d)
    with open(os.path.join(OUT, "runs.txt"), "w") as f:
        f.write("\n".join(dirs) + "\n")
    print(f"{len(dirs)} run directories -> {OUT}/runs.txt")


def analyse(runs):
    import h5py, json, torch
    import os_solver as osv
    from analyse import run_modes
    from model import StructuredV1
    from losses import disp_F
    for run in runs:
        base = os.path.join(OUT, run)
        print(f"===== {run}")
        d = os.path.join(base, "uniform")
        try:
            with h5py.File(os.path.join(d, "thinfilm_training_data.h5")) as f:
                h, q = f["h"][-1], f["q"][-1]
            print(f"uniform: max|h-1| {np.abs(h - 1).max():.2e}  max|q-1| {np.abs(q - 1).max():.2e}  (t = 100)")
        except Exception as e:
            print("uniform: not available", e)
        from model import load_run
        m, cfg = load_run(os.path.join(D.BASE, "runs", run))
        for c in COMB:
            d = os.path.join(base, f"comb_{c}")
            R, info = run_modes(d)
            if R is None:
                print(f"comb {c}: {info}"); continue
            Re, Ca, b = float(c[2:5]), 0.01, math.radians(45); cotb = 1 / math.tan(b)
            L = float([l for l in open(os.path.join(d, "input.txt")) if l.startswith("L ")][0].split("=")[1])
            rel = (R[:, 3] > 0.95) & (2 * np.pi / np.abs(R[:, 2]) < 0.35 * L)
            om = R[rel, 0]
            K, _ = osv.spatial_curve(om, Re, Ca, cotb)
            K = np.asarray(K)
            with torch.enable_grad():
                Lc = {k: v.detach() for k, v in m.linear(torch.tensor(D.pfeat(Re, Ca, b), dtype=torch.float64), Re, cotb).items()}
            k = torch.tensor(K); omt = torch.tensor(om)
            for _ in range(10):
                eps = 1e-6 * k.abs()
                F0 = disp_F(k, omt, Re, Re * Ca, cotb, Lc, cfg["sigma"])
                dF = (disp_F(k + eps, omt, Re, Re * Ca, cotb, Lc, cfg["sigma"]) - disp_F(k - eps, omt, Re, Re * Ca, cotb, Lc, cfg["sigma"])) / (2 * eps)
                k = k - F0 / dF
            gm = np.max(-K.imag)
            rom_g = R[rel, 1]          # run_modes: slope of ln|A| over x = spatial growth -k_i
            e_os = np.abs(rom_g - (-K.imag)) / gm; e_an = np.abs(rom_g - (-k.numpy().imag)) / gm
            lav = np.array([osv.lavalle_spatial(o, Re, Ca, cotb) for o in om]); e_lav = np.abs(-lav.imag - (-K.imag)) / gm
            print(f"comb {c}: {rel.sum()} modes (omega {om.min():.2f}-{om.max():.2f}); growth err vs OS median/max "
                  f"{np.median(e_os):.3f}/{e_os.max():.3f} (Lavalle {np.median(e_lav):.3f}/{e_lav.max():.3f}); "
                  f"ROM vs model analytic {np.median(e_an):.3f}/{e_an.max():.3f}; mean h {info['mean_h']:.4f}, max|h-1| {info['max_dev']:.2e}")
        d = os.path.join(base, "oracle")
        if os.path.exists(os.path.join(d, "thinfilm_training_data.h5")):
            cmd = [sys.executable, "-I", os.path.join(ROOT, "scripts/oracle_summary.py"), "--forcing", os.path.join(ORACLE, "rom_forcing.h5"),
                   "--run", f"lavalle={ORACLE}/runs/lavalle/thinfilm_training_data.h5",
                   "--run", f"oracle_TwMx={ORACLE}/runs/Tw_Mx/thinfilm_training_data.h5",
                   "--run", f"ml_sum_hhx={ORACLE}/runs/ml_sum_hhx/thinfilm_training_data.h5",
                   "--run", f"structured={d}/thinfilm_training_data.h5", "--csv", os.path.join(base, "oracle_summary.csv")]
            print(subprocess.run(cmd, capture_output=True, text=True).stdout[-3000:])
        else:
            print("oracle: not run yet")


if __name__ == "__main__":
    {"make": make, "analyse": analyse}[sys.argv[1]](sys.argv[2:])
