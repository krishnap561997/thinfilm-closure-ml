"""D3 / B.5: short comb-forced DNS from a flat film (plan structured-closure-os-plan.md, D3).

Production physics (case params.in: Re, Ca, beta, fluid, CFL, seed) on a short domain with the same
mesh and height: LX = 200, AR = 32, MAXLEVEL = 12 (dx = 200/2^12 = 800/2^14).  Inlet: the Chang
signal with MNOISE = M modes is a comb at k df, k = 1..M; FCUT_DIM = M/(TC T_N) makes the comb
period exactly TC = 100 (d omega = 2 pi/100); M = floor(FRAC_C omega_c/d omega) covers the OS unstable
band up to FRAC_C omega_c (B.4 case table); per-mode amplitude AMP (F0_NOISE = AMP sqrt(M/2)).

Phases (one SLURM job per case, run_comb.sh):
  spinup/   filmflow0.c,  t = 0 .. T_SPIN, dump at T_SPIN
  run_t<T_SPIN>/  filmflow_restart_profiles.c from that dump, t = T_SPIN .. T_SPIN + TC + 5,
            1D profiles every T_PROF (exact projection on the comb over one period)
Usage: python setup_comb_runs.py [--test]   (--test: one case, spin-up to t = 10 only, for timing)
Nothing is submitted.
"""
import argparse, csv, math, os, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STORE = os.environ["THINFILM_STORE"]
BASE = os.path.join(STORE, "datasets/BasC/DNS/comb")
CASES = ["Re020_Ca0_010_theta45", "Re040_Ca0_040_theta45", "Re080_Ca0_010_theta45",
         "Re120_Ca0_010_theta45", "Re200_Ca0_010_theta45"]
TC, AMP, FRAC_C, T_SPIN, T_PROF = 100.0, 3e-4, 0.97, 150.0, 0.25
OVR = dict(LX="200", AR="32", Zoom="32", MAXLEVEL="12", T_OUT="50", T_DUMP=f"{T_SPIN:g}")


def read_params(path):
    out = []
    for line in open(path):
        if "=" in line:
            k, v = line.split("=", 1); out.append((k.strip(), v.strip()))
    return out


def write_params(path, kv):
    with open(path, "w") as f:
        for k, v in kv:
            f.write(f"{k} = {v}\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); A = ap.parse_args()
    ct = {r["case"]: r for r in csv.DictReader(open(os.path.join(STORE, "analysis/orr_sommerfeld/case_table.csv")))}
    cases = ["Re120_Ca0_010_theta45"] if A.test else CASES
    base = os.path.join(BASE, "test") if A.test else BASE
    rows = []
    for case in cases:
        r = ct[case]; tn, wc = float(r["T0_s"]), float(r["os_omega_c"])
        dw = 2 * math.pi / TC
        M = int(FRAC_C * wc / dw)
        fcut = M / (TC * tn)
        kv = [(k, v) for k, v in read_params(os.path.join(ROOT, "dns/production/cases", case, "params.in"))
              if k not in OVR and k not in ("T_END", "MNOISE", "FCUT_DIM", "F0_NOISE")]
        kv += list(OVR.items()) + [("MNOISE", str(M)), ("FCUT_DIM", f"{fcut:.12g}"), ("F0_NOISE", f"{AMP * math.sqrt(M / 2):.12g}")]
        d = os.path.join(base, case)
        spin, run = os.path.join(d, "spinup"), os.path.join(d, f"run_t{T_SPIN:g}")
        os.makedirs(spin, exist_ok=False); os.makedirs(run)
        write_params(os.path.join(spin, "params.in"), kv + [("T_END", "10" if A.test else f"{T_SPIN:g}")])
        write_params(os.path.join(run, "params.in"), [(k, v) for k, v in kv if k != "T_DUMP"] +
                     [("T_END", f"{T_SPIN + TC + 5:g}"), ("T_DUMP", "100000"), ("T_OUT", "1000"),
                      ("T_PROF", f"{T_PROF:g}"), ("RESTART_FILE", "restart.dump")])
        with open(os.path.join(d, "COMB_INFO.txt"), "w") as f:
            f.write(f"case {case}\nT_N {tn:.10g}\nTC {TC:g}\nM {M}\ndomega {dw:.10g}\nomega_max_comb {M * dw:.6g}\n"
                    f"omega_c_OS {wc:.6g}\nFCUT_DIM_Hz {fcut:.6g}\nAMP_per_mode {AMP:g}\nT_SPIN {T_SPIN:g}\nT_PROF {T_PROF:g}\n"
                    f"window {T_SPIN + 5:g} .. {T_SPIN + 5 + TC:g}\n")
        rows.append((case, M, fcut, M * dw, wc))
    for c, M, fc, wm, wc in rows:
        print(f"{c}: M {M}, FCUT_DIM {fc:.1f} Hz, comb up to omega {wm:.3f} (OS omega_c {wc:.3f})")
    print("->", base)


if __name__ == "__main__":
    main()
