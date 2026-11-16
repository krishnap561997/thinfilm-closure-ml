"""B.6: linear-regime ROM runs with a frequency-comb inlet (plan task B.6).

The Chang inlet with MNOISE = M modes is a comb q_in = 1 + A sqrt(2/M) sum_k cos(k dw t + phi_k),
dw = 2 pi fcut TN / M.  fcut_dim is chosen so that the comb period 2 pi/dw is exactly TC = 100,
and the per-mode amplitude A sqrt(2/M) = AMP.  Every mode is linear and is separated exactly by a
projection over the last TC of the run (analyse.py).  Flat initial state (oracle argument "none").
Writes run directories under $THINFILM_STORE/analysis/orr_sommerfeld/b6_rom/<case>/<model>/ and
runs.txt (one directory per line) for run_array.sh.
"""
import math, os

STORE = os.environ["THINFILM_STORE"]
OUT = os.path.join(STORE, "analysis/orr_sommerfeld/b6_rom")
MODELS_DIR = os.path.join(STORE, "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/rom/models")
TC, AMP, DX, DT, TSAVE = 100.0, 1e-5, 0.2, 0.005, 0.5
CASES = {  # case: Re, Ca, beta, rho, mu, M, L  (L ~ 7 e-folds of the OS max spatial growth)
    "Re020_Ca0_010_theta45": (20.0, 0.01, 45.0, 1111.0375, 0.0044543213, 20, 250.0),
    "Re120_Ca0_010_theta45": (120.0, 0.01, 45.0, 1072.0, 0.00673, 24, 400.0),
    "Re200_Ca0_010_theta45": (200.0, 0.01, 45.0, 1030.6231, 0.0014260519, 24, 450.0),
}
MODELS = {
    "lavalle": None,
    "ml_sum_hhx": f"sum_model = {MODELS_DIR}/sum_h_hx.txt\n",
    "ml_mx_tw": f"mx_model = {MODELS_DIR}/mx_first_derivatives.txt\ntw_model = {MODELS_DIR}/tw_first_derivatives.txt\n",
}


def TN(Re, beta, rho, mu, g=9.81):
    nu = mu / rho
    h = (3 * Re * nu * nu / (g * math.sin(math.radians(beta)))) ** (1 / 3)
    return h / (nu * Re / h)


dirs = []
for case, (Re, Ca, beta, rho, mu, M, L) in CASES.items():
    tn = TN(Re, beta, rho, mu)
    fcut = M / (TC * tn)                       # Hz: dw = 2 pi / TC
    t_end = math.ceil((L / 1.2 + TC + 20) / 10) * 10
    for model, ml in MODELS.items():
        d = os.path.join(OUT, case, model); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "input.txt"), "w") as f:
            f.write(f"fcut_dim  = {fcut:.12g}\nMNOISE    = {M}\nF0_noise  = {AMP * math.sqrt(M / 2):.12g}\n\n"
                    f"t_end  = {t_end:g}\ndt     = {DT:g}\nt_save = {TSAVE:g}\n\nL = {L:g}\nN = {int(round(L / DX))}\n\n"
                    f"Re_input  = {Re:g}\ntheta_deg = {beta:g}\nrho       = {rho:.10g}\nmu        = {mu:.10g}\nCa_input  = {Ca:g}\n")
        if ml:
            open(os.path.join(d, "ml.txt"), "w").write(ml)
        with open(os.path.join(d, "comb.txt"), "w") as f:
            f.write(f"TN {tn:.12g}\nTC {TC:g}\nM {M}\nAMP {AMP:g}\ndomega {2 * math.pi / TC:.12g}\nt_end {t_end:g}\n")
        dirs.append(d)
open(os.path.join(OUT, "runs.txt"), "w").write("\n".join(dirs) + "\n")
print(f"{len(dirs)} run directories -> {OUT}/runs.txt")
