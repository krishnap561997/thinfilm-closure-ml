"""Helpers shared by the held-out MLP figures (not a figure itself)."""
import sys, re; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv, json, h5py, numpy as np
import style as S

G = S.FMLP / "runs/ablation_01/layers1_width16"
TARGETS = {"delta_Mx_ML": (S.DMX, S.BLUE), "delta_Tw_ML": (S.DTW, S.ORANGE)}


def cases():
    return sorted(p.stem for p in (G / "first_derivatives/delta_Mx_ML/seed42/eval_heldout").glob("*.h5"))


def nice(c):
    re_, ca, th = re.match(r"Re(\d+)_Ca(\d_\d+)_theta(.+)", c).groups()
    return f"Re{int(re_)} Ca{float(ca.replace('_', '.')):g} β{th.replace('_', '.')}°"


def load(group, target, case):
    with h5py.File(G / group / target / "seed42/eval_heldout" / f"{case}.h5") as g:
        P, Y, idx, x = g["prediction"][:, 0].astype(float), g["truth"][:, 0].astype(float), g["indices"][:], g["x"][:]
    t = np.load(S.FMLP / "cache_dir" / case / "t.npy")
    return dict(P=P, Y=Y, ti=idx[:, 0], x=x[idx[:, 1]], t=t[idx[:, 0]], sigma=Y.std())


def groups():
    sw = json.loads((G / "sweep.json").read_text())["runs"]
    out = {}
    for r in sw: out[r["input_group"]] = r["input_fields"]
    return out


def val_nrmse_curve(group, target):
    rows = list(csv.DictReader((G / group / target / "seed42/history.csv").open()))
    return np.sqrt(np.array([float(r["val_loss"]) for r in rows]))


TEX = {"h": "h", "q": "q", "h_x": "h_x", "h_xx": "h_{xx}", "q_x": "q_x", "q_xx": "q_{xx}"}
def label(fields): return "$" + ",\\ ".join(TEX[f] for f in fields) + "$"
