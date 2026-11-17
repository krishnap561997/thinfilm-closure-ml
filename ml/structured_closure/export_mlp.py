"""Export a flat-film-exact MLP run (train.py --model mlp) for lavalle-cpp.

Writes <run>/rom/flat_mlp.txt in the format of the ROM's MLP class (ml_closure.cpp: inputs, x_mean, x_std,
y_mean, y_std, layers, W, b; tanh hidden layers) and <run>/rom/sc_flatmlp.txt with keys
    flat_mlp = <file>      selects the FlatMLPClosure (S = N(state) - N(h, h^3, 0, 0, 0), added as -S)
    sigma = <sigma>        D_G width of the gradients (zero-sum operator), as in training
The model takes Re, Ca, angle from the ROM parameters, so one file serves every case.
Usage: python export_mlp.py RUN
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D  # noqa: E402
from model import load_run  # noqa: E402


def export_mlp(run):
    rd = os.path.join(D.BASE, "runs", run)
    m, cfg = load_run(rd)
    out = os.path.join(rd, "rom"); os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, "flat_mlp.txt")
    lin = [l for l in list(m.body) + [m.out] if isinstance(l, torch.nn.Linear)]
    with open(fn, "w") as f:
        f.write("targets S_flat_exact\n")
        f.write(f"inputs {len(m.NAMES)} {' '.join(m.NAMES)}\n")
        f.write("x_mean " + " ".join(f"{v:.12g}" for v in m.xm.tolist()) + "\n")
        f.write("x_std " + " ".join(f"{v:.12g}" for v in m.xs.tolist()) + "\n")
        f.write("y_mean 0\n" + f"y_std {float(m.ys):.12g}\n" + f"layers {len(lin)}\n")
        for l in lin:
            W, b = l.weight.detach(), l.bias.detach()
            f.write(f"W {W.shape[0]} {W.shape[1]}\n" + " ".join(f"{v:.12g}" for v in W.flatten().tolist()) + "\n")
            f.write(f"b {b.numel()}\n" + " ".join(f"{v:.12g}" for v in b.tolist()) + "\n")
    cf = os.path.join(out, "sc_flatmlp.txt")
    open(cf, "w").write(f"flat_mlp = {fn}\nsigma = {cfg['sigma']}\n")
    print(f"{run}: -> {fn}")
    return cf


if __name__ == "__main__":
    for r in sys.argv[1:]:
        export_mlp(r)
