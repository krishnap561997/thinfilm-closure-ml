"""Numpy evaluation of the MLP text files used by rom/lavalle-cpp (src/ml/ml_closure.cpp):
inputs normalised by x_mean/x_std, tanh hidden layers, linear output, y*y_std + y_mean.
'angle' is in radians (Parameters::theta)."""
import os
import numpy as np

ORACLE = os.path.join(os.environ.get("THINFILM_STORE", ""), "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz")
MODELS = os.path.join(ORACLE, "rom/models")


class MLP:
    def __init__(self, path):
        self.path = path
        tok = open(path).read().split()
        i, self.layers = 0, []
        while i < len(tok):
            t = tok[i]
            if t == "targets":
                self.target = tok[i + 1]; i += 2
            elif t == "inputs":
                n = int(tok[i + 1]); self.inputs = tok[i + 2:i + 2 + n]; i += 2 + n
            elif t in ("x_mean", "x_std"):
                n = len(self.inputs); setattr(self, t, np.array(tok[i + 1:i + 1 + n], float)); i += 1 + n
            elif t in ("y_mean", "y_std"):
                setattr(self, t, float(tok[i + 1])); i += 2
            elif t == "layers":
                i += 2
            elif t == "W":
                r, c = int(tok[i + 1]), int(tok[i + 2])
                W = np.array(tok[i + 3:i + 3 + r * c], float).reshape(r, c); i += 3 + r * c
                self.layers.append([W, None])
            elif t == "b":
                n = int(tok[i + 1]); self.layers[-1][1] = np.array(tok[i + 2:i + 2 + n], float); i += 2 + n
            else:
                raise ValueError(f"bad token {t} in {path}")

    def __call__(self, **kw):
        """kw: arrays/scalars for every input name (Re, Ca, angle in radians). Broadcasts."""
        cols = np.broadcast_arrays(*[np.asarray(kw[n], float) for n in self.inputs])
        a = (np.stack(cols, -1) - self.x_mean) / self.x_std
        for j, (W, b) in enumerate(self.layers):
            a = a @ W.T + b
            if j + 1 < len(self.layers):
                a = np.tanh(a)
        return a[..., 0] * self.y_std + self.y_mean


def sum_model():
    """S = dMx + dTw from the (h, h_x) sum model."""
    m = MLP(os.path.join(MODELS, "sum_h_hx.txt"))
    return lambda h, q, hx, qx, Re, Ca, ang: m(h=h, h_x=hx, Re=Re, Ca=Ca, angle=ang)


def sep_models():
    """S = dMx + dTw from the separate (h, q, h_x, q_x) models."""
    mx = MLP(os.path.join(MODELS, "mx_first_derivatives.txt"))
    tw = MLP(os.path.join(MODELS, "tw_first_derivatives.txt"))
    f = lambda m, h, q, hx, qx, Re, Ca, ang: m(h=h, q=q, h_x=hx, q_x=qx, Re=Re, Ca=Ca, angle=ang)
    return lambda *a: f(mx, *a) + f(tw, *a)
