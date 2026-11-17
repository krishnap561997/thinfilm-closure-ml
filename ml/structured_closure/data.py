"""Training data for the structured closure, from cache v2 (pipeline/cache_v2/build_cache_v2.py).

Per case: points (x, t) with X0 < x < 0.95 L (every STRIDE-th grid point) and t >= T0, holding
    h, q                          raw DNS depth-integrated fields
    h_x, q_x, h_xx, dF_x          Gaussian-filter derivatives D_G(sigma) (cache group s<sigma>)
    S = delta_Mx_ML + delta_Tw_ML the target (the ROM adds -S to the q-equation)
The subsampled arrays are stored once per (sigma, x0, t0, stride) in
$THINFILM_STORE/codes/ml-models/structured_closure/data/<tag>/<case>.npz and reused.

Split: ml/functional_mlp/data_config.json (development / held-out). Cases excluded on 17 Nov 2026
(docs/decisions.md) are in EXCLUDE and never loaded.
"""
import json, math, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STORE = os.environ["THINFILM_STORE"]
CACHE = os.path.join(STORE, "codes/ml-models/cache_v2")
BASE = os.path.join(STORE, "codes/ml-models/structured_closure")
EXCLUDE = {"Re010_Ca0_001_theta90", "Re020_Ca0_001_theta6_4", "Re040_Ca0_100_theta90"}
FIELDS = ("h", "q", "h_x", "q_x", "h_xx", "dF_x", "S")


def split():
    cfg = json.load(open(os.path.join(ROOT, "ml/functional_mlp/data_config.json")))
    dev = [c for c in cfg["development_cases"] if c not in EXCLUDE]
    held = [c for c in cfg["heldout_cases"] if c not in EXCLUDE]
    return dev, held


def tag(sigma, x0, t0, stride):
    return f"s{sigma:g}_x{x0:g}_t{t0:g}_st{stride}".replace(".", "p")


def c0_second(sigma, dx):
    """h_xx that scipy's truncated Gaussian derivative returns for h = 1 (nonzero: -3.5e-3 at sigma 0.5 on
    the DNS grid).  The ROM uses the zero-sum operator (gauss_deriv_hN), so training removes c0 * h."""
    from scipy.ndimage import gaussian_filter1d
    return float(gaussian_filter1d(np.ones(int(20 * sigma / dx) + 50), sigma / dx, order=2, mode="nearest")[int(10 * sigma / dx) + 25] / dx ** 2)


def load_case(case, sigma=0.5, x0=50.0, t0=300.0, stride=8):
    out = _load_raw(case, sigma, x0, t0, stride)
    dx = float(np.median(np.diff(out["x"]))) / stride
    out["h_xx"] = (out["h_xx"] - c0_second(sigma, dx) * out["h"]).astype(np.float32)   # zero-sum operator
    return out


def _load_raw(case, sigma, x0, t0, stride):
    import h5py
    d = os.path.join(BASE, "data", tag(sigma, x0, t0, stride)); fn = os.path.join(d, f"{case}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return {k: z[k] for k in z.files}
    with h5py.File(os.path.join(CACHE, f"{case}.h5")) as f:
        x, t = f["x"][:], f["t"][:]
        ix = np.where((x > x0) & (x < x[0] + 0.95 * (x[-1] - x[0])))[0][::stride]
        it = np.where(t >= t0 - 1e-9)[0]
        g = f["s" + f"{sigma:g}".replace(".", "p")]
        rd = lambda ds: ds[it[0]:it[-1] + 1][it - it[0]][:, ix].astype(np.float32)
        out = dict(h=rd(f["h"]), q=rd(f["q"]), h_x=rd(g["h_x"]), q_x=rd(g["q_x"]), h_xx=rd(g["h_xx"]),
                   dF_x=rd(g["dF_x"]), S=rd(f["delta_Mx_ML"]) + rd(f["delta_Tw_ML"]))
        a = f.attrs
        out.update(Re=np.float32(a["Re"]), Ca=np.float32(a["Ca"]), beta=np.float32(a["angle_rad"]),
                   x=x[ix].astype(np.float32), t=t[it].astype(np.float32))
    os.makedirs(d, exist_ok=True)
    np.savez(fn, **out)
    return out


def pfeat(Re, Ca, beta):
    """Normalised parameter features p = (ln Re, ln Ca, beta)."""
    return np.array([(math.log(Re) - math.log(60.0)), (math.log(Ca) - math.log(0.01)) / 1.5,
                     (beta - 0.8) / 0.6], np.float32)


def flat(d):
    """(nt, nx) fields -> 1D arrays; drops points with non-finite values."""
    arr = {k: d[k].ravel() for k in FIELDS}
    ok = np.ones_like(arr["h"], bool)
    for v in arr.values():
        ok &= np.isfinite(v)
    return {k: v[ok] for k, v in arr.items()}
