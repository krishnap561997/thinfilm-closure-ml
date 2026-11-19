"""Safeguard for the flat-film-exact MLP in the ROM (19 Nov): clamp the network inputs to the training range and
taper S to zero in near-dry film (rom/lavalle-cpp FlatMLPClosure keys clip_<v>, taper_h).

A closure name ending in "_sg" (e.g. mlp_s0.5_L10_seed0_sg) means the run without the suffix plus this safeguard;
broadband.py and rom_tests.py accept such names.

    python ml/structured_closure/safeguard.py ranges --sigma 0.5     # print (and cache) the training ranges
    python ml/structured_closure/safeguard.py config RUN             # write runs/RUN/rom/sc_flatmlp_sg.txt

Ranges: 0.01 and 99.99 percentiles over the development cases (data.load_case, every 7th point; 0.1-99.9 clamped
~0.1% of cells on the oracle case, inside the data) of h, dq = q - h^3,
h_x, q_x, h_xx (zero-sum D_G(sigma), as the ROM). Taper: smoothstep from h 0.15 to 0.25, set from the DNS minima of
the held-out cases (0.18-0.29), not tuned on the test runs.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D  # noqa: E402

SUFFIX = "_sg"
VARS = ["h", "dq", "h_x", "q_x", "h_xx"]
PCT = (0.01, 99.99)
TAPER = (0.15, 0.25)


def base_run(name):
    return name[:-len(SUFFIX)] if name.endswith(SUFFIX) else name


def ranges(sigma):
    fn = os.path.join(D.BASE, "data", f"clip_ranges_s{sigma:g}.json".replace(".", "p", 1))
    if os.path.exists(fn):
        return json.load(open(fn))
    dev, _ = D.split(); cols = {k: [] for k in VARS}
    for c in dev:
        d = D.load_case(c, sigma=sigma)
        h, q = d["h"].ravel()[::7].astype(float), d["q"].ravel()[::7].astype(float)
        cols["h"].append(h); cols["dq"].append(q - h ** 3)
        for k in ("h_x", "q_x", "h_xx"): cols[k].append(d[k].ravel()[::7].astype(float))
    out = {k: [float(v) for v in np.nanpercentile(np.concatenate(cols[k]), PCT)] for k in VARS}
    os.makedirs(os.path.dirname(fn), exist_ok=True); json.dump(out, open(fn, "w"), indent=1)
    return out


def config(run):
    """Safeguarded ROM config for a flat-MLP run (export_mlp output plus clip_/taper_h keys)."""
    from export_mlp import export_mlp
    run = base_run(run)
    cf = export_mlp(run)
    txt = open(cf).read()
    sigma = float([l for l in txt.splitlines() if l.startswith("sigma")][0].split("=")[1])
    R = ranges(sigma)
    out = os.path.join(os.path.dirname(cf), "sc_flatmlp_sg.txt")
    open(out, "w").write(txt.rstrip("\n") + "\n# safeguard (safeguard.py): training-range clamp and thin-film taper\n"
                         + "".join(f"clip_{k} = {R[k][0]:.6g} {R[k][1]:.6g}\n" for k in VARS)
                         + f"taper_h = {TAPER[0]:g} {TAPER[1]:g}\n")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "ranges":
        s = float(sys.argv[sys.argv.index("--sigma") + 1]) if "--sigma" in sys.argv else 0.5
        for k, v in ranges(s).items(): print(f"{k:5s} {v[0]:.4g} {v[1]:.4g}")
    elif sys.argv[1] == "config":
        for r in sys.argv[2:]: print(config(r))
