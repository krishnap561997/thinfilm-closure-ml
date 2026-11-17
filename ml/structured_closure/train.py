"""Train the structured closure V1 (plan tasks C.1/C.2) on cache v2.

Loss = sum over training cases (equal weight per case) of mean (S_pred - S)^2 / rms_c(S)^2
     + lam_lin * mean over training cases [ mean r_j^2 + r_D^2 ]          (L2; lam_lin = 0 is L0)
     + lam_F  * per-case-normalised mean (d_x F - dF_x)^2                 (anchor of the flux part)
r_j, r_D: losses.dispersion_residual against os_dispersion.json.

Output: $THINFILM_STORE/codes/ml-models/structured_closure/runs/<name>/
    model.pt, config.json, metrics.json (per-case NRMSE, linear errors, (a,b,c,d), min discriminant), log.txt
Usage: python train.py --name NAME [--sigma 0.5] [--lam_lin 1] [--lam_F 0] [--seed 0] [--epochs 100]
       [--cases A B --heldout C]   (smoke tests)
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D          # noqa: E402
from model import StructuredV1      # noqa: E402
from losses import dispersion_residual, disp_F, stack_targets      # noqa: E402


def build(cases, a, dev):
    """Concatenate cases into flat tensors with per-point case index."""
    cols = {k: [] for k in D.FIELDS}; idx = []; meta = []
    for i, c in enumerate(cases):
        d = D.load_case(c, a.sigma, a.x0, a.t0, a.stride)
        f = D.flat(d)
        for k in D.FIELDS:
            cols[k].append(f[k])
        idx.append(np.full(len(f["h"]), i, np.int64))
        Re, Ca, beta = float(d["Re"]), float(d["Ca"]), float(d["beta"])
        meta.append(dict(case=c, Re=Re, Ca=Ca, beta=beta, cotb=1 / math.tan(beta), We=Re * Ca, n=len(f["h"]),
                         rmsS=float(np.sqrt(np.mean(f["S"] ** 2))), rmsF=float(np.sqrt(np.mean(f["dF_x"] ** 2))),
                         p=D.pfeat(Re, Ca, beta)))
    T = {k: torch.tensor(np.concatenate(v), device=dev) for k, v in cols.items()}
    T["idx"] = torch.tensor(np.concatenate(idx), device=dev)
    pc = lambda key: torch.tensor([m[key] for m in meta], dtype=torch.float32, device=dev)
    C = dict(Re=pc("Re"), cotb=pc("cotb"), p=torch.tensor(np.stack([m["p"] for m in meta]), device=dev),
             wS=pc("rmsS") ** -2 / pc("n"), wF=pc("rmsF") ** -2 / pc("n"))
    return T, C, meta


def os_targets(meta, dev):
    js = json.load(open(os.path.join(D.BASE, "os_dispersion.json")))
    out = []
    for m in meta:
        r = js[m["case"]]
        if "error" in r:
            raise RuntimeError(f"no OS dispersion for {m['case']}: {r['error']}")
        out.append(dict(omega=torch.tensor(r["omega"], dtype=torch.float64, device=dev),
                        k=torch.tensor(np.array(r["k_re"]) + 1j * np.array(r["k_im"]), dtype=torch.complex128, device=dev),
                        Re=m["Re"], We=m["We"], cotb=m["cotb"], D_os=r["D_os"]))
    return out


def lin_double(L):
    return {k: v.double() for k, v in L.items()}


def lin_loss(model, C, tgs, sigma):
    """Batched over cases: mean over cases of [mean r_j^2 + r_D^2]; tgs = stack_targets(tg)."""
    L = {k: v.double()[:, None] for k, v in model.linear(C["p"], C["Re"], C["cotb"]).items()}
    r, rD = dispersion_residual(L, tgs, sigma)
    return (r ** 2).mean(-1).mean() + (rD ** 2).mean()


def roots(model, C, tg, sigma, i, it=8):
    """Newton from the OS roots onto the augmented relation; returns growth and k_r errors (B.8 scaling)."""
    with torch.enable_grad():
        L = {k: v.detach() for k, v in lin_double(model.linear(C["p"][i], C["Re"][i], C["cotb"][i])).items()}
    t = tg[i]; k = t["k"].clone()
    for _ in range(it):
        eps = 1e-6 * k.abs()
        F0 = disp_F(k, t["omega"], t["Re"], t["We"], t["cotb"], L, sigma)
        dF = (disp_F(k + eps, t["omega"], t["Re"], t["We"], t["cotb"], L, sigma)
              - disp_F(k - eps, t["omega"], t["Re"], t["We"], t["cotb"], L, sigma)) / (2 * eps)
        k = k - F0 / dF
    ki = (k.imag - t["k"].imag).abs().max() / t["k"].imag.abs().max()
    kr = ((k.real - t["k"].real) / t["k"].real).abs().max()
    Dg = float((3.6 - 3 * t["cotb"] / t["Re"] - L["b"] - 3 * L["c"]) / (3 / t["Re"] + L["a"]))
    return float(ki), float(kr), {n: float(L[n]) for n in ("a", "b", "c", "d", "kap", "g1", "g2", "g3", "Fh", "Fq")}, Dg


def evaluate(model, T, C, meta, chunk=1 << 17):
    """Per-case NRMSE of S (rms-normalised) and of the flux part against dF_x; min discriminant."""
    n = len(meta); se = torch.zeros(n, dtype=torch.float64, device=T["h"].device); sf = torch.zeros_like(se)
    dmin = torch.full((n,), float("inf"), dtype=torch.float64, device=T["h"].device)   # amin with include_self
    for s in range(0, len(T["h"]), chunk):
        sl = slice(s, s + chunk); i = T["idx"][sl]
        with torch.enable_grad():
            S, fl = model(T["h"][sl], T["q"][sl], T["h_x"][sl], T["q_x"][sl], T["h_xx"][sl], C["p"][i], C["Re"][i], C["cotb"][i])
            disc = model.discriminant(T["h"][sl], T["q"][sl], C["p"][i], C["Re"][i], C["cotb"][i])
        se.index_add_(0, i, ((S.detach() - T["S"][sl]) ** 2).double())
        sf.index_add_(0, i, ((fl.detach() - T["dF_x"][sl]) ** 2).double())
        dmin.index_reduce_(0, i, disc.detach().double(), "amin", include_self=True)
    out = {}
    for j, m in enumerate(meta):
        out[m["case"]] = dict(nrmse=float(torch.sqrt(se[j] / m["n"]) / m["rmsS"]),
                              flux_vs_dFx=float(torch.sqrt(sf[j] / m["n"]) / m["rmsF"]), min_disc=float(dmin[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lam_lin", type=float, default=1.0)
    ap.add_argument("--lam_F", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1 << 15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--no_g3", action="store_true")
    ap.add_argument("--x0", type=float, default=50.0)
    ap.add_argument("--t0", type=float, default=300.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--cases", nargs="*", help="training cases (default: development split)")
    ap.add_argument("--heldout", nargs="*", help="evaluation cases (default: held-out split)")
    ap.add_argument("--lin_cases", choices=["train", "os"], default="os",
                    help="where L2 is imposed: the training cases only, or every case of os_dispersion.json "
                         "(OS case table) except those with the parameters of an evaluation case")
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_minutes", type=float, default=0, help="stop training after this wall time (0: off)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.device == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    dev = torch.device(a.device)
    run = os.path.join(D.BASE, "runs", a.name); os.makedirs(run, exist_ok=True)
    logf = open(os.path.join(run, "log.txt"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    dcases, hcases = D.split()
    dcases = a.cases or dcases; hcases = a.heldout if a.heldout is not None else hcases
    t0 = time.time()
    T, C, meta = build(dcases, a, dev)
    TH, CH, metaH = build(hcases, a, dev) if hcases else (None, None, [])
    tg, tgH = os_targets(meta, dev), os_targets(metaH, dev)
    if a.lin_cases == "os":
        js = json.load(open(os.path.join(D.BASE, "os_dispersion.json")))
        key = lambda Re, Ca, bdeg: (round(Re, 2), round(Ca, 5), round(bdeg, 1))      # cache attrs are float32
        hold = {key(m["Re"], m["Ca"], math.degrees(m["beta"])) for m in metaH}
        lmeta = []
        for c, r in sorted(js.items()):
            if "error" in r or key(r["Re"], r["Ca"], r["beta_deg"]) in hold:
                continue
            b = math.radians(r["beta_deg"])
            lmeta.append(dict(case=c, Re=r["Re"], Ca=r["Ca"], beta=b, cotb=1 / math.tan(b), We=r["Re"] * r["Ca"],
                              p=D.pfeat(r["Re"], r["Ca"], b)))
        tgL = os_targets(lmeta, dev)
        CL = dict(Re=torch.tensor([m["Re"] for m in lmeta], dtype=torch.float32, device=dev),
                  cotb=torch.tensor([m["cotb"] for m in lmeta], dtype=torch.float32, device=dev),
                  p=torch.tensor(np.stack([m["p"] for m in lmeta]), device=dev))
    else:
        tgL, CL = tg, C
    tgs = stack_targets(tgL)
    log(f"L2 imposed at {len(tgL)} parameter points ({a.lin_cases})")
    N = len(T["h"])
    log(f"{a.name}: {len(dcases)} training cases, {N} points; {len(hcases)} eval cases; device {dev}, "
        f"threads {torch.get_num_threads()}; data {time.time() - t0:.0f}s")
    json.dump(dict(vars(a), train_cases=dcases, eval_cases=hcases, n_points=N), open(os.path.join(run, "config.json"), "w"), indent=1)
    model = StructuredV1(a.width, a.depth, use_g3=not a.no_g3).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    nb = math.ceil(N / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb, pct_start=0.05)
    ncase = len(meta); te = time.time(); hist = []
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(N, device=dev); acc = np.zeros(3)
        for b in range(nb):
            sl = perm[b * a.batch:(b + 1) * a.batch]; i = T["idx"][sl]
            S, fl = model(T["h"][sl], T["q"][sl], T["h_x"][sl], T["q_x"][sl], T["h_xx"][sl], C["p"][i], C["Re"][i], C["cotb"][i])
            scale = N / (ncase * len(sl))
            lS = (C["wS"][i] * (S - T["S"][sl]) ** 2).sum() * scale
            loss = lS
            lL = lin_loss(model, CL, tgs, a.sigma) if a.lam_lin > 0 else torch.zeros((), device=dev)
            loss = loss + a.lam_lin * lL
            lF = (C["wF"][i] * (fl - T["dF_x"][sl]) ** 2).sum() * scale if a.lam_F > 0 else torch.zeros((), device=dev)
            loss = loss + a.lam_F * lF
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            acc += [lS.item(), lL.item(), lF.item()]
        acc /= nb
        msg = f"ep {ep:4d}  S {acc[0]:.4f}  lin {acc[1]:.2e}  F {acc[2]:.4f}  {(time.time() - te) / ep:.1f}s/ep"
        if ep % a.val_every == 0 or ep == a.epochs:
            ev = evaluate(model, TH, CH, metaH) if metaH else {}
            msg += f"  | eval NRMSE mean {np.mean([v['nrmse'] for v in ev.values()]):.4f}" if ev else ""
        log(msg); hist.append(dict(ep=ep, S=acc[0], lin=acc[1], F=acc[2]))
        if a.max_minutes and (time.time() - te) > 60 * a.max_minutes:
            log(f"stopping at epoch {ep}: max_minutes reached"); break
    torch.save(model.state_dict(), os.path.join(run, "model.pt"))
    # final metrics
    M = dict(train=evaluate(model, T, C, meta), eval=evaluate(model, TH, CH, metaH) if metaH else {},
             s_per_epoch=(time.time() - te) / max(1, len(hist)), history=hist)
    for part, mt, cc, tt in (("train", meta, C, tg), ("eval", metaH, CH, tgH)):
        for j, m in enumerate(mt):
            ki, kr, L, Dg = roots(model, cc, tt, a.sigma, j)
            M[part][m["case"]].update(lin_err_ki=ki, lin_err_kr=kr, D=Dg, D_os=tt[j]["D_os"], **L)
    json.dump(M, open(os.path.join(run, "metrics.json"), "w"), indent=1)
    log(f"{'case':24s} {'part':5s} {'NRMSE':>6s} {'fluxF':>6s} {'mindisc':>8s} {'ki_err':>6s} {'kr_err':>6s} {'D/D_os':>6s}   a       b       c       d")
    for part in ("train", "eval"):
        for c, v in M[part].items():
            log(f"{c:24s} {part:5s} {v['nrmse']:6.3f} {v['flux_vs_dFx']:6.3f} {v['min_disc']:8.2e} {v['lin_err_ki']:6.3f} "
                f"{v['lin_err_kr']:6.3f} {v['D'] / v['D_os']:6.3f}  {v['a']:+.4f} {v['b']:+.3f} {v['c']:+.3f} {v['d']:+.3f}")
    log(f"done: {time.time() - t0:.0f}s total, {M['s_per_epoch']:.1f} s/epoch")


if __name__ == "__main__":
    main()
