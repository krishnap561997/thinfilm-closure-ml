"""Structured closure, variant V1 (docs/plans/structured-closure-os-plan.md section 3.2).

    d_t q + d_x[ Phi q^2/h + P_Lav(h) + dP(h) ] = kappa (Q - q)/h^2 + (1/We) h h_xxx + (2/Re) q_xx
    Q = h^3 (1 + g1 h_x + g2 q_x + g3 h_xx)

so the correction to Lavalle (the ROM adds -S) is
    S = d_x[(Phi - 1) q^2/h + dP] - [kappa (Q - q) - (3/Re)(h^3 - q)] / h^2.

All coefficient functions depend on (h; p), p = (ln Re, ln Ca, beta) normalised (data.pfeat):
    Phi   = 1 + softplus(N_Phi),  N_Phi non-decreasing in h (non-negative weights on the h path)
    P_tot' = P_Lav'(h) exp(N_s) > 0,  dP' = P_tot' - P_Lav'      (so V1 is hyperbolic for every state)
    kappa = (3/Re) exp(N_kappa)
    g1, g2, g3 free.
Lavalle is the zero of every network at initialisation (Phi - 1 ~ 2.5e-3).
P_Lav(h) = 2 Lambda^2 h^5/225 + (3 cot b / (2 Re)) h^2, Lambda = 3 (docs/equations.md sec. 3).

The uniform film (h const, q = h^3, zero gradients) gives S = 0 for any weights.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI0 = -6.0          # softplus(-6) = 2.5e-3: Phi starts next to 1


class FreeNet(nn.Module):
    def __init__(self, width=32, depth=2, nin=4):
        super().__init__()
        layers, n = [], nin
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.Tanh()]; n = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(n, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h, p):
        return self.out(self.body(torch.cat([h[:, None], p], 1)))[:, 0]


class MonoNet(nn.Module):
    """Output non-decreasing in h: h enters through non-negative weights, all later weights are
    non-negative and the activations are monotone; p enters the first layer freely."""
    def __init__(self, width=32, depth=2, npar=3, bias0=PHI0):
        super().__init__()
        self.wh = nn.Parameter(torch.randn(width) * 0.5 - 1.0)        # softplus -> >= 0
        self.wp = nn.Linear(npar, width)
        self.hidden = nn.ParameterList([nn.Parameter(torch.randn(width, width) * 0.3 - 2.0) for _ in range(depth - 1)])
        self.hb = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(depth - 1)])
        self.wo = nn.Parameter(torch.full((width,), -6.0))
        self.bo = nn.Parameter(torch.tensor(bias0))

    def forward(self, h, p):
        z = torch.tanh(F.softplus(self.wh)[None, :] * h[:, None] + self.wp(p))
        for W, b in zip(self.hidden, self.hb):
            z = torch.tanh(z @ F.softplus(W).T + b)
        return z @ F.softplus(self.wo) + self.bo


def plav_h(h, Re, cotb):
    return 0.4 * h ** 4 + 3.0 * cotb / Re * h


class StructuredV1(nn.Module):
    def __init__(self, width=32, depth=2, use_g3=True):
        super().__init__()
        self.phi = MonoNet(width, depth)
        self.ns = FreeNet(width, depth)
        self.nk = FreeNet(width, depth)
        self.g1, self.g2 = FreeNet(width, depth), FreeNet(width, depth)
        self.g3 = FreeNet(width, depth) if use_g3 else None

    def coeffs(self, h, p, Re, cotb):
        """Coefficient functions and Phi_h (autograd) at states h with features p."""
        h = h.detach().requires_grad_(True) if not h.requires_grad else h
        Phi = 1.0 + F.softplus(self.phi(h, p))
        Phi_h = torch.autograd.grad(Phi.sum(), h, create_graph=True)[0]
        pl = plav_h(h, Re, cotb)
        dPh = pl * (torch.exp(self.ns(h, p)) - 1.0)
        kap = 3.0 / Re * torch.exp(self.nk(h, p))
        g1, g2 = self.g1(h, p), self.g2(h, p)
        g3 = self.g3(h, p) if self.g3 is not None else torch.zeros_like(h)
        return dict(Phi=Phi, Phi_h=Phi_h, dPh=dPh, kap=kap, g1=g1, g2=g2, g3=g3, plav_h=pl)

    def forward(self, h, q, hx, qx, hxx, p, Re, cotb):
        """Returns S and its flux part d_x F (chain rule with the filtered gradients)."""
        c = self.coeffs(h, p, Re, cotb)
        Fh = c["Phi_h"] * q * q / h - (c["Phi"] - 1.0) * q * q / (h * h) + c["dPh"]
        Fq = 2.0 * (c["Phi"] - 1.0) * q / h
        flux = Fh * hx + Fq * qx
        Q = h ** 3 * (1.0 + c["g1"] * hx + c["g2"] * qx + c["g3"] * hxx)
        relax = (c["kap"] * (Q - q) - 3.0 / Re * (h ** 3 - q)) / (h * h)
        return flux - relax, flux

    def linear(self, p, Re, cotb):
        """Linear coefficients at the Nusselt state h = q = 1.  p: (n, 3) or (3,); Re, cotb: (n,) or scalars.
        Returns tensors of shape (n,) (or scalars for a single case)."""
        single = p.dim() == 1
        P = p[None, :] if single else p
        h = torch.ones(P.shape[0], dtype=P.dtype, device=P.device, requires_grad=True)
        c = self.coeffs(h, P, Re, cotb)
        Fh = c["Phi_h"] - (c["Phi"] - 1.0) + c["dPh"]
        Fq = 2.0 * (c["Phi"] - 1.0)
        out = dict(Fh=Fh, Fq=Fq, kap=c["kap"], g1=c["g1"], g2=c["g2"], g3=c["g3"], a=c["kap"] - 3.0 / Re,
                   b=Fh - c["kap"] * c["g1"], c=Fq - c["kap"] * c["g2"], d=-c["kap"] * c["g3"])
        return {k: v[0] for k, v in out.items()} if single else out

    def discriminant(self, h, q, p, Re, cotb):
        """Flux-Jacobian discriminant 4u^2[Phi(Phi-1) + h Phi_h] + 4 P_tot' (>= 0 for V1)."""
        c = self.coeffs(h, p, Re, cotb)
        u = q / h
        return 4 * u * u * (c["Phi"] * (c["Phi"] - 1) + h * c["Phi_h"]) + 4 * (c["plav_h"] + c["dPh"])


# ---------------------------------------------------------------------------------------------------------
class FlatExactMLP(nn.Module):
    """Flat-film-exact source closure (progress tracker "Next options", option 2):
        S = N(h, q, h_x, q_x, h_xx, Re, Ca, angle) - N(h, h^3, 0, 0, 0, Re, Ca, angle)
    so S = 0 on every flat film for any weights (S0 = e = 0).  N has exactly the layout of the ROM's MLP class
    (rom/lavalle-cpp/src/ml/ml_closure.cpp): inputs standardised by x_mean/x_std, tanh hidden layers, linear
    output times y_std.  The gradients are D_G(sigma) (zero-sum), all filtered, so in the dispersion relation the
    closure's b, c, d carry the filter factor (losses.disp_F keys bf, cf, df).  The ROM flux stays Lavalle's.
    Parameters are recovered from the features p of data.pfeat (invertible)."""
    NAMES = ["h", "q", "h_x", "q_x", "h_xx", "Re", "Ca", "angle"]

    def __init__(self, width=64, depth=2):
        super().__init__()
        self.register_buffer("xm", torch.zeros(8)); self.register_buffer("xs", torch.ones(8))
        self.register_buffer("ys", torch.ones(()))
        layers, n = [], 8
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.Tanh()]; n = width
        self.body = nn.Sequential(*layers); self.out = nn.Linear(n, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    @staticmethod
    def raw(p):
        return 60.0 * torch.exp(p[:, 0]), 0.01 * torch.exp(1.5 * p[:, 1]), 0.8 + 0.6 * p[:, 2]

    def N(self, h, q, hx, qx, hxx, Re, Ca, ang):
        x = (torch.stack([h, q, hx, qx, hxx, Re, Ca, ang], 1) - self.xm) / self.xs
        return self.out(self.body(x))[:, 0] * self.ys

    def forward(self, h, q, hx, qx, hxx, p, Re, cotb):
        Re_, Ca, ang = self.raw(p)
        z = torch.zeros_like(h)
        S = self.N(h, q, hx, qx, hxx, Re_, Ca, ang) - self.N(h, h ** 3, z, z, z, Re_, Ca, ang)
        return S, torch.zeros_like(S)

    def linear(self, p, Re, cotb):
        single = p.dim() == 1
        P = p[None, :] if single else p
        n = P.shape[0]
        mk = lambda v: torch.full((n,), v, dtype=P.dtype, device=P.device, requires_grad=True)
        h, q, hx, qx, hxx = mk(1.0), mk(1.0), mk(0.0), mk(0.0), mk(0.0)
        S, _ = self.forward(h, q, hx, qx, hxx, P, Re, cotb)
        a, bf, cf, df = torch.autograd.grad(S.sum(), (q, hx, qx, hxx), create_graph=True)
        z = torch.zeros_like(a)
        out = dict(Fh=z, Fq=z, kap=z, g1=z, g2=z, g3=z, a=a, b=bf, c=cf, d=df, bf=bf, cf=cf, df=df)
        return {k: v[0] for k, v in out.items()} if single else out

    def discriminant(self, h, q, p, Re, cotb):
        u = q / h
        return 4 * plav_h(h, Re, cotb) + 0 * u          # flux unchanged: Lavalle (Phi = 1)

    def set_scales(self, X, ys):
        """X: (n, 8) raw inputs of the training data; ys: output scale (rms of S)."""
        self.xm.copy_(X.mean(0)); self.xs.copy_(X.std(0).clamp_min(1e-6)); self.ys.fill_(float(ys))


def load_run(run_dir, dtype=torch.float64):
    """Model of a training run directory (config.json + model.pt), in double precision on CPU."""
    import json, os
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if cfg.get("model", "v1") == "mlp":
        m = FlatExactMLP(cfg.get("mlp_width", 64), cfg.get("mlp_depth", 2))
    else:
        m = StructuredV1(cfg["width"], cfg["depth"], use_g3=not cfg["no_g3"])
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    return m.to(dtype), cfg
