"""Linear constraint L2 of the structured closure: match the OS spatial dispersion over the forced band.

Linearised augmented ROM about h = q = 1 with eta, xi = (omega/k) eta, multiplied by k (the B.8
quartic, analysis/orr_sommerfeld/linear_targets.py k_aug, generalised):
    Lavalle:  (i/We) k^4 + (2 omega/Re + iG) k^2 + (2i omega - 9/Re) k + (3 omega/Re - i omega^2)
    closure:  i Fh k^2 + i Fq omega k  +  a (omega - 3k)  -  kappa Gf(k) (i g1 k^2 + i g2 omega k - g3 k^3)
with G = -0.6 + 3 cot b/Re, a = kappa - 3/Re.  The flux part (Fh, Fq) is differentiated by the ROM's
finite-volume flux (factor 1); the relaxation gradients are D_G(sigma), Gf(k) = exp(-k^2 sigma^2/2).
For sigma = 0 this is k_aug with b = Fh - kappa g1, c = Fq - kappa g2, d = -kappa g3.

Residual: at each OS root k_j, the first-order wavenumber error dk = -F(k_j)/F'(k_j) of the
augmented relation, scaled as in B.8: Re dk / k_r,j and Im dk / max|k_i|.  Plus the long-wave
coefficient (D_aug - D_OS)/D_OS, D_aug = (3.6 - 3 cot b/Re - b - 3c)/(3/Re + a).
"""
import torch


def disp_F(k, om, Re, We, cotb, L, sigma):
    G = -0.6 + 3.0 * cotb / Re
    lav = 1j / We * k ** 4 + (2 * om / Re + 1j * G) * k ** 2 + (2j * om - 9 / Re) * k + (3 * om / Re - 1j * om ** 2)
    Gf = torch.exp(-(k * sigma) ** 2 / 2)
    clo = 1j * L["Fh"] * k ** 2 + 1j * L["Fq"] * om * k + L["a"] * (om - 3 * k) \
        - L["kap"] * Gf * (1j * L["g1"] * k ** 2 + 1j * L["g2"] * om * k - L["g3"] * k ** 3)
    if "bf" in L:   # fully filtered source closure (model.FlatExactMLP): b, c, d all see D_G(sigma)
        clo = clo + Gf * (1j * L["bf"] * k ** 2 + 1j * L["cf"] * om * k - L["df"] * k ** 3)
    return lav + clo


def dispersion_residual(L, tgt, sigma):
    """L: model.linear(...) dict; tgt: one case (omega, k: (20,); Re, We, cotb, D_os floats) or a batch from
    stack_targets (L entries then of shape (n, 1)).  Returns (residuals, D residual)."""
    om, k = tgt["omega"], tgt["k"]
    Re, We, cotb = tgt["Re"], tgt["We"], tgt["cotb"]
    eps = 1e-5 * k.abs()
    F0 = disp_F(k, om, Re, We, cotb, L, sigma)
    dF = (disp_F(k + eps, om, Re, We, cotb, L, sigma) - disp_F(k - eps, om, Re, We, cotb, L, sigma)) / (2 * eps)
    dk = -F0 / dF
    r = torch.cat([dk.real / k.real, dk.imag / k.imag.abs().amax(-1, keepdim=True)], -1)
    D = (3.6 - 3 * cotb / Re - L["b"] - 3 * L["c"]) / (3 / Re + L["a"])
    return r, (D - tgt["D_os"]) / tgt["D_os"]


def stack_targets(tg):
    """List of per-case targets -> one batched target: omega, k (n, 20); Re, We, cotb, D_os (n, 1)."""
    col = lambda key: torch.tensor([[t[key]] for t in tg], dtype=torch.float64, device=tg[0]["omega"].device)
    return dict(omega=torch.stack([t["omega"] for t in tg]), k=torch.stack([t["k"] for t in tg]),
                Re=col("Re"), We=col("We"), cotb=col("cotb"), D_os=col("D_os"))
