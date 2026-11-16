"""Orr-Sommerfeld solver for a film falling down an inclined plane (passive gas).

Project nondimensionalisation (Nusselt scaling, docs/equations.md):
    U(z) = 3z - 3z^2/2,  U_s = 3/2,  U'' = -3,  Re = u_N h_N / nu,  Ca = mu u_N / sigma.

Perturbation: stream function psi = phi(z) exp[ik(x - ct)], u' = phi', w' = -ik phi,
interface eta = eta_hat exp[ik(x - ct)].  Unknowns x = (phi(z_j), eta_hat), A x = c B x.

    bulk        phi'''' - 2k^2 phi'' + k^4 phi = ik Re [(U - c)(phi'' - k^2 phi) - U'' phi]
    wall z=0    phi = phi' = 0
    kinematic   (c - U_s) eta_hat = phi(1)
    tangential  phi'' + k^2 phi + U'' eta_hat = 0                       (z = 1)
    normal      phi''' - 3k^2 phi' - ik Re (U_s - c) phi' - ik (3 cot b + k^2/Ca) eta_hat = 0

Temporal problem: real k -> complex c; growth rate = k c_i.
Spatial problem: real omega -> complex k with k c(k) = omega; spatial growth = -k_i.

Spurious eigenvalues are removed by keeping only eigenvalues that agree between two
resolutions N and N + dN (see `physical_eigs`).
"""
import numpy as np
import scipy.linalg as la

U_S = 1.5
UPP = -3.0


def cheb(N):
    """Chebyshev differentiation matrix (Trefethen), nodes x_j = cos(pi j / N)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2.0, np.ones(N - 1), 2.0] * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1.0 / c) / (X - X.T + np.eye(N + 1))
    return D - np.diag(D.sum(1)), x


_CACHE = {}


def _grid(N):
    if N not in _CACHE:
        D, x = cheb(N)
        z = (x + 1.0) / 2.0          # z[0] = 1 (surface), z[N] = 0 (wall)
        D = 2.0 * D
        D2 = D @ D
        _CACHE[N] = (z, D, D2, D2 @ D)
    return _CACHE[N]


def operators(k, Re, Ca, cotb, N=80):
    """Matrices A, B of the generalised eigenproblem A x = c B x (k may be complex)."""
    z, D, D2, D3 = _grid(N)
    I = np.eye(N + 1)
    U = 3.0 * z - 1.5 * z ** 2
    n = N + 2
    A = np.zeros((n, n), complex)
    B = np.zeros((n, n), complex)
    L = D2 - k * k * I
    A[:N + 1, :N + 1] = L @ L - 1j * k * Re * (np.diag(U) @ L - UPP * I)
    B[:N + 1, :N + 1] = -1j * k * Re * L
    # wall: phi(0) = 0, phi'(0) = 0   (rows N, N-1 replaced)
    A[N] = 0; B[N] = 0; A[N, N] = 1.0
    A[N - 1] = 0; B[N - 1] = 0; A[N - 1, :N + 1] = D[N]
    # tangential stress (row 0)
    A[0] = 0; B[0] = 0
    A[0, :N + 1] = D2[0] + k * k * I[0]; A[0, N + 1] = UPP
    # normal stress (row 1)
    A[1] = 0; B[1] = 0
    A[1, :N + 1] = D3[0] - 3 * k * k * D[0] - 1j * k * Re * U_S * D[0]
    A[1, N + 1] = -1j * k * (3.0 * cotb + k * k / Ca)
    B[1, :N + 1] = -1j * k * Re * D[0]
    # kinematic: phi(1) + U_s eta - c eta = 0
    A[N + 1, 0] = 1.0; A[N + 1, N + 1] = U_S; B[N + 1, N + 1] = 1.0
    # row balancing: leaves the eigenvalues unchanged but cuts the QZ round-off by orders
    # of magnitude (the D^4 rows are O(N^8), the BC rows much smaller).
    sc = 1.0 / np.maximum(np.abs(A).max(1), np.abs(B).max(1))
    return A * sc[:, None], B * sc[:, None]


def _eigs(k, Re, Ca, cotb, N, cmax=50.0, vectors=False):
    A, B = operators(k, Re, Ca, cotb, N)
    if vectors:
        w, V = la.eig(A, B)
    else:
        w, V = la.eigvals(A, B), None
    ok = np.isfinite(w) & (np.abs(w) < cmax)
    return (w[ok], V[:, ok]) if vectors else (w[ok], None)


def physical_eigs(k, Re, Ca, cotb, N=48, dN=16, rtol=1e-5):
    """Eigenvalues c of resolution N that are reproduced at N + dN to rtol."""
    w1, _ = _eigs(k, Re, Ca, cotb, N)
    w2, _ = _eigs(k, Re, Ca, cotb, N + dN)
    if len(w1) == 0 or len(w2) == 0:
        return w1[:0]
    d = np.abs(w1[:, None] - w2[None, :]).min(1)
    return w1[d <= rtol * np.maximum(1.0, np.abs(w1))]


def temporal_colloc(k, Re, Ca, cotb, N=48, dN=16, c_guess=None, rtol=1e-5):
    """Collocation only: c of the least stable physical mode (or the one closest to c_guess)."""
    w = physical_eigs(k, Re, Ca, cotb, N, dN, rtol)
    if len(w) == 0:
        raise RuntimeError(f"no converged eigenvalue at k={k}, Re={Re}, Ca={Ca}")
    if c_guess is None:
        return w[np.argmax(w.imag)]
    return w[np.argmin(np.abs(w - c_guess))]


# ---------------------------------------------------------------- shooting (polish)
def _rhs(z, y, k, Re, c):
    U = 3.0 * z - 1.5 * z * z
    p, p1, p2, p3 = y
    p4 = 2 * k * k * p2 - k ** 4 * p + 1j * k * Re * ((U - c) * (p2 - k * k * p) - UPP * p)
    return [p1, p2, p3, p4]


def surface_det(c, k, Re, Ca, cotb, rtol=1e-12):
    """Determinant of the surface conditions for the two wall-satisfying solutions,
    multiplied by (c - U_s) so that it has no pole.  Zero <=> c is an eigenvalue."""
    from scipy.integrate import solve_ivp
    cols = []
    for y0 in ([0, 0, 1, 0], [0, 0, 0, 1]):
        sol = solve_ivp(_rhs, (0.0, 1.0), np.array(y0, complex), method="DOP853",
                        rtol=rtol, atol=1e-14, args=(k, Re, c))
        p, p1, p2, p3 = sol.y[:, -1]
        # eta (c - U_s) = phi(1); write each condition times (c - U_s)
        T = (p2 + k * k * p) * (c - U_S) + UPP * p
        Nn = (p3 - 3 * k * k * p1 - 1j * k * Re * (U_S - c) * p1) * (c - U_S) \
            - 1j * k * (3 * cotb + k * k / Ca) * p
        cols.append((T, Nn))
    (T1, N1), (T2, N2) = cols
    return T1 * N2 - T2 * N1


def polish(c0, k, Re, Ca, cotb, tol=1e-11, maxit=30, stag_tol=1e-7):
    """Secant refinement of an eigenvalue c on the shooting determinant."""
    ca, cb = complex(c0), complex(c0) * (1 + 1e-7) + 1e-9
    fa, fb = surface_det(ca, k, Re, Ca, cotb), surface_det(cb, k, Re, Ca, cotb)
    steps = []
    for _ in range(maxit):
        if fb == fa:
            break
        cn = cb - fb * (cb - ca) / (fb - fa)
        ca, fa = cb, fb
        cb, fb = cn, surface_det(cn, k, Re, Ca, cotb)
        steps.append(abs(cb - ca))
        if steps[-1] < tol * max(1.0, abs(cb)):
            return cb
    # at large k^2/Ca the determinant is badly scaled and the secant stagnates at round-off;
    # accept if the last steps are all below stag_tol (still far better than collocation)
    if len(steps) >= 3 and max(steps[-3:]) < stag_tol * max(1.0, abs(cb)):
        return cb
    raise RuntimeError(f"polish did not converge at k={k}, Re={Re} (c0={c0}, last={cb})")


def temporal(k, Re, Ca, cotb, N=48, dN=16, c_guess=None, rtol=1e-5, do_polish=True):
    """Complex c of the least stable physical mode (or the one closest to c_guess):
    collocation with spurious-mode filter, then shooting polish (~1e-10)."""
    c = temporal_colloc(k, Re, Ca, cotb, N, dN, c_guess, rtol)
    if do_polish:
        cp = polish(c, k, Re, Ca, cotb)
        if abs(cp - c) > 1e-3 * max(1.0, abs(c)):
            raise RuntimeError(f"polish moved c from {c} to {cp} at k={k}, Re={Re}: mode jump?")
        c = cp
    return c


def eigenfunction(k, Re, Ca, cotb, N=48, c_guess=None):
    """(c, z, phi) of the selected mode, normalised so that eta_hat = 1. z[0]=1, z[-1]=0."""
    c0 = temporal(k, Re, Ca, cotb, N, c_guess=c_guess)
    w, V = _eigs(k, Re, Ca, cotb, N, vectors=True)
    j = np.argmin(np.abs(w - c0))
    v = V[:, j] / V[-1, j]
    z = _grid(N)[0]
    return w[j], z, v[:-1]


def spatial(omega, Re, Ca, cotb, k0, c0=None, N=48, tol=1e-9, maxit=40):
    """Complex k with k c(k) = omega for real omega, Newton with numerical derivative.

    k0: starting complex k (e.g. Gaster estimate); c0: guess for the mode's c to follow.
    Returns (k, c).  Spatial growth rate = -k.imag, phase speed = omega / k.real.
    """
    k = complex(k0)
    c = temporal(k, Re, Ca, cotb, N, c_guess=c0)

    def f(kk, cg):
        cc = temporal(kk, Re, Ca, cotb, N, c_guess=cg)
        return kk * cc - omega, cc

    for _ in range(maxit):
        F, c = f(k, c)
        h = 1e-6 * max(1.0, abs(k))
        F2, _ = f(k + h, c)
        dk = -F * h / (F2 - F)
        k += dk
        if abs(dk) < tol * max(1.0, abs(k)):
            F, c = f(k, c)
            return k, c
    raise RuntimeError(f"spatial Newton did not converge at omega={omega} (last k={k})")


def spatial_curve(omegas, Re, Ca, cotb, N=48):
    """Spatial branch k(omega) for increasing omegas, continued from a temporal/Gaster start."""
    omegas = np.asarray(omegas, float)
    # start: real k with k c_r(k) ~ omega_0 at small k (c_r ~ 3), Gaster for k_i
    w0 = omegas[0]
    kr = w0 / 3.0
    for _ in range(30):                  # fixed point on k_r = omega / c_r(k_r)
        kr = w0 / temporal(kr, Re, Ca, cotb, N).real
    c = temporal(kr, Re, Ca, cotb, N)
    dk = 1e-4 * kr
    cg = ((kr + dk) * temporal(kr + dk, Re, Ca, cotb, N, c_guess=c).real
          - (kr - dk) * temporal(kr - dk, Re, Ca, cotb, N, c_guess=c).real) / (2 * dk)
    k = complex(kr, -kr * c.imag / cg)
    out_k, out_c = [], []
    for om in omegas:
        k, c = spatial(om, Re, Ca, cotb, k0=k, c0=c, N=N)
        out_k.append(k); out_c.append(c)
    return np.array(out_k), np.array(out_c)


# ---------------------------------------------------------------- linearised Lavalle ROM
def lavalle_temporal(k, Re, Ca, cotb):
    """Least stable complex c of the linearised first-order Lavalle (2015) model.

    Quadratic in c:  -ik c^2 + (2ik + 3/Re + 2k^2/Re) c + ik G - 9/Re + ik^3/We = 0,
    G = -0.6 + 3 cot b / Re,  We = Re Ca.
    """
    We = Re * Ca
    G = -0.6 + 3.0 * cotb / Re
    r = np.roots([-1j * k, 2j * k + 3 / Re + 2 * k * k / Re, 1j * k * G - 9 / Re + 1j * k ** 3 / We])
    return r[np.argmax(r.imag)]


def lavalle_spatial(omega, Re, Ca, cotb):
    """Complex k of the linearised Lavalle model for real omega (the downstream, k_r > 0 branch
    that tends to the kinematic wave k = omega/3 as omega -> 0).

    Substituting c = omega/k into the quadratic and multiplying by k gives a quartic in k:
      (i/We) k^4 + (2 omega/Re + i G) k^2 + (2i omega - 9/Re) k + (3 omega/Re - i omega^2) = 0.
    """
    We = Re * Ca
    G = -0.6 + 3.0 * cotb / Re
    r = np.roots([1j / We, 0.0, 2 * omega / Re + 1j * G, 2j * omega - 9 / Re,
                  3 * omega / Re - 1j * omega ** 2])
    return r[np.argmin(np.abs(r - omega / 3.0))]


def follow(k_target, Re, Ca, cotb, nsteps=24, N=48):
    """c of the least stable real-k mode at Re(k_target), continued to complex k_target."""
    kr = k_target.real
    c = temporal(kr, Re, Ca, cotb, N)
    for s in np.linspace(0.0, 1.0, nsteps + 1)[1:]:
        c = temporal(kr + s * (k_target - kr), Re, Ca, cotb, N, c_guess=c)
    return c


def saddle_point(V, k0, Re, Ca, cotb, N=48, h=1e-5, tol=1e-10, maxit=40):
    """Saddle point d omega/dk = V of omega(k) = k c(k) (absolute/convective analysis on the
    ray x/t = V).  Returns (k_s, omega(k_s), c).  The growth on the ray is Im[omega - k V]."""
    k = complex(k0)
    c = follow(k, Re, Ca, cotb, N=N)
    w = lambda kk, cg: kk * temporal(kk, Re, Ca, cotb, N, c_guess=cg)
    for _ in range(maxit):
        c = temporal(k, Re, Ca, cotb, N, c_guess=c)
        w0, wp, wm = k * c, w(k + h, c), w(k - h, c)
        dk = -((wp - wm) / (2 * h) - V) / ((wp - 2 * w0 + wm) / h ** 2)
        k += dk
        if abs(dk) < tol:
            c = temporal(k, Re, Ca, cotb, N, c_guess=c)
            return k, k * c, c
    raise RuntimeError(f"saddle Newton did not converge (V={V}, k0={k0}, last k={k})")
