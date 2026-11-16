# Orr–Sommerfeld solver (plan `docs/plans/structured-closure-os-plan.md`, Part I)

Linear stability of the Nusselt film (passive gas), in the project's Nusselt scaling
(U = 3z − 3z²/2, Re = u_N h_N/ν, Ca = μu_N/σ). Pure numpy/scipy; every call takes milliseconds.

| File | Content |
|---|---|
| `os_solver.py` | `temporal(k, Re, Ca, cotb)` → complex c (growth = k c_i); `spatial`, `spatial_curve` → complex k(ω) (spatial growth −k_i); `eigenfunction` (φ(z) with η̂ = 1); `saddle_point` (dω/dk = V); `lavalle_temporal`, `lavalle_spatial` (linearised Lavalle 2015 model, analytic) |
| `verify.py` | Task B.3: Benney limit, Re_c, Brevdo et al. (1999) saddle points from Kalliadasis et al. Table 7.1, convergence, plan §0 table. Writes `$THINFILM_STORE/analysis/orr_sommerfeld/verification.json` |

Method: Chebyshev collocation (row-balanced generalised eigenproblem in (φ, η̂)), spurious modes
removed by agreement between N = 48 and 64, then each selected eigenvalue is polished by shooting
(DOP853 from the wall, secant on the 2×2 surface determinant). The two methods agree to ~1e-7, and
the polished value is independent of N to 1e-15.

Usage:
```bash
source env.sh
python analysis/orr_sommerfeld/verify.py          # ~5 s, prints PASS/FAIL per check
python -c "import sys; sys.path.insert(0,'analysis/orr_sommerfeld'); from os_solver import temporal; print(temporal(0.3,120,0.01,1.0))"
```

Mode selection: without `c_guess` the least stable converged mode is returned. With complex k
(spatial or saddle problems) the mode is followed by continuation with `c_guess`; check for
mode jumps when you change parameters a lot (`temporal` raises if the polish moves c by > 1e-3).

Converting from Kalliadasis et al. (2012) Nusselt scaling (velocity scale 3u_N): k is unchanged;
ω, c and V are 3× ours; We_book = Γ/(3Re)^{2/3} = 1/(3 Ca); Re and Ct = cot β are the same.
