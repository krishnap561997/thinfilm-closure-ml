Scripts behind ordering-matrix.md (sympy 1.14). Run in this order:
  python rowb1.py   # Re ~ eps^-1: q-gauge expansion, u1, tau_w, Galerkin solvability, consistency check
  python rowb23.py  # Re ~ eps^-2, eps^-3: profile-family closures (A2, A3, D, u_s) + non-hydrostatic pressure
  python checks.py  # Lavalle/Benney limit of the row-1 tau_w closure; Jacobian eigenvalues at sample s
  python scan.py    # hyperbolicity band in s for the 3-moment pressureless system
  python nh2.py     # non-hydrostatic source in the second-moment equation
common.py holds the derivative bookkeeping (h0..h8 = h and its x-derivatives, likewise q, s).
  python richard_compare.py  # comparison with the Richard/Teshukov enstrophy structure
  python richard2019_check.py # checks against Richard et al. (2019): eqs 32-33, 59, and EIM(22) == row-1 model
  python richard2016_check.py # checks against Richard et al. (2016): skewness, tau_w(phi), relaxation rates
