"""Tier-1 library for S = DeltaM_x + DeltaT_w (docs/plans/functional-form-plan.md, section 3.1).

Columns are sympy expressions in (h, q, h_x, q_x, h_xx, q_xx). Flux columns F are expanded
analytically with the chain rule (ddx), so data products are never differentiated numerically.
Each column carries a parameter factor and the row-1 q-gauge theory coefficient
(ordering-matrix section 9; valid for delta = eps Re <~ 1).
Bases:
  raw  T1..T5, F1..F4 as in the plan
  rg   conditioned: T3 -> G = q g / h^2 with g = h q_x - 3 q h_x (theory 1/35); T4 then has theory 0,
       because 1/35 G already carries the -3/35 q^2 h_x / h^2 of T4. Same span as raw.
"""
import numpy as np
import sympy as sp

h, q, hx, qx, hxx, qxx = sp.symbols('h q h_x q_x h_xx q_xx')
VARS = (h, q, hx, qx, hxx, qxx)
FIELDS = ('h', 'q', 'hx', 'qx', 'hxx', 'qxx')


def ddx(F):
    """d/dx of F(h, q, h_x, q_x) by the chain rule."""
    return sp.expand(sp.diff(F, h) * hx + sp.diff(F, q) * qx + sp.diff(F, hx) * hxx + sp.diff(F, qx) * qxx)


g = h * qx - 3 * q * hx
R = sp.Rational
# id: (expression in S, parameter factor, theory coefficient, flux before ddx or None)
_RAW = {
    'T1': ((h ** 3 - q) / h ** 2, '1/Re', R(1, 2), None),
    'T2': (h ** 4 * hx, '1', R(3, 5), None),
    'T3': (q * qx / h, '1', R(1, 35), None),
    'T4': (q ** 2 * hx / h ** 2, '1', R(-3, 35), None),
    'T5': (h * hx, 'cotb/Re', R(-1, 2), None),
    'F1': (None, '1', R(8, 35), (q - h ** 3) * q / h),
    'F2': (None, '1', R(1, 5), (q - h ** 3) * h ** 2),
    'F3': (None, 'Re', R(-2, 735), q ** 2 * qx - 3 * q ** 3 * hx / h),
    'F4': (None, 'cotb', R(1, 35), q * h ** 2 * hx),
}


def _columns(spec):
    return {k: dict(expr=sp.expand(e if e is not None else ddx(F)), factor=f, theory=t, flux=F)
            for k, (e, f, t, F) in spec.items()}


RAW = _columns(_RAW)
_RG ={'T1': _RAW['T1'], 'T2': _RAW['T2'], 'G': (q * g / h ** 2, '1', R(1, 35), None),
       'T4': (_RAW['T4'][0], '1', R(0), None), **{k: _RAW[k] for k in ['T5', 'F1', 'F2', 'F3', 'F4']}}
RG = _columns(_RG)
BASES = {'raw': RAW, 'rg': RG}


def factor_value(name, Re, Ca, beta_deg):
    cot = 0. if np.isclose(beta_deg, 90.) else 1. / np.tan(np.radians(beta_deg))   # tan(90 deg) ~ 1.6e16
    return {'1': 1., '1/Re': 1. / Re, 'cotb/Re': cot / Re, 'Re': Re, 'cotb': cot, '1/We': 1. / (Re * Ca)}[name]


GENERIC_FACTORS = ['1', 'Re', '1/Re', '1/We', 'cotb/Re']


def evaluate(basis, fields):
    """Numeric columns (without parameter factor): dict id -> array, fields: dict of FIELDS arrays."""
    args = [fields[f] for f in FIELDS]
    return {k: np.broadcast_to(sp.lambdify(VARS, c['expr'], 'numpy')(*args), fields['h'].shape)
            for k, c in BASES[basis].items()}


def monomials(basis, coefs):
    """Combined coefficient of each monomial for column coefficients {id: c} (factor included in c)."""
    out = {}
    for k, c in coefs.items():
        for mono, a in sp.expand(BASES[basis][k]['expr']).as_coefficients_dict().items():
            out[mono] = out.get(mono, 0.) + float(a) * c
    return {str(m): v for m, v in out.items()}


def theory_monomials(basis, Re, Ca, beta_deg):
    return monomials(basis, {k: float(c['theory']) * factor_value(c['factor'], Re, Ca, beta_deg)
                             for k, c in BASES[basis].items()})
