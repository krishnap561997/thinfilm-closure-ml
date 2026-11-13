#!/usr/bin/env python
"""Combined monomial coefficients of the per-case Tier-1 fit against row-1 theory (plan section 3.5).

Per-column Tier-1 coefficients are not identifiable on their own (T and F columns overlap and
q ~ h^3), so compare the coefficient of each monomial in (h, q, h_x, q_x, h_xx, q_xx), parameter
factors included. Usage: functional_form_monomials.py --phase0 <phase0.json> --cases <case> ...
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml' / 'functional_forms'))
import library as L  # noqa: E402
from fit_sparse import case_params  # noqa: E402


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--phase0', required=True)
    a.add_argument('--cases', nargs='+', required=True)
    a.add_argument('--window', default='full')
    a = a.parse_args()
    P = json.loads(Path(a.phase0).read_text())
    th, fit = {}, {}
    for c in a.cases:
        Re, Ca, b = case_params(c)
        th[c] = L.theory_monomials('raw', Re, Ca, b)
        co = P['per_case'][c][a.window]['coef']
        fit[c] = L.monomials('raw', {k: (v if v is not None and np.isfinite(v) else 0.) * L.factor_value(L.RAW[k]['factor'], Re, Ca, b)
                                     for k, v in co.items()})
    monos = sorted(set().union(*th.values()), key=lambda m: -abs(th[a.cases[0]].get(m, 0)))
    print(f"{'monomial':22s} " + ' '.join(f'{c[:23]:>23s}' for c in a.cases))
    print(f"{'':22s} " + ' '.join(f"{'theory':>11s} {'fit':>11s}" for _ in a.cases))
    for m in monos:
        print(f'{m:22s} ' + ' '.join(f'{th[c].get(m, 0):11.4g} {fit[c].get(m, 0):11.4g}' for c in a.cases))


if __name__ == '__main__':
    main()
