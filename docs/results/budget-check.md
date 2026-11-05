# Momentum-budget closure check

_Figures: [`figures/budget-check/`](figures/budget-check/)_

Does the depth-integrated mass and momentum budget close with the extracted terms, and does S_ML close the lavalle-cpp discretisation? Run before the oracle so that bookkeeping errors are not mistaken for physics.

- **Code:** `pipeline/extract_closures/check_momentum_budget.py` (commit 26aefd9), `run_budget_check.sh`. Metrics are defined in `pipeline/extract_closures/README.md`.
- **Data:** `$THINFILM_DATA/BasC/DNS/param_space/ClosureData/budget_check/` (`summary.csv`, `per_case/<case>.csv`, `run_info.txt`). SLURM job 48347255 (2 min), 22 closure files, t = 1000–2000 every 100, smoothing widths 0 and 1 h_N, mask 20 < x < 0.95 L, ROM N = 4000.
- **Related notebook:** `notebooks/analysis/verify_thinfilm_equation.ipynb` does the DNS-grid part (A) for one case.

## Findings

- Uses the (t, t−dt) pair in each snapshot for ∂ₜ. The stored deltas match the raw integrals to machine precision, so C_Lav + ΔC = C_DNS exactly.
- Restricting the DNS state to the ROM grid (N = 4000) and evaluating a Python replica of the lavalle-cpp q-RHS, d_t q − RHS_ROM ≈ S_ML:
  - inertial cases close well; at 1 h_N smoothing E1/E0 = |d_t q − RHS_ROM − S_ML| / |d_t q − RHS_ROM| is 0.04 (Re120 Ca0.01 β45, the oracle target, corr 0.999) and 0.10 (Re200 Ca0.01 β6.4);
  - Re40 Ca0.02 β6.4: 0.35, but the Lavalle misfit itself is only ~7% of d_t q there;
  - **Re10 Ca0.01 β45 (We = 0.1) does not close** (E1/E0 ≈ 0.9). Mass closes over 1 h_N windows (2%), but the one-step ∂ₜh has grid-scale jitter (15% of its variance below 1 h_N), and the DNS momentum budget leaves 13% of d_t q unexplained even at 1 h_N. Suspect ∫∂ₓp near the interface at large σ, or interfacial terms omitted by the passive-gas form. Treat strongly capillary low-Re cases with caution in training and in the oracle.
- **Full sweep (all 22 closure files, t = 1000–2000 every 100; results in `ClosureData/budget_check/`):**
  - at 1 h_N smoothing, the unexplained remainder E1 is ≤ 4.5% of d_t q for 19 of 22 cases; corr(E0, S_ML) ≥ 0.99 for 15. The exceptions are Re10 Ca0.01 β45 (16%), Re80 Ca0.005 β6.4 (9%) and Re20 Ca0.01 β45 (7%), We 0.1–0.4 (but We is not the criterion: Re40 Ca0.01 β90, We = 0.4, leaves only 3.7% and closes well);
  - where these three cases fail, they fail mostly at the same snapshots where the mass equation fails (e.g. Re80 Ca0.005 β6.4 at t = 1500–1600: mass 0.24, E1/E0 0.92; at t = 1000: mass 0.01, E1/E0 0.28). Mass involves no closure, so the one-step ∂ₜ is unreliable there. The per-snapshot mass residual is a cheap data-quality flag;
  - β = 6.4° cases show E1/E0 ≈ 0.1–0.45 only because Lavalle is already close there (E0 ≈ 7–12% of d_t q); E1 itself is ≤ 3% of d_t q.
- Rusanov numerical dissipation is ~2% of d_t q for Re120; Noble–Vila capillarity with r = √(h/We) h_x matches h h_xxx/We to ~2–7%.

## Where it fails

- **Three of the 22 cases do not close** (remainder E1 above 4.5% of ∂ₜq at 1 h_N smoothing): Re10 Ca0.01 β45 (16%), Re80 Ca0.005 β6.4 (9%) and Re20 Ca0.01 β45 (7%). All have We between 0.1 and 0.4, but We is not the criterion: Re40 Ca0.01 β90 (We = 0.4) leaves 3.7% and closes well.
- **Re10 Ca0.01 β45 fails before the ROM is involved.** Its one-step ∂ₜh is noisy (on average 17% of the variance is at wavelengths below 1 h_N, 7–27% by snapshot over t = 1000–2000, against 0.3% for Re120 Ca0.01 β45; the 15% quoted above was one snapshot), the mass residual is about 45% of ∂ₜh (pooled over the same times), and the DNS momentum budget alone leaves 13% of ∂ₜq unexplained at 1 h_N smoothing. The cause is not identified (candidates: the pressure integral near the interface at large σ; interface terms that the passive-gas form omits).
- **Re80 Ca0.005 β6.4 swings between snapshots**: E1/E0 is 0.24–0.92 and the mass residual 0.01–0.24 over the 11 snapshots from t = 1000 to 2000.
- **The mass residual is a flag, not a guarantee.** Its rank correlation with the remainder is 0.53 over 234 snapshots. Six of the 202 snapshots of the other 19 cases, all from Re10 Ca0.1 β45, have E1/E0 of 0.3–0.5 with a mass residual below 0.05; there the Lavalle misfit is small, so the ratio is large while the remainder itself is at most 4% of ∂ₜq.
- **The check cannot separate extraction error from noise in ∂ₜ.** It uses a difference over one DNS step (about 10⁻³ to 10⁻²), so grid-scale noise in ∂ₜ appears as a failure to close whether or not the closure terms are right.
