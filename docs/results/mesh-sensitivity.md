# Mesh sensitivity: MAXLEVEL 14 vs 15 (Re120, Ca0.01, β45°, 10 Hz, L = 800)

_Figures: [`figures/mesh-sensitivity/`](figures/mesh-sensitivity/)_

Runs: `length-check/L800` (level 14) and `length-check/L800_Level15` in the data store.
Scripts: `scripts/mesh_study/{find_transient_exit,compare_mesh}.py`. Closure files, figures and CSVs:
`$THINFILM_STORE/datasets/BasC/DNS/param_space/ClosureData/mesh_study/` (`compare/`).

## Setup
- Level 15 stopped at t = 650 (scancelled), so the overlap is t ≤ 650 only; level 15 was not restarted.
- The initial disturbance reaches x ≈ 790 at t ≈ 390 in both runs and has left by t ≈ 400–410, so the
  window is **t = 550–650** (11 snapshots, every 10; constant-frequency forcing, waves repeat).
- Closures from `extract_closure_terms.py` with identical settings (nx = 16384 vs 32768).
- Waves = crests of h (Gaussian-smoothed 0.3, prominence 0.5), x < 700, matched between runs at the same t
  by nearest crest (offset 2.5–8 < 12 tolerance); fields cut in ξ = x − x_crest, |ξ| ≤ 40. 65 matched waves.
- Not a clean mesh-only test: L800 sets DT = 0.003 (its dt sits at the cap), level 15 is CFL-limited (dt ≈ 0.0016).

## Results
**Waves.**
- Wave speed: median crest celerity 1.913 (L14) vs 1.937 (L15), **+1.2 %**. The difference grows downstream:
  1.75/1.85/1.93/1.93/1.91 vs 1.76/1.85/1.96/1.95/1.96 in x-bins of 140 starting at x = 60. Matched crest offset
  (L15 − L14) grows from ≈ 2.5 at x ≈ 180 to ≈ 8 at x ≈ 690.
- Crest height 1.834 vs 1.833, wavelength 89.8 vs 90.5, mean h 0.967 vs 0.975, rms h′ 0.334 vs 0.332, h range 0.22–2.11 vs 0.17–2.13.
- Crest-aligned h and q differ by 4 % and 5 % (median relative L2 over |ξ| ≤ 40, max 17 % and 23 %).
  The main difference is the capillary ripple train ahead of the crest: level 15 has slightly different ripple phases/amplitudes.

**Closures, crest-aligned (median over matched waves; L15/L14 peak ratio, relative L2 difference).**

| term | raw peak ratio | raw relL2 | smoothed (Gaussian 1.0) peak ratio | smoothed relL2 |
|---|---|---|---|---|
| ΔM_x | 0.993 | 0.10 | 0.988 | 0.07 |
| ΔT_w | 0.996 | 0.10 | 0.991 | 0.07 |
| ΔP_x | 1.048 | 0.87 | 0.969 | 0.44 |
| ΔD_x | 1.037 | 0.48 | 1.041 | 0.18 |
| S_ML | 1.030 | 0.62 | 0.984 | 0.10 |

- ΔM_x and ΔT_w are mesh-converged: peaks within 1 %, peak positions identical relative to the crest, correlation 0.998.
- ΔP_x (and so S_ML raw, and ΔD_x to a lesser extent) is dominated by grid-scale spikes at the interface
  (raw peak ≈ 1.6, against ≈ 0.05 after 1.0 smoothing); their locations and signs are not reproduced between meshes.
  This is noise in the extraction (SG window fixed in points, so its physical width halves at level 15; one-step ∂ₜ),
  not a wave-physics difference: after a common Gaussian smoothing S_ML agrees to 10 %.

**Budget check** (`check_momentum_budget.py`, t = 550, 600, 650; mass residual, DNS momentum residual / ∂ₜq):
L14 0.016, 0.072; L15 0.045, 0.128. Level 15 closes worse, consistent with the halved dt (the one-step ∂ₜ is noisier), not with a worse mesh.
ROM-grid check, 1 h_N smoothing: E1/E0 0.044 (L14), 0.059 (L15).

## Conclusion
Level 14 is adequate for the closures: ΔM_x, ΔT_w and the smoothed S_ML are mesh-insensitive once waves are aligned,
and the only systematic mesh effect on the waves is a +1.2 % celerity at level 15. Transient window only (t ≤ 650,
waves not saturated), with the DT mismatch above. Update the older note in `decisions.md` ("level 15 changes only the wave speed slightly"): now quantified.

## Where it fails

Figures: [`limit01`](figures/mesh-sensitivity/limit01_not_converged.png), [`fig02`](figures/mesh-sensitivity/fig02_dPx_grid_scale_noise.png), [`fig03`](figures/mesh-sensitivity/fig03_crest_offset.png).

- **ΔP_x and ΔD_x are not mesh-converged as extracted.** The crest-aligned relative L2 difference between level 15 and level 14 is 0.87 for ΔP_x and 0.48 for ΔD_x (0.44 and 0.18 after 1 h_N smoothing), and 0.62 for S_ML (0.10 smoothed), against 0.10 for ΔM_x and ΔT_w (0.07 smoothed) (`limit01`). The spikes sit at the interface at different places on the two meshes (`fig02`).
- **Level 15 closes the budget worse than level 14**: mass residual 0.045 against 0.016, DNS momentum residual 0.128 against 0.072, ROM remainder E1/E0 0.059 against 0.044 (3 snapshots, t = 550–650).
- **The waves themselves differ.** The crest speed is 1.2% higher on level 15, so the crest offset grows from 2.5 at x ≈ 180 to about 8 at x ≈ 690 (`fig03`), and crest-aligned h and q differ by 4–5% (median).
- **This is not a clean mesh test.** DT = 0.003 on level 14 against a CFL-limited step of about 0.0016 on level 15; level 15 was stopped at t = 650, so the comparison is 11 snapshots (about 2 forcing periods) of one case, and there is no level 16.
