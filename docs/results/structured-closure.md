# Structured closure — Phase A diagnostics (A.1–A.4)

_16 Nov 2026. Plan: `docs/plans/structured-closure-os-plan.md` §3.1. Scripts: `scripts/structured-closure/`. Outputs (CSV, figure): `$THINFILM_STORE/analysis/structured_closure/`. OS reference: `docs/results/orr-sommerfeld.md` (B.4 case table)._

Notation: S = ΔM_x + ΔT_w (the ROM adds −S). F = −S is what the oracle adds to the q-equation (`oracle_forcing.hpp`, kSign). Linearisation about h = 1 + η, q = 1 + ξ.

## A.1 The current MLPs at the uniform state

_Script `a1_uniform_state.py`. Models: `ml_sum_hhx` (inputs h, h_x) and `ml_mx_tw` (ΔM_x and ΔT_w, inputs h, q, h_x, q_x), the files in `oracle/.../rom/models/`. Cases: the 15 development and 7 held-out cases of `ml/functional_mlp/data_config.json`. Output: `a1_uniform_state.csv`._

Linear form: S ≈ S₀ + αη + aξ + bη_x + cξ_x. The plan's §1.1 decomposition assumes S vanishes on a uniform film; here e = α + 3a = dS(h, h³, 0, 0)/dh measures how far it does not.
- Long-wave limit: the kinematic speed becomes 3 − e/(3/Re + a), and D = (3.6 − 3 cot β/Re − b − 3c)/(3/Re + a). This is the §1.1 formula, which holds when e = 0.
- Finite k: the linearised Lavalle quadratic plus this linear part. Maximum temporal growth is compared with OS.

| | Sum model (h, h_x) | Separate models (h, q, h_x, q_x) |
|---|---|---|
| S₀/(3/Re), the offset relative to the friction scale | −0.16 to +0.49 | −0.09 to +0.30 |
| Kinematic-speed shift −e/(3/Re + a) | −1.7 to +1.1 | −0.83 to +0.10 |
| Oracle case (Re120 Ca0.01 β45, held out): shift | +0.44 | −0.72 |
| a (relaxation) | 0 (no q input) | 0.04–0.10, about 4 × 3/Re at Re 120 |
| D, augmented / Lavalle | 0.89–1.02 | 0.14–0.81 (much less long-wave growth) |
| Max temporal growth, augmented / OS, oracle case | 0.91 (Lavalle 0.67) | 1.9 |
| Max temporal growth, augmented / OS, all 22 cases | −0.02 (Re200 β45: instability removed) to 1.4 | 1.0–2.4 |

- **Neither model keeps the uniform film steady.**
  - S₀ ≠ 0, so the ROM's base state is not the Nusselt film. The cause is the training data: `flat_mask: true` with tolerance 0.05 removes |h − 1| < 0.05 from training, so the uniform state is extrapolated.
  - The uniform-film slope e shifts the kinematic speed by an O(1) amount. That is the largest effect in the table.
  - The sum model's +0.44 in the oracle case agrees with its closed-loop behaviour: its waves lead the DNS (`ml-a-posteriori.md`).
- **Linear growth is not controlled.** The separate models roughly double the most-amplified growth in every case. This is consistent with them failing earlier in closed loop. The sum model is close to OS in the oracle case by chance: in other cases it ranges from removing the instability to 1.4 × OS.
- **What this means for Part II:** the constraints S(h, h³, 0, 0) = 0 by construction and an L2 linear target are not cosmetic. The current models violate both by O(1) amounts.

## A.2 Crest celerity against crest height

_Script `a2_crest_celerity.py`. Oracle case, elapsed 0–100 after t = 1800. One main crest per wave (spacing ≥ 0.6 λ, λ = 87, h > 1.05, prominence ≥ 0.1), 100 < x < 750. Each crest is matched over Δt = 1. Figure: `a2_crest_celerity.png`._

Median crest celerity per bin of h_max:

| h_max | 1.00 | 1.15 | 1.30 | 1.45 | 1.60 | 1.75 | 1.90 | 2.05 | 2.20 | 2.35 |
|---|---|---|---|---|---|---|---|---|---|---|
| DNS | 1.88 | 1.88 | 1.84 | 1.78 | 1.71 | 1.75 | 1.92 | 1.78 | – | – |
| Oracle ΔT_w + ΔM_x | 1.89 | 1.88 | 1.84 | 1.79 | 1.71 | 1.80 | 1.91 | 1.77 | – | – |
| ML sum | 1.87 | 1.88 | 1.81 | 1.74 | 1.73 | 1.78 | 1.82 | 1.85 | 1.94 | 2.05 |
| ML separate | 1.96 | 1.93 | 1.87 | 1.80 | 1.81 | 1.85 | 1.91 | 2.02 | 2.15 | 2.26 |
| Lavalle | 1.91 | 1.84 | 1.71 | 1.86 | (2.35, few) | – | – | – | – | – |

- **Small crests move at the linear phase speed.** DNS 1.88 against OS ω/k_r = 1.85 at 10 Hz. Lavalle's linear value is 1.78, and its small crests run at about 1.8–1.9.
- **ML sum:** it lies on the DNS c(h_max) curve within the scatter. It simply reaches taller crests (up to 2.5), where celerity rises to about 2.05. Its celerity error is a consequence of its amplitude error.
- **ML separate:** it runs about 0.05–0.08 (3–4%) above the DNS curve at small and moderate heights, and well above it for h > 2. It has a speed error of its own.
- **Lavalle:** its waves saturate near h ≈ 1.4–1.5 with flat or multi-humped crests, and the crest matching scatters there. Only the medians are meaningful.

## A.3 Phase projection of the DNS correction

_Script `a3_phase_projection.py`. F = −S on x > 100 (outlet 5% excluded), Gaussian σ = 0.5 on all fields.
- Oracle case: 100 snapshots, every 1 time unit.
- Six broadband β = 45° cases spanning δ_lin = 2–65: `ClosureData`, every 10 time units, t ≥ 500.
- Least-squares fit F ≈ α q′ + β q′_x. Each projection is split by region: front (h_x < −0.5 std), back (h_x > +0.5 std), crest (h > 1.15), capillary ripples (h < 1 and |h_xx| > std), substrate (the rest).
- Output: `a3_phase_projection.csv`._

| Case | δ_lin | rms F | α | −β | R² |
|---|---|---|---|---|---|
| Re120 Ca0.01 β45, oracle (10 Hz) | 36.7 | 0.124 | +0.0011 | −0.32 | 0.21 |
| Re010 Ca0.01 β45 | 2.1 | 0.058 | −0.0020 | −0.13 | 0.18 |
| Re020 Ca0.01 β45 | 4.4 | 0.109 | +0.0004 | −0.25 | 0.26 |
| Re040 Ca0.04 β45 | 14.3 | 0.148 | −0.0040 | −0.24 | 0.13 |
| Re080 Ca0.01 β45 | 22.3 | 0.071 | +0.0088 | −0.10 | 0.08 |
| Re120 Ca0.01 β45, broadband | 36.7 | 0.128 | +0.0027 | −0.23 | 0.13 |
| Re200 Ca0.01 β45 | 64.5 | 0.112 | +0.0024 | −0.22 | 0.13 |

- **Most of F is neither in phase with q′ nor with q′_x** (R² 0.08–0.26). It is localised at fronts and ripples and is rich in harmonics.
- **Where:** for every case, F is negative on the wave front and positive on the back. Momentum is removed at the front and returned at the back.
  - In the in-phase projection, the front (−0.02 to −0.06) and the back (+0.01 to +0.06) almost cancel, leaving a small net α of either sign.
  - The q′_x projection is dominated by the front (72–97% of the total), then the back. Crest and substrate contribute little. Ripples are small and of either sign.
- **The sign is the same in all seven cases:** −β ≈ −0.1 to −0.3 throughout. There is no obvious trend with δ in this nonlinear window.
- **Interpretation, and a correction to the plan's wording.** The plan reads the q′ projection as "changes amplitude" and the ∂_x q′ projection as "changes speed". That holds for the q-equation alone, not for the coupled h–q system at long wavelength.
  - There, a ξ_x term is the c of §1.1 and enters D, i.e. the *growth*.
  - An in-phase term ∝ q′ ≈ 3η is of e type and shifts the *kinematic speed*.
  - Read that way, the robust −β < 0 is a growth-raising quadrature term. Converted naively to long wave (c = −β, a = 0), it raises D by |β| Re. For the oracle case that is about +38 on Lavalle's 143. This direction is consistent with OS at δ > 10, but the window is nonlinear, so this is only indicative.
  - At finite amplitude and k the two effects mix. Separating them properly is what B.7 (linear targets) and Phase C (fitting the structured form) are for.

## A.4 Linear-region regression (oracle case)

_Script `a4_linear_regression.py`. Oracle case, rom_forcing.h5 every 0.05 time units, elapsed 0–100, Gaussian σ = 0.5. Windows x ∈ (10, 60), (20, 60) and (10, 100); B.5 will replace the guessed limit._

- In the window, rms η is 0.027–0.038. Between x 20–40 and x 60–80, rms η grows at about 0.012, close to the OS spatial rate of 0.0129.
- S has a **mean of +6e-4 to +8e-4**, larger than its fluctuation (rms 2e-4 to 4e-4). That is about 3% of the friction scale 3/Re. The likely sources are gas drag (a passive gas gives zero) or a mean state slightly off Nusselt (mean η ≈ +0.004 near the inlet). Either way it is not a linear-wave term.
- Fits:
  - Full form, S = S₀ + eη + a(ξ − 3η) + bη_x + cξ_x: R² 0.77–0.85.
  - Two-term form, S = S₀ + Pη + Qη_x: R² 0.17–0.30.
- **The individual coefficients cannot be identified.** With a single forced frequency ξ ≈ c_ph η, so η_x and ξ_x are nearly collinear (condition number 19–42). b and c are large with opposite signs (b ≈ −0.8, c ≈ +0.5), and the long-wave D they imply swings between 124 and 306 across the windows.
- **The prediction at the forcing frequency is robust:**

| | −k_i at ω = 0.1345 | Phase speed |
|---|---|---|
| OS | 0.01293 | 1.847 |
| Lavalle | 0.01250 | 1.780 |
| Lavalle + full fit, three windows | 0.0131–0.0136 | 1.841–1.847 |
| Lavalle + two-term fit | 0.0134–0.0143 | 1.80–1.82 |

- So in the linear region the DNS correction moves Lavalle onto OS at the forced frequency: the phase speed to within 0.3% and the growth to within 1–5%. This validates the extraction pipeline in the regime where the truth is known exactly.
- The correction at 10 Hz is mostly a phase-speed correction, because Lavalle's growth is already close there (B.4).
- Determining (a, b, c) needs several frequencies: OS targets (B.8) or dense broadband profiles (D3).

## Where it fails

- **A.1 is a linearisation about h = q = 1, which is not an equilibrium of the MLP-augmented ROM when S₀ ≠ 0.** The growth numbers describe the closure's linear part, not the closed-loop base state. The ROM runs used raw central differences for h_x; the long-wave numbers do not depend on the filter, but finite-k ones would.
- **A.2 covers one case, 100 time units, and crests matched over Δt = 1.** Lavalle's flat crests make its scatter large. Celerity also depends on position (growth versus saturated region), which the h_max bins mix.
- **A.3 uses rule-based regions with thresholds relative to the case's std**, so the region shares differ between cases. Broadband snapshots every 10 time units give spatial statistics only. The projections are for the q-equation; the coupled interpretation is long-wave only.
- **A.4 uses a single frequency**, so the linear coefficients are not separately identifiable. The window x < 60 is a guess (B.5). The mean offset of S is unexplained.
