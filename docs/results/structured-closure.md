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

## Phase C: structured closure V1 (17 Nov)

_Code: `ml/structured_closure/` (README-level docstrings in each file); ROM: `rom/lavalle-cpp`, lines tagged `// STRUCTCLOSURE`. Runs, tables and ROM outputs: `$THINFILM_STORE/codes/ml-models/structured_closure/{runs,rom}/`. Summaries: `summarize_runs.py`, `rom_tests.py analyse`, `diag_oracle.py`._

### Setup

- **Form (plan §3.2, V1).** ∂ₜq + ∂ₓ[Φ q²/h + P_Lav + ΔP] = κ(Q − q)/h² + capillary + viscous, with Q = h³(1 + g₁h_x + g₂q_x + g₃h_xx).
  - All six coefficient functions are small networks (2 × 32, tanh) of (h; ln Re, ln Ca, β).
  - Φ = 1 + softplus(N), monotone in h; P_tot′ = P_Lav′ e^{N_s} > 0; κ = (3/Re)e^{N_κ}. So V1 is hyperbolic for every state.
  - S(h, h³, 0, 0, 0) = 0 for any weights, so S₀ = e = 0 exactly. Lavalle is the initial state of every network.
- **Data.** Cache v2 for the 15 development and 7 held-out cases. x from 50 to 95% of the domain, every 8th point; t ≥ 300. The three cases with a failed cache check are excluded (`decisions.md`, 17 Nov).
- **Loss.** Per-case 1/rms-weighted error in S = ΔM_x + ΔT_w, plus:
  - λ_lin × the L2 constraint;
  - optionally λ_F × an anchor of the flux part to ∂ₓΔF;
  - optionally λ_P × a power penalty (below).
- **L2 is a constraint on the dispersion relation, not on fitted (a, b, c, d).**
  - In the ROM the flux part is differentiated unfiltered by the finite-volume scheme, while the relaxation part sees the Gaussian gradients D_G(σ), with factor e^{−k²σ²/2}.
  - At 20 frequencies per point, the first-order wavenumber error δk = −F(k_OS)/F′(k_OS) of the linearised augmented ROM is penalised, together with (D − D_OS)/D_OS. The OS spatial roots k_OS come from `os_dispersion.py`.
  - The constraint is imposed at the 85 OS case-table points, minus the held-out parameters.
  - With σ = 0 it reproduces the B.8 growth errors to four digits (`test_structured.py`).
- **Speed.** GPU (L4): 1.1 s per epoch over 4.7 M points; 8 CPU cores: 14 s per epoch.

### Training sweep (seed 0, 100 epochs)

λ_lin {0, 10} × λ_F {0, 0.1} × σ {0.5, 0.25}. Growth error = max |k_i − k_i,OS| / max |k_i,OS| over the forced band.

| Run | NRMSE train | NRMSE held-out (max) | Growth error, train (median / max) | Growth error, held-out (median / max) | \|D/D_OS − 1\|, held-out max |
|---|---|---|---|---|---|
| σ 0.5, L0 | 0.226 | 0.321 (0.79) | 0.56 / 6.7 | 0.45 / 2.9 | 0.49 |
| σ 0.5, L10 | 0.252 | 0.295 (0.60) | 0.058 / 0.17 | 0.10 / 0.25 | 0.013 |
| σ 0.5, L10, F 0.1 | 0.255 | 0.306 (0.70) | 0.069 / 0.18 | 0.10 / 0.31 | 0.024 |
| σ 0.25, L0 | 0.193 | 0.273 (0.78) | 0.60 / 10.4 | 0.58 / 0.99 | 0.57 |
| **σ 0.25, L10** | 0.217 | **0.260** (0.62) | 0.072 / 0.17 | **0.10 / 0.16** | 0.012 |
| σ 0.25, L10, F 0.1 | 0.217 | 0.259 (0.62) | 0.065 / 0.16 | 0.083 / 0.16 | 0.010 |

- **Without L2 the data do not constrain linear growth:** errors reach 7–10× the OS peak.
- **L2 (λ_lin = 10) cuts the growth error about tenfold at no held-out cost.** The λ_lin scan (1, 10, 100) chose 10: λ = 100 costs 15% in held-out NRMSE.
- **σ = 0.25 fits better than σ = 0.5** (held-out 0.26 against 0.30).
- **The flux anchor λ_F does not help.**
- **The minimum hyperbolicity discriminant over all DNS states is > 0 in every run**, as guaranteed.

### ROM tests (C.10)

- **Uniform film, 100 time units:** h and q stay exactly 1, after the zero-sum derivative fix below.
- **Stiffness:** max κ dt/h² is 0.006–0.03, so explicit RK4 is fine (C.9 not needed).
- **Comb inlet (B.6 setup):** the ROM reproduces the model's own analytic linear dispersion to 0.5% (Re 20) and 3–4% (Re 120, 200), medians. So the solver implements what was trained.
- **Growth error against OS (median / max over the reliable modes):**

| | Re 20 | Re 120 | Re 200 |
|---|---|---|---|
| Lavalle | 0.04 / 0.10 | 0.27 / 0.41 | 0.36 / 0.53 |
| σ 0.5, L10 | 0.06 / 0.11 | 0.08 / 0.36 | 0.11 / 0.31 |
| σ 0.25, L10 | 0.03 / 0.04 | 0.08 / 0.31 | 0.11 / 0.35 |
| σ 0.5, L0 | 0.05 / 0.12 | 0.22 / 0.43 | 0.54 / 0.66 (goes nonlinear, \|h − 1\| 0.58) |

The maxima are at the top of the comb, above the band where L2 is imposed (ω up to 0.9 at Re 120).

- **Bug found by the uniform-film test.** The truncated Gaussian second derivative (scipy `gaussian_filter1d`, truncate = 4, and the ROM copy) is not zero on constants:

  | σ | Grid | h_xx(h = 1) |
  |---|---|---|
  | 0.5 | DNS | −3.5e-3 |
  | 0.5 | ROM | −1.8e-3 |
  | 0.25 | DNS | −1.65e-2 |

  With g₃ in the model the flat film drifted (|h − 1| = 7e-3 at t = 100), and training and ROM saw different biases.
  - **Fix:** a zero-sum operator in the ROM (`gauss_deriv_hN` subtracts c₀ × smoothed field) and c₀·h subtracted from h_xx in training.
  - Cache-v2 h_xx and q_xx still carry the bias (CLAUDE.md gotcha).

### Oracle a posteriori (Re 120 Ca 0.01 β 45, 10 Hz, from the DNS state, 100 time units)

h error = rms(h_ROM − h_DNS)/std(h_DNS) over 20 < x < 760; max h in brackets (DNS 2.0–2.1).

| Elapsed | Lavalle | Exact ΔT_w+ΔM_x | `ml_sum_hhx` | σ 0.5 L10 | σ 0.25 L10 | σ 0.5 L10 P100 | σ 0.25 L10 P100 |
|---|---|---|---|---|---|---|---|
| 1 | 0.115 | 0.060 | 0.057 | 0.063 | 0.068 | 0.060 | 0.062 |
| 5 | 0.54 | 0.15 | 0.17 | 0.17 | 0.18 | 0.18 | 0.19 |
| 10 | 0.61 | 0.17 | 0.22 | 0.26 | 0.23 | 0.35 | 0.33 |
| 20 | 0.59 | 0.20 | 0.28 | 0.47 | 0.40 | 0.70 | 0.65 |
| 50 | 0.59 (1.45) | 0.19 (2.10) | 0.47 (2.38) | 1.20 (2.33) | 0.60 (2.50) | 1.04 (2.35) | 1.06 (2.26) |
| 99.9 | 1.07 (1.43) | 0.55 (2.04) | 1.26 (2.50) | 1.94 (2.57) | 1.55 (3.05) | 1.17 (2.26) | 1.31 (2.24) |

- **Up to about 5 time units** the structured closures match `ml_sum_hhx` and are close to the exact closure.
- **From about 10** they drift faster than `ml_sum_hhx`. The best is σ 0.25 L10 (0.60 against 0.47 at t = 50).
- **Phase drift dominates the error.** Shift of the ROM waves relative to the DNS (+ = ahead) at maximum correlation over 100 < x < 700:

  | | t = 10 | t = 20 | t = 50 |
  |---|---|---|---|
  | `ml_sum_hhx` | +0.6 | +0.6 | 0.0 (corr 0.95) |
  | σ 0.5 L10 | +0.8 | +1.4 | +7.8 |
  | σ 0.25 L10 | +0.4 | −0.2 | −0.2 (corr 0.91) |
  | σ 0.25 L10 P100 | −0.4 | −2.2 | −7.2 |

- **Split runs: the flux part supplies the crest growth, the relaxation part damps.**
  - Flux part alone: the best short-time error of any run (0.040 at t = 1, exact closure 0.060), then crests up to 2.6–3.0.
  - Relaxation part alone: behaves like Lavalle (crests decay to 1.4).

### Diagnosis on the oracle case (a priori, `diag_oracle.py`)

Closures evaluated on the DNS states with the inputs the ROM gives them. Regions as in A.3.

- **All closures fit S equally well:** NRMSE 0.125 (`ml_sum_hhx`), 0.117 (σ 0.5 L10), 0.120 (σ 0.25 L10).
- **The closure's net power into the waves is a tiny residual, and every learned closure gets its sign wrong.**
  - P = ⟨(−S) q′⟩, q′ = q − ⟨q⟩ₓ.
  - True closure: front −0.32, back +0.19, crest +0.02, ripples −0.01; net −3.4e-4 (slightly damping).
  - Every learned closure gives net input: +2.2e-3 (`ml_sum_hhx`), +1.8e-3 (σ 0.5), +1.0e-3 (σ 0.25); the excess over the DNS is +2.6e-3, +2.1e-3 and +1.4e-3. For σ 0.5 the excess comes about 40% from the back face and 30% from the substrate (relaxation part), the rest from ripples, crest and front.
- **Power penalty** λ_P × mean r², r = Σₓ(S_pred − S)q′ / Σₓ|S q′| over 2 random whole snapshots per case per step (λ_P 10, 100):
  - It fixes the a priori net power (0.2–0.3× of the DNS value instead of 4–6×).
  - But a priori NRMSE rises to 0.16 on the oracle case, and **a posteriori is not better**: lower crests, but larger phase drift (table above).
  - So net power alone does not control the closed-loop error.

### Flat-film-exact MLP, oracle a posteriori (17 Nov, evening)

S = N(h, q, h_x, q_x, h_xx; p) − N(h, h³, 0, 0, 0; p), L2 at the 85 OS points, σ 0.25, λ_lin 0, 1, 10 (seed 0, lr 1e-2; `train.py --model mlp`, ROM `FlatMLPClosure`). The σ 0.5 L10 run and the comb runs are in the next section (18 Nov). Scripts: `scripts/oracle_summary.py`, `ml/structured_closure/diag_oracle.py`, `scripts/structured-closure/oracle_snapshots.py` (h(x) panels), `scripts/structured-closure/oracle_phase.py` (shift time series). Figures and csv in `$THINFILM_STORE/codes/ml-models/structured_closure/rom/{figures,}` (not in git).

A priori on the oracle DNS states (NRMSE; net power P/P_DNS, positive = same sign as the DNS, i.e. damping):

| Model | NRMSE | P/P_DNS |
|---|---|---|
| `ml_sum_hhx` | 0.125 | −6.5 |
| V1 σ 0.25 L10 | 0.120 | −3.0 |
| MLP L0 | 0.047 | +0.52 |
| MLP L1 | 0.052 | +4.1 |
| MLP L10 | 0.055 | +0.54 |

A posteriori (h error / std h_DNS; in brackets the error after removing the best-fit x-shift; max h ROM/DNS):

| t | exact | `ml_sum_hhx` | V1 σ 0.25 | MLP L0 | MLP L1 | MLP L10 |
|---|---|---|---|---|---|---|
| 10 | 0.17 | 0.22 | 0.23 | 0.22 | 0.22 | **0.18** |
| 50 | 0.19 (0.17) | **0.47 (0.47)**, 1.13 | 0.60 (0.59), 1.18 | 0.67 (0.49), 1.12 | 0.79 (0.55), 1.01 | 0.59 (0.47), 1.04 |
| 100 | 0.55 (0.44) | 1.25 (0.91), 1.19 | 1.54 (1.03), 1.47 | **0.58 (0.47)**, 1.12 | 1.21 (0.91), 1.09 | 0.70 (0.64), 1.02 |

Phase (DNS wavelength 89.8, celerity c = λ/T = 1.92; shift positive = ROM ahead):

| | shift t = 20 / 50 / 100 | drift / c, t < 50 |
|---|---|---|
| exact | 0.0 / −0.2 / −1.9 | −0.5% |
| `ml_sum_hhx` | +0.3 / −0.1 / +6.9 | −0.9% |
| V1 σ 0.25 | −0.5 / −0.3 / +9.4 | −1.0% |
| MLP L0 | −1.5 / −3.0 / +1.5 | −4.1% |
| MLP L1 | −1.3 / −4.7 / −9.0 | −5.8% |
| MLP L10 | −0.3 / −2.5 / −0.9 | −3.3% |

- **Adding q and the gradients as inputs halves the a priori error** and gives the first learned closures with the right sign of net power.
- **No crest overgrowth.** L10 keeps max h within 4% of the DNS to t = 100; `ml_sum_hhx` and V1 overshoot by 19–47%, then run ahead (taller waves are faster).
- **The new MLPs make the waves 3–6% too slow from the start**, while the amplitude is still right: a wave-speed error of the closure itself, not one driven by amplitude. L0 and L10 stop lagging at about 3 and recover; L1 keeps lagging. Not monotone in λ_lin (one seed).
- **A shape error is common to all learned closures.** After removing the shift, all reach 0.37–0.55 by t = 30–50 against 0.17–0.20 for the exact closure. The snapshots put it on the wave backs (rise too early) and in the substrate/ripple region. The new MLPs do not run away: 0.47 (L0) and 0.64 (L10) at t = 100 against 0.91–1.03.

### Flat-film-exact MLP: all seed-0 runs (18 Nov)

All four seed-0 runs finished; `rom_tests.py analyse mlp_s0.25_L0_seed0 mlp_s0.25_L1_seed0 mlp_s0.25_L10_seed0 mlp_s0.5_L10_seed0`.

**Uniform film:** h and q stay exactly 1 for 100 time units in every run.

**Comb inlet, growth error against OS** (median / max over the reliable modes; ROM against the model's own analytic dispersion in brackets, median):

| | Re 20 | Re 120 | Re 200 |
|---|---|---|---|
| Lavalle | 0.024–0.043 / 0.097 | 0.27–0.28 / 0.41 | 0.36–0.39 / 0.53 |
| MLP σ 0.25 L0 | 0.128 / 0.155 | 0.080 / 0.150 | **0.471 / 0.615** (11 of 23 modes) |
| MLP σ 0.25 L1 | 0.021 / 0.041 | 0.078 / 0.425 (0.034) | 0.113 / 0.338 (0.048) |
| MLP σ 0.25 L10 | 0.017 / 0.040 | 0.073 / 0.364 (0.029) | 0.084 / 0.328 (0.040) |
| MLP σ 0.5 L10 | 0.015 / 0.041 | 0.068 / 0.379 (0.031) | 0.090 / 0.312 (0.044) |

**Oracle, σ 0.5 L10 against σ 0.25 L10** (h error; max h ROM/DNS):

| t | exact | `ml_sum_hhx` | σ 0.25 L10 | σ 0.5 L10 |
|---|---|---|---|---|
| 10 | 0.17 | 0.22 | 0.18 | 0.21, 2.02/2.01 |
| 20 | 0.20 | 0.28 | 0.30 | 0.33, 2.01/2.01 |
| 50 | 0.19 | 0.47 | 0.59 | **0.39**, 2.29/2.12 |
| 100 | 0.55 | 1.26 | 0.70 | **0.62**, 2.23/2.08 |

- **With the OS constraint the MLP's linear growth is within 7–11% of OS (median) at Re 120–200**, against 27–39% for Lavalle, and the ROM reproduces the model's own prediction to 3–5%. The max errors (0.31–0.43) are at the top of the comb band, above the frequencies the constraint covers.
- **Without it (L0) the Re 200 waves are damped** (max |h − 1| 1e-3; only 11 modes detected) and Re 20 is 13% off. The constraint is needed for linear behaviour outside the oracle regime.
- **σ 0.5 L10 is the best learned closure on the oracle case so far:** 0.39 at t = 50 and 0.62 at t = 100, against 0.47 and 1.26 for `ml_sum_hhx`. It is still about 0.2 above the exact closure at t = 50, and its crests are 6–10% high from t ≈ 50.
- One seed each; the seed-1 runs of both L10 models are running (see the tracker).

### ROM speed-up of the flat-film-exact MLP (18 Nov)

`FlatMLPClosure` evaluated the network twice per cell and RK stage. The flat-film term N(h, h³, 0, 0, 0; p) depends only on h for a given case, so it is now tabulated once per run (h 0.02–8, dh 1e-4, h = 1 exactly a node so the uniform film stays exact; outside the range the network is evaluated; `n0_table = 0` in the config restores per-cell evaluation). The cell loop runs with OpenMP (`-fopenmp`; threads from `OMP_NUM_THREADS`).

| | s per step (4000 cells) |
|---|---|
| before | 0.164 |
| table, 1 thread | 0.085 |
| table, 4 threads | 0.024 |

Check on the oracle case, 400 steps: max |Δh| 1.5e-9 and max |Δq| 3.4e-9 against the old binary (the interpolation error); 1 and 4 threads give identical results. A 2000-time-unit broadband run takes about 3 h on 4 cores.

### Broadband held-out a posteriori test: setup (gate GD)

`ml/structured_closure/broadband.py` (`make`, `analyse`) and `run_broadband.sh`. The held-out DNS cases have h every 10 time units only (closure files), so the comparison is statistical:

- **ROM:** flat start, the DNS Chang inlet parameters (F0 0.025, M 1000, f_cut 28 Hz, q_in = 1 + F(t) as in the DNS), L 800, N 4000, dt 0.005, t 0–2000, output every 10. The random phases differ from the DNS, so realisations differ.
- **Statistics over t ≥ 1000 and 20 < x < 760:** h mean, std and skewness; main-crest height (peaks with prominence ≥ std h; mean and 90th percentile); mean wavelength (1 / spectral centroid over 300 < x < 760).
- **Distances to the DNS:** envelope error (rms over 20-wide x-bins of std_ROM(x) − std_DNS(x), relative to the mean DNS std), PDF distance (Wasserstein-1 of h over std h_DNS), crest-height ratio.
- **Cases:** the 7 held-out cases (Re20 Ca0.01 β45, Re40 Ca0.04 β45, Re80 Ca0.005 β6.4, Re80 Ca0.01 β90, Re120 Ca0.01 β45, Re120 Ca0.02 β90, Re200 Ca0.01 β6.4). **Closures:** Lavalle, `ml_sum_hhx`, V1 σ 0.25 L10, MLP σ 0.25 L10 and σ 0.5 L10 (seed 0); seed 1 of both MLPs follows.
- DNS statistics for reference (t 1000–2000): std h 0.21–0.39; main-crest mean 1.16 (β 6.4 cases) to 1.66 (Re40 Ca0.04); skewness −0.5 (β 6.4) to +1.3.

### Dense broadband a posteriori runs (18 Nov)

`ml/structured_closure/dense_aposteriori.py` (`forcing`, `make`, `analyse`). The dense restarts in `BasC/DNS/dense/` have h and q profiles every 0.01 for 10 time units; their inlet forcing is active (restarted after the 10 Nov fix). From them, `forcing` builds `rom/rom_forcing.h5` (as `pipeline/oracle/make_rom_forcing.py`, without closure terms), and the ROM runs from the DNS state with the DNS inlet, as in the oracle case. No exact-closure reference (the profiles carry no closure terms). Figure: `scripts/structured-closure/dense_error_fig.py`; csv `rom/dense/dense_summary.csv` (error every 0.1). Linear-wave figure for the same closures: `scripts/structured-closure/comb_growth_fig.py`.

h error / std h_DNS over 20 < x < 760 at t = 5 / 10 after the DNS state (max h at t = 10, ROM/DNS):

| | Re 20 Ca 0.01 β 45 (held out) | Re 120 Ca 0.01 β 45 (held out) | Re 40 Ca 0.002 β 45 (training) |
|---|---|---|---|
| Lavalle | 0.22 / 0.28 (1.90/2.10) | 0.45 / 0.59 (1.57/2.08) | 0.25 / 0.39 (1.72/2.18) |
| `ml_sum_hhx` | 0.11 / 0.22 (2.12) | 0.24 / 0.42 (2.12) | **0.06 / 0.10** (2.13) |
| V1 σ 0.25 L10 | 0.17 / 0.25 (2.07) | 0.23 / 0.36 (2.14) | 0.16 / 0.32 (1.90) |
| MLP σ 0.25 L10 | **0.09 / 0.11** (2.15) | **0.20 / 0.25** (2.05) | 0.09 / 0.22 (2.03) |
| MLP σ 0.5 L10 | 0.10 / 0.14 (2.11) | 0.21 / 0.30 (2.04) | 0.11 / 0.22 (1.95) |

- **On both held-out broadband cases the flat-film-exact MLPs have the lowest error:** at t = 10 about half of `ml_sum_hhx` at Re 20 and 30–40% lower at Re 120, with crest heights within 2–3% of the DNS.
- **σ 0.25 is slightly ahead of σ 0.5 over these 10 time units,** the reverse of the oracle ranking at t = 50; one seed, so not separable yet.
- **Exception: Re 40 Ca 0.002 (We 0.08, a training case).** `ml_sum_hhx` is best (0.10 against 0.22), and the new MLPs let the crests sag (1.95–2.03 against 2.18). This strongly capillary regime is where the DNS momentum budget is least reliable.

### Broadband held-out a posteriori results, seed 0 (19 Nov)

`broadband.py analyse` (CSV `rom/broadband/broadband_seed0.csv`); figures `scripts/structured-closure/broadband_fig.py` (`metrics`, `waves`) → `rom/figures/broadband_{metrics,waves}.png`. PDF distance (W1/std, lower is better) and main-crest ratio (ROM/DNS), t 1000–2000:

| Case | Lavalle | `ml_sum_hhx` | V1 σ 0.25 | MLP σ 0.25 | MLP σ 0.5 |
|---|---|---|---|---|---|
| Re20 Ca0.01 β45 | 0.14 / 0.85 | 0.30 / 1.08 | 0.05 / 0.93 | 0.06 / 0.97 | 0.05 / 0.95 |
| Re40 Ca0.04 β45 | 0.20 / 0.82 | 0.25 / 1.07 | 0.30 / 1.04 | blew up (t 641) | 0.29 / 1.16 |
| Re80 Ca0.005 β6.4 | 0.11 / 0.94 | 0.26 / 0.97 | 0.08 / 0.98 | 0.09 / 1.00 | **0.04** / 1.00 |
| Re80 Ca0.01 β90 | 0.32 / 0.76 | 0.35 / 1.04 | 0.38 / 1.47 | **0.03** / 0.96 | blew up (t 401) |
| Re120 Ca0.01 β45 | 0.29 / 0.84 | 0.43 / 1.10 | 0.46 / 1.38 | 0.13 / 1.03 | **0.06** / 1.00 |
| Re120 Ca0.02 β90 | 0.28 / 0.72 | 0.27 / 1.05 | 0.49 / 1.40 | 0.20 / 0.89 | **0.07** / 0.96 |
| Re200 Ca0.01 β6.4 | 0.29 / 0.94 | 2.01 / 1.96 | 0.69 / 1.37 | 0.36 / 1.09 | **0.26** / 1.07 |

- **MLP σ 0.5 L10 is the best closure on 4 of its 6 completed cases and ties with V1 on Re 20** (PDF distance 0.04–0.07 on those five against Lavalle 0.11–0.29; crest ratio 0.95–1.00). Envelope error 0.06–0.07 against Lavalle 0.28–0.31 at Re 120.
- **Robustness:** two MLP runs stopped with loss of hyperbolicity (discriminant < 0): σ 0.25 on Re40 Ca0.04 β45 at t 641, σ 0.5 on Re80 Ca0.01 β90 at t 401. No other closure failed.
- **Re40 Ca0.04 β45** (We 1.6, most capillary held-out case): σ 0.5 is worse than Lavalle (0.29 against 0.20; crests 16% high, wavelength 28 against 33).
- **Re200 β6.4:** σ 0.5 waves are too long (centroid wavelength 31 against 14 in the DNS), although the h PDF and crest heights are close. `ml_sum_hhx` fails there (crests ×2).
- **Lavalle** under-predicts crests in every case (ratio 0.72–0.94) and gives negative skewness where the DNS is positive. **V1** over-predicts crests by 37–47% at β 90 and Re ≥ 120.
- **Seed 1 (ROM tests, 19 Nov):** comb growth error against OS within ±1 percentage point of seed 0 (median 1.8–1.9% at Re 20, 5–6% at Re 120, about 9% at Re 200). Oracle h error at t = 50 / 100: σ 0.5 0.37 / 0.75 (seed 0 0.39 / 0.62), σ 0.25 0.48 / 0.94. σ 0.5 is better than σ 0.25 for both seeds; the t = 100 error varies by about 0.15 between seeds. Seed-1 broadband runs: job 51936778 (running, 2 at a time).

## Where it fails

- **Phase C results are one seed.** The a posteriori differences between nearby models (σ, λ_P) are large and of both signs (phase drift +7.8 to −7.2 at t = 50). The seed spread is not known; seed 1 is the backup.
- **The broadband held-out test is statistical.** The held-out DNS has h only every 10 time units, so the ROM uses its own noise realisation and only distributions, crests, envelope and wavelength are compared; one ROM realisation per case and closure, 100 snapshots.
- **The flat-film-exact MLP is not guaranteed hyperbolic.** 2 of 14 seed-0 broadband runs blew up (loss of hyperbolicity); a run that fails is a failed closure, not a missing data point.
- **V1 coefficients depend on h only, and S is linear in the gradients.** Training NRMSE (0.22–0.25) is close to held-out, which points to a limit of the form rather than of network capacity (not tested).
- **The L2 constraint covers the forced band only.** Above it (comb ω > 0.9 at Re 120) the growth error reaches 0.3.
- **The flat-film-exact MLP results are two seeds (broadband seed 1 running) and four (σ, λ_lin) pairs.** Its wave-speed error is not yet traced to the linear part (check: model phase speed against OS at ω ≈ 0.134). The shift is a single global cross-correlation over x = 20–760; jumps near t ≈ 80 are probably waves crossing the window ends.
- **The power diagnosis uses 100 snapshots of one case,** and the region shares depend on thresholds.

- **A.1 is a linearisation about h = q = 1, which is not an equilibrium of the MLP-augmented ROM when S₀ ≠ 0.** The growth numbers describe the closure's linear part, not the closed-loop base state. The ROM runs used raw central differences for h_x; the long-wave numbers do not depend on the filter, but finite-k ones would.
- **A.2 covers one case, 100 time units, and crests matched over Δt = 1.** Lavalle's flat crests make its scatter large. Celerity also depends on position (growth versus saturated region), which the h_max bins mix.
- **A.3 uses rule-based regions with thresholds relative to the case's std**, so the region shares differ between cases. Broadband snapshots every 10 time units give spatial statistics only. The projections are for the q-equation; the coupled interpretation is long-wave only.
- **A.4 uses a single frequency**, so the linear coefficients are not separately identifiable. The window x < 60 is a guess (B.5). The mean offset of S is unexplained.
