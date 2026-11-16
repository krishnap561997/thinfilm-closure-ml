# Plan: Orr–Sommerfeld reference and a structured closure for ΔM_x + ΔT_w

_Written 16 Nov 2026 in Claude (chat) with Krishna, from repo `main` @ 719b7da, for Claude Code on HiPerGator to carry out. First task: put this file in the repo at `docs/plans/structured-closure-os-plan.md` (task 0.0)._

Read first: `CLAUDE.md`, `docs/plans/functional-form-plan.md` (the 13 Nov plan, called "FF plan" below), `docs/results/ml-a-posteriori.md`, `docs/results/functional-form-structure.md`, `docs/results/ordering-matrix.md` §3, `rom/lavalle-cpp/README.md`.

---

## 0. Why this plan exists

Three concerns about the current closure (the 1×16 MLPs and the sparse forms), raised on 14 Nov:

1. **Linear regime.** For x < 100 the waves are small and linear stability theory applies. The augmented ROM should reproduce it. Nothing so far tests this, and nothing in the MLP enforces it.
2. **Amplitude, not celerity.** At Re 120 Lavalle's main error is the wave amplitude; its celerity is roughly right. The closure should be built so that it corrects growth and amplitude and leaves the long-wave speed alone.
3. **Numerical stability.** The ML term is added as a pointwise source with central-differenced gradients. It bypasses the Rusanov upwinding, and nothing guarantees hyperbolicity.

A fourth point: a 1×16 tanh network is not sparse in a monomial library, so "shallow MLP ⇒ easy symbolic form" does not follow. The per-case fit of the sum reaches NRMSE 0.134 while the pooled sparse form stays at 0.35–0.40, so the gradient structure is easy and the parameter dependence is hard.

This plan has two parts:

- **Part I: an Orr–Sommerfeld (OS) solver** as the linear ground truth, used to test the DNS, the ROM and the closure in the linear regime.
- **Part II: a structured closure** (flux part + relaxation part) that is hyperbolic by construction, preserves the uniform film, can be constrained to the OS linear behaviour, and whose components are 1–2D functions that symbolic regression can handle.

### Preliminary result that motivates Part I (16 Nov, to be reproduced in task B.3)

Temporal dispersion at Re 120, Ca 0.01, β 45°, from the prototype in Appendix A (verified only against the Benney limit so far) against the linearised Lavalle model:

| k | OS c_r | OS growth | Lavalle c_r | Lavalle growth |
|---|---|---|---|---|
| 0.04 | 1.951 | 0.0151 | 1.900 | 0.0152 |
| 0.08 | 1.805 | 0.0221 | 1.758 | 0.0203 |
| 0.15 | 1.731 | 0.0287 | 1.702 | 0.0227 |
| 0.30 | 1.718 | 0.0338 | 1.715 | 0.0207 |
| 0.50 | 1.789 | 0.0274 | 1.797 | 0.0142 |

- Lavalle's phase speed is within about 3% of OS at every k.
- Lavalle's growth rate is right for long waves and too low for shorter ones: about 8% low at k ≈ 0.08 (close to the 10 Hz forcing, ω ≈ 0.134) and about 40% low at k ≈ 0.3, where OS is most amplified. Lavalle's most amplified wavenumber is about 0.15–0.2, against about 0.3 for OS.
- So, in the linear regime of this case, the missing physics is growth and not speed. This agrees with concern 2, and it means the correction **must have a nonzero linear part** at this Re. A closure that is purely nonlinear ("Lavalle-linear") would keep Lavalle's growth-rate error.
- Caveats: temporal analysis (the DNS is a spatial problem), passive gas, one case, prototype code.

## 1. Background formulas

### 1.1 Long-wave reduction of the augmented ROM

Linearise about the Nusselt state: h = 1 + η, q = 1 + ξ. Write the linear part of S = ΔM_x + ΔT_w (the ROM adds −S) as

S_lin = a (ξ − 3η) + b η_x + c ξ_x + (higher derivatives).

Eliminating ξ at long wavelength gives η_t + 3η_x = −D η_xx with

**D = (3.6 − 3 cot β/Re − b − 3c) / (3/Re + a).**

- a = b = c = 0 gives D = 1.2 Re − cot β, the Benney value (Lavalle is exact at threshold).
- A term in S that is nonzero on a uniform film (S(h, h³, 0, 0) ≠ 0) changes the base state and the kinematic speed 3. It must be excluded by construction.
- Relaxation (a) and gradient terms (b, c) change growth. Odd higher derivatives change dispersion, i.e. the speed of short waves.
- b + 3c < 0 or a < 0 increases growth.

### 1.2 What we know about the current closed-loop failure

From `ml-a-posteriori.md`: the sum model (inputs h, h_x) matches the oracle for about 10 time units, then over-amplifies (max h 2.50 against 2.08 at elapsed 99.9). It has no q input, so it has no restoring response when the ROM's q drifts. Whether that is the cause is a hypothesis; task A.1 and Part II test it.

## 2. Part I — Orr–Sommerfeld solver

### 2.1 Scope

**What it is.** The exact linear stability problem for 2D perturbations of the Nusselt film: Navier–Stokes linearised about U(z) = 3z − 3z²/2 with a deformable free surface, passive gas, gravity and surface tension. No long-wave assumption, so it is valid for any k and any Re for which the base flow is laminar.

**What it gives, per case (Re, Ca, β):**

- temporal dispersion c(k): phase speed c_r and growth rate k c_i;
- spatial dispersion k(ω): spatial growth rate −k_i and phase speed ω/k_r for a forced frequency (the DNS situation);
- neutral curve, cut-off wavenumber and frequency, most amplified wavenumber and frequency;
- the eigenfunction φ(z), hence u′(z), w′(z), p′(z) of the growing mode.

**What it does not cover.**

- Finite amplitude: it holds only while the waves are small (roughly x < 50–100 at F₀ = 0.025; task B.5 measures where it stops).
- The gas phase: passive-gas boundary conditions, like the ROM. If the DNS gas matters, the DNS will differ slightly (check the density and viscosity ratios in `filmflow0.c`; a two-layer OS is a later extension, not planned).
- Spanwise (3D) modes: not needed, the DNS is 2D.
- Non-parallel effects at the inlet: neglected (the inlet imposes U_N, so the base flow is parallel).

### 2.2 Equations (project nondimensionalisation, status doc §1)

Stream function ψ = φ(z) exp[ik(x − ct)], u′ = φ′, w′ = −ikφ, interface η = η̂ exp[ik(x − ct)]. U_s = 3/2, U″ = −3.

- **Bulk:** φ⁗ − 2k²φ″ + k⁴φ = ik Re [(U − c)(φ″ − k²φ) − U″φ].
- **Wall (z = 0):** φ = 0, φ′ = 0.
- **Kinematic (z = 1):** (c − U_s) η̂ = φ(1).
- **Tangential stress (z = 1):** φ″ + k²φ + U″ η̂ = 0.
- **Normal stress (z = 1):** φ‴ − 3k²φ′ − ik Re (U_s − c) φ′ − ik (3 cot β + k²/Ca) η̂ = 0.

Keeping η̂ as an extra unknown makes this a linear generalised eigenproblem A x = c B x with x = (φ, η̂).

### 2.3 Tasks

Code in `analysis/orr_sommerfeld/` (pure Python: numpy, scipy). Results in `docs/results/orr-sommerfeld.md` with a "Where it fails" section. Everything here runs in seconds on a login-node-safe core count; use `hpg-dev` for the batch over cases.

- **B.1 Temporal solver.** Chebyshev collocation on z ∈ [0, 1], N = 60–100, generalised eigenproblem as in Appendix A. Return the least stable physical mode; filter spurious eigenvalues by convergence between two N.
- **B.2 Spatial solver.** For real ω find complex k with ω(k) = ω: Newton or secant on k, calling the temporal operator with complex k, started from the temporal solution (Gaster estimate k_i ≈ −ω_i/c_g). Track the root continuously in ω.
- **B.3 Verification.**
  1. Benney limit: growth → k²(6/5 Re − cot β) − k⁴/(3Ca), c → 3 as k → 0 (the prototype passes for Re 0.5–10).
  2. Critical Reynolds number Re_c = (5/6) cot β.
  3. One published curve: Liu, Paul & Gollub (1993) or Brevdo et al. (1999) spatial growth rates, or the tables in Kalliadasis et al. (book, ch. 3). Convert their scaling to ours and state the conversion.
  4. Convergence in N; reproduce the table in §0.
- **B.4 Case table.** For all cases in `datasets/MANIFEST_dns.csv`: k_max, ω_max, cut-off k_c and ω_c, c_r at k_max, spatial growth at ω_max. Convert ω_c to Hz per case and compare with the forcing cut-off f_cut = 28 Hz (this closes the open "check f_cut against 2f₀" item). Report ε_lin = k_max as the per-case long-wave parameter and δ_lin = ε_lin Re (input to FF plan D2; note that this uses OS, not the linearised Lavalle model, and compare the two).
- **B.5 DNS check in the linear region.** For data with dense-in-time 1D profiles (the oracle run, profiles every 0.01; the dense restarts under `BasC/DNS/dense/`): Fourier-transform h(x, t) in time at each x, and for each forced frequency fit the amplitude envelope exp(−k_i x) and the phase k_r x over the range where the fit is exponential. Compare −k_i(ω) and ω/k_r with OS. Report the x at which each frequency departs from linear growth (this defines the "linear region" per case; the x < 100 figure is so far a guess).
  - The production broadband runs store snapshots every 10 time units, too coarse for this. Extra short restarts with profiles for x < 150 are decision D3; do not submit DNS without approval.
- **B.6 ROM check in the linear region.** No derivation needed: run `lavalle-cpp` with a small monochromatic inlet (amplitude 1e-4) at about 8 frequencies across the unstable band, and measure −k_i and ω/k_r the same way. Do it for (i) Lavalle, (ii) `ml_sum_hhx`, (iii) every Part II candidate. Also code the analytic linearised-Lavalle dispersion (the quadratic in Appendix A) as a cross-check of the method. This is the linear-regime test of concern 1.
- **B.7 Linear closure targets.** From the eigenfunction (normalised to η̂ = 1), compute the linear response of each closure term along the OS mode and subtract the linearised Lavalle closure:
  - q′ = φ(1) + U_s η̂ (check: equals c η̂);
  - (∫u²)′ = ∫₀¹ 2Uφ′ dz + U_s² η̂; T_w′ = φ″(0)/Re; P_x′ from p′ via the x-momentum equation.
  - Output: complex ΔM_x′/η̂, ΔT_w′/η̂, ΔP_x′/η̂ against k per case. The real and imaginary parts are the in-phase and quadrature parts of each correction relative to h, i.e. which part changes growth and which changes speed (concern 2).
  - Compare with the DNS corrections in the linear region from B.5 (amplitude ratio and phase per frequency). This validates the extraction pipeline (Savitzky–Golay or Gaussian derivatives) where the truth is known exactly.
- **B.8 Linear coefficients for the closure.** Fit (a, b, c, and optionally one h_xx coefficient d) per case so that the linearised augmented ROM matches the OS spatial dispersion over the forced band (weights: uniform in ω up to ω_c). Report how well four numbers can do it, and how they vary with (Re, Ca, β). These are the targets for constraint L2 in §3.3.
- **Gate GB:** B.3 passes. Krishna reviews B.4–B.6 and decides D2.

## 3. Part II — structured closure

### 3.1 Phase A: cheap diagnostics on existing data (do these first)

Scripts in `scripts/structured-closure/`; results in `docs/results/structured-closure.md`.

- **A.1 The current MLP at the uniform state.** Evaluate `ml_sum_hhx` and the `ml_mx_tw` pair at h = 1, q = 1, zero gradients for each case: the offset S₀ and the slopes (a, b, c). Compute D (§1.1) with and without them. A nonzero S₀ means the ROM's uniform film is not steady.
- **A.2 Amplitude against celerity.** From the existing oracle-case runs (`lavalle`, `Tw_Mx`, `ml_sum_hhx`, `ml_mx_tw`) and the DNS: track crests, and plot crest celerity against crest height for each. If all lie on one curve, the celerity error is a consequence of the amplitude error.
- **A.3 Phase projection of the DNS correction.** Crest-aligned, in the frame moving with each wave: project −S on q′ = q − mean(q) (changes amplitude) and on ∂ₓq′ (changes speed), split by region (front, crest, back, capillary ripples, substrate). Do it for the oracle case and for 4–6 broadband cases spanning δ. This answers "what sign, and where".
- **A.4 Linear-region regression.** In the linear region of the oracle data (limit from B.5; until then x < 60), regress the DNS S on (ξ − 3η, η_x, ξ_x) to get empirical (a, b, c); compare with B.8.

### 3.2 The model

The ROM momentum equation becomes

∂ₜq + ∂ₓ[ Φ q²/h + P_Lav(h) + ΔP(h) ] = κ (Q − q)/h² + (1/We) h h_xxx + (2/Re) q_xx

with, for parameters p = (Re, Ca, β) (fed as log Re, log Ca, cot β; a variant with δ_lin from B.4 as an extra input):

- **Flux part.** Φ ≥ 1 and P_Lav′ + ΔP′ ≥ 0.
  - Variant **V1 (guaranteed):** Φ = Φ(h; p) = 1 + softplus(N_Φ) with N_Φ non-decreasing in h (non-negative weights on h). The discriminant is 4u²[Φ(Φ − 1) + hΦ_h] + 4(P_Lav′ + ΔP′) ≥ 0, so the system is hyperbolic for every state.
  - Variant **V2 (penalised):** Φ = Φ(h, r; p), r = q/h³. Not guaranteed; add a training penalty on discriminant < margin over a box of states (training range padded by 30%), and the runtime limiter of §3.5.
  - ΔP: parametrise P_tot′(h; p) = softplus(N_P) and integrate from h = 1 by quadrature (the constant is irrelevant).
- **Relaxation part.** κ = (3/Re) exp(N_κ(h; p)) > 0. Q = h³ (1 + G), G = g₁(h; p) h_x + g₂(h; p) q_x (+ g₃(h; p) h_xx, optional), so Q = h³ on a uniform film. Gradients are the Gaussian-filtered ones, D_G(σ = 0.5 h_N), as in the FF plan §2.2.
- **Lavalle is the zero of every network:** N_Φ → −∞ (Φ = 1), ΔP = 0, N_κ = 0, g = 0. Initialise there.
- **Implied S**, for fitting and for comparison with everything else:
  S = ∂ₓ[(Φ − 1) q²/h + ΔP] − [κ (Q − q) − (3/Re)(h³ − q)]/h², with the x-derivative expanded by the chain rule (autograd), not differentiated numerically.
- **Linear coefficients at the Nusselt state** (for §3.3), with F = (Φ − 1)q²/h + ΔP:
  a = κ(1) − 3/Re; b = F_h − κ(1) g₁(1); c = F_q − κ(1) g₂(1).
- Each network is 1×16 (2×16 as a sensitivity). Every learned function has one state argument (h) plus p, which is what makes Phase E possible.

What this gives by construction: the uniform film stays a solution; the long-wave speed 3 is untouched; relaxation is always toward Q; V1 is always hyperbolic; the flux is conservative and goes through the Riemann solver.

What it does not give: Lavalle's energy inequality. The correction has to add energy to raise the crests, so no dissipative bound applies. Stability beyond hyperbolicity is tested, not proved.

### 3.3 Phase C: training

Code in `ml/structured_closure/`. Data: cache v2 of FF plan task 1.1 (σ = 0.5; fields h, q, h_x, q_x, h_xx, ΔF, `delta_Mx_ML`, `delta_Tw_ML`), same split (`ml/functional_mlp/data_config.json`: 15 development, 7 held-out), full time record. If cache v2 does not exist yet, build it first (shared with the FF plan).

- **C.1 Loss.** ‖S_pred − S‖² with the per-case 1/rms weighting of `fit_sparse.py`, plus:
  - λ_F ‖(Φ − 1)q²/h + ΔP − ΔF_data − const‖², an optional anchor of the flux part to the measured flux defect, λ_F ∈ {0, 0.1, 1};
  - the linear constraint, one of **L0** none; **L1** (a, b, c) = 0, "Lavalle-linear"; **L2** (a, b, c) = OS targets from B.8. Default L2 once B.8 exists; L0 until then. Given §0, L1 is a comparison row, not a candidate, unless B.5–B.6 contradict the prototype.
  - V2 only: the hyperbolicity penalty.
- **C.2 Runs.** V1 and V2 × {L0, L2} × λ_F, three seeds each, CPU or at most 2 GPU jobs at a time. Then ablations: no flux part; no relaxation part; g₃ on; δ_lin as input.
- **C.3 A priori evaluation.** Metrics M1 and M2 of the FF plan §2.3 on LOCO (if affordable), held-out and β-holdout, with the reference rows: Lavalle (S = 0), `ml_sum_hhx`, the separate 1×16 models summed, and the best sparse form. Also report per case: (a, b, c), the minimum discriminant over DNS states, and plots of Φ, P_tot′, κ, g₁, g₂ against h.
- **Expectation to state honestly:** the a priori error will probably be above the unconstrained MLP's. The claim to test is closed-loop behaviour, not a priori accuracy.

### 3.4 Phase C (continued): the solver

`rom/lavalle-cpp`, lines tagged `// STRUCTCLOSURE`. The Rusanov flux already takes a shape factor and its derivatives into the wave speed (`ClosurePoint::phi_sf`, `dphi_sf_dh`, `dphi_sf_dq` in `hyperbolic_flux.cpp`), so the flux part is a small change.

- **C.4** Add `dP`, `dP_dh` to `ClosurePoint`; use them in the flux and in `characteristics()`.
- **C.5** `StructuredClosure : ClosureModel`. The parameters are fixed in a run, so for V1 export 1D tables in h (Φ, Φ_h, ΔP, ΔP′, κ, g₁, g₂, g₃ on h ∈ [0.01, 5], 5001 points, cubic interpolation) from Python (`export_tables.py`). V2 evaluates the small network directly.
- **C.6** Relaxation source: replaces the Lavalle friction (L h − τ_w)/Re in `source_terms.cpp` when the model is active. Filtered gradients from the existing `gauss_deriv` (same operator as training).
- **C.7** Training-range envelope: outside [h_lo, h_hi] (0.1 and 99.9 percentiles of the training h, padded 10%) blend every table smoothly to its Lavalle value over a width of 0.1 in h; clip the filtered gradients entering G at their training 99.9 percentiles. Because the blend is a function of h only, the flux stays a function of state. Log how often the envelope is active.
- **C.8** V2 only: `hyp_limiter = 1` blends Φ → Φ(h) of V1 or to 1 at faces where the discriminant is below a margin, and counts the events. V1 keeps the existing throw, which should never fire.
- **C.9** Stiffness monitor: log max κ dt/h². Only if it exceeds 0.5, treat the −κ q/h² part point-implicitly (separate task, ask first).
- **C.10 Tests.**
  1. Lavalle tables → output byte-identical to the default solver.
  2. S from C++ against Python on the DNS state at t = 1800 and 1850, ≤ 1e-10 (extend `scripts/ml_hook_check.py`).
  3. Uniform film h = q = 1, unforced, 100 time units: stays uniform to round-off.
  4. Linear test B.6.
- **Gate GC:** tests 1–3 pass.

### 3.5 Phase D: closed loop

Oracle case (Re120 Ca0.01 β45, 10 Hz, from t = 1800, 100 time units), harness of FF plan task 1.3 if it exists, otherwise `scripts/oracle_summary.py`.

- Runs: the C.2 candidates; reference rows `lavalle`, `Tw_Mx` (oracle), `ml_sum_hhx`.
- Metrics: E_h and Ē_h (FF plan §2.4), max h, h_min, minimum discriminant, plus three new ones:
  - linear region: −k_i and ω/k_r of the forcing frequency from B.6, against OS and DNS;
  - crest celerity against crest height (A.2);
  - envelope and limiter activity.
- With and without `reinit_every = 10` (FF plan 1.3), to separate closure error from phase drift.
- Selection: the FF plan §2.5 definition of "beats the ML sum", plus: stable for 100 time units and linear growth within 10% of OS over the forced band.
- Then broadband: 2–3 cases, statistics only (h PDF, crest heights, celerity, spectra) as in FF plan task 3.5, and the spatial growth of each frequency band for x < x_linear against OS.
- **Gate GD (about 6 Dec):** Krishna decides which closure Paper 1 carries (D1).

### 3.6 Phase E: closed forms for the components

Only for the selected model.

- For each learned function (Φ, P_tot′, κ, g₁, g₂) and each case: the curve against h over the training range. Fit low-order rational or power-law forms in h; then model the coefficients against p (and δ_lin). PySR as a cross-check (test `import pysr` with `envs/pysr_env.sh` first; cap the compute).
- Compare with theory at low δ: the row-1 q-gauge closures (ordering matrix §3.2) written in this form, and the Tier-1 coefficients of the FF plan §3.1.
- Put the closed forms back into the solver (tables from formulas) and rerun Phase D. The functional form of Paper 1 is this set of formulas if it holds up in closed loop.

### 3.7 Phase F: rollout training (conditional; ask before starting)

Only if the selected model still drifts in closed loop where the oracle does not.

- A differentiable copy of the ROM numerics (PyTorch or JAX): Rusanov + MUSCL + RK4 + Noble–Vila. It must match `lavalle-cpp` on a Lavalle run to 1e-10 before use.
- Loss: h and q error against dense DNS profiles after rollouts of 2–5 time units, started from DNS states. Dense data today: the oracle case and the `BasC/DNS/dense/` restarts.
- Fine-tune from the a priori model; keep the constraints of §3.2.

## 4. Schedule (alongside the FF plan; DFD talk 22 Nov)

| Dates | Work |
|---|---|
| 16–22 Nov | 0.0; Phase A; B.1–B.4 (all cheap, no DNS) |
| 23–29 Nov | B.5–B.8; gate GB; C.1–C.3 (needs cache v2) |
| 30 Nov–6 Dec | C.4–C.10; Phase D on the oracle case; gate GD |
| 7–20 Dec | Phase D broadband; Phase E; Phase F if approved |
| From 4 Jan | as in the FF plan (draft to co-authors, submission 15 Jan 2027) |

## 5. Decisions needed from Krishna

- **D1:** how this relates to the FF plan. Proposed: both run; the monomial route S of the FF plan stays as the theory-library baseline, and the Paper 1 closure is whichever wins in closed loop at gate GD.
- **D2:** the linear target, after gate GB: L2 (match OS) as proposed, or L1 if the DNS and the ROM disagree with the prototype table in §0.
- **D3:** short DNS restarts with dense profiles for x < 150 on a few broadband cases, for B.5 and B.7 (cost to be estimated in B.5; nothing submitted without approval).
- **D4:** V1 only, or V2 as well, if V1's a priori error is much worse.
- **D5:** rollout training (Phase F) before submission, or left for later.
- **D6:** whether the OS comparison goes into the Paper 1 validation section (2.3).

## 6. Rules (from `CLAUDE.md`)

- At most 2 GPU jobs at a time and 64 CPUs on qos `bala1s`; `bala1s-b` beyond that; `hpg-dev` for short tests.
- No data, figures, logs or caches in git. Never `git pull` in or move the data store. Outputs under `$THINFILM_STORE/codes/ml-models/structured_closure/<run>/` and `$THINFILM_STORE/analysis/orr_sommerfeld/`.
- No new DNS without approval. DNS builds use `~/basilisk-old/src`; restarts link the dump as `restart.dump`.
- Update `project-status.md` and `decisions.md` in the same commit as the code that changes the science. Every results write-up has a "Where it fails" section, and reports numbers with the data window, metric and split.
- Work on a branch `structured-closure`, one PR per phase.

## 7. Risks

- **A priori accuracy of the structured model** may be clearly below the MLP, especially V1 (Φ depends on h only). Mitigation: V2, g₃, δ_lin input; the comparison that matters is closed loop.
- **Four linear coefficients may not match OS** over the whole band (B.8). Then report the mismatch and weight the fit toward the forced and most amplified frequencies.
- **The linear region in the DNS may be short** at F₀ = 0.025, leaving few wavelengths for B.5. Mitigation: D3 with a smaller forcing amplitude.
- **Gas-phase effects** could make the DNS differ from passive-gas OS by a few percent; B.5 on the oracle case will show the size.
- **One dense closed-loop case.** As in the FF plan §8: the closed-loop evidence rests on Re120 Ca0.01 β45 until more monochromatic cases exist.
- **The prototype numbers in §0** have not been checked against a published curve yet (B.3).

## Appendix A: prototype temporal solver and Lavalle dispersion (16 Nov)

Passes the Benney limit for Re 0.5–10 at k = 0.01 (growth within 1–3%). Not yet checked against published curves. Starting point for B.1.

```python
import numpy as np, scipy.linalg as la

def cheb(N):
    x = np.cos(np.pi*np.arange(N+1)/N)
    c = np.r_[2, np.ones(N-1), 2]*(-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1)).T
    D = np.outer(c, 1/c)/(X - X.T + np.eye(N+1))
    return D - np.diag(D.sum(1)), x

def os_temporal(k, Re, Ca, cotb, N=60):
    D, x = cheb(N); z = (x+1)/2; D = 2*D      # z[0] = 1 (surface), z[N] = 0 (wall)
    D2 = D@D; D3 = D2@D; I = np.eye(N+1)
    U = 3*z - 1.5*z**2; Upp = -3.0; Us = 1.5
    n = N+2                                    # unknowns: phi (N+1 values), eta
    A = np.zeros((n, n), complex); B = np.zeros((n, n), complex)
    L = D2 - k*k*I
    A[:N+1, :N+1] = L@L - 1j*k*Re*(np.diag(U)@L - Upp*I)
    B[:N+1, :N+1] = -1j*k*Re*L
    A[N] = 0; B[N] = 0; A[N, N] = 1                         # phi(0) = 0
    A[N-1] = 0; B[N-1] = 0; A[N-1, :N+1] = D[N]             # phi'(0) = 0
    A[0] = 0; B[0] = 0                                      # tangential stress
    A[0, :N+1] = D2[0] + k*k*I[0]; A[0, N+1] = Upp
    A[1] = 0; B[1] = 0                                      # normal stress
    A[1, :N+1] = D3[0] - 3*k*k*D[0] - 1j*k*Re*Us*D[0]
    A[1, N+1] = -1j*k*(3*cotb + k*k/Ca)
    B[1, :N+1] = -1j*k*Re*D[0]
    A[N+1, 0] = 1; A[N+1, N+1] = Us; B[N+1, N+1] = 1        # kinematic
    w = la.eigvals(A, B); w = w[np.isfinite(w) & (abs(w) < 50)]
    return w[np.argmax(w.imag)]                             # complex c; growth = k*c.imag

def lavalle_temporal(k, Re, Ca, cotb):
    We = Re*Ca; G = -0.6 + 3*cotb/Re
    r = np.roots([-1j*k, 2j*k + 3/Re + 2*k*k/Re, 1j*k*G - 9/Re + 1j*k**3/We])
    return r[np.argmax(r.imag)]
```
