# What we did on 16 Nov 2026, and why

_Written for Krishna, to understand the work delegated on 16 Nov: the Orr–Sommerfeld reference, the linear-regime tests, the diagnosis of the current ML closures, the comb-forced DNS and cache v2. Technical results: `docs/results/orr-sommerfeld.md` and `docs/results/structured-closure.md`. Task status and job IDs: `docs/plans/structured-closure-progress.md`._

---

## Part 1. What we did, and the main reasons

- **Built an exact linear-stability solver (Orr–Sommerfeld, OS) for the falling film and verified it.**
  - Small waves on a flat film obey linear theory exactly, and OS is that theory with no long-wave assumption. It gives the "true answer" for how fast each wave frequency grows and how fast it travels.
  - It is the only ground truth we have that does not come from our own DNS or ROM, so it can check both.
  - It turns the plan's first concern ("does the augmented ROM get the linear regime right?") into a testable number.

- **Tabulated the OS growth, speed and cut-off for all 85 cases (B.4).**
  - It shows where Lavalle's linear error is large. The error depends on one parameter, δ_lin = k_max·Re: Lavalle is 7–18% too unstable at small δ and 25–45% too stable at large δ.
  - It answered the open question about the 28 Hz forcing cut-off. In many high-Re cases the DNS forcing does not reach the most amplified frequency.

- **Diagnosed the current MLP closures at the flat film (Phase A, A.1–A.4).**
  - The closed-loop runs drift after about 10 time units, and we needed to know why before designing a replacement.
  - Finding: neither MLP gives zero correction on a flat film (it was never in their training data). They shift the long-wave speed, and they get the linear growth wrong (the separate-models MLP doubles it).

- **Checked the DNS against OS in the linear region (B.5) and the extracted closure terms against the exact OS values (B.7).**
  - If the DNS or the extraction were wrong for small waves, everything built on them would be suspect.
  - Result: growth agrees to 1%, and the extracted ΔM_x and ΔT_w agree to 3–6%. This validates the DNS and the extraction pipeline where the truth is known.

- **Tested the ROM (Lavalle, and Lavalle + MLPs) in the linear regime with a frequency-comb inlet (B.6).**
  - The goal is to separate errors of the *model* from errors of the *numerics*, and to measure what the MLPs actually do to linear waves in the solver.
  - Result: the numerics are accurate (about 1%). The MLPs miss OS by 20–28%, and at higher Re they leave the linear regime altogether.

- **Computed the linear targets for a new closure: four numbers (a, b, c, d) per case (B.8).**
  - They are the constraint that makes a learned closure reproduce OS for small waves (decision D2 = L2).
  - Four numbers are enough: within 10% of OS growth over the forced band in every case.

- **Investigated the "inlet problem".**
  - It looked as if the ML-augmented ROM blew up near the inlet.
  - It was a bookkeeping artefact in the output (fixed). The real causes were excess linear growth and *spurious flat-film equilibria* of the MLPs.

- **Set up and launched five short comb-forced DNS runs (decision D3).**
  - The existing DNS check of linear theory uses only one case at one frequency (10 Hz).
  - These runs test about 20–30 frequencies in each of five cases spanning δ, and they let us fit (a, b, c, d) from the DNS itself.

- **Started cache v2: training data with full time records and the same derivative operator as the ROM.**
  - The next step, training the new structured closure, needs it.
  - It also removes two problems of the old cache: missing flat-film samples, and a derivative filter different from the solver's.
  - Along the way: re-extracted Re050 and four out-of-date closure files, and extracted seven new cases.

---

## Part 2. Concepts you need

This section builds up the ideas in the order they are used. Symbols follow the project scaling: lengths in h_N, velocities in u_N (the mean Nusselt velocity), time in T_N = h_N/u_N (2.14 ms for Re120 Ca0.01 β45).

### 2.1 The base state: the flat (Nusselt) film

- A film of uniform thickness h = 1 flowing steadily has the half-parabolic velocity profile U(z) = 3z − 1.5z² (surface velocity 1.5, mean 1). Its flow rate is q = 1. In general a flat film of thickness h carries q = h³.
- This is the state that inlet noise destabilises. Every wave we see in the DNS started as a tiny disturbance of it.
- Anything we add to the ROM must leave this state alone: a flat film must stay flat and steady. That sounds trivial, but the current MLPs violate it (§3.3).

### 2.2 Linear waves and the dispersion relation

- Disturb the flat film by a tiny wave, h = 1 + η with η = ε·exp[i(kx − ωt)]: wavenumber k (2π/wavelength) and angular frequency ω.
- While ε is small, the equations can be linearised: products of small quantities are dropped. Then each k behaves independently, and the equations allow only certain pairs (k, ω). That relation is the **dispersion relation**.
- k and ω can be complex. The imaginary parts are growth or decay:
  - η ∝ exp(ω_i t) grows in time if ω_i > 0;
  - η ∝ exp(−k_i x) grows in space if k_i < 0.
- **Phase speed** c = ω/k_r is how fast crests move. **Group velocity** c_g = dω/dk is how fast a packet of waves, and its energy, moves.

### 2.3 Temporal versus spatial stability, and why we use the spatial one

- **Temporal problem:** k is real (a periodic disturbance in x), and we ask how fast it grows in time. The answer is a complex c, with growth rate k·c_i per time unit. This is the textbook form.
- **Spatial problem:** ω is real (we shake the inlet at a fixed frequency), and we ask how the wave grows as it travels downstream. The answer is a complex k, with spatial growth rate −k_i per unit length. **This is what the DNS and the ROM do:** they are forced at the inlet. All our comparisons with simulations use spatial rates.
- The two are related approximately by −k_i ≈ ω_i/c_g (Gaster's relation), but we solve the spatial problem exactly.

### 2.4 The characteristic numbers of an unstable film

- **Cut-off (neutral) wavenumber k_c and frequency ω_c.** Shorter, faster waves are damped by surface tension. Above ω_c every disturbance decays.
- **Most amplified wavenumber k_max (temporal) and frequency ω_max (spatial).** These are the waves that win if the inlet noise is broadband.
- **Long-wave limit (k → 0).** Growth ≈ k²(6Re/5 − cot β) − k⁴/(3Ca). The first term is inertia (destabilising) against the cross-stream component of gravity (stabilising). The second is surface tension. Two consequences:
  - Instability needs Re > Re_c = (5/6) cot β.
  - Very long waves travel at the **kinematic speed 3**, which is dq/dh for q = h³ at h = 1.
  - This is Benney's result. It is exact as k → 0, and any consistent long-wave model, including Lavalle's, reproduces it.
- **δ_lin = k_max·Re.** A measure of how much inertia acts over one unstable wavelength, our version of the "reduced Reynolds number" in Kalliadasis et al.
  - Small δ means long-wave models are accurate.
  - Large δ means first-order models such as Lavalle's lose accuracy, because second-order effects (viscous dispersion, inertia corrections) matter.
  - Our cases span δ ≈ 1–66.

### 2.5 What the Orr–Sommerfeld problem is

- It is the exact linearised Navier–Stokes problem for 2D disturbances of the Nusselt film. The disturbance is written with a stream function ψ = φ(z)·exp[ik(x − ct)]; φ(z) is its vertical shape. Then:
  - u′ = φ′ and w′ = −ikφ;
  - the interface moves by η̂·exp[ik(x − ct)].
- **The equation is fourth order in z**: φ⁗ − 2k²φ″ + k⁴φ = ik·Re[(U − c)(φ″ − k²φ) − U″φ].
- **Boundary conditions:**
  - no slip at the wall (φ = φ′ = 0);
  - at the free surface, the kinematic condition (the interface moves with the fluid);
  - zero tangential stress (passive gas);
  - the normal-stress balance: pressure, gravity and surface tension.
- Only special values of c satisfy all of these, so it is an **eigenvalue problem**. Among the solutions, the least-stable **surface mode** is the one that produces the film's long waves.
- **It differs from Benney and Lavalle** because it makes no long-wave or depth-averaging assumption. It is exact for any wavelength, as long as the waves are small and the gas is passive.
- **The eigenfunction φ(z) is the full velocity profile of the wave.** From it we can compute the *exact* depth-integrated quantities of a small wave: q′, (∫u²)′, the wall shear τ_w′ and the pressure integral. That gives the exact linear values of the closure terms ΔM_x, ΔT_w, ΔP_x (B.7). For small waves, this is the closure the ML must learn.
- **How we solve it numerically:**
  - Chebyshev collocation: represent φ by a polynomial in z and enforce the equations at special points. This turns the ODE into a matrix eigenproblem.
  - Matrix methods also produce spurious eigenvalues, so we keep only those that agree at two resolutions.
  - Each eigenvalue is then refined by **shooting**: integrate the ODE from the wall and adjust c until the surface conditions are met. That is an independent second method, and the two agree to about 1e-7.
  - One numerical lesson: the raw matrices are badly scaled. Without rescaling the rows, adding resolution made the answer *worse* (round-off).
- **How we know it is right (B.3):**
  - Benney's long-wave growth, with the error vanishing as k²;
  - Re_c = (5/6) cot β to 1e-5;
  - the published OS saddle points of Brevdo et al. (1999), tabulated in Kalliadasis et al. (Table 7.1), reproduced to the printed digits (see §2.6);
  - convergence in resolution;
  - the prototype table in the plan.
  - Kalliadasis et al. use a velocity scale 3u_N, so their ω, c and V are three times ours, and their We is 1/(3Ca).

### 2.6 Convective versus absolute instability (saddle points)

- **Convectively unstable:** a disturbance grows but is carried downstream, so at a fixed position the flow returns to flat once the disturbance has passed. Films on inclines are convectively unstable. That is why the inlet noise matters: it is amplified as it travels.
- **Absolutely unstable:** a disturbance grows in place and eventually fills the domain.
- **How to tell:** the answer lies in the *saddle points* of ω(k) in the complex k-plane. Brevdo et al. computed them for an experimental case, and reproducing their numbers tests our solver for complex k, the same machinery the spatial problem needs.
- **The same idea guided the inlet diagnosis (§3.6).** Growth in time at a fixed position near the inlet is not what a convectively unstable film does. That pointed to the boundary or the closure, not the physics.

### 2.7 The reduced model's own dispersion relation

- Linearising Lavalle's two-equation model (h, q) about the flat film gives a quadratic for c at each k: the "Lavalle dispersion relation". It reproduces Benney's long-wave limit exactly, but at finite k:
  - **Phase speed:** within about 3% of OS (Re120 Ca0.01 β45). Lavalle's speed is good.
  - **Growth:** too low at moderate k, by 8% near the 10 Hz forcing and by about 40% at the OS maximum k ≈ 0.3. Lavalle's peak growth is at k ≈ 0.17 instead of 0.3.
  - **Over all 85 cases:** the error grows with δ_lin, as listed in Part 1.
- So in the linear regime **the missing physics is growth, not speed**, and it depends on δ. This is concern 2 of the plan, now made quantitative.

### 2.8 How a closure changes the linear behaviour: the coefficients (a, b, c, d)

- The ROM adds −S to its q-equation, with S = ΔM_x + ΔT_w. For tiny waves (h = 1 + η, q = 1 + ξ), any smooth S reduces to its linear part:

  S_lin = a(ξ − 3η) + b∂ₓη + c∂ₓξ + d∂ₓₓη

- **What each term does:**
  - **a, relaxation.** ξ − 3η is the linearised q − h³, how far the flow rate is from the flat-film value for that thickness. Lavalle's wall friction pulls q toward h³ at rate 3/Re, and a changes that rate to 3/Re + a.
  - **b and c, slopes of h and q.** They act like corrections to the pressure term and to the momentum flux.
  - **d, curvature of h.** It matters only for shorter waves.
- **Why this form.** On a flat film (ξ = 3η, no gradients) every term vanishes, so the flat film stays steady and the long-wave speed stays 3. A closure that adds a constant S₀, or a term in η alone, breaks both. That is the A.1 finding about the current MLPs.
- **Long-wave growth.** For long waves the growth is governed by D = (3.6 − 3 cot β/Re − b − 3c)/(3/Re + a), which equals 1.2Re − cot β for Lavalle. B.8 fixes D to the exact OS value (which Lavalle already gets right) and fits (a, b, d) so the ROM reproduces the OS spatial dispersion over the forced band; c follows from the D condition.
- **Result of B.8.** (a, b, c) alone is not enough for δ > 5. Adding d brings the growth within 10% of OS in all 85 cases.
  - Decision **D2 = L2** makes these the training constraint.
  - That is why the h_xx term (g₃) becomes part of the default structured closure.
- **In phase versus quadrature, and a counter-intuitive point.** For a wave η = A·exp[i(kx − ωt)], a correction proportional to η is *in phase* (real part relative to η), and one proportional to ∂ₓη is in *quadrature* (imaginary part).
  - In the coupled h–q system at long wavelength, ξ ≈ 3η.
  - So an in-phase term behaves like a change of the kinematic speed.
  - A quadrature term enters D, so it changes the **growth**.
  - The plan's wording for A.3 had this the other way round (it holds for the q-equation alone, not for the coupled system).

### 2.9 Flat-film equilibria and spurious equilibria

- **The uniform state.** A flat film of thickness h is steady when friction balances gravity, (3/Re)(h − q/h²) = 0, i.e. q = h³. With q fixed by the inlet (q = 1), the only solution is h = 1.
- **What a closure that is nonzero on a flat film does.** The steady flat film then satisfies (3/Re)(h − 1/h²) = S(h). Two consequences:
  - the physical solution moves (for the sum model at Re 200, to h = 1.108, an 11% thicker film);
  - new, spurious solutions can appear. Both MLPs have them at h ≈ 2–3 for every Re tested.
- **Why it matters.** The ROM can lock onto a spurious state. In the B.6 test at Re 200 it did, at h = 2.92 exactly. A crest that reaches h ≈ 2.5 may also be pulled toward a nearby spurious state. That is a hypothesis for the sum model's crest over-amplification in the oracle run (2.5 against the DNS's 2.1). It is not tested yet.
- **The structured closure excludes this by construction:** S = 0 on any flat film, and relaxation always pulls q toward h³.

### 2.10 Hyperbolicity

- The (h, q) system without its diffusive and capillary terms is a pair of first-order conservation laws. They must have real characteristic speeds, i.e. be *hyperbolic*; otherwise the problem is ill-posed and the scheme blows up.
- Lavalle's model is hyperbolic. A learned term inserted as a pointwise source with derivative inputs is outside that structure (concern 3 of the plan).
- The structured closure puts its gradient-dependent part into the flux, (Φ − 1)q²/h + ΔP(h), constrained so the system stays hyperbolic. It goes through the Riemann solver like Lavalle's own flux.

### 2.11 Measuring growth in a simulation: inlet forcing, the comb, projection

- **Monochromatic forcing** (one frequency, as in the oracle run) gives growth at one ω only.
- **The comb.** If the inlet signal is a sum of M cosines at frequencies k·Δω (k = 1…M), the whole signal repeats exactly every T = 2π/Δω.
  - Over exactly one period, the frequencies are orthogonal, so projecting h(x, t) onto exp(iω_k t) gives each mode's complex amplitude A_k(x) with no leakage.
  - Then ln|A_k(x)| against x has slope −k_i (the growth), and the phase of A_k has slope k_r (so the phase speed is ω_k/k_r).
- **The inlet signal both codes already use** (Chang noise, M modes up to f_cut) *is* such a comb. Choosing M and f_cut gives an exact comb with no code change. We used T = 100 time units.
- **Where to fit.**
  - Near the inlet there is an *adjustment zone*. The inlet imposes a velocity profile that is not exactly the wave's own, so for x < 60 the growth is not modal.
  - Beyond roughly |A| ≈ 0.2–0.3 the waves become nonlinear. For the oracle case the clean linear region is x ≈ 60–150.
  - Harmonics (20 Hz, 30 Hz for 10 Hz forcing) are generated nonlinearly by the fundamental. They grow faster and travel at its speed, so they are **bound** waves and OS does not apply to them.

### 2.12 Derivative operators and the training cache

- The DNS closure extraction uses **Savitzky–Golay** (SG) filters for derivatives. The ROM uses a **Gaussian derivative filter** D_G(σ) with σ in units of h_N.
- If the ML is trained on one operator and run with another, its inputs differ between training and use.
- Cache v2 stores the D_G derivatives (σ = 0.25, 0.5, 1.0) computed with exactly the ROM's operator. The check ∂ₓΔF ≈ ΔM_x agrees to 2%, 7% and 19% at those σ. The difference is just the extra smoothing of the Gaussian.
- The old cache also removed all points with |h − 1| < 0.05 (the "flat mask"). That is why the MLPs never learned that S = 0 on a flat film. Cache v2 keeps everything; masks become a training choice.

### 2.13 A priori versus a posteriori

- **A priori:** compare the predicted correction with the extracted one on DNS states (NRMSE).
- **A posteriori (closed loop):** put the closure into the ROM and see whether the ROM's waves match the DNS.
- A closure can be good a priori and still drift in closed loop. Phase A showed why that can happen:
  - wrong behaviour at the flat state;
  - wrong linear growth;
  - spurious equilibria.
- A priori error says nothing about these, which is why the new closure is built with constraints rather than judged by a priori error alone.

---

## Part 3. Detailed report of 16 Nov

Each item gives what was done, why, the outcome, and where to find it. Commits are on branch `structured-closure` (not pushed).

### 3.1 Setup

- **What.** Read the plan (written in chat on 16 Nov) and created branch `structured-closure` from `main`. Committed the plan (task 0.0) and the Kalliadasis et al. book PDF. Copied in the functional-form plan, marked *secondary*.
- **Why.** The plan asks for one PR per phase, kept independent of the functional-form Phase 0 work on `functional-form-v2`.
- **Note.** HiPerGator has no PDF tools. The book's text, including Table 7.1, was extracted with a small standard-library script.

### 3.2 Orr–Sommerfeld solver and verification (B.1–B.3)

- **What.** `analysis/orr_sommerfeld/os_solver.py` provides:
  - temporal modes and spatial branches k(ω);
  - eigenfunctions;
  - saddle points;
  - the analytic linearised Lavalle dispersion (temporal and spatial).
- **Outcome.**
  - All checks pass (§2.5). The prototype numbers in the plan's §0 are confirmed.
  - Spatial growth at the oracle case's 10 Hz forcing: OS 0.0129, Lavalle 0.0125 (only 3% low). At ω = 0.45 Lavalle is 37% low.
- **Where.** `docs/results/orr-sommerfeld.md` (B.1–B.3), `verify.py`.

### 3.3 OS case table (B.4)

- **What.** OS and Lavalle stability numbers for all 85 manifest cases (2.4 min job).
- **Outcomes.**
  - **Lavalle's growth error depends on δ_lin.** It is 7–18% too high for δ < 5, about right for δ 5–15, and 25–45% too low for δ > 30. OS's k_max is up to 2.4× Lavalle's, so δ computed from Lavalle can be up to 2.3× smaller.
  - **The forcing cut-off (28 Hz) often stops short of the unstable band.** Of the 36 DNS cases, it lies below the OS cut-off frequency in 31, and below the most amplified frequency in 13 (β 45°/90°, Re ≥ 40). Only 7 force the first harmonic.
    - So at high Re the broadband DNS do not force the linearly dominant wave: their waves grow from the lower forced frequencies plus nonlinear harmonics.
    - At β 6.4° and low Re the opposite holds: the forcing reaches deep into the stable band.
    - This closed an open question in `decisions.md`; the production runs are not changed.
- **Where.** `docs/results/orr-sommerfeld.md` (B.4), `case_table.csv` in the store.

### 3.4 Phase A: the current MLPs examined (A.1–A.4)

- **A.1, the flat state.** Neither the sum model (inputs h, h_x) nor the separate models (h, q, h_x, q_x) gives S = 0 on a flat film.
  - The offset is up to half the friction term.
  - Both shift the long-wave speed by an O(1) amount: +0.44 (sum) and −0.72 (separate) in the oracle case. The sum model's shift matches its closed-loop waves running ahead of the DNS.
  - **Cause:** the training data excluded the flat film (flat mask).
  - **Linear growth:** the sum model ranges from removing the instability (Re200 β45) to 1.4× OS; the separate models give 1.0–2.4× OS.
- **A.2, crest speed against crest height.** Small DNS crests travel at the OS phase speed (1.88 against 1.85).
  - The sum model follows the DNS speed–height curve; its waves are just too tall. Its celerity error is a consequence of its amplitude error.
  - The separate models run 3–4% fast at the same height, so they also have a speed error of their own.
- **A.3, where the DNS correction acts.** In seven cases (δ 2–65), the forcing F = −S removes momentum on the wave front and returns it on the back.
  - Most of F is localised (fronts, ripples) and is not a simple multiple of q′ or ∂ₓq′.
  - The interpretation of the two projections is the swapped one (§2.8).
- **A.4, linear-region regression (oracle case).** Fitting the DNS correction in the linear region moves Lavalle's dispersion onto OS at 10 Hz (speed 1.84 against 1.85; growth within 5%).
  - With a single frequency the four coefficients cannot be separated.
  - The DNS correction also has an unexplained mean of about 3% of the friction scale (possibly gas drag).
- **Where.** `docs/results/structured-closure.md`, `scripts/structured-closure/`.

### 3.5 Linear-regime tests and targets (B.5–B.8), gate GB

- **B.5, DNS against OS (oracle case).**
  - From x = 60 to 130, growth matches OS to 1%, and the phase speed is 1.5% low (most likely mesh: level 15 waves were 1.2% faster than level 14).
  - The linear region is x ≈ 60–150, not the guessed x < 60–100.
  - The existing dense broadband restarts are too short (10 time units) to separate frequencies. That motivated the comb DNS.
- **B.6, ROM against OS** (comb inlet; three cases × three models).
  - **Lavalle:** the ROM matches analytic Lavalle to about 1%, so the measurement reflects the model, not the scheme.
  - **The MLPs where they stay linear:** 20–28% off OS, and they agree with their own A.1 linearisation.
  - **The MLPs at Re 120 and 200:** they go nonlinear. The cause is twice the OS growth (separate models), or the shifted base state plus a spurious equilibrium (sum model).
- **B.7, exact linear closure terms.**
  - ΔM_x and ΔT_w are each mostly in quadrature with h and cancel to 10–40% (median |S′|/|ΔM_x′| 0.10–0.42). This is the linear-regime root of the ΔM_x/ΔT_w cancellation seen since October.
  - ΔP_x and ΔD_x are negligible for small waves.
  - In the oracle case the *extracted* DNS ΔM_x and ΔT_w match these exact values to 3–6%. The pipeline is validated.
  - Small extra finding: the depth-integrated momentum equation in `docs/equations.md` drops a surface viscous term of order k²/Re. It is small but nonzero, and it is part of the unclosed remainder.
- **B.8, (a, b, c, d).**
  - Four coefficients reproduce OS growth within 10% over the forced band in all 85 cases; three do not.
  - Oracle case: a = −0.0056, b = −1.04, c = 0.61, d = 0.087.
- **Gate GB decision: D2 = L2.** The new closure's linear part will match OS, with g₃ in the default model.

### 3.6 The "inlet problem" and the first-cell fix

- **What.** In the failing B.6 runs the first cell of the output drifted (h to 2.4, q to 1.5). We traced it to the boundary routine:
  - it overwrites the first physical cell with the inlet state only in the copy used to compute each step's right-hand side;
  - the stored value was still advanced in time, but never used again.
- **Outcome.**
  - It is a write-out artefact; the dynamics were never affected. Fixed (`5df0595`): the interior output is bit-identical before and after, and only cell 0 changes.
  - It also affected earlier oracle outputs (q in cell 0 had reached 7.9), so metrics that include that cell, such as a max of q, were slightly contaminated.
  - The real reasons for the failing runs are in §3.5 (B.6) and §2.9.

### 3.7 The comb-forced DNS (D3)

- **What.** Five runs (Re 20, 40, 80, 120, 200; β 45°):
  - production physics and mesh on a 200-long domain (¼ of the production cost per time unit);
  - a comb of 21–31 frequencies at 3·10⁻⁴ each, covering the OS unstable band up to 0.97 ω_c;
  - spin-up to t = 150, then a restart writing 1D profiles every 0.25 for one comb period.
- **Cost.** About 1–2 h each on 16 CPUs.
- **Tests before launch.**
  - A timing test (15 s per time unit). Its first attempt stopped early because of how Basilisk ends runs; that was a test-setting issue, not a code problem.
  - A restart test: profiles are written and the forcing is continuous across the restart.
- **Why.** To test OS at many frequencies and several δ, to check the B.7 closure values per frequency, and to fit (a, b, c, d) from DNS rather than only from OS.
- **Where.** `dns/comb/`. Analysis: `analysis/orr_sommerfeld/comb_dns/analyse_comb.py`. **Your instructions:** `docs/plans/comb-dns-instructions.md`.

### 3.8 Cache v2 and the closure extractions

- **What.** `pipeline/cache_v2/build_cache_v2.py` writes one file per case. It holds:
  - the full record (every 10 time units) at DNS resolution;
  - raw h, q, ∫u², the flux defect ΔF, and the targets ΔM_x, ΔT_w, ΔP_x v2;
  - D_G derivatives at three σ;
  - the split from `data_config.json`, plus provenance and quality flags.
- **Snapshot hygiene.** Snapshots overwritten by the unforced restarts of 10 Nov are dropped wherever they are still present.
- **Extractions.**
  - Re050, a development case whose closure file had been deleted, is being re-extracted.
  - Seven new cases are being extracted (two are 3 Nov runs still in progress).
  - Four stale closure files are being fully redone into `.new` files. They will be swapped in after the cache build that is reading the old ones.
- **Why.** Phase C (training the structured closure) needs consistent data: same derivative operator as the ROM, flat states included, full record, no bad snapshots.
- **Where.** `docs/decisions.md` (16 Nov), `docs/plans/structured-closure-progress.md` (jobs and pending swap).

### 3.9 What changed in our understanding today

1. **Lavalle's linear defect is growth, and its sign depends on δ.** The new closure must have a nonzero linear part, and four coefficients describe it.
2. **The DNS and the extraction are right for small waves.** That gives confidence in the training targets.
3. **The current MLPs' drift has identifiable structural causes,** none of which a priori error could show:
   - a nonzero flat-film correction (from the flat mask);
   - a shifted long-wave speed;
   - wrong linear growth;
   - spurious equilibria.
4. **The ΔM_x/ΔT_w cancellation is already present in linear theory.** Modelling their sum, or building a structure that respects it, is the right target.
5. **The broadband DNS forcing does not reach the most amplified frequencies at high Re.** It matters when interpreting broadband statistics, and when planning new runs.

---

## Part 4. Where this goes next

### 4.1 The goal

The goal is an ML-augmented ROM for a film with a passive gas. Its closure must:
- (i) reproduce the DNS waves in closed loop (amplitude, speed, statistics);
- (ii) be stable and well posed;
- (iii) reduce, where possible, to interpretable formulas.

The structured closure is built for (i)–(ii) by design, and Phase E delivers (iii).

### 4.2 Immediate steps (jobs already running)

1. **Closure-file swap and cache v2 completion.** After the extractions and the cache build finish: swap in the four redone files, add Re050 and the new cases, check ∂ₓΔF ≈ ΔM_x for all cases, and refresh the manifest.
2. **Comb DNS analysis.** You run it with the instructions file; then I write up:
   - OS against DNS at about 20–30 frequencies × 5 cases;
   - the per-frequency ΔM_x/ΔT_w check;
   - (a, b, c, d) from the DNS.
   - If the DNS-fitted coefficients agree with the OS ones, L2 stands as is. If they differ (gas effects, mesh), we decide which to use.

### 4.3 Phase C: the structured closure (23 Nov–6 Dec)

- **The model.** The ROM's momentum equation becomes Lavalle's, plus:
  - **a flux part:** a shape factor Φ(h) ≥ 1 and an extra pressure ΔP(h). Both go through the Riemann solver, built so the system stays hyperbolic (variant V1);
  - **a relaxation part:** friction rate κ(h) > 0, pulling q toward Q = h³(1 + g₁h_x + g₂q_x + g₃h_xx).
- **What it gives by construction:**
  - every learned function is a small network of h and the parameters (Re, Ca, β);
  - Lavalle is the zero of every network;
  - a flat film is untouched, so there are no spurious equilibria;
  - the long-wave speed stays 3.
- **Training (C.1–C.3)** on cache v2 (σ = 0.5). The loss is the a priori error on S, plus the L2 constraint that ties the networks' values at h = 1 to the B.8 (a, b, c, d) of each case.
  - Ablations: no flux part, no relaxation part, δ_lin as an extra input.
  - Expect the a priori error to be somewhat worse than the free MLP. The claim being tested is closed-loop behaviour.
- **Solver (C.4–C.10).** Tables of the learned 1D functions are loaded into `lavalle-cpp`. Tests:
  - "all networks at zero" gives byte-identical Lavalle output;
  - C++ and Python agree to 1e-10;
  - a flat film stays flat for 100 time units;
  - the B.6 comb test now matches OS.

### 4.4 Phase D: closed loop, and gate GD (~6 Dec)

- **Oracle case:** run the candidates against the references (Lavalle, the oracle ΔT_w + ΔM_x, the ML sum model). Metrics:
  - the h error;
  - crest height and speed;
  - linear growth against OS;
  - stability.
- Optionally re-initialise the ROM from the DNS every 10 time units, to separate closure error from phase drift.
- **Broadband:** 2–3 cases, statistics only (h PDF, crest heights, celerity, spectra).
- **Gate GD:** you decide which closure Paper 1 carries (decision D1).

### 4.5 How the functional-form plan fits in

- **Shared infrastructure:**
  - cache v2 (FF task 1.1, now being built);
  - the closed-loop harness (FF task 1.3), used by Phase D;
  - the analytic-closure hook `form_model` (FF task 1.2), which can host closed-form versions of the structured closure.
- **Phase E is where the two plans meet.** Each learned function of the selected structured model (Φ, P′, κ, g₁, g₂, g₃) is a curve in h for given parameters. We fit low-order formulas to them, then model the coefficients against (Re, Ca, β, δ_lin).
  - The FF plan's monomial library (route S, Tier-1 coefficients) is the theory-based baseline and comparison.
  - The closed forms go back into the solver and must hold up in closed loop.
  - **If they do, they are Paper 1's functional form.**
- **The FF Phase 0 results** (Tier-1 fits, talk figures) are on branch `functional-form-v2` and should be merged into `main` with their own PR.
- **ΔP_x** (and memory, if needed) stays on its own track (`docs/results/dpx-closure.md`). The structured closure covers ΔM_x + ΔT_w.

### 4.6 Decisions still open

- **D1:** which closure Paper 1 carries. Decide at gate GD.
- **D4:** V1 only (hyperbolic by construction) or also V2 (more flexible, with a penalty and a runtime limiter), if V1's a priori error is much worse.
- **D5:** rollout training (Phase F) before submission, or later.
- **D6:** whether the OS comparison goes into Paper 1's validation section (§2.3). My suggestion is yes: B.5 and B.7 are strong evidence for the DNS and the extraction.
- **Git:** pushing `structured-closure` and opening the Phase A/B PR; merging `functional-form-v2`.
- **To test later:**
  - whether spurious equilibria cause the sum model's crest over-amplification;
  - what causes the 3% mean offset in the DNS correction;
  - the mesh explanation for the 1.5% phase-speed gap.

### 4.7 Caveats to keep in mind

- **OS is passive-gas linear theory.** The DNS has a real gas phase; agreement so far is within 1–6%.
- **The linear-regime evidence from the DNS is one case at one frequency** until the comb runs are analysed.
- **(a, b, c, d) are fitted to OS over the forced band.** Over the full unstable band, four numbers are not enough at high δ (risk 2 of the plan), but the misfit is limited to short capillary waves.
- **The interpretations in Phase A are diagnostic,** made from existing runs. The structured closure is what tests them.
