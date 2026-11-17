# What we did on 17 Nov 2026, and why

_Written for Krishna, to understand the work of 17 Nov: the comb-forced DNS results, cache v2, and the first structured closure (training, ROM implementation, first a posteriori test and its diagnosis), and in the evening the flat-film-exact MLP in closed loop (Part 6). It continues `2026-11-16-structured-closure-explainer.md`; read that first for Orr–Sommerfeld, (a, b, c, d) and the structured form. Technical results: `docs/results/orr-sommerfeld.md` (comb DNS) and `docs/results/structured-closure.md` ("Phase C"). Task status: `docs/plans/structured-closure-progress.md`._

---

## Part 1. What we did, and the main reasons

- **Analysed the five comb-forced DNS runs against Orr–Sommerfeld (OS).**
  - Yesterday OS had been checked against the DNS in only one case at one frequency. The training targets of the new closure come from OS, so they had to be trusted across the parameter space.
  - Result: at Re 80, 120 and 200 the DNS follows OS across the whole forced band (growth 2–3%, phase speed 1–1.5% low), and the extracted ΔM_x and ΔT_w match OS per frequency to 1–3%.
  - Re 40 (Ca 0.04) needed a window further upstream, because its waves were already nonlinear in the standard window. Re 20 is fine for the closure terms, but the correction Lavalle needs there is smaller than the measurement scatter.

- **Built cache v2 and fixed the training-data choices.**
  - Every training run reads cache v2, so it had to be complete and clean first: full time record, the derivatives the ROM uses, overwritten snapshots dropped.
  - Three partial cases failed the consistency check and were excluded (your decision). The derivative filter σ = 0.5 is the default and 0.25 the comparison (your decision).

- **Wrote the structured closure V1 and its training code.**
  - It is the core of the plan. It is a closure whose good properties hold *by construction*, so they don't depend on the training data:
    - a flat film stays flat;
    - the equations stay hyperbolic (well posed);
    - Lavalle is the starting point.
  - The linear (OS) constraint was formulated on the **dispersion relation**, not on four fitted numbers. The reason is that the two parts of the closure see derivatives differently in the ROM (Part 2.3).

- **Extended the OS constraint to all 85 OS case-table points** (your decision).
  - OS targets cost nothing (no DNS), so the closure can be told the right linear behaviour at many more parameter points than the 15 training cases. This helps it generalise to new (Re, Ca, β).

- **Chose λ_lin = 10 and ran an 8-run sweep on GPU.**
  - The scan λ = 1, 10, 100 showed the trade-off between linear accuracy and data fit. 10 halves the linear error of 1 at no cost; 100 costs 15% data accuracy.
  - GPU was 12× faster than CPU, so you relaxed the GPU limit for this session.

- **Implemented the closure in the ROM (`lavalle-cpp`) and tested it (C.4–C.10).**
  - An a posteriori test needs the closure inside the solver, and the solver must be shown to implement exactly what was trained.
  - Tests:
    - a flat film stays flat;
    - linear waves grow as the model predicts;
    - no hyperbolicity violations;
    - no stiffness.

- **Found and fixed a hidden bug in the Gaussian derivative.**
  - The uniform-film test failed: the standard Gaussian second derivative is not zero on a constant field (it is truncated at 4σ).
  - That broke the flat-film guarantee, and the training data and the ROM saw different errors. The same bias is in every h_xx feature of cache v2.

- **Ran the first a posteriori test** (oracle case, Re 120 Ca 0.01 β 45, 10 Hz).
  - This is the test that counts: does the closure work when the ROM feeds it its own state?
  - Result: as good as the existing `ml_sum_hhx` for about 5 time units, then it drifts faster.

- **Diagnosed the drift instead of guessing.**
  - Closed-loop runs with each half of the closure, and an "energy budget" of the closure on the DNS data, showed what goes wrong:
    - the closure's net energy input to the waves is a tiny difference of large terms, and every learned closure gets its sign wrong;
    - the flux part drives the growth, the relaxation part damps.

- **Tried one fix (a power penalty), which did not work, and stopped.**
  - The penalty corrects the energy budget a priori, but the a posteriori error did not improve. The waves now drift in phase.
  - So the next decision (seeds, broadband test, a richer form) is yours, not another blind variant.

---

## Part 2. Concepts you need

### 2.1 The structured closure in one picture

The ROM's momentum equation with the structured closure is

∂ₜq + ∂ₓ[ Φ q²/h + P_Lav(h) + ΔP(h) ] = κ(h)(Q − q)/h² + capillary + viscous,  with Q = h³(1 + g₁h_x + g₂q_x + g₃h_xx).

It has two parts.

- **Flux part (Φ, ΔP).**
  - It replaces the momentum flux q²/h by Φ q²/h and adds a pressure-like term ΔP(h).
  - It sits *inside* the conservative derivative ∂ₓ, so it goes through the solver's Riemann solver and changes the characteristic wave speeds. It roughly corresponds to ΔM_x (momentum-flux correction).
- **Relaxation part (κ, g₁, g₂, g₃).**
  - Lavalle's friction pulls q toward h³ (the flat-film flux) at rate 3/Re. The structured closure pulls q toward a *target* Q, which differs from h³ only where there are gradients, at a learned rate κ.
  - It roughly corresponds to ΔT_w (wall-shear correction).

Six small networks give Φ, ΔP, κ, g₁, g₂ and g₃ as functions of h and of (Re, Ca, β). The correction relative to Lavalle is

S = ∂ₓ[(Φ − 1)q²/h + ΔP] − [κ(Q − q) − (3/Re)(h³ − q)]/h².

### 2.2 What is guaranteed by construction, and what is not

- **The flat film stays flat, exactly.**
  - With no gradients, Q = h³, and on a flat film q = h³, so both parts vanish for any weights.
  - So S₀ = e = 0 exactly. The old MLPs violated this by O(1) amounts.
- **The equations stay hyperbolic, for every state.**
  - Φ ≥ 1 and increasing in h, plus a positive total pressure slope, make the discriminant of the flux Jacobian, 4u²[Φ(Φ − 1) + hΦ_h] + 4P_tot′, positive.
  - The ROM never throws a "loss of hyperbolicity" error.
- **Lavalle is the starting point:** every network starts at "no correction".
- **Not guaranteed, only encouraged by the loss:**
  - matching OS linear growth (a penalty, so close but not exact);
  - matching the data;
  - closed-loop accuracy. Closed-loop accuracy is the open problem.

### 2.3 Why the OS constraint is on the dispersion relation

- Yesterday's B.8 gave four OS-fitted numbers (a, b, c, d). The plan was to match the closure's linearisation to them.
- **The complication:** in the ROM the two parts of the closure see different derivatives.
  - The flux part goes through the finite-volume scheme: essentially the exact derivative, factor ik for a wave e^{ikx}.
  - The relaxation part uses the Gaussian-filtered gradients: factor ik·e^{−k²σ²/2}.

  So the same (b, c, d) mean different growth curves depending on which part produces them.
- **The solution:** insert the closure's linearisation, with the right factor on each part, into the ROM's dispersion relation. Then check at 20 frequencies how far the predicted wavenumber is from the OS one:

  δk ≈ −F(k_OS)/F′(k_OS),

  where F is the augmented dispersion function. Relative errors in growth and wavenumber are penalised, plus the long-wave coefficient D.
- With σ = 0 this reproduces yesterday's B.8 numbers to four digits, which is the check that it is right.

### 2.4 λ_lin and the loss

Loss = (error in S on the DNS data) + λ_lin × (dispersion error at the OS points) [+ λ_F × flux anchor] [+ λ_P × power error].

| λ_lin | Effect |
|---|---|
| 0 ("L0") | no constraint: linear growth errors up to 10× the OS peak |
| 10 | growth errors about 10× smaller, no loss of held-out accuracy |
| 100 | the constraint dominates; data accuracy drops by 15% |

### 2.5 The Gaussian derivative and σ

- **What σ is.** σ is the width of the Gaussian used to compute h_x, q_x and h_xx, in units of the film thickness h_N.
  - σ = 0.5 means 0.64 mm in the oracle case.
  - The filter halves a derivative at a wavelength of about 5.3σ: 2.7 h_N for σ = 0.5, 1.3 h_N for σ = 0.25.
  - In grid cells, σ = 0.5 is 10 DNS cells or 2.5 ROM cells.
- **Why filter at all:**
  1. training and the ROM must see identical inputs;
  2. the DNS and ROM grids differ (dx 0.05 against 0.2), and raw differences are noisy and depend on the grid;
  3. it keeps the closure from reacting to grid-scale noise in closed loop.
- **The bug.** Cut off at 4σ, the discrete second-derivative kernel does not sum exactly to zero, so h_xx of h = 1 is −3.5e-3 (σ 0.5, DNS grid) instead of 0, and a different value on the ROM grid.
  - The fix makes the operator exactly zero on constants.
  - Lesson: test the exact properties (here, "flat film stays flat") in the real solver, not only in Python.

### 2.6 The closure's energy budget ("power")

- **Definition.** The closure adds −S to the q-equation. Its power into the wave motion is P = ⟨(−S) q′⟩, where q′ is q minus its spatial mean. P > 0 feeds the waves; P < 0 damps them.
- **What the DNS shows** (oracle case):
  - the true closure takes energy out on the wave fronts (−0.32) and puts it in on the back faces (+0.19);
  - the net over the wave is only −3.4e-4, about 0.1% of either.
- **Consequence:** a closure that is 99% right pointwise can still have the wrong *sign* of net power. All the learned closures (old MLPs and new structured ones) feed net energy (+1.0e-3 to +2.2e-3), so their crests grow too high in closed loop.

### 2.7 Phase drift

- **The error measure.** The a posteriori error rms(h_ROM − h_DNS)/std(h_DNS) mixes amplitude errors and position errors.
  - If the ROM waves travel slightly faster or slower than the DNS waves, they drift out of phase.
  - The error then rises toward √2 ≈ 1.4 even if the wave shapes are perfect.
- **The check.** The shift that best aligns ROM and DNS waves separates the two effects. `ml_sum_hhx` stays in phase; some structured models drift ±7 units by t = 50.

### 2.8 A priori, a posteriori, and why they disagree

- **A priori:** evaluate the closure on DNS states. All closures here have NRMSE ≈ 0.12 on the oracle case.
- **A posteriori:** run the ROM with the closure. Small errors that change wave speed or energy accumulate over many wave periods.
- So a closure can win a priori and lose a posteriori. Today's result is exactly that.

---

## Part 3. Detailed report of 17 Nov

### 3.1 Questions answered first

- **OS assumptions; OS against DNS; how closure terms come from OS; how the ML model is linearised; the OS solver's role; how S₀ = e = 0 is enforced.** All are in Part 5 of yesterday's explainer.
- **Are the models only valid in the linear regime?** No. The data term trains them on fully nonlinear waves. The OS constraint only pins the behaviour near the flat film, where every wave is born.

### 3.2 Comb-forced DNS (D3)

- **Setup:** five cases at β 45°, LX = 200. A comb of 21–31 frequencies at 3e-4 each, up to 0.97 ω_c. Each frequency is separated exactly by projecting over one comb period.
- **Re 80–200:** DNS growth within 2–3% of OS (median), phase speed 1–1.5% below OS at every frequency (the same offset as the oracle case, attributed to the mesh). Extracted ΔM_x and ΔT_w within 1–3%.
- **Fitted (a, b, c, d):** the DNS values have the OS signs and are 15–25% smaller, along the one combination that is weakly determined once D is fixed.
- **Re 40, Ca 0.04:** the 31 modes add up to a wave height of 0.28 in the standard window (nonlinear). In x = 30–100 it behaves like Re 80–200.
- **Re 20:** closure terms match (3–4%), but Lavalle is already within 3% of OS there, smaller than the DNS scatter, so the fitted coefficients carry no information.
- **Open:** mean h is 0.3–0.55% below Nusselt in all runs.

### 3.3 Cache v2

- 36 cases, every snapshot to t = 2000, D_G(σ) derivatives for σ = 0.25, 0.5 and 1, and the momentum-flux defect ΔF.
- The four redone closure files were swapped in.
- Consistency check ∂ₓΔF against ΔM_x: NRMSE below 0.2 in most cases. The Ca 0.06–0.1 cases are better at σ = 0.25. Three partial cases fail (0.84–0.96) and are excluded.

### 3.4 Training

- **Data:** 15 development and 7 held-out cases. x from 50 to 95% of the domain; t ≥ 300 (start-up front gone); every 8th point. That is 4.7 M training points.
- **Smoke tests:**
  - λ_lin = 1 was too weak: the data term pulled the linear coefficients away from OS.
  - The GPU job first failed on a device bug, now fixed.
- **Sweep results (seed 0):**
  - With the OS constraint (λ_lin 10): growth error 0.06–0.07 (training, median) and 0.10 (held-out, median), long-wave D within 1–2%.
  - Held-out NRMSE: 0.26 at σ 0.25, 0.30 at σ 0.5.
  - Without the constraint: growth errors up to 10× the OS peak.
  - The flux anchor λ_F made no difference.

### 3.5 ROM implementation

- **Flux part:** `StructuredClosure` reads one table in h per case, exported by `export_tables.py`. Φ and Φ_h enter the existing shape-factor hook of the Rusanov flux; ΔP and ΔP′ enter a new hook in the flux and in its wave speed.
- **Relaxation part:** a source term using the same Gaussian derivative as training, now zero-sum.
- **Tests:**
  - uniform film exact;
  - κ dt/h² ≤ 0.03 (no stiffness);
  - comb runs reproduce the model's own linear dispersion to 0.5–4%;
  - growth against OS 8–11% (Lavalle 27–36%) at Re 120–200.

### 3.6 Oracle a posteriori and diagnosis

- **h error at t = 50:**

  | Exact closure | `ml_sum_hhx` | Structured σ 0.25 | Structured σ 0.5 | Lavalle |
  |---|---|---|---|---|
  | 0.19 | 0.47 | 0.60 | 1.20 | 0.59 |

  Crests reach 2.3–2.5 against the DNS 2.1.
- **Split runs:**
  - The flux part alone gives the best short-time error of any run (0.040 at t = 1, better than the exact closure), then over-grows the crests (2.6–3.0).
  - The relaxation part alone behaves like Lavalle.
- **Energy budget:** every learned closure feeds net energy where the DNS removes a little (Part 2.6).
- **Power penalty** (λ_P 10, 100):
  - it fixes the a priori energy budget;
  - it worsens the a priori fit (0.12 to 0.16);
  - it does not improve the a posteriori error. The waves drift in phase instead (−7 units by t = 50 for σ 0.25 with λ_P 100).

### 3.7 What changed in our understanding today

1. The OS targets are confirmed by the DNS across Re, so the linear constraint rests on solid ground.
2. The structured form delivers its guarantees in the real solver, and the linear constraint transfers to the ROM.
3. **The hard part is nonlinear and closed-loop.** Net energy and wave speed come out of small residuals of large terms that a pointwise loss cannot see. The old MLP has the same tendency, only weaker.
4. **Matching the energy budget a priori is not enough;** wave speed matters as much.

---

## Part 4. Where this goes next

### 4.1 Options, cheapest first

1. **Seed spread (the backup seed 1).** Nearby models give a posteriori results of opposite sign (phase +7.8 against −7.2). Before reading anything into them we need to know how much changes with only the random seed. About 15 minutes.
2. **Broadband held-out a posteriori (Phase D, gate GD).** The oracle case is monochromatic and the training data are broadband, so it is a single out-of-distribution test. The plan's real test compares ROM and DNS wave statistics for the held-out broadband cases.
3. **A richer form (V2).**
   - Coefficients depending on r = q/h³ as well as h (see 5.4). Crest speed depends on more than h (A.2).
   - Optionally, more data: the 11 good "extra" cases.
4. **Not a priority:** bigger networks (5.3).

### 4.2 How the functional-form plan connects

- It shares cache v2 (built today) and will share the closed-loop harness of Phase D.
- Its monomial fits are the baseline that the structured closure must beat.
- The structured closure's 1D coefficient curves Φ(h), κ(h), g(h) are what Phase E will try to replace with closed forms. That is why V1 keeps them as functions of h alone.

---

## Part 5. Questions and answers (17 Nov)

### 5.1 What do we do with the Re 20 and Re 40 comb cases?

- **Re 20: keep as it is.** It validates the closure terms. The growth correction is below the measurement scatter, and no rerun can fix that.
- **Re 40: reanalysed in an upstream window** (x = 30–100), where it agrees with OS. No new DNS needed.

### 5.2 Why smooth, and what do σ = 0.5 and 0.25 mean?

See Part 2.5. σ is a physical length in units of h_N, not grid cells. The grid only sets a lower limit: σ must stay above the ROM spacing (0.2 h_N), which is why 0.25 is the smallest sensible value.

### 5.3 Would wider or deeper networks help?

- **Probably not.**
  - Each network is effectively a smooth 1D curve in h per case, which 2 × 32 units already represent well.
  - Training error (0.22–0.25) is close to held-out error, which points to a limit of the *form*, not of capacity.
  - Larger networks also give less smooth curves, and kinks feed directly into wave speed in closed loop.
- **A quick test** (width 64, depth 3) would settle it; it was not run.

### 5.4 Why are the networks not functions of q or of the derivatives of h and q?

- **The gradients are there,** just not inside the networks.
  - S is linear in h_x, q_x and h_xx: through ∂ₓ of the flux, and through Q = h³(1 + g₁h_x + g₂q_x + g₃h_xx).
  - The networks give the *coefficients* of those terms. What V1 excludes is coefficients that depend on q or on the gradients, i.e. nonlinear gradient terms such as h_x².
- **Flux part (Φ, ΔP): restricted on purpose.**
  - These enter the hyperbolic flux. The guarantee "hyperbolic for every state" relies on Φ depending on h only, being ≥ 1 and increasing in h.
  - If Φ depended on q, the term ∂Φ/∂q enters the flux Jacobian, and the simple construction no longer guarantees a positive discriminant. The plan's V2 allows Φ(h, r), r = q/h³, with a penalty and a runtime limiter instead of a guarantee.
  - If the flux depended on gradients, it would no longer be a local hyperbolic flux at all. Wave speeds and the Riemann solver would be ill defined.
- **Relaxation part (κ, g₁, g₂, g₃): restricted for simplicity, not necessity.**
  - These are source terms. They do not enter the flux, so making them depend on r = q/h³ would **not** affect hyperbolicity.
  - It would not break the flat-film guarantee either, because Q − q still vanishes there.
  - They were kept as functions of h for three reasons:
    - V1 is the minimal version;
    - 1D curves are easy to interpret and to replace by closed forms (Phase E);
    - fewer inputs extrapolate more safely.
  - Making the relaxation coefficients depend on gradients is riskier: nonlinear gradient terms can act anti-diffusively and are sensitive to the filter and grid.
- **What follows for the next step.** A cheap intermediate between V1 and V2 keeps the flux part as is (guaranteed hyperbolic) and lets κ, g₁, g₂ and g₃ depend on (h, r). Since the crest speed depends on more than h (A.2), and the closed-loop error is now dominated by wave speed, this is a sensible next variant.

### 5.5 What is r?

- **Definition.** r = q/h³: the local flow rate divided by the flow rate a flat film of the same local thickness would carry. In Nusselt scaling a flat film of thickness h carries q = h³.
  - **r = 1:** locally in equilibrium (half-parabola profile, gravity balanced by wall friction).
  - **r > 1:** more flow than equilibrium, as on the steep fronts of large waves where inertia matters.
  - **r < 1:** less flow than equilibrium, e.g. in decelerating capillary ripples and the thin substrate.
- **Link to Lavalle's friction.** Lavalle's friction is (3/Re)(h³ − q)/h² = (3/Re)·h·(1 − r). So r is exactly what Lavalle's model keeps pushing back to 1: a natural measure of non-equilibrium.
- **Why r rather than q:** it is dimensionless and O(1), and every flat film has r = 1 whatever its thickness. A dependence on r therefore means a dependence on how non-equilibrium the flow is, separately from h.
- **Effect of using r in the relaxation coefficients (κ, g₁–g₃):**
  - the flat-film guarantee is unchanged (Q − q still vanishes there);
  - the linear behaviour depends only on their values at h = 1, r = 1;
  - hyperbolicity is unaffected (source terms).
- **Effect of using r in Φ (V2):** the r-dependence enters the flux Jacobian, so the hyperbolicity guarantee is lost.
- **Name clash:** in `lavalle-cpp`, `r` is a different variable, the Noble–Vila augmented variable r = h·w used for the capillary terms. The code for V1.5/V2 will use a distinct name.

### 5.6 Why can't the networks have inputs h, q, h_x, q_x? Would it break the linear loss?

- **They can, and the linear loss still works.** The restrictions in V1 come from the hyperbolicity guarantee and from interpretability, not from the linear loss.
- **Relaxation coefficients κ, g₁–g₃ as networks of (h, q, h_x, q_x):**
  - **Flat-film guarantee: kept.** With zero gradients, Q = h³(1 + g₁·0 + g₂·0 + g₃·0) = h³ whatever the g's are, and κ multiplies Q − q = 0.
  - **Hyperbolicity: unaffected** (source terms).
  - **Linear loss: unchanged in form.** On the flat film Q − q = 0 and h_x = q_x = h_xx = 0. So the derivatives of κ and g with respect to their inputs always multiply something that is zero there, and drop out of the linearisation. The linear coefficients only need the networks' *values* at (h, q, h_x, q_x) = (1, 1, 0, 0): a = κ − 3/Re, b = F_h − κg₁, c = F_q − κg₂, d = −κg₃, exactly as now.
  - **Risk of gradient inputs:** g₁(h_x)·h_x is effectively a nonlinear gradient term (h_x², h_x³, …). At steep fronts such terms can turn anti-diffusive and destabilise the ROM, and they depend more on the filter σ and the grid. That is testable, not forbidden.
- **Flux coefficients Φ, ΔP:**
  - **With q (or r):** allowed, and the linear loss still works. Automatic differentiation adds the ∂Φ/∂q terms to F_q automatically. The hyperbolicity guarantee is lost: this is V2 (penalty + runtime limiter).
  - **With gradients:** not as a flux. A flux that depends on h_x is no longer a function of the state, and the Riemann solver's wave speeds are undefined. Such dependence belongs in the source part.
- **A plain MLP can also be made flat-film-exact.** Use S = N(h, q, h_x, q_x) − N(h, h³, 0, 0), which is zero on every flat film by construction. With the same OS linear constraint, this is the old unconstrained MLP with the two main defects of Phase A removed.
  - It gives no hyperbolicity or interpretability guarantees.
  - It is a useful baseline: comparing it with V1/V1.5 shows whether the structure itself helps in closed loop.

---

## Part 6. Evening: the flat-film-exact MLP in closed loop

### 6.1 Questions answered first

- **Is the a posteriori "h error" a mean absolute relative error?** No. It is an rms error normalised by the wave amplitude: error(t) = √(mean over x of (h_ROM − h_DNS)²) / std_x(h_DNS), over 20 < x < 760 (`scripts/oracle_summary.py`).
  - 0 is a perfect match; about 1 means the error is as large as the waves.
  - About 1.4 is what waves of the right amplitude give when completely out of phase.
- **Why does OS look poor at Re 200 (slide 2)?** It is the DNS points that leave the curve, not OS.
  - Up to ω ≈ 1 the DNS is within 2–5% of OS. Above that the points scatter: 0.0064 against 0.0092 at ω = 1.07, and 20–40% low at ω ≈ 1.4–1.5.
  - Re 120 has the same dip at ω = 1.07, so it is a measurement artefact. The strongest modes are at ω ≈ 0.44–0.57, and their sum frequencies (0.50 + 0.57 = 1.07, …) fall exactly on comb teeth. The weakly growing high-ω modes are then biased by a small second-order product of the strong modes. Short waves are also less well resolved.
  - The "reliable" flag is looser at Re 200 (the domain holds more short wavelengths), so a few contaminated points get through. OS is exact linear theory and independently verified, so we trust it there.
- **What is δ on slide 3?** δ_lin = k_max·Re: the wavenumber of the fastest-growing wave times Re. It is not a height difference. It is a reduced Reynolds number, measuring how much inertia matters at the scale of the waves the film actually forms. Small δ is near-Benney, where Lavalle does well; large δ is inertia-dominated, where Lavalle's linear error grows.
- **Should the OS constraint apply only in the linear regime?** It already does, by construction.
  - The L2 loss never evaluates the closure on nonlinear states. It linearises the closure about the flat film (h = 1, q = 1), which uses only the network's value and first derivatives there, and compares the dispersion relation of that linearised model with OS.
  - In V1 it constrains only Φ, ΔP, κ, g₁–g₃ and their h-slopes at h = 1. In the flat-film-exact MLP it constrains only the gradient of N at the flat film. Everything away from the flat film is fitted to the nonlinear DNS data.
  - Caveat: a smooth network cannot change arbitrarily fast, so pinning the tangent pulls on behaviour at small-to-moderate amplitude, with strength set by λ_lin. The held-out NRMSE did not get worse at λ_lin = 10 for V1. For the MLP the constraint costs more data fit (smoke run: loss 0.10 at L0 against 0.17–0.59 at L10).

### 6.2 What was run

- **Model:** S = N(h, q, h_x, q_x, h_xx; Re, Ca, β) − N(h, h³, 0, 0, 0; Re, Ca, β). The second term makes S exactly zero on every flat film (Phase A defect 1), and the OS constraint fixes linear growth (defect 2). Otherwise it is the old `ml_sum_hhx` source term: the ROM flux stays Lavalle's, with no hyperbolicity or interpretability structure.
- **Runs:** σ 0.25 with λ_lin 0, 1, 10, one seed. σ 0.5 L10 and the comb runs were still running.
- **Test:** the oracle case (Re 120, Ca 0.01, β 45°, 10 Hz), started from the DNS state, compared over 100 time units.

### 6.3 Results in four points

1. **A priori it is much better.** NRMSE on the oracle DNS states 0.05 against 0.12 for `ml_sum_hhx` and V1, and the right sign of net power (about half the DNS damping at λ_lin 0 and 10). Giving the network q and the gradients is what does this.
2. **Its crests stay the right height.** This was the long-time failure of `ml_sum_hhx` and V1 (crests 1.2–1.5× too tall, then running ahead because taller waves travel faster). At t = 100 the new MLPs have h error 0.58 (L0) and 0.70 (L10) against 1.25 and 1.54. L0 is as close as the exact closure (0.55).
3. **It has a new, early failure: its waves are 3–6% too slow.** They lag by 2.5–3 (of a wavelength of 90) by t ≈ 40–50 while the amplitude is still right, so this is a wave-speed error of the closure itself. At t = 50 that makes L10 (0.59) worse than `ml_sum_hhx` (0.47). L0 and L10 then recover; L1 keeps lagging.
4. **A shape error is common to all learned closures.** With the shift removed, every learned closure reaches about 0.45–0.55 by t = 50 against 0.17 for the exact closure. It sits on the wave backs (which rise too early) and in the substrate and ripple region. This is now the largest gap.

### 6.4 How to read `oracle_snapshots_mlp.png`

(`scripts/structured-closure/oracle_snapshots.py`; figure in `$THINFILM_STORE/codes/ml-models/structured_closure/rom/figures/`, not in git.)

- **Layout:** 3 rows × 6 columns. Rows are times since the ROM start: t = 10, 50, 100. Columns are closures: Lavalle, exact (DNS ΔT_w + ΔM_x fed in), `ml_sum_hhx` (current MLP), V1 σ 0.25 L10, MLP L0, MLP L10.
- **Each panel:** h(x) over x = 400–700 (about three wavelengths). Black is the DNS, the colour is the ROM. The text in the corner gives the h error (as in 6.1, over the whole domain) and the best-fit x-shift of the ROM against the DNS (positive = ROM ahead).
- **What each column shows:**
  - **Lavalle:** crests far too low (about 1.45 against 2.1) and flat-topped; by t = 100 the waves lag by 11.
  - **Exact:** on top of the DNS to t = 50; at t = 100 slightly deeper troughs. This is the target.
  - **`ml_sum_hhx`:** in phase at t = 50 but crests too tall (2.4); by t = 100 tall (2.5), ahead by 7, and the substrate between waves is too smooth and too flat (the capillary ripples are lost).
  - **V1 σ 0.25:** spurious secondary bumps at t = 50; at t = 100 crests reach 3.0 with a wave split in two.
  - **MLP L0:** crests a little high, lagging by 3 at t = 50; back in phase at t = 100, ripples close to the DNS.
  - **MLP L10:** best at t = 10 (0.18); crest height right throughout; lags by 2.4 at t = 50; wave backs rise slightly too early.
- **What to take from it:** the old closures fail on amplitude, the new ones on phase and back shape. The capillary ripples in front of each crest are reproduced by the exact closure and the new MLPs; `ml_sum_hhx` and V1 smooth them out by t = 100, while Lavalle keeps them but in the wrong place.

### 6.5 How to read `oracle_phase_mlp.png`

(`scripts/structured-closure/oracle_phase.py`; same folder; numbers in `oracle_phase_mlp.csv`.)

- **Left: x-shift of the ROM against the DNS over time** (cross-correlation over x = 20–760, sub-cell). A constant slope is a wave-speed error; the DNS celerity is c = 1.92, so a slope of −0.06 is 3% slow. The new MLPs slope down from the start; `ml_sum_hhx` and V1 stay near zero, then turn up after t ≈ 50 as their crests overgrow. The jumps near t ≈ 80 are probably waves crossing the ends of the window.
- **Middle: h error after removing that shift.** What is left is shape error. All learned closures rise together to about 0.45–0.55 by t = 50; the exact closure stays near 0.2. After t ≈ 60 `ml_sum_hhx`, V1 and MLP L1 keep rising; MLP L0 and L10 level off at 0.5–0.65.
- **Right: max h, ROM over DNS.** Lavalle about 0.7; exact 1.0; `ml_sum_hhx` and V1 climb to 1.2–1.5; the new MLPs stay at 1.0–1.15.

### 6.6 What it means and what is next

- The flat-film-exact constraint plus richer inputs fixed the energy and amplitude failure, which was the long-time failure of all earlier closures.
- What remains: (i) a few-percent wave-speed error, and (ii) a shape error on the wave backs and substrate common to all learned closures.
- **Cheap next check:** compute each model's linear phase speed at the forcing frequency (ω ≈ 0.134) and compare with OS. If the linear speed is right, the speed error is nonlinear (crest-celerity relation, A.2); if not, the L2 band or weight needs to cover that frequency.
- **Still pending:** the σ 0.5 L10 oracle run and the comb runs (linear growth in the ROM), then slide 5.
