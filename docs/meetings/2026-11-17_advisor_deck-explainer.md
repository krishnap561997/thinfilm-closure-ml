# Advisor deck, 17 Nov 2026: what each slide says and how to present it

_For Krishna, to present `2026-11-17_advisor_structured-closure.pptx`. The deck and its PNGs are not in git: rebuild them with `scripts/meetings/advisor_2026_11_17_figs.py` (figures) and `scripts/meetings/advisor_2026_11_17_deck.py` (deck; needs `python-pptx` on `PYTHONPATH`). Background: `docs/explainers/2026-11-16-structured-closure-explainer.md` and `2026-11-17-structured-closure-explainer.md`._

**The story in one line:** the current ML closures drift because they are wrong at the flat film and in linear growth. We built an exact linear reference and a closure that is right there by construction, and it works in the ROM. The remaining problem is nonlinear: the closure's net energy input to the waves.

**Timing:** about 2 minutes per slide, 10–12 minutes in total, leaving room for questions.

---

## Slide 1: Why the current ML closures drift, and the plan

**Figure** (`f1_ml_drift.png`). Oracle case (Re 120, Ca 0.01, β 45°, 10 Hz). The ROM is started from the DNS state, and the h error relative to the DNS wave amplitude is plotted against time.
- **Grey, Lavalle:** the waves collapse (crests at 1.45 against 2.1), so the error jumps to about 0.6.
- **Dashed teal, exact closure:** the DNS ΔT_w + ΔM_x fed into the ROM. Error about 0.2 for 50 time units. This is the best a closure of this kind can do.
- **Coral, current MLP:** follows the exact closure for about 10 time units, then drifts. By t = 100 it is worse than Lavalle.

**Talking points**
- The current MLP is good a priori and for short times. The problem is closed-loop drift.
- Phase A traced two defects:
  1. **Not zero on a flat film.** Training masked out |h − 1| < 0.05, so the MLP never saw a flat film. Its output there is not zero, which shifts the base state and the wave speed by O(1).
  2. **Wrong linear growth:** 20–28% off Orr–Sommerfeld; the separate-model MLP doubles it.
- **The plan:** a closure that is right at the flat film *by construction* and is trained to the exact linear behaviour.

**Likely question:** "Isn't the drift just error accumulation?"
- Partly. But a closure that shifts wave speed or adds energy at the flat film produces systematic drift, not random accumulation.
- The exact-closure run shows the ROM itself can hold 0.2 for 50 time units.

---

## Slide 2: An exact linear reference, confirmed by the DNS

**Figure** (`f2_comb_os.png`). Spatial growth rate −k_i against forcing frequency ω at Re 80, 120 and 200 (β 45°, Ca 0.01).
- **Navy line: Orr–Sommerfeld (OS).** The exact linear theory of the full 2D equations, with no long-wave assumption.
- **Dashed grey: Lavalle's model.**
- **Coral dots: DNS.** Short comb-forced runs, LX = 200, with 21–31 frequencies forced at once (3e-4 each) and each separated exactly by projecting over one comb period.

**Talking points**
- **OS is verified.** It matches the Benney limit, Re_c = (5/6) cot β, and the published saddle points of Brevdo et al. (in Kalliadasis et al.) to the printed digits.
- **The DNS follows OS across the whole forced band:**
  - growth within 2–3% (median);
  - phase speed 1–1.5% below OS at every frequency, a systematic offset consistent with mesh resolution.
- **The extracted closure terms match too.** ΔM_x and ΔT_w extracted from the DNS match the exact OS values to 1–3% per frequency. This validates the extraction pipeline in the regime where the truth is known.
- **Lavalle is 24–37% low in growth,** and the deficit grows with Re.

**Likely questions**
- **"What about Re 20 and Re 40?"**
  - Re 40 (Ca 0.04) agrees in an upstream window (x = 30–100); further downstream its waves are already nonlinear.
  - At Re 20 Lavalle is already within 3% of OS, smaller than the DNS scatter, so there is nothing to correct.
- **"Why not just use the DNS for the linear behaviour?"** OS is exact and costs seconds per case, so it gives targets at any (Re, Ca, β), including parameters with no DNS.

---

## Slide 3: Four coefficients fix Lavalle's linear error

**Figure** (`f3_linear_error.png`). For all 85 cases of the parameter table: the maximum growth error over the forced band relative to the maximum OS growth, against δ_lin = k_max Re.
- **Grey: Lavalle.**
- **Teal: Lavalle plus a linear correction with four coefficients (a, b, c, d),** fitted to OS with the long-wave limit imposed exactly.

**Talking points**
- **One parameter organises Lavalle's error: δ_lin.** δ_lin is k_max·Re, the inertia of the most unstable wave. Lavalle is 7–18% too unstable at small δ and 25–50% too stable at large δ.
- **Four numbers fix it.** The linear part of the closure, S = a(ξ − 3η) + b η_x + c ξ_x + d η_xx, brings all 85 cases within 10%.
- **The h_xx term d is essential** beyond δ ≈ 5. That is why the closure has a g₃ (h_xx) term.
- **This becomes the training constraint (L2).** In practice we impose the OS dispersion relation directly at 20 frequencies per point, because the flux and relaxation parts of the closure see different derivative filters in the ROM. It is imposed at all 85 OS points, which needs no DNS.

**Likely question:** "Is 10% good enough?" It is over the forced band, where the waves are born. Above it (short capillary waves) four coefficients are not enough at high δ, but those waves barely grow.

---

## Slide 4: Structured closure works as designed in the ROM

**Left figure** (`f4a_schematic.png`). The structure of the closure in the q-equation:
- **Flux part:** Φ(h) multiplies the momentum flux q²/h, and ΔP(h) adds to the pressure-like flux. Both sit inside ∂ₓ, so they are conservative and enter the ROM's Riemann solver, changing the characteristic wave speeds. Roughly the ΔM_x correction.
- **Relaxation part:** instead of Lavalle's friction pulling q toward h³ at rate 3/Re, the closure pulls q toward a target Q = h³(1 + g₁h_x + g₂q_x + g₃h_xx) at a learned rate κ(h). Roughly the ΔT_w correction.
- Six small networks give Φ, ΔP, κ, g₁, g₂ and g₃ as functions of h and of (Re, Ca, β).

**Guarantees** (the card below the schematic):
- **Exact flat film:** with no gradients, Q = h³ = q, so S = 0 for any weights.
- **Always hyperbolic:** Φ ≥ 1, increasing in h, plus a positive pressure slope.
- **Starts at Lavalle:** every network initialised to zero correction.
- **ROM checks:** the flat film stays flat to round-off, and there is no stiffness (κ dt/h² ≤ 0.03).

**Right figure** (`f4b_rom_comb.png`). Linear waves in the ROM at Re 120, with the comb inlet:
- **OS:** navy line.
- **Lavalle:** grey squares, far too low.
- **Structured closure without the OS constraint:** gold triangles, up to 40% too high. The data alone don't pin linear growth.
- **With the constraint:** teal circles, on the OS curve over the forced band (ω ≲ 0.9). Above it, where the constraint does not act, it drifts high.

**Talking points**
- **The OS constraint cuts the growth error about tenfold** (8–11% against Lavalle's 27–36% at Re 120–200) **at no cost in held-out accuracy** (NRMSE 0.26 at σ = 0.25).
- **The ROM reproduces the trained model's own linear prediction to 3–4%,** so the solver implements what was trained.
- **The uniform-film test found a hidden bug.** The truncated Gaussian second derivative (scipy and our C++ copy) is not zero on constant fields: h_xx(1) = −3.5e-3 on the DNS grid, a different value on the ROM grid. It is now fixed with a zero-sum operator.

**Likely questions**
- **"Why are the coefficients functions of h only?"**
  - For Φ and ΔP, it is what guarantees hyperbolicity.
  - For the relaxation coefficients, it was a simplicity choice. They can take (h, q/h³, h_x, q_x) without losing the flat-film or hyperbolicity guarantees, and that is a next step.
- **"What is σ?"** The width of the Gaussian derivative filter in units of h_N. σ = 0.25 fits better than 0.5.

---

## Slide 5: Closed loop: the energy balance is the hard part

**Left figure** (`f5a_aposteriori.png`). Same oracle test as slide 1, up to t = 60, now with the structured closure (navy, σ = 0.25 with the OS constraint).
- It matches the current MLP and is close to the exact closure for about 5 time units.
- It then drifts somewhat faster: h error 0.60 against 0.47 at t = 50.
- The crests grow too high (2.5 against 2.1), and the waves drift slightly out of phase.

**Middle figure** (`f5b_power.png`). Net power the closure puts into the wave motion, P = ⟨(−S) q′⟩ (q′ = q minus its spatial mean), evaluated on the DNS states of the oracle case:
- **DNS, the true closure: −0.34 × 10⁻³.** Slightly damping.
- **Current MLP: +2.23 × 10⁻³.**
- **Structured V1: +1.80 × 10⁻³ (σ 0.5) and +1.02 × 10⁻³ (σ 0.25).**

**Talking points**
- **Pointwise, every closure fits the oracle data equally well** (NRMSE ≈ 0.12).
- **The net energy input is a ~0.1% residual of large opposite terms.** The true closure removes energy on the wave fronts (−0.32) and adds it on the backs (+0.19). A closure can be 99% right pointwise and still have the wrong sign of net energy, and every learned closure does. That explains the over-grown crests.
- **Split runs show the two parts of the closure behave sensibly.** The flux part alone drives the crest growth, while the relaxation part alone damps like Lavalle. The full closure is a balance between them that is slightly off.
- **We tried penalising the power error in training.** It fixes the energy budget a priori but not the closed-loop error: the waves then drift in phase instead. So energy alone isn't the whole story; wave speed matters too.

**Next steps** (the four cards):
1. **Seed spread:** how much of the closed-loop differences is training noise. Nearby models drift in opposite directions.
2. **Flat-film-exact MLP:** S = N(h, q, h_x, q_x, h_xx) − N(h, h³, 0, 0, 0) with the OS constraint. It is the current MLP with both Phase A defects removed, and it tests whether the structure helps. Running now.
3. **Richer forms:** V1.5 (relaxation coefficients depending on q/h³) and V2 (Φ too, with hyperbolicity by penalty).
4. **Broadband held-out a posteriori runs (gate GD).** The oracle case is a single monochromatic test; the plan's real test compares ROM and DNS wave statistics on the held-out broadband cases.

**Likely question:** "Why not train on closed-loop error directly?" It is possible (differentiable or windowed rollouts) but expensive and stiff to optimise. The cheaper fixes above come first. If the drift persists, a short-rollout loss is the natural next step.

---

## Numbers to have at hand

| Item | Value |
|---|---|
| DNS vs OS growth, Re 80–200 | 2–3% (median); phase speed −1 to −1.5% |
| Extracted ΔM_x, ΔT_w vs OS | 1–3% per frequency |
| Lavalle growth error | 24–37% (Re 80–200); 7–18% too unstable at δ < 5, 25–50% too stable at δ > 30 |
| 4-coefficient fit | all 85 cases within 10% |
| Structured V1, ROM growth error vs OS | 8–11% (Re 120–200), Lavalle 27–36% |
| Held-out NRMSE | 0.26 (σ 0.25), 0.30 (σ 0.5); current MLP a priori 0.29 |
| Oracle h error at t = 50 | exact 0.19, current MLP 0.47, structured σ 0.25 0.60, Lavalle 0.59 |
| Net power, oracle case (×10⁻³) | DNS −0.34; MLP +2.23; V1 σ 0.5 +1.80; V1 σ 0.25 +1.02 |
