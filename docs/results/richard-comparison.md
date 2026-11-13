# Comparison: three-moment closure (ordering matrix §4) vs Richard et al. enstrophy models

_Prepared 8 Nov 2026. This is a companion to `ordering-matrix.md`. All numbers below come from `richard_compare.py` (added to `ordering-matrix-scripts.zip`)._

## Sources

- **Richard, Ruyer-Quil & Vila (2016), JFM 804, 162–200.** Full text, uploaded by you on 9 Nov and used in §6. When §§1–4 were first written, only the abstract was available (the paper is paywalled).
- **Richard, Gisclon, Ruyer-Quil & Vila (2019), EJMB 76, 7–25.** Full text, uploaded by you on 9 Nov and used in §5.
- **Ruyer-Quil, Bresch, Gisclon, Richard et al. (2023), JFM**, on sheared droplets. Open access. It sets out the same enstrophy construction in full:
  - the definition of enstrophy;
  - a Teshukov-type conservative structure that neglects ⟨u′³⟩;
  - a relaxation source with an arbitrary constant B;
  - the energy equation.
  
  Its base flow is Couette rather than semi-parabolic.

§§1–4 compare against the **structure** of the Richard approach as written out in the 2023 paper, because that was all that was available at the time. §3 lists what still had to be checked against the 2016 coefficients. **All of it is now answered in §6, and §6.6 supersedes §4.** Your Literature folder lists Richard 2019 and 2024 but not 2016, so it may be worth adding the 2016 PDF there.

## 1. Mapping between the two formulations

| | Ordering matrix §4 | Richard et al. |
|---|---|---|
| Third variable | A₂ = ∫u² dz, with deformation amplitude s | enstrophy φ = h⁻³∫(u − U)² dz |
| Relation | A₂ = q²/h + h³φ | same |
| Third equation | second moment: ∂ₜA₂ + ∂ₓA₃ = (2/Re)(3q − 𝒟) − 2∫u∂ₓp | work–energy theorem, the same equation; their φ equation is it minus 2U × momentum |
| Pressure in the third equation | cancels in the φ form (checked) | cancels (2023, Eq. A3 minus A4) |

The two sets of equations have the same exact content. They differ only in how three things are closed: **A₃**, **the source in the φ (A₂) equation**, and **the wall shear**. Those are compared below.

## 2. Closure by closure

### 2.1 Flux A₃ (third moment) and skewness

Writing ⟨·⟩ for a depth average, A₃ = h⟨u³⟩ = h[U³ + 3U·h²φ + ⟨u′³⟩].

- **Richard structure.** This neglects ⟨u′³⟩, following Teshukov's shear-shallow-water (SSW) system. The 2023 paper justifies this because ⟨u′³⟩ = 0 at leading order for Couette flow.
- **Falling film.** For the semi-parabola, the skewness is not zero: ⟨u′³⟩ = −(2/35)U³. In my families it is σ(s) = f₃ − 3f₂ + 2, which equals −2/35 + O(s²) for the quartic mode.
- **The consequence is an inconsistency.** I took the weak-inertia slaved limit of each three-moment model and solved for the first-order flux correction q₁:

| A₃ closure | q₁ (slaved, δ ≪ 1) | Wall shear correction | Linear threshold |
|---|---|---|---|
| Quartic family, skewness kept | (6/5)Re h⁶h_x − cot β h³h_x + h³h_xxx/(3Ca) | −(3/5)Re h⁴h_x | (5/6)cot β ✓ (Benney) |
| Sextic family, skewness kept | identical | identical | (5/6)cot β ✓ |
| SSW, skewness dropped | **(17/15)**Re h⁶h_x − cot β h³h_x + h³h_xxx/(3Ca) | **−(2/5)**Re h⁴h_x | **(15/17)**cot β ✗ (6% error) |

- My three-moment model is first-order consistent **whichever mode is chosen**. Only the semi-parabola's skewness has to be right.
- A Teshukov flux carried over from the sheared-film papers to a falling film loses consistency.
- The 2016 paper states that its model is consistent, so it must handle this point somehow. **Resolved in §6.1:** they keep the skewness and move it into the relaxation sources.

### 2.2 Source in the φ (A₂) equation

- **Richard.** The source is −(B/(Re h))(φ − φ_eq), with B arbitrary. In their ordering, φ − φ_eq = O(ε²), so consistency does not fix B; in the 2023 paper it is "chosen large".
- **This work.** The source is derived from the profile family: S_φ = (2/(Re h²))(q τ_w/h − 𝒟) = (2/Re)(q²/h⁵)(s − c s²). Linearising gives the same relaxation form with B fixed: **B = 35 (quartic) or 21 (sextic)**.
- **Why this matters for the matrix.** In the δ ≫ 1 rows, the profile is not slaved, so the relaxation rate is part of the leading-order dynamics and cannot be left free. A test that does not use any DNS: for a uniform film, the exact viscous decay rates of profile perturbations are ((2n+1)π/2)² in units of 1/(Re h²).

| | slow mode | second mode |
|---|---|---|
| exact | 2.4674 | 22.207 |
| quartic | **2.4688** (+0.06%) | 42.53 |
| sextic | 2.4462 (−0.9%) | **25.75** |
| friction 3q/h² with no φ coupling (as in the 2023 model) | 3 (+22%) for the q mode | set by the chosen B |

Both families capture the slow mode almost exactly. The quartic is better on the slow mode and the sextic on the second. A model in which the friction does not depend on φ relaxes q 22% too fast.

### 2.3 Wall shear

- **2023 model.** τ_w = 3U/h, independent of φ, plus their applied shear.
- **This work.** τ_w = (q/h²)(3 + s): the wall shear depends on the profile deformation.
  - This dependence produces the correct slow decay rate in §2.2.
  - It also gives the correct first-order correction −(3/5)Re h⁴h_x in the slaved limit.
  - **Resolved in §6.2:** the 2016 model does couple τ_w to φ, and linearised, its coupling is identical to ours.

### 2.4 Hyperbolicity and energy

- **SSW structure.** It is always hyperbolic. Its speeds are U and U ± √(3h²φ), which are real whenever φ ≥ 0 (verified).
- **My families (skewness kept).** These lose hyperbolicity for 1.61 < s < 2.91 (quartic) and 0.48 < s < 1.31 (sextic). **This is the price of consistency.**
- **Energy.** Both models have an energy equation, because the A₂ equation is the energy equation. Mine has dissipation 𝒟(s) ≥ 0 and gravity work 3q. Richard et al. (2019) argue that no two-equation model is both consistent and energy-compatible. The two-equation q-gauge model of matrix row 1 is consistent but momentum-based, so it falls under that statement.

## 3. Things to check in the 2016 PDF (all resolved in §6)

1. How they handle ⟨u′³⟩ ≠ 0 for the semi-parabola. Options include an extra flux term, a modified definition of the third variable, or a weighted (non-plain) momentum equation.
2. Whether τ_w (their friction term) depends on φ.
3. Their value of B, if one is fixed for falling films, compared with 35 and 21.
4. Their kinematic-wave speed and solitary-wave comparisons, to use as benchmarks if we adopt a three-moment model.

## 4. Conclusions and recommendation

1. **The three-moment model with skewness kept could serve as a single closure framework across the whole matrix:**
   - in the weak-inertia limit it is first-order consistent (Benney and Lavalle recovered for either mode);
   - its energy equation is built in;
   - its viscous relaxation is physically correct to better than 1% for the slow mode;
   - it remains the minimal consistent truncation when δ ≫ 1.
   
   This argues for using it instead of the two-equation row-1 model even when Re ~ ε⁻¹.
2. **Its weak point is hyperbolicity.** Where Richard et al. guarantee hyperbolicity (via SSW), our model has a non-hyperbolic band of s.
   - The slaved estimate s ≈ −(3/5)Re h³h_x is roughly 10 at strong fronts for Re ~ 100. That is beyond the realizable branch of the one-mode family (s < 4.5) and crosses the band.
   - So a single deformation mode is probably too restrictive at fronts. A DNS projection of the profiles onto ψ_a and ψ_b (`ordering-matrix.md` §6, item 4) is needed before choosing between:
     - two modes (a four-equation model, in line with Ruyer-Quil & Msheik's 2021 four-equation work);
     - an SSW flux with the skewness moved into a consistent correction;
     - letting the ML term absorb the remainder.
3. **For the ML split:** if the ROM baseline becomes a three-moment model, the closure targets change. They become A₃, τ_w(h, q, A₂) and 𝒟, with A₂ taken from the DNS directly; ∫u² is already written by `column_profiles.h`. This is a bigger change than switching to the q-gauge two-equation model, so it should wait until the Paper 1 deadline is past.

---

## 5. Addendum (9 Nov): Richard, Gisclon, Ruyer-Quil & Vila (2019), EJMB 76, 7–25

You uploaded the 2019 paper, which covers two-equation models. It is not the 2016 three-equation paper, so §3 items 1–3 remain open. It does, however, settle several points (checked in `richard2019_check.py`).

### 5.1 What it independently confirms

- **First-order fields.** Their Eqs. (32)–(33), with λ = 3 and cos θ/Fr² = 3 cot β/Re, give τ̃₁(0) = 3Re h⁴h_x − 3 cot β h h_x and q₁ = (6/5)Re h⁶h_x − cot β h³h_x. These are exactly our fixed-h τ_w⁽¹⁾ and Benney correction (skill script and matrix §3.5).
- **Profile variance.** Their Eq. (35), ⟨(u₀ − U₀)²⟩ = λ²h⁴/45, matches our f₂(0) = 6/5.
- **Lavalle's flux coefficient.** Their consistency condition (59), F₁ = (1/9 − C₁/10)(4/5)λ²h⁵, gives 2λ²h⁵/225 for C₁ = 1. This is the Lavalle coefficient in status doc §3, so the 1/45 − 1/75 = 2/225 bookkeeping is confirmed.
- **Why a third variable is needed.** They show that a consistent two-equation model can satisfy both the averaged momentum and the energy equations only in the MIME class (∂F₁/∂U = 0), and that even then its energy is not the physical energy. This is the same conclusion that motivated the three-moment model in matrix §4.

### 5.2 What it changes: our row-1 two-equation model is their EIM model

Our q-gauge row-1 model (matrix §3.3) is **identical** to the energy-integral-method (EIM) form, their Eq. (22) (E₂ = hU²/10, F₂ = 19hU³/70), once that is rewritten as a q equation. The symbolic difference is 0. This is the Ruyer-Quil–Manneville first-order model. Their analysis of the EIM class shows that:

- it does not satisfy Galilean invariance, and it is not compatible with the averaged momentum equation in their MIM form (18), with the corrections in the flux and friction C₁(λh − 3U/h);
- it admits **no capillary energy**, so the Noble–Vila extended (augmented) model and its nonlinear-stability proof do not apply (their §5 and Appendix D);
- its solitary-wave speed on a vertical wall is 7% high (their Table 1).

The third point that matters for us is the capillary scheme: `lavalle-cpp` uses exactly that Noble–Vila augmented capillarity scheme. **Matrix §8, step 4 (put the q-gauge row-1 closures into `lavalle-cpp`) should therefore be withdrawn.** It would remove the property that makes the solver's capillary treatment provably stable. (I've corrected the matrix note.)

### 5.3 Two cheap improvements to the existing ROM that they recommend

Both stay inside the MIME class, so the augmented scheme still applies.

1. **Diffusive terms.** `lavalle-cpp` carries the add-on (2/Re)q_xx. Richard et al. instead derive the consistent linear O(ε/Re) terms of a MIME model and choose the form that reproduces the Orr–Sommerfeld kinematic-wave speed limit c → 3/2 (their α = 4/3, Eqs. 105–117). With C₁ = 1, their Eq. (121) in our units is:

   $$\frac{1}{Re}\Big[6\,\partial_x(hU_x)-3\,\partial_x(U h_x)+3\frac{U}{h}h_x^2\Big]$$

   - Linearised about Nusselt flow along kinematic waves (U′ = 2h′), theirs gives (9/2)U″ and our 2q_xx gives 3U″.
   - So the current ROM diffusion is two-thirds of their consistent value. That is my reading of their Eq. (106); please check it.
   - They show α = 4/3 gives the best neutral curves for both water on a vertical wall and the Liu–Gollub glycerin–water case (β = 6.4°, their Fig. 4).
2. **C₁.** Lavalle's model is the MIME model with C₁ = 1. It gives the best neutral stability curves, but its solitary-wave speeds are about 10% too fast on a vertical wall and 15% at Fr′⁻² = 0.3 (their Table 1).
   - Their "equivalent MIME" model, C₁ = 31/30 − √6/45 ≈ 0.979 with K = 315 − 90√6, matches the DNS solitary-wave speed on a vertical wall.
   - It changes only the coefficients of the flux, the friction and the hydrostatic term.
   - Its parameters are Eqs. (120)–(125).

### 5.4 Relevance to the ML closures and to Paper 1

- **The model's validity range.** Their models, like Lavalle's, are derived for Re, Fr and θ of O(1) (κ ~ 1, i.e. We ~ ε², as in our skill's default). The tuning of C₁ and α is an explicit calibration of how the model is extrapolated beyond that range, against DNS and Orr–Sommerfeld. That is the same role our ΔC terms play, done with one or two constants instead of a network. This is a useful framing for the Paper 1 introduction and for choosing baselines (Lavalle C₁ = 1 vs equivalent MIME vs α = 4/3).
- **The ΔC split depends on the baseline.** Condition (59) shows that, for any consistent MIME model, choosing C₁ moves terms between flux (F₁), friction and hydrostatic pressure. So how a DNS difference is divided among ΔM_x, ΔT_w and ΔP_x depends on which consistent baseline is used. This reinforces matrix §3.6, item 3.
- **The oracle result is consistent with their findings.** They report that two-equation models predict wave speed acceptably but wave amplitude poorly, and that this cannot be fixed within two equations. Our oracle run fits that pattern: max h was 1.45 against 2.12 in the DNS with Lavalle alone. In their Liu–Gollub comparison (Fig. 5c), the three-equation model of Richard et al. (2016) gives the closest agreement with experiment.

### 5.5 Still open at that point (resolved in §6)

- §3 items 1–3: skewness handling, a τ_w–φ coupling, and the value of B.
- The Liu & Gollub depth profiles in their Fig. 5 (Re = 19.3, β = 6.4°, 1.5 Hz) correspond to the ROM's existing good case ("good at Re = 19.33, β = 6.4°, f = 1.5 Hz", status doc §7). That makes Fig. 5 a ready benchmark for the α = 4/3 change.

---

## 6. Addendum (9 Nov): Richard, Ruyer-Quil & Vila (2016), JFM 804, 162–200

With the 2016 paper in hand, all the §3 items can now be answered (checked in `richard2016_check.py`).

### 6.1 Item 1: skewness. They keep it, then move it into the sources.

- **Same skewness as ours.** Their Eqs. (3.45)–(3.46) give ⟨u′³⟩ = −2λ³h⁶/945 = −(2/35)U³, the same value as §2.1.
- **They keep it, and say why.** They explicitly do **not** drop it, because the no-slip wall means the film is not weakly sheared (their §3.4). That answers the §2.1 concern: their model is consistent, and the SSW-with-skewness-dropped variant in our table is not their model.
- **How they keep the SSW structure.** The skewness flux −∂ₓ(h⟨u′³⟩/2) depends only on h at leading order, so its x-derivative is proportional to h_x. They then use the first-order identity (3.56), which expresses h_x through ⟨u₁⟩ and φ₁, to **rewrite it as relaxation source terms** (Eq. 3.66). The conservative part keeps the exact Euler/SSW form, P = h³φ + h²cos θ/(2F²), so the model is unconditionally hyperbolic and consistent at first order.
- **The limit of this trick.** Identity (3.56) holds only when the profile deviations are slaved, i.e. ⟨u₁⟩ and φ₁ are tied to h_x. That is the weak-inertia ordering (Re = O(1), δ ≲ 1).
  - In matrix rows Re ~ ε⁻² and ε⁻³ the skewness flux is an O(1) inertial term with no first-order relation to trade it against. **So their construction is consistent for row 1 but not for rows 2–3.** There, our approach of keeping the skewness in the flux is the consistent one, and the hyperbolicity issue in §2.4 is real.

### 6.2 Item 2: τ_w is coupled to φ, and it matches ours

Their wall shear is Eqs. (5.10)–(5.11), with f₁ = 7/3 and f̃₂ = 315U/(2λ²h³), both fixed by first-order consistency through (3.56)–(3.57):

τ_w = λh + (1 + f₁)(3U/h − λh) − f̃₂(φ − λ²h²/45)

Linearised about Nusselt flow, theirs gives 3 + 10 u′ − (35/2) φ′, and **our quartic family gives exactly the same, 3 + 10 u′ − (35/2) φ′**. The sextic family does not. So the quartic mode ψ_a reproduces their asymptotically derived wall-shear closure. Together with the slow-decay result of §2.2, this settles the mode choice in ψ_a's favour.

### 6.3 Item 3: relaxation, with B not free

The free constant B appears only in the 2023 sheared-droplet paper. In 2016, the enstrophy relaxation follows from f₁ and f̃₂ (Eq. 3.76). The comparison for homogeneous viscous relaxation (h fixed, x-independent), in units of 1/(Re h²):

| | slow mode | second mode |
|---|---|---|
| exact viscous modes | 2.4674 | 22.21 |
| Richard 2016 as published (skewness moved into the sources) | 2.252 (−9%) | 31.08 |
| Richard 2016 with skewness kept as a flux | **2.4688** | 42.53 |
| our quartic three-moment model | **2.4688** | 42.53 |

- **Identical before the rewrite.** With the skewness kept as a flux, their model is linearly identical to ours.
- **What the rewrite costs.** Moving the skewness into the sources is what lowers the slow rate to −9%. The relaxation terms it creates stay active even when h_x = 0, where the real skewness flux would vanish.
- **The trade-off.** Unconditional hyperbolicity, at the cost of about 9% in unslaved relaxation. For row 1 that is a good trade. For rows 2–3 it is not.

### 6.4 Item 4: benchmarks they provide

- **Liu & Gollub (1994).** θ = 6.4°, Re = 19.33, 1.5 Hz. Their Fig. 5 shows the amplitude almost exact and the wavelength slightly long. Their Fig. 6 covers 3, 4.5 and 7 Hz. This is the same setting where `lavalle-cpp` already does well.
- **Tihon et al. (2006) wall shear.** θ = 5°, Re = 35 (their Fig. 11). The model predicts **flow reversal (τ_w < 0) at the first capillary trough**, in agreement with experiment and DNS. Our DNS shows local backflow even at Re = 10, β = 45° with broadband forcing (status doc §7). Their Eq. (5.11) is a ready analytical baseline for T_w that includes this effect; Lavalle's closure cannot produce it.
- **Noise-driven waves.** Their §5.4 uses the Chang et al. (1996) forcing: N = 1000 random-phase modes, cut-off 100 Hz, A = 0.001. Our production DNS uses the same construction (F₀ = 0.025, M = 1000, f_cut = 28 Hz), so the model can be run on our broadband cases directly.
- **Nakoryakov et al. (1977).** Their enstrophy-region comparison (Fig. 8) is a profile-shape benchmark, matching the DNS profile-projection test proposed in matrix §6, item 4.

### 6.5 The diffusion coefficient (confirms §5.3)

Their Eq. (3.31) is **exactly Lavalle's model** (flux 2λ²h⁵/225, friction (1/Re)(λh − 3U/h)) with its consistent second-order viscous term (ε/Re)∂ₓ((9/2)h ∂ₓU). The 9/2 comes from ⟨u₂⟩ feeding through the relaxation form of the friction (their Eq. 3.30). When the friction is written as the exact τ_xz(0), the coefficient is 15/4 instead (their Eq. 3.13).

- So for `lavalle-cpp`, which writes friction in relaxation form, the consistent replacement for the ad hoc (2/Re)q_xx is **(9/2Re)∂ₓ(h ∂ₓ(q/h))**.
- Linearised along kinematic waves, the current term gives 3U″ against the consistent 4.5U″. That confirms the "two-thirds" reading in §5.3.
- Richard et al. (2019) then rewrite the same term using α = 4/3 to fix the short-wave kinematic speed, which gives their Eq. (121).

**Hypothesis for ΔP_x.** Lavalle splits the 2q_xx add-on as q_xx/Re into P_x and q_xx/Re into D_x (status doc §3). The DNS ΔD_x is negligible, so the mismatch in the consistent second-order viscous term should show up in ΔP_x. This is probably a small part of ΔP_x, since its Ca dependence points to capillarity first. It is cheap to test: compare ΔP_x before and after replacing the P_x viscous part with the consistent 9/2 form.

### 6.6 Updated conclusions (replace §4)

1. **For the production regime (row 1, δ ≲ 1, Re ≈ 10–120): Richard et al. (2016) is the natural next ROM baseline.**
   - It is consistent at first order, with second-order diffusion.
   - It is unconditionally hyperbolic, has an energy equation, and its wall shear is coupled to the profile through φ.
   - It predicts flow reversal and is validated against Liu–Gollub and Tihon.
   - It uses **the same Noble–Vila W-augmented capillarity scheme as `lavalle-cpp`**. In code terms, it adds one conservation law (energy, or hφ) to the existing h, q, W system, changes the flux to P = h³φ + h²cos θ/(2F²), and replaces the sources with Eqs. (5.2)–(5.4).
   - That is a modest change to the solver, but it changes the ML targets (τ_w now depends on φ, and A₂ is evolved). So it belongs **after the 15 Dec submission**, as the Paper 2 baseline.
2. **For rows 2–3 (δ ≫ 1):** their skewness-to-source rewrite is not justified there. The consistent option is our three-moment model with skewness kept in the flux and the quartic mode, which is linearly identical to theirs before the rewrite. It needs a fix for the non-hyperbolic band at large positive s, for example a second mode or a hyperbolic regularisation. This remains open.
3. **Before the submission, the only change I would make to `lavalle-cpp` is the diffusion term:** 9/2 per their Eq. (3.31), or the α = 4/3 form of Richard et al. (2019). It is cheap, consistent, and keeps the augmented scheme valid.
4. **The quartic mode ψ_a is the right single deformation mode.** It matches their asymptotic τ_w(φ) and the exact slow viscous decay rate. The sextic is dropped.
