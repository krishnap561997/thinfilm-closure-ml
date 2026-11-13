# Ordering matrix for passive-gas falling-film closures

**Ca ∈ {ε, ε², ε³} × Re ∈ {ε⁻¹, ε⁻², ε⁻³}**

_Prepared 8 Nov 2026 with the `thinfilm-longwave-derivation` skill. Every expression below was derived or checked with sympy; the scripts are in `scripts/ordering-matrix/` (see §7). This is a draft for review: the judgement calls I made on my own are listed in §6._

---

## 0. Summary

1. **None of the nine cells allows a Lavalle-type slaved closure.**
   - In the momentum equation multiplied by Re, inertia has order δ = εRe = ε^{1−b} for Re = ε^{−b}. So in every row, inertia is as large as the viscous–gravity balance or larger.
   - The velocity profile is therefore never a local function of h and its derivatives. A closure always needs q as an independent variable, plus an assumption about the profile.
   - Lavalle's model is the Re ~ 1 row, which is not part of this matrix.

2. **Row Re ~ ε⁻¹ (δ ~ 1)** still gives a two-equation (h, q) model, provided the expansion is done in the **q-gauge**. The result is equivalent to the Ruyer-Quil–Manneville simplified weighted-residual model. Compared with Lavalle:
   - T_w gains a flux-disequilibrium term, (q_eq − q)/(2h²). The local equilibrium flux q_eq includes the hydrostatic and capillary gradients.
   - T_w also gains an inertial term, (1/35)(q/h²)h⁴∂ₓ(q/h³).
   - Lavalle's rearrangements (q²/5h → Λ²h⁵/45, moving −Λ²h⁵/75 into the flux) are **not valid** in this row, because they assume q ≈ h³.
   - In the δ ≪ 1 limit, the closure reduces exactly to Lavalle's −(Re/15)Λ²h⁴h_x.
   - The resulting model is identical to the energy-integral (EIM) model. Per Richard et al. (2019), that means it is not Galilean invariant and has no capillary energy (`richard-comparison.md` §5).

3. **Rows Re ~ ε⁻² and ε⁻³ (δ ≫ 1)** are inertia-dominated:
   - The leading-order flow is inviscid and pressure-free (shallow shear flow). The profile shape is advected rather than set by viscosity.
   - A two-equation model is no longer consistent. The minimal consistent truncation of the moment hierarchy is **three equations, for (h, q, A₂ = ∫u²)**, with one profile-deformation amplitude s.
   - Closures for A₃, the wall shear and the dissipation are given for two candidate deformation modes.
   - The pressure-free system **loses hyperbolicity** when s lies in a band of positive values, i.e. when the wall shear exceeds 3q/h². That is likely near wave fronts.

4. **Capillarity matters at first order in only three cells:**
   - (Ca ε², Re ε⁻¹), where it enters at O(ε);
   - (Ca ε³, Re ε⁻¹), where it enters at O(1);
   - (Ca ε³, Re ε⁻²), where it enters at O(ε).
   
   In all other cells it is second order or weaker.

5. **Non-hydrostatic (dispersive, Serre-type) pressure** is O(ε²) relative to the dominant balance in every cell. It enters the retained model only in the Re ~ ε⁻³ row, at second order. A closed form is given for the semi-parabolic profile, and it reduces to the Serre term for plug flow.

6. **Streamwise viscous diffusion** (D_x, and Lavalle's 2q_xx/Re) is beyond the retained order in every cell: ε², ε³ and ε⁴ relative, row by row.

---

## 1. Setup

The setup uses Nusselt scaling with a passive gas, as in the status doc §1. Because Fr² = Re sin β/3, Fr is not a free parameter: the hydrostatic term is Re·cos β/Fr² = 3 cot β. In this matrix cot β is taken to be O(1); see §6, item 1.

The x-momentum equation multiplied by Re reads:

$$
Re\,(u_t+uu_x+wu_z)=3+u_{zz}-3\cot\beta\,h_x+\frac{1}{Ca}h_{xxx}-Re\,\partial_x p_{NH}+\big[u_{xx}+\partial_x(w_z|_h+w_z)\big]
$$

Here $p_{NH}(z)=\int_z^h (w_t+uw_x+ww_z)\,dz'$ is the non-hydrostatic pressure. The curvature is linearised.

With ∂ₓ, ∂ₜ ~ ε, w ~ ε, Ca = ε^a and Re = ε^{−b}, the order of each term in this equation is:

| term | order (viscous term = 1) |
|---|---|
| inertia, Re(u_t + …) | ε^{1−b} |
| gravity 3, viscous u_zz | 1 |
| hydrostatic, 3 cot β h_x | ε |
| capillary, h_xxx/Ca | ε^{3−a} |
| non-hydrostatic pressure | ε^{3−b} |
| streamwise diffusion and viscous normal stress | ε² |

The next table gives the same orders relative to the dominant balance, which is what decides truncation:

| | inertia | visc + gravity | hydrostatic | capillary (a = 1, 2, 3) | NH pressure | diffusion |
|---|---|---|---|---|---|---|
| **Re ~ ε⁻¹** | 1 | 1 | ε | ε², ε, **1** | ε² | ε² |
| **Re ~ ε⁻²** | 1 | ε | ε² | ε³, ε², **ε** | ε² | ε³ |
| **Re ~ ε⁻³** | 1 | ε² | ε³ | ε⁴, ε³, ε² | ε² | ε⁴ |

**Truncation.** Models are consistent to O(ε) relative to the dominant balance. For the Re ~ ε⁻³ row, nothing other than inertia enters at O(ε), so I also give the O(ε²) model.

**Gauge.** I use the q-gauge throughout: q is independent, and ∫u⁽ⁿ⁾dz = 0 for n ≥ 1. The Lavalle gauge, which writes q⁽⁰⁾(h) and then replaces q⁽⁰⁾ by q, needs q − h³ to be small. That is not the case in any row here (skill, Step 6).

<span style="color:red">**What this means.** The expansion does not fix how the exact flux q = ∫u dz is shared among the orders, so a choice has to be made. The **Lavalle gauge** takes u⁽⁰⁾ = 3(hz − z²/2), the Nusselt profile for the local h, whose flux is fixed at q⁽⁰⁾ = h³. Closures are derived in terms of q⁽⁰⁾, and q⁽⁰⁾ is then replaced by the actual q. That swap is accurate only when q − h³ is small, i.e. at weak inertia (Re ~ 1); Lavalle's rewriting q²/5h → Λ²h⁵/45 relies on the same assumption. The **q-gauge** takes u⁽⁰⁾ = (3q/h³)(hz − z²/2), the same shape but carrying the actual flux, and requires every correction to carry zero net flux. q is then a genuinely independent variable, and the departure from equilibrium appears explicitly in the closures, e.g. the term (q_eq − q)/(2h²) in τ_w. In every row of this matrix inertia is at least as large as the viscous–gravity balance, so q − h³ is not small and the q-gauge is required. The two gauges agree when inertia is weak: the q-gauge reduces to Lavalle's −(Re/15)Λ²h⁴h_x (§3.5).</span>

---

## 2. Exact results (all cells)

These follow from the boundary-layer equations without any closure assumption. They define the closure targets.

**Mass**

$$h_t+q_x=0$$

**First moment (IBL, status doc §2)**

$$q_t+\partial_x A_2=\frac{1}{Re}(3h-\tau_w)-\int_0^h\partial_x p\,dz,\qquad A_n\equiv\int_0^h u^n dz$$

**Second moment**

This comes from multiplying x-momentum by 2u and integrating. It uses 2u Du/Dt = (u²)ₜ + (u³)ₓ + (wu²)_z, together with u(0) = 0 and u_z(h) = 0:

$$\partial_t A_2+\partial_x A_3=\frac{2}{Re}\big(3q-\mathcal D\big)-2\int_0^h u\,\partial_x p\,dz,\qquad \mathcal D\equiv\int_0^h u_z^2\,dz$$

**Pressure term**

$$P_x\equiv-\int_0^h\partial_x p\,dz=-\frac{\cos\beta}{Fr^2}h h_x+\frac{1}{We}h h_{xxx}-\partial_x\!\int_0^h z\,\frac{Dw}{Dt}\,dz+(\text{viscous, }O(\epsilon^2))$$

The non-hydrostatic part reduces to $-\partial_x\int_0^h z\,Dw/Dt\,dz$ after exchanging the order of integration, using $p_{NH}(h)=0$.

The exact hierarchy does not close. M_x needs A₂, the A₂ equation needs A₃ and 𝒟, and so on (the Benney 1973 moment chain). Every model below is a truncation of this hierarchy, together with a profile assumption.

---

## 3. Row Re ~ ε⁻¹ (δ ~ 1)

At leading order, δ(u_t + uu_x + wu_z) = 3 + u_zz (+ h_xxx/Ca when a = 3). This is a nonlinear PDE in z, so the profile is not slaved.

I follow the standard approach: expand around the semi-parabola carrying the actual flux q, count gradients to first order, and keep δ as a coefficient. The underlying assumption is that the deviations from the semi-parabola are small. That has to be checked against DNS profiles; see §6, item 4.

### 3.1 Velocity profile

$$u=\frac{q}{h}\Big[\varphi(\eta)+s_a\,\psi_a(\eta)+s_b\,\psi_b(\eta)\Big],\qquad \eta=z/h$$

The shape functions are:

- $\varphi=3(\eta-\tfrac12\eta^2)$, the semi-parabola with mean 1;
- $\psi_a=\eta-3\eta^2+\tfrac52\eta^3-\tfrac58\eta^4$, the **flux-disequilibrium mode**;
- $\psi_b=\eta-\tfrac{15}{2}\eta^3+\tfrac{85}{8}\eta^4-\tfrac{21}{4}\eta^5+\tfrac78\eta^6$, the **inertial mode**.

<span style="color:red">**Where these shapes come from.** They are not chosen; they come out of the first-order solution (`rowb1.py`). Substituting u⁽⁰⁾ = (q/h)φ(η) into x-momentum leaves an ODE u⁽¹⁾_zz = F(z). The right-hand side F has two kinds of forcing. One is constant across the depth: the gravity–drag imbalance 3 − 3q/h³, the hydrostatic gradient and the capillary term, together with a parabola-shaped q_t piece from the time derivative. The other is inertial, Re(u⁽⁰⁾_t + u⁽⁰⁾u⁽⁰⁾_x + w⁽⁰⁾u⁽⁰⁾_z), which is a polynomial of degree up to 4 in η. Integrating twice with u⁽¹⁾(0) = 0, u⁽¹⁾_z(h) = 0 and ∫u⁽¹⁾dz = 0 (the q-gauge) gives three conditions on a second-order ODE, so the third acts as a solvability condition. It fixes q_t, giving the Galerkin equation of §3.3. What remains separates by source into two shapes. ψ_a (quartic, ψ_a″ = 5φ − 6) comes from the constant-plus-parabola forcing and is driven by q_eq − q. ψ_b (sextic, ψ_b″ quartic) comes from the inertial forcing; its amplitude is ∝ Re, and both inertial terms turn out to share this one shape. Each mode is normalised to ψ′(0) = 1, so its amplitude is its fractional contribution to the excess wall shear: τ_w = (q/h²)(3 + s_a + s_b).</span>

Both modes satisfy ψ(0) = 0, ψ′(1) = 0, ∫₀¹ψ = 0 and ψ′(0) = 1. Their amplitudes are slaved algebraically:

$$s_a=\frac{q_{eq}-q}{2q},\qquad q_{eq}=h^3\Big(1-\cot\beta\,h_x+\frac{h_{xxx}}{3\,Ca}\Big),\qquad s_b=\frac{Re}{35}\,h^4\,\partial_x\!\Big(\frac{q}{h^3}\Big)$$

q_eq is the local-equilibrium flux, including the pressure gradients. In s_b, h⁴∂ₓ(q/h³) = h q_x − 3q h_x.

**Cross-stream velocity**

$$w=-\partial_x\Big|_z\Big[q\big(\Phi+s_a\Psi_a+s_b\Psi_b\big)\Big]$$

with:

- $\Phi=\tfrac32\eta^2-\tfrac12\eta^3$;
- $\Psi_a=\tfrac12\eta^2-\eta^3+\tfrac58\eta^4-\tfrac18\eta^5$;
- $\Psi_b=\tfrac12\eta^2-\tfrac{15}{8}\eta^4+\tfrac{17}{8}\eta^5-\tfrac78\eta^6+\tfrac18\eta^7$.

### 3.2 Closures

**Wall shear**

$$\tau_w=\frac{q}{h^2}\,(3+s_a+s_b)$$

Written as T_w = τ_w/Re in project groups:

$$
T_w=\frac{1}{Re}\frac{3q}{h^2}+\frac{1}{Re}\frac{h^3-q}{2h^2}-\frac{\cos\beta}{6Fr^2}h h_x+\frac{1}{6We}h h_{xxx}+\frac{1}{35}\Big(\frac{q\,q_x}{h}-\frac{3q^2h_x}{h^2}\Big)
$$

As in Lavalle, the inertial term carries no 1/Re.

**Momentum flux**

$$M_x=\partial_x\!\Big(\frac{6q^2}{5h}\Big)$$

The profile-consistent value is $A_2=\frac{q^2}{h}\big(\tfrac65-\tfrac{2}{35}s_a-\tfrac{2}{21}s_b\big)+O(s^2)$. Its derivative is second order, so it is not retained.

**Pressure**

$$P_x=-\frac{\cos\beta}{Fr^2}h h_x+\frac{1}{We}h h_{xxx}$$

The capillary term is O(ε^{3−a}), so it is dropped when a = 1.

**Streamwise diffusion:** D_x is O(ε²) and is not retained.

### 3.3 Equivalent single equation

Integrating the profile with these closures gives exactly (checked symbolically) the solvability condition ∫u₁ = 0. That condition is the Galerkin projection with the parabola as weight:

$$q_t=\frac{5}{6}\Big[\frac{1}{Re}\Big(3h-\frac{3q}{h^2}\Big)-\frac{\cos\beta}{Fr^2}h h_x+\frac{1}{We}h h_{xxx}\Big]-\frac{17}{7}\frac{q}{h}q_x+\frac{9}{7}\frac{q^2}{h^2}h_x$$

This is the Ruyer-Quil–Manneville simplified model, written in project units.

### 3.4 Dependence on Ca within this row

| Ca | Capillary order | Effect |
|---|---|---|
| ε | ε² | Dropped from q_eq. The first-order model has no capillary cutoff and is ill-posed for ripples, so it needs a labelled add-on as regularisation. |
| ε² | ε | Enters exactly as written above. |
| ε³ | 1 | Same expressions. Capillarity enters only through q_eq, so it adds no new terms. |

In the Lavalle gauge, the Ca ε³ case would generate terms in K² and K·Re with derivatives up to h_{7x} (output of the skill script). This is the strongest practical argument for using the q-gauge.

### 3.5 Checks

- **Lavalle limit.** Substituting q = h³ + q₁ (Benney) into τ_w gives 3q/h² − (3/5)Re h⁴h_x = 3q/h² − (Re/15)Λ²h⁴h_x. The cot β and h_xxx terms cancel, as they should.
- **Benney limit.** Eliminating q from §3.3 gives q₁ = (6/5)Re h⁶h_x − cot β h³h_x. This is Benney's correction, which implies the threshold Re_c = (5/6)cot β.
- **Uniform state.** h = q = 1 gives s_a = s_b = 0, so it is a steady solution.

### 3.6 Implications for the ML closures (hypotheses to test)

1. **The cancellation between ΔT_w and ΔM_x.** Relative to the q-gauge row-1 closures, Lavalle's two closures both err through the same departure, q − h³:
   - Lavalle's M_x is off by ∂ₓ[(q² − h⁶)/(5h)];
   - Lavalle's T_w misses (h³ − q)/(2Re h²) + (1/35)(q q_x/h − 3q²h_x/h²) + (3/5)h⁴h_x, plus the pressure pieces.
   
   So ΔM_x and ΔT_w should be strongly correlated in δ ~ 1 cases, which fits the oracle observation that they largely cancel (status doc §7). Whether the cancellation is close to exact depends on the signs and magnitudes of these two error terms in each case. That has to be checked on the data and is not established here.
2. **Candidate inputs for ΔT_w** are (q_eq − q)/h² and h⁴∂ₓ(q/h³). The second carries q_x with a larger weight than h_x, which is consistent with the a priori finding that q_x matters more than h_x.
3. **Splitting depends on the gauge.** One sixth of the pressure terms ends up in T_w in the q-gauge. How a DNS difference divides between ΔT_w and ΔP_x is therefore a gauge choice, not a physical statement.

---

## 4. Rows Re ~ ε⁻² and Re ~ ε⁻³ (δ ≫ 1)

### 4.1 Leading order and why two equations are not enough

At O(1) the flow obeys $u_t+uu_x+wu_z=0$: inviscid, pressure-free, shallow shear flow. Every profile is a solution, and u is conserved along particle paths. The profile is therefore neither slaved nor relaxed toward the semi-parabola on the wave time scale. Viscosity enters only at relative order 1/δ, i.e. ε in row 2 and ε² in row 3.

A consistent model has to evolve information about the profile shape. The minimal truncation of the moment hierarchy (§2) is three moments, (h, q, A₂), with a one-parameter profile family:

$$u=\frac{q}{h}\big[\varphi(\eta)+s\,\psi(\eta)\big],\qquad \tau_w=\frac{q}{h^2}(3+s),\qquad w=-\partial_x|_z\big[q(\Phi+s\Psi)\big]$$

Here ψ is one of the two row-1 modes. In row 1 the amplitudes are slaved algebraically; in these rows the amplitude becomes a dynamic variable.

### 4.2 Closures for the profile family

The closures are $A_2=\frac{q^2}{h}f_2(s)$, $A_3=\frac{q^3}{h^2}f_3(s)$, $\mathcal D=\frac{q^2}{h^3}f_D(s)$ and $u|_h=\frac{q}{h}f_s(s)$:

| | ψ_a (quartic) | ψ_b (sextic) |
|---|---|---|
| f₂ | 6/5 − (2/35)s + (2/315)s² | 6/5 − (2/21)s + (194/9009)s² |
| f₃ | 54/35 − (6/35)s + (8/385)s² − (2/15015)s³ | 54/35 − (114/385)s + (376/5005)s² − (1366/969969)s³ |
| f_D | 3 + s²/7 | 3 + (101/231)s² |
| f_s | (12 − s)/8 | (6 − s)/4 |
| s(X), X = hA₂/q² | 9/2 − (3/2)√(70X − 75) | 429/194 − (3/970)√(4854850X − 5314595) |
| realizable X ≥ | 15/14 (branch s < 9/2) | 7433/6790 (branch s < 429/194) |
| non-hyperbolic band (pressure-free) | 1.61 < s < 2.91 | 0.48 < s < 1.31 |

For comparison, the Cauchy–Schwarz bound is X ≥ 1 (plug flow). The s(X) branch is the one with s → 0 as X → 6/5.

### 4.3 Models by cell

**Row Re ~ ε⁻²:** inertia is O(1), viscosity and gravity are O(ε).

$$
\begin{aligned}
&h_t+q_x=0\\
&q_t+\partial_x A_2=\tfrac{1}{Re}(3h-\tau_w)\;\big[+\tfrac{1}{We}h h_{xxx}\big]\\
&\partial_t A_2+\partial_x A_3=\tfrac{2}{Re}(3q-\mathcal D)\;\big[+\tfrac{2}{We}q\,h_{xxx}\big]
\end{aligned}
$$

- The bracketed capillary terms are present only for Ca ~ ε³. For Ca ~ ε and Ca ~ ε², the first-order models are identical.
- At the next order (ε²), these terms enter:
  - the hydrostatic terms, −(cos β/Fr²)h h_x in the q equation and −2(cos β/Fr²)q h_x in the A₂ equation;
  - capillarity for Ca ~ ε²;
  - the non-hydrostatic terms of §4.4.

**Row Re ~ ε⁻³:** the first-order model is pressure-free and inviscid.

$$q_t+\partial_xA_2=0,\qquad \partial_tA_2+\partial_xA_3=0$$

The second-order model (ε²) adds:

- the viscous terms of row 2;
- the non-hydrostatic terms of §4.4;
- capillarity for Ca ~ ε³.

The hydrostatic term and capillarity for Ca ~ ε or ε² are O(ε³) or weaker.

**Pressure term P_x by cell**

| | Ca ~ ε | Ca ~ ε² | Ca ~ ε³ |
|---|---|---|---|
| Re ~ ε⁻¹ | −(cos β/Fr²)h h_x | −(cos β/Fr²)h h_x + (1/We)h h_xxx | same as Ca ~ ε², with capillarity at O(1) |
| Re ~ ε⁻² | 0 (ε² terms next) | 0 (ε² terms next) | (1/We)h h_xxx |
| Re ~ ε⁻³ | 0 to O(ε²) except NH | 0 to O(ε²) except NH | NH + (1/We)h h_xxx at O(ε²) |

### 4.4 Non-hydrostatic pressure (Re ~ ε⁻³ at second order; ε² terms of every row)

This is computed for the semi-parabolic profile (s = 0), with leading-order time derivatives h_t = −q_x and q_t = −∂ₓ(6q²/5h).

**In the q equation:**

$$P_{x,NH}=-\partial_x\mathcal I,\qquad
\mathcal I=-\frac{241}{1400}h\,q\,q_{xx}+\frac{51}{175}q^2h_{xx}+\frac{829h^2q_x^2-1715h\,h_xq\,q_x+810h_x^2q^2}{1400\,h}$$

As a check, the same calculation for plug flow reproduces the Serre/Green–Naghdi term $\tfrac{h^3}{3}(U_x^2-U_{xt}-UU_{xx})$ exactly.

**In the A₂ equation:** the source is −2∫u∂ₓp_NH dz. Its closed form is long and is in the output of `nh2.py`.

**For s ≠ 0:** the general formula $-\partial_x\int_0^h z\,Dw/Dt\,dz$ applies with w from §4.1. I have not expanded it.

### 4.5 Checks and caveats

- **Uniform state.** h = q = 1 with s = 0 is steady in both rows, since τ_w = 3 and 𝒟 = 3.
- **Hyperbolicity.** The pressure-free three-moment system has complex characteristic speeds for s in the bands listed in §4.2. Since A₂ and A₃ are homogeneous in (h, q), these bands do not depend on h or q.
  - s > 0 means the wall shear exceeds 3q/h², which is expected at wave fronts. So the closure can lose hyperbolicity exactly where ΔT_w is largest.
  - The hydrostatic term (next order) and capillarity both act to restore it. Which mode to use is the main open choice in these rows.
- **Wall layer.** For δ ≫ 1, viscous effects over one wave period are confined to a Stokes layer of thickness about δ^{−1/2}h. That is about ε^{1/2}h in row 2 and εh in row 3. A single global polynomial cannot resolve this layer, so the τ_w closure is the least reliable part of these models. It is also where an ML correction would add most.
- **Physical range.** Re ~ ε⁻³ means Re ≈ 10³ to 10⁶ for ε between 0.1 and 0.01. That is beyond the laminar falling-film range, so row 3 is a formal limit only. Row 2 at ε ≈ 0.1 means Re ≈ 100, which matters for the Re 80–200 cases with short waves.

---

## 5. Cell summary

| Ca \ Re | ε⁻¹ (δ ~ 1) | ε⁻² (δ ~ ε⁻¹) | ε⁻³ (δ ~ ε⁻²) |
|---|---|---|---|
| **ε** | (h, q); T_w as in §3.2 with q_eq = h³(1 − cot β h_x); no capillarity (ill-posed for ripples) | (h, q, A₂); viscous terms at O(ε); P_x = 0 | (h, q, A₂); pressure-free inviscid at O(ε); viscous and NH at O(ε²) |
| **ε²** | (h, q); full §3.2, capillarity at O(ε) | same as Ca ~ ε at first order; capillarity at ε² | same as Ca ~ ε |
| **ε³** | (h, q); §3.2 with capillarity at O(1), inside q_eq | (h, q, A₂); capillarity in both moment equations at O(ε) | as Ca ~ ε, plus capillarity at O(ε²) |

In every cell, M_x = ∂ₓA₂: 6q²/(5h) in row 1, and (q²/h)f₂(s) in rows 2–3. D_x is never retained.

---

## 6. Choices I made on my own (please check)

1. **cot β = O(1).** For β = 6.4°, cot β ≈ 8.9, which may be closer to ε⁻¹ if ε ≈ 0.1. In that case the hydrostatic term moves up one order:
   - in row 1 it becomes O(1), but it still enters only through q_eq, so nothing changes in form;
   - in row 2 it enters the first-order model, in both moment equations;
   - it also widens the hyperbolic range.
2. **Truncation.** Models are first order relative to the dominant balance, with second order added for row 3.
3. **q-gauge everywhere.** The Lavalle gauge is inconsistent in all nine cells.
4. **Which mode to promote in rows 2–3** (ψ_a or ψ_b, or both, giving a four-equation model). A direct test on the DNS is possible:
   - project u(z)·h/q − φ(η) from the 2D snapshots onto ψ_a and ψ_b;
   - check which mode explains the variance and what values s takes along a wave;
   - compare those values against the non-hyperbolic bands.
   
   The same projection tests the "small deviation" assumption of row 1.
5. **Curvature is linearised.** Where |h_x| is not small (fronts, ripples), the full curvature of status doc §5 should replace h_xxx.
6. **The non-hydrostatic term** is closed only for s = 0.
7. **Literature check.** This is done in `richard-comparison.md` §§2, 5 and 6, covering the 2016, 2019 and 2023 papers. Summary:
   - our quartic three-moment model is linearly identical to Richard et al. (2016) before their skewness-to-relaxation rewrite;
   - that rewrite buys unconditional hyperbolicity, but it is valid only for row 1;
   - ψ_a is confirmed as the mode to use (item 4 above is now settled in its favour).

## 7. Verification log and scripts

`scripts/ordering-matrix/` contains:

- **`rowb1.py`:** the q-gauge expansion for row 1.
  - It confirms ∫u₁ = 0.
  - Its solvability condition is the Galerkin equation of §3.3.
  - The plain IBL with the §3.2 closures, minus that Galerkin equation, is **identically 0**.
- **`checks.py`:** the Lavalle/Benney limit of τ_w (first-order term −(3/5)Re h⁴h_x ✓), and Jacobian eigenvalues at sample values of s.
- **`rowb23.py`:** the family integrals in §4.2 (with checks of the boundary conditions and zero mean), and the non-hydrostatic integral, with the plug-flow Serre check ✓.
- **`scan.py`:** the hyperbolicity bands.
- **`nh2.py`:** the non-hydrostatic source in the A₂ equation.

I did the Benney reduction of §3.3 by hand (coefficients 6/5 and −cot β ✓).

## 8. Suggested next steps

1. Estimate ε and δ = εRe for each DNS case, and place each case in a row of the matrix. Also compute the budget-ratio diagnostic from the earlier discussion.
2. Project the DNS profiles onto φ, ψ_a and ψ_b (§6, item 4). This one diagnostic tests the row-1 assumption and chooses the mode for rows 2–3.
3. Test hypothesis §3.6.1 on the existing closure files: regress ΔM_x and ΔT_w on ∂ₓ[(q² − h⁶)/h] and (q − h³)/h².
4. ~~Try the q-gauge closures of §3.2 in `lavalle-cpp`.~~ **Withdrawn (9 Nov).** The row-1 model is identical to the EIM/Ruyer-Quil–Manneville model. Richard et al. (2019) show it has no capillary energy, so it would break the Noble–Vila augmented capillarity scheme that `lavalle-cpp` relies on. Their MIME-class improvements are safer: α = 4/3 diffusion and/or C₁ ≈ 0.979. See `richard-comparison.md` §5.

---

## 9. Candidate library for symbolic regression of ΔT_w and ΔM_x

_Added 9 Nov. The terms below come from the row-1 q-gauge closures (§3), from Lavalle's own terms (so the regression can correct his coefficients), and from the slaved first-order enstrophy of Richard et al. (2016) (`richard-comparison.md` §6). They are grouped in tiers: start with Tier 1 and add Tiers 2–3 only if the residual requires them._

### 9.1 Targets and constraints

1. **Regress the flux ΔF, not ΔM_x.** The target is ΔF = ∫u² − (q²/h + Λ²h⁵/45), with ΔM_x = ∂ₓΔF (decision of 29 Oct). Every theory term below is naturally a flux, and differentiating noisy data first would amplify the derivative terms.
2. **Also fit the combination ΔT_w + ∂ₓΔF.** By the theory, the two errors are driven by the same departure, q − h³ (§3.6, item 1). Separate fits will therefore have strongly correlated, partly cancelling coefficients, and the combination is better conditioned. It also matches the oracle finding that ΔT_w and ΔM_x must be applied together.
3. **Every term must vanish for uniform Nusselt flow (h constant, q = h³),** because Lavalle's model is exact there. So bare monomials such as h or q/h² are excluded, and non-derivative terms enter only through q − h³.
4. **Project units throughout:** T_w = τ_w/Re, and cos β/Fr² = 3 cot β/Re, 1/We = 1/(Re Ca).

### 9.2 Library for ΔT_w (pointwise)

The "Expected coefficient" column gives the value predicted by the row-1 q-gauge theory relative to Lavalle, valid for δ = εRe ≲ 1. Note that q_eq − q appears through both (h³ − q) and the cot β and capillary terms.

| Tier | Term | Parameter factor | Expected coefficient | Origin and justification |
|---|---|---|---|---|
| 1 | (h³ − q)/h² | 1/Re | 1/2 | Flux disequilibrium, mode ψ_a with amplitude s_a (§3.1). Absent from Lavalle, who assumes q ≈ h³. It is the main q-dependence, and it can drive τ_w negative (backflow). |
| 1 | h⁴h_x | 1 | 3/5 | Undoes Lavalle's inertial term −(3/5)h⁴h_x. At δ ~ 1 its effective coefficient changes, so the regression should find it. |
| 1 | q q_x/h | 1 | 1/35 | Inertial mode ψ_b, amplitude s_b = (Re/35)h⁴∂ₓ(q/h³). Carries q_x, consistent with the a priori finding that q_x matters more than h_x. |
| 1 | q²h_x/h² | 1 | −3/35 | Second half of the same inertial mode. Offered separately so the regression can test whether the combination h⁴∂ₓ(q/h³) holds. |
| 1 | h h_x | cot β/Re | −1/2 | Hydrostatic part of q_eq (equivalently −1/6 of cos β/Fr² h h_x). In the q-gauge, one sixth of the pressure term sits in T_w. |
| 2 | h⁵h_x | 1 | from slaved φ₁ | Richard (2016) τ_w depends on enstrophy, and φ₁ ∝ Re[(3/7)λ²h³ − cos θ/F²]h²h_x (their Eq. 3.36). |
| 2 | h²h_x | cot β/Re | from slaved φ₁ | cos β part of the same enstrophy correction. |
| 2 | (q − h³)h_x/h, (q − h³)q_x/h² | 1/Re or 1 | — | Nonlinear coupling of disequilibrium and gradients. The freedom in Richard's f₁, f̃₂ (their §3.4) lives here. |
| 3 | h h_xxx | 1/We | 1/6 | Capillary part of q_eq. Higher derivative, so expected small, but it links to ΔP_x. |
| 3 | h⁵h_xx, h⁴h_x² | Re | — | Second-order inertial terms, Richard (2016) App. A. |
| 3 | h²h_xx, h h_x² | 1/Re | — | Second-order viscous terms, same source. |

### 9.3 Library for ΔF (momentum flux)

| Tier | Term | Parameter factor | Expected coefficient | Origin and justification |
|---|---|---|---|---|
| 1 | (q − h³)q/h | 1 | 8/35 | Lavalle's substitution q²/5h → Λ²h⁵/45 (invalid when q ≠ h³), plus the ψ_a contribution to ∫u². Together with the next term it spans (q² − h⁶)/h. |
| 1 | (q − h³)h² | 1 | 1/5 | Second basis function for the same substitution error. |
| 1 | q²q_x − 3q³h_x/h | Re | −2/735 | Inertial mode ψ_b inside ∫u², from A₂ = (q²/h)(6/5 − (2/35)s_a − (2/21)s_b). |
| 1 | q h²h_x | cot β | 1/35 | Hydrostatic part of s_a inside ∫u². |
| 2 | h⁸h_x | Re | 6/175 (Lavalle gauge) or 18/35 (slaved φ₁) | First-order correction to ∫u² at fixed h; also the λ² part of the slaved enstrophy h³φ₁. |
| 2 | h⁵h_x | cot β | −2/5 | cos β part of the slaved enstrophy h³φ₁. |
| 3 | q h²h_xxx | 1/Ca | −1/105 | Capillary part of s_a inside ∫u². |

The Tier-2 h⁸h_x coefficient depends on which gauge the comparison is made in. That is a reminder that the split between ΔF and ΔT_w is itself a gauge choice (§3.6, item 3).

### 9.4 Practical notes

1. **Parameter factors.** The dataset spans Re, Ca and β, so there are two options:
   - fit each case separately and examine how each coefficient trends with δ = εRe;
   - include the factors in the columns above as fixed multipliers (parametric SINDy).
   
   The theory predicts the Tier-1 coefficients only for δ ≲ 1, and how they depart from those values as δ grows is the actual scientific result. I would start per case.
2. **Collinearity.** With h ≈ 1 ± 0.5, terms such as h⁴h_x, h⁵h_x and h²h_x are nearly collinear. Rewriting the library in r = q/h³ − 1 and h⁴∂ₓ(q/h³), the natural row-1 variables, conditions it much better. Both are still functions of h, q, h_x and q_x. Sparse regression with sequential thresholding, or a stability-selection pass, will help decide among the near-duplicates.
3. **What a Markovian library cannot capture.** Richard et al. (2016) and §4 both show that τ_w depends on the enstrophy φ, which is not slaved when δ ≳ 1. A library in (h, q, h_x, q_x) captures only its slaved part. As a **diagnostic term only, not a ROM input**, add φ − h²/5, computed from the DNS ∫u² as φ = (∫u² − q²/h)/h³. A large coefficient on it measures how much a two-equation closure fundamentally misses, and makes the quantitative case for the three-equation baseline (`richard-comparison.md` §6.6).
4. **Validation.** For the lowest-δ cases, Tier 1 alone should recover the expected coefficients in §§9.2–9.3. That checks the pipeline (derivative estimation, units, sign conventions) before the regression is trusted on high-δ cases.
5. **Scope.** ΔP_x needs its own library, built around curvature κ = h_xx/(1 + h_x²)^{3/2}, h h_xxx and the viscous second-order pressure (`richard-comparison.md` §6.5). It is not covered here. ΔD_x is negligible and was dropped (status doc §7).
6. **If the baseline changes.** Switching `lavalle-cpp` to the consistent 9/2 diffusion, or to C₁ ≈ 0.979 (`richard-comparison.md` §5.3), changes the baseline and hence the expected coefficients. Recompute §§9.2–9.3 before comparing.
