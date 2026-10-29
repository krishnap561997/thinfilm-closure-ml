# Governing equations and closures

Everything here is nondimensional, using Nusselt scaling. This file is the reference for the
Lavalle baseline closures and for the definitions of the ML correction targets.

## 1. Scaling

- **Scales:** length h_N, velocity u_N, time h_N/u_N, pressure ρ_l u_N².
- **Nusselt profile:** $U_N(z) = \tfrac{3}{2}u_N\left[2\tfrac{z}{h_N} - \left(\tfrac{z}{h_N}\right)^2\right]$, with $u_N = \rho_l g\sin\beta\, h_N^2/(3\mu_l)$. The surface velocity is $1.5\,u_N$.
- **Dimensionless groups:** $Re = \rho_l u_N h_N/\mu_l = \rho_l^2 g\sin\beta\, h_N^3/(3\mu_l^2)$, $Ca = \mu_l u_N/\sigma$, $Fr = u_N/\sqrt{g h_N}$.
- **Dependent groups:**
  - $Fr^2 = Re\sin\beta/3$
  - $We = Re\,Ca$
  - $\Lambda = Re\sin\beta/Fr^2 = 3$
  - $q^{(0)} = \Lambda h^3/3 = h^3$
- **DNS control parameters:** (Re, Ca, β) and the inlet forcing.

In the Basilisk sources (`dns/production/filmflow0.c`) this appears as:

- ρ₁ = 1 and μ₁ = 1/Re;
- σ = 1/(Re·Ca);
- gravity G = (3/Re, −(3/Re)/tan β);
- inlet velocity $u = 1.5\,(2z - z^2)\,[1 + F(t)]$.

## 2. Depth-integrated two-equation model (passive gas, $\partial_z u|_h = 0$)

$$\partial_t h + \partial_x q = 0$$

$$\partial_t q + \underbrace{\partial_x\!\int_0^h u^2\,dz}_{M_x} = \underbrace{-\int_0^h \partial_x p\,dz}_{P_x} - \underbrace{\frac{1}{Re}\partial_z u|_{0}}_{T_w} + \underbrace{\frac{1}{Re}\int_0^h \partial_{xx}u\,dz}_{D_x} + \frac{\Lambda}{Re}h$$

## 3. Lavalle et al. (2015) first-order closures

$$M_{x,Lav} = \partial_x\!\left(\frac{q^2}{h} + \frac{\Lambda^2 h^5}{45}\right)$$

$$P_{x,Lav} \simeq \frac{h\,h_{xxx}}{We} - \partial_x\!\left(\frac{\cos\beta}{2Fr^2}h^2\right) + \frac{q_{xx}}{Re}$$

$$T_{w,Lav} = \frac{1}{Re}\frac{3q}{h^2} - \partial_x\!\left(\frac{\Lambda^2 h^5}{75}\right)$$

$$D_{x,Lav} \simeq \frac{q_{xx}}{Re}$$

Notes:

- **Momentum flux.** Shkadov gives $6q^2/(5h) = q^2/h + q^2/(5h)$. Replacing $q \to \Lambda h^3/3$ in the second term gives the $\Lambda^2 h^5/45$ contribution.
- **Wall shear.** The 1/Re factor applies **only** to $3q/h^2$, because $\tau_w = 3q/h^2 - \tfrac{Re}{15}\Lambda^2 h^4 h_x$. Only this form reproduces the $2\Lambda^2h^5/225$ coefficient below, since 1/45 − 1/75 = 2/225. ONR report Eq. (8) wrongly puts 1/Re in front of both terms.
- **Pressure gradient.** The $q_{xx}/Re$ term is the approximated second-order dissipative contribution from the z-momentum equation.
- **Streamwise diffusion.** $D_x$ is formally O(ε²).

### Final Lavalle model

$$\partial_t q + \partial_x\!\left(\frac{q^2}{h} + \frac{2\Lambda^2h^5}{225} + \frac{\cos\beta}{2Fr^2}h^2\right) = \frac{1}{Re}\left(\Lambda h - \frac{3q}{h^2}\right) + \frac{h\,h_{xxx}}{We} + \frac{2}{Re}q_{xx}$$

## 4. ML corrections

The DNS closures are split into the Lavalle closure plus a learned correction:

$$C_{DNS} = C_{Lav} + \Delta C_{ML},\quad C\in\{M_x, T_w, P_x, D_x\}$$

The augmented ROM is the final Lavalle model with $+S_{ML}$ added to the right-hand side:

$$S_{ML} = -\Delta M_x + \Delta P_x - \Delta T_w + \Delta D_x$$

In code these corrections are named `delta_Mx_ML`, `delta_Px_ML`, `delta_Tw_ML` and `delta_Dx_ML`. Some older files also have `delta_M_ML`, which is the undifferentiated flux.

## 5. Extraction from DNS (`pipeline/extract_closures/`)

- **Integrated quantities.** h, q, ∫u² and ∫∂ₓp are computed as α-weighted column integrals of cell-centred values.
- **Derivatives.** Spatial derivatives use a Savitzky–Golay filter, with the window and polynomial order chosen by a sensitivity study.
- **Wall gradient.** $\partial_z u|_0 \approx (3675u_1 - 1225u_2 + 441u_3 - 75u_4)/(840\Delta z)$, using cell centres at $(2k-1)\Delta z/2$. This is exact for a quartic with u(0) = 0, which was verified.
- **Interface velocity.** Obtained by linear interpolation across the interfacial cell, the last liquid cell and the first gas cell.

## 6. General passive-gas form without the long-wave assumption (paper Appendix A)

$$\partial_t q + \partial_x\!\left(\frac{q^2}{h}\Phi_{sf}\right) = \frac{1}{Re}\left(\Lambda h - \tau_w\right) + \frac{1}{We}\frac{h\,h_{xxx}}{(1+h_x^2)^{3/2}} - \frac{\cos\beta}{Fr^2}h h_x + T_{\epsilon^2} + T_{\epsilon^3} + T_{\epsilon^4}$$

- $\Phi_{sf}$ is the normalized shape factor.
- $\Phi_i = u|_h h/q$ is the normalized interface velocity.
- The higher-order terms T are given in paper Eqs. (A23)–(A25).
- The hydrostatic term must be $\cos\beta/Fr^2\,h h_x$. The draft currently has a spurious factor of 2.
