# Decision log

Record each decision as a short entry: what was decided and why. Newest entries go at the top.

## 2026-10 — Repository restructure

- Code and data were split. The old /blue checkout is now the data store, and git holds only source code, small configs and manifests.
- Pre-restructure state is tagged `pre-restructure-2026-10`. Every removed binary, log, notebook output and figure is recoverable from that tag.

## Earlier (reconstructed from the ONR report, Oct 2026)

**Baseline model**

- The baseline ROM is the first-order Lavalle et al. (2015) two-equation model.
- It is Galilean invariant, has a mathematical energy structure, admits Noble–Vila stable capillarity, and gives good neutral curves (Richard et al. 2019).

**DNS setup**

- The DNS uses broadband inlet forcing (Chang et al. 1996) rather than a sweep over monochromatic frequencies, so that a single run samples many wave states.
- Settings: F₀ = 0.025, M = 1000, f_cut = 28 Hz.
- Open question: f_cut is dimensional and the same for every case. Check it against 2f₀ for each case.
- The parameter space is (Re, Ca, β), following Dhopeshwar et al. (2025), so that inertia and capillarity can be varied independently.
- Candidate cases were kept only if a glycerol–water mixture can realise them (≤ 90 wt%, h_N between 0.1 and 2 mm, u_N < 0.5 m/s).
- MAXLEVEL 14 was chosen over 15 for cost; level 15 changes only the wave speed slightly.

**ML scope**

- ΔD_x is dropped from the ML targets because it is negligible.
- Priority targets are ΔM_x and ΔT_w; ΔP_x is under investigation.
