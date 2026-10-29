# Decision log

Record each decision as a short entry: what was decided and why. Newest entries go at the top.

## 2026-10-29 — Paper 1 scope, venue and closure targets

- **Venue and dates.** Target IJHMT or AI Thermal Fluids. Submit 15 Dec 2026. Results for the APS DFD talk (22 Nov) freeze on 15 Nov.
- **Scope.** Paper 1 covers ΔM_x, ΔT_w and ΔP_x, with ΔP_x via curvature inputs. A memory-model Paper 2 happens only if the oracle shows ΔP_x matters *a posteriori* and curvature does not fix it; otherwise the next paper is the gas-sheared or heated film.
- **Flux-form target for M_x.** ΔF = ∫u² − (q²/h + Λ²h⁵/45), the momentum part only, and ΔM_x = ∂ₓΔF. This keeps the equation conservative and gives a smoother target. The −Λ²h⁵/75 that Lavalle moves into the flux from T_w stays with T_w; ΔT_w remains a pointwise source measured against the full T_w,Lav.
- **Oracle a posteriori first.** Run the ROM with exact DNS corrections over short horizons from a DNS state. Start with the monochromatic Re = 120, β = 45°, 10 Hz case, re-run as a short restart window with output every 0.1–0.5 time units, because 10-unit h5 output is too coarse to interpolate in time.
- **Statistical metrics for broadband cases.** Broadband ROM and DNS trajectories decorrelate, so compare h PDFs, amplitude and celerity, spectra and growth. Use the monochromatic cases for snapshot-by-snapshot comparison.

## 2026-10 — Basilisk version

- All thin-film DNS are built with `~/basilisk-old/src`, which is compatible with the HDF5/XDMF output headers.
- The newer Basilisk install belongs to the heterogeneous-nucleation project and cannot write these HDF5 files.
- Eight case Makefiles reference `$BASILISK`, but those cases were also run with basilisk-old.

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
