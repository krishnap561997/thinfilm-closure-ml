# Oracle *a posteriori* runs

Run the ROM with the exact DNS corrections ΔC(x, t) and compare with the DNS. This tests the closure framework independently of any learned model. Source for Paper 1 Sec. 5.2.

## Setup (Re120, Ca0.01, β45°, 10 Hz, t = 1800–1900)

- **DNS restart:** `dns/oracle/filmflow_restart_mono.c` from L800 `dump-001800` (job 48362419, 1 h 38 min). Inlet u = U_N(z)[1 + 0.03 sin(2π f t)], f = 10 Hz (0.0214 nondimensional, period ≈ 47). 1D column profiles every 0.01 (`dns/include/column_profiles.h`), 2D snapshots every 10, dumps every 50.
- **Conversion:** `profiles_to_closures.py` (job array 48561227) → `closures/closures_part_*.h5`. On the same state it matches the 2D extraction to round-off.
- **ROM forcing:** `pipeline/oracle/make_rom_forcing.py` (job 48562006): conservative cell averages of ΔM_x, ΔT_w, ΔP_x, ΔD_x onto N = 4000, plus the DNS state, initial state and inlet values.
- **ROM:** `rom/lavalle-cpp` oracle module (commit 47b9338), N = 4000, dt = 0.005, output every 0.1. DNS initial state with r = √(h/We) h_x; inlet h(0, t), q(0, t) from the DNS; corrections interpolated linearly in time at every RK4 stage and added to the q-equation as −ΔM_x − ΔT_w + ΔP_x + ΔD_x (ΔM_x = ∂ₓ[∫u² − (q²/h + Λ²h⁵/45)] only; the Lavalle flux and wall friction are unchanged). Job 48562007, ≈ 50 s per run.
- **Code layout:** all oracle code is in `rom/lavalle-cpp/{include,src}/oracle/`, hooked in by 6 lines tagged `// ORACLE`. Default runs are byte-identical to the pre-change code.
- **Data:** `$THINFILM_DATA/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/`: `run_t1800/`, `closures/`, `rom/{rom_forcing.h5, runs/*, summary.txt, summary.csv}`.
- **Summary / checks:** `scripts/oracle_summary.py`, `scripts/oracle_checks.py`.

## Checks

- **The restart reproduces L800:** h difference of 0.3%, 0.6% and 0.9% of std(h) at t = 1810, 1820 and 1830. The reconstructed source (the original was lost) is adequate.
- **Output interval 0.01 is fine** for ΔM_x, ΔT_w, h and q: linear-interpolation error over 0.02 is ≤ 0.06%. **ΔP_x is the exception:** 14% on the DNS grid, 5% on the ROM grid, so it has high-frequency temporal content.
- **Sanity tests of the oracle code:** a zero-forcing oracle run reproduces the default run exactly; constant ΔT_w, ΔM_x, ΔP_x = 0.01 change q over 2 time units by −0.0195, −0.0195, +0.0195 (expected ∓0.02, the shortfall being wall-friction relaxation).

## Results

h error relative to std(h_DNS), 20 < x < 760, as a function of elapsed time since t = 1800:

| elapsed | Lavalle | +ΔT_w | +ΔT_w+ΔM_x | all four |
|---|---|---|---|---|
| 0.1 | 0.006 | 0.007 | 0.006 | 0.003 |
| 1 | 0.115 | 0.283 | 0.060 | 0.014 |
| 2 | 0.260 | 0.680 | 0.095 | 0.028 |
| 5 | 0.536 | 1.63 | 0.150 | 0.164 |
| 10 | 0.606 | – | 0.172 | 0.329 |
| 20 | 0.590 | crashed at 7.9 | 0.199 | 0.555 |
| 50 | 0.585 | – | 0.192 | crashed at ≈30 |
| 99.9 | 1.07 | – | 0.550 | – |

Max h (ROM / DNS) at elapsed 50: Lavalle 1.45 / 2.12; ΔT_w + ΔM_x 2.10 / 2.12.

- **Lavalle alone underpredicts the waves** (max h 1.45 against 2.1), the known failure at this Re.
- **ΔT_w + ΔM_x** restores the wave height and is stable for the full 100 time units; the error rises near the end (cause not identified, possibly phase drift).
- **ΔT_w alone crashes** within 8 time units: h → 0 and the hyperbolicity check fails. ΔT_w and ΔM_x appear to largely cancel, so applying one without the other destabilises the ROM. This matters for how the networks are trained.
- **All four terms** is the most accurate early (0.014 after 1 time unit) but grows faster and crashes at t ≈ 1830. ΔP_x is the suspect given its temporal content; not yet confirmed.

## Open items

1. Correlation of ΔM_x and ΔT_w; train on the sum?
2. Rerun "all" with ΔP_x smoothed, and ΔT_w + ΔM_x + ΔP_x without ΔD_x, to find what causes the blow-up.
3. Re-initialise the ROM from the DNS every 5–10 time units, to separate closure error from accumulated phase drift.
4. Other cases (broadband forcing needs statistical metrics, see `docs/decisions.md`).
