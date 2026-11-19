# Progress tracker: structured-closure / Orr–Sommerfeld plan

_Context for Claude sessions. Updated 17 Nov 2026 (end of day 2). Plan: `structured-closure-os-plan.md`. Branch: `structured-closure` (from `main` @ 719b7da, not pushed). Results: `docs/results/orr-sommerfeld.md`, `docs/results/structured-closure.md`. Explainer for Krishna: `docs/explainers/2026-11-16-structured-closure-explainer.md`._

## Decisions taken (16 Nov)

- **Branch from main.** The functional-form plan (FF plan) is copied in, marked *secondary*. Its Phase 0 lives on `functional-form-v2` (not merged).
- **D2 = L2:** the closure's linear part must match OS through (a, b, c, d), fitted over the forced band with D = D_OS imposed. Consequence: g₃ (the h_xx term) is in the default structured model.
- **D3 = yes, as short comb-forced DNS from scratch** (not broadband restarts): five β = 45° cases, LX = 200, comb up to 0.97 ω_c.
- **Inlet problem:** investigated. It was a write-out artefact, now fixed (`5df0595`).
- **Extract all pending closure files,** and fully redo the four stale ones (`decisions.md`, 16 Nov).

## Task status

| Task | Status | Where |
|---|---|---|
| 0.0 plan in repo | done | `docs/plans/` |
| B.1 temporal OS, B.2 spatial OS | done | `analysis/orr_sommerfeld/os_solver.py` |
| B.3 verification | done, all pass | `verify.py` |
| B.4 case table (85 cases) | done | `case_table.py`, `summarize_case_table.py` |
| B.5 DNS linear region | oracle case done; comb DNS done 17 Nov (Re 80–200 match OS; Re 40 matches in x 30–100; Re 20 correction below scatter) | `b5_dns_linear.py`, `comb_dns/analyse_comb.py` |
| B.6 ROM linear (comb inlet) | done (3 cases × Lavalle, sum, separate) | `b6_rom/` |
| B.7 linear closure targets | done (85 cases); DNS check done for the oracle case at 10 Hz | `linear_targets.py`, `b7_dns_check.py` |
| B.8 (a, b, c, d) fits | done (85 cases) | `linear_targets.csv`, columns `band_abcd_*` |
| Gate GB | passed; D2 = L2 | `orr-sommerfeld.md`, "Gate GB summary" |
| A.1–A.4 Phase A | done | `scripts/structured-closure/` |
| ROM first-cell fix | done (`5df0595`) | `rom/lavalle-cpp/src/main.cpp` (`INLETFIX`) |
| ROM `none` oracle option | done | `main.cpp` |
| Cache v2 (FF 1.1, shared) | building | `pipeline/cache_v2/build_cache_v2.py` |
| C.1–C.3 training | V1 sweep done 17 Nov (seed 0; seed 1 = backup): `ml/structured_closure/` (`train.py`, `run_sweep.sh`, `summarize_runs.py`); L2 = OS dispersion residual at 85 OS points (`os_dispersion.py`, `losses.py`), λ_lin = 10 | runs in `$THINFILM_STORE/codes/ml-models/structured_closure/runs/` |
| C.4–C.10 solver | done 17 Nov: `StructuredClosure` (tables in h, `export_tables.py`), ΔP flux hook, relaxation source with zero-sum D_G; uniform film exact, comb runs match the model's linear dispersion (3–4%) | `rom/lavalle-cpp` tag `// STRUCTCLOSURE`; `rom_tests.py make/analyse`, `run_rom_tests.sh` |
| Phase D closed loop | first oracle run 17 Nov: structured ≈ ml_sum to t = 5, then crest over-amplification (max h 2.3–3.0 vs DNS 2.1), h error 0.6–1.2 at t = 50 (ml_sum 0.47, exact closure 0.19). Diagnosed 17 Nov (see facts below); power penalty tried, did not help a posteriori | `structured_closure/rom/<run>/oracle_summary.csv`, `oracle_split_summary.csv`, `oracle_power_summary.csv`; `diag_oracle.py` |
| Phase E closed forms, F rollout | not started | |

## Jobs submitted on 16 Nov (check once, do not poll)

| Jobs | What | When done |
|---|---|---|
| 51592372–76 | comb DNS (Re 20, 40, 80, 120, 200 at β 45), `$THINFILM_STORE/datasets/BasC/DNS/comb/<case>/` | Krishna runs `analyse_comb.py` (`docs/plans/comb-dns-instructions.md`); Claude writes up B.5/B.7/B.8 from DNS |
| 51592781 | Re050 closure extraction | build its cache v2: `build_cache_v2.py --cases Re050_Ca0_010_theta45` |
| 51592842 | cache v2 build, 28 cases | **FAILED**: HDF5 lock on a closure file still being written by an extraction. Replaced by 51595693 |
| 51595693 | `pipeline/cache_v2/swap_and_build.sh`, which waits (`afterok`) for extractions 51592928–33 and 35–38 (51592781 and 51592934 had already completed), checks and swaps the four `.new` files (old → `.stale_20261116`), then builds cache v2 for all closure files | read the `cache_v2_full_<id>.out` tail ("swapped …" lines), `check.csv`, `manifest.json`. If it is stuck with DependencyNeverSatisfied, an extraction failed: inspect it, then run the script by hand |
| 51592928–34 | closure extraction: Re10 Ca0.005, Re20 Ca0.002 β90, Re20 Ca0.005 β6.4, Re40 Ca0.002, Re40 Ca0.1 β90, and the partial 3 Nov runs Re10 Ca0.001 β90 and Re20 Ca0.001 β6.4 | add to cache v2 (split `extra`) |
| 51592935–38 | **full redo** of Re10 Ca0.01, Re10 Ca0.1, Re40 Ca0.1 β45, Re10 Ca0.04 β90 into `…h5.new` | **after 51592842**: rename old → `.stale_20261116`, `.new` → `.h5`; rebuild these in cache v2 |

## Pending actions, in order

1. When job 51595693 finishes: confirm the four swaps, report `check.csv`, and refresh `datasets/MANIFEST_dns.csv`. The comb DNS runs wait for burst nodes (Krishna's decision: stay on `bala1s-b`).
2. Write the comb DNS results into `orr-sommerfeld.md` (B.5 multi-frequency, B.7 per-frequency check, B.8 from DNS) and compare with the OS targets.
3. Phase C on cache v2 (σ = 0.5), V1 first, L2 constraint with g₃. Exclude Re10 Ca0.001 β90, Re20 Ca0.001 β6.4, Re40 Ca0.1 β90 (decisions.md, 17 Nov).
4. Later: re-extract the two 3 Nov partial cases when their runs finish.

## Facts to remember

- **Scaling.** T_N = h_N/u_N, from (Re, β, ρ, μ) with g = 9.81: 2.1389 ms for the oracle case. ω = 2π f T_N.
- **Kalliadasis et al. book** (`docs/plans/Book - Kalliadasis Ruyer-Quil - Falling Liquid Films.pdf`, committed on request): its velocity scale is 3u_N, so ω, c and V are 3× ours and We_b = 1/(3Ca). No PDF tools on HiPerGator; text was extracted with a stdlib zlib script in the scratchpad.
- **ROM inlet.** The Chang inlet with `MNOISE = M` is an exact comb at k·f_cut/M; the same holds for the DNS (`filmflow0.c`). Use it for linear tests.
- **Basilisk** ends a run once no time-scheduled event is left. Keep T_OUT and T_DUMP ≤ T_END in tests.
- **MLP spurious equilibria.** Both current MLPs have spurious uniform equilibria at h ≈ 2–3. A hypothesis, not yet tested, is that these cause the sum model's crest over-amplification.
- **A.3 interpretation.** At long wavelength, a ∂ₓq′ term in the q-equation changes *growth* and an in-phase term changes *speed*. The plan's A.3 wording has this the other way round.

## 17 Nov facts (structured closure)

- **Gaussian second derivative is not zero on constants** (scipy truncate=4 and the ROM copy): h_xx(1) = −3.5e-3 (σ 0.5, DNS grid), −1.8e-3 (ROM grid), −1.65e-2 (σ 0.25, DNS). Cache-v2 h_xx/q_xx carry it. Structured closure: ROM uses zero-sum `gauss_deriv_hN`, training subtracts c0·h (`data.c0_second`).
- **GPU is 12× faster than 8 CPU cores** for training (1.1 vs 14 s/epoch); the user relaxed the 2-GPU limit for this session only.
- **L0 (no linear constraint) gives growth errors up to 7–10× the OS peak**; L2 (λ_lin 10) cuts them ~10× at no held-out cost. λ_F anchor does not help.
- Running analysis that imports torch then scipy needs `LD_LIBRARY_PATH=$THINFILM_ENV/lib` (system libstdc++ clash).
- **Oracle a posteriori diagnosis (17 Nov).** A priori all closures fit S to 11–12.5% on the oracle case, but the closure's net power into the waves, <(−S) q'>, is a ~0.1% residual of large front (−0.32) and back (+0.19) contributions; every learned closure gives net input (+1.0e-3 to +2.2e-3) where the DNS gives −3.4e-4. Closed-loop split: flux part alone amplifies crests (max h 2.6–3.0), relaxation alone damps like Lavalle.
- **Power penalty (λ_P 10, 100; `train.py --lam_P`) fixes the a priori net power but not the a posteriori error**: a priori NRMSE rises to 0.16, and the waves drift in phase (−7 units by t = 50 for σ 0.25 λ_P 100; +7.8 for σ 0.5 without penalty), while ml_sum_hhx stays in phase (corr 0.95). Best structured so far: σ 0.25, λ_lin 10, no penalty (in phase, h error 0.60 vs ml_sum 0.47 at t = 50, max h 2.50 vs 2.38).
- **Open:** seed spread of these a posteriori numbers is unknown (seed 1 = backup); broadband held-out a posteriori (gate GD) not yet run; a richer form for wave speed: "V1.5" (κ, g₁–g₃ depend on (h, r = q/h³); flux part unchanged, so still guaranteed hyperbolic) or V2 (also Φ(h, r), hyperbolicity by penalty + limiter). Either needs the ROM to evaluate networks of (h, r) instead of 1D tables (the MLP class in `ml_closure.cpp` can be reused).
- Explainer for Krishna: `docs/explainers/2026-11-17-structured-closure-explainer.md`.

## Next options (17 Nov, end of day). Priority: a posteriori accuracy over interpretability (user's decision)

All use the same training code (`ml/structured_closure/train.py`, L2 at the 85 OS points, λ_lin 10) and the same oracle test (`rom_tests.py`). The ROM change is shared: evaluate networks directly instead of 1D tables (reuse the `MLP` class of `rom/lavalle-cpp/src/ml/ml_closure.cpp`). Do 0 first.

0. **Seed spread:** seed 1 of σ 0.25 / σ 0.5 L10 (`SEED=1 sbatch --export=ALL --array=...` with `run_sweep.sh`), oracle runs. Needed before reading the ±7-unit phase differences.
1. **V1.5:** relaxation coefficients κ, g₁–g₃ as networks of (h, r = q/h³, h_x, q_x; p); flux part (Φ(h), ΔP(h)) unchanged, so still guaranteed hyperbolic. The flat film stays exact; the linear coefficients need only the network values at (1, 1, 0, 0). Risk: gradient inputs act as nonlinear gradient terms (possibly anti-diffusive at fronts).
2. **Flat-film-exact plain MLP + L2:** S = N(h, q, h_x, q_x; p) − N(h, h³, 0, 0; p), a source term like the old `ml_sum_hhx` with the Phase A defects removed (S₀ = e = 0 exactly; OS-matched linear growth). No hyperbolicity or structure guarantees (the ROM flux stays Lavalle's). Easiest ROM path (MLClosure hook plus the subtraction). Baseline for "does the structure help in closed loop?".
3. **V2:** Φ(h, r) in the flux as well; hyperbolicity by penalty plus runtime limiter (plan C.8).
4. **Then:** broadband held-out a posteriori (Phase D, gate GD) for the best of these; optionally add the 11 extra cases to training.

## Meeting material (17 Nov)

- Advisor deck (5 slides, not in git): `docs/meetings/2026-11-17_advisor_structured-closure.pptx`; slide explainer `docs/meetings/2026-11-17_advisor_deck-explainer.md`; rebuild with `scripts/meetings/advisor_2026_11_17_{figs,deck}.py`. Slide 5 is to be updated with the flat-film-exact MLP oracle runs (`figs.py --extra NAME=<run>/oracle_summary.csv`).
- **Flat-film-exact MLP (option 2):** `train.py --model mlp` (`model.FlatExactMLP`, `export_mlp.py`; ROM `FlatMLPClosure`, config key `flat_mlp`). Runs `mlp_s0.25_L{0,1,10}_seed0`, `mlp_s0.5_L10_seed0` (lr 1e-2). The OS constraint costs this unstructured network much more data fit than V1 (smoke: loss 0.10 at L0 vs 0.17–0.59 at L10). ROM runs are ~15× slower than V1 (network evaluated twice per cell per stage): oracle ~1 h, comb Re200 ~2.5 h; `run_rom_tests.sh` now has a 4 h limit. Jobs: 51798464 (uniform + oracle), 51800485 (comb + `oracle_b` copies).
  - **Oracle results (σ 0.25, L0/L1/L10):** see `docs/results/structured-closure.md`, "Flat-film-exact MLP". Right crest heights and power sign; waves 3–6% slow; shape error ~0.3 above exact common to all learned closures. Pending: `mlp_s0.5_L10` oracle (task 19 or `oracle_b`), all comb runs (51800485), then `rom_tests.py analyse` and slide 5.
  - **Next check (cheap):** linear phase speed of each model against OS at the forcing frequency ω ≈ 0.134, to see whether the wave-speed error is linear.
  - **18 Nov, all seed-0 runs analysed** (`rom_tests.py analyse`): σ 0.5 L10 is the best learned closure on the oracle (h error 0.39 at t = 50, 0.62 at t = 100; ml_sum 0.47/1.26, exact 0.19/0.55); comb growth error vs OS 7–11% median for L1/L10 at Re 120–200 (Lavalle 27–39%); L0 fails at Re 200 (47%, waves damped), so the OS constraint matters.
  - **Priority (18 Nov, Krishna):** draft mid-December; maximise ML accuracy; Phase E (closed forms, PySR) de-emphasised; slides on hold.
  - **ROM speed-up (18 Nov):** `FlatMLPClosure` tabulates N(h, h³, 0, 0, 0; p) once per run (h 0.02–8, dh 1e-4, h = 1 a node; `n0_table = 0` disables) and runs the cell loop with OpenMP (`-fopenmp`, `OMP_NUM_THREADS`). Oracle check, 400 steps: max |Δh| 1.5e-9 against the old binary, threads bitwise identical; 0.164 → 0.085 s/step (table) → 0.024 s/step (4 threads).
  - **Seeds:** `SWEEP=mlp SEED=1` tasks 2, 3 (σ 0.25/0.5, L10): job 51936231. Chain job 51936523 (`chain_seed_rom.sh`, afterok) then submits their ROM tests and broadband runs (`rom/broadband/runs_seed1.txt`).
  - **Broadband held-out a posteriori (gate GD):** `broadband.py` (make/analyse) + `run_broadband.sh` (bala1s-b, 4 threads). ROM from a flat film with the DNS Chang inlet parameters (F0 0.025, M 1000, 28 Hz), L 800, t 0–2000, output every 10; statistics over t ≥ 1000, 20 < x < 760 against the DNS closure files (h every 10): h std/skewness, main-crest height (mean, p90), spectral-centroid wavelength, envelope error, PDF distance (W1/std). 7 held-out cases × {lavalle, ml_sum_hhx, v1_s0.25_L10_F0, mlp_s0.25_L10, mlp_s0.5_L10} seed 0: job 51936424. Analyse: `broadband.py analyse --closure lavalle ml_sum_hhx v1_s0.25_L10_F0_seed0 mlp_s0.25_L10_seed0 mlp_s0.5_L10_seed0 --csv ...`.
  - **Not done:** oracle runs with `reinit_every = 10` (not implemented in lavalle-cpp).
