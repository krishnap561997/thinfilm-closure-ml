# Progress tracker: structured-closure / Orr–Sommerfeld plan

_Context for Claude sessions. Updated 16 Nov 2026 (end of day 1). Plan: `structured-closure-os-plan.md`. Branch: `structured-closure` (from `main` @ 719b7da, not pushed). Results: `docs/results/orr-sommerfeld.md`, `docs/results/structured-closure.md`. Explainer for Krishna: `docs/explainers/2026-11-16-structured-closure-explainer.md`._

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
| Phase D closed loop | first oracle run 17 Nov: structured ≈ ml_sum to t = 5, then crest over-amplification (max h 2.3–3.0 vs DNS 2.1), h error 0.6–1.2 at t = 50 (ml_sum 0.47, exact closure 0.19). Diagnosis next | `structured_closure/rom/<run>/oracle_summary.csv` |
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
