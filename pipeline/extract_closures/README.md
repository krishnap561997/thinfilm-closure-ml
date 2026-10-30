# pipeline/extract_closures

DNS → depth-integrated quantities and Lavalle discrepancy closures (ΔC = C_DNS − C_Lav). Definitions are in `docs/equations.md` §5.

| Script | What it does |
|---|---|
| `extract_closure_terms.py <config.json>` | Reads 2D Basilisk snapshots (t and t−dt in each file) and writes a closure file: fields of shape (nx, nt), `t`, `tm`, `dt`, `parameters/`. Uses Savitzky–Golay derivatives and repairs the interface velocity. |
| `generate_extraction_jobs.py` | Writes one config and SLURM job per case into `ClosureData/configs/` and `run_extraction/`. |
| `profiles_to_closures.py <profiles/> <out.h5>` | Same output from the in-situ 1D column profiles written by `dns/include/column_profiles.h`, reusing the functions above. On the same state it matches the 2D route to round-off. |
| `check_momentum_budget.py <closure.h5>` | Budget check. (A) On the DNS grid: do mass and momentum close with the extracted terms? (B) On the ROM grid: does S_ML close a Python replica of the lavalle-cpp discrete q-RHS? Options: `--times 1000:2000:100`, `--smooth 0,1` (Gaussian width in h_N), `--csv` per time, `--summary-csv` pooled. |
| `run_budget_check.sh` | SLURM wrapper that runs the budget check on every closure file into `ClosureData/budget_check/`. Submit with `sbatch --output=$THINFILM_DATA/BasC/DNS/param_space/ClosureData/budget_check/logs/budget_%j.out run_budget_check.sh`. |

**Budget-check metrics** (in `summary.csv` and `per_case/*.csv`, as ratios of rms over 20 < x < 760):

- `mass_rel` = |∂ₜh + q_x| / |∂ₜh|. No closure enters, so a large value flags an unreliable one-step ∂ₜ at that snapshot.
- `dns_res_rel` = momentum residual on the DNS grid / |∂ₜq|.
- `S_rel` = |S_ML| / |∂ₜq|, the size of the correction.
- `rom_E0_rel_w*` = |∂ₜq − RHS_ROM| / |∂ₜq|, the Lavalle ROM misfit. `rom_E1_rel_w*` is the misfit left after adding S_ML, and `rom_E1_over_E0_w*` is the fraction of the misfit S_ML fails to explain. `w0` is unsmoothed and `w1` uses a 1 h_N Gaussian.
- `rom_corr_w*` = corr(E0, S_ML).
