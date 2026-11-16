# Comb-forced DNS (D3): how to check the runs and run the analysis

_16 Nov 2026. These are the five short comb-forced DNS runs (decision D3 of `structured-closure-os-plan.md`) and their analysis. Background: `docs/results/orr-sommerfeld.md` (B.5–B.8)._

## What is running

Five SLURM jobs on `bala1s-b`, 16 CPUs each, about 1–2 h each:

| Job | Case | Comb modes | FCUT_DIM | ω range | OS ω_c |
|---|---|---|---|---|---|
| 51592372 | Re020_Ca0_010_theta45 | 21 | 62.7 Hz | 0.063–1.32 | 1.41 |
| 51592373 | Re040_Ca0_040_theta45 | 31 | 93.6 Hz | 0.063–1.95 | 2.05 |
| 51592374 | Re080_Ca0_010_theta45 | 28 | 114.4 Hz | 0.063–1.76 | 1.84 |
| 51592375 | Re120_Ca0_010_theta45 | 29 | 135.6 Hz | 0.063–1.82 | 1.92 |
| 51592376 | Re200_Ca0_010_theta45 | 31 | 284.5 Hz | 0.063–1.95 | 2.04 |

- **Setup:** production physics (Re, Ca, β, fluid, CFL, seed) on a short domain with the same mesh resolution: LX = 200, AR = 32, MAXLEVEL 12, so dx = 0.0488 and the height is 6.25.
- **Inlet:** a comb of M modes, 3e-4 each, with a period of exactly 100 time units. It covers the OS unstable band up to 0.97 ω_c.
- **Each job:** `spinup/` runs `filmflow0` from a flat film to t = 150 (dump-000150). Then `run_t150/` runs `filmflow_restart_profiles` from that dump for t = 150–255, writing 1D profiles every 0.25 (420 files).
- **Files:**
  - setup: `dns/comb/setup_comb_runs.py`;
  - job script: `dns/comb/run_comb.sh`;
  - data: `$THINFILM_STORE/datasets/BasC/DNS/comb/<case>/` (`COMB_INFO.txt`, `comb_<jobid>.out`, `spinup/`, `run_t150/`).

## 1. Check that the jobs finished

```bash
source $THINFILM_ROOT/env.sh          # or: cd <repo>; source env.sh
sacct -j 51592372,51592373,51592374,51592375,51592376 --format=JobID,JobName,State,Elapsed,ExitCode -X
for c in $THINFILM_STORE/datasets/BasC/DNS/comb/Re*/; do
  echo "$(basename $c): $(tail -n 1 $c/comb_*.out)"          # expect: done <date>; 420 profiles
done
```

- `COMPLETED` with "420 profiles": go to step 2.
- `FAILED` or `TIMEOUT`: look at the end of the case's `comb_<jobid>.out` and of `spinup/log.err` or `run_t150/log.err` (`tail -n 20`).
  - If the spin-up finished (`spinup/dump-000150` exists) but the restart failed, rerun only the restart:
    ```bash
    cd $THINFILM_STORE/datasets/BasC/DNS/comb/<case>/run_t150
    rm -rf profiles; ln -sf ../spinup/dump-000150 restart.dump
    sbatch --job-name=comb_restart --account=bala1s --qos=bala1s-b --cpus-per-task=16 --mem=32gb -t 06:00:00 \
      --wrap "source $THINFILM_ROOT/env.sh; module purge; module load gcc/14.2.0 imagemagick/7.0.8-20 hdf5; \
              export OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=cores KMP_STACKSIZE=1g LD_LIBRARY_PATH=\$HPC_HDF5_LIB:\$LD_LIBRARY_PATH; \
              cp $THINFILM_STORE/datasets/BasC/DNS/comb/build/filmflow_restart_profiles .; ./filmflow_restart_profiles params.in 2> log.err"
    ```

## 2. Run the analysis

The analysis takes a few minutes for all five cases (no DNS data is written). Run it in a short interactive or `hpg-dev` job rather than on the login node:

```bash
cd $THINFILM_ROOT && source env.sh
sbatch --account=bala1s --qos=bala1s --partition=hpg-dev --cpus-per-task=2 --mem=16gb -t 00:30:00 \
  -o $THINFILM_STORE/analysis/orr_sommerfeld/logs/comb_analysis_%j.out \
  --wrap "source $THINFILM_ROOT/env.sh; python -I $THINFILM_ROOT/analysis/orr_sommerfeld/comb_dns/analyse_comb.py"
```

Options:
- `--cases Re120_Ca0_010_theta45 ...` to analyse a subset;
- `--xa 60 --xb 150` to change the x-window of the growth fits. B.5 found an inlet adjustment zone for x < 60.

## 3. What to look at

- **Printed summary:** the end of `comb_analysis_<jobid>.out` (`tail -n 120`). One block per case:
  - `settling`: the mean h over the window and its drift between the two halves of the window. Both should be small (≲ 1e-4). If not, the start-up transient has not left x < 150; rerun with a later window or a longer spin-up.
  - `reliable modes`: modes with R² > 0.95 of ln|A| over the window and a wavelength below twice the window. The lowest frequencies (wavelength ≳ 180) are excluded by design.
  - `max growth error`: DNS against OS, Lavalle against OS, and Lavalle against DNS (relative to the maximum OS growth). Also the phase-speed error of the DNS against OS.
  - `fit (a,b,c,d) to DNS / OS`: the B.8 linear-closure coefficients fitted to the DNS k(ω) and to OS at the same frequencies, with the fit errors.
  - A table of every second or third mode: growth, phase speed, x where growth stops being linear (`x_nl`), and the complex ΔM_x/η and ΔT_w/η from the DNS next to the OS targets.
- **Files** in `$THINFILM_STORE/analysis/orr_sommerfeld/comb_dns/`:
  - `comb_fits.csv`: one row per case, all summary numbers;
  - `comb_modes_<case>.csv`: every mode;
  - `comb_growth_<case>.png`: growth and phase speed against ω (OS line, Lavalle dashed, DNS circles = reliable, crosses = excluded).

## 4. What to send back to Claude

The `tail -n 120` of `comb_analysis_<jobid>.out`, or just say that it has run. Claude reads `comb_fits.csv` and writes the results into `docs/results/orr-sommerfeld.md` (B.5 extension, B.7 check, B.8 DNS fit).

## Notes

- The comb frequencies are k·Δω with Δω = 2π/100, and the projection uses exactly one period (t = 155–255), so the modes are separated exactly.
- Nonlinear products of comb modes also fall on comb frequencies. At 3e-4 per mode and x < 150, they are below about 2% of each mode (B.5: linear up to |A| ≈ 0.27).
- Each comb mode starts at 3e-4. Modes near ω_c grow little and stay at that level, which is well above the 1e-7 reliability floor.
- Testing options (`--base`, `--t0`, `--tc`) exist for the smoke test on `comb/test/`. Do not use them for the real analysis.
