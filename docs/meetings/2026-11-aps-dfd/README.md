# APS DFD 2026 Interact session material (19 Nov)

`story.md` is the talk and e-poster story (working draft). These figures and results are committed here on request (19 Nov), as an exception to the rule that PNGs stay out of git. The originals are in `$THINFILM_STORE/codes/ml-models/structured_closure/rom/`.

| File | What | Regenerate |
|---|---|---|
| `broadband_metrics.png` | held-out broadband cases: h-distribution distance and crest ratio per case, 5 closures, seed 0 | `scripts/structured-closure/broadband_fig.py metrics --csv broadband_seed0.csv` |
| `broadband_waves.png` | h(x) at t = 2000 and h PDFs, DNS / Lavalle / previous MLP / MLP σ 0.5 | `broadband_fig.py waves --closures lavalle ml_sum_hhx mlp_s0.5_L10_seed0` |
| `comb_growth_mlp.png` | linear growth in the ROM (comb inlet) against OS and the comb DNS | `scripts/structured-closure/comb_growth_fig.py` |
| `dense_error_mlp.png` | h error against time from the dense DNS restarts | `scripts/structured-closure/dense_error_fig.py --csv dense_summary.csv` |
| `broadband_seed0.csv` | `ml/structured_closure/broadband.py analyse` output (t 1000–2000) | |
| `dense_summary.csv` | `ml/structured_closure/dense_aposteriori.py analyse` output | |

The two crashed MLP runs are marked × in `broadband_metrics.png`. The safeguard reruns were pending at the time. The oracle figures (`oracle_snapshots_mlp.png`, `oracle_phase_mlp.png`, 17 Nov) predate the σ 0.5 model and are not included.
