# Curvature included in every new cache

Extract these files into your existing thinfilm_mlp code directory, replacing the Python files included. Your data/sweep JSON configurations and SLURM scripts are not included and will remain intact. Preserve any custom modifications to the included Python files before replacing them.

prepare_data.py now always calculates curvature = h_xx / (1 + h_x**2)**1.5 from the original unnormalized DNS derivatives and stores time-major float32 curvature.npy for each case. h_x and h_xx must exist with shape (nx, nt). Computation uses float64 in time blocks before storage. Missing derivatives or wrong shapes stop preparation. No separate augmentation or second cache is needed.

In data_config.json, set data_dir to your corrected simulation directory, check development_cases and heldout_cases, and set cache_dir to a NEW nonexistent directory. Keep the same case memberships for reruns of the same parameter cases. Keep all desired outputs in available_targets (including int_px_dy if used). Do not add curvature to available_targets: it is automatically cached as an input.

Run your existing preparation job:

    sbatch run_prepare.sh

The bundled original run_prepare.sh reads data_config.json by default (or the CONFIG environment variable). Wait for successful completion and manifest.json before generating sweeps.

In your sweep_config.json, set cache to the absolute path of that new cache and add this top-level field:

    "input_fields": ["h", "q", "curvature"]

Then run:

    python generate_sweep.py --config sweep_config.json --output runs/sweep_curvature_01
    sbatch runs/sweep_curvature_01/submit_array.sh

Use a new sweep output directory. Review the generated SLURM account/QoS directives as in your existing workflow; this update does not know your account/QoS values. The generator uses the hpg-turin partition and an L4 GPU. Existing custom changes to generate_sweep.py should be merged if necessary.

For a baseline sweep, use the SAME cache path and input_fields ["h", "q"], with a separate sweep directory. Omitting input_fields defaults to h and q. All models use the same selected centers. Alongside the existing h/q validity rules, centers are excluded if curvature is nonfinite anywhere within the common maximum-radius neighborhood. This check applies to all models, including h/q-only models. Flat-film masking remains center-only and controlled by your existing configuration.

With curvature enabled, each neighbor contributes [h, q, curvature, signed_dx], followed by global Re, Ca, angle. Input dimension is 4*(2*j+1)+3. Curvature normalization is fitted only on training data. The hidden architecture, 100-epoch sweep budget, and configured split policy are unchanged. Curvature uses the DNS derivative fields, not differences across the selected sparse stride.

Existing trained checkpoints still require their original cache; train new models on the corrected cache. The separate augment_curvature.py workflow is optional for legacy caches and is unnecessary here.

Validation: synthetic end-to-end tests cover the analytic curvature formula, common finite-data eligibility, both input choices, training/evaluation, sweep propagation, and the original pipeline. Production DNS and cluster GPU jobs were not run here.
