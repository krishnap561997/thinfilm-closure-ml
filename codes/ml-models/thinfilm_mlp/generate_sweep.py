"""Freeze a Cartesian-product experiment list and create a SLURM array launcher."""
import argparse
import itertools
import json
import re
import shlex
from pathlib import Path
import numpy as np


def generate(config, output):
    cfg = json.loads(Path(config).read_text())
    cache = Path(cfg['cache']).expanduser().resolve()
    manifest = json.loads((cache/'manifest.json').read_text())
    input_fields = cfg.get('input_fields', ['h', 'q'])
    if input_fields not in (['h', 'q'], ['h', 'q', 'curvature']):
        raise ValueError('Unsupported input_fields')
    for entry in manifest['cases']:
        for field in input_fields:
            if not (cache / entry['name'] / f'{field}.npy').is_file():
                raise FileNotFoundError(f'Missing {field} in {entry["name"]}; augment cache first')
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Choose a new sweep directory; existing sweeps are immutable')
    js, strides = cfg['half_windows'], cfg['strides']
    if not js or not strides or not cfg['target_groups']:
        raise ValueError('Window, stride and target lists must be nonempty')
    if any(type(j) is not int or j < 0 for j in js) or any(type(s) is not int or s < 1 for s in strides):
        raise ValueError('Nonnegative integer windows and positive integer strides required')
    if cfg['max_concurrent'] < 1:
        raise ValueError('max_concurrent must be positive')
    for name, targets in cfg['target_groups'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Target group names must use letters, digits, hyphens or underscores')
        if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(manifest['config']['available_targets']):
            raise ValueError(f'Invalid targets for {name}')
    allowed = {'hidden_width','hidden_layers','batch_size','lr','weight_decay','workers','seed'}
    if set(cfg['training']) - allowed:
        raise ValueError(f'Unknown training options: {set(cfg["training"]) - allowed}')
    grids = [np.load(cache/e['name']/'x.npy', mmap_mode='r') for e in manifest['cases']]
    radius = manifest['config']['max_half_width']
    runs, skipped = [], []
    for j, s in itertools.product(sorted(set(js)), sorted(set(strides))):
        # No neighbor spacing exists at j=0; keep only the smallest requested stride.
        if j == 0 and s != min(strides):
            skipped.append(dict(half_window=j, stride=s, reason='duplicate local-only baseline'))
            continue
        span = j*s
        if any(span >= len(x) or (span and np.max(x[span:]-x[:-span]) > radius+1e-10) for x in grids):
            skipped.append(dict(half_window=j, stride=s, reason='exceeds maximum radius in at least one case'))
            continue
        for group, targets in cfg['target_groups'].items():
            name = f'{group}_j{j}_s{s}'
            runs.append(dict(task_id=len(runs), name=name, group=group, half_window=j, stride=s,
                             targets=targets, run_dir=str(out/name)))
    if not runs:
        raise ValueError('No valid combinations')
    out.mkdir(parents=True)
    (out/'logs').mkdir()
    payload = dict(cache=str(cache), cache_id=manifest['id'], epochs=100, no_early_stopping=True,
                   input_fields=input_fields, training=cfg['training'], runs=runs, skipped=skipped)
    (out/'sweep.json').write_text(json.dumps(payload, indent=2))
    (out/'requested_config.json').write_text(json.dumps(cfg, indent=2))
    worker = Path(__file__).resolve().with_name('run_sweep_task.py')
    # Quote command arguments and SBATCH output path for directories with spaces.
    script = f'''#!/bin/bash
#SBATCH --job-name=thinfilm_sweep
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-{len(runs)-1}%{int(cfg['max_concurrent'])}
#SBATCH --output="{out}/logs/%A_%a.log"
set -euo pipefail
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
python -u {shlex.quote(str(worker))} --sweep {shlex.quote(str(out/'sweep.json'))} --task-id "$SLURM_ARRAY_TASK_ID"
'''
    (out/'submit_array.sh').write_text(script)
    for run in runs:
        print(f'{run["task_id"]:3d}  {run["name"]}')
    for item in skipped:
        print(f'Skipped j={item["half_window"]}, s={item["stride"]}: {item["reason"]}')
    print(f'{len(runs)} jobs. Submit: sbatch {shlex.quote(str(out/"submit_array.sh"))}')
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='sweep_config.json')
    p.add_argument('--output', default='runs/sweep_01')
    a = p.parse_args()
    generate(a.config, a.output)
