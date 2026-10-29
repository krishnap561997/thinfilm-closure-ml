"""Freeze local-input ablations grouped by network size and create a SLURM array."""
import argparse
import itertools
import json
import re
import shlex
from pathlib import Path

from datasets import LocalDataset


def generate(config, output):
    """Validate the study and enumerate architecture/input/target/seed combinations."""
    cfg = json.loads(Path(config).read_text())
    cache = Path(cfg['cache']).expanduser().resolve()
    manifest = json.loads((cache / 'manifest.json').read_text())
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Choose a new sweep directory; existing sweeps are immutable')

    # Architecture and seed order are preserved, grouping task IDs by network size.
    networks, groups, targets, seeds = (
        cfg['networks'], cfg['input_groups'], cfg['targets'], cfg['seeds']
    )
    if not networks or not groups or not targets or not seeds:
        raise ValueError('Networks, input groups, targets and seeds must be nonempty')
    if type(cfg['max_concurrent']) is not int or cfg['max_concurrent'] < 1:
        raise ValueError('max_concurrent must be a positive integer')
    epochs = cfg.get('epochs', 100)
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if type(cfg.get('no_early_stopping', True)) is not bool:
        raise ValueError('no_early_stopping must be boolean')
    if any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    if (len(set(targets)) != len(targets)
            or not set(targets) <= {'delta_Mx_ML', 'delta_Tw_ML'}
            or not set(targets) <= set(manifest['config']['available_targets'])):
        raise ValueError('Targets must be unique available correction names')

    sizes = []
    for net in networks:
        if set(net) != {'hidden_layers', 'hidden_width'}:
            raise ValueError('Each network requires hidden_layers and hidden_width')
        if any(type(v) is not int or v < 1 for v in net.values()):
            raise ValueError('Network dimensions must be positive integers')
        sizes.append((net['hidden_layers'], net['hidden_width']))
    if len(set(sizes)) != len(sizes):
        raise ValueError('Duplicate network sizes')

    # Network sizes and seeds belong to each run, never the shared options.
    allowed = {'batch_size', 'lr', 'weight_decay', 'workers', 'patience'}
    training = cfg['training']
    if set(training) - allowed:
        raise ValueError(f'Unknown training options: {set(training) - allowed}')
    for key in ['batch_size', 'patience', 'workers']:
        if key in training and (type(training[key]) is not int
                                or training[key] < (0 if key == 'workers' else 1)):
            raise ValueError(f'Invalid {key}')
    for key in ['lr', 'weight_decay']:
        if key in training:
            value = training[key]
            if (type(value) not in (int, float) or not 0 <= value < float('inf')
                    or (key == 'lr' and value == 0)):
                raise ValueError(f'Invalid {key}')

    seen = set()
    for name, fields in groups.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Input group names must use letters, digits, hyphens or underscores')
        if not isinstance(fields, list) or not fields or any(not isinstance(f, str) for f in fields):
            raise ValueError(f'Invalid input fields for {name}')
        signature = frozenset(fields)
        if signature in seen:
            raise ValueError(f'Duplicate input subset: {name}')
        seen.add(signature)
        # Validate headers and derivative/index alignment without accessing test
        # or held-out data. All experiments reuse the same frozen sample rows.
        for split in ['train', 'val']:
            LocalDataset(cache, split, targets, fields)

    runs = []
    for net in networks:
        architecture = f'layers{net["hidden_layers"]}_width{net["hidden_width"]}'
        for (group, fields), target, seed in itertools.product(groups.items(), targets, seeds):
            name = f'{architecture}/{group}/{target}/seed{seed}'
            runs.append(dict(
                task_id=len(runs), name=name, architecture_group=architecture,
                input_group=group, input_fields=fields, targets=[target],
                hidden_layers=net['hidden_layers'], hidden_width=net['hidden_width'],
                seed=seed, run_dir=str(out / name),
            ))

    # Freeze the resolved experiment list; workers consume this file on resume.
    out.mkdir(parents=True)
    payload = dict(
        schema=2, input_mode='local', cache=str(cache), cache_id=manifest['id'],
        epochs=epochs, no_early_stopping=cfg.get('no_early_stopping', True),
        training=training, runs=runs,
    )
    (out / 'sweep.json').write_text(json.dumps(payload, indent=2) + '\n')
    (out / 'requested_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    write_launchers(payload, out, cfg['max_concurrent'])
    for run in runs:
        print(f'{run["task_id"]:3d}  {run["name"]}')
    print(f'{len(runs)} experiments. Submit one network folder at a time.')
    return payload


def write_launchers(payload, out, max_concurrent):
    """Write an independent manifest and SLURM launcher for each network size.

    Group-local task IDs start at zero; output run directories stay unchanged.
    The root manifest remains the complete inventory of all experiments.
    """
    worker = Path(__file__).resolve().with_name('run_sweep_task.py')
    architectures = list(dict.fromkeys(r['architecture_group'] for r in payload['runs']))
    for architecture in architectures:
        folder = out / architecture
        folder.mkdir(exist_ok=True)
        (folder / 'logs').mkdir(exist_ok=True)
        group_runs = [dict(r, task_id=i) for i, r in enumerate(
            r for r in payload['runs'] if r['architecture_group'] == architecture
        )]
        group_payload = dict(payload, runs=group_runs)
        (folder / 'sweep.json').write_text(json.dumps(group_payload, indent=2) + '\n')
        script = f'''#!/bin/bash
#SBATCH --job-name=thinfilm_{architecture}
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --array=0-{len(group_runs)-1}%{max_concurrent}
#SBATCH --output="{folder}/logs/%A_%a.log"
set -euo pipefail
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
python -u {shlex.quote(str(worker))} --sweep {shlex.quote(str(folder / 'sweep.json'))} --task-id "$SLURM_ARRAY_TASK_ID"
'''
        (folder / 'submit_array.sh').write_text(script)
        print(f'{architecture}: {len(group_runs)} jobs. Submit: sbatch {shlex.quote(str(folder / "submit_array.sh"))}')

    # Replace the old all-network launcher with instructions so it cannot submit
    # every architecture accidentally. Submit directly from a network folder.
    instructions = '#!/bin/bash\n# Submit one network size at a time.\n'
    for architecture in architectures:
        instructions += f'echo {shlex.quote("sbatch " + shlex.quote(str(out / architecture / "submit_array.sh")))}\n'
    instructions += 'exit 1\n'
    (out / 'submit_array.sh').write_text(instructions)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='sweep_config.json')
    p.add_argument('--output', default='runs/ablation_01')
    a = p.parse_args()
    generate(a.config, a.output)
