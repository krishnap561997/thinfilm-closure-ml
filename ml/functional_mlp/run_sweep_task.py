"""Run/resume one fixed sweep experiment without changing its cached samples."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def command_for(sweep, task_id, device='cuda'):
    if not 0 <= task_id < len(sweep['runs']):
        raise ValueError('Task id outside sweep range')
    r = sweep['runs'][task_id]
    # Local sweep entries carry their own input subset, architecture and seed.
    command = [sys.executable, '-u', str(Path(__file__).resolve().with_name('train.py')),
        '--cache', sweep['cache'], '--run-dir', r['run_dir'],
        '--epochs', str(sweep['epochs']), '--device', device]
    if sweep.get('input_mode', 'window') == 'local':
        if len(r['targets']) != 1:
            raise ValueError('Each local sweep model must predict exactly one target')
        command.extend(['--input-mode', 'local', '--input-fields', *r['input_fields'],
                        '--hidden-layers', str(r['hidden_layers']),
                        '--hidden-width', str(r['hidden_width']), '--seed', str(r['seed'])])
    else:
        # Retain the ability to execute previously frozen window experiments.
        command.extend(['--input-mode', 'window',
                        '--half-window', str(r['half_window']), '--stride', str(r['stride']),
                        '--input-fields', *sweep.get('input_fields', ['h', 'q'])])
    command.extend(['--targets', *r['targets']])
    if sweep.get('no_early_stopping', True):
        command.append('--no-early-stopping')
    for key, value in sweep['training'].items():
        command.extend(['--'+key.replace('_','-'), str(value)])
    if (Path(r['run_dir'])/'last.pt').is_file():
        command.append('--resume')
    return command


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep', required=True)
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    sweep = json.loads(Path(a.sweep).read_text())
    manifest = json.loads((Path(sweep['cache'])/'manifest.json').read_text())
    if manifest['id'] != sweep['cache_id']:
        raise ValueError('Cache identity changed since sweep generation')
    command = command_for(sweep, a.task_id, a.device)
    print(shlex.join(command), flush=True)
    if not a.dry_run:
        subprocess.run(command, check=True)
