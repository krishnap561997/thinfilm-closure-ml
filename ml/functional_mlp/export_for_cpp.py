#!/usr/bin/env python
"""Export a functional_mlp checkpoint (local input mode, tanh MLP) to the plain-text
format read by rom/lavalle-cpp/src/ml/ml_closure.cpp.

Usage: export_for_cpp.py <best.pt> <out.txt>

Format (whitespace separated): 'inputs' n name... ; 'x_mean' ; 'x_std' ; 'y_mean' ; 'y_std' ;
'layers' L ; for each layer 'W' rows cols (row-major) then 'b' n. Tanh between layers.
Input order is the training order: the listed fields, then Re, Ca, angle (radians).
"""
import sys
import torch

ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert ck['input_mode'] == 'local', 'only local-input models are supported'
sd = ck['model']
names = list(ck['args']['input_fields']) + ['Re', 'Ca', 'angle']
assert len(names) == len(sd['x_mean'])
ws = sorted({int(k.split('.')[2]) for k in sd if k.startswith('mlp.network.')})
with open(sys.argv[2], 'w') as f:
    f.write(f"targets {' '.join(ck['args']['targets'])}\n")
    f.write(f"inputs {len(names)} {' '.join(names)}\n")
    for k in ('x_mean', 'x_std', 'y_mean', 'y_std'):
        f.write(k + ' ' + ' '.join(f'{v:.9g}' for v in sd[k].double().tolist()) + '\n')
    f.write(f'layers {len(ws)}\n')
    for i in ws:
        W, b = sd[f'mlp.network.{i}.weight'].double(), sd[f'mlp.network.{i}.bias'].double()
        f.write(f'W {W.shape[0]} {W.shape[1]}\n' + ' '.join(f'{v:.9g}' for v in W.flatten().tolist()) + '\n')
        f.write(f'b {b.numel()}\n' + ' '.join(f'{v:.9g}' for v in b.tolist()) + '\n')
print('wrote', sys.argv[2], names)
