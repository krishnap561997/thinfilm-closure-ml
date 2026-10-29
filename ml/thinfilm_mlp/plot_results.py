"""Plot a saved prediction file at one snapshot, preserving spatial positions."""
import argparse
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('predictions')
p.add_argument('--time-index', type=int, required=True)
p.add_argument('--output', default='prediction_profile.png')
a = p.parse_args()
with h5py.File(a.predictions) as f:
    index = f['indices'][:]
    take = np.flatnonzero(index[:, 0] == a.time_index)
    if not len(take):
        raise ValueError('No retained samples at that time index')
    ix = index[take, 1]
    x = f['x'][:][ix]
    truth, pred = f['truth'][take], f['prediction'][take]
    names = f.attrs['targets']
    time = f['t'][a.time_index]
order = np.argsort(x)
fig, axes = plt.subplots(len(names), 1, figsize=(10, 3 * len(names)), squeeze=False, constrained_layout=True)
for j, name in enumerate(names):
    ax = axes[j, 0]
    ax.plot(x[order], truth[order, j], label='DNS')
    ax.plot(x[order], pred[order, j], '--', label='MLP')
    ax.set(xlabel='x', ylabel=name, title=f't = {time:g}')
    ax.legend(); ax.grid(alpha=.3)
fig.savefig(a.output, dpi=160)
