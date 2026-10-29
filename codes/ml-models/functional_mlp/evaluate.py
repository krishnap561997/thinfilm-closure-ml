"""Evaluate a saved model, streaming full-split metrics and optional predictions."""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np
import torch
from datasets import LocalDataset, WindowDataset
from model import PhysicalMLP
from train import loader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default=str(Path(__file__).resolve().parent / 'cache_dir'))
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['train','val','test','heldout'], default='test')
    p.add_argument('--output', required=True, help='New output directory')
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else torch.device(a.device)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    args = ck['args']
    # New checkpoints record the input representation. Older checkpoints used windows.
    input_mode = ck.get('input_mode', 'window')
    if input_mode == 'local':
        ds = LocalDataset(a.cache, a.split, args['targets'], args['input_fields'])
        if ck['input_order'] != ds.input_order or ck['architecture']['input_dim'] != ds.input_dim:
            raise ValueError('Checkpoint local input layout differs')
    elif input_mode == 'window':
        ds = WindowDataset(a.cache, a.split, args['targets'], args['half_window'], args['stride'], args.get('input_fields', ['h', 'q']))
    else:
        raise ValueError(f'Unknown checkpoint input mode: {input_mode}')
    if ds.manifest['id'] != ck['cache_id']:
        raise ValueError('Evaluation cache does not match training cache')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    model = PhysicalMLP(ck['architecture'], ck['normalization']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    nc, ny = len(ds.entries), len(ds.targets)
    count = np.zeros(nc, np.int64)
    sums = np.zeros((nc, ny, 4), float)  # sum y, sum y², sum abs error, sum squared error
    handles = {}
    try:
        if a.save_predictions:
            for ci, entry in enumerate(ds.entries):
                f = h5py.File(out / f'{entry["name"]}.h5', 'w')
                handles[ci] = f
                n = int(ds.ends[ci] - ds.starts[ci])
                f.create_dataset('prediction', (n, ny), dtype='f4')
                f.create_dataset('truth', (n, ny), dtype='f4')
                f.create_dataset('indices', data=ds.maps(ci)[a.split + '_indices'])
                f.create_dataset('x', data=ds.maps(ci)['x'])
                f.create_dataset('t', data=np.load(ds.root / entry['name'] / 't.npy'))
                f.attrs['targets'] = ds.targets
                f.attrs['index_columns'] = ['time_index', 'space_index']
                f.attrs['checkpoint'] = str(Path(a.checkpoint).resolve())
                f.attrs['cache_id'] = ds.manifest['id']
        with torch.no_grad():
            for x, y, ci in loader(ds, a.batch_size, a.workers):
                pred = model(x.to(device)).cpu().numpy().astype(float)
                truth = y.numpy().astype(float)
                if not np.isfinite(pred).all():
                    raise FloatingPointError('Nonfinite predictions')
                for k in np.unique(ci.numpy()):
                    mask = ci.numpy() == k
                    yy, pp = truth[mask], pred[mask]
                    error = pp - yy
                    sums[k] += np.stack((yy.sum(0), (yy**2).sum(0), np.abs(error).sum(0), (error**2).sum(0)), axis=-1)
                    if a.save_predictions:
                        sl = slice(count[k], count[k] + len(yy))
                        handles[k]['prediction'][sl] = pp
                        handles[k]['truth'][sl] = yy
                    count[k] += len(yy)
    finally:
        for f in handles.values():
            f.close()
    rows = []
    for name, n, ss in [(e['name'], count[k], sums[k]) for k, e in enumerate(ds.entries)] + [('POOLED', count.sum(), sums.sum(0))]:
        for j, target in enumerate(ds.targets):
            sy, sy2, sae, sse = ss[j]
            sst = max(sy2 - sy * sy / n, 0.)
            rows.append(dict(case=name, target=target, n=int(n), RMSE=np.sqrt(sse/n), MAE=sae/n,
                        R2=1-sse/sst if sst > 1e-20 else float('nan'), NRMSE=np.sqrt(sse/sst) if sst > 1e-20 else float('nan')))
    for target in ds.targets:
        rr = [r for r in rows if r['target'] == target and r['case'] != 'POOLED']
        rows.append(dict(case='CASE_AVERAGE', target=target, n=int(count.sum()),
            **{key: float(np.mean([r[key] for r in rr])) for key in ['RMSE','MAE','R2','NRMSE']}))
    with (out / 'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'Saved {a.split} metrics to {out / "metrics.csv"}')


if __name__ == '__main__':
    main()
