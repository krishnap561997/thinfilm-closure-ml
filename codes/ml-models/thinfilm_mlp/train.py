"""Train a window MLP using a shared cache; never access held-out cases."""
import argparse
import csv
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import WindowDataset
from model import PhysicalMLP


def atomic_save(payload, path):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def loader(ds, batch, workers, shuffle=False, generator=None):
    return DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=shuffle,
                      generator=generator, pin_memory=torch.cuda.is_available())


def fit_normalization(ds, batch, workers):
    # Shared statistics across all positions for h and q; separate global scalars.
    nf = len(ds.input_fields)
    n, sy, sy2, sx, sx2 = 0, np.zeros(len(ds.targets)), np.zeros(len(ds.targets)), np.zeros(nf + 3), np.zeros(nf + 3)
    for X, Y, _ in loader(ds, batch, workers):
        x, y = X.numpy().astype(float), Y.numpy().astype(float)
        win = x[:, :-3].reshape(-1, ds.width, ds.channels)
        sx[:nf] += win[:, :, :nf].sum(axis=(0, 1))
        sx2[:nf] += (win[:, :, :nf] ** 2).sum(axis=(0, 1))
        sx[nf:] += x[:, -3:].sum(axis=0)
        sx2[nf:] += (x[:, -3:] ** 2).sum(axis=0)
        sy += y.sum(axis=0)
        sy2 += (y ** 2).sum(axis=0)
        n += len(x)
    counts = np.array([n * ds.width] * nf + [n] * 3)
    means = sx / counts
    stds = np.sqrt(np.maximum(sx2 / counts - means ** 2, 0))
    stds = np.where(stds > 1e-12, stds, 1.)
    ym = sy / n
    ys = np.sqrt(np.maximum(sy2 / n - ym ** 2, 0))
    ys = np.where(ys > 1e-12, ys, 1.)
    scale = ds.manifest['config']['max_half_width'] or 1.
    return dict(x_mean=np.r_[np.tile(np.r_[means[:nf], 0.], ds.width), means[nf:]].tolist(),
                x_std=np.r_[np.tile(np.r_[stds[:nf], scale], ds.width), stds[nf:]].tolist(),
                y_mean=ym.tolist(), y_std=ys.tolist())


def validation(model, dl, weights, device, ncases):
    sums, counts = np.zeros(ncases), np.zeros(ncases)
    model.eval()
    with torch.no_grad():
        for x, y, ci in dl:
            x, y = x.to(device), y.to(device)
            error = ((model.standardized(x) - (y - model.y_mean) / model.y_std) ** 2 * weights).sum(1) / weights.sum()
            sums += np.bincount(ci.numpy(), weights=error.cpu().numpy(), minlength=ncases)
            counts += np.bincount(ci.numpy(), minlength=ncases)
    # Equal case weighting for model selection.
    return float(np.mean(sums / counts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default='cache/default')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--half-window', type=int, default=4)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--input-fields', nargs='+', default=['h', 'q'])
    p.add_argument('--targets', nargs='+', default=['delta_Dx_ML','delta_Mx_ML','delta_Px_ML','delta_Tw_ML'])
    p.add_argument('--target-weights', nargs='+', type=float)
    p.add_argument('--hidden-width', type=int, default=64)
    p.add_argument('--hidden-layers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-early-stopping', action='store_true', help='Always finish the requested epoch count')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.workers < 0 or a.patience < 1 or a.weight_decay < 0:
        p.error('Invalid training settings')
    if a.target_weights is None:
        a.target_weights = [1.] * len(a.targets)
    if len(a.target_weights) != len(a.targets) or not all(np.isfinite(w) and w > 0 for w in a.target_weights):
        p.error('Supply one positive finite weight per target')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else torch.device(a.device)
    train = WindowDataset(a.cache, 'train', a.targets, a.half_window, a.stride, a.input_fields)
    val = WindowDataset(a.cache, 'val', a.targets, a.half_window, a.stride, a.input_fields)
    run = Path(a.run_dir)
    if not a.resume and run.exists() and any(run.iterdir()):
        raise FileExistsError('Use a new run-dir or --resume to avoid overwriting a run')
    run.mkdir(parents=True, exist_ok=True)
    architecture = dict(input_dim=train.channels * train.width + 3, output_dim=len(a.targets),
                        hidden_width=a.hidden_width, hidden_layers=a.hidden_layers)
    checkpoint = None
    if a.resume:
        checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['cache_id'] != train.manifest['id']:
            raise ValueError('Resume cache differs')
        for k, v in vars(a).items():
            if k not in ('resume','epochs','workers','device','cache','run_dir') and checkpoint['args'].get(k, (['h', 'q'] if k == 'input_fields' else (False if k == 'no_early_stopping' else None))) != v:
                raise ValueError(f'Resume configuration differs: {k}')
        norm = checkpoint['normalization']
    else:
        print('Computing normalization from selected training samples only...', flush=True)
        norm = fit_normalization(train, a.batch_size, a.workers)
    model = PhysicalMLP(architecture, norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=8)
    generator = torch.Generator().manual_seed(a.seed)
    first, best, stale, history = 0, float('inf'), 0, []
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        first, best, stale, history = checkpoint['epoch'], checkpoint['best'], checkpoint['stale'], checkpoint['history']
        generator.set_state(checkpoint['generator_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    dl = loader(train, a.batch_size, a.workers, True, generator)
    vl = loader(val, a.batch_size, a.workers)
    weights = torch.tensor(a.target_weights, device=device)
    # Equal case contribution even when a small case has fewer than the sample cap.
    count = np.diff(np.r_[0, train.ends])
    case_weights = torch.tensor(len(train) / (len(count) * count), dtype=torch.float32, device=device)
    metadata = dict(args=vars(a), architecture=architecture, normalization=norm,
        cache_id=train.manifest['id'], input_order=f'[{",".join(a.input_fields)},signed_dx] for increasing offsets, then Re,Ca,angle_radians',
        window_by_case={e['name']: dict(spacing=e['dx'] * a.stride, half_width=e['dx'] * a.stride * a.half_window) for e in train.entries},
        versions=dict(torch=str(torch.__version__), numpy=np.__version__))
    (run / 'run_config.json').write_text(json.dumps(metadata, indent=2))
    print(f'{device}: {len(train):,} training samples; {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    for epoch in range(first, a.epochs):
        model.train()
        total = 0.
        for x, y, ci in dl:
            x, y, ci = x.to(device), y.to(device), ci.to(device)
            optimizer.zero_grad(set_to_none=True)
            error = ((model.standardized(x) - (y - model.y_mean) / model.y_std) ** 2 * weights).sum(1) / weights.sum()
            loss = (error * case_weights[ci]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        vl_loss = validation(model, vl, weights, device, len(val.entries))
        if not np.isfinite(vl_loss):
            raise FloatingPointError('Nonfinite validation loss')
        improved = vl_loss < best
        best, stale = (vl_loss, 0) if improved else (best, stale + 1)
        scheduler.step(vl_loss)
        history.append(dict(epoch=epoch+1, train_loss=total/len(train), val_loss=vl_loss, lr=optimizer.param_groups[0]['lr']))
        state = dict(metadata, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            epoch=epoch+1, best=best, stale=stale, history=history, generator_rng=generator.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        atomic_save(state, run / 'last.pt')
        if improved:
            atomic_save(state, run / 'best.pt')
        with (run / 'history.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        print(f'Epoch {epoch+1}: train={total/len(train):.6g}, val={vl_loss:.6g}, best={best:.6g}', flush=True)
        if not a.no_early_stopping and stale >= a.patience:
            print('Early stopping. best.pt retains the best validation epoch.')
            break


if __name__ == '__main__':
    main()
