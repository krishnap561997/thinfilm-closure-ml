"""One line per structured-closure run: data error (train/held-out), linear growth error, long-wave D,
min discriminant.  Usage: python summarize_runs.py [PATTERN]  (default: v1_*)"""
import glob, json, os, sys
import numpy as np
B = os.path.join(os.environ["THINFILM_STORE"], "codes/ml-models/structured_closure/runs")
pat = sys.argv[1] if len(sys.argv) > 1 else "v1_*"
print(f"{'run':28s} {'NRMSE tr':>8s} {'NRMSE ho':>8s} {'ho max':>6s} {'ki tr med/max':>13s} {'ki ho med/max':>13s} {'|D/Dos-1| ho':>12s} {'min disc':>8s} {'s/ep':>5s} {'pow ho med|.|':>13s}")
for d in sorted(glob.glob(os.path.join(B, pat))):
    fn = os.path.join(d, "metrics.json")
    if not os.path.exists(fn):
        print(f"{os.path.basename(d):28s} (no metrics)"); continue
    M = json.load(open(fn)); tr, ho = M["train"], M["eval"]
    f = lambda D, k: np.array([v[k] for v in D.values()])
    print(f"{os.path.basename(d):28s} {f(tr,'nrmse').mean():8.3f} {f(ho,'nrmse').mean():8.3f} {f(ho,'nrmse').max():6.3f} "
          f"{np.median(f(tr,'lin_err_ki')):6.3f}/{f(tr,'lin_err_ki').max():.3f} {np.median(f(ho,'lin_err_ki')):6.3f}/{f(ho,'lin_err_ki').max():.3f} "
          f"{np.abs(f(ho,'D')/f(ho,'D_os')-1).max():12.3f} {min(f(tr,'min_disc').min(), f(ho,'min_disc').min()):8.1e} {M['s_per_epoch']:5.1f} {np.median(np.abs(f(ho,'pow_err'))) if 'pow_err' in next(iter(ho.values())) else float('nan'):13.4f}")
