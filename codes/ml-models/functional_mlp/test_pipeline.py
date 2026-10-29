"""Small synthetic end-to-end check; no production DNS data required."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from prepare_data import prepare
from datasets import WindowDataset

HERE = Path(__file__).resolve().parent


class PipelineTest(unittest.TestCase):
    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = json.loads((HERE / 'data_config.json').read_text())
            cfg.update(data_dir=str(root), cache_dir=str(root/'cache'), development_cases=['A','B'], heldout_cases=['C'])
            # Fix synthetic split expectations independently of production policy.
            cfg['fractions'] = dict(train=.65, val=.15, test=.20)
            cfg['sample_caps'] = dict(train=50, val=30, test=30, heldout=30)
            x, t = np.arange(0, 10, .0488), np.arange(20.)
            for name in ['A','B','C']:
                h = np.ones((len(x),len(t)))
                h[x < 6., :] += .2
                h[x > 9.5, :] = 3.  # Outlet artifact must not affect front detection.
                h[:,0] = 1.
                q = .5 + .1*h + .001*t[None,:]
                with h5py.File(root/f'thinfilm_closure_data_{name}.h5','w') as f:
                    f['x'], f['t'], f['h'], f['q'] = x,t,h,q
                    f['h_x'] = np.zeros_like(h)
                    f['h_xx'] = np.zeros_like(h)
                    f['q_x'] = np.broadcast_to(x[:, None] + t[None, :], h.shape)
                    f['q_xx'] = np.ones_like(h)
                    # A bad derivative at x=0 must not invalidate neighbors.
                    f['h_x'][0, :] = np.nan
                    for j,key in enumerate(cfg['available_targets']):
                        f[key] = (j+1)*h + q
                    g = f.create_group('parameters')
                    for key,value in dict(Re=10 if name!='C' else 9999,Ca=.01,angle=np.pi/4).items():
                        g[key] = value
            config = root/'data.json'; config.write_text(json.dumps(cfg)); prepare(config)
            ds = WindowDataset(root/'cache','train',cfg['available_targets'][:2],4,3)
            self.assertEqual(len(ds),100)
            for case in ['A', 'B', 'C']:
                folder = root/'cache'/case
                self.assertFalse((folder/'curvature.npy').exists())
                for key in cfg['local_fields']:
                    self.assertFalse((folder/f'{key}.npy').exists())
                splits = ['heldout'] if case == 'C' else ['train', 'val', 'test']
                for split in splits:
                    ids = np.load(folder/f'{split}_indices.npy')
                    with h5py.File(root/f'thinfilm_closure_data_{case}.h5', 'r') as f:
                        for key in cfg['local_fields']:
                            values = np.load(folder/f'{split}_{key}.npy')
                            self.assertEqual(values.shape, (len(ids),))
                            np.testing.assert_array_equal(
                                values, f[key][:][ids[:, 1], ids[:, 0]].astype('float32'))
            for e in ds.entries:
                idx = np.load(root/'cache'/e['name']/'train_indices.npy')
                self.assertTrue((idx[:,0] < 13).all())
                self.assertTrue((idx[:,0] > 0).all())
                self.assertTrue((x[idx[:,1]] < 6).all())
            vals = WindowDataset(root/'cache','val',ds.targets,0,1)
            tests = WindowDataset(root/'cache','test',ds.targets,0,1)
            self.assertEqual(vals.entries[0]['splits']['val']['time_indices'],[13,14,15])
            self.assertEqual(tests.entries[0]['splits']['test']['time_indices'],[16,17,18,19])
            a,y,_ = ds[0]
            np.testing.assert_allclose(a[2:-3:3],np.arange(-4,5)*3*.0488,atol=1e-7)
            self.assertEqual(a.shape,(30,))
            smaller = WindowDataset(root/'cache','train',ds.targets,0,1)
            np.testing.assert_array_equal(y,smaller[0][1])
            with self.assertRaises(ValueError):
                WindowDataset(root/'cache','train',ds.targets,41,1)
            # Center-only mask permits flat neighbors ahead of the wave.
            front_ds = WindowDataset(root/'cache','val',ds.targets,40,1)
            self.assertTrue(any(np.any(xx[:-3:3] == 1) for xx,_,_ in front_ds))
            run = root/'run'
            cmd = [sys.executable,str(HERE/'train.py'),'--cache',str(root/'cache'),'--run-dir',str(run),
                   '--targets',*ds.targets,'--epochs','1','--batch-size','32','--hidden-width','8','--hidden-layers','1','--device','cpu']
            subprocess.run(cmd,check=True)
            cmd[cmd.index('--epochs')+1] = '2'
            subprocess.run(cmd+['--resume'],check=True)
            import torch
            ck = torch.load(run/'best.pt',weights_only=False)
            self.assertEqual(ck['normalization']['x_mean'][-3],10.)
            for split in ['test','heldout']:
                subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--cache',str(root/'cache'),
                    '--checkpoint',str(run/'best.pt'),'--split',split,'--output',str(root/split),'--save-predictions'],check=True)
            # Rebuilding into a different directory with unchanged policy reproduces indices.
            cfg['cache_dir'] = str(root/'repeat'); config.write_text(json.dumps(cfg)); prepare(config)
            for name in ['A','B']:
                np.testing.assert_array_equal(np.load(root/'cache'/name/'train_indices.npy'),np.load(root/'repeat'/name/'train_indices.npy'))
            cfg['cache_dir'] = str(root/'unmasked'); cfg['flat_mask'] = False
            config.write_text(json.dumps(cfg)); prepare(config)
            old = json.loads((root/'cache'/'manifest.json').read_text())
            new = json.loads((root/'unmasked'/'manifest.json').read_text())
            self.assertGreater(new['cases'][0]['splits']['train']['eligible'],old['cases'][0]['splits']['train']['eligible'])
            from generate_sweep import generate
            sweep_cfg = json.loads((HERE/'sweep_config.json').read_text())
            sweep_cfg.update(
                cache=str(root/'cache'),
                networks=[dict(hidden_layers=1, hidden_width=8)],
                input_groups=dict(baseline=['h', 'q'], derivatives=['h', 'q', 'h_x', 'q_xx']),
                targets=['delta_Mx_ML', 'delta_Tw_ML'], seeds=[42],
            )
            sweep_cfg['training'].update(batch_size=100)
            sc = root/'sweep_config.json'; sc.write_text(json.dumps(sweep_cfg))
            sweep = generate(sc, root/'sweep')
            self.assertEqual(len(sweep['runs']),4)
            self.assertEqual(len({r['run_dir'] for r in sweep['runs']}), 4)
            self.assertTrue(all(len(r['targets']) == 1 for r in sweep['runs']))
            self.assertTrue(all('layers1_width8/' in r['name'] for r in sweep['runs']))
            from run_sweep_task import command_for
            command = command_for(sweep, 0, 'cpu')
            self.assertEqual(command[command.index('--input-mode') + 1], 'local')
            self.assertNotIn('--half-window', command)
            self.assertIn('--no-early-stopping', command)
            subprocess.run(['bash','-n',str(root/'sweep'/'submit_array.sh')],check=True)
            sweep_command = [sys.executable,str(HERE/'run_sweep_task.py'),'--sweep',str(root/'sweep'/'sweep.json'),
                             '--task-id','0','--device','cpu']
            subprocess.run(sweep_command,check=True,capture_output=True,text=True)
            final = torch.load(Path(sweep['runs'][0]['run_dir'])/'last.pt',weights_only=False)
            self.assertEqual(final['epoch'],100)
            self.assertEqual(len(final['history']),100)
            self.assertTrue(final['args']['no_early_stopping'])
            subprocess.run(sweep_command,check=True,capture_output=True,text=True)
            self.assertEqual(torch.load(Path(sweep['runs'][0]['run_dir'])/'last.pt',weights_only=False)['epoch'],100)



if __name__ == '__main__':
    unittest.main()
