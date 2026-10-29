"""Verify derived features, unchanged splits, normalization and model compatibility."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from prepare_data import prepare
from augment_curvature import augment
from datasets import WindowDataset
from train import fit_normalization
from generate_sweep import generate
from run_sweep_task import command_for

HERE=Path(__file__).resolve().parent

class CurvatureTest(unittest.TestCase):
    def test_curvature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            x=np.arange(0,10,.0488);t=np.arange(20.)
            h=1.2+.1*np.sin(x[:,None]+t[None,:]*.01)
            hx=.1*np.cos(x[:,None]+t[None,:]*.01)
            hxx=-.1*np.sin(x[:,None]+t[None,:]*.01)
            raw=root/'thinfilm_closure_data_A.h5'
            with h5py.File(raw,'w') as f:
                for k,v in dict(x=x,t=t,h=h,q=h*.5,h_x=hx,h_xx=hxx,target=hxx*3).items():f[k]=v
                g=f.create_group('parameters')
                for k,v in dict(Re=10.,Ca=.01,angle=.7).items():g[k]=v
            cfg=json.loads((HERE/'data_config.json').read_text())
            cfg.update(data_dir=str(root),cache_dir=str(root/'old'),development_cases=['A'],heldout_cases=[],available_targets=['target'])
            cfg['sample_caps']=dict(train=40,val=20,test=20,heldout=None)
            cp=root/'config.json';cp.write_text(json.dumps(cfg));prepare(cp)
            expected=(hxx/(1+hx**2)**1.5).T
            np.testing.assert_allclose(np.load(root/'old'/'A'/'curvature.npy'),expected,rtol=1e-6,atol=1e-8)
            # Simulate a legacy cache to retain coverage of the optional upgrader.
            mp=root/'old'/'manifest.json'
            legacy=json.loads(mp.read_text());legacy.pop('derived_inputs')
            mp.write_text(json.dumps(legacy))
            (root/'old'/'A'/'curvature.npy').unlink()
            result=augment(root/'old',root/'new')
            self.assertNotEqual(result['id'], result['parent_cache_id'])
            for split in ['train','val','test']:
                self.assertEqual((root/'old'/'A'/f'{split}_indices.npy').read_bytes(),(root/'new'/'A'/f'{split}_indices.npy').read_bytes())
            actual=np.load(root/'new'/'A'/'curvature.npy')
            expected=(hxx/(1+hx**2)**1.5).T
            np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8)
            old=WindowDataset(root/'old','train',['target'],3,2)
            base=WindowDataset(root/'new','train',['target'],3,2)
            ds=WindowDataset(root/'new','train',['target'],3,2,['h','q','curvature'])
            np.testing.assert_array_equal(old[0][0],base[0][0])
            X,Y,_=ds[0]
            self.assertEqual(X.shape,(31,))
            it,ix=np.load(root/'new'/'A'/'train_indices.npy')[0]
            np.testing.assert_allclose(X[:-3].reshape(-1,4)[:,2],expected[it,ix+np.arange(-3,4)*2],rtol=1e-6)
            norm=fit_normalization(ds,16,0)
            xx=np.stack([ds[i][0] for i in range(len(ds))])
            self.assertAlmostEqual(norm['x_mean'][2],float(xx[:,:-3].reshape(-1,7,4)[:,:,2].mean()),places=7)
            cmd=[sys.executable,str(HERE/'train.py'),'--cache',str(root/'new'),'--run-dir',str(root/'run'),
                '--input-fields','h','q','curvature','--targets','target','--epochs','2','--half-window','3','--stride','2',
                '--batch-size','16','--hidden-width','8','--hidden-layers','1','--device','cpu']
            subprocess.run(cmd,check=True)
            subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--cache',str(root/'new'),
                '--checkpoint',str(root/'run'/'best.pt'),'--split','val','--output',str(root/'eval'),'--save-predictions'],check=True)
            sc=json.loads((HERE/'sweep_config.json').read_text())
            sc.update(cache=str(root/'new'),input_fields=['h','q','curvature'],target_groups={'physical':['target']},half_windows=[3],strides=[2])
            sp=root/'sweep_cfg.json';sp.write_text(json.dumps(sc));sw=generate(sp,root/'sweep')
            self.assertIn('curvature',command_for(sw,0,'cpu'))
            # Never silently drop selected centers when derivatives are invalid.
            with h5py.File(raw,'a') as f:f['h_x'][ix,it]=np.nan
            with self.assertRaisesRegex(ValueError,'nonfinite curvature'):
                augment(root/'old',root/'bad')
            self.assertFalse((root/'bad'/'manifest.json').exists())
            # Fresh preparation filters the same centers for both input choices.
            cfg['cache_dir']=str(root/'fresh_invalid');cfg['sample_caps']['train']=None
            cp.write_text(json.dumps(cfg));prepare(cp)
            ids=np.load(root/'fresh_invalid'/'A'/'train_indices.npy')
            affected=(ids[:,0]==it)&(np.abs(x[ids[:,1]]-x[ix])<=cfg['max_half_width']+1e-10)
            self.assertFalse(affected.any())
            base=WindowDataset(root/'fresh_invalid','train',['target'],3,2)
            curved=WindowDataset(root/'fresh_invalid','train',['target'],3,2,['h','q','curvature'])
            self.assertEqual(len(base),len(curved))

if __name__=='__main__':unittest.main()
