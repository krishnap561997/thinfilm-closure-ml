"""Small synthetic regression tests; python -m unittest -v test_statistics.py."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import analyze_closures as mod


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.src=self.root/'thinfilm_closure_data_Re040_Ca0_100_theta45.h5'
        rng=np.random.default_rng(40)
        self.x=np.arange(31)*.125
        self.h=1+.25*rng.standard_normal((31,17))
        self.q=rng.standard_normal((31,17))
        self.hx=rng.uniform(-.1,.1,(31,17))
        self.y=2*self.h-.7*self.q+rng.standard_normal((31,17))*.1
        self.h[4,3]=np.nan
        self.y[8,2]=np.nan
        with h5py.File(self.src,'w') as f:
            f['x']=self.x;f['t']=np.arange(17.)
            f['h']=self.h;f['q']=self.q.T # exercise transposed layout
            f['h_x']=self.hx;f['Y']=self.y;f['constant']=np.ones_like(self.y)
        cfg={
          'data_dir':str(self.root),'output_dir':str(self.root/'results'),
          'simulation':{'Re':40,'Ca':.1,'beta':45},
          'qois':['Y','constant'],'offsets':{'min':-2,'max':2},
          'regions':{'All':{},'Highlands':{'h_gt':1.15,'abs_hx_le':.05,'require_finite_hx':True},
                     'Lowlands':{'h_lt':.85,'require_finite_hx':True},'Empty':{'h_gt':100}},
          'pdf':{'bins':11,'n_std':1},'correlation_use_fluctuations':True,'pdf_use_fluctuations':False}
        self.config=self.root/'config.json';self.config.write_text(json.dumps(cfg))
        self.cfg=mod.load_config(self.config)
    def tearDown(self):self.tmp.cleanup()
    def output(self):return next((self.root/'results').glob('*/*statistics.h5'))
    def test_notebook_parity(self):
        mod.analyze_one(self.src,self.cfg)
        hp,_=mod.fluctuations(self.h);qp,_=mod.fluctuations(self.q);yp,_=mod.fluctuations(self.y)
        K=2;sl=slice(K,-K);base=np.isfinite(self.h[sl])
        for n in self.cfg['offsets']:
            base &= np.isfinite(hp[K+n:31-K+n]) & np.isfinite(qp[K+n:31-K+n])
        masks={'All':base,'Highlands':base & np.isfinite(self.hx[sl]) & (self.h[sl]>1.15) & (np.abs(self.hx[sl])<=.05),
               'Lowlands':base & np.isfinite(self.hx[sl]) & (self.h[sl]<.85)}
        with h5py.File(self.output(),'r') as f:
            np.testing.assert_array_equal(f['distances'][:],np.arange(-2,3)*.125)
            for region,mask in masks.items():
                mask=mask & np.isfinite(yp[sl])
                for field,a in [('h',hp),('q',qp)]:
                    expected=[np.corrcoef(yp[sl][mask],a[K+n:31-K+n][mask])[0,1] for n in range(-2,3)]
                    np.testing.assert_allclose(f[f'qois/Y/{region}/pearson/{field}'][:],expected,atol=1e-13)
                rg=f[f'qois/Y/{region}/pdf'];values=self.y[sl][mask]
                self.assertEqual(rg.attrs['n_points'],len(values))
                self.assertAlmostEqual(rg.attrs['mean'],values.mean())
                self.assertAlmostEqual(rg.attrs['rms'],np.sqrt(np.mean(values**2)))
                raw=rg['raw'];self.assertAlmostEqual(np.sum(raw['density'][:]*np.diff(raw['bin_edges'][:])),1)
                std=rg['standardized'];z=(values-values.mean())/values.std()
                counts=np.histogram(z,bins=std['bin_edges'][:])[0]
                np.testing.assert_array_equal(std['counts'][:],counts)
                self.assertAlmostEqual(np.sum(std['density'][:]*np.diff(std['bin_edges'][:])),counts.sum()/len(values))
            self.assertTrue(np.isnan(f['qois/constant/All/pearson/h'][:]).all())
            self.assertEqual(f['qois/Y/Empty/pearson'].attrs['n_points'],0)
            self.assertTrue(np.isnan(f['qois/Y/Empty/pdf/raw/density'][:]).all())
    def test_repeat_config_and_source_change(self):
        self.assertEqual(mod.analyze_one(self.src,self.cfg),'saved')
        self.assertEqual(mod.analyze_one(self.src,self.cfg),'skipped')
        with h5py.File(self.src,'a') as f:f['Y'][1,1]+=1
        self.assertEqual(mod.analyze_one(self.src,self.cfg),'saved')
        newer=dict(self.cfg);newer['offsets']=[0]
        self.assertNotEqual(mod.config_hash(newer),mod.config_hash(self.cfg))
        self.assertEqual(mod.select_files(self.cfg,'single'),[self.src])
        self.assertEqual(mod.select_files(self.cfg,'scan'),[self.src])
    def test_source_change_does_not_publish(self):
        from unittest.mock import patch
        real=mod.calculate
        def change(*args):
            real(*args)
            with h5py.File(self.src,'a') as f:f['Y'][1,1]+=1
        with patch.object(mod,'calculate',change):
            with self.assertRaisesRegex(RuntimeError,'Source changed'):
                mod.analyze_one(self.src,self.cfg)
        self.assertFalse(list((self.root/'results').glob('*/*statistics.h5')))
    def test_raw_correlations(self):
        cfg=dict(self.cfg);cfg['correlation_use_fluctuations']=False
        self.assertEqual(mod.analyze_one(self.src,cfg),'saved')
        with h5py.File(self.output(),'r') as f:
            self.assertTrue(np.isfinite(f['qois/Y/All/pearson/h'][:]).all())

if __name__=='__main__':unittest.main()
