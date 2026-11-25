"""Signed-objective contracts; optionally exercise actual Keras in the production env."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dctr_c2st_generic.tests.test_pipeline import fixture
from dctr_c2st_generic.config import validate
from dctr_c2st_generic.training import run_channel
from dctr_c2st_generic.plot_signed_physical import match_rows, verify_factors
from dctr_c2st_generic.compare_signed_positive import load_full


class SignedPipeline(unittest.TestCase):
    def test_signed_jobs_and_positive_closure(self):
        cfg, _, t, b = fixture(True)
        cfg.update(compatibility=False, normalization="yield", dctr_objective="signed_mse",
                   validation_vars=[], regions=["dycr"], region_ids={"dycr":3}, plots=False)
        cfg = validate(cfg)
        b.loc[b.index % 4 == 0, "weight_before"] = -.25
        b.loc[1, "weight_before"] = 0.
        b["weight_dy"] = b.weight_before
        wt, wb = t.weight_before.to_numpy(), b.weight_before.to_numpy()
        calls = []
        class Model:
            def __init__(self, n, loss, **kw): self.loss = loss
            def fit(self,x,y,sample_weight,validation_data,**kw):
                calls.append((self.loss,y.copy(),sample_weight.copy(),validation_data))
                return NS(history={"loss":[.1], "val_loss":[.12]})
            def predict(self,x,**kw): return np.full((len(x),1),.55)
            def save(self,path): path.touch()
        fake_tf = NS(keras=NS(backend=NS(clear_session=lambda:None),callbacks=NS(
            ReduceLROnPlateau=lambda **kw:None,EarlyStopping=lambda **kw:None)))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{
            "tensorflow":fake_tf,"c2st_models":NS(build_binary_classifier=Model)}), patch.object(
            pd.DataFrame,"to_parquet",lambda df,path,**kw:df.to_pickle(path)), patch.object(
            pd,"read_parquet",pd.read_pickle):
            out = Path(tmp)/"all"
            run_channel(t,b,cfg,out,"all")
            saved = pd.read_pickle(out/"base_rows.parquet")
            np.testing.assert_array_equal(saved.event_id,b.event_id)
            match_rows(b,saved,"base",signed=True)
            test = pd.read_pickle(out/"outer_test_fold.parquet")
            self.assertTrue((test.weight_before < 0).any())
            with np.load(out/"splits.npz") as sp, np.load(out/"dctr_factors_base.npz") as fa:
                verify_factors(saved, fa["dctr_factor"], fa["fold_id"], fa["fold_caps"], sp, cfg["folds"])
                for j, name in enumerate(["fold_0","fold_1","fold_2","final"]):
                    loss,y,w,val = calls[j]
                    self.assertEqual(loss,"mean_squared_error")
                    for split,yy,ww in [("train",y,w),("val",val[1],val[2])]:
                        ti,si,bi = (sp[f"{name}_{split}_{k}"] for k in ("target","subtract","base"))
                        raw = np.r_[wt[ti],-wb[si],wb[bi]].astype(float)
                        self.assertEqual(yy.shape,(len(raw),1))
                        np.testing.assert_array_equal(yy[:,0],np.r_[np.ones(len(ti)+len(si)),np.zeros(len(bi))])
                        np.testing.assert_allclose(ww,raw/np.mean(abs(raw)),rtol=1e-6)
                for loss,_,w,val in calls[4:]:
                    self.assertEqual(loss,"binary_crossentropy")
                    self.assertTrue((w>0).all() and (val[2]>0).all())
                bb = sp["base_test"]
                yields = json.loads((out/"yield_closure.json").read_text())
                self.assertAlmostEqual(yields["stages"]["dctr"]["base_sumw_test"],
                                       np.sum((wb*fa["dctr_factor"])[bb],dtype=float))
                # Both full plotting entry points must reuse OOF negative factors.
                # Any unexpected inference also fails: the stub cannot load a model.
                from dctr_c2st_generic import plot_signed_physical as pp
                from dctr_c2st_generic import compare_signed_positive as cp
                for frame in (t,b):
                    for col in pp.OBSERVABLES: frame[col]=frame.x
                def tables(c,side,*args): return (t if side=="target" else b).copy()
                with patch.object(cp,"load_signed_cached",tables):
                    _,_,f,ids,_=load_full(out,None)
                np.testing.assert_array_equal(f,fa["dctr_factor"])
                np.testing.assert_array_equal(ids,fa["fold_id"])
                with patch.object(pp,"load_signed_cached",tables),patch.object(pp,"draw") as draw:
                    pp.run(out,Path(tmp)/"plots")
                    for call in draw.call_args_list:
                        if call.args[9]=="dctr":
                            np.testing.assert_array_equal(call.args[2],f)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") and importlib.util.find_spec("pyarrow"),
                         "requires TensorFlow and pyarrow")
    def test_real_signed_fold_and_export(self):
        from dctr_c2st_generic.apply import DCTRReweighter
        cfg, _, t, b = fixture(True)
        cfg.update(dctr_objective="signed_mse", compatibility=False, normalization="yield",
                   folds=2, epochs=1, save_fold_models=True)
        b.loc[b.index % 4 == 0, "weight_before"] = -.25
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"all"
            run_channel(t, b, validate(cfg), out, "all")
            with np.load(out/"splits.npz") as sp, np.load(out/"dctr_factors_base.npz") as fa:
                for i, name in enumerate(("fold_0","fold_1","final")):
                    idx = sp[f"{name}_hold_base"]
                    model = DCTRReweighter(out/("inference" if name=="final" else name))
                    pred = model.predict(b.iloc[idx], processes=b.process.to_numpy()[idx])
                    np.testing.assert_allclose(pred, fa["dctr_factor"][idx], rtol=1e-5)
                    self.assertTrue(np.all(fa["fold_id"][idx]==i))
                self.assertTrue(np.all(fa["dctr_factor"][~b.corrected.to_numpy()]==1.))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"),"requires TensorFlow")
    def test_real_keras_signed_reduction(self):
        from c2st_models import build_binary_classifier
        m=build_binary_classifier(1,hidden=[],optimizer="sgd",learning_rate=.01,loss="mean_squared_error")
        # Set p=0.25: (2 * (1-p)^2 - 1 * p^2) / 2
        m.set_weights([np.zeros((1,1),np.float32),np.array([-np.log(3)],np.float32)])
        loss=m.test_on_batch(np.zeros((2,1),np.float32),np.array([[1],[0]],np.float32),
                             sample_weight=np.array([2,-1],np.float32))
        self.assertAlmostEqual(float(loss),(2*.75**2-.25**2)/2,places=6)

if __name__=="__main__": unittest.main()
