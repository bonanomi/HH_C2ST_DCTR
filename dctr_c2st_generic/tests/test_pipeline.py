"""End-to-end orchestration parity, plus optional real TensorFlow/parquet parity.

The dependency-light recorder test verifies arrays/seeds/caps/metrics using a
small deterministic stand-in. It does NOT validate neural optimization or I/O.
The real test runs automatically when TensorFlow and pyarrow are installed.
"""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
import c2st_config as original_cfg
from dctr_c2st_generic.config import validate
from dctr_c2st_generic.training import run_channel

HAS_REAL = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("pyarrow") is not None


class RecorderModel:
    calls = []
    def __init__(self, features, seed=0, **kwargs):
        self.seed = seed
    def fit(self, x, y, sample_weight, validation_data, **kwargs):
        self.calls.append((self.seed, x.copy(), y.copy(), sample_weight.copy(),
                           tuple(a.copy() for a in validation_data)))
        self.coefficient = np.sum(x * (sample_weight * (2*y.astype(float)-1))[:, None], axis=0) / len(x)
        return types.SimpleNamespace(history={"loss": [.7, .6], "val_loss": [.72, .65]})
    def predict(self, x, **kwargs):
        logit = np.clip(x @ self.coefficient * .1, -10, 10)
        return (1/(1+np.exp(-logit))).astype(np.float32)[:, None]
    def save(self, path):
        Path(path).write_text("test recorder; NOT a trained TensorFlow model")


def fixture(signed):
    rng = np.random.default_rng(53)
    def frame(n, shift):
        return pd.DataFrame({"x": rng.normal(shift, 1, n).astype(np.float32),
                             "z": rng.normal(0, 1, n).astype(np.float32), "channel": "all"})
    t, b = frame(180, .2), frame(220, 0)
    t["weight_uncorrected"], t["weight"] = np.float32(1), np.float32(1)
    b["weight_uncorrected"] = rng.uniform(.3, 1.1, len(b)).astype(np.float32)
    b.loc[3, "weight_uncorrected"] = -.3
    b["is_dy"] = np.arange(len(b)) < 160
    b["weight"] = b.weight_uncorrected * np.where(b.is_dy, 1.1, 1).astype(np.float32)
    tables = {"C": b.iloc[:160].copy(), "S": b.iloc[160:].copy(), "T": t.copy()}
    cfg = validate({"target_processes": {"T": ["t"]},
                    "base_processes": {"C": {"datasets": ["c"], "is_dy": True}, "S": {"datasets": ["s"], "is_dy": False}},
                    "target_source": "a", "base_source": "a", "sources": {"a": {"format": "parquet", "root": "."}},
                    "features": ["x", "z"], "channels": {"all": ["all"]}, "folds": 3,
                    "normalization": "legacy", "compatibility": True, "reference_stages": ["dy"],
                    "subtract_processes": ["S"] if signed else [], "plots": False,
                    "epochs": 2, "batch_size": 64,
                    "dctr_model": {"hidden": [5], "batch_normalization": True, "optimizer": "sgd", "learning_rate": .005},
                    "closure_model": {"hidden": [5], "batch_normalization": False, "optimizer": "adam", "learning_rate": .001}})
    for side, f in (("target", t), ("base", b)):
        f["weight_before"], f["weight_dy"] = f.weight_uncorrected, f.weight
        f["process"] = "T" if side == "target" else np.where(f.is_dy, "C", "S")
        f["event_id"] = [f"{side}/{i}" for i in range(len(f))]
        f["source"], f["dataset"], f["branch"], f["row"] = side, side, 0, np.arange(len(f))
        f["corrected"] = side == "base"
        if side == "base" and signed: f["corrected"] = f.is_dy
        if side == "target": f["is_dy"] = False
    return cfg, tables, t, b


def parity(signed, real=False):
    cfg, tables, target, base = fixture(signed)
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        tmp = Path(tmp)
        stub_loader = types.ModuleType("dyvr_lib")
        stub_loader.discover_store_layout = lambda *a: None
        stub_loader.load_all = lambda *a, **kw: tables
        injected = {"dyvr_lib": stub_loader}
        if not real:
            injected["tensorflow"] = types.SimpleNamespace(keras=types.SimpleNamespace(
                backend=types.SimpleNamespace(clear_session=lambda: None),
                callbacks=types.SimpleNamespace(ReduceLROnPlateau=lambda **kw: None, EarlyStopping=lambda **kw: None)),
                config=types.SimpleNamespace(list_physical_devices=lambda *a: []))
            injected["c2st_models"] = types.SimpleNamespace(build_binary_classifier=RecorderModel)
            stack.enter_context(patch.object(pd.DataFrame, "to_parquet", lambda self, path, **kw: self.to_pickle(path)))
            stack.enter_context(patch.object(pd, "read_parquet", lambda path, **kw: pd.read_pickle(path)))
            RecorderModel.calls = []
        stack.enter_context(patch.dict(sys.modules, injected))
        path = Path(__file__).parents[2]/"c2st_final_closure/train_dctr_crossfit_closure.py"
        spec = importlib.util.spec_from_file_location("legacy_parity", path)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        old.cfg = types.SimpleNamespace(**{k: v for k, v in vars(original_cfg).items() if k.isupper()})
        for key, value in {"FEATURES": ["x", "z"], "LOAD_FEATURES": ["x", "z"], "VALIDATION_VARS": [],
                           "SELECTIONS": {}, "DATA_PROCESSES": cfg["target_processes"], "MC_PROCESSES": cfg["base_processes"],
                           "CHANNELS": ["all"], "EPOCHS": 2, "BATCH_SIZE": 64, "HIDDEN": (5,), "DCTR_HIDDEN": (5,)}.items():
            setattr(old.cfg, key, value)
        args = ["train", "--channels", "all", "--folds", "3", "--output", str(tmp/"old")]
        if signed: args += ["--dctr-target", "dy_only"]
        with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()): old.main()
        old_calls = list(RecorderModel.calls) if not real else []
        if not real: RecorderModel.calls = []
        with redirect_stdout(io.StringIO()): run_channel(target, base, cfg, tmp/"new", "all")
        olddir = tmp/"old"/("dy_only" if signed else "")/"all"
        newdir = tmp/"new"
        a = np.load(olddir/"dctr_factors_mc.npz"); b = np.load(newdir/"dctr_factors_base.npz")
        assert_array_equal(a["fold_id"], b["fold_id"])
        assert_allclose(a["dctr_factor"], b["dctr_factor"], rtol=1e-6 if real else 0, atol=1e-7 if real else 0)
        for stage in ("before", "dy", "dctr"):
            pa, pb = np.load(olddir/f"closure_{stage}_test.npz"), np.load(newdir/f"closure_{stage}_test.npz")
            assert_allclose(pa["p_test"], pb["p_test"], rtol=1e-6 if real else 0, atol=1e-7 if real else 0)
            assert_array_equal(pa["w_test"], pb["w_test"])
        if not real:
            if len(old_calls) != len(RecorderModel.calls): raise AssertionError("Different fit call counts")
            for a, b in zip(old_calls, RecorderModel.calls):
                if a[0] != b[0]: raise AssertionError("Model seed mismatch")
                for x, y in zip(a[1:4], b[1:4]): assert_array_equal(x, y)
                for x, y in zip(a[4], b[4]): assert_array_equal(x, y)
        return True


class PipelineParity(unittest.TestCase):
    def test_inclusive_recorder_parity(self): self.assertTrue(parity(False))
    def test_signed_recorder_parity(self): self.assertTrue(parity(True))

    @unittest.skipUnless(HAS_REAL, "Requires TensorFlow and pyarrow; run in the production environment")
    def test_inclusive_tensorflow_parity(self): self.assertTrue(parity(False, True))

    @unittest.skipUnless(HAS_REAL, "Requires TensorFlow and pyarrow; run in the production environment")
    def test_signed_tensorflow_parity(self): self.assertTrue(parity(True, True))


def generic_profiles(output, plots=False):
    """Exercise generic weighted target/base profiles with explicit test doubles."""
    fake_tf = types.SimpleNamespace(keras=types.SimpleNamespace(
        backend=types.SimpleNamespace(clear_session=lambda: None),
        callbacks=types.SimpleNamespace(ReduceLROnPlateau=lambda **kw: None, EarlyStopping=lambda **kw: None)))
    with patch.dict(sys.modules, {"tensorflow": fake_tf, "c2st_models": types.SimpleNamespace(build_binary_classifier=RecorderModel)}), \
         patch.object(pd.DataFrame, "to_parquet", lambda self, path, **kw: self.to_pickle(path)), \
         patch.object(pd, "read_parquet", lambda path, **kw: pd.read_pickle(path)):
        for signed in (False, True):
            for mode in ("shape", "yield"):
                cfg, _, target, base = fixture(signed)
                cfg.update(compatibility=False, normalization=mode, plots=plots and signed and mode == "yield")
                target["weight_before"] *= np.float32(1.4)
                target["weight_dy"] = target.weight_before
                cfg["target_weights"] = {"mode": "unit", "scale": 1.4}
                dest = Path(output)/f"{mode}_{'signed' if signed else 'inclusive'}"
                RecorderModel.calls = []
                with redirect_stdout(io.StringIO()): run_channel(target, base, cfg, dest, "all")
                data = np.load(dest/"dctr_factors_base.npz")
                assert_array_equal(data["dctr_factor"][~data["corrected"]], 1.)
                if not np.all(np.isfinite(data["dctr_factor"])): raise AssertionError("Nonfinite factors")
                if not (dest/"inference/metadata.json").exists(): raise AssertionError("Missing inference contract")
                if cfg["plots"] and len(list((dest/"plots").glob("*.png"))) < 10:
                    raise AssertionError("Missing diagnostic plots")


class GenericProfiles(unittest.TestCase):
    def test_weighted_shape_yield_and_subtraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            generic_profiles(tmp, plots=True)

    @unittest.skipUnless(HAS_REAL, "Requires TensorFlow and pyarrow; run in the production environment")
    def test_real_joined_loader_and_training(self):
        from dctr_c2st_generic.examples.toy import create_toy
        from dctr_c2st_generic.config import load_config
        from dctr_c2st_generic.training import run
        from dctr_c2st_generic.apply import DCTRReweighter
        with tempfile.TemporaryDirectory() as tmp:
            path = create_toy(Path(tmp)/"inputs", n=300)
            cfg = load_config(path); cfg.update(epochs=1, plots=False, folds=2)
            run(cfg, Path(tmp)/"result")
            rw = DCTRReweighter(Path(tmp)/"result/all/inference")
            values = pd.DataFrame({"x": [0., 1.], "z": [1., 2.]})
            weights = rw.reweight(values, np.array([-2., 3.]))
            self.assertLess(weights[0], 0); self.assertGreater(weights[1], 0)


if __name__ == "__main__": unittest.main()
