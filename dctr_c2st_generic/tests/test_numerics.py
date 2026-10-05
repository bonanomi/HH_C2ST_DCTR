"""Dependency-light tests runnable without TensorFlow, awkward, or pyarrow."""
import ast
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from c2st_core import stage_weights, normalize_signed_sample_weights, split_class_indices
from dctr_c2st_generic.config import validate, load_config
from dctr_c2st_generic.loading import category_mask, weight_product, prepare_tables, branch_files
from dctr_c2st_generic.numerics import (sample_weights, closure_weights, odds, factor_summary,
                                       reweight_signed, shuffled_folds, inner_train_val)
from dctr_c2st_generic.training import training_plan, transform, fit_transformer
from dctr_c2st_generic.apply import DCTRReweighter


def minimal():
    return validate({"target_processes": {"T": ["t"]}, "base_processes": {"B": ["b"]},
                     "target_source": "s", "base_source": "s",
                     "sources": {"s": {"format": "parquet", "root": "."}},
                     "features": ["x"], "channels": {"all": ["all"]}, "folds": 3})


def legacy_functions(*names):
    """Execute original pure functions verbatim, avoiding irrelevant HEP/TF imports."""
    path = Path(__file__).parents[2] / "c2st_final_closure/train_dctr_crossfit_closure.py"
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"np": np, "normalize_signed_sample_weights": normalize_signed_sample_weights}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Contracts(unittest.TestCase):
    def test_required_roles(self):
        c = minimal(); del c["base_processes"]
        with self.assertRaises(ValueError): validate(c)

    def test_unknown_config_rejected(self):
        with self.assertRaises(ValueError): validate({**minimal(), "normalisation": "yield"})

    def test_role_structures(self):
        c = minimal()
        self.assertEqual(c["base_processes"]["B"], {"datasets": ["b"]})
        c["base_processes"]["B"]["is_dy"] = True
        self.assertTrue(validate(c)["base_processes"]["B"]["is_dy"])

    def test_subtraction_requires_component(self):
        c = minimal(); c["subtract_processes"] = ["B"]
        with self.assertRaises(ValueError): validate(c)

    def test_invalid_cap_and_epsilon(self):
        for field, value in (("cap_quantile", 0), ("cap_quantile", 1.1), ("eps", .5)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate({**minimal(), field: value})

    def test_category_union_no_double_count(self):
        assert_array_equal(category_mask([[1], [3, 4], [4], [10]], [3, 4]), [False, True, True, False])

    def test_unit_weights_do_not_fetch(self):
        def fail(*args): raise AssertionError("unit weights must not read weight producers")
        assert_array_equal(weight_product({"mode": "unit", "scale": .5}, fail, 4), [.5] * 4)

    def test_weight_product_order_and_sign(self):
        values = {"a": np.array([1.17, -2., 3]), "b": np.array([.5, .2, .3])}
        result = weight_product({"mode": "product", "fields": ["a", "b"]}, lambda p, c, o: values[c], 3)
        expected = np.ones(3, np.float32)
        for key in ("a", "b"): expected *= values[key].astype(np.float32)
        assert_array_equal(result, expected)

    def test_legacy_inclusive_weights_exact(self):
        wt, wb = np.ones(40, np.float32), np.linspace(.3, 4, 80, dtype=np.float32)
        it, ib, empty = np.arange(13), np.arange(21), np.array([], int)
        a, b = stage_weights(len(wt), wb, wb[ib], len(it))
        assert_array_equal(sample_weights(wt, wb, (it, empty, ib), (it, empty, ib), "legacy"), np.r_[a, b])

    def test_legacy_signed_sample_exact(self):
        old = legacy_functions("make_dy_only_sample")["make_dy_only_sample"]
        xt = np.arange(100, dtype=np.float32).reshape(50, 2)
        xb = np.arange(160, dtype=np.float32).reshape(80, 2)
        t, s, b = np.arange(20), np.arange(10), np.arange(10, 50)
        wb = np.linspace(.1, 1, 80, dtype=np.float32)
        _, _, expected = old(xt, t, xb, s, b, wb)
        got = sample_weights(np.ones(50), wb, (t, s, b), (t, s, b), "legacy_signed")
        assert_array_equal(got, expected)

    def test_shape_vs_yield(self):
        indices = (np.arange(10), np.array([], int), np.arange(10))
        wt, wb = np.full(10, 2.), np.ones(10)
        shape = sample_weights(wt, wb, indices, indices, "shape")
        physical = sample_weights(wt, wb, indices, indices, "yield")
        self.assertAlmostEqual(shape[:10].sum() / shape[10:].sum(), 1.)
        self.assertAlmostEqual(physical[:10].sum() / physical[10:].sum(), 2.)

    def test_signed_residual_shape_balance(self):
        it, ins, ib = np.arange(10), np.arange(5), np.arange(5, 15)
        wt, wb = np.full(10, 2.), np.ones(15)
        w = sample_weights(wt, wb, (it, ins, ib), (it, ins, ib), "shape")
        self.assertAlmostEqual(w[:15].sum() / w[15:].sum(), 1., places=6)
        self.assertTrue(np.all(w[10:15] < 0))

    def test_nonpositive_residual_rejected(self):
        idx = (np.arange(2), np.arange(3), np.arange(3, 6))
        with self.assertRaises(ValueError): sample_weights(np.ones(2), np.ones(6), idx, idx, "yield")

    def test_training_balance_ignores_held_out_weights(self):
        ref = (np.arange(5), np.array([], int), np.arange(5))
        wt, wb = np.ones(10), np.ones(10)
        expected = sample_weights(wt, wb, ref, ref, "shape")
        wt[5:], wb[5:] = 1000, 500
        assert_array_equal(sample_weights(wt, wb, ref, ref, "shape"), expected)

    def test_exposure_scales_remove_luminosity_ratio(self):
        idx = (np.arange(20), np.array([], int), np.arange(10))
        # Twice the target exposure, identical per-luminosity rate.
        w = sample_weights(np.full(20, .5), np.ones(10), idx, idx, "yield")
        self.assertAlmostEqual(w[:20].sum() / w[20:].sum(), 1.)

    def test_closure_balance_train_only(self):
        wt, wb = np.ones(10), np.ones(20)
        it, ib = np.arange(5), np.arange(8)
        expected = closure_weights(wt, wb, it, ib, it, ib)
        wt[5:], wb[8:] = 30, 60
        assert_array_equal(closure_weights(wt, wb, it, ib, it, ib), expected)

    def test_zero_reference_weight_is_allowed(self):
        wt, wb = np.ones(5, np.float32), np.array([0, 1, 2, 3, 4], np.float32)
        i = np.arange(5)
        got = closure_weights(wt, wb, i, i, i, i, compatibility=True)
        a, b = stage_weights(5, wb, wb, 5)
        assert_array_equal(got, np.r_[a, b])

    def test_odds_and_signed_application(self):
        assert_allclose(odds([.2, .5, .8], 1e-6), [.25, 1, 4])
        assert_array_equal(reweight_signed([-2., 0., 3.], [.5, 2., 2.]), [-1, 0, 6])
        with self.assertRaises(ValueError): odds([np.nan], 1e-6)

    def test_cap_diagnostics(self):
        d = factor_summary([1., 2., 2.], [1., 2., 1.], 2.)
        self.assertEqual(d["weighted_mean"], 1.75)
        self.assertAlmostEqual(d["fraction_at_cap"], 2/3)

    def test_split_primitive_legacy_exact(self):
        old = legacy_functions("shuffled_folds", "inner_train_val")
        for a, b in zip(shuffled_folds(np.arange(103), 5, 19), old["shuffled_folds"](np.arange(103), 5, 19)):
            assert_array_equal(a, b)
        for a, b in zip(inner_train_val(np.arange(99), .15, 31), old["inner_train_val"](np.arange(99), .15, 31)):
            assert_array_equal(a, b)

    def test_plan_no_leakage_and_complete_coverage(self):
        c = minimal(); corr = np.r_[np.ones(150, bool), np.zeros(50, bool)]
        outer, jobs = training_plan(180, 200, corr, c)
        held = []
        for job in jobs:
            train, val = job["train"], job["val"]
            for train_idx, val_idx, hold in zip(train, val, (job["hold_target"], job["hold_subtract"], job["hold_base"])):
                self.assertFalse(set(train_idx) & set(val_idx))
                self.assertFalse(set(train_idx) & set(hold))
                self.assertFalse(set(val_idx) & set(hold))
            self.assertFalse(set(train[0]) & set(outer["target_test"]))
            self.assertFalse(set(np.r_[train[1], train[2]]) & set(outer["base_test"]))
            held.extend(job["hold_base"])
        assert_array_equal(np.sort(held), np.flatnonzero(corr))
        self.assertEqual(len(set(held)), len(held))

    def test_scaler_ignores_holdout(self):
        c = minimal(); c["scaler"] = "standard"
        t, b = pd.DataFrame({"x": [1., 2., 3., 1e9]}), pd.DataFrame({"x": [2., 3., 4., -1e9]})
        scaler = fit_transformer(t, b, [0, 1, 2], [0, 1, 2], c)
        self.assertEqual(scaler.mean_[0], 2.5)

    def test_physical_tables_not_mutated(self):
        c = minimal()
        def table(prefix):
            return pd.DataFrame({"event_id": [prefix+str(i) for i in range(6)], "weight_before": [1, 2, 3, 4, -2, 0]})
        t, b = table("t"), table("b")
        tp, bp, report = prepare_tables(t, b, c)
        self.assertEqual(len(tp), 4); self.assertEqual(len(b), 6)
        self.assertEqual(report["base"]["full_signed_sumw"], 8)
        with self.assertRaises(ValueError): prepare_tables(t, t.copy(), c)

    def test_duplicate_branch_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            for sub in ("ds_v1", "ds_v2"):
                path = Path(tmp)/"nominal"/sub; path.mkdir(parents=True)
                (path/"events_0.parquet").touch()
            with self.assertRaises(ValueError): branch_files(tmp, "ds")

    def test_component_inference_mask_and_cap(self):
        obj = DCTRReweighter.__new__(DCTRReweighter)
        obj.metadata = {"features": ["x"], "batch_size": 9, "eps": 1e-6, "cap": 2.,
                        "corrected_processes": ["C"], "subtract_processes": ["S"]}
        from sklearn.preprocessing import StandardScaler
        frame = pd.DataFrame({"x": [1., 2., 3.]})
        obj.scaler = StandardScaler().fit(frame)
        obj.model = types.SimpleNamespace(predict=lambda x, **kw: np.full((len(x), 1), .9))
        with self.assertRaises(ValueError): obj.predict(frame)
        f = obj.predict(frame, processes=["C", "S", "C"])
        assert_array_equal(f, [2, 1, 2])
        assert_array_equal(obj.reweight(frame, [-2., 3., 4.], processes=["C", "S", "C"]), [-4, 3, 8])

    def test_example_config_loads(self):
        root = Path(__file__).parents[1] / "examples"
        for name in ("legacy_inclusive.py", "legacy_dy_only.py", "data25_over_data24.py"):
            c = load_config(root/name)
            self.assertTrue(c["features"])


if __name__ == "__main__": unittest.main()
