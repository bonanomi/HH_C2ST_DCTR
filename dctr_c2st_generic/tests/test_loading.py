"""CF-loader contract tests with in-memory parquet readers, no external inputs.

Tests exercise branch discovery, column selection, products, alignment safeguards
and region unions. They do not substitute for real parquet serialization checks.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from dctr_c2st_generic.config import validate
from dctr_c2st_generic.loading import load_side


class LoaderContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.tables = {}
        def write(where, frame):
            path = self.root/where/"nominal"/"ds"/"events_0.parquet"
            path.parent.mkdir(parents=True); path.touch()
            self.tables[str(path)] = frame
        self.write = write
        write("red", pd.DataFrame({"category_ids": [[3, 40], [3, 4, 40], [4, 40], [1, 40], [3, 30]]}))
        write("store/cf.ProduceColumns/prod__dl_ml_inputs_v1", pd.DataFrame({"x": np.arange(5, dtype=float)}))
        pq = types.ModuleType("pyarrow.parquet")
        pq.read_schema = lambda path: types.SimpleNamespace(names=list(self.tables[str(path)].columns))
        pq.read_table = lambda path, columns: types.SimpleNamespace(to_pandas=lambda: self.tables[str(path)][columns].copy())
        pa = types.ModuleType("pyarrow"); pa.parquet = pq
        self.stack.enter_context(patch.dict(sys.modules, {"pyarrow": pa, "pyarrow.parquet": pq}))
        self.cfg = validate({"target_processes": {"T": ["ds"]}, "base_processes": {"B": ["ds"]},
                             "target_source": "s", "base_source": "s",
                             "sources": {"s": {"format": "cf", "store_root": self.root/"store", "reduction_dir": self.root/"red",
                                                "alignment_ok": {"ds": True}}},
                             "features": ["x"], "channels": {"all": ["2mu"]}, "channel_ids": {"2mu": 40},
                             "region_ids": {"dy": 3, "tt": 4}, "regions": ["dy", "tt"]})

    def tearDown(self): self.stack.close()

    def test_union_unit_without_weight_producer(self):
        out = load_side(self.cfg, "base", "all")
        assert_array_equal(out.x, [0, 1, 2])
        assert_array_equal(out.row, [0, 1, 2])
        assert_array_equal(out.weight_before, [1, 1, 1])
        self.assertFalse(out.event_id.duplicated().any())

    def test_nominal_and_reference_weights(self):
        self.write("store/cf.ProduceColumns/prod__event_weights_v1", pd.DataFrame({"w": [1., -2., 3., 4., 5.]}))
        self.write("store/cf.ProduceColumns/prod__dy_correction_weight_v1", pd.DataFrame({"r": [2.] * 5}))
        spec = {"mode": "product", "fields": ["w"]}
        self.cfg["base_weights"] = spec
        self.cfg["reference_stages"] = ["official"]
        self.cfg["base_processes"]["B"]["reference_weights"] = {"official": {"mode": "product", "fields": ["w", {"producer": "dy_correction_weight", "column": "r"}]}}
        out = load_side(self.cfg, "base", "all")
        assert_array_equal(out.weight_before, [1, -2, 3])
        assert_array_equal(out.weight_official, [2, -4, 6])

    def test_required_missing_weight_fails(self):
        self.cfg["base_weights"] = {"mode": "product", "fields": ["w"]}
        with self.assertRaises(KeyError): load_side(self.cfg, "base", "all")

    def test_optional_missing_warns(self):
        self.cfg["base_weights"] = {"mode": "product", "fields": [{"column": "w", "optional": True}]}
        with warnings.catch_warnings(record=True) as records:
            out = load_side(self.cfg, "base", "all")
        self.assertTrue(records)
        assert_array_equal(out.weight_before, np.ones(3))

    def test_misaligned_count_fails(self):
        self.write("store/cf.ProduceColumns/prod__event_weights_v1", pd.DataFrame({"w": [1., 2.]}))
        self.cfg["base_weights"] = {"mode": "product", "fields": ["w"]}
        with self.assertRaisesRegex(ValueError, "Misaligned"): load_side(self.cfg, "base", "all")

    def test_alignment_not_assumed(self):
        self.cfg["sources"]["s"]["alignment_ok"]["ds"] = False
        with self.assertRaisesRegex(ValueError, "verify alignment"): load_side(self.cfg, "base", "all")

    def test_selection_and_combined_channels(self):
        self.cfg["channels"]["all"].append("2e")
        self.cfg["channel_ids"]["2e"] = 30
        self.cfg["selections"] = {"x": [1, 4]}
        assert_array_equal(load_side(self.cfg, "target", "all").x, [1, 2])


if __name__ == "__main__": unittest.main()
