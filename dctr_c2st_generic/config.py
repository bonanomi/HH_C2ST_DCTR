"""Configuration loading and validation; no TensorFlow or HEP imports required."""
from copy import deepcopy
from pathlib import Path
import runpy
import json
import re
import numpy as np

DEFAULTS = {
    "subtract_processes": [], "reference_stages": [],
    "regions": None, "region_ids": {}, "channel_ids": {}, "channels": {},
    "selections": {}, "validation_vars": [], "scaler": "hep",
    "normalization": "shape", "compatibility": False,
    "test_size": .30, "val_size": .15, "seed": 0, "folds": 5,
    "cap_quantile": .995, "eps": 1e-6, "max_events_per_class": None,
    "epochs": 50, "batch_size": 8192, "early_stopping_patience": 5,
    "reduce_lr_patience": 2, "reduce_lr_factor": .2,
    "save_fold_models": False, "plots": True, "plot_bins": 40,
    "dctr_model": {"hidden": [50], "batch_normalization": True,
                   "optimizer": "sgd", "learning_rate": .005},
    "closure_model": {"hidden": [50], "batch_normalization": False,
                      "optimizer": "adam", "learning_rate": .001},
}


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe artifact name: {value!r}")
    return value


def process_entries(processes):
    """Both historical list-valued and dict-valued process maps are accepted."""
    result = {}
    for label, entry in processes.items():
        item = {"datasets": entry} if isinstance(entry, (list, tuple)) else dict(entry)
        if not item.get("datasets") or isinstance(item["datasets"], str):
            raise ValueError(f"{label}: datasets must be a nonempty list")
        result[label] = item
    return result


def validate(user):
    cfg = deepcopy(DEFAULTS)
    unknown = set(user) - set(DEFAULTS) - {
        "target_processes", "base_processes", "target_source", "base_source",
        "target_weights", "base_weights", "sources", "features", "description",
    }
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg.update(deepcopy(user))
    for key in ("target_processes", "base_processes", "features", "sources", "channels"):
        if not cfg.get(key):
            raise ValueError(f"Specify nonempty {key}")
    for side in ("target", "base"):
        cfg[f"{side}_processes"] = process_entries(cfg[f"{side}_processes"])
        if cfg.get(f"{side}_source") not in cfg["sources"]:
            raise ValueError(f"Specify {side}_source from sources")
        # No inference of physical Data/MC identity from class role.
        cfg.setdefault(f"{side}_weights", {"mode": "unit"})
        for entry in cfg[f"{side}_processes"].values():
            if entry.get("source", cfg[f"{side}_source"]) not in cfg["sources"]:
                raise ValueError("Unknown per-process source")
    sub = cfg["subtract_processes"]
    if len(set(sub)) != len(sub) or set(sub) - set(cfg["base_processes"]):
        raise ValueError("subtract_processes must be unique base-process names")
    if len(sub) == len(cfg["base_processes"]):
        raise ValueError("At least one base process must remain corrected")
    if cfg["normalization"] not in ("shape", "yield", "legacy"):
        raise ValueError("normalization must be shape, yield, or legacy")
    if cfg["compatibility"] != (cfg["normalization"] == "legacy"):
        raise ValueError("compatibility=True requires normalization='legacy' and vice versa")
    if cfg["scaler"] not in ("hep", "robust", "standard"):
        raise ValueError("scaler must be hep, robust, or standard")
    if cfg["compatibility"] and cfg["scaler"] != "hep":
        raise ValueError("Legacy compatibility requires the historical hep scaler")
    if not 0 < cfg["test_size"] < 1 or not 0 < cfg["val_size"] < 1:
        raise ValueError("Split fractions must lie in (0,1)")
    if not isinstance(cfg["folds"], int) or not 2 <= cfg["folds"] < 32767:
        raise ValueError("folds must be an integer in [2,32766]")
    if not 0 < cfg["eps"] < .5:
        raise ValueError("eps must lie in (0,.5)")
    q = cfg["cap_quantile"]
    if q is not None and not 0 < q <= 1:
        raise ValueError("cap_quantile must be None or in (0,1]")
    for name, members in cfg["channels"].items():
        safe_name(name)
        if not members or isinstance(members, str):
            raise ValueError("channels maps output names to lists of physical channels")
    for stage in cfg["reference_stages"]:
        safe_name(stage)
        if stage in ("before", "dctr"):
            raise ValueError("before and dctr are reserved stage names")
    if len(set(cfg["reference_stages"])) != len(cfg["reference_stages"]):
        raise ValueError("Duplicate reference stages")
    if cfg["regions"] is not None:
        if not cfg["regions"] or set(cfg["regions"]) - set(cfg["region_ids"]):
            raise ValueError("regions must be None or nonempty names from region_ids")
    members = {c for values in cfg["channels"].values() for c in values}
    if any(s.get("format", "cf") == "cf" for s in cfg["sources"].values()):
        if members - set(cfg["channel_ids"]):
            raise ValueError("All CF channels need channel_ids")
    if len(set(cfg["features"])) != len(cfg["features"]):
        raise ValueError("Duplicate features")
    for feature, bounds in cfg["selections"].items():
        if len(bounds) != 2 or (bounds[0] is not None and bounds[1] is not None and bounds[0] >= bounds[1]):
            raise ValueError(f"Invalid selection for {feature}")
    for key in ("epochs", "batch_size", "plot_bins"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["max_events_per_class"] is not None and cfg["max_events_per_class"] < 2:
        raise ValueError("max_events_per_class must be None or >=2")
    if cfg["max_events_per_class"] is not None and not cfg["compatibility"]:
        raise ValueError("Generic mode requires max_events_per_class=None: independent class truncation changes yield/subtraction normalization")
    for name in ("dctr_model", "closure_model"):
        model = cfg[name]
        if set(model) != {"hidden", "batch_normalization", "optimizer", "learning_rate"}:
            raise ValueError(f"{name}: specify hidden, batch_normalization, optimizer, learning_rate")
        if any(int(n) != n or n < 1 for n in model["hidden"]) or model["optimizer"] not in ("adam", "sgd") or model["learning_rate"] <= 0:
            raise ValueError(f"Invalid {name}")
    return cfg


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text()) if path.suffix == ".json" else runpy.run_path(str(path))["CONFIG"]
    return validate(raw)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False) + "\n")
