"""EDIT paths, datasets, exposure scales and alignment declarations before use.
No paths, 2025 dataset names, or luminosities are guessed.
"""
import c2st_config as old

# Fill these with your actual dataset identifiers (not file paths).
DATA24 = {"Data24": ["REPLACE_WITH_2024_DATASET"]}
DATA25 = {"Data25": ["REPLACE_WITH_2025_DATASET"]}

# 1 means raw event yields. For an exposure-normalized rate correction, set
# target scale = L_ref/L25 and base scale = L_ref/L24, using compatible units.
# Equal values cancel in odds. Choose L_ref so weights remain numerically useful.
TARGET_EXPOSURE_SCALE = 1.0
BASE_EXPOSURE_SCALE = 1.0

CONFIG = {
    "description": "Data25 / Data24; transfer hypothesis for MC25",
    "target_processes": DATA25, "base_processes": DATA24,
    "target_source": "2025", "base_source": "2024",
    "sources": {
        "2024": {"format": "cf", "store_root": "/REPLACE/2024/cf_store",
                 "reduction_dir": "/REPLACE/2024/cf.MergeReducedEvents", "shift": "nominal",
                 # Set entries True only after checking positional alignment.
                 "alignment_ok": {d: False for ds in DATA24.values() for d in ds}},
        "2025": {"format": "cf", "store_root": "/REPLACE/2025/cf_store",
                 "reduction_dir": "/REPLACE/2025/cf.MergeReducedEvents", "shift": "nominal",
                 "alignment_ok": {d: False for ds in DATA25.values() for d in ds}},
    },
    "target_weights": {"mode": "unit", "scale": TARGET_EXPOSURE_SCALE},
    "base_weights": {"mode": "unit", "scale": BASE_EXPOSURE_SCALE},
    "subtract_processes": [], "reference_stages": [],
    "region_ids": {"ar": 1, "dycr": 3, "ttcr": 4}, "regions": ["dycr"],
    # Example region union: ["dycr", "ttcr"]. Each row is loaded only once.
    "channel_ids": {"2e": 30, "2mu": 40, "emu": 50},
    "channels": {"2mu": ["2mu"], "2e": ["2e"]},
    # To pool channels into one model: {"combined": ["2mu", "2e"]}.
    "features": list(old.FEATURES), "validation_vars": [], "selections": {},
    "normalization": "yield", "compatibility": False,
    "dctr_model": {"hidden": [50], "batch_normalization": True,
                   "optimizer": "sgd", "learning_rate": .005},
    "closure_model": {"hidden": [50], "batch_normalization": False,
                      "optimizer": "adam", "learning_rate": .001},
}
