"""Historical Data/all-pre-DY-MC workflow, preserving normalization and numerics.
Run from the repository root. Existing c2st_config.py remains authoritative here.
"""
from copy import deepcopy
import c2st_config as old

# Kept explicit to avoid importing the HEP loader just to inspect this config.
NOMINAL_FIELDS = [
    "stitched_normalization_weight", "trigger_weight", "normalized_pu_weight",
    "muon_id_weight", "muon_iso_weight", "electron_weight", "electron_reco_weight",
    "normalized_ht_njet_nhf_btag_weight", "normalized_murmuf_envelope_weight",
    "normalized_mur_weight", "normalized_muf_weight", "normalized_pdf_weight",
    "normalized_isr_weight", "normalized_fsr_weight", "top_pt_theory_weight",
]
BASE_WEIGHTS = {"mode": "product", "fields": [
    {"producer": "event_weights", "column": f, "optional": True} for f in NOMINAL_FIELDS]}
base = deepcopy(old.MC_PROCESSES)
for entry in base.values():
    ref = deepcopy(BASE_WEIGHTS)
    if entry.get("is_dy", False):
        ref["fields"].append({"producer": "dy_correction_weight", "column": "dy_correction_weight", "optional": True})
    entry["reference_weights"] = {"dy": ref}

CONFIG = {
    "description": "Compatibility: Data vs all pre-DY MC",
    "target_processes": deepcopy(old.DATA_PROCESSES), "base_processes": base,
    "target_source": "legacy", "base_source": "legacy",
    "sources": {"legacy": {"format": "cf", "store_root": old.STORE_ROOT,
                            "reduction_dir": old.REDUCTION_DIR, "shift": old.SHIFT,
                            "alignment_ok": deepcopy(old.ALIGNMENT_OK)}},
    "target_weights": {"mode": "unit"}, "base_weights": BASE_WEIGHTS,
    "subtract_processes": [], "reference_stages": ["dy"],
    "region_ids": {"ar": 1, "dycr": 3, "ttcr": 4}, "regions": ["dycr"],
    "channel_ids": {"1e": 10, "1mu": 20, "2e": 30, "2mu": 40, "emu": 50, "ge3lep": 60},
    "channels": {c: [c] for c in old.CHANNELS},
    "features": list(old.FEATURES), "validation_vars": list(old.VALIDATION_VARS),
    "selections": deepcopy(old.SELECTIONS), "normalization": "legacy", "compatibility": True,
    "test_size": old.TEST_SIZE, "val_size": old.VAL_SIZE_WITHIN_TRAINVAL,
    "seed": old.RANDOM_STATE, "max_events_per_class": old.MAX_EVENTS_PER_CLASS,
    "epochs": old.EPOCHS, "batch_size": old.BATCH_SIZE,
    "early_stopping_patience": old.EARLY_STOPPING_PATIENCE,
    "reduce_lr_patience": old.REDUCE_LR_PATIENCE, "reduce_lr_factor": old.REDUCE_LR_FACTOR,
    "closure_model": {"hidden": old.HIDDEN, "batch_normalization": old.BATCH_NORMALIZATION,
                      "optimizer": old.OPTIMIZER, "learning_rate": old.LEARNING_RATE},
    "dctr_model": {"hidden": old.DCTR_HIDDEN, "batch_normalization": old.DCTR_BATCH_NORMALIZATION,
                   "optimizer": old.DCTR_OPTIMIZER, "learning_rate": old.DCTR_LEARNING_RATE},
}
