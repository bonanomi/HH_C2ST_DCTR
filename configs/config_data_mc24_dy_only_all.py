from copy import deepcopy
import runpy

# Reuse your existing model settings and training controls.
CONFIG = deepcopy(
    runpy.run_path("config_data25_over_data24_2mu.py")["CONFIG"]
)

# Reuse the verified 2024 input paths and process lists.
plot_cfg = runpy.run_path("config_plot_data_mc_years.py")["PLOT_CONFIG"]
year24 = deepcopy(plot_cfg["years"]["2024"])

# Repository's nominal MC product BEFORE the official DY correction.
from dctr_c2st_generic.examples.legacy_inclusive import BASE_WEIGHTS

def clean_processes(groups):
    result = {}
    for label, entry in groups.items():
        item = (
            {"datasets": list(entry)}
            if isinstance(entry, (list, tuple))
            else deepcopy(entry)
        )
        # All populations use the explicitly configured 2024 source/weights.
        for key in ("source", "weights", "reference_weights"):
            item.pop(key, None)
        result[label] = item
    return result

CONFIG.update({
    "description": "Data24 / inclusive pre-DY MC24; ptll and njet and mll",
    "sources": {"2024": year24["source"]},
    "target_source": "2024",
    "base_source": "2024",

    "target_processes": clean_processes(year24["data_processes"]),
    "base_processes": clean_processes(year24["mc_processes"]),

    "target_weights": {"mode": "unit"},
    "base_weights": deepcopy(BASE_WEIGHTS),

    # Inclusive: no background subtraction; correct every base component.
    "subtract_processes": [],
    "reference_stages": [],

    "channels": {"2mu": ["2mu"]},
    "channel_ids": deepcopy(plot_cfg["channel_ids"]),
    "regions": ["dycr"],
    "region_ids": deepcopy(plot_cfg["region_ids"]),
    "selections": deepcopy(plot_cfg.get("selections", {})),

    "features": ["mli_ll_pt", "mli_n_jet", "mli_mll"],
    "validation_vars": [],

    "normalization": "yield",
    "compatibility": False,
    "max_events_per_class": None,
    "folds": 5,
    "save_fold_models": True,
    "plots": True,
})

CONFIG["description"] = (
    "Data24 - non-DY MC24 versus DY24; ptll and njet and mll"
)

CONFIG["subtract_processes"] = [
    name
    for name, entry in CONFIG["base_processes"].items()
    if not entry.get("is_dy", False)
]

assert any(
    entry.get("is_dy", False)
    for entry in CONFIG["base_processes"].values()
), "No DY processes identified: check is_dy flags"

print("Subtracted processes:", CONFIG["subtract_processes"])
