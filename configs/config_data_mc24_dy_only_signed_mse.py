"""Start from the actual completed run, including its verified source paths."""
import json
from pathlib import Path

previous = Path("c2st_artifacts/data_mc24_training_full/config.json")
CONFIG = json.loads(previous.read_text())
base = CONFIG["base_processes"]
dy = {p for p, entry in base.items() if entry.get("is_dy", False)}
if not dy or set(base) - set(CONFIG["subtract_processes"]) != dy:
    raise ValueError("Expected a DY-only baseline with explicit is_dy flags")
CONFIG.update(
    description="Data24 minus non-DY vs signed pre-DY DY24; ptll and njet",
    dctr_objective="signed_mse", normalization="yield", compatibility=False,
    max_events_per_class=None, features=["mli_ll_pt", "mli_n_jet"],
    validation_vars=["mli_met_pt", "mli_mll"], channels={"2mu": ["2mu"]},
    regions=["dycr"], folds=5,
)
