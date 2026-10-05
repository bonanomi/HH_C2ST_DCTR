"""Create toy parquet inputs/config for a real TensorFlow smoke run.
python -m dctr_c2st_generic.examples.toy --output /tmp/dctr_toy
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from dctr_c2st_generic.config import write_json


def create_toy(output, n=3000):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(71)
    for name, shift, rows in (("target", .3, n), ("base", 0., n)):
        pd.DataFrame({"x": rng.normal(shift, 1., rows), "z": rng.normal(0, 1, rows),
                      "channel": "all", "region": "toy", "physical_weight": 1.2 if name == "target" else 1.}).to_parquet(output / f"{name}.parquet", index=False)
    cfg = {
        "target_processes": {"target": ["target.parquet"]}, "base_processes": {"base": ["base.parquet"]},
        "target_source": "toy", "base_source": "toy",
        "sources": {"toy": {"format": "parquet", "root": str(output)}},
        "target_weights": {"mode": "product", "fields": ["physical_weight"]},
        "base_weights": {"mode": "product", "fields": ["physical_weight"]},
        "features": ["x", "z"], "channels": {"all": ["all"]},
        "region_ids": {"toy": 1}, "regions": ["toy"], "scaler": "standard",
        "folds": 3, "epochs": 8, "batch_size": 256,
        "normalization": "yield", "cap_quantile": .995,
    }
    write_json(output / "config.json", cfg)
    return output / "config.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    print(create_toy(args.output))
