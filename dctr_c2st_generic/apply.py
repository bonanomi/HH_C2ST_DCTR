"""Apply an exported model to independent events, retaining physical weight signs."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from .training import transform
from .numerics import odds, reweight_signed


class DCTRReweighter:
    """Load a trusted inference bundle. predict() returns positive target/base factors.

    For component-only corrections, supply process labels from the training config,
    or an explicit boolean mask when transferring to a differently named sample.
    Passing a mask is an explicit physics choice; it does not invert the factor.
    """
    def __init__(self, bundle):
        import tensorflow as tf
        self.bundle = Path(bundle)
        self.metadata = json.loads((self.bundle / "metadata.json").read_text())
        self.scaler = joblib.load(self.bundle / "scaler.joblib")
        self.model = tf.keras.models.load_model(self.bundle / "model.keras", compile=False)

    def predict(self, frame, *, processes=None, mask=None):
        if processes is not None and mask is not None:
            raise ValueError("Supply process labels OR a mask")
        if processes is not None:
            processes = np.asarray(processes)
            if processes.shape != (len(frame),):
                raise ValueError("Process-label length mismatch")
            known = self.metadata["corrected_processes"] + self.metadata["subtract_processes"]
            if not np.all(np.isin(processes, known)):
                raise ValueError("Unknown process labels; specify an explicit transfer mask")
            mask = np.isin(processes, self.metadata["corrected_processes"])
        elif mask is None:
            if self.metadata["subtract_processes"]:
                raise ValueError("Component-only inference requires process labels or an explicit mask")
            mask = np.ones(len(frame), bool)
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(frame),):
            raise ValueError("mask must be a boolean array with one entry per event")
        result = np.ones(len(frame), dtype=np.float32)
        if not mask.any():
            return result
        features = self.metadata["features"]
        selected = frame.loc[mask, features]
        if not np.all(np.isfinite(selected.to_numpy())):
            raise ValueError("Nonfinite inference features")
        x = transform(selected, self.scaler, features)
        p = self.model.predict(x, batch_size=self.metadata["batch_size"], verbose=0).reshape(-1)
        f = odds(p, self.metadata["eps"])
        cap = self.metadata["cap"]
        if cap is not None:
            f = np.minimum(f, cap)
        result[mask] = f.astype(np.float32)
        return result

    def reweight(self, frame, weights, **kwargs):
        return reweight_signed(weights, self.predict(frame, **kwargs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--weight-column", help="Physical signed weight; omitted means unit weights")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--process-column")
    scope.add_argument("--all-events", action="store_true", help="Explicit transfer to every supplied event")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_parquet(args.input)
    if {"dctr_factor", "weight_dctr"} & set(frame):
        raise ValueError("Input already has DCTR columns; avoid accidental double application")
    model = DCTRReweighter(args.bundle)
    kwargs = {"processes": frame[args.process_column].to_numpy()} if args.process_column else {}
    if args.all_events:
        kwargs = {"mask": np.ones(len(frame), bool)}
    factor = model.predict(frame, **kwargs)
    weights = frame[args.weight_column].to_numpy() if args.weight_column else np.ones(len(frame))
    frame["dctr_factor"], frame["weight_dctr"] = factor, reweight_signed(weights, factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
