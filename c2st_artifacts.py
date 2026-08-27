from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def channel_dir(root: Path, channel: str) -> Path:
    return Path(root) / channel


def save_metadata(root: Path, metadata: dict) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_metadata(root: Path) -> dict:
    return json.loads((Path(root) / 'metadata.json').read_text())


def save_scaler(root: Path, channel: str, scaler) -> Path:
    out = channel_dir(root, channel)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'scaler.joblib'
    joblib.dump(scaler, path, compress=3)
    return path


def load_scaler(root: Path, channel: str):
    return joblib.load(channel_dir(root, channel) / 'scaler.joblib')


def save_test_fold(root: Path, channel: str, df: pd.DataFrame) -> Path:
    out = channel_dir(root, channel)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'test_fold.parquet'
    df.to_parquet(path, index=False, compression='zstd')
    return path


def load_test_fold(root: Path, channel: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(channel_dir(root, channel) / 'test_fold.parquet', columns=columns)


def save_stage_arrays(root: Path, channel: str, stage: str, **arrays) -> Path:
    out = channel_dir(root, channel)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{stage}_test.npz'
    # Test arrays are already compact float32/uint8; uncompressed npz loads much faster and
    # avoids CPU-heavy decompression for multi-million-row validation jobs.
    np.savez(path, **arrays)
    return path


def load_stage_arrays(root: Path, channel: str, stage: str):
    return np.load(channel_dir(root, channel) / f'{stage}_test.npz', mmap_mode='r')


def model_path(root: Path, channel: str, stage: str) -> Path:
    return channel_dir(root, channel) / f'model_{stage}.keras'


def save_stage_metrics(root: Path, channel: str, stage: str, metrics: dict) -> None:
    path = channel_dir(root, channel) / f'{stage}_metrics.json'
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def load_stage_metrics(root: Path, channel: str, stage: str) -> dict:
    return json.loads((channel_dir(root, channel) / f'{stage}_metrics.json').read_text())
