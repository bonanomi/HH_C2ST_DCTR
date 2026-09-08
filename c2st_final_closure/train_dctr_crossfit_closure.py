from __future__ import annotations

"""Cross-fitted DCTR derivation followed by an independent closure C2ST.

Two DCTR targets are supported:

``inclusive`` (the historical/default mode)
    class 1 = Data, class 0 = all positive-weight pre-DY MC.  The learned
    factor is applied to every MC event.

``dy_only``
    class 1 = Data - non-DY MC, class 0 = DY MC.  Non-DY MC enters the
    target class with a negative subtraction weight and only DY events receive
    a learned DCTR factor.  Generator-level negative-weight MC events remain
    excluded from NN training, exactly as in the historical closure workflow.

Both modes use the same outer train/validation/test split, k-fold cross-fitting,
validation-derived DCTR cap, and fresh closure classifiers.  The untouched
outer test is never used to fit the scaler, a DCTR model, a DCTR cap, or a
closure classifier.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c2st_config as cfg
import dyvr_lib
from c2st_models import build_binary_classifier
from c2st_core import (
    apply_scaler,
    fit_scaler,
    normalize_signed_sample_weights,
    split_class_indices,
    stage_weights,
    weighted_bce,
)


DEFAULT_OUT = cfg.ARTIFACT_DIR / "dctr_crossfit_closure"
DCTR_TARGETS = ("inclusive", "dy_only")


def target_root(base: Path, dctr_target: str) -> Path:
    """Keep the historical inclusive artifact path backwards compatible."""
    return base if dctr_target == "inclusive" else base / dctr_target



def build_standard_model(n_features: int, seed: int) -> tf.keras.Model:
    """Build the fresh C2ST/closure classifier from the standard NN config."""
    return build_binary_classifier(
        n_features,
        hidden=cfg.HIDDEN,
        optimizer=cfg.OPTIMIZER,
        learning_rate=cfg.LEARNING_RATE,
        batch_normalization=cfg.BATCH_NORMALIZATION,
        seed=seed,
    )


def build_dctr_model(n_features: int, seed: int) -> tf.keras.Model:
    """Build a DCTR derivation classifier from the DCTR-specific config."""
    return build_binary_classifier(
        n_features,
        hidden=cfg.DCTR_HIDDEN,
        optimizer=cfg.DCTR_OPTIMIZER,
        learning_rate=cfg.DCTR_LEARNING_RATE,
        batch_normalization=cfg.DCTR_BATCH_NORMALIZATION,
        seed=seed,
    )

def callbacks(verbose: int = 1):
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.REDUCE_LR_FACTOR,
            patience=cfg.REDUCE_LR_PATIENCE,
            min_lr=1e-5,
            verbose=verbose,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.EARLY_STOPPING_PATIENCE,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=verbose,
        ),
    ]


def channel_tables(tables: dict[str, pd.DataFrame], channel: str):
    """Concatenate one channel while preserving whether each MC row is DY."""
    data_parts = []
    mc_parts = []
    for label, df in tables.items():
        if not len(df):
            continue
        if label in cfg.DATA_PROCESSES:
            data_parts.append(df)
        elif label in cfg.MC_PROCESSES:
            tmp = df.copy()
            tmp["is_dy"] = bool(cfg.MC_PROCESSES[label].get("is_dy", False))
            mc_parts.append(tmp)

    data = pd.concat(data_parts, ignore_index=True) if data_parts else pd.DataFrame()
    mc = pd.concat(mc_parts, ignore_index=True) if mc_parts else pd.DataFrame()
    if len(data):
        data = data.loc[data["channel"] == channel].reset_index(drop=True)
    if len(mc):
        mc = mc.loc[mc["channel"] == channel].reset_index(drop=True)
    return data, mc


def maybe_subsample(df: pd.DataFrame, maximum: int | None, seed: int):
    if maximum is None or len(df) <= maximum:
        return df
    return df.sample(n=maximum, random_state=seed).reset_index(drop=True)


def make_pair(x_data, idx_data, x_mc, idx_mc):
    """Materialize one Data+MC NN matrix for the requested class indices."""
    xd = x_data[idx_data]
    xm = x_mc[idx_mc]
    x = np.concatenate([xd, xm], axis=0).astype(np.float32, copy=False)
    y = np.concatenate([
        np.ones(len(idx_data), dtype=np.uint8),
        np.zeros(len(idx_mc), dtype=np.uint8),
    ])
    return x, y


def balanced_weights(n_data_total: int, raw_mc_all, raw_mc_subset, n_data_subset: int):
    return stage_weights(n_data_total, raw_mc_all, raw_mc_subset, n_data_subset)


def fit_binary_model(
    x_data,
    idx_data_train,
    idx_data_val,
    x_mc,
    idx_mc_train,
    idx_mc_val,
    raw_mc_all,
    seed: int,
    label: str,
):
    """Historical inclusive Data-vs-all-MC DCTR/closure fit."""
    x_train, y_train = make_pair(x_data, idx_data_train, x_mc, idx_mc_train)
    x_val, y_val = make_pair(x_data, idx_data_val, x_mc, idx_mc_val)

    wd_train, wm_train = balanced_weights(
        len(x_data), raw_mc_all, raw_mc_all[idx_mc_train], len(idx_data_train)
    )
    wd_val, wm_val = balanced_weights(
        len(x_data), raw_mc_all, raw_mc_all[idx_mc_val], len(idx_data_val)
    )
    w_train = np.concatenate([wd_train, wm_train]).astype(np.float32, copy=False)
    w_val = np.concatenate([wd_val, wm_val]).astype(np.float32, copy=False)

    model = build_dctr_model(x_train.shape[1], seed)
    print(
        f"=== {label}: train={len(y_train):_}, val={len(y_val):_}, "
        f"batch={cfg.BATCH_SIZE:_}, optimizer={cfg.DCTR_OPTIMIZER}, "
        f"lr={cfg.DCTR_LEARNING_RATE:g}, hidden={tuple(cfg.DCTR_HIDDEN)}, "
        f"batchnorm={cfg.DCTR_BATCH_NORMALIZATION} ==="
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        validation_data=(x_val, y_val, w_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks(),
        verbose=2,
    )

    del x_train, y_train, x_val, y_val, w_train, w_val
    del wd_train, wm_train, wd_val, wm_val
    gc.collect()
    return model


def make_dy_only_sample(
    x_data,
    idx_data,
    x_mc,
    idx_non_dy,
    idx_dy,
    raw_before,
):
    """Build (Data - non-DY MC) vs DY with signed target weights.

    Label 1 is the target distribution.  Data has +1 weight while non-DY MC
    has -weight_uncorrected.  Label 0 is DY with +weight_uncorrected.
    A single common normalization rescales the signed weights so mean |w|=1;
    this does not change the signed BCE optimum but keeps the numerical loss
    scale stable.
    """
    x = np.concatenate([
        x_data[idx_data],
        x_mc[idx_non_dy],
        x_mc[idx_dy],
    ], axis=0).astype(np.float32, copy=False)
    y = np.concatenate([
        np.ones(len(idx_data) + len(idx_non_dy), dtype=np.uint8),
        np.zeros(len(idx_dy), dtype=np.uint8),
    ])
    raw_signed = np.concatenate([
        np.ones(len(idx_data), dtype=np.float64),
        -np.asarray(raw_before[idx_non_dy], dtype=np.float64),
        np.asarray(raw_before[idx_dy], dtype=np.float64),
    ])
    w = normalize_signed_sample_weights(raw_signed).astype(np.float32, copy=False)
    return x, y, w


def fit_dy_only_model(
    x_data,
    idx_data_train,
    idx_data_val,
    x_mc,
    idx_non_dy_train,
    idx_non_dy_val,
    idx_dy_train,
    idx_dy_val,
    raw_before,
    seed: int,
    label: str,
):
    """Fit DY vs (Data - non-DY) using signed sample weights."""
    x_train, y_train, w_train = make_dy_only_sample(
        x_data, idx_data_train, x_mc, idx_non_dy_train, idx_dy_train, raw_before
    )
    x_val, y_val, w_val = make_dy_only_sample(
        x_data, idx_data_val, x_mc, idx_non_dy_val, idx_dy_val, raw_before
    )

    model = build_dctr_model(x_train.shape[1], seed)
    print(
        f"=== {label}: train={len(y_train):_}, val={len(y_val):_}, "
        f"batch={cfg.BATCH_SIZE:_}, signed target weights, "
        f"optimizer={cfg.DCTR_OPTIMIZER}, lr={cfg.DCTR_LEARNING_RATE:g}, "
        f"hidden={tuple(cfg.DCTR_HIDDEN)}, "
        f"batchnorm={cfg.DCTR_BATCH_NORMALIZATION} ==="
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        validation_data=(x_val, y_val, w_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks(),
        verbose=2,
    )

    del x_train, y_train, w_train, x_val, y_val, w_val
    gc.collect()
    return model


def dctr_from_probability(p, eps: float):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return p / (1.0 - p)


def cap_from_validation(model, x_mc, idx_mc_val, quantile, eps):
    if quantile is None or len(idx_mc_val) == 0:
        return None
    p = model.predict(x_mc[idx_mc_val], batch_size=cfg.BATCH_SIZE, verbose=0).reshape(-1)
    factors = dctr_from_probability(p, eps)
    finite = factors[np.isfinite(factors) & (factors >= 0)]
    if not len(finite):
        return None
    return float(np.quantile(finite, quantile))


def predict_dctr(model, x_mc, indices, cap_value, eps):
    if len(indices) == 0:
        return np.empty(0, dtype=np.float32)
    p = model.predict(x_mc[indices], batch_size=cfg.BATCH_SIZE, verbose=0).reshape(-1)
    factor = dctr_from_probability(p, eps)
    if cap_value is not None:
        factor = np.minimum(factor, cap_value)
    return factor.astype(np.float32)


def shuffled_folds(indices: np.ndarray, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(shuffled, n_folds)]


def inner_train_val(indices: np.ndarray, val_fraction: float, seed: int, allow_empty: bool = False):
    indices = np.asarray(indices, dtype=np.int64).copy()
    if len(indices) == 0:
        if allow_empty:
            return indices, indices
        raise ValueError("Cannot split an empty population")
    if len(indices) == 1:
        if allow_empty:
            return indices, np.empty(0, dtype=np.int64)
        raise ValueError("Need at least two events for an internal train/validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(val_fraction * len(indices))))
    if n_val >= len(indices):
        n_val = len(indices) - 1
    return indices[n_val:], indices[:n_val]


def _concat_other_folds(folds, held_out: int):
    pieces = [folds[j] for j in range(len(folds)) if j != held_out and len(folds[j])]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def crossfit_dctr_trainval_inclusive(
    x_data,
    data_trainval,
    x_mc,
    mc_trainval,
    raw_before,
    n_folds: int,
    cap_quantile: float | None,
    eps: float,
    seed: int,
    save_fold_models_dir: Path | None = None,
):
    """Historical out-of-fold Data-vs-all-MC factors."""
    data_folds = shuffled_folds(data_trainval, n_folds, seed)
    mc_folds = shuffled_folds(mc_trainval, n_folds, seed + 1000)
    factors = np.full(len(x_mc), np.nan, dtype=np.float32)
    fold_id = np.full(len(x_mc), -1, dtype=np.int16)
    cap_values = []

    for k in range(n_folds):
        hold_m = mc_folds[k]
        cand_d = _concat_other_folds(data_folds, k)
        cand_m = _concat_other_folds(mc_folds, k)

        train_d, val_d = inner_train_val(cand_d, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 10 * k + 1)
        train_m, val_m = inner_train_val(cand_m, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 10 * k + 2)

        model = fit_binary_model(
            x_data, train_d, val_d,
            x_mc, train_m, val_m,
            raw_before,
            seed + k,
            label=f"DCTR inclusive cross-fit fold {k + 1}/{n_folds}",
        )
        cap_value = cap_from_validation(model, x_mc, val_m, cap_quantile, eps)
        fold_factor = predict_dctr(model, x_mc, hold_m, cap_value, eps)
        factors[hold_m] = fold_factor
        fold_id[hold_m] = k
        cap_values.append(cap_value)
        print(
            f"  fold {k + 1}: held-out MC={len(hold_m):_}, "
            f"cap={cap_value if cap_value is not None else 'none'}, "
            f"factor mean={float(np.mean(fold_factor)):.5g}, max={float(np.max(fold_factor)):.5g}"
        )
        if save_fold_models_dir is not None:
            save_fold_models_dir.mkdir(parents=True, exist_ok=True)
            model.save(save_fold_models_dir / f"dctr_fold_{k}.keras")
        del model, cand_d, cand_m, train_d, val_d, train_m, val_m, fold_factor
        tf.keras.backend.clear_session()
        gc.collect()

    if np.any(~np.isfinite(factors[mc_trainval])):
        missing = int(np.sum(~np.isfinite(factors[mc_trainval])))
        raise RuntimeError(f"Cross-fitting failed to assign DCTR factors to {missing} train/val MC rows")
    return factors, fold_id, cap_values


def crossfit_dctr_trainval_dy_only(
    x_data,
    data_trainval,
    x_mc,
    mc_trainval,
    is_dy,
    raw_before,
    n_folds: int,
    cap_quantile: float | None,
    eps: float,
    seed: int,
    save_fold_models_dir: Path | None = None,
):
    """Out-of-fold factors for DY using (Data - non-DY MC) as the target."""
    dy_trainval = mc_trainval[is_dy[mc_trainval]]
    non_dy_trainval = mc_trainval[~is_dy[mc_trainval]]
    if len(dy_trainval) < n_folds * 2 or len(data_trainval) < n_folds * 2:
        raise ValueError("Not enough Data/DY events for the requested number of DY-only folds")

    data_folds = shuffled_folds(data_trainval, n_folds, seed)
    dy_folds = shuffled_folds(dy_trainval, n_folds, seed + 1000)
    non_dy_folds = shuffled_folds(non_dy_trainval, n_folds, seed + 2000)

    # Non-DY events are deliberately left at factor 1.  Only DY receives a learned correction.
    factors = np.ones(len(x_mc), dtype=np.float32)
    fold_id = np.full(len(x_mc), -1, dtype=np.int16)
    cap_values = []

    for k in range(n_folds):
        hold_dy = dy_folds[k]
        cand_d = _concat_other_folds(data_folds, k)
        cand_dy = _concat_other_folds(dy_folds, k)
        cand_non = _concat_other_folds(non_dy_folds, k)

        train_d, val_d = inner_train_val(cand_d, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 10 * k + 1)
        train_dy, val_dy = inner_train_val(cand_dy, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 10 * k + 2)
        train_non, val_non = inner_train_val(
            cand_non, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 10 * k + 3, allow_empty=True
        )

        model = fit_dy_only_model(
            x_data, train_d, val_d,
            x_mc, train_non, val_non, train_dy, val_dy,
            raw_before,
            seed + k,
            label=f"DCTR DY-only cross-fit fold {k + 1}/{n_folds}",
        )
        cap_value = cap_from_validation(model, x_mc, val_dy, cap_quantile, eps)
        fold_factor = predict_dctr(model, x_mc, hold_dy, cap_value, eps)
        factors[hold_dy] = fold_factor
        fold_id[hold_dy] = k
        cap_values.append(cap_value)
        print(
            f"  fold {k + 1}: held-out DY={len(hold_dy):_}, "
            f"cap={cap_value if cap_value is not None else 'none'}, "
            f"factor mean={float(np.mean(fold_factor)):.5g}, max={float(np.max(fold_factor)):.5g}"
        )
        if save_fold_models_dir is not None:
            save_fold_models_dir.mkdir(parents=True, exist_ok=True)
            model.save(save_fold_models_dir / f"dctr_fold_{k}.keras")
        del model, cand_d, cand_dy, cand_non, train_d, val_d, train_dy, val_dy, train_non, val_non
        del fold_factor
        tf.keras.backend.clear_session()
        gc.collect()

    if np.any(~np.isfinite(factors[dy_trainval])):
        missing = int(np.sum(~np.isfinite(factors[dy_trainval])))
        raise RuntimeError(f"DY-only cross-fitting failed to assign {missing} train/val DY factors")
    return factors, fold_id, cap_values


def fit_final_dctr_for_outer_test_inclusive(
    x_data, data_trainval, x_mc, mc_trainval, mc_test, raw_before,
    cap_quantile, eps, seed,
):
    train_d, val_d = inner_train_val(data_trainval, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 3001)
    train_m, val_m = inner_train_val(mc_trainval, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 3002)
    model = fit_binary_model(
        x_data, train_d, val_d,
        x_mc, train_m, val_m,
        raw_before,
        seed + 3000,
        label="DCTR inclusive final model for untouched outer test",
    )
    cap_value = cap_from_validation(model, x_mc, val_m, cap_quantile, eps)
    factor_test = predict_dctr(model, x_mc, mc_test, cap_value, eps)
    return model, factor_test, cap_value


def fit_final_dctr_for_outer_test_dy_only(
    x_data, data_trainval, x_mc, mc_trainval, mc_test, is_dy, raw_before,
    cap_quantile, eps, seed,
):
    dy_trainval = mc_trainval[is_dy[mc_trainval]]
    non_trainval = mc_trainval[~is_dy[mc_trainval]]
    dy_test = mc_test[is_dy[mc_test]]

    train_d, val_d = inner_train_val(data_trainval, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 3001)
    train_dy, val_dy = inner_train_val(dy_trainval, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 3002)
    train_non, val_non = inner_train_val(
        non_trainval, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 3003, allow_empty=True
    )
    model = fit_dy_only_model(
        x_data, train_d, val_d,
        x_mc, train_non, val_non, train_dy, val_dy,
        raw_before,
        seed + 3000,
        label="DCTR DY-only final model for untouched outer test",
    )
    cap_value = cap_from_validation(model, x_mc, val_dy, cap_quantile, eps)
    factor_test_dy = predict_dctr(model, x_mc, dy_test, cap_value, eps)
    return model, dy_test, factor_test_dy, cap_value


def train_closure_stage(
    channel: str,
    stage: str,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    raw_mc_all,
    raw_mc_train,
    raw_mc_val,
    raw_mc_test,
    n_data_total: int,
    n_data_train: int,
    n_data_val: int,
    n_data_test: int,
    output_dir: Path,
    seed: int,
):
    """Train one fresh closure C2ST and evaluate only on the untouched outer test."""
    wd_train, wm_train = stage_weights(n_data_total, raw_mc_all, raw_mc_train, n_data_train)
    wd_val, wm_val = stage_weights(n_data_total, raw_mc_all, raw_mc_val, n_data_val)
    wd_test, wm_test = stage_weights(n_data_total, raw_mc_all, raw_mc_test, n_data_test)
    w_train = np.concatenate([wd_train, wm_train]).astype(np.float32, copy=False)
    w_val = np.concatenate([wd_val, wm_val]).astype(np.float32, copy=False)
    w_test = np.concatenate([wd_test, wm_test]).astype(np.float32, copy=False)

    model = build_standard_model(x_train.shape[1], seed)
    print(
        f"=== closure [{channel}, {stage}]: optimizer={cfg.OPTIMIZER}, "
        f"lr={cfg.LEARNING_RATE:g}, hidden={tuple(cfg.HIDDEN)}, "
        f"batchnorm={cfg.BATCH_NORMALIZATION} ==="
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        validation_data=(x_val, y_val, w_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks(),
        verbose=2,
    )
    p_test = model.predict(x_test, batch_size=cfg.BATCH_SIZE, verbose=0).reshape(-1).astype(np.float32)
    auc = float(roc_auc_score(y_test, p_test, sample_weight=w_test))
    bce = weighted_bce(y_test, p_test, w_test)

    model.save(output_dir / f"closure_model_{stage}.keras")
    np.savez(output_dir / f"closure_{stage}_test.npz", p_test=p_test, w_test=w_test)
    metrics = {
        "channel": channel,
        "stage": stage,
        "auc": auc,
        "distance_from_half": abs(auc - 0.5),
        "weighted_bce": bce,
        "n_test": int(len(y_test)),
    }
    (output_dir / f"closure_{stage}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{channel}, {stage}] AUC={auc:.6f}, |AUC-0.5|={abs(auc-0.5):.6f}")

    del model, p_test, w_train, w_val, w_test
    del wd_train, wm_train, wd_val, wm_val, wd_test, wm_test
    tf.keras.backend.clear_session()
    gc.collect()
    return metrics


def make_outer_test_fold(data_df, test_d, mc_df, test_m, dctr_factor_test):
    cols = cfg.LOAD_FEATURES
    d = data_df.iloc[test_d][cols].copy()
    d.insert(0, "y", np.ones(len(d), dtype=np.uint8))
    d["is_dy"] = False
    d["weight_uncorrected"] = np.float32(1.0)
    d["weight"] = np.float32(1.0)
    d["dctr_factor"] = np.float32(1.0)
    d["weight_dctr"] = np.float32(1.0)

    m = mc_df.iloc[test_m][cols + ["weight_uncorrected", "weight", "is_dy"]].copy()
    m.insert(0, "y", np.zeros(len(m), dtype=np.uint8))
    m["dctr_factor"] = dctr_factor_test.astype(np.float32, copy=False)
    m["weight_dctr"] = (
        m["weight_uncorrected"].to_numpy(dtype=np.float32, copy=False)
        * dctr_factor_test.astype(np.float32, copy=False)
    )
    return pd.concat([d, m], ignore_index=True)


def print_dy_only_composition(channel: str, data_df: pd.DataFrame, mc_df: pd.DataFrame):
    w = mc_df["weight_uncorrected"].to_numpy(dtype=np.float64, copy=False)
    is_dy = mc_df["is_dy"].to_numpy(dtype=bool, copy=False)
    dy_sum = float(w[is_dy].sum())
    non_sum = float(w[~is_dy].sum())
    target = float(len(data_df) - non_sum)
    total_mc = float(w.sum())
    dy_purity = dy_sum / total_mc if total_mc > 0 else np.nan
    print("\nDCTR DY-only target composition")
    print(f"  Data:        rows={len(data_df):_}, target weight={float(len(data_df)):.6g}")
    print(f"  DY MC:       rows={int(is_dy.sum()):_}, sumw={dy_sum:.6g}, sum|w|={float(np.abs(w[is_dy]).sum()):.6g}")
    print(f"  non-DY MC:   rows={int((~is_dy).sum()):_}, subtraction sumw={non_sum:.6g}, sum|w|={float(np.abs(w[~is_dy]).sum()):.6g}")
    print(f"  Data-nonDY:  effective signed target yield={target:.6g}")
    print(f"  DY fraction of positive-weight MC yield={dy_purity:.3%}")
    if target <= 0:
        print("  WARNING: integrated Data-nonDY target is non-positive; inspect the subtraction before interpreting DCTR.")
    return {
        "data_rows": int(len(data_df)),
        "dy_rows": int(is_dy.sum()),
        "non_dy_rows": int((~is_dy).sum()),
        "dy_sumw": dy_sum,
        "non_dy_sumw": non_sum,
        "data_minus_non_dy_sumw": target,
        "dy_positive_mc_yield_fraction": float(dy_purity),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=cfg.CHANNELS)
    ap.add_argument("--folds", type=int, default=5,
                    help="number of DCTR cross-fitting folds over the outer train+validation sample")
    ap.add_argument("--dctr-target", choices=DCTR_TARGETS, default="inclusive",
                    help="inclusive: Data vs all MC; dy_only: (Data - non-DY MC) vs DY")
    ap.add_argument("--cap-quantile", type=float, default=0.995,
                    help="cap factors at this validation-DY/MC quantile; use 0 to disable")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--save-fold-models", action="store_true",
                    help="save all k cross-fit DCTR networks (normally unnecessary and disk-heavy)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    cap_quantile = None if args.cap_quantile <= 0 else args.cap_quantile
    out_root = target_root(args.output, args.dctr_target)
    out_root.mkdir(parents=True, exist_ok=True)
    print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))
    print("DCTR target:", args.dctr_target)

    metadata = {
        "purpose": "nested/cross-fitted DCTR closure C2ST",
        "dctr_target": args.dctr_target,
        "dctr_target_definition": (
            "Data vs all positive-weight pre-DY MC"
            if args.dctr_target == "inclusive"
            else "(Data - non-DY positive-weight MC) vs DY positive-weight MC"
        ),
        "features": cfg.FEATURES,
        "validation_vars": cfg.VALIDATION_VARS,
        "load_features": cfg.LOAD_FEATURES,
        "selections": {k: list(v) for k, v in cfg.SELECTIONS.items()},
        "folds": args.folds,
        "cap_quantile": cap_quantile,
        "eps": args.eps,
        "outer_test_size": cfg.TEST_SIZE,
        "outer_val_size_within_trainval": cfg.VAL_SIZE_WITHIN_TRAINVAL,
        "random_state": cfg.RANDOM_STATE,
        "model_config": {
            "closure_c2st": {
                "hidden": list(cfg.HIDDEN),
                "batch_normalization": bool(cfg.BATCH_NORMALIZATION),
                "optimizer": cfg.OPTIMIZER,
                "learning_rate": cfg.LEARNING_RATE,
            },
            "dctr": {
                "hidden": list(cfg.DCTR_HIDDEN),
                "batch_normalization": bool(cfg.DCTR_BATCH_NORMALIZATION),
                "optimizer": cfg.DCTR_OPTIMIZER,
                "learning_rate": cfg.DCTR_LEARNING_RATE,
            },
        },
        "stages": {
            "before": "weight_uncorrected",
            "dy": "weight (official DY correction included)",
            "dctr": (
                "weight_uncorrected * out-of-fold DCTR factor for all MC"
                if args.dctr_target == "inclusive"
                else "weight_uncorrected * out-of-fold DCTR factor for DY; non-DY factor = 1"
            ),
        },
        "negative_mc_policy": (
            "generator-level weight_uncorrected <= 0 events are excluded from classifier training/closure; "
            "in dy_only mode positive-weight non-DY events enter the DCTR target with a negative subtraction sign"
        ),
        "dy_only_signed_loss": (
            "Keras BCE with signed sample weights normalized by one common sum|w| factor"
            if args.dctr_target == "dy_only" else None
        ),
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    layout = dyvr_lib.discover_store_layout(cfg.STORE_ROOT, cfg.REDUCTION_DIR)
    all_summary = []

    for channel_i, channel in enumerate(args.channels):
        print(f"\n{'='*90}\nCHANNEL {channel}\n{'='*90}")
        channel_out = out_root / channel
        channel_out.mkdir(parents=True, exist_ok=True)

        tables = dyvr_lib.load_all(
            layout,
            cfg.MC_PROCESSES,
            cfg.DATA_PROCESSES,
            cfg.ALIGNMENT_OK,
            cfg.SHIFT,
            feature_fields=cfg.LOAD_FEATURES,
            validate_classification=False,
            keep_region="dycr",
            keep_channels=(channel,),
            selections=cfg.SELECTIONS,
            compact_dtypes=True,
        )
        data_df, mc_full = channel_tables(tables, channel)
        del tables
        gc.collect()
        if not len(data_df) or not len(mc_full):
            print(f"[{channel}] empty Data or MC, skipping")
            continue

        neg_event_frac = float((mc_full["weight_uncorrected"] <= 0).mean())
        absw = np.abs(mc_full["weight_uncorrected"].to_numpy(dtype=np.float64, copy=False))
        neg_mask_full = mc_full["weight_uncorrected"].to_numpy(dtype=np.float64, copy=False) <= 0
        neg_absw_frac = float(absw[neg_mask_full].sum() / absw.sum()) if absw.sum() > 0 else np.nan

        mc_df = mc_full.loc[mc_full["weight_uncorrected"] > 0].reset_index(drop=True)
        del mc_full, absw, neg_mask_full
        data_df = maybe_subsample(data_df, cfg.MAX_EVENTS_PER_CLASS, cfg.RANDOM_STATE)
        mc_df = maybe_subsample(mc_df, cfg.MAX_EVENTS_PER_CLASS, cfg.RANDOM_STATE)
        print(
            f"[{channel}] Data={len(data_df):_}, positive-weight MC={len(mc_df):_}, "
            f"negative-event fraction={neg_event_frac:.3%}, negative |sumw| fraction={neg_absw_frac:.3%}"
        )
        target_composition = None
        if args.dctr_target == "dy_only":
            target_composition = print_dy_only_composition(channel, data_df, mc_df)
            if not mc_df["is_dy"].any():
                raise RuntimeError(f"[{channel}] no DY MC available for --dctr-target dy_only")

        # OUTER split is identical for both target definitions.
        train_d, val_d, test_d = split_class_indices(
            len(data_df), cfg.TEST_SIZE, cfg.VAL_SIZE_WITHIN_TRAINVAL, cfg.RANDOM_STATE
        )
        train_m, val_m, test_m = split_class_indices(
            len(mc_df), cfg.TEST_SIZE, cfg.VAL_SIZE_WITHIN_TRAINVAL, cfg.RANDOM_STATE + 1
        )
        trainval_d = np.concatenate([train_d, val_d])
        trainval_m = np.concatenate([train_m, val_m])

        scaler_fit = pd.concat([
            data_df.iloc[train_d][cfg.FEATURES],
            mc_df.iloc[train_m][cfg.FEATURES],
        ], ignore_index=True)
        scaler = fit_scaler(scaler_fit, cfg.FEATURES)
        del scaler_fit
        joblib.dump(scaler, channel_out / "scaler.joblib", compress=3)

        print(f"[{channel}] transforming Data/MC feature matrices once ...")
        x_data = apply_scaler(data_df, cfg.FEATURES, scaler)
        x_mc = apply_scaler(mc_df, cfg.FEATURES, scaler)
        raw_before = mc_df["weight_uncorrected"].to_numpy(dtype=np.float32, copy=True)
        raw_dy = mc_df["weight"].to_numpy(dtype=np.float32, copy=True)
        is_dy = mc_df["is_dy"].to_numpy(dtype=bool, copy=True)

        fold_models_dir = channel_out / "fold_models" if args.save_fold_models else None
        seed_base = cfg.RANDOM_STATE + 100 * channel_i
        if args.dctr_target == "inclusive":
            dctr_factor, fold_id, fold_caps = crossfit_dctr_trainval_inclusive(
                x_data, trainval_d, x_mc, trainval_m, raw_before,
                n_folds=args.folds, cap_quantile=cap_quantile, eps=args.eps,
                seed=seed_base, save_fold_models_dir=fold_models_dir,
            )
            final_dctr_model, factor_test, final_cap = fit_final_dctr_for_outer_test_inclusive(
                x_data, trainval_d, x_mc, trainval_m, test_m, raw_before,
                cap_quantile, args.eps, cfg.RANDOM_STATE + 5000 + 100 * channel_i,
            )
            dctr_factor[test_m] = factor_test
            fold_id[test_m] = args.folds
            del factor_test
        else:
            dctr_factor, fold_id, fold_caps = crossfit_dctr_trainval_dy_only(
                x_data, trainval_d, x_mc, trainval_m, is_dy, raw_before,
                n_folds=args.folds, cap_quantile=cap_quantile, eps=args.eps,
                seed=seed_base, save_fold_models_dir=fold_models_dir,
            )
            final_dctr_model, dy_test, factor_test_dy, final_cap = fit_final_dctr_for_outer_test_dy_only(
                x_data, trainval_d, x_mc, trainval_m, test_m, is_dy, raw_before,
                cap_quantile, args.eps, cfg.RANDOM_STATE + 5000 + 100 * channel_i,
            )
            dctr_factor[dy_test] = factor_test_dy
            fold_id[dy_test] = args.folds
            # non-DY factors stay exactly one and their fold id stays -1 (not corrected).
            del dy_test, factor_test_dy

        final_dctr_model.save(channel_out / "dctr_model_final.keras")
        del final_dctr_model
        tf.keras.backend.clear_session()
        gc.collect()

        closure_mc = np.concatenate([trainval_m, test_m])
        if np.any(~np.isfinite(dctr_factor[closure_mc])):
            raise RuntimeError(f"[{channel}] non-finite DCTR factor remains on closure population")
        if args.dctr_target == "dy_only" and not np.all(dctr_factor[~is_dy] == 1.0):
            raise RuntimeError(f"[{channel}] a non-DY event received a DY-only DCTR factor different from 1")

        np.savez(
            channel_out / "dctr_factors_mc.npz",
            dctr_factor=dctr_factor,
            fold_id=fold_id,
            is_dy=is_dy,
            dctr_target=np.asarray(args.dctr_target),
            mc_train=train_m,
            mc_val=val_m,
            mc_test=test_m,
            fold_caps=np.asarray([np.nan if x is None else x for x in fold_caps], dtype=np.float64),
            final_test_cap=np.asarray(np.nan if final_cap is None else final_cap, dtype=np.float64),
        )

        outer_test = make_outer_test_fold(data_df, test_d, mc_df, test_m, dctr_factor[test_m])
        outer_test.to_parquet(channel_out / "outer_test_fold.parquet", index=False, compression="zstd")
        del outer_test

        x_train, y_train = make_pair(x_data, train_d, x_mc, train_m)
        x_val, y_val = make_pair(x_data, val_d, x_mc, val_m)
        x_test, y_test = make_pair(x_data, test_d, x_mc, test_m)

        physical_stage_weights = {
            "before": raw_before,
            "dy": raw_dy,
            "dctr": raw_before * dctr_factor,
        }
        metrics_by_stage = {}
        for stage_i, (stage, raw_stage) in enumerate(physical_stage_weights.items()):
            metrics = train_closure_stage(
                channel, stage,
                x_train, y_train, x_val, y_val, x_test, y_test,
                raw_stage, raw_stage[train_m], raw_stage[val_m], raw_stage[test_m],
                len(data_df), len(train_d), len(val_d), len(test_d),
                channel_out,
                seed=cfg.RANDOM_STATE + 7000 + 100 * channel_i + stage_i,
            )
            metrics["dctr_target"] = args.dctr_target
            metrics_by_stage[stage] = metrics
            all_summary.append(metrics)

        comparison = {
            "channel": channel,
            "dctr_target": args.dctr_target,
            "auc_before": metrics_by_stage["before"]["auc"],
            "auc_dy": metrics_by_stage["dy"]["auc"],
            "auc_dctr": metrics_by_stage["dctr"]["auc"],
            "delta_dy_vs_before": metrics_by_stage["dy"]["auc"] - metrics_by_stage["before"]["auc"],
            "delta_dctr_vs_before": metrics_by_stage["dctr"]["auc"] - metrics_by_stage["before"]["auc"],
            "delta_dctr_vs_dy": metrics_by_stage["dctr"]["auc"] - metrics_by_stage["dy"]["auc"],
            "negative_event_fraction_excluded": neg_event_frac,
            "negative_absw_fraction_excluded": neg_absw_frac,
            "final_dctr_cap": final_cap,
            "target_composition": target_composition,
        }
        (channel_out / "comparison.json").write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2))

        del x_train, y_train, x_val, y_val, x_test, y_test
        del x_data, x_mc, raw_before, raw_dy, dctr_factor, fold_id, is_dy
        del train_d, val_d, test_d, train_m, val_m, test_m, trainval_d, trainval_m
        del scaler, data_df, mc_df, physical_stage_weights
        gc.collect()

    pd.DataFrame(all_summary).to_csv(out_root / "closure_metrics.csv", index=False)
    print(f"\nAll closure artifacts written under {out_root.resolve()}")


if __name__ == "__main__":
    main()
