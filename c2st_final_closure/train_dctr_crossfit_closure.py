from __future__ import annotations

"""Train and validate an out-of-fold DCTR correction with an independent closure C2ST.

The workflow is deliberately separated from c2st_nn.py:

1. Load one channel at a time and retain the same DY-VR phase space/config selections.
2. Remove negative-weight MC *for classifier training only* (ordinary BCE needs a positive
   sample measure). The signed-weight deployment issue is discussed in README_DCTR_CLOSURE.md.
3. Make one OUTER train/validation/test split. The outer test set is never used to fit DCTR.
4. Cross-fit DCTR factors on the outer train+validation population:
      - each row receives a factor from a DCTR network that did not train on that row;
      - each fold's cap is derived from that fold model's internal validation MC, not from the
        held-out fold to which the cap is applied.
5. Train one final DCTR model on outer train+validation and apply it to the untouched outer test.
6. Train three NEW closure classifiers on exactly the same outer split:
      before : nominal weight without the DY correction
      dy     : nominal weight with the official DY correction
      dctr   : nominal weight without DY correction x cross-fitted DCTR factor
7. Evaluate all three on the exact same untouched outer test events and save their AUCs/predictions.

The target result for a successful correction is an AUC closer to 0.5. The script does not
assume DCTR must outperform the nominal DY correction; that is what the closure test measures.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2st_config as cfg
import dyvr_lib
from c2st_core import apply_scaler, fit_scaler, split_class_indices, stage_weights, weighted_bce


DEFAULT_OUT = cfg.ARTIFACT_DIR / "dctr_crossfit_closure"


def build_model(n_features: int, seed: int) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for n_nodes in cfg.HIDDEN:
        layers.append(tf.keras.layers.Dense(n_nodes, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss="binary_crossentropy",
    )
    return model


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
    data_parts = [df for label, df in tables.items() if label in cfg.DATA_PROCESSES and len(df)]
    mc_parts = [df for label, df in tables.items() if label in cfg.MC_PROCESSES and len(df)]
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
    """Materialize one Data+MC NN matrix for the requested class-specific indices."""
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
    """Fit one Data-vs-MC network using class-balanced physical MC weights."""
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

    model = build_model(x_train.shape[1], seed)
    print(
        f"=== {label}: train={len(y_train):_}, val={len(y_val):_}, "
        f"batch={cfg.BATCH_SIZE:_} ==="
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


def dctr_from_probability(p, eps: float):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return p / (1.0 - p)


def cap_from_validation(model, x_mc, idx_mc_val, quantile, eps):
    if quantile is None:
        return None
    if len(idx_mc_val) == 0:
        return None
    p = model.predict(x_mc[idx_mc_val], batch_size=cfg.BATCH_SIZE, verbose=0).reshape(-1)
    factors = dctr_from_probability(p, eps)
    finite = factors[np.isfinite(factors) & (factors >= 0)]
    if not len(finite):
        return None
    return float(np.quantile(finite, quantile))


def predict_dctr(model, x_mc, indices, cap_value, eps):
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


def inner_train_val(indices: np.ndarray, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(indices)
    n_val = max(1, int(round(val_fraction * len(indices))))
    if n_val >= len(indices):
        n_val = max(1, len(indices) - 1)
    return indices[n_val:], indices[:n_val]


def crossfit_dctr_trainval(
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
    """Out-of-fold DCTR factors for the OUTER train+validation MC population.

    Each held-out MC fold receives predictions from a classifier that did not train on that fold.
    Data is folded in parallel so the held-out DCTR evaluation population is class-independent.
    Only MC DCTR factors are needed downstream.
    """
    data_folds = shuffled_folds(data_trainval, n_folds, seed)
    mc_folds = shuffled_folds(mc_trainval, n_folds, seed + 1000)
    factors = np.full(len(x_mc), np.nan, dtype=np.float32)
    fold_id = np.full(len(x_mc), -1, dtype=np.int16)
    cap_values = []

    for k in range(n_folds):
        hold_d = data_folds[k]
        hold_m = mc_folds[k]
        cand_d = np.concatenate([data_folds[j] for j in range(n_folds) if j != k])
        cand_m = np.concatenate([mc_folds[j] for j in range(n_folds) if j != k])

        train_d, val_d = inner_train_val(cand_d, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 10 * k + 1)
        train_m, val_m = inner_train_val(cand_m, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 10 * k + 2)

        model = fit_binary_model(
            x_data, train_d, val_d,
            x_mc, train_m, val_m,
            raw_before,
            seed + k,
            label=f"DCTR cross-fit fold {k + 1}/{n_folds}",
        )
        cap_value = cap_from_validation(model, x_mc, val_m, cap_quantile, eps)
        fold_factor = predict_dctr(model, x_mc, hold_m, cap_value, eps)
        factors[hold_m] = fold_factor
        fold_id[hold_m] = k
        cap_values.append(cap_value)

        print(
            f"  fold {k + 1}: held-out MC={len(hold_m):_}, "
            f"cap={cap_value if cap_value is not None else 'none'}, "
            f"factor mean={float(np.mean(fold_factor)):.5g}, "
            f"max={float(np.max(fold_factor)):.5g}"
        )

        if save_fold_models_dir is not None:
            save_fold_models_dir.mkdir(parents=True, exist_ok=True)
            model.save(save_fold_models_dir / f"dctr_fold_{k}.keras")

        del model, hold_d, hold_m, cand_d, cand_m, train_d, val_d, train_m, val_m, fold_factor
        tf.keras.backend.clear_session()
        gc.collect()

    if np.any(~np.isfinite(factors[mc_trainval])):
        missing = int(np.sum(~np.isfinite(factors[mc_trainval])))
        raise RuntimeError(f"Cross-fitting failed to assign DCTR factors to {missing} train/val MC rows")

    return factors, fold_id, cap_values


def fit_final_dctr_for_outer_test(
    x_data,
    data_trainval,
    x_mc,
    mc_trainval,
    mc_test,
    raw_before,
    cap_quantile,
    eps,
    seed,
):
    """Fit a final DCTR model with NO outer-test events and predict the outer-test MC."""
    train_d, val_d = inner_train_val(data_trainval, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 3001)
    train_m, val_m = inner_train_val(mc_trainval, cfg.VAL_SIZE_WITHIN_TRAINVAL, seed + 3002)
    model = fit_binary_model(
        x_data, train_d, val_d,
        x_mc, train_m, val_m,
        raw_before,
        seed + 3000,
        label="DCTR final model for untouched outer test",
    )
    cap_value = cap_from_validation(model, x_mc, val_m, cap_quantile, eps)
    factor_test = predict_dctr(model, x_mc, mc_test, cap_value, eps)
    return model, factor_test, cap_value


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

    model = build_model(x_train.shape[1], seed)
    print(f"=== closure [{channel}, {stage}] ===")
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
    np.savez(
        output_dir / f"closure_{stage}_test.npz",
        p_test=p_test,
        w_test=w_test,
    )
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
    d["weight_uncorrected"] = np.float32(1.0)
    d["weight"] = np.float32(1.0)
    d["dctr_factor"] = np.float32(1.0)
    d["weight_dctr"] = np.float32(1.0)

    m = mc_df.iloc[test_m][cols + ["weight_uncorrected", "weight"]].copy()
    m.insert(0, "y", np.zeros(len(m), dtype=np.uint8))
    m["dctr_factor"] = dctr_factor_test.astype(np.float32, copy=False)
    m["weight_dctr"] = (
        m["weight_uncorrected"].to_numpy(dtype=np.float32, copy=False)
        * dctr_factor_test.astype(np.float32, copy=False)
    )
    out = pd.concat([d, m], ignore_index=True)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=cfg.CHANNELS)
    ap.add_argument("--folds", type=int, default=5,
                    help="number of DCTR cross-fitting folds over the outer train+validation sample")
    ap.add_argument("--cap-quantile", type=float, default=0.995,
                    help="cap DCTR factors at this quantile measured on each DCTR model's internal validation MC; use 0 to disable")
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
    args.output.mkdir(parents=True, exist_ok=True)
    print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))

    metadata = {
        "purpose": "nested/cross-fitted DCTR closure C2ST",
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
        "stages": {
            "before": "weight_uncorrected",
            "dy": "weight (official DY correction included)",
            "dctr": "weight_uncorrected * out-of-fold DCTR factor",
        },
        "negative_mc_policy": "excluded from classifier training/closure C2ST because BCE requires non-negative sample weights",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    layout = dyvr_lib.discover_store_layout(cfg.STORE_ROOT, cfg.REDUCTION_DIR)
    all_summary = []

    for channel_i, channel in enumerate(args.channels):
        print(f"\n{'='*90}\nCHANNEL {channel}\n{'='*90}")
        channel_out = args.output / channel
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

        # OUTER split: the final test rows are protected from every DCTR training step.
        train_d, val_d, test_d = split_class_indices(
            len(data_df), cfg.TEST_SIZE, cfg.VAL_SIZE_WITHIN_TRAINVAL, cfg.RANDOM_STATE
        )
        train_m, val_m, test_m = split_class_indices(
            len(mc_df), cfg.TEST_SIZE, cfg.VAL_SIZE_WITHIN_TRAINVAL, cfg.RANDOM_STATE + 1
        )
        trainval_d = np.concatenate([train_d, val_d])
        trainval_m = np.concatenate([train_m, val_m])

        # Fit preprocessing on OUTER TRAIN only. Outer validation and test do not influence it.
        scaler_fit = pd.concat([
            data_df.iloc[train_d][cfg.FEATURES],
            mc_df.iloc[train_m][cfg.FEATURES],
        ], ignore_index=True)
        scaler = fit_scaler(scaler_fit, cfg.FEATURES)
        del scaler_fit
        joblib.dump(scaler, channel_out / "scaler.joblib", compress=3)

        # Convert once to compact float32 arrays. This avoids repeatedly carrying pandas copies
        # through k-fold NN fitting and keeps peak RAM bounded.
        print(f"[{channel}] transforming Data/MC feature matrices once ...")
        x_data = apply_scaler(data_df, cfg.FEATURES, scaler)
        x_mc = apply_scaler(mc_df, cfg.FEATURES, scaler)
        raw_before = mc_df["weight_uncorrected"].to_numpy(dtype=np.float32, copy=True)
        raw_dy = mc_df["weight"].to_numpy(dtype=np.float32, copy=True)

        # Cross-fitted correction on closure train+validation rows.
        fold_models_dir = channel_out / "fold_models" if args.save_fold_models else None
        dctr_factor, fold_id, fold_caps = crossfit_dctr_trainval(
            x_data,
            trainval_d,
            x_mc,
            trainval_m,
            raw_before,
            n_folds=args.folds,
            cap_quantile=cap_quantile,
            eps=args.eps,
            seed=cfg.RANDOM_STATE + 100 * channel_i,
            save_fold_models_dir=fold_models_dir,
        )

        # Final deployable DCTR model: uses no outer test events whatsoever.
        final_dctr_model, factor_test, final_cap = fit_final_dctr_for_outer_test(
            x_data,
            trainval_d,
            x_mc,
            trainval_m,
            test_m,
            raw_before,
            cap_quantile,
            args.eps,
            cfg.RANDOM_STATE + 5000 + 100 * channel_i,
        )
        dctr_factor[test_m] = factor_test
        fold_id[test_m] = args.folds  # sentinel: predicted by final trainval model
        final_dctr_model.save(channel_out / "dctr_model_final.keras")
        del final_dctr_model, factor_test
        tf.keras.backend.clear_session()
        gc.collect()

        if np.any(~np.isfinite(dctr_factor[np.concatenate([trainval_m, test_m])])):
            raise RuntimeError(f"[{channel}] non-finite DCTR factor remains on closure population")

        # Save reusable per-MC correction factors and provenance.
        np.savez(
            channel_out / "dctr_factors_mc.npz",
            dctr_factor=dctr_factor,
            fold_id=fold_id,
            mc_train=train_m,
            mc_val=val_m,
            mc_test=test_m,
            fold_caps=np.asarray([np.nan if x is None else x for x in fold_caps], dtype=np.float64),
            final_test_cap=np.asarray(np.nan if final_cap is None else final_cap, dtype=np.float64),
        )

        # Save only OUTER test raw variables/physical weights for independent plotting later.
        outer_test = make_outer_test_fold(data_df, test_d, mc_df, test_m, dctr_factor[test_m])
        outer_test.to_parquet(channel_out / "outer_test_fold.parquet", index=False, compression="zstd")
        del outer_test

        # Build the three closure NN matrices once; stages differ only by MC sample weights.
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
                channel,
                stage,
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                raw_stage,
                raw_stage[train_m],
                raw_stage[val_m],
                raw_stage[test_m],
                len(data_df),
                len(train_d),
                len(val_d),
                len(test_d),
                channel_out,
                seed=cfg.RANDOM_STATE + 7000 + 100 * channel_i + stage_i,
            )
            metrics_by_stage[stage] = metrics
            all_summary.append(metrics)

        comparison = {
            "channel": channel,
            "auc_before": metrics_by_stage["before"]["auc"],
            "auc_dy": metrics_by_stage["dy"]["auc"],
            "auc_dctr": metrics_by_stage["dctr"]["auc"],
            "delta_dy_vs_before": metrics_by_stage["dy"]["auc"] - metrics_by_stage["before"]["auc"],
            "delta_dctr_vs_before": metrics_by_stage["dctr"]["auc"] - metrics_by_stage["before"]["auc"],
            "delta_dctr_vs_dy": metrics_by_stage["dctr"]["auc"] - metrics_by_stage["dy"]["auc"],
            "negative_event_fraction_excluded": neg_event_frac,
            "negative_absw_fraction_excluded": neg_absw_frac,
            "final_dctr_cap": final_cap,
        }
        (channel_out / "comparison.json").write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2))

        # Free channel-scale objects before starting the next channel.
        del x_train, y_train, x_val, y_val, x_test, y_test
        del x_data, x_mc, raw_before, raw_dy, dctr_factor, fold_id
        del train_d, val_d, test_d, train_m, val_m, test_m, trainval_d, trainval_m
        del scaler, data_df, mc_df, physical_stage_weights
        gc.collect()

    pd.DataFrame(all_summary).to_csv(args.output / "closure_metrics.csv", index=False)
    print(f"\nAll closure artifacts written under {args.output.resolve()}")


if __name__ == "__main__":
    main()
