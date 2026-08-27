from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2st_config as cfg
import dyvr_lib
from c2st_artifacts import (
    model_path, save_metadata, save_scaler, save_stage_arrays, save_stage_metrics, save_test_fold,
)
from c2st_core import apply_scaler, fit_scaler, split_class_indices, stage_weights, weighted_bce


def build_model(n_features: int):
    tf.keras.backend.clear_session()
    tf.random.set_seed(cfg.RANDOM_STATE)
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for n_nodes in cfg.HIDDEN:
        layers.append(tf.keras.layers.Dense(n_nodes, activation='relu'))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss='binary_crossentropy',
    )
    return model


def channel_tables(tables, channel):
    data_parts = [df for label, df in tables.items() if label in cfg.DATA_PROCESSES and len(df)]
    mc_parts = [df for label, df in tables.items() if label in cfg.MC_PROCESSES and len(df)]
    data = pd.concat(data_parts, ignore_index=True) if data_parts else pd.DataFrame()
    mc = pd.concat(mc_parts, ignore_index=True) if mc_parts else pd.DataFrame()
    if len(data):
        data = data[data['channel'] == channel].reset_index(drop=True)
    if len(mc):
        mc = mc[mc['channel'] == channel].reset_index(drop=True)
    return data, mc


def maybe_subsample(df, maximum, seed):
    if maximum is None or len(df) <= maximum:
        return df
    return df.sample(n=maximum, random_state=seed).reset_index(drop=True)


def transform_pair(data_df, data_idx, mc_df, mc_idx, scaler):
    n_data = len(data_idx)
    n_mc = len(mc_idx)
    x = np.empty((n_data + n_mc, len(cfg.FEATURES)), dtype=np.float32)
    if n_data:
        x[:n_data] = apply_scaler(data_df.iloc[data_idx], cfg.FEATURES, scaler)
    if n_mc:
        x[n_data:] = apply_scaler(mc_df.iloc[mc_idx], cfg.FEATURES, scaler)
    y = np.concatenate((np.ones(n_data, dtype=np.uint8), np.zeros(n_mc, dtype=np.uint8)))
    return x, y


def make_test_fold(data_df, data_idx, mc_df, mc_idx):
    cols = cfg.LOAD_FEATURES
    d = data_df.iloc[data_idx][cols].copy()
    d.insert(0, 'y', np.ones(len(d), dtype=np.uint8))
    d['weight_uncorrected'] = np.float32(1.0)
    d['weight'] = np.float32(1.0)
    m = mc_df.iloc[mc_idx][cols + ['weight_uncorrected', 'weight']].copy()
    m.insert(0, 'y', np.zeros(len(m), dtype=np.uint8))
    out = pd.concat([d, m], ignore_index=True)
    del d, m
    return out


def train_stage(channel, stage, x_train, y_train, x_val, y_val, x_test, y_test,
                data_df, mc_df, train_d, train_m, val_d, val_m, test_d, test_m):
    weight_col = cfg.WEIGHT_COLUMNS[stage]
    raw_mc_all = mc_df[weight_col].to_numpy(dtype=np.float32, copy=False)

    wd_train, wm_train = stage_weights(len(data_df), raw_mc_all, raw_mc_all[train_m], len(train_d))
    wd_val, wm_val = stage_weights(len(data_df), raw_mc_all, raw_mc_all[val_m], len(val_d))
    wd_test, wm_test = stage_weights(len(data_df), raw_mc_all, raw_mc_all[test_m], len(test_d))
    w_train = np.concatenate((wd_train, wm_train)).astype(np.float32, copy=False)
    w_val = np.concatenate((wd_val, wm_val)).astype(np.float32, copy=False)
    w_test = np.concatenate((wd_test, wm_test)).astype(np.float32, copy=False)

    model = build_model(x_train.shape[1])
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=cfg.REDUCE_LR_FACTOR,
            patience=cfg.REDUCE_LR_PATIENCE, min_lr=1e-5, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=cfg.EARLY_STOPPING_PATIENCE,
            min_delta=1e-4, restore_best_weights=True, verbose=1,
        ),
    ]

    print(f'=== training [{channel}, {stage}] train={len(y_train):_} val={len(y_val):_} test={len(y_test):_} ===')
    model.fit(
        x_train, y_train, sample_weight=w_train,
        validation_data=(x_val, y_val, w_val),
        epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )
    p_test = model.predict(x_test, batch_size=cfg.BATCH_SIZE, verbose=0).reshape(-1).astype(np.float32)
    auc = float(roc_auc_score(y_test, p_test, sample_weight=w_test))
    bce = weighted_bce(y_test, p_test, w_test)

    out_model = model_path(cfg.ARTIFACT_DIR, channel, stage)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model)
    save_stage_arrays(cfg.ARTIFACT_DIR, channel, stage, p_test=p_test, w_test=w_test)
    save_stage_metrics(cfg.ARTIFACT_DIR, channel, stage, {
        'channel': channel, 'stage': stage, 'auc': auc, 'weighted_bce': bce,
        'n_test': int(len(y_test)), 'model': str(out_model),
    })
    print(f'[{channel}, {stage}] AUC={auc:.6f}, weighted BCE={bce:.6f}')

    del model, p_test, w_train, w_val, w_test, wd_train, wm_train, wd_val, wm_val, wd_test, wm_test
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices('GPU')
    print('TensorFlow GPUs:', gpus)

    save_metadata(cfg.ARTIFACT_DIR, {
        'features': cfg.FEATURES,
        'validation_vars': cfg.VALIDATION_VARS,
        'load_features': cfg.LOAD_FEATURES,
        'selections': {k: list(v) for k, v in cfg.SELECTIONS.items()},
        'scaling': {
            'long_tailed_kinematics': 'RobustScaler',
            'bounded_counts_angles_scores': 'MinMaxScaler',
        },
        'channels': cfg.CHANNELS,
        'stages': cfg.STAGES,
        'weight_columns': cfg.WEIGHT_COLUMNS,
        'random_state': cfg.RANDOM_STATE,
        'test_size': cfg.TEST_SIZE,
        'val_size_within_trainval': cfg.VAL_SIZE_WITHIN_TRAINVAL,
        'hidden': list(cfg.HIDDEN),
        'batch_size': cfg.BATCH_SIZE,
    })

    layout = dyvr_lib.discover_store_layout(cfg.STORE_ROOT, cfg.REDUCTION_DIR)

    # Load/train one channel at a time. This deliberately trades some extra parquet I/O for a
    # much lower peak RSS than retaining both channels and all process groups simultaneously.
    for channel in cfg.CHANNELS:
        print(f'\n===== loading {channel} only =====')
        tables = dyvr_lib.load_all(
            layout, cfg.MC_PROCESSES, cfg.DATA_PROCESSES, cfg.ALIGNMENT_OK, cfg.SHIFT,
            feature_fields=cfg.LOAD_FEATURES,
            validate_classification=False,
            keep_region='dycr', keep_channels=(channel,), selections=cfg.SELECTIONS,
            compact_dtypes=True,
        )
        data_df, mc_full = channel_tables(tables, channel)
        del tables
        gc.collect()

        if not len(data_df) or not len(mc_full):
            print(f'[{channel}] empty Data or MC; skipping')
            del data_df, mc_full
            continue

        dropped_frac = float((mc_full['weight_uncorrected'] <= 0).mean())
        mc_df = mc_full.loc[mc_full['weight_uncorrected'] > 0].reset_index(drop=True)
        del mc_full
        data_df = maybe_subsample(data_df, cfg.MAX_EVENTS_PER_CLASS, cfg.RANDOM_STATE)
        mc_df = maybe_subsample(mc_df, cfg.MAX_EVENTS_PER_CLASS, cfg.RANDOM_STATE)
        print(f'[{channel}] Data={len(data_df):_}, MC positive-weight={len(mc_df):_}, dropped={dropped_frac:.2%}')

        train_d, val_d, test_d = split_class_indices(
            len(data_df), cfg.TEST_SIZE, cfg.VAL_SIZE_WITHIN_TRAINVAL, cfg.RANDOM_STATE)
        train_m, val_m, test_m = split_class_indices(
            len(mc_df), cfg.TEST_SIZE, cfg.VAL_SIZE_WITHIN_TRAINVAL, cfg.RANDOM_STATE + 1)

        # Fit the scaler only on training rows. The temporary frame is freed before transformed
        # train/validation matrices are allocated.
        scaler_fit = pd.concat([
            data_df.iloc[train_d][cfg.FEATURES],
            mc_df.iloc[train_m][cfg.FEATURES],
        ], ignore_index=True)
        scaler = fit_scaler(scaler_fit, cfg.FEATURES)
        del scaler_fit
        save_scaler(cfg.ARTIFACT_DIR, channel, scaler)
        gc.collect()

        x_train, y_train = transform_pair(data_df, train_d, mc_df, train_m, scaler)
        x_val, y_val = transform_pair(data_df, val_d, mc_df, val_m, scaler)
        x_test, y_test = transform_pair(data_df, test_d, mc_df, test_m, scaler)

        test_fold = make_test_fold(data_df, test_d, mc_df, test_m)
        save_test_fold(cfg.ARTIFACT_DIR, channel, test_fold)
        del test_fold
        gc.collect()

        for stage in cfg.STAGES:
            train_stage(
                channel, stage, x_train, y_train, x_val, y_val, x_test, y_test,
                data_df, mc_df, train_d, train_m, val_d, val_m, test_d, test_m,
            )

        # Nothing from the train/val matrices is needed downstream; validations start fresh from disk.
        del x_train, y_train, x_val, y_val, x_test, y_test
        del train_d, val_d, test_d, train_m, val_m, test_m, scaler, data_df, mc_df
        gc.collect()

    print(f'\nArtifacts written under: {cfg.ARTIFACT_DIR.resolve()}')


if __name__ == '__main__':
    main()
