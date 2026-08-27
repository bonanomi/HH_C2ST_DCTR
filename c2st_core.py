from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def categorize_hep_features(features):
    robust_patterns = [r'_pt$', r'_ht$', r'_lt$', r'^mli_mbb$', r'^mli_mll$', r'_mllMET$', r'_mbbllMET$', r'_mass$']
    robust = [f for f in features if any(re.search(p, f) for p in robust_patterns)]
    minmax = [f for f in features if f not in robust]
    return robust, minmax


def make_scaler(features):
    """Build the feature transformer used by the NN.

    Long-tailed kinematic variables use RobustScaler *only*.  Chaining a
    MinMaxScaler afterwards would make the final scale depend again on the
    observed extrema and can compress the bulk of variables with TeV-scale
    tails. Bounded/count/angular inputs continue to use MinMaxScaler.
    """
    robust, minmax = categorize_hep_features(features)
    transforms = []
    if robust:
        transforms.append(('kinematics', RobustScaler(), robust))
    if minmax:
        transforms.append(('bounded_counts', MinMaxScaler(), minmax))
    return ColumnTransformer(transforms, remainder='drop')


def fit_scaler(df, features):
    scaler = make_scaler(features)
    scaler.fit(df[features])
    return scaler


def apply_scaler(df, features, scaler):
    x = scaler.transform(df[features])
    raw_cols = [name.split('__', 1)[1] for name in scaler.get_feature_names_out()]
    pos = {name: j for j, name in enumerate(raw_cols)}
    x = x[:, [pos[f] for f in features]]
    return np.asarray(x, dtype=np.float32)


def split_class_indices(n, test_size=0.30, val_size_within_trainval=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(test_size * n))
    test = idx[:n_test]
    trainval = idx[n_test:]
    n_val = int(round(val_size_within_trainval * len(trainval)))
    val = trainval[:n_val]
    train = trainval[n_val:]
    return train, val, test


def weighted_bce(y, p, w):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.sum(w * loss, dtype=np.float64) / np.sum(w, dtype=np.float64))


def class_balance_factors(n_data, raw_mc_weights):
    raw_mc_weights = np.asarray(raw_mc_weights, dtype=np.float64)
    mc_scale = n_data / raw_mc_weights.sum(dtype=np.float64)
    global_scale = (n_data + len(raw_mc_weights)) / (2.0 * n_data)
    return float(mc_scale), float(global_scale)


def stage_weights(n_data_total, raw_mc_all, raw_mc_subset, n_data_subset):
    mc_scale, global_scale = class_balance_factors(n_data_total, raw_mc_all)
    w_data = np.full(n_data_subset, global_scale, dtype=np.float32)
    w_mc = np.asarray(raw_mc_subset, dtype=np.float32) * np.float32(mc_scale * global_scale)
    return w_data, w_mc
