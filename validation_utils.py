from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def load_aligned(root, channel, stages=('before', 'after')):
    from c2st_artifacts import load_test_fold, load_stage_arrays
    test = load_test_fold(root, channel)
    y = test['y'].to_numpy(dtype=np.uint8, copy=False)
    stage_data = {}
    for stage in stages:
        arr = load_stage_arrays(root, channel, stage)
        p = np.asarray(arr['p_test'], dtype=np.float32)
        w = np.asarray(arr['w_test'], dtype=np.float32)
        if len(p) != len(y) or len(w) != len(y):
            raise ValueError(f'{channel}/{stage}: artifact length mismatch')
        stage_data[stage] = {'p': p, 'w': w}
    return test, y, stage_data


def quantile_edges(values, n_bins):
    values = np.asarray(values)
    edges = np.quantile(values[np.isfinite(values)], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError('Not enough distinct values to construct bins')
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def weighted_auc(y, p, w):
    return float(roc_auc_score(y, p, sample_weight=w))


def prepare_fixed_score_auc(p):
    """Precompute score ordering and tie groups for repeated weighted-AUC evaluations."""
    p = np.asarray(p)
    order = np.argsort(p, kind='mergesort')
    ps = p[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]
    return order, starts


def weighted_auc_preordered(y, w, order, starts):
    """Weighted AUC in O(n) once score ordering is known; exact tie handling."""
    ys = np.asarray(y, dtype=np.uint8)[order]
    ws = np.asarray(w, dtype=np.float64)[order]
    pos = ws * (ys == 1)
    neg = ws * (ys == 0)
    pos_g = np.add.reduceat(pos, starts)
    neg_g = np.add.reduceat(neg, starts)
    total_pos = pos_g.sum(dtype=np.float64)
    total_neg = neg_g.sum(dtype=np.float64)
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    neg_before = np.cumsum(neg_g, dtype=np.float64) - neg_g
    concordant = np.sum(pos_g * (neg_before + 0.5 * neg_g), dtype=np.float64)
    return float(concordant / (total_pos * total_neg))


def paired_bootstrap_delta_auc(y, p_before, w_before, p_after, w_after,
                               n_resamples=50, random_state=0, subsample=None):
    """Memory-bounded paired bootstrap using one integer multiplicity vector per replicate.

    Sampling n row indices with replacement is represented by bincount multiplicities. This is
    exactly the ordinary nonparametric bootstrap but avoids materializing five resampled arrays.
    Score sorting is cached once per model, so each replicate is O(n) rather than O(n log n).
    """
    y = np.asarray(y, dtype=np.uint8)
    pb = np.asarray(p_before, dtype=np.float32)
    wb = np.asarray(w_before, dtype=np.float32)
    pa = np.asarray(p_after, dtype=np.float32)
    wa = np.asarray(w_after, dtype=np.float32)
    rng = np.random.default_rng(random_state)

    if subsample is not None and 0 < subsample < len(y):
        sel = rng.choice(len(y), size=subsample, replace=False)
        y, pb, wb, pa, wa = y[sel], pb[sel], wb[sel], pa[sel], wa[sel]

    order_b, starts_b = prepare_fixed_score_auc(pb)
    order_a, starts_a = prepare_fixed_score_auc(pa)
    observed_b = weighted_auc_preordered(y, wb, order_b, starts_b)
    observed_a = weighted_auc_preordered(y, wa, order_a, starts_a)
    deltas = np.empty(n_resamples, dtype=np.float64)
    n = len(y)

    for i in range(n_resamples):
        # int32 draw halves temporary memory versus NumPy's default int64 indices.
        draws = rng.integers(0, n, size=n, dtype=np.int32)
        counts = np.bincount(draws, minlength=n).astype(np.float32, copy=False)
        del draws
        auc_b = weighted_auc_preordered(y, wb * counts, order_b, starts_b)
        auc_a = weighted_auc_preordered(y, wa * counts, order_a, starts_a)
        deltas[i] = auc_a - auc_b
        del counts

    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        'auc_before': observed_b,
        'auc_after': observed_a,
        'delta_auc_observed': observed_a - observed_b,
        'ci_low': float(lo), 'ci_high': float(hi),
        'excludes_zero': bool(lo > 0 or hi < 0),
        'bootstrap_distribution': deltas,
    }


def fixed_model_permutation_pvalue(y, p, w, n_resamples=100, random_state=0,
                                   subsample=None, alternative='greater'):
    """Fixed-model label permutation check with cached score sorting.

    This intentionally remains the narrower diagnostic null from additions.py; it does not
    retrain the classifier for each label permutation and therefore is not the exact C2ST null.
    """
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(p, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    rng = np.random.default_rng(random_state)
    if subsample is not None and 0 < subsample < len(y):
        sel = rng.choice(len(y), size=subsample, replace=False)
        y, p, w = y[sel], p[sel], w[sel]
    order, starts = prepare_fixed_score_auc(p)
    observed = weighted_auc_preordered(y, w, order, starts)
    null = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        yp = rng.permutation(y)
        null[i] = weighted_auc_preordered(yp, w, order, starts)
    if alternative == 'greater':
        pval = (np.sum(null >= observed) + 1) / (n_resamples + 1)
    elif alternative == 'less':
        pval = (np.sum(null <= observed) + 1) / (n_resamples + 1)
    else:
        center = null.mean()
        pval = (np.sum(np.abs(null - center) >= abs(observed - center)) + 1) / (n_resamples + 1)
    return {'observed_auc': observed, 'null_distribution': null, 'p_value': float(pval)}
