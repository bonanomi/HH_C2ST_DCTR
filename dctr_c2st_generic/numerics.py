"""Pure numerical contracts: labels, weights, folds, odds, and diagnostics."""
import numpy as np
from c2st_core import normalize_signed_sample_weights, stage_weights


def shuffled_folds(indices, n_folds, seed):
    indices = np.asarray(indices, dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(indices)
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(indices, n_folds)]


def inner_train_val(indices, fraction, seed, allow_empty=False):
    indices = np.asarray(indices, dtype=np.int64).copy()
    if len(indices) < 2:
        if allow_empty:
            return indices, np.empty(0, dtype=np.int64)
        raise ValueError("Need at least two events for an internal split")
    np.random.default_rng(seed).shuffle(indices)
    n = min(len(indices) - 1, max(1, int(round(fraction * len(indices)))))
    return indices[n:], indices[:n]


def other_folds(folds, k):
    return np.concatenate([fold for j, fold in enumerate(folds) if j != k])


def odds(probability, eps):
    p = np.asarray(probability, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid classifier probabilities")
    p = np.clip(p, eps, 1 - eps)
    return p / (1 - p)


def sample_weights(wt, wb, indices, reference, normalization):
    """Return weights in target, subtraction, corrected-base order.

    `reference` is (target, subtraction, corrected-base) TRAIN indices only in
    generic mode. Shape balance scales the corrected base to the residual target.
    Legacy inclusive reproduces full-cohort arithmetic; legacy subtraction uses
    the historical common mean-|w| normalization independently in each sample.
    """
    it, ins, ib = indices
    rt, rns, rb = reference
    if normalization == "legacy":
        if not np.all(wt == 1):
            raise ValueError("Legacy inclusive requires unit target weights")
        a, b = stage_weights(len(wt), wb, wb[ib], len(it))
        return np.concatenate([a, b]).astype(np.float32, copy=False)
    target_yield = np.sum(wt[rt], dtype=np.float64) - np.sum(wb[rns], dtype=np.float64)
    base_yield = np.sum(wb[rb], dtype=np.float64)
    if normalization not in ("legacy", "legacy_signed") and (target_yield <= 0 or base_yield <= 0):
        raise ValueError("Nonpositive training residual-target or corrected-base yield")
    scale = target_yield / base_yield if normalization == "shape" else 1.
    raw = np.concatenate([np.asarray(wt[it], dtype=np.float64),
                          -np.asarray(wb[ins], dtype=np.float64),
                          np.asarray(wb[ib], dtype=np.float64) * scale])
    return normalize_signed_sample_weights(raw).astype(np.float32, copy=False)


def closure_weights(wt, wb, it, ib, train_t, train_b, compatibility=False):
    """Closure C2ST is class-balanced; physical yield checks are separate."""
    if np.any(~np.isfinite(wt)) or np.any(~np.isfinite(wb)) or np.any(wt < 0) or np.any(wb < 0):
        raise ValueError("Closure C2ST needs finite nonnegative stage weights")
    if np.sum(wt[train_t], dtype=np.float64) <= 0 or np.sum(wb[train_b], dtype=np.float64) <= 0:
        raise ValueError("Each closure training class must have positive total weight")
    if compatibility:
        if not np.all(wt == 1):
            raise ValueError("Legacy closure requires unit target weights")
        a, b = stage_weights(len(wt), wb, wb[ib], len(it))
        return np.concatenate([a, b]).astype(np.float32)
    ts = np.sum(wt[train_t], dtype=np.float64)
    bs = np.sum(wb[train_b], dtype=np.float64)
    common = (len(train_t) + len(train_b)) / (2 * ts)
    return np.concatenate([wt[it] * common, wb[ib] * (ts / bs * common)]).astype(np.float32)


def factor_summary(factor, weights, cap=None):
    f, w = np.asarray(factor, float), np.asarray(weights, float)
    if not len(f) or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Expected nonempty finite positive factors")
    return {
        "count": len(f), "quantiles": dict(zip(
            ["0", "0.01", "0.05", "0.5", "0.95", "0.99", "1"],
            np.quantile(f, [0, .01, .05, .5, .95, .99, 1]).tolist())),
        "mean": float(f.mean()),
        "weighted_mean": float(np.sum(w * f) / w.sum()) if w.sum() != 0 else None,
        "cap": cap,
        "fraction_at_cap": float(np.mean(np.isclose(f, cap, rtol=2e-6, atol=1e-8))) if cap is not None else 0.,
        "fraction_below_0.01": float(np.mean(f < .01)),
    }


def reweight_signed(weights, factor):
    """Apply a positive correction WITHOUT removing or taking abs of signed weights."""
    w, f = np.asarray(weights), np.asarray(factor)
    if w.shape != f.shape or np.any(~np.isfinite(w)) or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Weights/factors must have equal shape and be finite; factors must be positive")
    return w * f
