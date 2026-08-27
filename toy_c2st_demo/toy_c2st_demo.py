#!/usr/bin/env python3
"""
toy_c2st_demo.py
================

A minimal, fully standalone demonstration of a classifier two-sample test (C2ST).

The goal is to reproduce, with a controlled toy problem, the logic used in the
real Data/MC study:

    1. Generate "Data" and "MC" from known distributions.
    2. Construct a deliberately incomplete MC correction ("nominal").
    3. Construct an almost exact correction ("closure").
    4. Train the SAME neural-network classifier on

           Data vs MC_nominal
           Data vs MC_closure

       using the SAME physical Data/MC events and the SAME train/validation/test split.
    5. Compare the held-out weighted AUCs.

Interpretation
--------------
If Data and weighted MC have identical feature distributions, no classifier can
systematically tell them apart and the expected AUC is 0.5.

If Data and weighted MC differ, the classifier can exploit the disagreement and
the held-out AUC rises above 0.5.

This script deliberately creates a case where

    AUC(Data vs nominal MC)  >> 0.5
    AUC(Data vs closure MC)  ~= 0.5

so that the meaning of the C2ST can be seen directly.

Important conceptual distinction
---------------------------------
The MC event weights are used as *sample weights* in the classifier loss.
Before training, the MC class is multiplied by one additional GLOBAL factor so
that the effective total Data and MC training weights are equal. This removes
overall normalization as a trivial class-prior difference and makes the C2ST
primarily a test of shape / multivariate distribution agreement.

The relative event-to-event MC weights are NOT removed.

Dependencies
------------
    numpy
    scipy
    scikit-learn
    matplotlib
    tensorflow

Example
-------
    python toy_c2st_demo.py

For a faster smoke test:
    python toy_c2st_demo.py --events 30000 --epochs 20

Outputs are written to ./toy_c2st_outputs by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Use a non-interactive backend so the script works on batch / NAF nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import tensorflow as tf


# ======================================================================================
# 1. Reproducibility and toy-distribution definitions
# ======================================================================================

DEFAULT_SEED = 7

# Both samples use the same covariance. Only the mean differs.
#
# This is useful pedagogically because:
#   - the Data/MC discrepancy is obvious but not pathological;
#   - the exact density-ratio p_data(x)/p_mc(x) is analytically available;
#   - the "closure" reweighting can therefore be known by construction.
COVARIANCE = np.array(
    [
        [1.00, 0.45],
        [0.45, 1.00],
    ],
    dtype=np.float64,
)

DATA_MEAN = np.array([0.0, 0.0], dtype=np.float64)

# Raw MC is deliberately displaced from Data.
MC_MEAN = np.array([0.9, -0.7], dtype=np.float64)

# The nominal correction only moves MC halfway toward Data.
#
# After applying the nominal weights, weighted MC behaves approximately like a
# Gaussian centered on this intermediate mean rather than on DATA_MEAN.
#
# Therefore a substantial residual discrepancy remains for the C2ST to find.
NOMINAL_TARGET_MEAN = np.array([0.45, -0.35], dtype=np.float64)


# ======================================================================================
# 2. Utility functions
# ======================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone toy demonstration of a classifier two-sample test."
    )
    parser.add_argument(
        "--events",
        type=int,
        default=100_000,
        help="Number of Data events AND number of MC events (default: 100000).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of NN training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Keras batch size (default: 2048).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("toy_c2st_outputs"),
        help="Directory where plots/results are written.",
    )
    return parser.parse_args()


def gaussian_density_ratio(
    x: np.ndarray,
    target_mean: np.ndarray,
    source_mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Return the exact density ratio

        p_target(x) / p_source(x)

    for two multivariate normal distributions with a common covariance.

    In a real Data/MC problem we do NOT know the Data density analytically.
    DCTR/C2ST-style methods use a classifier to estimate a density ratio instead.

    Here we intentionally know the answer, which lets us construct a toy
    "perfect correction" and test whether the classifier recognizes the closure.
    """
    log_target = multivariate_normal.logpdf(
        x,
        mean=target_mean,
        cov=covariance,
    )
    log_source = multivariate_normal.logpdf(
        x,
        mean=source_mean,
        cov=covariance,
    )

    # Work in log-space first for numerical stability.
    return np.exp(log_target - log_source)


def kish_effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish effective sample size:

        N_eff = (sum w)^2 / sum(w^2)

    Large variations in event weights reduce the effective statistical power.
    """
    weights = np.asarray(weights, dtype=np.float64)
    denominator = np.sum(weights**2)
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights) ** 2 / denominator)


def build_model(n_features: int, seed: int) -> tf.keras.Model:
    """
    Deliberately small neural network.

    The toy problem has only two input variables. A huge network would obscure
    the lesson: the classifier is merely a flexible tool for finding differences
    between two distributions.

    Architecture:
        2 inputs -> Dense(32, ReLU) -> Dense(32, ReLU) -> sigmoid output
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


def make_common_split(
    n_data: int,
    n_mc: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build ONE shared train/validation/test split for all weighting scenarios.

    Event ordering is

        [all Data events, all MC events]

    and labels are

        Data = 1
        MC   = 0

    Sharing the split is important: nominal and closure AUCs are then evaluated
    on exactly the same physical test events.
    """
    y = np.concatenate(
        [
            np.ones(n_data, dtype=np.int8),
            np.zeros(n_mc, dtype=np.int8),
        ]
    )
    idx = np.arange(len(y))

    # First hold out 20% for the final test set.
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=0.20,
        stratify=y,
        random_state=seed,
    )

    # Then use 20% of train+val as validation:
    # 0.8 * 0.2 = 0.16 of the full sample.
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.20,
        stratify=y[idx_trainval],
        random_state=seed,
    )

    return idx_train, idx_val, idx_test, y


def make_balanced_sample_weights(
    y: np.ndarray,
    idx_train: np.ndarray,
    mc_physical_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Construct C2ST sample weights.

    Data events have weight 1.

    MC events retain their RELATIVE physical weights, but the entire MC class is
    multiplied by one global factor chosen from the TRAINING subset:

        class_scale =
            N_data_train / sum(MC physical weights in training)

    Why?
    ----
    Without this global balancing, the loss also contains information about the
    different effective Data/MC class normalizations.

    With balancing, the NN is asked the cleaner question:

        "Can the event FEATURES distinguish Data from weighted MC?"

    The same global scale derived on the training subset is also applied to
    validation and test MC events. Weighted AUC is invariant under such a common
    multiplicative MC factor.
    """
    n_data = int(np.sum(y))
    n_mc = len(y) - n_data

    if len(mc_physical_weights) != n_mc:
        raise ValueError(
            f"Expected {n_mc} MC weights, got {len(mc_physical_weights)}."
        )

    full_weights = np.ones(len(y), dtype=np.float64)
    full_weights[n_data:] = mc_physical_weights

    train_is_data = y[idx_train] == 1
    train_is_mc = ~train_is_data

    mc_train_sumw = np.sum(full_weights[idx_train][train_is_mc], dtype=np.float64)
    n_data_train = int(np.sum(train_is_data))

    if mc_train_sumw <= 0:
        raise ValueError("Training MC sum of weights must be positive.")

    class_scale = n_data_train / mc_train_sumw

    full_weights[n_data:] *= class_scale

    return full_weights.astype(np.float32), float(class_scale)


def train_one_c2st(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    mc_physical_weights: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    """
    Train and evaluate one classifier two-sample test.

    The only thing that changes between calls is `mc_physical_weights`.

    Therefore any change in AUC is caused by how the MC weighting changes the
    distribution seen by the classifier, not by changing events or splits.
    """
    sample_weight, class_scale = make_balanced_sample_weights(
        y=y,
        idx_train=idx_train,
        mc_physical_weights=mc_physical_weights,
    )

    model = build_model(
        n_features=X.shape[1],
        seed=seed,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print(f"\n{'=' * 80}")
    print(f"Training C2ST: {name}")
    print(f"{'=' * 80}")
    print(f"train events: {len(idx_train):,}")
    print(f"val events:   {len(idx_val):,}")
    print(f"test events:  {len(idx_test):,}")
    print(f"MC class-balancing scale: {class_scale:.6g}")

    history = model.fit(
        X[idx_train],
        y[idx_train],
        sample_weight=sample_weight[idx_train],
        validation_data=(
            X[idx_val],
            y[idx_val],
            sample_weight[idx_val],
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    p_test = model.predict(
        X[idx_test],
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    y_test = y[idx_test]
    w_test = sample_weight[idx_test]

    auc = roc_auc_score(
        y_test,
        p_test,
        sample_weight=w_test,
    )

    print(f"\nHeld-out weighted AUC ({name}): {auc:.5f}")

    return {
        "name": name,
        "model": model,
        "history": history.history,
        "p_test": p_test.astype(np.float32),
        "y_test": y_test,
        "w_test": w_test,
        "auc": float(auc),
        "class_scale": class_scale,
    }


# ======================================================================================
# 3. Plotting helpers
# ======================================================================================

def normalized_histogram(
    values: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Histogram normalized to unit integral.

    These plots deliberately compare SHAPES. The C2ST itself is also class-balanced,
    so this visualization matches the conceptual question being asked by the NN.
    """
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    total = np.sum(counts)
    if total > 0:
        counts = counts / total
    return counts


def plot_input_features(
    data: np.ndarray,
    mc: np.ndarray,
    w_nominal: np.ndarray,
    w_closure: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot x1 and x2 before/after weighting.

    This makes the toy problem understandable before looking at any NN output.
    """
    for i, label in enumerate(["x1", "x2"]):
        all_values = np.concatenate([data[:, i], mc[:, i]])
        lo, hi = np.quantile(all_values, [0.002, 0.998])
        bins = np.linspace(lo, hi, 61)
        centers = 0.5 * (bins[:-1] + bins[1:])

        h_data = normalized_histogram(data[:, i], bins)
        h_raw = normalized_histogram(mc[:, i], bins)
        h_nominal = normalized_histogram(mc[:, i], bins, w_nominal)
        h_closure = normalized_histogram(mc[:, i], bins, w_closure)

        fig, (ax, ratio) = plt.subplots(
            2,
            1,
            figsize=(7, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )

        ax.step(centers, h_data, where="mid", label="Data", color="k")
        ax.step(centers, h_raw, where="mid", label="MC raw", color="0.5")
        ax.step(centers, h_nominal, where="mid", label="MC nominal", color="tab:red")
        ax.step(centers, h_closure, where="mid", label="MC closure", color="darkorange")
        ax.set_ylabel("normalized events")
        ax.set_title(f"Toy input feature: {label}")
        ax.legend()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio.step(
                centers,
                np.divide(
                    h_raw,
                    h_data,
                    out=np.full_like(h_raw, np.nan),
                    where=h_data > 0,
                ),
                where="mid",
                color="0.5",
                label="raw",
            )
            ratio.step(
                centers,
                np.divide(
                    h_nominal,
                    h_data,
                    out=np.full_like(h_nominal, np.nan),
                    where=h_data > 0,
                ),
                where="mid",
                color="tab:red",
                label="nominal",
            )
            ratio.step(
                centers,
                np.divide(
                    h_closure,
                    h_data,
                    out=np.full_like(h_closure, np.nan),
                    where=h_data > 0,
                ),
                where="mid",
                color="darkorange",
                label="closure",
            )

        ratio.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ratio.set_ylabel("MC / Data")
        ratio.set_xlabel(label)
        ratio.set_ylim(0.5, 1.5)

        fig.tight_layout()
        fig.savefig(output_dir / f"input_{label}.png", dpi=160)
        plt.close(fig)


def plot_roc(
    nominal: dict,
    closure: dict,
    output_dir: Path,
) -> None:
    """Plot the two held-out ROC curves."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for result, color in [
        (nominal, "tab:red"),
        (closure, "darkorange"),
    ]:
        fpr, tpr, _ = roc_curve(
            result["y_test"],
            result["p_test"],
            sample_weight=result["w_test"],
        )
        ax.plot(
            fpr,
            tpr,
            label=f"{result['name']} (AUC={result['auc']:.3f})",
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Classifier two-sample test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_comparison.png", dpi=160)
    plt.close(fig)


def plot_classifier_scores(
    nominal: dict,
    closure: dict,
    output_dir: Path,
) -> None:
    """
    Plot P(Data) for Data and MC.

    When closure is good, the two score distributions should nearly overlap and
    the NN output should have very little ranking power.
    """
    bins = np.linspace(0, 1, 61)

    for result, color in [
        (nominal, "tab:red"),
        (closure, "darkorange"),
    ]:
        y = result["y_test"]
        p = result["p_test"]
        w = result["w_test"]

        is_data = y == 1
        is_mc = y == 0

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            p[is_data],
            bins=bins,
            weights=w[is_data],
            density=True,
            histtype="step",
            linewidth=1.5,
            color="k",
            label="Data",
        )
        ax.hist(
            p[is_mc],
            bins=bins,
            weights=w[is_mc],
            density=True,
            histtype="step",
            linewidth=1.5,
            color=color,
            label=result["name"],
        )

        ax.set_xlabel("NN output P(Data)")
        ax.set_ylabel("density")
        ax.set_title(f"{result['name']}: classifier scores, AUC={result['auc']:.3f}")
        ax.legend()
        fig.tight_layout()
        safe_name = result["name"].lower().replace(" ", "_")
        fig.savefig(output_dir / f"classifier_scores_{safe_name}.png", dpi=160)
        plt.close(fig)


def plot_weight_distributions(
    w_nominal: np.ndarray,
    w_closure: np.ndarray,
    output_dir: Path,
) -> None:
    """Inspect the toy MC correction factors themselves."""
    combined = np.concatenate([w_nominal, w_closure])
    xmax = np.quantile(combined, 0.995)
    bins = np.linspace(0, xmax, 80)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        w_nominal,
        bins=bins,
        histtype="step",
        density=True,
        label="nominal MC weight",
        color="tab:red",
    )
    ax.hist(
        w_closure,
        bins=bins,
        histtype="step",
        density=True,
        label="closure MC weight",
        color="darkorange",
    )
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("multiplicative MC weight")
    ax.set_ylabel("density")
    ax.set_title("Toy MC reweighting factors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mc_weight_distributions.png", dpi=160)
    plt.close(fig)


# ======================================================================================
# 4. Main toy experiment
# ======================================================================================

def main() -> None:
    args = parse_args()

    if args.events < 1_000:
        raise ValueError("--events should be at least 1000 for a meaningful demonstration.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility for NumPy and TensorFlow.
    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    print("\nGenerating toy samples...")
    print(f"  Data events: {args.events:,}")
    print(f"  MC events:   {args.events:,}")

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    # "Data" is sampled from the target distribution.
    data = rng.multivariate_normal(
        mean=DATA_MEAN,
        cov=COVARIANCE,
        size=args.events,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # RAW MC
    # ------------------------------------------------------------------
    # MC comes from a displaced distribution.
    mc = rng.multivariate_normal(
        mean=MC_MEAN,
        cov=COVARIANCE,
        size=args.events,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # NOMINAL MC WEIGHT
    # ------------------------------------------------------------------
    # Pretend that the analysis already has a known correction, but that this
    # correction is incomplete. It reweights raw MC toward an intermediate
    # Gaussian, not all the way to Data.
    #
    # This is analogous to a nominal physics correction that improves agreement
    # but leaves residual mismodeling.
    w_nominal = gaussian_density_ratio(
        mc,
        target_mean=NOMINAL_TARGET_MEAN,
        source_mean=MC_MEAN,
        covariance=COVARIANCE,
    )

    # ------------------------------------------------------------------
    # CLOSURE MC WEIGHT
    # ------------------------------------------------------------------
    # This is the exact toy density ratio p_data(x)/p_mc(x).
    #
    # By construction, weighting MC with this factor should reproduce the Data
    # feature density, up to finite-sample fluctuations.
    #
    # In the real problem this exact ratio is unknown. A classifier-based
    # reweighter such as DCTR attempts to ESTIMATE it from Data and MC.
    w_closure = gaussian_density_ratio(
        mc,
        target_mean=DATA_MEAN,
        source_mean=MC_MEAN,
        covariance=COVARIANCE,
    )

    # Keep weights in float32 for the ML pipeline.
    w_nominal = w_nominal.astype(np.float32)
    w_closure = w_closure.astype(np.float32)

    print("\nToy MC-weight diagnostics")
    for name, weights in [
        ("nominal", w_nominal),
        ("closure", w_closure),
    ]:
        print(
            f"  {name:8s}: "
            f"mean={np.mean(weights):.4f}, "
            f"median={np.median(weights):.4f}, "
            f"max={np.max(weights):.4f}, "
            f"Kish N_eff={kish_effective_sample_size(weights):,.0f}"
        )

    # ------------------------------------------------------------------
    # Build the NN input matrix.
    # ------------------------------------------------------------------
    #
    # The classifier gets only event features. It does NOT receive the MC weight
    # as an input feature.
    #
    # The event weight enters only as sample_weight in model.fit().
    X = np.vstack([data, mc]).astype(np.float32)

    idx_train, idx_val, idx_test, y = make_common_split(
        n_data=len(data),
        n_mc=len(mc),
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # C2ST 1: Data vs incompletely corrected MC
    # ------------------------------------------------------------------
    nominal_result = train_one_c2st(
        name="MC nominal",
        X=X,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        mc_physical_weights=w_nominal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # C2ST 2: Data vs almost perfectly corrected MC
    # ------------------------------------------------------------------
    closure_result = train_one_c2st(
        name="MC closure",
        X=X,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        mc_physical_weights=w_closure,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )

    # ------------------------------------------------------------------
    # Save plots.
    # ------------------------------------------------------------------
    plot_input_features(
        data=data,
        mc=mc,
        w_nominal=w_nominal,
        w_closure=w_closure,
        output_dir=args.output_dir,
    )

    plot_roc(
        nominal=nominal_result,
        closure=closure_result,
        output_dir=args.output_dir,
    )

    plot_classifier_scores(
        nominal=nominal_result,
        closure=closure_result,
        output_dir=args.output_dir,
    )

    plot_weight_distributions(
        w_nominal=w_nominal,
        w_closure=w_closure,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------
    # Save a compact machine-readable summary.
    # ------------------------------------------------------------------
    summary = {
        "seed": args.seed,
        "events_per_class": args.events,
        "data_mean": DATA_MEAN.tolist(),
        "mc_mean": MC_MEAN.tolist(),
        "nominal_target_mean": NOMINAL_TARGET_MEAN.tolist(),
        "auc_nominal": nominal_result["auc"],
        "auc_closure": closure_result["auc"],
        "nominal_class_scale": nominal_result["class_scale"],
        "closure_class_scale": closure_result["class_scale"],
        "nominal_kish_neff": kish_effective_sample_size(w_nominal),
        "closure_kish_neff": kish_effective_sample_size(w_closure),
    }

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Final human-readable interpretation.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL C2ST RESULT")
    print("=" * 80)
    print(f"Data vs nominal MC : AUC = {nominal_result['auc']:.5f}")
    print(f"Data vs closure MC : AUC = {closure_result['auc']:.5f}")
    print()
    print("Interpretation:")
    print("  * AUC = 0.5 corresponds to no classifier-visible separation.")
    print("  * The nominal weighting deliberately leaves a residual mismatch,")
    print("    so its AUC should be noticeably above 0.5.")
    print("  * The exact closure weighting makes weighted MC follow the Data")
    print("    density, so its AUC should approach 0.5 within finite statistics")
    print("    and ordinary NN-training fluctuations.")
    print()
    print(f"Plots and summary written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
