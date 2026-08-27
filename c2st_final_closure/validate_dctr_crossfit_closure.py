from __future__ import annotations

"""Independent plots and quantitative checks for train_dctr_crossfit_closure.py."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2st_config as cfg
from validation_utils import paired_bootstrap_delta_auc

DEFAULT_ROOT = cfg.ARTIFACT_DIR / "dctr_crossfit_closure"
STAGES = ("before", "dy", "dctr")
COLORS = {"before": "tab:blue", "dy": "tab:red", "dctr": "darkorange"}
LABELS = {"before": "MC before", "dy": "MC DY corrected", "dctr": "MC DCTR"}


def load_channel(root: Path, channel: str):
    channel_dir = root / channel
    test = pd.read_parquet(channel_dir / "outer_test_fold.parquet")
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    stages = {}
    for stage in STAGES:
        arr = np.load(channel_dir / f"closure_{stage}_test.npz", mmap_mode="r")
        p = np.asarray(arr["p_test"], dtype=np.float32)
        w = np.asarray(arr["w_test"], dtype=np.float32)
        if len(p) != len(test) or len(w) != len(test):
            raise ValueError(f"{channel}/{stage}: prediction/test-fold length mismatch")
        stages[stage] = {"p": p, "w": w}
    return test, y, stages


def equal_width_edges(values, bins: int, plot_range=None):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("No finite values available")
    if plot_range is None:
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = map(float, plot_range)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid range ({lo}, {hi})")
    return np.linspace(lo, hi, bins + 1)


def normalize_to_data(data, mc):
    ds = np.sum(data, dtype=np.float64)
    ms = np.sum(mc, dtype=np.float64)
    if ds <= 0 or ms <= 0:
        return mc.astype(np.float64, copy=False), np.nan
    scale = ds / ms
    return mc * scale, scale


def plot_auc_summary(root: Path, channels, outdir: Path):
    rows = []
    for channel in channels:
        comp = json.loads((root / channel / "comparison.json").read_text())
        for stage in STAGES:
            rows.append({"channel": channel, "stage": stage, "auc": comp[f"auc_{stage}"]})
    df = pd.DataFrame(rows)
    x = np.arange(len(channels), dtype=float)
    width = 0.23
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for j, stage in enumerate(STAGES):
        vals = [float(df[(df.channel == ch) & (df.stage == stage)].auc.iloc[0]) for ch in channels]
        ax.bar(x + (j - 1) * width, vals, width=width, label=stage, color=COLORS[stage])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(x, channels)
    ax.set_ylabel("weighted test AUC")
    ax.set_ylim(0.48, max(0.65, float(df.auc.max()) + 0.03))
    ax.set_title("Independent closure C2ST: lower is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "closure_auc_comparison.png", dpi=170)
    plt.close(fig)
    df.to_csv(outdir / "closure_auc_comparison.csv", index=False)
    return df


def plot_classifier_scores(channel, y, stages, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    bins = np.linspace(0, 1, 61)
    for ax, stage in zip(axes, STAGES):
        p, w = stages[stage]["p"], stages[stage]["w"]
        is_data, is_mc = y == 1, y == 0
        ax.hist(p[is_data], bins=bins, weights=w[is_data], density=True,
                histtype="step", label="Data", color="k")
        ax.hist(p[is_mc], bins=bins, weights=w[is_mc], density=True,
                histtype="step", label=LABELS[stage], color=COLORS[stage])
        ax.set_title(stage)
        ax.set_xlabel("closure classifier P(Data)")
        ax.legend()
    axes[0].set_ylabel("normalized weighted density")
    fig.suptitle(f"{channel}: closure-classifier outputs")
    fig.tight_layout()
    fig.savefig(outdir / f"closure_scores_{channel}.png", dpi=170)
    plt.close(fig)


def plot_rocs(channel, y, stages, outdir):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for stage in STAGES:
        p, w = stages[stage]["p"], stages[stage]["w"]
        fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
        # Read AUC from the saved metrics so no convention mismatch is introduced here.
        metrics = json.loads((DEFAULT_ROOT / channel / f"closure_{stage}_metrics.json").read_text()) if DEFAULT_ROOT.exists() else None
        label = stage if metrics is None else f"{stage} (AUC={metrics['auc']:.3f})"
        ax.plot(fpr, tpr, label=label, color=COLORS[stage])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{channel}: independent closure ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"closure_roc_{channel}.png", dpi=170)
    plt.close(fig)


def plot_feature_closure(channel, test, var, bins, normalization, outdir):
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    values = test[var].to_numpy(dtype=np.float32, copy=False)
    raw_before = test["weight_uncorrected"].to_numpy(dtype=np.float32, copy=False)
    raw_dy = test["weight"].to_numpy(dtype=np.float32, copy=False)
    raw_dctr = test["weight_dctr"].to_numpy(dtype=np.float32, copy=False)

    finite = np.isfinite(values)
    in_range = finite & (values >= bins[0]) & (values <= bins[-1])
    is_data = (y == 1) & in_range
    is_mc = (y == 0) & in_range

    data, _ = np.histogram(values[is_data], bins=bins, weights=raw_before[is_data])
    mc_before, _ = np.histogram(values[is_mc], bins=bins, weights=raw_before[is_mc])
    mc_dy, _ = np.histogram(values[is_mc], bins=bins, weights=raw_dy[is_mc])
    mc_dctr, _ = np.histogram(values[is_mc], bins=bins, weights=raw_dctr[is_mc])
    scales = {"before": 1.0, "dy": 1.0, "dctr": 1.0}
    if normalization == "shape":
        mc_before, scales["before"] = normalize_to_data(data, mc_before)
        mc_dy, scales["dy"] = normalize_to_data(data, mc_dy)
        mc_dctr, scales["dctr"] = normalize_to_data(data, mc_dctr)

    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, (ax, ratio) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.step(centers, data, where="mid", label="Data", color="k")
    ax.step(centers, mc_before, where="mid", label="MC before", color="tab:blue")
    ax.step(centers, mc_dy, where="mid", label="MC DY corrected", color="tab:red")
    ax.step(centers, mc_dctr, where="mid", label="MC DCTR", color="darkorange")
    ax.set_ylabel("weighted events")
    suffix = "shape-normalized" if normalization == "shape" else "physical normalization"
    ax.set_title(f"{channel}: outer-test closure on {var} ({suffix})")
    ax.legend()

    with np.errstate(divide="ignore", invalid="ignore"):
        for arr, stage, label in [
            (mc_before, "before", "before"),
            (mc_dy, "dy", "DY corr"),
            (mc_dctr, "dctr", "DCTR"),
        ]:
            ratio.step(
                centers,
                np.divide(arr, data, out=np.full_like(arr, np.nan, dtype=float), where=data > 0),
                where="mid", label=label, color=COLORS[stage],
            )
    ratio.axhline(1.0, color="k", ls="--", lw=1)
    ratio.set_ylim(0.5, 1.5)
    ratio.set_ylabel("MC/Data")
    ratio.set_xlabel(var)
    fig.tight_layout()
    fig.savefig(outdir / f"outer_test_feature_{channel}_{var}.png", dpi=170)
    plt.close(fig)
    return scales


def plot_dctr_factors(channel, test, outdir, bins=100, qmax=0.999):
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    f = test["dctr_factor"].to_numpy(dtype=np.float32, copy=False)[y == 0]
    f = f[np.isfinite(f) & (f >= 0)]
    xmax = float(np.quantile(f, qmax))
    edges = np.linspace(0, max(xmax, np.finfo(np.float32).eps), bins + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(f, bins=edges, density=True, histtype="step", color="darkorange")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("out-of-sample DCTR factor")
    ax.set_ylabel("normalized MC density")
    ax.set_title(f"{channel}: DCTR factors on untouched outer test")
    fig.tight_layout()
    fig.savefig(outdir / f"outer_test_dctr_factors_{channel}.png", dpi=170)
    plt.close(fig)


def paired_bootstrap_table(channel, y, stages, n_resamples, subsample, seed):
    rows = []
    for a, b in [("before", "dy"), ("before", "dctr"), ("dy", "dctr")]:
        res = paired_bootstrap_delta_auc(
            y,
            stages[a]["p"], stages[a]["w"],
            stages[b]["p"], stages[b]["w"],
            n_resamples=n_resamples,
            random_state=seed,
            subsample=subsample,
        )
        rows.append({
            "channel": channel,
            "from": a,
            "to": b,
            "auc_from": res["auc_before"],
            "auc_to": res["auc_after"],
            "delta_auc": res["delta_auc_observed"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
            "excludes_zero": res["excludes_zero"],
        })
    return rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--channels", nargs="+", default=cfg.CHANNELS)
    ap.add_argument("--vars", nargs="+", default=["mli_ll_pt", "mli_n_jet"])
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--range", dest="plot_range", nargs=2, type=float, default=None,
                    metavar=("MIN", "MAX"),
                    help="one common range for all --vars; omit for each variable's finite min/max")
    ap.add_argument("--normalization", choices=["shape", "physical"], default="shape")
    ap.add_argument("--bootstrap-resamples", type=int, default=50)
    ap.add_argument("--bootstrap-subsample", type=int, default=0,
                    help="0 = full outer test; otherwise fixed paired subsample for faster iteration")
    ap.add_argument("--output", type=Path, default=cfg.PLOT_DIR / "dctr_crossfit_closure")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    auc_df = plot_auc_summary(args.root, args.channels, args.output)
    print("\nAUC summary")
    print(auc_df.to_string(index=False))

    bootstrap_rows = []
    scale_rows = []
    for channel in args.channels:
        test, y, stages = load_channel(args.root, channel)
        plot_classifier_scores(channel, y, stages, args.output)

        # ROC plotting reads metrics from args.root explicitly here (rather than relying on globals).
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for stage in STAGES:
            fpr, tpr, _ = roc_curve(y, stages[stage]["p"], sample_weight=stages[stage]["w"])
            metrics = json.loads((args.root / channel / f"closure_{stage}_metrics.json").read_text())
            ax.plot(fpr, tpr, label=f"{stage} (AUC={metrics['auc']:.3f})", color=COLORS[stage])
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{channel}: independent closure ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / f"closure_roc_{channel}.png", dpi=170)
        plt.close(fig)

        plot_dctr_factors(channel, test, args.output)

        for var in args.vars:
            if var not in test.columns:
                print(f"[{channel}] {var} not saved in outer test; skipping")
                continue
            bins = equal_width_edges(test[var].to_numpy(), args.bins, args.plot_range)
            scales = plot_feature_closure(
                channel, test, var, bins, args.normalization, args.output
            )
            scale_rows.append({
                "channel": channel,
                "var": var,
                "range_min": bins[0],
                "range_max": bins[-1],
                "normalization": args.normalization,
                **{f"scale_{k}": v for k, v in scales.items()},
            })

        bootstrap_rows.extend(paired_bootstrap_table(
            channel,
            y,
            stages,
            n_resamples=args.bootstrap_resamples,
            subsample=(None if args.bootstrap_subsample <= 0 else args.bootstrap_subsample),
            seed=cfg.RANDOM_STATE,
        ))

    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv(args.output / "paired_auc_bootstrap.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(args.output / "feature_shape_scales.csv", index=False)
    print("\nPaired AUC differences (to - from)")
    print(boot_df.to_string(index=False))
    print(f"\nPlots/tables written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
