from __future__ import annotations

"""Independent plots/checks for the cross-fitted DCTR closure study.

Use ``--dctr-target inclusive`` or ``dy_only`` to validate one training.  Use
``--dctr-target both`` after both trainings exist to compare the official DY,
inclusive-DCTR and DY-only-DCTR prescriptions on the common outer test.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c2st_config as cfg
from validation_utils import paired_bootstrap_delta_auc

DEFAULT_ROOT = cfg.ARTIFACT_DIR / "dctr_crossfit_closure"
BASE_STAGES = ("before", "dy", "dctr")
SINGLE_COLORS = {"before": "tab:blue", "dy": "tab:red", "dctr": "darkorange"}
SINGLE_LABELS = {"before": "MC before", "dy": "MC DY corrected", "dctr": "MC DCTR"}
COMPARE_STAGES = ("before", "dy", "dctr_inclusive", "dctr_dy_only")
COMPARE_COLORS = {
    "before": "tab:blue",
    "dy": "tab:red",
    "dctr_inclusive": "darkorange",
    "dctr_dy_only": "tab:green",
}
COMPARE_LABELS = {
    "before": "MC before",
    "dy": "MC official DY",
    "dctr_inclusive": "MC inclusive DCTR",
    "dctr_dy_only": "MC DY-only DCTR",
}


def target_root(root: Path, target: str) -> Path:
    return root if target == "inclusive" else root / target


def load_single(root: Path, channel: str, target: str):
    base = target_root(root, target)
    channel_dir = base / channel
    test = pd.read_parquet(channel_dir / "outer_test_fold.parquet")
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    stages = {}
    for stage in BASE_STAGES:
        arr = np.load(channel_dir / f"closure_{stage}_test.npz", mmap_mode="r")
        p = np.asarray(arr["p_test"], dtype=np.float32)
        w = np.asarray(arr["w_test"], dtype=np.float32)
        if len(p) != len(test) or len(w) != len(test):
            raise ValueError(f"{target}/{channel}/{stage}: prediction/test-fold length mismatch")
        stages[stage] = {"p": p, "w": w}
    return test, y, stages, channel_dir


def load_both(root: Path, channel: str):
    test_i, y_i, stages_i, dir_i = load_single(root, channel, "inclusive")
    test_d, y_d, stages_d, dir_d = load_single(root, channel, "dy_only")
    if len(test_i) != len(test_d) or not np.array_equal(y_i, y_d):
        raise ValueError(f"{channel}: inclusive and DY-only outer tests are not aligned")
    for col in ("weight_uncorrected", "weight"):
        if not np.allclose(test_i[col].to_numpy(), test_d[col].to_numpy(), rtol=0, atol=1e-6):
            raise ValueError(f"{channel}: outer-test {col} differs between DCTR targets")
    stages = {
        "before": stages_i["before"],
        "dy": stages_i["dy"],
        "dctr_inclusive": stages_i["dctr"],
        "dctr_dy_only": stages_d["dctr"],
    }
    test = test_i.copy()
    test["weight_dctr_inclusive"] = test_i["weight_dctr"].to_numpy()
    test["weight_dctr_dy_only"] = test_d["weight_dctr"].to_numpy()
    test["dctr_factor_inclusive"] = test_i["dctr_factor"].to_numpy()
    test["dctr_factor_dy_only"] = test_d["dctr_factor"].to_numpy()
    return test, y_i, stages, {"inclusive": dir_i, "dy_only": dir_d}


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


def metrics_auc(channel_dir: Path, stage: str) -> float:
    return float(json.loads((channel_dir / f"closure_{stage}_metrics.json").read_text())["auc"])


def plot_auc_summary_single(root: Path, channels, target: str, outdir: Path):
    rows = []
    base = target_root(root, target)
    for channel in channels:
        comp = json.loads((base / channel / "comparison.json").read_text())
        for stage in BASE_STAGES:
            rows.append({"channel": channel, "stage": stage, "auc": comp[f"auc_{stage}"]})
    df = pd.DataFrame(rows)
    x = np.arange(len(channels), dtype=float)
    width = 0.23
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for j, stage in enumerate(BASE_STAGES):
        vals = [float(df[(df.channel == ch) & (df.stage == stage)].auc.iloc[0]) for ch in channels]
        ax.bar(x + (j - 1) * width, vals, width=width, label=stage, color=SINGLE_COLORS[stage])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(x, channels)
    ax.set_ylabel("weighted test AUC")
    ax.set_ylim(0.48, max(0.65, float(df.auc.max()) + 0.03))
    ax.set_title(f"Independent closure C2ST ({target} DCTR): lower is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "closure_auc_comparison.png", dpi=170)
    plt.close(fig)
    df.to_csv(outdir / "closure_auc_comparison.csv", index=False)
    return df


def plot_auc_summary_both(root: Path, channels, outdir: Path):
    rows = []
    for channel in channels:
        _, _, _, dirs = load_both(root, channel)
        values = {
            "before": metrics_auc(dirs["inclusive"], "before"),
            "dy": metrics_auc(dirs["inclusive"], "dy"),
            "dctr_inclusive": metrics_auc(dirs["inclusive"], "dctr"),
            "dctr_dy_only": metrics_auc(dirs["dy_only"], "dctr"),
        }
        rows.extend({"channel": channel, "stage": k, "auc": v} for k, v in values.items())
    df = pd.DataFrame(rows)
    x = np.arange(len(channels), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.arange(len(COMPARE_STAGES)) - (len(COMPARE_STAGES) - 1) / 2
    for j, stage in enumerate(COMPARE_STAGES):
        vals = [float(df[(df.channel == ch) & (df.stage == stage)].auc.iloc[0]) for ch in channels]
        ax.bar(x + offsets[j] * width, vals, width=width, label=COMPARE_LABELS[stage], color=COMPARE_COLORS[stage])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(x, channels)
    ax.set_ylabel("weighted test AUC")
    ax.set_ylim(0.48, max(0.65, float(df.auc.max()) + 0.03))
    ax.set_title("Independent closure C2ST: inclusive vs DY-only DCTR")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "closure_auc_comparison_both_targets.png", dpi=170)
    plt.close(fig)
    df.to_csv(outdir / "closure_auc_comparison_both_targets.csv", index=False)
    return df


def plot_classifier_scores(channel, y, stages, labels, colors, outdir):
    n = len(stages)
    ncols = 2 if n == 4 else n
    nrows = 2 if n == 4 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 if n == 4 else 15, 8 if n == 4 else 4.5), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(0, 1, 61)
    for ax, stage in zip(axes, stages):
        p, w = stages[stage]["p"], stages[stage]["w"]
        is_data, is_mc = y == 1, y == 0
        ax.hist(p[is_data], bins=bins, weights=w[is_data], density=True, histtype="step", label="Data", color="k")
        ax.hist(p[is_mc], bins=bins, weights=w[is_mc], density=True, histtype="step", label=labels[stage], color=colors[stage])
        ax.set_title(stage)
        ax.set_xlabel("closure classifier P(Data)")
        ax.legend()
    axes[0].set_ylabel("normalized weighted density")
    fig.suptitle(f"{channel}: closure-classifier outputs")
    fig.tight_layout()
    fig.savefig(outdir / f"closure_scores_{channel}.png", dpi=170)
    plt.close(fig)


def plot_roc(channel, y, stages, aucs, labels, colors, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for stage in stages:
        fpr, tpr, _ = roc_curve(y, stages[stage]["p"], sample_weight=stages[stage]["w"])
        ax.plot(fpr, tpr, label=f"{labels[stage]} (AUC={aucs[stage]:.3f})", color=colors[stage])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{channel}: independent closure ROC")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"closure_roc_{channel}.png", dpi=170)
    plt.close(fig)


def plot_feature_closure_single(channel, test, var, bins, normalization, outdir, target):
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    values = test[var].to_numpy(dtype=np.float32, copy=False)
    raw = {
        "before": test["weight_uncorrected"].to_numpy(dtype=np.float32, copy=False),
        "dy": test["weight"].to_numpy(dtype=np.float32, copy=False),
        "dctr": test["weight_dctr"].to_numpy(dtype=np.float32, copy=False),
    }
    finite = np.isfinite(values)
    in_range = finite & (values >= bins[0]) & (values <= bins[-1])
    is_data, is_mc = (y == 1) & in_range, (y == 0) & in_range
    data, _ = np.histogram(values[is_data], bins=bins, weights=raw["before"][is_data])
    hist = {stage: np.histogram(values[is_mc], bins=bins, weights=raw[stage][is_mc])[0] for stage in BASE_STAGES}
    scales = {stage: 1.0 for stage in BASE_STAGES}
    if normalization == "shape":
        for stage in BASE_STAGES:
            hist[stage], scales[stage] = normalize_to_data(data, hist[stage])
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.step(centers, data, where="mid", label="Data", color="k")
    for stage in BASE_STAGES:
        label = SINGLE_LABELS[stage] if stage != "dctr" else f"MC DCTR ({target})"
        ax.step(centers, hist[stage], where="mid", label=label, color=SINGLE_COLORS[stage])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio.step(centers, np.divide(hist[stage], data, out=np.full_like(hist[stage], np.nan, dtype=float), where=data > 0), where="mid", label=stage, color=SINGLE_COLORS[stage])
    suffix = "shape-normalized" if normalization == "shape" else "physical normalization"
    ax.set_ylabel("weighted events")
    ax.set_title(f"{channel}: outer-test closure on {var} ({suffix})")
    ax.legend()
    ratio.axhline(1.0, color="k", ls="--", lw=1)
    ratio.set_ylim(0.5, 1.5)
    ratio.set_ylabel("MC/Data")
    ratio.set_xlabel(var)
    fig.tight_layout()
    fig.savefig(outdir / f"outer_test_feature_{channel}_{var}.png", dpi=170)
    plt.close(fig)
    return scales


def plot_feature_closure_both(channel, test, var, bins, normalization, outdir):
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    values = test[var].to_numpy(dtype=np.float32, copy=False)
    raw = {
        "before": test["weight_uncorrected"].to_numpy(dtype=np.float32, copy=False),
        "dy": test["weight"].to_numpy(dtype=np.float32, copy=False),
        "dctr_inclusive": test["weight_dctr_inclusive"].to_numpy(dtype=np.float32, copy=False),
        "dctr_dy_only": test["weight_dctr_dy_only"].to_numpy(dtype=np.float32, copy=False),
    }
    finite = np.isfinite(values)
    in_range = finite & (values >= bins[0]) & (values <= bins[-1])
    is_data, is_mc = (y == 1) & in_range, (y == 0) & in_range
    data, _ = np.histogram(values[is_data], bins=bins, weights=raw["before"][is_data])
    hist = {stage: np.histogram(values[is_mc], bins=bins, weights=raw[stage][is_mc])[0] for stage in COMPARE_STAGES}
    scales = {stage: 1.0 for stage in COMPARE_STAGES}
    if normalization == "shape":
        for stage in COMPARE_STAGES:
            hist[stage], scales[stage] = normalize_to_data(data, hist[stage])
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.step(centers, data, where="mid", label="Data", color="k")
    for stage in COMPARE_STAGES:
        ax.step(centers, hist[stage], where="mid", label=COMPARE_LABELS[stage], color=COMPARE_COLORS[stage])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio.step(centers, np.divide(hist[stage], data, out=np.full_like(hist[stage], np.nan, dtype=float), where=data > 0), where="mid", label=COMPARE_LABELS[stage], color=COMPARE_COLORS[stage])
    suffix = "shape-normalized" if normalization == "shape" else "physical normalization"
    ax.set_ylabel("weighted events")
    ax.set_title(f"{channel}: inclusive vs DY-only DCTR on {var} ({suffix})")
    ax.legend(fontsize=8)
    ratio.axhline(1.0, color="k", ls="--", lw=1)
    ratio.set_ylim(0.5, 1.5)
    ratio.set_ylabel("MC/Data")
    ratio.set_xlabel(var)
    fig.tight_layout()
    fig.savefig(outdir / f"outer_test_feature_both_{channel}_{var}.png", dpi=170)
    plt.close(fig)
    return scales


def plot_dctr_factors_single(channel, test, outdir, target, bins=100, qmax=0.999):
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    mask = y == 0
    if target == "dy_only" and "is_dy" in test:
        mask &= test["is_dy"].to_numpy(dtype=bool, copy=False)
    f = test["dctr_factor"].to_numpy(dtype=np.float32, copy=False)[mask]
    f = f[np.isfinite(f) & (f >= 0)]
    xmax = float(np.quantile(f, qmax))
    edges = np.linspace(0, max(xmax, np.finfo(np.float32).eps), bins + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(f, bins=edges, density=True, histtype="step", color="darkorange")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("out-of-sample DCTR factor")
    ax.set_ylabel("normalized MC density")
    ax.set_title(f"{channel}: {target} DCTR factors on outer test")
    fig.tight_layout()
    fig.savefig(outdir / f"outer_test_dctr_factors_{channel}.png", dpi=170)
    plt.close(fig)


def plot_factor_comparison(channel, test, outdir):
    y = test["y"].to_numpy(dtype=np.uint8, copy=False)
    is_dy = test["is_dy"].to_numpy(dtype=bool, copy=False)
    mask = (y == 0) & is_dy
    fi = test["dctr_factor_inclusive"].to_numpy(dtype=float, copy=False)[mask]
    fd = test["dctr_factor_dy_only"].to_numpy(dtype=float, copy=False)[mask]
    finite = np.isfinite(fi) & np.isfinite(fd) & (fi >= 0) & (fd >= 0)
    fi, fd = fi[finite], fd[finite]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(fi, fd, s=4, alpha=0.2)
    lim = float(np.quantile(np.concatenate([fi, fd]), 0.995))
    lim = max(lim, 1.0)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("inclusive DCTR factor (DY events)")
    ax.set_ylabel("DY-only DCTR factor")
    ax.set_title(f"{channel}: DCTR factor comparison")
    fig.tight_layout()
    fig.savefig(outdir / f"dctr_factor_inclusive_vs_dy_only_{channel}.png", dpi=170)
    plt.close(fig)

    ratio = np.divide(fd, fi, out=np.full_like(fd, np.nan), where=fi > 0)
    ratio = ratio[np.isfinite(ratio)]
    pd.DataFrame({
        "channel": [channel],
        "n_dy_outer_test": [len(fi)],
        "inclusive_mean": [float(np.mean(fi))],
        "dy_only_mean": [float(np.mean(fd))],
        "ratio_median": [float(np.median(ratio))],
        "ratio_q16": [float(np.quantile(ratio, 0.16))],
        "ratio_q84": [float(np.quantile(ratio, 0.84))],
    }).to_csv(outdir / f"dctr_factor_comparison_{channel}.csv", index=False)


def paired_bootstrap_table(channel, y, stages, stage_order, n_resamples, subsample, seed):
    rows = []
    for a, b in combinations(stage_order, 2):
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
    ap.add_argument("--dctr-target", choices=("inclusive", "dy_only", "both"), default="inclusive")
    ap.add_argument("--channels", nargs="+", default=cfg.CHANNELS)
    ap.add_argument("--vars", nargs="+", default=["mli_ll_pt", "mli_n_jet"])
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--range", dest="plot_range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                    help="one common range for all --vars; omit for each variable's finite min/max")
    ap.add_argument("--normalization", choices=["shape", "physical"], default="shape")
    ap.add_argument("--bootstrap-resamples", type=int, default=50)
    ap.add_argument("--bootstrap-subsample", type=int, default=0,
                    help="0 = full outer test; otherwise fixed paired subsample for faster iteration")
    ap.add_argument("--output", type=Path, default=cfg.PLOT_DIR / "dctr_crossfit_closure")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = args.output / args.dctr_target if args.dctr_target != "inclusive" else args.output
    outdir.mkdir(parents=True, exist_ok=True)

    if args.dctr_target == "both":
        auc_df = plot_auc_summary_both(args.root, args.channels, outdir)
    else:
        auc_df = plot_auc_summary_single(args.root, args.channels, args.dctr_target, outdir)
    print("\nAUC summary")
    print(auc_df.to_string(index=False))

    bootstrap_rows = []
    scale_rows = []
    for channel in args.channels:
        if args.dctr_target == "both":
            test, y, stages, dirs = load_both(args.root, channel)
            stage_order = COMPARE_STAGES
            labels, colors = COMPARE_LABELS, COMPARE_COLORS
            aucs = {
                "before": metrics_auc(dirs["inclusive"], "before"),
                "dy": metrics_auc(dirs["inclusive"], "dy"),
                "dctr_inclusive": metrics_auc(dirs["inclusive"], "dctr"),
                "dctr_dy_only": metrics_auc(dirs["dy_only"], "dctr"),
            }
            plot_factor_comparison(channel, test, outdir)
        else:
            test, y, stages, channel_dir = load_single(args.root, channel, args.dctr_target)
            stage_order = BASE_STAGES
            labels, colors = SINGLE_LABELS, SINGLE_COLORS
            aucs = {stage: metrics_auc(channel_dir, stage) for stage in BASE_STAGES}
            plot_dctr_factors_single(channel, test, outdir, args.dctr_target)

        plot_classifier_scores(channel, y, stages, labels, colors, outdir)
        plot_roc(channel, y, stages, aucs, labels, colors, outdir)

        for var in args.vars:
            if var not in test.columns:
                print(f"[{channel}] {var} not saved in outer test; skipping")
                continue
            bins = equal_width_edges(test[var].to_numpy(), args.bins, args.plot_range)
            if args.dctr_target == "both":
                scales = plot_feature_closure_both(channel, test, var, bins, args.normalization, outdir)
            else:
                scales = plot_feature_closure_single(channel, test, var, bins, args.normalization, outdir, args.dctr_target)
            scale_rows.append({
                "channel": channel,
                "var": var,
                "range_min": bins[0],
                "range_max": bins[-1],
                "normalization": args.normalization,
                **{f"scale_{k}": v for k, v in scales.items()},
            })

        bootstrap_rows.extend(paired_bootstrap_table(
            channel, y, stages, stage_order,
            n_resamples=args.bootstrap_resamples,
            subsample=(None if args.bootstrap_subsample <= 0 else args.bootstrap_subsample),
            seed=cfg.RANDOM_STATE,
        ))

    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv(outdir / "paired_auc_bootstrap.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(outdir / "feature_shape_scales.csv", index=False)
    print("\nPaired AUC differences (to - from)")
    print(boot_df.to_string(index=False))
    print(f"\nPlots/tables written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
