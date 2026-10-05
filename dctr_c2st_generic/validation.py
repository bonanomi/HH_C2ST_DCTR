"""Automatic loss, factor, AUC and physical/shape closure diagnostics."""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import write_json
from .numerics import factor_summary


def filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def validate_channel(directory, cfg):
    directory = Path(directory)
    plotdir = directory / "plots"
    plotdir.mkdir(exist_ok=True)
    frame = pd.read_parquet(directory / "outer_test_fold.parquet")
    stages = ["before", *cfg["reference_stages"], "dctr"]
    target, base = frame.y.to_numpy() == 1, frame.y.to_numpy() == 0
    corrected = base & frame.corrected.to_numpy(bool)
    diag = json.loads((directory / "dctr_final_diagnostics.json").read_text())
    factors = frame.loc[corrected, "dctr_factor"].to_numpy()
    write_json(plotdir / "factor_summary.json", factor_summary(
        factors, frame.loc[corrected, "weight_before"].to_numpy(), diag["cap"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(np.log10(factors), bins=cfg["plot_bins"], histtype="step")
    ax.set(xlabel="log10(applied target/base factor)", ylabel="Corrected-base events", title="Untouched outer test")
    fig.tight_layout(); fig.savefig(plotdir / "factors.png", dpi=140); plt.close(fig)
    for path in sorted(directory.glob("*_history.json")):
        record = json.loads(path.read_text())
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for key in ("loss", "val_loss"):
            values = record["history"][key]
            ax.plot(np.arange(1, len(values) + 1), values, label=key)
        ax.set(xlabel="Epoch", ylabel="Weighted BCE", title=path.stem)
        ax.legend(); fig.tight_layout(); fig.savefig(plotdir / f"{path.stem}.png", dpi=140); plt.close(fig)
    aucs = [json.loads((directory / f"closure_{s}_metrics.json").read_text())["auc"] for s in stages]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.scatter(stages, aucs); ax.axhline(.5, color="k", ls="--")
    ax.set(ylabel="Closure AUC (closer to 0.5 is better)", title="Positive-weight outer test")
    fig.tight_layout(); fig.savefig(plotdir / "auc.png", dpi=140); plt.close(fig)
    fractions = json.loads((directory / "yield_closure.json").read_text())
    ft, fb = fractions["test_fraction_target"], fractions["test_fraction_base"]
    for j, feature in enumerate(dict.fromkeys(cfg["features"] + cfg["validation_vars"])):
        values = frame[feature].to_numpy(dtype=float)
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low - .5, high + .5
        edges = np.linspace(low, high, cfg["plot_bins"] + 1)
        for mode in ("physical", "shape"):
            fig, (ax, ratio) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
            # Target reference stages may carry alternative weights, so plot each
            # stage's corresponding target and compute its own ratio.
            for stage in stages:
                w = frame[f"weight_{stage}"].to_numpy(dtype=float)
                ht = np.histogram(values[target], edges, weights=w[target] / ft)[0]
                hb = np.histogram(values[base], edges, weights=w[base] / fb)[0]
                et = np.sqrt(np.histogram(values[target], edges, weights=(w[target] / ft)**2)[0])
                eb = np.sqrt(np.histogram(values[base], edges, weights=(w[base] / fb)**2)[0])
                if mode == "shape":
                    scale = ht.sum() / hb.sum() if hb.sum() else np.nan
                    hb, eb = hb * scale, eb * abs(scale)
                line = ax.stairs(hb, edges, label=f"base: {stage}")
                color = line.get_edgecolor()
                centers = (edges[1:] + edges[:-1]) / 2
                ax.errorbar(centers, ht, yerr=et, fmt=".", color=color, alpha=.6, label=f"target: {stage}")
                r = np.divide(hb, ht, out=np.full_like(hb, np.nan), where=ht != 0)
                error = np.sqrt(np.divide(eb**2, ht**2, out=np.full_like(hb, np.nan), where=ht != 0)
                                + np.divide(hb**2 * et**2, ht**4, out=np.full_like(hb, np.nan), where=ht != 0))
                ratio.errorbar(centers, r, yerr=error, fmt=".", color=color)
            ax.set(ylabel="Weighted events (test-fraction corrected)", title=f"{feature}: {mode} closure")
            ax.legend(fontsize=8, ncol=2)
            ratio.axhline(1., color="k", ls="--"); ratio.set(xlabel=feature, ylabel="Base/target")
            fig.tight_layout(); fig.savefig(plotdir / f"{j:03d}_{filename(feature)}_{mode}.png", dpi=140); plt.close(fig)
    write_json(plotdir / "README.json", {
        "population": "Positive-weight classifier outer test; NOT full signed-MC closure",
        "uncertainties": "sumw2 only; no learned-factor uncertainty or cross-bin covariance",
        "yield_closure": "../yield_closure.json", "normalization": cfg["normalization"],
        "interpretation": "AUC is a shape diagnostic. Inspect physical yields, signed-loss stability, factor quantiles, weighted mean, and cap fraction separately.",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    args = ap.parse_args()
    cfg = json.loads((args.input / "config.json").read_text())
    for channel in cfg["channels"]:
        validate_channel(args.input / channel, cfg)


if __name__ == "__main__":
    main()
