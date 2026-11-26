"""Raw final-DCTR train/outer-test scores (not closure-C2ST predictions)."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import validate
from .plot_signed_physical import OBSERVABLES, load_signed_cached, match_rows
from .training import transform


def run(root, out, cache, bins=60, chunk=131072):
    import tensorflow as tf
    bundle = root / "inference"
    meta = json.loads((bundle / "metadata.json").read_text())
    cfg = validate(meta["config"])
    features = meta["features"]
    if features != cfg["features"] or cfg["max_events_per_class"] is not None:
        raise ValueError("Expected matching feature lists and a non-truncated run")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    load_cfg = dict(cfg, validation_vars=list(dict.fromkeys(
        cfg["validation_vars"] + [v for v in OBSERVABLES if v not in features])))
    model = tf.keras.models.load_model(bundle / "model.keras", compile=False)
    scaler = joblib.load(bundle / "scaler.joblib")
    edges = np.linspace(0.4, 0.8, bins + 1)
    hist = {(s, c): np.zeros((3, bins)) for s in ("train", "test") for c in (0, 1)}
    with np.load(root / "splits.npz", allow_pickle=False) as splits:
        for side, roles in (("target", ("target",)), ("base", ("subtract", "base"))):
            frame = load_signed_cached(load_cfg, side, root.name, cache)
            saved = pd.read_parquet(root / f"{side}_rows.parquet")
            pos = match_rows(frame, saved, side, signed=cfg.get("dctr_objective") == "signed_mse")
            test = splits[f"{side}_test"]
            for role in roles:
                train = splits[f"final_train_{role}"]
                hold = splits[f"final_hold_{role}"]
                if np.intersect1d(train, test).size or not np.isin(hold, test).all():
                    raise ValueError("Inconsistent final-train/outer-test assignment")
                label = int(role != "base")
                for split, idx in (("train", train), ("test", hold)):
                    acc = hist[split, label]
                    for offset in range(0, len(idx), chunk):
                        rows = frame.iloc[pos[idx[offset:offset+chunk]]]
                        x = transform(rows, scaler, features)
                        p = model.predict(x, batch_size=meta["batch_size"], verbose=0).reshape(-1)
                        if len(p) != len(rows) or not np.all(np.isfinite(p) & (p>=0) & (p<=1)):
                            raise ValueError("Invalid sigmoid scores")
                        w = rows.weight_before.to_numpy(dtype=float)
                        if role == "subtract":
                            w = -w
                        for a, weights in zip(acc, (None, w, w*w)):
                            a += np.histogram(p, edges, weights=weights)[0]
            del frame, saved, pos
    render(hist, edges, out, bool(cfg["subtract_processes"]), meta.get("cap"))


def render(hist, edges, out, subtraction, cap):
    records = []
    for (s, c), (n, h, h2) in hist.items():
        records.append(pd.DataFrame(dict(split=s,label=c,low=edges[:-1],high=edges[1:],
                                         entries=n.astype(int),sumw=h,sumw2=h2)))
    pd.concat(records, ignore_index=True).to_csv(out/"score_bins.csv",index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11,7), sharex=True, constrained_layout=True)
    for c in (0,1):
        title = (("DY", "Data minus non-DY")[c] if subtraction else ("Base","Target")[c])
        for row, measure in enumerate(("Entries", "Signed yield")):
            ax = axes[row,c]
            for s, color, style in (("train","C0","-"),("test","C1","--")):
                n,h,h2 = hist[s,c]
                y = n if row==0 else h
                total = y.sum()
                if total <= 0:
                    raise ValueError(f"Cannot unit-normalize {s=} {c=} {measure}, see score_bins.csv")
                ax.stairs(y/total, edges, color=color, linestyle=style, label=s)
                if s=="test":
                    # Sumw2 error with the observed total treated as fixed.
                    err=np.sqrt(n if row==0 else h2)/total
                    ax.fill_between(edges, np.r_[y/total-err,(y/total-err)[-1]],
                                    np.r_[y/total+err,(y/total+err)[-1]],
                                    step="post",color=color,alpha=.18)
            if cap is not None:
                ax.axvline(cap/(1+cap),color="0.45",ls=":",label="final cap threshold")
            ax.axhline(0,color="0.75",lw=.6)
            # Raw counts do not subtract background events in the residual class.
            ax.set_title(("Corrected base" if c==0 else "Target + subtraction rows")
                         if row==0 else title)
            ax.set_ylabel(measure+" fraction / bin")
            ax.set_xlim(edges[0], edges[-1])
            ax.legend(fontsize=8)
            if row==1: ax.set_xlabel("Raw final DCTR sigmoid score p")
    fig.savefig(out/"dctr_score_train_test.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache-dir",type=Path,default=Path("c2st_cache/signed_physical"))
    parser.add_argument("--bins",type=int,default=60)
    a=parser.parse_args()
    if a.bins<2: parser.error("--bins must be at least 2")
    run(a.channel_dir,a.output,a.cache_dir,a.bins)
