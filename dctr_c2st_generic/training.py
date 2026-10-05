"""Generic cross-fit DCTR followed by fresh closure classifiers.

The compatibility profile preserves the original arithmetic and random seeds.
Generic profiles fit each DCTR scaler and normalization on its own training rows.
"""
from pathlib import Path
import gc
import sys
import platform
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from c2st_core import fit_scaler, apply_scaler, split_class_indices, weighted_bce
from .config import load_config, write_json
from .loading import load_side, prepare_tables
from .numerics import (sample_weights, closure_weights, odds, shuffled_folds,
                       inner_train_val, other_folds, factor_summary)


def fit_transformer(target, base, it, ib, cfg):
    frame = pd.concat([target.iloc[it][cfg["features"]], base.iloc[ib][cfg["features"]]], ignore_index=True)
    if cfg["scaler"] == "hep":
        return fit_scaler(frame, cfg["features"])
    klass = RobustScaler if cfg["scaler"] == "robust" else StandardScaler
    return klass().fit(frame[cfg["features"]])


def transform(frame, scaler, features):
    if hasattr(scaler, "transformers_"):
        return apply_scaler(frame, features, scaler)
    return np.asarray(scaler.transform(frame[features]), dtype=np.float32)


def callbacks(cfg):
    import tensorflow as tf
    return [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=cfg["reduce_lr_factor"], patience=cfg["reduce_lr_patience"], min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                                        min_delta=1e-4, restore_best_weights=True, verbose=1)]


def fit_model(x, y, w, xv, yv, wv, cfg, profile, seed, output):
    from c2st_models import build_binary_classifier
    model = build_binary_classifier(x.shape[1], seed=seed, **cfg[profile])
    history = model.fit(x, y, sample_weight=w, validation_data=(xv, yv, wv),
                        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                        callbacks=callbacks(cfg), verbose=2).history
    loss, val = np.asarray(history["loss"]), np.asarray(history["val_loss"])
    stability = {
        "history": history, "epochs_run": len(loss),
        "finite": bool(np.all(np.isfinite(loss)) and np.all(np.isfinite(val))),
        "minimum_loss": float(np.min(loss)), "minimum_val_loss": float(np.min(val)),
        "negative_loss_seen": bool(np.any(loss < 0) or np.any(val < 0)),
        "last_minus_first_loss": float(loss[-1] - loss[0]),
        "last_minus_first_val_loss": float(val[-1] - val[0]),
        "signed_weights": bool(np.any(w < 0)),
    }
    write_json(output, stability)
    if not stability["finite"]:
        raise FloatingPointError(f"Nonfinite training loss; inspect {output}")
    if stability["signed_weights"] and stability["negative_loss_seen"]:
        print(f"WARNING: negative signed BCE observed; inspect loss stability and factors: {output}")
    return model


def training_plan(nt, nb, corrected, cfg, channel_i=0):
    """Return all splits before fitting, for audit and deterministic regression tests."""
    seed = cfg["seed"]
    tt, tv, te = split_class_indices(nt, cfg["test_size"], cfg["val_size"], seed)
    bt, bv, be = split_class_indices(nb, cfg["test_size"], cfg["val_size"], seed + 1)
    if any(not len(x) for x in (tt, tv, te, bt, bv, be)):
        raise ValueError("Empty outer split; increase the sample size")
    tpool, bpool = np.concatenate([tt, tv]), np.concatenate([bt, bv])
    cp, sp = bpool[corrected[bpool]], bpool[~corrected[bpool]]
    if min(len(tpool), len(cp)) < 2 * cfg["folds"]:
        raise ValueError("Need at least 2*folds target and corrected-base train/validation rows")
    start = seed + 100 * channel_i
    tfolds = shuffled_folds(tpool, cfg["folds"], start)
    bfolds = shuffled_folds(cp, cfg["folds"], start + 1000)
    sfolds = shuffled_folds(sp, cfg["folds"], start + 2000)
    jobs = []
    for k in range(cfg["folds"]):
        t, v = inner_train_val(other_folds(tfolds, k), cfg["val_size"], start + 10 * k + 1)
        b, bv_ = inner_train_val(other_folds(bfolds, k), cfg["val_size"], start + 10 * k + 2)
        s, sv = inner_train_val(other_folds(sfolds, k), cfg["val_size"], start + 10 * k + 3, True)
        jobs.append({"name": f"fold_{k}", "seed": start + k,
                     "train": (t, s, b), "val": (v, sv, bv_),
                     "hold_base": bfolds[k], "hold_target": tfolds[k], "hold_subtract": sfolds[k]})
    final_seed = seed + 5000 + 100 * channel_i
    t, v = inner_train_val(tpool, cfg["val_size"], final_seed + 3001)
    b, bv_ = inner_train_val(cp, cfg["val_size"], final_seed + 3002)
    s, sv = inner_train_val(sp, cfg["val_size"], final_seed + 3003, True)
    jobs.append({"name": "final", "seed": final_seed + 3000,
                 "train": (t, s, b), "val": (v, sv, bv_),
                 "hold_base": be[corrected[be]], "hold_target": te, "hold_subtract": be[~corrected[be]]})
    return {"target_train": tt, "target_val": tv, "target_test": te,
            "base_train": bt, "base_val": bv, "base_test": be}, jobs


def sample(xt, xb, indices):
    t, s, b = indices
    return (np.concatenate([xt[t], xb[s], xb[b]], axis=0).astype(np.float32, copy=False),
            np.concatenate([np.ones(len(t) + len(s), np.uint8), np.zeros(len(b), np.uint8)]))


def predict(model, x, cfg):
    return model.predict(x, batch_size=cfg["batch_size"], verbose=0).reshape(-1)


def fit_dctr_job(target, base, cfg, job, shared_scaler, output):
    import tensorflow as tf
    name = job["name"]
    t, s, b = job["train"]
    scaler = shared_scaler if cfg["compatibility"] else fit_transformer(target, base, t, np.concatenate([s, b]), cfg)
    xt, xb = transform(target, scaler, cfg["features"]), transform(base, scaler, cfg["features"])
    x, y = sample(xt, xb, job["train"])
    xv, yv = sample(xt, xb, job["val"])
    wt, wb = target.weight_before.to_numpy(), base.weight_before.to_numpy()
    norm = "legacy_signed" if cfg["compatibility"] and cfg["subtract_processes"] else cfg["normalization"]
    w = sample_weights(wt, wb, job["train"], job["train"], norm)
    wv = sample_weights(wt, wb, job["val"], job["train"], norm)
    model = fit_model(x, y, w, xv, yv, wv, cfg, "dctr_model", job["seed"], output / f"dctr_{name}_history.json")
    cap = None
    if cfg["cap_quantile"] is not None:
        v = odds(predict(model, xb[job["val"][2]], cfg), cfg["eps"])
        cap = float(np.quantile(v, cfg["cap_quantile"]))
    raw = odds(predict(model, xb[job["hold_base"]], cfg), cfg["eps"])
    factors = (np.minimum(raw, cap) if cap is not None else raw).astype(np.float32)
    diagnostics = factor_summary(factors, wb[job["hold_base"]], cap)
    diagnostics["fraction_raw_above_cap"] = float(np.mean(raw > cap)) if cap is not None else 0.
    diagnostics["train_target_sumw"] = float(wt[t].sum(dtype=np.float64))
    diagnostics["train_subtract_sumw"] = float(wb[s].sum(dtype=np.float64))
    diagnostics["train_corrected_base_sumw"] = float(wb[b].sum(dtype=np.float64))
    diagnostics["train_residual_target_sumw"] = diagnostics["train_target_sumw"] - diagnostics["train_subtract_sumw"]
    write_json(output / f"dctr_{name}_diagnostics.json", diagnostics)
    if cfg["save_fold_models"] or name == "final":
        bundle = output / ("inference" if name == "final" else name)
        bundle.mkdir()
        model.save(bundle / "model.keras")
        joblib.dump(scaler, bundle / "scaler.joblib", compress=3)
        write_json(bundle / "metadata.json", {
            "features": cfg["features"], "eps": cfg["eps"], "cap": cap,
            "normalization": cfg["normalization"], "batch_size": cfg["batch_size"],
            "target_label": 1, "base_label": 0, "direction": "target/base",
            "corrected_processes": [p for p in cfg["base_processes"] if p not in cfg["subtract_processes"]],
            "subtract_processes": cfg["subtract_processes"],
            "scope": "final outer-trainval model" if name == "final" else "cross-fit fold",
            "config": cfg,
        })
    del model, xt, xb, x, xv
    tf.keras.backend.clear_session()
    gc.collect()
    return factors, cap


def run_channel(target, base, cfg, output, channel, channel_i=0):
    import tensorflow as tf
    target, base, population = prepare_tables(target, base, cfg)
    if cfg["compatibility"] and not np.all(target.weight_before.to_numpy() == 1):
        raise ValueError("Legacy compatibility requires unit target weights")
    corrected = base.corrected.to_numpy(dtype=bool)
    outer, jobs = training_plan(len(target), len(base), corrected, cfg, channel_i)
    if not len(outer["base_test"][corrected[outer["base_test"]]]):
        raise ValueError("No corrected-base outer-test events")
    output.mkdir(parents=True)
    write_json(output / "population.json", population)
    # Both sides retain exact row-to-index mapping, including after subsampling.
    identity = ["event_id", "source", "dataset", "branch", "row", "process", "weight_before"]
    target[identity].to_parquet(output / "target_rows.parquet", index=False)
    base[identity + ["corrected"]].to_parquet(output / "base_rows.parquet", index=False)
    provenance = dict(outer)
    for job in jobs:
        for split in ("train", "val"):
            for component, indices in zip(("target", "subtract", "base"), job[split]):
                provenance[f"{job['name']}_{split}_{component}"] = indices
        for key in ("hold_target", "hold_subtract", "hold_base"):
            provenance[f"{job['name']}_{key}"] = job[key]
    np.savez_compressed(output / "splits.npz", **provenance)
    closure_scaler = fit_transformer(target, base, outer["target_train"], outer["base_train"], cfg)
    joblib.dump(closure_scaler, output / "closure_scaler.joblib", compress=3)
    factors = np.ones(len(base), np.float32)
    fold_id = np.full(len(base), -1, np.int16)
    caps = []
    for j, job in enumerate(jobs):
        print(f"[{channel}] DCTR {job['name']}")
        f, cap = fit_dctr_job(target, base, cfg, job, closure_scaler, output)
        factors[job["hold_base"]], fold_id[job["hold_base"]] = f, j
        caps.append(cap)
    if not np.all(factors[~corrected] == 1) or np.any(fold_id[corrected] < 0):
        raise AssertionError("Incomplete or incorrect factor assignment")
    np.savez_compressed(output / "dctr_factors_base.npz", dctr_factor=factors, fold_id=fold_id,
                        corrected=corrected, fold_caps=np.asarray([np.nan if c is None else c for c in caps]))
    test_t, test_b = outer["target_test"], outer["base_test"]
    t, b = target.iloc[test_t].copy(), base.iloc[test_b].copy()
    t["y"], b["y"] = np.uint8(1), np.uint8(0)
    t["dctr_factor"], b["dctr_factor"] = np.float32(1), factors[test_b]
    t["weight_dctr"], b["weight_dctr"] = t.weight_before, b.weight_before.to_numpy() * factors[test_b]
    pd.concat([t, b], ignore_index=True).to_parquet(output / "outer_test_fold.parquet", index=False)
    xt, xb = transform(target, closure_scaler, cfg["features"]), transform(base, closure_scaler, cfg["features"])
    matrices = {}
    for split in ("train", "val", "test"):
        it, ib = outer[f"target_{split}"], outer[f"base_{split}"]
        matrices[split] = (np.concatenate([xt[it], xb[ib]]).astype(np.float32),
                           np.concatenate([np.ones(len(it), np.uint8), np.zeros(len(ib), np.uint8)]))
    metrics = []
    yields = {}
    for i, stage in enumerate(["before", *cfg["reference_stages"], "dctr"]):
        wt = target[f"weight_{stage}" if stage != "dctr" else "weight_before"].to_numpy()
        wb = base[f"weight_{stage}" if stage != "dctr" else "weight_before"].to_numpy()
        if stage == "dctr":
            wb = wb * factors
        ws = {split: closure_weights(wt, wb, outer[f"target_{split}"], outer[f"base_{split}"],
                                     outer["target_train"], outer["base_train"], cfg["compatibility"])
              for split in ("train", "val", "test")}
        x, y = matrices["train"]
        xv, yv = matrices["val"]
        xte, yte = matrices["test"]
        model = fit_model(x, y, ws["train"], xv, yv, ws["val"], cfg, "closure_model",
                          cfg["seed"] + 7000 + 100 * channel_i + i, output / f"closure_{stage}_history.json")
        p = predict(model, xte, cfg).astype(np.float32)
        auc = float(roc_auc_score(yte, p, sample_weight=ws["test"]))
        metric = {"channel": channel, "stage": stage, "auc": auc, "distance_from_half": abs(auc - .5),
                  "weighted_bce": weighted_bce(yte, p, ws["test"]), "n_test": len(yte)}
        metrics.append(metric)
        model.save(output / f"closure_model_{stage}.keras")
        np.savez_compressed(output / f"closure_{stage}_test.npz", p_test=p, w_test=ws["test"], y_test=yte)
        write_json(output / f"closure_{stage}_metrics.json", metric)
        # Correct different realized outer-test fractions caused by integer rounding.
        ft, fb = len(test_t) / len(target), len(test_b) / len(base)
        ts, bs = wt[test_t].sum(dtype=np.float64) / ft, wb[test_b].sum(dtype=np.float64) / fb
        yields[stage] = {"target_sumw_test": float(wt[test_t].sum(dtype=np.float64)),
                         "base_sumw_test": float(wb[test_b].sum(dtype=np.float64)),
                         "target_yield_estimate": float(ts), "base_yield_estimate": float(bs),
                         "base_over_target": float(bs / ts),
                         "target_sumw2_estimate": float(np.square(wt[test_t].astype(float)).sum() / ft**2),
                         "base_sumw2_estimate": float(np.square(wb[test_b].astype(float)).sum() / fb**2)}
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    write_json(output / "yield_closure.json", {"population": "positive-weight classifier outer test",
                                               "test_fraction_target": len(test_t) / len(target),
                                               "test_fraction_base": len(test_b) / len(base), "stages": yields})
    if cfg["plots"]:
        from .validation import validate_channel
        validate_channel(output, cfg)
    return metrics


def run(cfg, output):
    import tensorflow as tf
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg)
    write_json(output / "environment.json", {"python": platform.python_version(), "tensorflow": tf.__version__,
                                             "numpy": np.__version__, "pandas": pd.__version__,
                                             "argv": sys.argv})
    metrics = []
    for i, channel in enumerate(cfg["channels"]):
        target, base = load_side(cfg, "target", channel), load_side(cfg, "base", channel)
        metrics.extend(run_channel(target, base, cfg, output / channel, channel, i))
        del target, base
        gc.collect()
    pd.DataFrame(metrics).to_csv(output / "closure_metrics.csv", index=False)
    write_json(output / "status.json", {"completed": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--normalization", choices=("shape", "yield", "both"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.normalization and cfg["compatibility"]:
        raise ValueError("Do not override compatibility normalization; use a generic profile")
    if args.normalization == "both":
        for mode in ("shape", "yield"):
            run({**cfg, "normalization": mode}, args.output / mode)
    else:
        if args.normalization:
            cfg["normalization"] = args.normalization
        run(cfg, args.output)


if __name__ == "__main__":
    main()
