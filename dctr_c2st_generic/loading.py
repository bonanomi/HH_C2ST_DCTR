"""Branch-wise configurable CF/parquet loading, with source-qualified row identity.

Class role does not imply Data or MC. All weight products, optional fields,
region/category IDs and producer paths are supplied by configuration.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


def branch_files(directory, dataset, shift="nominal"):
    """Historical file ordering; reject ambiguous branch matches instead of guessing."""
    found = {}
    for pattern in (f"{shift}/{dataset}*/events_*.parquet", f"{shift}/{dataset}*/columns_*.parquet"):
        for path in sorted(Path(directory).glob(pattern)):
            branch = int(path.stem.rsplit("_", 1)[-1])
            if branch in found and found[branch] != path:
                raise ValueError(f"Ambiguous files for {dataset}, branch {branch}: {found[branch]}, {path}")
            found[branch] = path
    return found


def producer_path(source, name):
    if name == "reduction":
        return Path(source["reduction_dir"])
    if name in source.get("producers", {}):
        return Path(source["producers"][name])
    matches = sorted((Path(source["store_root"]) / "cf.ProduceColumns").glob(f"prod__{name}*"))
    matches = [p for p in matches if p.is_dir()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous producer {name}; set sources[name]['producers'] explicitly")
    return matches[0] if matches else None


def read_columns(path, columns, optional=False):
    import pyarrow.parquet as pq
    if path is None:
        if optional:
            return None
        raise FileNotFoundError(f"No parquet file for columns {columns}")
    schema = set(pq.read_schema(path).names)
    missing = set(columns) - schema
    if missing:
        if optional:
            return None
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    return pq.read_table(path, columns=columns).to_pandas()


def weight_product(spec, fetch, n):
    """Float32 multiplication order intentionally matches the historical loader.

    spec = {'mode':'unit'|'product', 'fields':[{'producer':..., 'column':...,
             'optional':False}], 'scale':1.0}. scale can encode exposure ratios.
    """
    unknown = set(spec) - {"mode", "fields", "scale"}
    if unknown:
        raise ValueError(f"Unknown weight options: {sorted(unknown)}")
    mode = spec.get("mode", "unit")
    if mode not in ("unit", "product"):
        raise ValueError(f"Unknown weight mode {mode!r}")
    fields = spec.get("fields", [])
    if mode == "unit" and fields:
        raise ValueError("unit weights cannot specify fields")
    if mode == "product" and not fields:
        raise ValueError("product weights require fields; use unit for no weights")
    w = np.ones(n, dtype=np.float32)
    for field in fields:
        if isinstance(field, str):
            field = {"column": field, "producer": "event_weights"}
        if set(field) - {"producer", "column", "optional"}:
            raise ValueError(f"Unknown field options: {field}")
        values = fetch(field.get("producer", "event_weights"), field["column"], field.get("optional", False))
        if values is not None:
            w *= np.asarray(values, dtype=np.float32)
    scale = float(spec.get("scale", 1.))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Weight scale must be finite and positive")
    if scale != 1:
        w *= np.float32(scale)
    if np.any(~np.isfinite(w)):
        raise ValueError("Nonfinite weight product")
    return w


def category_mask(categories, ids):
    ids = set(ids)
    return np.fromiter((any(int(x) in ids for x in row) for row in categories), bool, len(categories))


def load_side(cfg, side, channel):
    frames = []
    fields = list(dict.fromkeys(cfg["features"] + cfg["validation_vars"] + list(cfg["selections"])))
    for process, entry in cfg[f"{side}_processes"].items():
        source_name = entry.get("source", cfg[f"{side}_source"])
        source = cfg["sources"][source_name]
        spec = entry.get("weights", cfg[f"{side}_weights"])
        reference = entry.get("reference_weights", {})
        unknown = set(reference) - set(cfg["reference_stages"])
        if unknown:
            raise ValueError(f"{process}: unknown reference stages {unknown}")
        requested = {source.get("feature_producer", "dl_ml_inputs"): {f: False for f in fields}}
        for ws in [spec, *reference.values()]:
            for field in ws.get("fields", []):
                field = {"column": field, "producer": "event_weights"} if isinstance(field, str) else field
                group = requested.setdefault(field.get("producer", "event_weights"), {})
                col, optional = field["column"], field.get("optional", False)
                group[col] = group.get(col, True) and optional
        for dataset in entry["datasets"]:
            if source.get("format", "cf") == "cf":
                if not source.get("alignment_ok", {}).get(dataset, False):
                    raise ValueError(f"{source_name}/{dataset}: verify alignment and set alignment_ok; no silent skipping")
                paths = branch_files(source["reduction_dir"], dataset, source.get("shift", "nominal"))
            elif source["format"] == "parquet":
                # Explicit prejoined tables: datasets are paths/globs under root.
                paths = dict(enumerate(sorted(Path(source["root"]).glob(dataset))))
            else:
                raise ValueError(f"Unknown source format: {source.get('format')}")
            if not paths:
                raise FileNotFoundError(f"No files for {source_name}/{dataset}")
            for branch, path in sorted(paths.items()):
                cache = {}
                if source.get("format", "cf") == "cf":
                    categories = read_columns(path, [source.get("category_column", "category_ids")]).iloc[:, 0]
                    n = len(categories)
                    def fetch(producer, column, optional=False):
                        key = (producer, column)
                        if key not in cache:
                            directory = producer_path(source, producer)
                            file = branch_files(directory, dataset, source.get("shift", "nominal")).get(branch) if directory else None
                            import pyarrow.parquet as pq
                            wanted = requested.get(producer, {column: optional})
                            schema = set(pq.read_schema(file).names) if file is not None else set()
                            missing_required = [c for c, opt in wanted.items() if c not in schema and not opt]
                            if missing_required:
                                raise KeyError(f"{dataset}/{branch}: missing {producer} fields {missing_required}")
                            present = [c for c in wanted if c in schema]
                            result = read_columns(file, present) if present else None
                            if result is not None and len(result) != n:
                                raise ValueError(f"Misaligned row count: {file}, expected {n}, got {len(result)}")
                            for c in wanted:
                                cache[(producer, c)] = result[c].to_numpy() if c in present else None
                                if c not in present:
                                    warnings.warn(f"Optional field {producer}/{c} missing for {dataset}, branch {branch}; using 1")
                        return cache[key]
                    mask = category_mask(categories, [cfg["channel_ids"][c] for c in cfg["channels"][channel]])
                    if cfg["regions"] is not None:
                        mask &= category_mask(categories, [cfg["region_ids"][r] for r in cfg["regions"]])
                    if not mask.any():
                        continue
                    features = {f: fetch(source.get("feature_producer", "dl_ml_inputs"), f) for f in fields}
                else:
                    table = pd.read_parquet(path)
                    n = len(table)
                    def fetch(producer, column, optional=False):
                        if column not in table:
                            if not optional:
                                raise KeyError(f"{path}: missing {column}")
                            warnings.warn(f"Optional field {column} missing in {path}; using 1")
                            return None
                        return table[column].to_numpy()
                    mask = table[source.get("channel_column", "channel")].isin(cfg["channels"][channel]).to_numpy()
                    if cfg["regions"] is not None:
                        mask &= table[source.get("region_column", "region")].isin(cfg["regions"]).to_numpy()
                    features = {f: fetch("table", f) for f in fields}
                for field, (lower, upper) in cfg["selections"].items():
                    values = features[field]
                    mask &= np.isfinite(values)
                    if lower is not None:
                        mask &= values >= lower
                    if upper is not None:
                        mask &= values < upper
                if not mask.any():
                    continue
                out = pd.DataFrame({f: np.asarray(v[mask], np.float32) for f, v in features.items()})
                if not np.all(np.isfinite(out[fields].to_numpy())):
                    raise ValueError(f"Nonfinite features in {dataset}/{branch}; define an explicit selection")
                nominal = weight_product(spec, fetch, n)[mask]
                out["weight_before"] = nominal
                for stage in cfg["reference_stages"]:
                    out[f"weight_{stage}"] = weight_product(reference[stage], fetch, n)[mask] if stage in reference else nominal
                out["process"] = process
                out["source"] = source_name
                out["dataset"] = dataset
                out["branch"] = branch
                out["row"] = np.flatnonzero(mask)
                # Canonical input path makes aliases to the same file detectable.
                out["event_id"] = [f"{path.resolve()}::{row}" for row in np.flatnonzero(mask)]
                out["corrected"] = side == "base" and process not in cfg["subtract_processes"]
                out["is_dy"] = bool(entry.get("is_dy", False))
                frames.append(out)
    if not frames:
        raise ValueError(f"No {side} rows for {channel}")
    result = pd.concat(frames, ignore_index=True)
    if result.event_id.duplicated().any():
        raise ValueError(f"Repeated {side} input rows (overlapping dataset/process definitions)")
    return result


def prepare_tables(target, base, cfg):
    """Keep physical tables intact; filter nonpositive weights only for classifiers."""
    if set(target.event_id) & set(base.event_id):
        raise ValueError("Target and base contain the same input rows; independent splits would leak")
    report, result = {}, []
    for side, table in (("target", target), ("base", base)):
        weights = table.weight_before.to_numpy(dtype=np.float64)
        good = weights > 0
        report[side] = {
            "full_rows": len(table), "full_signed_sumw": float(weights.sum()),
            "nonpositive_rows_excluded": int((~good).sum()),
            "nonpositive_absw_fraction": float(np.abs(weights[~good]).sum() / np.abs(weights).sum()) if np.abs(weights).sum() else 0.,
        }
        selected = table.loc[good].reset_index(drop=True)
        maximum = cfg["max_events_per_class"]
        if maximum is not None and len(selected) > maximum:
            selected = selected.sample(n=maximum, random_state=cfg["seed"]).reset_index(drop=True)
        if len(selected) < 4:
            raise ValueError(f"Too few positive-weight {side} events")
        report[side]["classifier_rows"] = len(selected)
        report[side]["classifier_sumw"] = float(selected.weight_before.sum())
        result.append(selected)
    return *result, report
