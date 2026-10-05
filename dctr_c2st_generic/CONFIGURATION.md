# Configuration reference

Use a Python file defining `CONFIG` or a JSON file containing the same mapping.
Python configs are executed, so load only trusted configs. Relative paths are
relative to the working directory; the examples assume the repository root.
The CLI requires `--config` and `--output`. Nonempty output directories are
rejected, preventing mixed-run artifacts.

## Processes and roles

```python
CONFIG = {
    "target_processes": {
        "observed": ["dataset_a", "dataset_b"],
    },
    "base_processes": {
        "corrected": {"datasets": ["dataset_c"], "is_dy": True},
        "fixed_background": {"datasets": ["dataset_d"], "is_dy": False},
    },
    "subtract_processes": ["fixed_background"],
    # ... remaining required fields ...
}
```

Both list-valued and dict-valued entries work on **either** side. A dict entry
supports `datasets`, optional `source`, `weights`, `reference_weights`, and
`is_dy`. `is_dy` is retained as provenance; it does not implicitly choose a weight
or determine which processes get corrected. `subtract_processes` is the explicit
correction-scope definition.

All subtracted names must be base-process keys and at least one base group must
remain corrected. An empty list derives one correction for the total base.
Closure always compares target against the complete base: corrected components
plus fixed components. Identical group labels are allowed across roles, but input
rows themselves must not overlap. No Data-versus-MC classification is inferred
from a dataset name.

## Sources

Required `sources` maps arbitrary names to source definitions. `target_source`
and `base_source` choose each role's default. A process entry's `source` overrides
that default, permitting several productions within a role.

CF example:

```python
"sources": {
    "year24": {
        "format": "cf",                  # default format
        "store_root": "/path/to/cf_store_tree",
        "reduction_dir": "/path/to/cf.MergeReducedEvents",
        "shift": "nominal",              # default
        "alignment_ok": {"dataset_a": True},
        "feature_producer": "dl_ml_inputs", # default
        "category_column": "category_ids", # default
        # Optional explicit paths, useful when discovery finds multiple matches:
        "producers": {"event_weights": "/exact/producer/directory"},
    },
},
"target_source": "year24",
"base_source": "year24",
```

Producer discovery uses `store_root/cf.ProduceColumns/prod__NAME*`. Branch
discovery preserves the historical `SHIFT/DATASET*/events_N.parquet` and
`columns_N.parquet` convention, with branches sorted numerically. Ambiguous
matches fail rather than picking an arbitrary file. Producers are read lazily:
unit-weight Data does not require an event-weight producer.

Feature columns and weight fields are read once per needed producer per branch.
Requested columns must exist; explicitly optional weight fields warn and act as
one when absent. Every loaded producer must have the same row count as reduction.
Users must verify actual positional/event alignment and declare it in
`alignment_ok`. The 2025 example deliberately starts with false declarations;
copying the 2024 declarations is not validation of the 2025 production.

A prejoined-table alternative is available:

```python
"sources": {
    "tables": {
        "format": "parquet", "root": "/path/to/tables",
        "channel_column": "channel", "region_column": "region",
    },
},
"base_processes": {"base": ["base_*.parquet"]},
```

Here dataset entries are relative path/glob patterns under `root`. Each file is
already joined and carries features, weights, channel and optional region.
Weight `producer` names are ignored in this format; column names address the
joined table. Files are loaded individually, and overlapping glob definitions
are rejected through row identity. CF-only alignment declarations are unnecessary.

## Weight definitions

`target_weights` and `base_weights` default to `{"mode": "unit"}`. Each process
may override its side's default with `weights`.

```python
"base_weights": {
    "mode": "product",
    "fields": [
        {"producer": "event_weights", "column": "nominal_weight"},
        {"producer": "event_weights", "column": "optional_sf", "optional": True},
    ],
    "scale": 1.0,
},
"target_weights": {"mode": "unit", "scale": 1.0},
```

A string field is shorthand for a required column from `event_weights`. Fields
are multiplied in listed order in float32, matching the old loader arithmetic.
`scale` is a finite positive constant multiplied after the field product. A
`unit` specification cannot list fields; a `product` must have at least one.
No correction is implicitly enabled because a group is MC, Data or DY.

Examples of explicit normalization:

- Unweighted samples: unit mode with scale 1 on both sides.
- Rates at common exposure: unit or product mode with per-side/process
  `scale=L_ref/L_year`.
- Legacy physical MC weights: the ordered nominal field product in
  `examples/legacy_inclusive.py`. The official DY correction is excluded here.

The nominal generic column is called `weight_before`. In the legacy example it
is precisely `weight_uncorrected`. The `dy` reference is the old `weight`, with
the official DY correction included for DY. The learned stage is always
`weight_before * dctr_factor`, not an automatic multiplication of a reference
stage. To learn a residual correction on top of a known correction, explicitly
put that known correction into nominal `weights`.

### Reference stages

```python
"reference_stages": ["official"],
"base_processes": {
    "corrected": {
        "datasets": ["dataset_c"],
        "reference_weights": {
            "official": {
                "mode": "product",
                "fields": ["nominal_weight",
                           {"producer": "known_correction", "column": "factor"}],
            },
        },
    },
},
```

Each reference specification is a **complete weight definition**, not an extra
factor automatically multiplied by nominal. Groups without an override reuse
their nominal weights. Target processes may also define reference weights.
Each stage receives its own fresh closure classifier. References do not enter
DCTR derivation. `before` and `dctr` are reserved; stage names must be safe
artifact names.

## Regions, channels and features

```python
"region_ids": {"ar": 1, "dycr": 3, "ttcr": 4},
"regions": ["dycr", "ttcr"],
"channel_ids": {"2e": 30, "2mu": 40, "emu": 50},
"channels": {"same_flavor": ["2e", "2mu"]},
"features": ["mli_mll", "mli_met_pt"],
"validation_vars": ["mli_n_jet"],
"selections": {"mli_met_pt": [None, 200.]},
"scaler": "hep",
```

CF IDs must correspond to the actual stored category representation. No mll
window or DY region is hardcoded in the generic loader. `regions=None` disables
region filtering; otherwise names form an OR union. A row passing multiple
requested regions is retained once. For joined parquet, region names refer to
the configured region column; the `region_ids` names still declare valid regions
but their numeric values are unused.

Each key in `channels` is an independent output/model group, whose list is a
union of physical channels. Several keys run separate models; one key can pool
channels. If the same physical channel is intentionally listed under multiple
keys, those are separate overlapping studies, not independent measurements.
Pooling regions/channels without including an identifying feature learns a
mixture-level ratio; it does not guarantee closure separately within each group.

Selections apply to both roles: lower bounds inclusive, upper bounds exclusive,
`None` disables a bound. Selection fields are loaded even if they are not model
features. Nonfinite selected feature/diagnostic values fail explicitly. The code
does not silently impute or drop them. Configure a finite-value selection using
`[None, None]` on a field if dropping those events is intentional.

Scalers: `hep` reuses the existing regex-based RobustScaler/MinMaxScaler split;
`robust` applies RobustScaler to all features; `standard` applies StandardScaler
to all features. Compatibility requires `hep`. Generic scalers are fit inside
each DCTR training fold; closure has its own outer-training scaler.

## Training controls

| Key | Default | Meaning |
| --- | --- | --- |
| `normalization` | `shape` | `shape`, `yield`, or explicit `legacy` |
| `compatibility` | `False` | Must be true exactly when normalization is legacy |
| `test_size` | `0.30` | Outer-test fraction per role |
| `val_size` | `0.15` | Validation fraction within train+validation |
| `seed` | `0` | Historical deterministic split/model seed base |
| `folds` | `5` | Cross-fit folds; at least 2 |
| `cap_quantile` | `0.995` | Unweighted validation odds quantile; `None` disables |
| `eps` | `1e-6` | Clip probabilities to `[eps, 1-eps]` |
| `epochs` | `50` | Maximum epochs |
| `batch_size` | `8192` | Fit and prediction batch size |
| `early_stopping_patience` | `5` | Validation-loss early stopping, best weights restored |
| `reduce_lr_patience` | `2` | Validation-loss LR schedule patience |
| `reduce_lr_factor` | `0.2` | LR reduction factor |
| `save_fold_models` | `False` | Save each fold bundle as well as final bundle |
| `plots` | `True` | Generate all diagnostics after each channel |
| `plot_bins` | `40` | Histogram bins |
| `max_events_per_class` | `None` | Historical compatibility-only subsampling |

Independent class truncation can change relative yields/subtraction. Generic
mode therefore rejects `max_events_per_class` rather than silently altering
physical normalization. Use explicit, consistently normalized input samples
for reduced-size generic studies. Small empty splits fail with a clear error.

Model profiles are independent, complete dictionaries:

```python
"dctr_model": {
    "hidden": [50], "batch_normalization": True,
    "optimizer": "sgd", "learning_rate": 0.005,
},
"closure_model": {
    "hidden": [50], "batch_normalization": False,
    "optimizer": "adam", "learning_rate": 0.001,
},
```

These are defaults, not hard-coded architectures. For example `[128, 64, 32]`
creates three hidden layers. Supported optimizers are the repository factory's
`adam` and `sgd`. Callbacks preserve historical `min_delta=1e-4` and
`min_lr=1e-5`. Labels, odds direction and sign preservation are contracts, not
user-selectable conventions.

## Output indices and inference

`dctr_factors_base.npz` is ordered exactly like `base_rows.parquet`, after
classifier filtering. Fold IDs `0..K-1` identify out-of-fold predictions; `K`
is the final model's outer-test prediction; `-1` denotes a fixed component.
`fold_caps` has K fold caps followed by the final cap; NaN denotes no cap.
Never assume these array positions are raw input parquet row numbers.

The final bundle stores its own scaler, features, cap, epsilon, correction scope
and resolved config. Inference preserves input row order and checks feature
finiteness. An unknown process name fails; an explicit transfer mask permits
applying to a new process/year intentionally. Fixed components return one.

There is no feature-based overlap detection at inference. Do not apply a final
bundle to its training population and call that an independent closure test.
