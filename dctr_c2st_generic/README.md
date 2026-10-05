# Generic DCTR + C2ST closure

Derive a positive factor that reweights **base (label 0)** toward **target
(label 1)**, then evaluate it with fresh closure classifiers and physical-yield
checks. The existing `c2st_final_closure` and shared repository modules are
unchanged. This folder is an additional workflow, not a replacement.

The target/base names describe statistical roles, not Data/MC identity. Either
side may contain Data or MC from different years, regions, or productions.

## Start here

Run every command below from the **repository root** (the directory containing
`c2st_config.py`). Use the existing `environment.yaml` environment: TensorFlow,
NumPy, pandas, scikit-learn, joblib, matplotlib and pyarrow are required for full
runs. The generic loader itself does not require awkward. Python 3.11 is the
repository environment version.

### Reproduce the existing setup

```bash
python -m dctr_c2st_generic.training \
  --config dctr_c2st_generic/examples/legacy_inclusive.py \
  --output c2st_artifacts/generic_legacy_inclusive

python -m dctr_c2st_generic.training \
  --config dctr_c2st_generic/examples/legacy_dy_only.py \
  --output c2st_artifacts/generic_legacy_dy_only
```

These examples read the existing `c2st_config.py` paths, process lists, features,
selections and model profiles. They preserve pre-DY training weights, official
DY reference weights, seeds, row ordering, split arithmetic, historical scaler,
class balancing and validation quantile capping. Output names are generic and
are intentionally different from the old validator's artifact interface; use
this folder's validator.

For the historical DY-only case, all non-DY base processes are listed in
`subtract_processes`. They enter the target with negative subtraction weights,
remain in the total closure comparison, and receive factor **exactly one**.

### Compare Data 2025 with Data 2024

Copy `examples/data25_over_data24.py` to your own config and edit:

1. Input paths and actual dataset names for both years.
2. Verified per-dataset `alignment_ok` declarations.
3. Region(s), channel(s), features and selections.
4. Exposure scales, if comparing rates at a common luminosity.

```bash
python -m dctr_c2st_generic.training \
  --config my_data25_over_data24.py \
  --normalization both \
  --output c2st_artifacts/data25_over_data24
```

This produces separate `shape/` and `yield/` runs with the same split seeds.
`--normalization shape` or `--normalization yield` runs one mode. Without an
override, the config controls the mode. Compatibility examples deliberately
reject overrides; use a generic config to change their statistical definition.

**Direction:** target = Data25, base = Data24. The learned factor maps Data24
toward Data25. Applying that same factor to MC25 is the intended, separate
transfer hypothesis. It does not map Data25 back to Data24.

**Normalization:** shape-only can fix a differential mismatch, but cannot by
itself determine an overall rate correction. The `yield` mode retains the
relative input yields. For Data comparisons, those yields include integrated
luminosity unless you explicitly rescale them. A luminosity-normalized yield
ratio is usually the meaningful candidate when the intent is to transfer a
change in acceptance or efficiency rather than luminosity.

### Apply to independent analysis events

A final model/scaler/cap bundle is saved at
`RUN/CHANNEL/inference/`. With a combined channel config there is one combined
bundle; otherwise choose the matching channel's bundle.

```bash
python -m dctr_c2st_generic.apply \
  --bundle c2st_artifacts/data25_over_data24/yield/2mu/inference \
  --input mc25_2mu.parquet \
  --weight-column weight \
  --all-events \
  --output mc25_2mu_reweighted.parquet
```

Use the physical weight column appropriate for your analysis; `weight` above is
only an example. If that column includes an official correction, it stays
included: the application is a multiplication, not an implicit replacement.
The CLI preserves existing columns and adds `dctr_factor` and `weight_dctr`.
It rejects existing DCTR output columns to prevent accidental double application.
If `--weight-column` is omitted, it starts from unit weights.

For component-only inference, use `--process-column process` with the configured
process names. For transfer to differently named samples, `--all-events` is an
explicit choice to correct all supplied events. The Python API also accepts an
explicit boolean selection mask:

```python
from dctr_c2st_generic.apply import DCTRReweighter

rw = DCTRReweighter("RUN/2mu/inference")
factor = rw.predict(events, mask=events["is_selected_component"].to_numpy(bool))
corrected_signed_weights = rw.reweight(
    events, events["physical_weight"].to_numpy(),
    mask=events["is_selected_component"].to_numpy(bool),
)
```

Positive factors preserve the sign of negative generator-weight events. No
negative-event filtering happens during application. There is no automatic
inversion, yield rescaling, clipping beyond the saved cap, feature selection,
region cut, or luminosity adjustment at inference. Inputs must use the training
feature definitions, units and intended phase space. Use only trusted model and
joblib bundles.

Do not use the final model to claim closure on its own training rows. Use the
saved out-of-fold factors for the training/validation population and the final
model only for the untouched outer test or independent samples.

## What is configurable?

- Required `target_processes` and `base_processes`; both accept the old list- or
  dict-valued group structures.
- Independent source paths, per-process source overrides and weight products.
- Optional subtracted base processes and arbitrary reference-weight stages.
- Region unions and separate or pooled channels, with category IDs supplied by
  config rather than embedded in the loader.
- Arbitrary feature lists, validation-only features, selections and scalers.
- Separate DCTR and closure model profiles, with arbitrary hidden-layer layouts.
- Shape versus yield normalization, split fractions, folds, seeds and capping.

See [CONFIGURATION.md](CONFIGURATION.md) for the full contract.

## Outputs and diagnostics

Each run saves its resolved `config.json`, software versions,
`closure_metrics.csv`, and a completion marker. Each channel saves:

| Artifact | Purpose |
| --- | --- |
| `target_rows.parquet`, `base_rows.parquet` | Ordered classifier-row provenance and nominal weights |
| `splits.npz` | Outer and per-model train/validation/held-out indices |
| `dctr_factors_base.npz` | Factors, fold IDs, corrected mask and caps |
| `outer_test_fold.parquet` | Target and total base with physical stage weights |
| `inference/` | Final model, corresponding scaler, odds/cap and config metadata |
| `dctr_*_history.json` | Training/validation loss and stability summaries |
| `dctr_*_diagnostics.json` | Factor quantiles, weighted mean, cap fraction and training yields |
| `closure_*` | Fresh closure models, test predictions and AUC/BCE metrics |
| `yield_closure.json` | Physical target/base yield comparison on the outer test |
| `population.json` | Signed input yields and nonpositive classifier exclusions |
| `plots/` | Losses, factors, AUC and both physical/shape feature closure |

AUC **closer to 0.5** means better classifier closure. AUC does not establish
normalization closure. Inspect `yield_closure.json` and the physical plots too.
For signed subtraction, inspect all losses, factor quantiles, weighted mean,
cap fraction and closure plots before interpreting the correction.

The training closure sample excludes nonpositive nominal weights, matching the
old workflow. Its plots are **not a full signed-MC closure measurement**. The
full signed application must be validated separately on analysis samples.

Rebuild plots without retraining:

```bash
python -m dctr_c2st_generic.validation --input RUN
```

For `--normalization both`, validate `RUN/shape` and `RUN/yield` separately.

## Tests and toy example

```bash
python -m unittest discover -s dctr_c2st_generic/tests -v

python -m dctr_c2st_generic.examples.toy --output /tmp/dctr_toy
python -m dctr_c2st_generic.training \
  --config /tmp/dctr_toy/config.json --normalization both \
  --output /tmp/dctr_toy/results
```

The toy has a shifted feature and a known relative yield, so it exercises both
normalization choices. It is an integration example, not a precision benchmark.
Do not interpret an eight-epoch toy run as evidence of physics performance.
