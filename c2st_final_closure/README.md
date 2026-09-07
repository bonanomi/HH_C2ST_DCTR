# Final DCTR closure test: cross-fitting + an independent C2ST

This directory contains the strictest DCTR validation in the repository.  It derives an out-of-sample DCTR correction, freezes it, trains a **new** classifier, and measures Data/MC closure on a protected outer test sample.

The ideal closure C2ST result is **AUC = 0.5**.  Smaller `|AUC - 0.5|` means less classifier-visible Data/MC separation.

A detailed explanation of the nested split and weight provenance is available in [`TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md`](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md).

## DCTR target definitions

The training script supports two target definitions.

### `inclusive` — historical/default mode

The DCTR network distinguishes

```text
class 1: Data
class 0: all pre-DY MC
```

using the existing class-balanced positive-weight BCE.  Its odds estimate

$$
r_{\rm inclusive}(x) \simeq \frac{p_{\rm Data}(x)}{p_{\rm all\,MC}(x)}
$$

and the factor is applied to every MC event:

$$
w_{\rm DCTR}=w_{\rm uncorrected}\,r_{\rm inclusive}(x).
$$

This is the behavior of the original cross-fit closure code and remains the default for backwards compatibility.

### `dy_only` — process-specific DY mode

The DCTR network instead distinguishes

```text
class 1: Data - non-DY MC
class 0: DY MC
```

The target class is implemented with signed weights:

```text
Data       : label 1, +1
non-DY MC  : label 1, -weight_uncorrected
DY MC      : label 0, +weight_uncorrected
```

Generator-level MC events with `weight_uncorrected <= 0` are still excluded from NN training, as in the historical pipeline.  The negative sign above is introduced deliberately to perform the **background subtraction**.

The signed sample weights are multiplied by one common normalization factor so that their mean absolute value is one.  This keeps the BCE numerical scale stable without altering the event-to-event ratios or signs.

The resulting odds estimate the process-specific correction

$$
r_{\rm DY}(x) \simeq
\frac{p_{\rm Data}(x)-p_{\rm nonDY}(x)}{p_{\rm DY}(x)}.
$$

Only DY events receive the learned factor:

$$
w_i^{\rm DCTR,DY-only}=
\begin{cases}
w_i^{\rm uncorrected} r_{\rm DY}(x_i), & i\in {\rm DY},\\
w_i^{\rm uncorrected}, & i\notin {\rm DY}.
\end{cases}
$$

For convenience the stored `dctr_factor` is exactly `1` for non-DY MC, so `weight_dctr = weight_uncorrected * dctr_factor` remains valid for every MC row.

This construction is the DY analogue of CMS DCTR applications in which a target process is trained against Data after subtraction of the other simulated backgrounds.

## Statistical layout

Both target modes use the same leakage protection:

1. Data and positive-weight MC are split into **outer train / validation / test** populations.
2. The scaler is fitted on outer train only.
3. DCTR is cross-fitted on outer train+validation.
4. Each corrected training/validation event receives a factor from a network that did not train on that event.
5. A final DCTR model trained only on outer train+validation supplies factors for the untouched outer test.
6. Three fresh closure classifiers are trained from scratch using the same outer split:
   - `before`: `weight_uncorrected`
   - `dy`: `weight` (official DY correction)
   - `dctr`: the DCTR prescription selected by `--dctr-target`
7. All closure AUCs are evaluated on exactly the same outer-test rows.

For `dy_only`, Data, DY, and non-DY MC are folded independently inside outer train+validation.  Only held-out DY rows receive a factor; the corresponding held-out Data and non-DY folds are excluded from the fold model as well.

The closure C2ST still uses the normal positive-weight, stage-specific class balancing.  The signed subtraction is used **only while deriving the DY-only DCTR model**.

## Run the training

Run from the repository root.

Historical inclusive mode:

```bash
python -m c2st_final_closure.train_dctr_crossfit_closure \
    --channels 2mu \
    --folds 5 \
    --dctr-target inclusive
```

DY-only mode:

```bash
python -m c2st_final_closure.train_dctr_crossfit_closure \
    --channels 2mu \
    --folds 5 \
    --dctr-target dy_only
```

The default cap is the 99.5% quantile derived from the appropriate internal-validation MC population.  In DY-only mode the cap is derived from **DY validation MC only**.  Disable capping for diagnostics with:

```bash
--cap-quantile 0
```

With five folds each run trains 5 cross-fit DCTR networks, 1 final DCTR network and 3 fresh closure classifiers per channel.

## Artifacts

Inclusive artifacts retain the historical location:

```text
c2st_artifacts/dctr_crossfit_closure/<channel>/
```

DY-only artifacts are kept separate:

```text
c2st_artifacts/dctr_crossfit_closure/dy_only/<channel>/
```

Each channel contains:

```text
scaler.joblib
dctr_model_final.keras
dctr_factors_mc.npz
outer_test_fold.parquet
closure_model_before.keras
closure_model_dy.keras
closure_model_dctr.keras
closure_before_test.npz
closure_dy_test.npz
closure_dctr_test.npz
comparison.json
```

`dctr_factors_mc.npz` also stores `is_dy` and the target mode.  `outer_test_fold.parquet` contains `is_dy`, `dctr_factor`, and `weight_dctr` so the correction can be inspected independently.

The DY-only training additionally prints and stores diagnostics for the target composition, including the DY yield, non-DY subtraction, `Data - nonDY` integrated target and DY fraction of the positive-weight MC yield.

## Run validation

Validate one mode:

```bash
python -m c2st_final_closure.validate_dctr_crossfit_closure \
    --dctr-target dy_only \
    --channels 2mu \
    --vars mli_ll_pt mli_n_jet \
    --bins 60
```

For example, with an explicit range:

```bash
python -m c2st_final_closure.validate_dctr_crossfit_closure \
    --dctr-target dy_only \
    --channels 2mu \
    --vars mli_ll_pt \
    --range 0 200 \
    --bins 60 \
    --normalization shape
```

After **both** target modes have been trained with the same configuration, compare them directly:

```bash
python -m c2st_final_closure.validate_dctr_crossfit_closure \
    --dctr-target both \
    --channels 2mu \
    --vars mli_ll_pt mli_n_jet \
    --bins 60
```

The combined validation compares:

```text
before
official DY
inclusive DCTR
DY-only DCTR
```

and produces:

- a four-way closure-AUC comparison;
- ROC curves and fresh closure-classifier score distributions;
- feature-by-feature Data/MC closure plots;
- all pairwise paired-bootstrap AUC differences;
- a scatter comparison of inclusive and DY-only DCTR factors on the same DY outer-test events;
- a compact CSV summary of the factor comparison.

Feature plots are shape-normalized by default.  Use `--normalization physical` to inspect absolute yields.

## Interpreting the comparison

The main question is whether a process-specific correction

```text
(Data - nonDY) / DY
```

closes the full Data/MC prediction as well as the inclusive

```text
Data / all-MC
```

correction while being more directly interpretable as a genuine DY correction.

If the inclusive and DY-only factors agree on DY events and give similar closure, the original inclusive DCTR was likely dominated by the DY mismatch.  If they differ substantially, the inclusive reweighter was also compensating discrepancies in the non-DY mixture.
