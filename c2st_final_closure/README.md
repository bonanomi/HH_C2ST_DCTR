# Final DCTR closure test: cross-fitting + an independent C2ST

This is the final, stricter validation stage built on top of the existing C2ST/DCTR pipeline.
It answers one focused question:

> After deriving a multidimensional DCTR correction from Data vs nominal (pre-DY-correction) MC,
> does a **new classifier**, evaluated on events that were never used to derive the correction,
> find the DCTR-weighted MC harder to distinguish from Data than (a) uncorrected MC or (b) MC with
> the official DY correction?

The ideal C2ST result is **AUC = 0.5**. Smaller `|AUC - 0.5|` means better closure.

## DCTR cross-fit closure documentation

A detailed walkthrough of the final DCTR closure procedure is available in
[`TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md`](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md).

### Quick index

- [Purpose and overall workflow](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#1-purpose-of-this-script)
- [What does `OUTER` mean?](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#3-terminology-what-does-outer-mean)
- [Which Data, MC, and event weights are used?](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#4-which-data-and-mc-enter-the-script)
- [How the OUTER train/validation/test split works](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#7-step-3--make-the-outer-trainvalidationtest-split)
- [How the NN sample weights are class-balanced](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#11-the-class-balancing-weights-used-by-the-neural-networks)
- [DCTR training vs closure training](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#12-the-two-different-training-stages)
- [How K-fold DCTR cross-fitting works](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#13-step-4--cross-fit-dctr-factors-on-outer-trainvalidation)
- [How the DCTR factor and cap are obtained](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#18-how-is-the-dctr-factor-obtained)
- [How the final DCTR model is applied to the untouched outer test](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#22-step-5--train-the-final-dctr-model-for-outer-test)
- [Training the three final closure classifiers: before, DY, DCTR](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#28-step-6--train-three-new-closure-classifiers)
- [How the final AUC comparison is evaluated](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#35-step-7--evaluate-all-three-on-exactly-the-same-outer-test)
- [“Who sees what?” summary](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#38-who-sees-what-summary)
- [Weight flow diagram](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#39-weight-flow-diagram)
- [What files are produced](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#40-what-files-are-produced)
- [How to interpret the final AUCs](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#41-how-to-interpret-the-final-aucs)
- [Why both cross-fitting and an outer test are needed](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#44-why-do-we-need-both-cross-fitting-and-an-outer-test)
- [Compact end-to-end walkthrough](TRAIN_DCTR_CROSSFIT_CLOSURE_WALKTHROUGH.md#45-compact-step-by-step-walkthrough-matching-the-script-header)

## Why this script is separate

Do not reuse the original DCTR network as the closure classifier. A network used to construct a
reweighting is not an independent judge of that same reweighting. The final study therefore trains
fresh closure networks and evaluates them on a protected outer test sample.

## Files

- `train_dctr_crossfit_closure.py` — derives out-of-fold DCTR factors, trains the three independent
  closure C2STs (`before`, `dy`, `dctr`), and saves all outer-test artifacts.
- `validate_dctr_crossfit_closure.py` — plots AUCs, ROCs, classifier scores, feature closure and
  DCTR factors; also computes paired bootstrap intervals for AUC differences.
- `c2st_config.py`, `c2st_core.py`, `dyvr_lib.py`, `validation_utils.py` — existing utilities used by
  the study.

## Statistical layout

For each channel, Data and positive-weight MC are first divided into an **outer** train, validation
and test population using the existing split fractions from `c2st_config.py`.

The outer test set has a special status:

- it is never used to train a DCTR model;
- it is never used to determine the DCTR cap;
- it is used only for the final closure C2ST evaluation and downstream plots.

### DCTR factors for outer train + validation

The outer train+validation population is divided into `K` folds. For each fold:

1. hold that fold out;
2. train a Data-vs-MC DCTR classifier on the other `K-1` folds;
3. use an internal validation subset of those `K-1` folds for early stopping;
4. determine the optional DCTR cap from that internal-validation MC;
5. predict `p(Data|x)` for the held-out MC fold;
6. store `p/(1-p)` (after optional capping) as that fold's DCTR factor.

Thus each outer-train/validation MC event receives a correction from a network that did not train on
that event.

### DCTR factors for the outer test

A final DCTR model is trained only from the outer train+validation population. It predicts DCTR
factors for the outer test MC. Therefore **no outer-test event participates in DCTR fitting**.

The final DCTR model and scaler are saved and are the natural objects to use later on genuinely new
MC events.

## The three closure classifiers

Three fresh neural networks are then trained with identical input features, architecture, outer split
and class-balancing convention. Only the physical MC weight prescription changes:

- `before`: `weight_uncorrected`
- `dy`: `weight` (the official DY correction is included)
- `dctr`: `weight_uncorrected * dctr_factor`

Each stage receives its own single global class-balancing factor, exactly as in the original C2ST.
This means the closure AUC tests **multidimensional shape agreement**, not the absolute MC yield.

All three networks are evaluated on the exact same outer test rows.

## Run the training

From the `c2st_pipeline` directory:

```bash
conda activate c2st
/usr/bin/time -v python -m c2st_final_closure.train_dctr_crossfit_closure.py --folds 5
```

To start with one channel while profiling runtime:

```bash
python -m c2st_final_closure.train_dctr_crossfit_closure.py --channels 2mu --folds 5
```

The default DCTR factor cap is the 99.5% quantile (`--cap-quantile 0.995`) determined independently
for each fold from its model's internal-validation MC. Disable capping for a diagnostic run with:

```bash
python -m c2st_final_closure.train_dctr_crossfit_closure.py --channels 2mu --folds 5 --cap-quantile 0
```

Cross-fitting requires more training than the basic C2ST. With 5 folds the script trains, per channel:

- 5 DCTR cross-fit networks;
- 1 final DCTR model for the untouched outer test / future deployment;
- 3 closure networks (`before`, `dy`, `dctr`).

That is 9 networks per channel. TensorFlow will use the NAF GPU automatically when visible.

## Outputs

Artifacts are written under:

```text
c2st_artifacts/dctr_crossfit_closure/
```

For each channel the main files are:

```text
2mu/
├── scaler.joblib
├── dctr_model_final.keras
├── dctr_factors_mc.npz
├── outer_test_fold.parquet
├── closure_model_before.keras
├── closure_model_dy.keras
├── closure_model_dctr.keras
├── closure_before_test.npz
├── closure_dy_test.npz
├── closure_dctr_test.npz
└── comparison.json
```

`outer_test_fold.parquet` contains raw test features and physical weights, including:

- `weight_uncorrected`
- `weight`
- `dctr_factor`
- `weight_dctr = weight_uncorrected * dctr_factor`

## Run the validation

```bash
python -m c2st_final_closure.validate_dctr_crossfit_closure.py
```

Useful examples:

```bash
# ll_pt from 0 to 200 GeV, 60 equal-width bins
python -m c2st_final_closure.validate_dctr_crossfit_closure.py \
    --vars mli_ll_pt \
    --range 0 200 \
    --bins 60
```

```bash
# multiple variables, using each variable's full finite range
python -m c2st_final_closure.validate_dctr_crossfit_closure.py \
    --vars mli_ll_pt mli_n_jet mli_mbb mli_met_pt \
    --bins 60
```

By default feature plots are shape-normalized independently to Data in the displayed bins. This makes
it easy to compare which of `before`, `DY`, or `DCTR` best reproduces the differential shape. Use
`--normalization physical` when you explicitly want the raw physical-yield comparison instead.

The validation script creates:

- `closure_auc_comparison.png` — direct before/DY/DCTR AUC comparison;
- one ROC plot per channel;
- one closure-classifier score plot per channel;
- DCTR factor distributions on the untouched outer test;
- Data/MC feature closure plots for requested variables;
- `paired_auc_bootstrap.csv` — paired bootstrap confidence intervals for:
  - DY − before,
  - DCTR − before,
  - DCTR − DY.

A negative `delta_auc` is an improvement if all AUCs are above 0.5.

## Interpreting the result

An example outcome could be:

```text
before  AUC = 0.600
DY      AUC = 0.570
DCTR    AUC = 0.525
```

This would mean both corrections improve closure, with the DCTR correction leaving less
classifier-visible discrepancy on the protected outer test.

Another possible result is:

```text
before  AUC = 0.600
DY      AUC = 0.555
DCTR    AUC = 0.590
```

Then the nominal DY correction would generalize better than DCTR. A DCTR plot that looked excellent
on the sample used to construct it would not be sufficient evidence of a successful correction; this
independent closure C2ST is designed precisely to reveal that failure mode.

The most convincing result is not only an AUC near 0.5. Also inspect held-out or physics-relevant
variables in `outer_test_fold.parquet`. A reweighter that closes only its own training variables but
creates distortions elsewhere is not satisfactory.

## Negative MC weights

The current C2ST/DCTR classifier training retains only MC rows with `weight_uncorrected > 0`.
Ordinary binary cross entropy expects non-negative sample weights; feeding signed NLO weights into BCE
is not a well-defined probability-density classification problem.

This does **not** mean negative-weight events should be removed from the real physics prediction.
For deployment, the final DCTR model depends only on event features:

```python
p_data = model.predict(X)
dctr_factor = p_data / (1.0 - p_data)
new_signed_weight = original_signed_weight * dctr_factor
```

Because the DCTR factor is positive, a negative original event weight stays negative.

The training script reports both:

- fraction of MC events excluded because their weight is non-positive;
- fraction of total absolute MC weight carried by those events.

If those fractions are sizeable or negative-weight events populate different phase-space regions,
this approximation deserves a dedicated systematic study. A future extension could evaluate the
final DCTR model on the full signed MC sample and make physical closure plots with all signed events.

## Deployment on new MC

Use `scaler.joblib` and `dctr_model_final.keras`. The new event's existing weight is not supplied to
the network. Only the configured feature values are transformed and evaluated:

```python
X = scaler.transform(new_mc[cfg.FEATURES])
p_data = model.predict(X).reshape(-1)
factor = p_data / (1.0 - p_data)
new_weight = original_weight_uncorrected * factor
```

Apply the same configured phase-space selections and the same feature ordering used in training.
If the production analysis uses a fixed DCTR cap, save and reuse the cap from the training artifact
rather than re-estimating it on the new sample.
