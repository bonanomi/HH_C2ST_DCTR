# Walkthrough of `train_dctr_crossfit_closure.py`

## 1. Purpose of this script

`train_dctr_crossfit_closure.py` performs the final, independent closure test of the DCTR reweighting strategy.

The central physics question is:

> **After reweighting MC with a DCTR correction, can a completely new classifier still distinguish Data from MC?**

The script compares three MC descriptions of exactly the same phase space:

1. **before**  
   MC with the nominal event weight **without** the official DY correction.

2. **dy**  
   MC with the nominal event weight **including** the official DY correction.

3. **dctr**  
   MC with the nominal event weight **without** the official DY correction, multiplied by a DCTR factor learned from Data and MC.

For each of these three descriptions, a **new closure classifier** is trained to separate Data from MC.

The final quantity of interest is the held-out AUC:

$$
\mathrm{AUC}_{\rm before},\qquad
\mathrm{AUC}_{\rm DY},\qquad
\mathrm{AUC}_{\rm DCTR}.
$$

For this study, an AUC closer to 0.5 means **better Data/MC closure**.

The desired behavior would therefore be something like

```text
before : AUC = 0.60
DY     : AUC = 0.56
DCTR   : AUC = 0.52
```

rather than a larger AUC.

This script is deliberately more complicated than the basic C2ST because it must ensure that the DCTR correction is evaluated honestly: an event should not be assigned a DCTR factor from a network that trained on that same event.

---

# 2. An overall view of the workflow

The full procedure can be summarized as

```text
                         Data + MC
                            │
                            │
                  apply DY-VR selections
                            │
                            │
              remove negative-weight MC
              for classifier training only
                            │
                            ▼
                  OUTER train/val/test
                   /        |        \
                  /         |         \
          outer train   outer val   OUTER TEST
               \          /             │
                \        /              │
                 train+val              │
                    │                   │
                    │                   │ protected
                    ▼                   │ from DCTR fit
          K-fold DCTR cross-fitting     │
                    │                   │
                    │                   │
          out-of-fold DCTR factors      │
          for outer train+val MC        │
                    │                   │
                    └──────┐            │
                           │            │
              final DCTR model          │
             trained on train+val ──────┘
                           │
                           ▼
                DCTR factors for
                untouched outer test
                           │
                           ▼
       ┌────────────────────────────────────┐
       │ Train THREE new closure classifiers│
       │ on the same outer split            │
       │                                    │
       │ before : weight_uncorrected        │
       │ DY     : weight                    │
       │ DCTR   : weight_uncorrected × DCTR │
       └────────────────────────────────────┘
                           │
                           ▼
             evaluate all three on
             the SAME OUTER TEST events
                           │
                           ▼
               compare held-out AUCs
```

The most important idea is that there are **two different kinds of neural networks** in this script:

1. **DCTR networks**  
   Their purpose is to *derive a correction factor*.

2. **closure C2ST networks**  
   Their purpose is to *test whether that correction worked*.

These networks must not be confused.

---

# 3. Terminology: what does “OUTER” mean?

The word **OUTER** refers to the highest-level train/validation/test split of the entire closure study.

It is called “outer” because another level of splitting happens *inside* the outer training+validation population when the DCTR factors are cross-fitted.

The outer split therefore surrounds the whole DCTR derivation procedure.

For each channel, Data and MC are split into:

```text
OUTER TRAIN
OUTER VALIDATION
OUTER TEST
```

The crucial rule is:

> **The OUTER TEST events must never be used to fit any DCTR model, fit the scaler, choose a DCTR cap, or train the closure classifiers.**

They are the final protected events on which the three closure AUCs are measured.

This gives the final AUC comparison a clean interpretation.

---

# 4. Which Data and MC enter the script?

The relevant code starts in `main()` with

```python
tables = dyvr_lib.load_all(
    layout,
    cfg.MC_PROCESSES,
    cfg.DATA_PROCESSES,
    cfg.ALIGNMENT_OK,
    cfg.SHIFT,
    feature_fields=cfg.LOAD_FEATURES,
    validate_classification=False,
    keep_region="dycr",
    keep_channels=(channel,),
    selections=cfg.SELECTIONS,
    compact_dtypes=True,
)
```

## 4.1 Data

The Data samples are those listed in

```python
cfg.DATA_PROCESSES
```

and are restricted to

```text
region  = dycr
channel = the channel currently being processed
```

for example `2mu` or `2e`.

Any additional physics selections configured in

```python
cfg.SELECTIONS
```

are also applied.

Therefore this script does **not** use arbitrary events from the full analysis.

It uses the same DY validation-region phase space defined by the project configuration.

---

## 4.2 MC

The MC samples are those listed in

```python
cfg.MC_PROCESSES
```

and are restricted to exactly the same region, channel, and configured selections as Data.

The helper

```python
channel_tables(...)
```

concatenates all relevant Data processes into one Data table and all relevant MC processes into one MC table for the current channel.

At this point we have conceptually

```text
data_df
mc_full
```

where `mc_full` contains all MC events, including events with negative `weight_uncorrected`.

---

# 5. What happens to negative MC weights?

The script first measures their importance:

```python
neg_event_frac
neg_absw_frac
```

where

- `neg_event_frac` is the fraction of MC rows with `weight_uncorrected <= 0`;
- `neg_absw_frac` is the fraction of total \(|w|\) carried by those rows.

Then it defines

```python
mc_df = mc_full.loc[
    mc_full["weight_uncorrected"] > 0
].reset_index(drop=True)
```

so the classifier study uses only positive-weight MC.

This is done because an ordinary binary-cross-entropy classifier needs a positive sample measure. Negative `sample_weight` values do not have the usual probabilistic interpretation of weighted maximum likelihood.

This does **not** mean that negative-weight events should be removed from a final physics prediction.

The policy here is only:

```text
classifier training / closure study:
    positive-weight MC

real deployment of a positive DCTR factor:
    apply the factor multiplicatively to the original
    signed MC weight, preserving its sign
```

The script records the excluded fractions so that one can judge how important this approximation is.

---

# 6. Optional subsampling

The code calls

```python
data_df = maybe_subsample(...)
mc_df   = maybe_subsample(...)
```

using

```python
cfg.MAX_EVENTS_PER_CLASS
```

If this configuration value is `None`, all selected events are retained.

If a maximum is set, Data and MC are randomly reduced independently to that maximum.

The final study should normally use the full intended sample once memory/runtime have been validated.

---

# 7. Step 3 — make the OUTER train/validation/test split

The relevant code is

```python
train_d, val_d, test_d = split_class_indices(
    len(data_df),
    cfg.TEST_SIZE,
    cfg.VAL_SIZE_WITHIN_TRAINVAL,
    cfg.RANDOM_STATE,
)

train_m, val_m, test_m = split_class_indices(
    len(mc_df),
    cfg.TEST_SIZE,
    cfg.VAL_SIZE_WITHIN_TRAINVAL,
    cfg.RANDOM_STATE + 1,
)
```

Data and MC are split separately.

The suffixes mean:

```text
_d = Data indices
_m = MC indices
```

so

```text
train_d = outer-training Data indices
val_d   = outer-validation Data indices
test_d  = outer-test Data indices

train_m = outer-training MC indices
val_m   = outer-validation MC indices
test_m  = outer-test MC indices
```

Then the script defines

```python
trainval_d = np.concatenate([train_d, val_d])
trainval_m = np.concatenate([train_m, val_m])
```

These combined outer train+validation populations are the events that may be used to derive DCTR.

The protected events are

```text
test_d
test_m
```

and these are excluded from **every DCTR fitting step**.

---

# 8. Why do we still have an OUTER validation set?

The final closure classifiers need the usual three-way split:

```text
train
validation
test
```

The outer validation set is used for

- Keras validation loss;
- learning-rate reduction;
- early stopping;

when the final three closure networks are trained.

However, for deriving DCTR factors on the closure train/validation population, `train_d + val_d` and `train_m + val_m` are temporarily treated as one larger population and cross-fitted.

This is safe because cross-fitting guarantees that the DCTR factor assigned to each of those rows comes from a DCTR network that did not train on that row.

---

# 9. Preprocessing: where is the scaler fitted?

Before any DCTR fitting, the script creates

```python
scaler_fit = pd.concat([
    data_df.iloc[train_d][cfg.FEATURES],
    mc_df.iloc[train_m][cfg.FEATURES],
], ignore_index=True)
```

and fits

```python
scaler = fit_scaler(scaler_fit, cfg.FEATURES)
```

Therefore the feature scaler sees **OUTER TRAIN only**.

It does not see

```text
outer validation
outer test
```

when its parameters are estimated.

This is an important leakage protection.

Afterward the same fitted scaler is applied to every Data and MC event:

```python
x_data = apply_scaler(data_df, cfg.FEATURES, scaler)
x_mc   = apply_scaler(mc_df, cfg.FEATURES, scaler)
```

These arrays contain the NN input features only.

The physical event weights are **not NN input features**.

---

# 10. The two physical MC weight definitions

The script extracts

```python
raw_before = mc_df["weight_uncorrected"].to_numpy(...)
raw_dy     = mc_df["weight"].to_numpy(...)
```

These have different meanings.

## `raw_before`

```text
weight_uncorrected
```

is the nominal physical MC event weight **without** the official DY correction.

It already contains the other event-weight components included in the analysis setup.

This is the starting point for both

- the `before` comparison;
- the DCTR derivation.

---

## `raw_dy`

```text
weight
```

is the physical MC event weight **after including the official DY correction**.

This is used for the `dy` closure stage.

---

# 11. The class-balancing weights used by the neural networks

This is one of the most important details.

The neural networks are not trained directly with raw MC weights.

The helper

```python
stage_weights(...)
```

first calculates a global MC class factor

$$
C_{\rm MC}=
\frac{N_{\rm Data}}
{\sum_i w_i^{\rm MC}},
$$

using the relevant full Data and MC population supplied to it.

The helper also uses a common global factor

$$
C_{\rm global}=
\frac{N_{\rm Data}+N_{\rm MC}}
     {2N_{\rm Data}}.
$$

The Data NN sample weights become

$$
w_i^{\rm Data,NN}=C_{\rm global},
$$

while MC NN sample weights become

$$
w_i^{\rm MC,NN}=
w_i^{\rm physical}
C_{\rm MC}
C_{\rm global}.
$$

The common `C_global` factor does not change the optimum of the weighted loss; it simply keeps the average weight scale well behaved.

The important factor is $(C_{\rm MC})$, which balances the total effective Data and MC class weights.

Thus the networks retain the **relative event-to-event physical MC weighting**, but do not use overall Data/MC normalization as a trivial class discriminator.

This is what gives the classifier odds their useful density-ratio interpretation.

---

# 12. The two different training stages

Before following the folds, it is useful to distinguish the networks by role.

## Stage A — DCTR networks

Input:

```text
Data features
MC features
```

Labels:

```text
Data = 1
MC   = 0
```

MC physical sample weight:

```text
weight_uncorrected
```

plus the global C2ST class-balancing factor.

Purpose:

```text
learn a Data / pre-DY-MC density ratio
```

Output:

$$
r(x)=
\frac{p(\mathrm{Data}\mid x)}
     {1-p(\mathrm{Data}\mid x)}.
$$

This output becomes the DCTR multiplicative correction.

---

## Stage B — closure networks

Three new networks are trained later.

They do **not** create DCTR weights.

They test three different MC weighting prescriptions:

```text
before
DY
DCTR
```

Their held-out AUCs are the final result of this script.

---

# 13. Step 4 — cross-fit DCTR factors on outer train+validation

The function is

```python
crossfit_dctr_trainval(...)
```

and is called with

```python
trainval_d
trainval_m
raw_before
```

Only the outer train+validation population enters this procedure.

The outer test is absent.

---

# 14. Why cross-fitting is needed

Imagine that we trained one DCTR classifier on events A, B, C and then used that same classifier to calculate DCTR weights for A, B, C.

The classifier could partially exploit finite-sample fluctuations or memorization.

If we then evaluated closure using those same DCTR-weighted events, the result could look artificially good.

Cross-fitting avoids this.

The rule is:

> Every outer train/validation MC row receives its DCTR factor from a network that did not train on that row.

---

# 15. Constructing the K folds

Inside

```python
crossfit_dctr_trainval(...)
```

the code builds independent shuffled folds:

```python
data_folds = shuffled_folds(data_trainval, n_folds, seed)
mc_folds   = shuffled_folds(mc_trainval, n_folds, seed + 1000)
```

For five folds, conceptually:

```text
Data train+val:
    D1 D2 D3 D4 D5

MC train+val:
    M1 M2 M3 M4 M5
```

For fold `k`, one Data fold and one MC fold are held out:

```python
hold_d = data_folds[k]
hold_m = mc_folds[k]
```

The remaining four folds form candidate fitting samples:

```python
cand_d
cand_m
```

---

# 16. Each cross-fit fold has another internal train/validation split

For the candidate events, the script calls

```python
train_d, val_d = inner_train_val(cand_d, ...)
train_m, val_m = inner_train_val(cand_m, ...)
```

These are **internal DCTR train/validation sets**.

They should not be confused with the outer train and outer validation sets.

For one fold the structure is therefore:

```text
OUTER TRAIN+VALIDATION
        │
        ├── held-out fold
        │       hold_d
        │       hold_m
        │
        └── other K-1 folds
                │
                ├── internal DCTR train
                │
                └── internal DCTR validation
```

The DCTR network trains only on the internal DCTR training rows.

Its Keras early stopping uses the internal DCTR validation rows.

It never trains on `hold_d` or `hold_m`.

---

# 17. Exactly which weights are passed to a cross-fit DCTR network?

The fold model is trained through

```python
fit_binary_model(...)
```

with

```python
raw_mc_all = raw_before
```

That means the physical MC weight underlying this DCTR classifier is always

$$
w_i^{\rm physical}=w_i^{\rm before}
=\texttt{weight\_uncorrected}.
$$

The Data sample weights are class-balancing weights.

The MC sample weights are

$$
w_i^{\rm before}
\times
C_{\rm MC}
\times
C_{\rm global}.
$$

No official DY correction is included.

No DCTR factor is included.

This is essential:

> The DCTR network learns how to correct the **pre-DY** MC toward Data.

---

# 18. How is the DCTR factor obtained?

For an MC event, the DCTR network predicts

$$
p(x)=P(\mathrm{Data}\mid x).
$$

The helper

```python
dctr_from_probability(...)
```

computes

$$
r(x)=
\frac{p(x)}{1-p(x)}.
$$

Numerically, the probability is clipped to

```python
[eps, 1-eps]
```

before forming the ratio.

Because the DCTR classifier was trained with balanced effective class priors, this odds ratio can be interpreted approximately as a Data/MC density ratio in the feature space.

---

# 19. Where is the DCTR cap obtained?

For each fold, the function

```python
cap_from_validation(...)
```

runs the DCTR network on

```python
val_m
```

which is the fold model's **internal validation MC**.

The requested quantile, by default

```text
0.995
```

is measured from those DCTR factors.

For example,

$$
r_{\rm cap}=
Q_{0.995}
\left[
r(x_i),\,
i\in\text{internal validation MC}
\right].
$$

Then the held-out fold receives

$$
r_{\rm capped}(x)=
\min(r(x),r_{\rm cap}).
$$

This ordering is deliberate.

The cap is **not derived from the held-out MC fold**.

Otherwise information from the events receiving the correction would influence the correction prescription.

---

# 20. Assigning an out-of-fold DCTR factor

After training fold `k`, the script evaluates

```python
predict_dctr(
    model,
    x_mc,
    hold_m,
    cap_value,
    eps,
)
```

Only `hold_m` receives these predictions.

Therefore each MC row in outer train+validation eventually obtains exactly one DCTR factor.

For five folds:

```text
M1 <- model trained without M1
M2 <- model trained without M2
M3 <- model trained without M3
M4 <- model trained without M4
M5 <- model trained without M5
```

This is what **out-of-fold** means.

The array

```python
dctr_factor
```

stores these factors in the original MC row positions.

The array

```python
fold_id
```

records which fold model produced each factor.

---

# 21. What happens to the held-out Data fold?

The code folds Data in parallel:

```python
hold_d
```

but downstream only MC requires a DCTR correction factor.

The held-out Data fold is still useful conceptually because the fold model is trained without that portion of Data as well.

Thus each fold's DCTR evaluation population is independent on both sides of the Data/MC comparison.

---

# 22. Step 5 — train the final DCTR model for outer test

At this point every MC event in outer train+validation has an out-of-fold DCTR factor.

The outer test still has none.

For the final outer-test correction, the script calls

```python
fit_final_dctr_for_outer_test(...)
```

using

```text
data_trainval
mc_trainval
```

but **not**

```text
test_d
test_m
```

The complete outer train+validation population is internally split once more:

```python
train_d, val_d = inner_train_val(data_trainval, ...)
train_m, val_m = inner_train_val(mc_trainval, ...)
```

Then one final DCTR classifier is trained.

Again, its MC training weights are based on

```python
raw_before
```

so it learns Data versus pre-DY MC.

---

# 23. Which events determine the final test cap?

The final model's DCTR cap is computed from

```python
val_m
```

inside outer train+validation.

The protected `test_m` events are not used to determine the cap.

Then the final model is applied to

```python
mc_test
```

to produce

```python
factor_test
```

and these factors are inserted into

```python
dctr_factor[test_m]
```

Thus:

```text
outer train+val MC:
    factor from a cross-fit model that did not train on that row

outer test MC:
    factor from the final DCTR model trained only on outer train+val
```

No MC event receives a factor from a DCTR network that trained on that event.

---

# 24. Summary of DCTR factor provenance

This is worth keeping as a reference.

| MC event population | DCTR model used | Did that model train on the event? |
|---|---|---:|
| outer train | corresponding cross-fit fold model | No |
| outer validation | corresponding cross-fit fold model | No |
| outer test | final DCTR model trained on outer train+val | No |

This is the core leakage-protection property of the script.

---

# 25. The reusable DCTR weight

Once an MC event has a DCTR factor \(r_i\), its DCTR-corrected physical weight is defined as

$$
w_i^{\rm DCTR}=
w_i^{\rm before}
r_i.
$$

In code:

```python
raw_before * dctr_factor
```

The DCTR factor itself is positive.

In a future real deployment on signed MC, the intended operation would be

$$
w_i^{\rm new,signed}=
w_i^{\rm old,signed}
r_i,
$$

so a negative event remains negative.

The classifier does not need the event's weight to calculate $r_i$. It only needs its input features.

---

# 26. Saving the DCTR factors

The script saves

```text
dctr_factors_mc.npz
```

containing

```text
dctr_factor
fold_id
mc_train
mc_val
mc_test
fold_caps
final_test_cap
```

This records both the correction and its provenance.

It also saves

```text
dctr_model_final.keras
```

which is the deployable DCTR model trained without outer-test events.

---

# 27. Saving the untouched outer test fold

The helper

```python
make_outer_test_fold(...)
```

writes

```text
outer_test_fold.parquet
```

For Data it stores

```text
y = 1
weight_uncorrected = 1
weight = 1
dctr_factor = 1
weight_dctr = 1
```

For MC it stores

```text
y = 0
weight_uncorrected
weight
dctr_factor
weight_dctr = weight_uncorrected × dctr_factor
```

plus the configured raw features.

This file is deliberately sufficient for independent downstream plotting without loading the full training population again.

---

# 28. Step 6 — train three new closure classifiers

Now the DCTR correction has been fully derived.

Only at this point does the script start the final closure C2ST.

It constructs three fixed NN input matrices:

```python
x_train, y_train
x_val,   y_val
x_test,  y_test
```

using

```text
the exact same outer Data events
the exact same outer MC events
the exact same feature values
the exact same labels
```

for all three stages.

The only difference between stages is the MC sample weight.

This is very important.

---

# 29. Stage `before`

The physical MC weight is

```python
raw_stage = raw_before
```

or

$$
w_i^{\rm physical,before}=
w_i^{\rm uncorrected}.
$$

The closure NN therefore asks:

> Can Data be distinguished from the MC prediction before applying either the official DY correction or DCTR?

The network receives:

```text
inputs:
    cfg.FEATURES

labels:
    Data = 1
    MC   = 0

Data sample weights:
    class-balancing factor

MC sample weights:
    weight_uncorrected × class-balancing factors
```

Its held-out result is

```text
AUC_before
```

---

# 30. Stage `dy`

The physical MC weight is

```python
raw_stage = raw_dy
```

or

$$
w_i^{\rm physical,DY}=
w_i^{\rm official\ DY}.
$$

The events and feature matrices are unchanged.

Only the MC sample weights are different.

The closure NN asks:

> After applying the official DY correction, how distinguishable are Data and MC?

The network receives:

```text
inputs:
    exactly the same cfg.FEATURES

labels:
    exactly the same Data/MC labels

Data sample weights:
    class-balancing factor

MC sample weights:
    official corrected weight × class-balancing factors
```

Its held-out result is

```text
AUC_DY
```

---

# 31. Stage `dctr`

The physical MC weight is constructed as

```python
raw_stage = raw_before * dctr_factor
```

or

$$
w_i^{\rm physical,DCTR}=
w_i^{\rm uncorrected}
r_i^{\rm DCTR}.
$$

For outer train and validation MC, \(r_i\) is cross-fitted.

For outer test MC, \(r_i\) comes from the final DCTR model that never saw outer test.

The closure NN asks:

> After replacing the official DY correction with the learned DCTR correction, how distinguishable are Data and MC?

The network receives:

```text
inputs:
    exactly the same cfg.FEATURES

labels:
    exactly the same Data/MC labels

Data sample weights:
    class-balancing factor

MC sample weights:
    weight_uncorrected
    × out-of-sample DCTR factor
    × class-balancing factors
```

Its held-out result is

```text
AUC_DCTR
```

---

# 32. Important: the DCTR closure NN is NOT the DCTR NN

This is easy to confuse.

The first set of networks learned

```text
Data vs pre-DY MC
```

and produced the DCTR correction.

The later `dctr` closure network is a fresh network initialized from scratch.

Its purpose is adversarial:

> Try as hard as possible to find a remaining difference between Data and DCTR-reweighted MC.

A successful DCTR correction should therefore make this fresh classifier perform poorly:

$$
\mathrm{AUC}_{\rm DCTR}\rightarrow0.5.
$$

---

# 33. The exact closure-stage class balancing

For each stage, `train_closure_stage()` calls

```python
stage_weights(...)
```

using the corresponding physical weighting prescription.

For example, the DCTR stage uses

```text
raw_mc_all   = weight_uncorrected × DCTR
raw_mc_train = same quantity on train_m
raw_mc_val   = same quantity on val_m
raw_mc_test  = same quantity on test_m
```

A stage-specific class-balancing factor is therefore calculated.

This is intentional.

The objective is to compare **multivariate shape agreement**, not allow the closure classifier to win simply because one weighting prescription predicts a different total MC normalization.

Consequently:

```text
AUC_before
AUC_DY
AUC_DCTR
```

primarily compare how well the three weighting prescriptions reproduce the Data feature distribution.

Absolute yield closure should be studied separately using physical-weight histograms without C2ST class balancing.

---

# 34. Why each stage gets a different global MC balance factor

The three physical MC weight sums may differ:

$$
\sum w^{\rm before}
\neq
\sum w^{\rm DY}
\neq
\sum w^{\rm DCTR}.
$$

If the same arbitrary class factor were used for all stages, differences in total normalization could affect the classifier loss.

Instead, each stage balances its own MC class against Data.

This deliberately removes the trivial normalization information.

The event-to-event relative weighting within each stage remains intact.

---

# 35. Step 7 — evaluate all three on exactly the same outer test

The function

```python
train_closure_stage(...)
```

trains on

```text
x_train
y_train
```

uses

```text
x_val
y_val
```

for early stopping, and finally predicts

```python
p_test = model.predict(x_test)
```

The same `x_test` and `y_test` are supplied to all three stages.

Only `w_test` changes according to the stage.

The weighted AUC is then

```python
roc_auc_score(
    y_test,
    p_test,
    sample_weight=w_test,
)
```

The script also calculates weighted binary cross entropy.

---

# 36. Why the common outer test is important

Suppose the three classifiers were evaluated on three independently drawn test samples.

Then an AUC difference could partly come from test-sample fluctuations.

Instead this script compares

```text
same Data test rows
same MC test rows
same features
same labels
different MC weighting prescriptions
different independently trained closure networks
```

This makes downstream paired resampling comparisons much more meaningful.

---

# 37. What exactly has the outer test been used for?

Before final evaluation, the outer test has been used only in ways that do not fit model parameters:

- the already-fitted scaler is applied to its features;
- the final DCTR model predicts DCTR factors for its MC events;
- those DCTR factors are capped using a cap derived elsewhere;
- the three trained closure classifiers predict scores for it.

The outer test has **not** been used to:

- fit the scaler;
- train any cross-fit DCTR model;
- train the final DCTR model;
- choose any DCTR cap;
- train any closure classifier;
- control early stopping.

This is what makes it a genuine final test sample.

---

# 38. “Who sees what?” summary

## Scaler

```text
sees:
    OUTER TRAIN Data
    OUTER TRAIN MC

does not fit on:
    OUTER VALIDATION
    OUTER TEST
```

---

## Cross-fit DCTR fold model

For fold \(k\):

```text
trains on:
    K-1 folds, after an internal train/validation split

does not train on:
    held-out Data fold k
    held-out MC fold k
    any OUTER TEST event
```

Its held-out MC fold receives its DCTR factors.

---

## Final DCTR model

```text
trains/validates on:
    OUTER TRAIN + OUTER VALIDATION

does not see:
    OUTER TEST
```

Its purpose is to assign DCTR factors to outer-test MC.

---

## `before` closure classifier

```text
train:
    OUTER TRAIN

validation:
    OUTER VALIDATION

test:
    OUTER TEST

MC physical weighting:
    weight_uncorrected
```

---

## `dy` closure classifier

```text
train:
    same OUTER TRAIN

validation:
    same OUTER VALIDATION

test:
    same OUTER TEST

MC physical weighting:
    official corrected weight
```

---

## `dctr` closure classifier

```text
train:
    same OUTER TRAIN

validation:
    same OUTER VALIDATION

test:
    same OUTER TEST

MC physical weighting:
    weight_uncorrected × out-of-sample DCTR factor
```

---

# 39. Weight flow diagram

A useful way of remembering the whole procedure is:

```text
                         weight_uncorrected
                                │
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  │                           │
                  ▼                           ▼
          DCTR derivation              BEFORE closure C2ST
                  │
         Data vs pre-DY MC
                  │
          class-balanced BCE
                  │
                  ▼
              p(Data|x)
                  │
                  ▼
             p / (1-p)
                  │
                  ▼
             DCTR factor
                  │
                  ▼
       weight_uncorrected × DCTR
                  │
                  ▼
            DCTR closure C2ST


Meanwhile:

weight_uncorrected
        │
        × official DY correction
        │
        ▼
      weight
        │
        ▼
   DY closure C2ST
```

The three final C2ST branches are therefore

$$
\boxed{
w_{\rm before}=w_{\rm uncorrected}
}
$$

$$
\boxed{
w_{\rm DY}=w_{\rm official\ corrected}
}
$$

$$
\boxed{
w_{\rm DCTR}=w_{\rm uncorrected}\,r_{\rm DCTR}(x)
}
$$

followed in every case by the C2ST-specific global class balancing.

---

# 40. What files are produced?

For each channel, the script writes an output directory containing several artifacts.

## `scaler.joblib`

The feature scaler fitted on outer training data only.

---

## `dctr_model_final.keras`

The final DCTR model trained on outer train+validation.

This is the model that can be used to evaluate genuinely unseen MC events.

---

## `dctr_factors_mc.npz`

Contains DCTR factors and fold provenance for the MC used by the closure experiment.

---

## `outer_test_fold.parquet`

Compact physical information for the protected outer test sample.

Useful for independent plots.

---

## `closure_model_before.keras`

Fresh classifier trained to distinguish Data from pre-DY MC.

---

## `closure_model_dy.keras`

Fresh classifier trained to distinguish Data from officially DY-corrected MC.

---

## `closure_model_dctr.keras`

Fresh classifier trained to distinguish Data from DCTR-corrected MC.

---

## `closure_before_test.npz`

Contains test predictions and C2ST test weights for the `before` closure classifier.

Analogous files are saved for `dy` and `dctr`.

---

## `comparison.json`

Contains the main AUC comparison:

```text
auc_before
auc_dy
auc_dctr
delta_dy_vs_before
delta_dctr_vs_before
delta_dctr_vs_dy
```

plus negative-weight diagnostics and the final DCTR cap.

---

# 41. How to interpret the final AUCs

The most direct comparison is

$$
|\mathrm{AUC}-0.5|.
$$

Smaller means less residual classifier-visible Data/MC separation.

For example:

```text
before = 0.610
DY     = 0.570
DCTR   = 0.525
```

suggests

```text
raw/pre-DY MC:
    substantial mismatch

official DY correction:
    improves agreement

DCTR:
    improves agreement further
```

But

```text
before = 0.610
DY     = 0.550
DCTR   = 0.590
```

would mean that the official DY correction generalizes better than DCTR.

The script does **not** assume DCTR must win.

That is exactly what the independent closure classifier is designed to test.

---

# 42. AUC close to 0.5 is necessary but not the whole story

A good final study should also inspect:

- ROC curves;
- closure-classifier output distributions;
- feature-by-feature Data/MC ratios;
- held-out variables not used by DCTR;
- DCTR-factor tails;
- effective sample size;
- dependence of results on DCTR capping;
- stability against network architecture;
- negative-weight fractions;
- analysis-relevant phase-space regions.

A global AUC compresses a high-dimensional comparison into one number.

The validation scripts should therefore be used together with the final closure AUC.

---

# 43. Why not simply evaluate the original DCTR NN after reweighting?

Because that would not be an independent test.

The DCTR model was optimized specifically to distinguish Data from pre-DY MC.

The correction is mathematically constructed from its own output.

Therefore asking the same network whether its own correction worked can produce an overly circular conclusion.

The closure classifier is initialized and trained from scratch after DCTR derivation.

It provides an independent adversary:

> **Can a new classifier still find residual Data/MC information after reweighting?**

This is a much stronger closure test.

---

# 44. Why do we need both cross-fitting and an outer test?

They solve two different problems.

## Cross-fitting

Cross-fitting makes the DCTR factors used during closure training/validation out-of-sample.

Without it, the closure network could be trained on MC carrying DCTR factors produced by a network that had already trained on those same rows.

---

## Outer test

The outer test gives a final population completely protected from DCTR fitting and closure fitting.

Without it, we would not have a clean unbiased final AUC.

Thus:

```text
cross-fitting
    protects the DCTR correction used on closure train/val

outer test
    protects the final reported closure measurement
```

Both are useful.

---

# 45. Compact step-by-step walkthrough matching the script header

The script header lists seven steps. Here they are in expanded form.

## Step 1 — load one channel

From configured Data and MC processes, load only:

```text
DY validation region
requested channel
configured selections
requested features
physical weight columns
```

---

## Step 2 — remove negative MC for classifier training

Measure their frequency first, then retain

```python
weight_uncorrected > 0
```

for the classifier study.

---

## Step 3 — make one OUTER split

Split Data and MC separately into

```text
outer train
outer validation
outer test
```

The outer test is protected from all DCTR fitting.

Fit preprocessing on outer train only.

---

## Step 4 — cross-fit DCTR on outer train+validation

Divide outer train+validation into \(K\) folds.

For each fold:

1. hold out one Data fold and one MC fold;
2. use the other folds as candidates;
3. make an internal DCTR train/validation split;
4. train a Data-vs-pre-DY-MC classifier using `weight_uncorrected`;
5. derive the DCTR cap from internal validation MC;
6. predict DCTR factors only for the held-out MC fold.

After all folds, every outer train/validation MC event has an out-of-fold factor.

---

## Step 5 — final DCTR model for outer test

Train one more Data-vs-pre-DY-MC DCTR classifier using only outer train+validation.

Derive its cap from its own internal validation MC.

Apply it to outer-test MC.

Now all MC rows have an out-of-sample DCTR factor.

---

## Step 6 — train three new closure classifiers

Use one fixed outer split.

### Before

$$
w_{\rm MC}=w_{\rm uncorrected}.
$$

### DY

$$
w_{\rm MC}=w_{\rm official\ DY}.
$$

### DCTR

$$
w_{\rm MC}
=w_{\rm uncorrected}
r_{\rm DCTR}(x).
$$

Each is class-balanced before entering binary cross entropy.

The three closure classifiers are independent NNs trained from scratch.

---

## Step 7 — final evaluation

All three networks predict on the same outer-test feature matrix.

Calculate weighted

```text
AUC
binary cross entropy
```

and save predictions.

The final headline quantities are

```text
AUC_before
AUC_DY
AUC_DCTR
```

with closer to 0.5 indicating better multivariate Data/MC closure.

---

# 46. Recommended mental model

When reading the code, keep three layers separate.

## Layer 1 — events

```text
Data events
MC events
```

These remain the same physical rows across the final comparison.

---

## Layer 2 — correction prescription

The MC row can carry one of three physical weights:

```text
before
DY
DCTR
```

This changes the distribution represented by MC.

---

## Layer 3 — diagnostic classifier

A fresh closure NN asks:

```text
"With this weighting prescription, can I still tell
Data and MC apart from their features?"
```

This third layer is only a measurement tool.

It does not define the correction.

---

# 47. Final conceptual summary

The procedure is intentionally nested:

$$
\boxed{
\text{derive correction}
\quad\rightarrow\quad
\text{freeze correction}
\quad\rightarrow\quad
\text{train new classifier}
}\\
\boxed{
\quad\rightarrow\quad
\text{measure closure on untouched events}
}
$$

The OUTER split guarantees that the final test is protected.

The K-fold cross-fitting guarantees that DCTR factors used to train the final closure classifier are themselves out-of-sample.

The three closure networks then answer the final comparison cleanly:

> **Which MC weighting prescription leaves the least multivariate Data/MC information for a new classifier to exploit?**

That is the role of `train_dctr_crossfit_closure.py`.
