# C2ST Data/MC Validation Pipeline

This document is a walk-through of the C2ST (classifier two-sample test) pipeline used to compare collision **Data** with simulated **Monte Carlo (MC)** events in the DY validation region.

The code is deliberately split into a **training/export stage** and several **independent validation scripts**. This keeps memory use under control, makes each check reproducible, and prevents TensorFlow state or large training arrays from affecting later validation steps.

---

## 1. Introduction to the problem

A traditional Data/MC validation often consists of one-dimensional histograms:

- plot Data and MC in `mli_ll_pt`;
- plot them in `mli_n_jet`;
- plot them in `mli_mbb`;
- inspect Data/MC ratios.

Those checks are essential, but they can miss discrepancies that appear only in **correlations between variables**. For example, Data and MC could agree separately in `mli_ll_pt` and `mli_n_jet`, while disagreeing in the joint distribution `(mli_ll_pt, mli_n_jet)`.

A classifier two-sample test addresses this by training a neural network to distinguish Data from MC using many observables simultaneously.

The label convention used throughout the code is:

```text
Data -> y = 1
MC   -> y = 0
```

If Data and MC have identical feature distributions, no classifier should be able to distinguish them better than random ordering. The expected ROC AUC is then approximately

$\mathrm{AUC} = 0.5.$

If the AUC is larger than 0.5, the classifier has found differences between Data and MC in the supplied feature space.

An important interpretation rule is:

> AUC measures **distinguishability**, not automatically whether a discrepancy is dangerous for the physics analysis.

With millions of events, even small differences can be statistically detectable. The purpose of the validation scripts is therefore to determine **where the difference comes from, whether a correction improves it, and whether it matters in the analysis phase space**.

---

# 2. The two main questions in this project

The same C2ST machinery can answer two related but distinct questions.

## 2.1 Does the DY correction improve Data/MC agreement?

The nominal DY correction is applied as an additional multiplicative event weight. The pipeline therefore trains two classifiers using the same events and the same train/validation/test split:

```text
before -> MC uses weight_uncorrected
after  -> MC uses weight (including the DY correction)
```

The relevant quantities are then

$$
\mathrm{AUC}_{\rm before},
\qquad
\mathrm{AUC}_{\rm after},
\qquad
\Delta\mathrm{AUC}
= \mathrm{AUC}_{\rm after} - \mathrm{AUC}_{\rm before}.
$$

A negative `delta AUC` indicates that the correction made Data and MC harder to distinguish.

For studying the correction itself, it is useful to run both:

1. a **targeted C2ST**, using the variables in which the DY correction is parameterized (for example `mli_ll_pt` and `mli_n_jet`);
2. a **global C2ST**, using the full feature set.

The targeted test asks whether the correction does what it was designed to do. The global test asks whether that improvement is important compared with all other residual Data/MC discrepancies.

## 2.2 How good is the overall nominal Data/MC model?

The `after` classifier uses the nominal corrected MC prediction, including the DY correction and all the other event weights. Its AUC answers the broader question:

> Can the classifier still distinguish Data from the MC prediction as it is actually used in the analysis?

This is a global modeling diagnostic. It should be complemented with localization studies and conventional Data/MC plots.

---

# 3. Repository structure

The pipeline is organized approximately as follows:

```text
c2st_pipeline/
|
|-- c2st_config.py
|-- dyvr_lib.py
|-- c2st_core.py
|-- c2st_artifacts.py
|-- c2st_nn.py
|
|-- validation_utils.py
|-- validate_metrics.py
|-- validate_auc_bins.py
|-- validate_significance.py
|-- validate_dctr.py
|-- validate_scaled_features.py
|-- validate_model_reload.py
|
|-- run_all_validations.sh
|
|-- c2st_artifacts/             # created by c2st_nn.py
`-- c2st_validation_plots/      # created by validation scripts
```

Each file has a deliberately narrow role.

### `c2st_config.py`

The user-facing configuration: paths, processes, features, cuts, NN architecture, split sizes, and training hyperparameters.

### `dyvr_lib.py`

Reads the ColumnFlow-produced Parquet files, combines the required event-weight components, identifies the DY region/channel, and performs memory-efficient event selection.

### `c2st_core.py`

Contains reusable preprocessing and statistics helpers:

- feature categorization;
- scalers;
- train/validation/test splitting;
- weighted binary cross entropy;
- class balancing.

### `c2st_nn.py`

The main training program. It loads one channel at a time, trains `before` and `after` networks, and saves only the artifacts needed downstream.

### `c2st_artifacts.py`

Defines the common format used to save and reload:

- metadata;
- scalers;
- test folds;
- test predictions;
- test weights;
- models;
- metrics.

### Validation scripts

Each validation is an independent process. This is intentional: after `c2st_nn.py` finishes, all large TensorFlow and training arrays are released by the operating system before a validation job starts.

---

# 4. Environment setup

If Miniforge is not installed yet, first download and install it.

On a typical 64-bit Linux machine:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

During the installer:

- accept the license;
- install Miniforge in the default location, usually `~/miniforge3`;
- allow the installer to initialize the shell if prompted.

After installation, either open a new shell or activate Miniforge manually with

```bash
source ~/miniforge3/bin/activate
```

It is recommended to create a dedicated environment for the C2ST/DCTR code rather than installing everything into the base environment:

```bash
conda create -n c2st python=3.11 -y
conda activate c2st
```

A typical Miniforge environment can then be activated in future sessions with

```bash
source ~/miniforge3/bin/activate
conda activate c2st
```

The scientific stack used by the code includes at least:

```bash
conda install -c conda-forge \
    numpy pandas scipy scikit-learn matplotlib mplhep \
    awkward pyarrow joblib
```

TensorFlow is normally installed through pip in the activated environment:

```bash
pip install 'tensorflow[and-cuda]'
```

After installation, it is useful to check that TensorFlow imports correctly:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

On a GPU node, you can additionally verify that TensorFlow sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

A successful GPU setup should return at least one device, for example:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

At this point the environment should contain everything required to run the C2ST, DCTR, plotting, and validation scripts.

GPU acceleration matters mainly for:

- NN training;
- NN inference.

The data loader, pandas operations, histograms, AUC calculations, and most resampling validations are CPU-side operations.

---

# 5. `c2st_config.py`: the central user configuration


## 5.1 Input paths

```python
STORE_ROOT = Path(...)
REDUCTION_DIR = Path(...)
SHIFT = 'nominal'
```

These identify the already-produced ColumnFlow outputs. The loader does not run ColumnFlow tasks itself; it reads Parquet outputs that already exist on disk.

## 5.2 Output paths

```python
ARTIFACT_DIR = Path('c2st_artifacts')
PLOT_DIR = Path('c2st_validation_plots')
```

`ARTIFACT_DIR` stores persistent NN outputs and test-fold information.

`PLOT_DIR` stores validation figures.

## 5.3 Process definitions

`MC_PROCESSES` groups MC datasets into logical physics processes. For example the different DY mass and jet-bin samples are grouped together.

Each MC group contains

```python
{
    'datasets': [...],
    'is_dy': True or False,
}
```

`is_dy=True` tells the loader that the process should receive the DY correction when constructing the `after` weight.

`DATA_PROCESSES` groups Data datasets/eras.

## 5.4 Alignment information

```python
ALIGNMENT_OK = {...}
```

The reduction and producer outputs are expected to be row-aligned branch by branch. If the production changes, the alignment should be checked again rather than blindly reusing an old dictionary.

The helper in `dyvr_lib.py` is:

```python
check_alignment(...)
```

It compares branch numbers and Parquet row counts without loading the full event content.

---

# 6. Choosing NN input features

The main input list is

```python
FEATURES = [...]
```

Only these variables are passed to the neural network.

A useful strategy is to run different feature sets for different scientific questions.

### Targeted DY-weight test

For example:

```python
FEATURES = [
    'mli_ll_pt',
    'mli_n_jet',
]
```

This asks whether the correction improves agreement in the variables most directly associated with its derivation.

### Global Data/MC test

Use the full set of analysis-related inputs. This asks whether the complete corrected MC prediction agrees with Data in the multidimensional feature space.

### Single-feature diagnostics

Running one feature at a time can identify whether one variable alone accounts for most of the global AUC.

For example:

```text
AUC(mli_ll_pt only) = 0.59
AUC(all features)   = 0.60
```

would indicate that `mli_ll_pt` dominates the separation.

In contrast:

```text
best one-feature AUC = 0.53
all-feature AUC      = 0.60
```

suggests that correlations between features are important.

---

# 7. `VALIDATION_VARS` versus `FEATURES`

The configuration contains

```python
VALIDATION_VARS = []
```

A validation variable is loaded and saved in the test fold but **not used as an NN input**.

This is especially useful for DCTR closure studies. A DCTR correction can trivially improve variables used to train the classifier. A more interesting test is whether the learned correction also improves a physically related variable that the network never saw.

Example:

```python
VALIDATION_VARS = [
    'mli_some_held_out_variable',
]
```

Then this variable can be plotted later with `validate_dctr.py`, while remaining absent from NN training.

---

# 8. Configurable physics selections

The configuration supports selections such as

```python
SELECTIONS = {
    'mli_met_pt': (None, 200.0),
}
```

The convention is:

```text
feature -> (lower_bound, upper_bound)
```

where

- lower bound is inclusive;
- upper bound is exclusive;
- `None` disables that side of the interval;
- non-finite values are rejected.

Examples:

```python
SELECTIONS = {
    'mli_met_pt': (0.0, 200.0),
    'mli_n_jet': (2.0, None),
}
```

These selections are applied **identically to Data and MC**.

## 8.1 Why selections happen inside the loader

The loader applies cuts branch by branch before retaining the large arrays. This avoids reading millions of events into permanent pandas tables only to discard them later.

The flow is approximately

```text
read one branch
    |
    +-- determine DY region/channel
    +-- apply configured selections
    +-- discard rejected events
    +-- read/retain required feature values
    +-- construct compact DataFrame
```

This substantially reduces peak memory.

## 8.2 Do not use selections just to make scaling easier

A selection should represent the phase space that you actually intend to validate.

If `MET > 200 GeV` is physically outside the scope of the study, then `MET < 200 GeV` is a legitimate selection.

If the high-MET tail is part of the analysis, it should not be removed merely because it makes preprocessing inconvenient. In that case the scaling strategy should be improved instead.

---

# 9. `LOAD_FEATURES`

The loader needs every variable used for one of three purposes:

1. NN input;
2. validation-only variable;
3. event selection.

Therefore the configuration constructs

```python
LOAD_FEATURES = list(dict.fromkeys(
    FEATURES + VALIDATION_VARS + list(SELECTIONS)
))
```

---

# 10. `dyvr_lib.py`: how events are loaded

The optimized loader is a crucial part of the pipeline because the input samples can contain tens of millions of events.

## 10.1 Branch-by-branch loading strategy

The data loader:

1. reads one branch;
2. uses `category_ids` to select the requested region/channel;
3. applies configured selections;
4. reads only requested features;
5. computes event weights;
6. stores only selected rows;
7. moves to the next branch.

The retained feature and weight columns use `float32` when `compact_dtypes=True`.

This is important because a `float32` column requires half the memory of `float64`.

## 10.2 Region and channel selection

For the C2ST run, `c2st_nn.py` calls the loader with approximately

```python
keep_region='dycr'
keep_channels=(channel,)
```

so the full analysis-region and unrelated-channel populations never need to remain in memory.

## 10.3 Event weights

For Data:

```text
weight_uncorrected = 1
weight             = 1
```

For MC, `weight_uncorrected` is the product of the configured nominal MC factors, while `weight` additionally contains the DY correction for DY processes.

Conceptually:

$$
w_{\rm before} = w_{\rm nominal\;MC\;without\;DY\;corr}
$$

and

$$
w_{\rm after} = w_{\rm before}\,w_{\rm DY}.
$$

This allows the same physical event sample to be used for both stages.

---

# 11. Negative MC event weights

The current C2ST training removes events satisfying

```python
weight_uncorrected <= 0
```

before constructing the training sample.

This is because ordinary binary cross entropy expects non-negative sample weights. A signed NLO event sample is not a conventional positive probability density, so directly passing negative sample weights to a probabilistic classifier is not well-defined.

The fraction removed is printed during training.

## 11.1 This does not mean negative events should disappear from the physics prediction

The current training strategy and eventual deployment should be distinguished.

### During NN training

Use a positive-weight population to learn a positive kinematic correction.

### During application to an independent MC sample

The NN uses only event features:

```python
p_data = model.predict(X)
```

The DCTR factor is positive:

$$
r(x)=\frac{p_{\rm Data}(x)}{1-p_{\rm Data}(x)}.
$$

For a signed MC event, the correction can then be applied multiplicatively:

$$
w_{\rm new}=w_{\rm original}\,r(x).
$$

If `w_original` is negative, the corrected weight remains negative because `r(x)` is positive.

Example:

```text
original MC weight = -0.27
DCTR factor        =  1.15
new MC weight      = -0.3105
```

Thus the event sign is preserved.

## 11.2 Important caveat

Because negative-weight events were not used to train the current classifier, applying the learned correction to them assumes that the learned density-ratio behavior is also meaningful in the regions occupied by those events.

A useful diagnostic is therefore to measure:

- fraction of MC events with negative weight;
- fraction of total absolute MC weight carried by them;
- whether they populate unusual regions of important observables.

---

# 12. Feature preprocessing (`c2st_core.py`)

Different HEP observables have very different numerical behavior.

## 12.1 Long-tailed kinematics

Variables matching patterns such as

```text
*_pt
*_ht
*_lt
masses
```

use

```python
RobustScaler()
```

only.

`RobustScaler` uses robust location/scale information rather than the global minimum and maximum. This avoids a small number of TeV-scale events compressing the physically dense part of the distribution.

## 12.2 Bounded/count/angle/score features

The remaining features use

```python
MinMaxScaler()
```

which is reasonable for variables that have a naturally bounded or compact range.

## 12.3 The scaler is fitted only on training events

This is essential.

The validation and test events must not influence the fitted preprocessing parameters. Otherwise information leaks from the evaluation sample into the training pipeline.

---

# 13. Train/validation/test splitting

Data and MC are independently shuffled with deterministic random seeds and split into

```text
training
validation
test
```

The test set is never used to fit the scaler or train the NN.

The same split is used for `before` and `after` within a channel. This is particularly important for the paired before/after significance test: row `i` in the before result corresponds to the same physical event as row `i` in the after result.

---

# 14. Why the MC weights are class-balanced for NN training

The physical MC weights generally do not sum to the same value as the number of Data events.

Before training, the code multiplies all MC event weights by one common factor so that the effective Data and MC class weights are balanced.

Conceptually:

$$
C_{\rm MC}=\frac{N_{\rm Data}}{\sum_i w_i^{\rm MC}}.
$$

The relative MC event weights are unchanged; only the entire MC class receives one global multiplicative factor.

This is useful because the C2ST should primarily ask:

> Can the classifier distinguish Data and MC using the event features?

rather than learning a trivial difference in total class normalization.

## 14.1 Does balancing artificially force AUC to 0.5?

No.

If Data and MC have identical shapes but unequal total weights, an unbalanced classifier can learn a different constant output probability, but a constant score still has AUC 0.5.

Global class balancing primarily improves the interpretation/calibration and makes DCTR density-ratio extraction cleaner.

## 14.2 Why the global factor does not change weighted AUC

Multiplying every MC weight by the same constant multiplies both the weighted-AUC numerator and its MC normalization by the same factor. It therefore cancels.

The physically important **relative** weights remain present.

---

# 15. Neural-network architecture

The architecture is configured through

```python
HIDDEN = (128, 128, 128)
```

which creates

```text
Input
  |
Dense(128, ReLU)
  |
Dense(128, ReLU)
  |
Dense(128, ReLU)
  |
Dense(1, sigmoid)
```

The final output is interpreted as

$$
p(x) = P(\mathrm{Data}\mid x).
$$

The model uses:

```text
optimizer: Adam
loss: binary cross entropy
learning rate: 1e-3 (configurable)
```

and the callbacks:

- `ReduceLROnPlateau`;
- `EarlyStopping` with best-weight restoration.

The goal is not to build the largest possible network. For a C2ST, a useful check is that the measured AUC is reasonably stable against modest architecture changes.

For example, compare:

```text
1 x 64
1 x 128
3 x 128
3 x 256 (optional stress test)
```

If AUC has already saturated, increasing model capacity does not add useful information.

---

# 16. Memory-aware training in `c2st_nn.py`

The training script intentionally processes **one channel at a time**:

```text
load 2mu
train before
train after
save outputs
free arrays

load 2e
train before
train after
save outputs
free arrays
```

This avoids retaining both channels simultaneously.

It also builds separate compact arrays for

```text
X_train
X_val
X_test
```

rather than storing one giant global `X` plus many indexed copies.

After each stage, TensorFlow sessions and Python objects are explicitly cleaned:

```python
tf.keras.backend.clear_session()
gc.collect()
```

The operating system will completely release process memory once `c2st_nn.py` exits, which is another reason validations are separate programs.

---

# 17. Running the training

From the pipeline directory:

```bash
python c2st_nn.py
```

For memory profiling on NAF:

```bash
/usr/bin/time -v python c2st_nn.py
```

At the end inspect

```text
Maximum resident set size (kbytes)
```

TensorFlow warnings such as

```text
Allocation of ... exceeds 10% of free system memory
```

are warnings rather than failures. They indicate a large CPU-memory allocation and should be interpreted together with the actual peak RSS and batch/job memory limit.

---

# 18. What is saved after training?

The training output is intentionally much smaller than the full training dataset.

A typical structure is

```text
c2st_artifacts/
|
|-- metadata.json
|
|-- 2mu/
|   |-- scaler.joblib
|   |-- model_before.keras
|   |-- model_after.keras
|   |-- test_fold.parquet
|   |-- before_test.npz
|   |-- after_test.npz
|   |-- before_metrics.json
|   `-- after_metrics.json
|
`-- 2e/
    `-- ...
```

## 18.1 `metadata.json`

Records important run configuration such as:

- feature list;
- validation variables;
- selections;
- scaling strategy;
- channels/stages;
- split parameters;
- network architecture;
- batch size.

This is important for reproducibility.

## 18.2 `scaler.joblib`

The fitted preprocessing transformation for that channel.

It must be applied to future events before passing them to the saved NN.

## 18.3 `model_before.keras` / `model_after.keras`

The trained TensorFlow models.

## 18.4 `test_fold.parquet`

Contains only the held-out test-fold rows, including:

- `y`;
- raw requested feature values;
- raw `weight_uncorrected`;
- raw `weight`.

These are physical/raw test-fold quantities rather than the class-balanced NN weights.

## 18.5 `before_test.npz` / `after_test.npz`

Contain compact test arrays such as:

- `p_test`;
- class-balanced `w_test`.

These are used for C2ST metrics.

---

# 19. `validate_metrics.py`

This is the first validation to run after training.

It summarizes basic held-out classifier performance, including quantities such as:

- weighted AUC;
- weighted binary cross entropy;
- ROC curves;
- classifier-output distributions.

Typical use:

```bash
python validate_metrics.py
```

## Interpretation

### AUC near 0.5

No classifier-visible separation at the sensitivity of the chosen model/features.

### AUC above 0.5

Residual Data/MC distinguishability exists.

There is no universal HEP threshold saying that a particular AUC is automatically acceptable or unacceptable. With millions of events, small discrepancies can be highly significant.

The next question is therefore where the discrepancy occurs and whether it is relevant to the analysis.

---

# 20. `validate_auc_bins.py`

This script evaluates classifier AUC in bins of a chosen raw physics variable.

It is useful for answering questions such as:

> Is the global AUC driven by the high-`ll_pt` tail?

or

> Does Data/MC agreement degrade at large jet multiplicity?

Example:

```bash
python validate_auc_bins.py --var mli_ll_pt --bins 10
```

Quantile-based edges are used for this AUC localization study. Quantile bins contain roughly similar numbers of events and are therefore useful when the variable has a long tail.

The script reports quantities including:

- Data count;
- MC count;
- weighted sums;
- MC Kish effective sample size;
- weighted AUC in each bin.

## Kish effective sample size

For weighted events,

$$
N_{\rm eff}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

A bin can contain many raw MC events while having a much smaller effective statistical sample if a few large weights dominate.

---

# 21. Quantile bins versus equal-width bins

There are two common binning strategies in the validation code.

## Quantile binning

Edges are chosen from quantiles of the observed sample.

Advantages:

- roughly equal event statistics per bin;
- useful for localization tests in long-tailed distributions.

Disadvantages:

- bin widths can be very different;
- less intuitive when the variable has physically meaningful fixed intervals.

## Equal-width binning in a fixed range

For the DCTR plots, the updated interface supports for example

```bash
python validate_dctr.py \
    --vars mli_ll_pt \
    --range 0 200 \
    --bins 60 \
    --normalization shape
```

This creates exactly 60 equal-width bins between 0 and 200.

This is completely valid and is often preferable for a conventional physics plot.

The option `--normalization shape` tests whether the Data and MC distributions agree *only* at shape level. Any residual normalization mismatch is taken off.
Run with `--normalization physical` to have a yield and shape comparison, even though the lack of negative event weights makes this comparison unfair (i.e. large discrepancies in yield between Data and MC are observed).

### Important nuance for `--normalization shape`

In shape-normalized DCTR plots, each MC histogram is normalized to Data using the events that fall inside the plotted bins/range.

Therefore changing

```text
--range
```

can change the plot-specific normalization factor. This is intentional: the plot then asks for shape closure **inside that displayed phase space**.

---

# 22. `validate_significance.py`

This script performs two different statistical checks.

## 22.1 Paired bootstrap of `delta AUC`

The primary before/after question is

$$
\Delta\mathrm{AUC}
=\mathrm{AUC}_{\rm after}-\mathrm{AUC}_{\rm before}.
$$

Because before and after use the same held-out events, the comparison is paired.

Example:

```bash
python validate_significance.py \
    --bootstrap-resamples 50 \
    --permutation-resamples 100
```

The implementation is memory-aware. Rather than repeatedly creating several million-row resampled arrays and re-sorting predictions, it:

- sorts fixed classifier scores once;
- represents each bootstrap sample with row multiplicities;
- evaluates weighted AUC using the cached ordering.

The important output is the confidence interval on `delta AUC`.

For example:

```text
delta AUC = -0.035
95% CI = [-0.041, -0.029]
```

would provide evidence that the correction reproducibly reduces classifier distinguishability.

## 22.2 Fixed-model label permutation check

The second test permutes labels while keeping the already-trained classifier fixed.

This is a useful sanity check, but it is **not the exact formal C2ST null**, which would require retraining the classifier for every permutation.

With very large datasets, the p-value will often become extremely small for even modest discrepancies. Therefore effect size and localization are generally more informative than a tiny p-value alone.

---

# 23. `validate_scaled_features.py`

This script reloads the test fold and fitted scaler, transforms only the test events, and plots scaled feature distributions.

Example:

```bash
python validate_scaled_features.py --channel 2mu
```

It uses the **class-balanced C2ST test weights** stored in `before_test.npz` and `after_test.npz`.

Therefore the total Data and MC normalizations in these plots are expected to be close by construction.

This is primarily a **shape / NN-input diagnostic**, not a physical-yield closure plot.

This distinction is important when comparing it with `validate_dctr.py`.

---

# 24. `validate_model_reload.py`

This is a serialization/reproducibility check.

It reloads:

- the saved scaler;
- the saved Keras model;
- raw test features;

then runs inference again and compares the new predictions with those saved during training.

Example:

```bash
python validate_model_reload.py \
    --channel 2mu \
    --stage after
```

If the maximum prediction difference is tiny, the persisted model/scaler pair correctly reproduces the training-time inference.

This check is useful before using the model on a new dataset.

---

# 25. DCTR: what it is doing

The DCTR-style correction uses the classifier output as a density-ratio estimator.

The network is trained with

```text
Data = 1
MC   = 0
```

and balanced effective class priors.

If

$$
p(x)=P(\mathrm{Data}\mid x),
$$

then the classifier odds approximately estimate

$$
r(x)=\frac{p(x)}{1-p(x)}
\approx
\frac{f_{\rm Data}(x)}{f_{\rm MC}(x)}.
$$

The DCTR multiplicative factor is therefore

```python
dctr_weight = p_data / (1.0 - p_data)
```

For this project, DCTR is derived from the **before-DY classifier** and applied to the **uncorrected MC event weight**.

Conceptually:

$$
w_{\rm MC}^{\rm DCTR} =
w_{\rm MC}^{\rm before}\,r(x).
$$

This lets us compare two alternative reweightings starting from the same uncorrected MC prediction:

```text
existing correction:
weight_uncorrected * DY_weight

classifier correction:
weight_uncorrected * DCTR_weight
```

---

# 26. DCTR capping

Classifier odds can become very large when

$$
p(x)\rightarrow 1.
$$

This can occur in regions with poor Data/MC overlap or limited statistics.

The code therefore optionally caps the DCTR factor at a chosen MC quantile, for example

```text
cap_quantile = 0.995
```

The validation reports:

- cap value;
- number of capped events;
- fraction of weighted MC affected.

A large capped fraction is a warning that the result depends strongly on a poorly constrained tail.

---

# 27. Plotting the DCTR-weight distribution

Use the DCTR validation option that plots the MC-only DCTR factors.

For example:

```bash
python validate_dctr.py \
    --vars mli_ll_pt \
    --plot-weights
```

The diagnostic compares the raw and capped DCTR factors and marks the cap.

This plot helps determine whether the correction is:

```text
mostly around 1 with a modest spread
```

or instead contains a large, unstable high-weight tail.

A logarithmic y-axis is useful for this diagnostic.

---

# 28. DCTR closure modes

`validate_dctr.py` supports two conceptually different normalization modes.

## 28.1 Shape normalization

Typical command:

```bash
python validate_dctr.py \
    --vars mli_ll_pt \
    --range 0 200 \
    --bins 60 \
    --normalization shape
```

The plotted histograms are:

```text
Data
MC before
MC DY corrected
MC DCTR
```

and each MC alternative is independently normalized to the Data integral in the displayed bins.

Therefore this plot asks:

> Ignoring the overall yield difference, which weighting gives the best differential shape agreement with Data in this range?

This is the recommended mode for comparing the **effect of the DY versus DCTR reweighting on feature shapes**.

The corresponding scale factors should be printed or recorded so that the imposed normalization is transparent.

## 28.2 Physical normalization

Typical command:

```bash
python validate_dctr.py \
    --vars mli_ll_pt \
    --range 0 200 \
    --bins 60 \
    --normalization physical
```

No per-curve normalization-to-Data is imposed.

This asks:

> Does the absolute weighted MC yield agree with Data as well as the shape?

The two modes should not be confused.

A shape-normalized closure can look excellent while the physical MC normalization is wrong.

---

# 29. DCTR plot range and binning

The updated DCTR interface separates two choices:

```text
--range MIN MAX
--bins N
```

For example:

```bash
--range 0 200 --bins 60
```

means:

- consider the displayed interval 0 to 200;
- divide it into 60 equal-width bins.

The bin width is therefore

$$
\frac{200-0}{60}\approx 3.33\;\mathrm{GeV}.
$$

If `--range` is omitted, the finite minimum and maximum of the plotted variable are used.

The plot colors are intentionally fixed to make different runs visually comparable:

```text
Data            -> black
MC before       -> tab:blue
MC DY corrected -> tab:red
MC DCTR         -> darkorange
```

The ratio panel uses the same color convention.

---

# 30. What does a good DCTR closure mean?

Suppose the shape-normalized ratio panel shows

```text
MC before       -> clear slope / deviations
MC DY corrected -> closer to one
MC DCTR         -> very close to one
```

Then DCTR has found a multidimensional event reweighting that improves the distribution of the plotted variable beyond the existing DY correction.

However, several questions remain:

1. Was the plotted feature used to train the DCTR classifier?
2. Does closure also improve for held-out variables?
3. Are the DCTR weights stable or dominated by large tails?
4. Does the improvement persist on independent events?
5. Is the correction physically interpretable and robust enough to use beyond a diagnostic study?

A closure on a training feature is useful as a sanity check but is not by itself strong evidence of generalization.

---

# 31. Applying a saved NN/DCTR correction to new MC events

For a genuinely independent MC sample, the event's original MC weight is **not needed as an NN input**.

The procedure is:

1. apply the same physics selections;
2. construct the same feature columns in the same order;
3. apply the saved scaler;
4. evaluate the saved `before` NN;
5. compute the DCTR factor;
6. multiply the original physical MC weight by that factor.

Conceptually:

```python
X = scaler.transform(new_events[FEATURES])
p_data = model.predict(X)
dctr = p_data / (1.0 - p_data)
new_weight = original_weight_uncorrected * dctr
```

If capping is part of the chosen correction prescription, the same fixed cap should be applied consistently. A production correction should generally not redefine its cap separately on every new sample.

---

# 32. Recommended study sequence for validating the DY correction

A sensible workflow is:

## Step 1: targeted input space

Train using only variables directly related to the correction, for example

```python
FEATURES = [
    'mli_ll_pt',
    'mli_n_jet',
]
```

Record:

```text
AUC before
AUC after
delta AUC
bootstrap CI(delta AUC)
```

This answers whether the correction works in its intended phase space.

## Step 2: individual features

Repeat one variable at a time to identify dominant one-dimensional discrepancies.

## Step 3: full feature set

Train with the global feature list.

If before and after both remain at AUC ~0.60 while the targeted test improves strongly, the likely interpretation is:

> The DY correction improves its intended variables, but other Data/MC mismodeling dominates the global classifier.

## Step 4: localize global separation

Use:

```text
validate_auc_bins.py
conventional Data/MC plots
classifier-output distributions
```

## Step 5: DCTR diagnostic

Compare:

```text
MC before
MC DY corrected
MC DCTR
```

in both training and held-out variables.

## Step 6: connect discrepancies to the physics analysis

Ask whether the regions identified by C2ST are important for:

- signal acceptance;
- final discriminant;
- background yields;
- systematic uncertainties;
- control-to-signal extrapolation.

This is what determines whether a statistically visible discrepancy is actually concerning.

---

# 33. How to interpret AUC values

There is no universal acceptance threshold such as “AUC < 0.55 means good MC”.

A useful qualitative interpretation is:

```text
AUC ~ 0.50
No classifier-visible separation.

AUC ~ 0.51-0.55
Small discrepancy; can still be extremely statistically significant
with a very large dataset.

AUC ~ 0.55-0.60
Meaningful multivariate distinguishability; localization is warranted.

AUC >= ~0.60
Substantial classifier-visible separation; do not declare overall
Data/MC agreement good without understanding the source and impact.
```

These are diagnostic heuristics, not formal thresholds.

The final statement should combine multiple pieces of information:

- `AUC_after`;
- `delta AUC`;
- bootstrap confidence interval on `delta AUC`;
- AUC in bins of physical variables;
- Data/MC feature ratios;
- classifier-output distributions;
- effective MC statistics;
- impact on analysis-relevant observables.

---

# 34. AUC and normalization are different questions

An important conceptual point is that ROC AUC is fundamentally sensitive to **ranking / shape separation**, not an overall normalization mismatch.

This is why the C2ST uses globally class-balanced weights.

To assess absolute Data/MC yields, use the raw physical weights and a physical-normalization plot.

To assess multidimensional shape agreement, use the class-balanced C2ST and/or explicitly shape-normalized histograms.

Both questions are scientifically useful, but they should be reported separately.

---

# 35. Common pitfalls

## 35.1 Treating AUC as an automatic pass/fail criterion

AUC > 0.5 means the classifier can distinguish the samples. It does not by itself quantify the effect on the final analysis.

## 35.2 Changing the test set between before and after

Do not do this. The paired comparison assumes identical physical test events.

## 35.3 Fitting preprocessing on the full dataset

The scaler must be fitted on training rows only.

## 35.4 Using a validation-only variable as an NN input by accident

Keep it in `VALIDATION_VARS`, not `FEATURES`.

## 35.5 Removing a tail solely because it is inconvenient for scaling

Use `RobustScaler` for long-tailed observables. Only use a hard selection when it defines the intended physics phase space.

## 35.6 Interpreting a shape-normalized plot as a yield closure

In `--normalization shape`, the MC integral is imposed to match Data in the displayed range.

## 35.7 Inferring Data/MC identity from event weights

Do not identify Data by checking whether `weight == 1`. Some MC events can also have weights near or exactly one. Use the explicit `y` label.

## 35.8 Forgetting the label convention in DCTR

The current convention is

```text
Data = 1
MC = 0
```

so the correction is

$$
\frac{p_{\rm Data}}{1-p_{\rm Data}}.
$$

Inverting this ratio would reweight in the wrong direction.

## 35.9 Using negative sample weights in binary cross entropy

The current standard NN training cannot directly interpret signed event weights as ordinary likelihood weights.

## 35.10 Forgetting that the DCTR cap is part of the prescription

If a correction is ever deployed beyond a diagnostic test, the cap should be defined and documented reproducibly.

---

# 36. Memory and performance checklist

For very large samples:

- load only requested features;
- apply cuts during branch loading;
- keep compact `float32` columns;
- process one channel at a time;
- do not retain full training matrices after training;
- save only test-fold information needed downstream;
- run validations as separate programs;
- use subsampling while developing expensive bootstrap/permutation checks;
- monitor peak RSS with `/usr/bin/time -v`;
- use GPU for TensorFlow training/inference, not automatically for pandas/histograms.

---

# 37. Reproducibility checklist before quoting a result

Before recording an AUC or closure result in a thesis/note, record:

- input production paths/version;
- channel;
- region;
- process composition;
- feature list;
- validation-variable list;
- event selections;
- negative-weight treatment;
- before/after weight definitions;
- scaler strategy;
- NN architecture;
- batch size;
- random seed;
- train/validation/test fractions;
- event cap/subsampling, if any;
- DCTR cap prescription;
- DCTR histogram range/binning;
- normalization mode (`shape` or `physical`).

Most of the training configuration is written to `metadata.json`, but plot-specific arguments should also be recorded with the produced figures/results.

---

# 38. Suggested commands for a complete run

## Train models

```bash
/usr/bin/time -v python c2st_nn.py
```

## Basic metrics

```bash
python validate_metrics.py
```

## Localize AUC in a variable

```bash
python validate_auc_bins.py \
    --var mli_ll_pt \
    --bins 10
```

## Significance / stability of before-after change

```bash
python validate_significance.py \
    --bootstrap-resamples 50 \
    --permutation-resamples 100
```

## DCTR shape comparison in a physically interesting interval

```bash
python validate_dctr.py \
    --vars mli_ll_pt \
    --range 0 200 \
    --bins 60 \
    --normalization shape \
    --plot-weights
```

## DCTR physical-yield comparison

```bash
python validate_dctr.py \
    --vars mli_ll_pt \
    --range 0 200 \
    --bins 60 \
    --normalization physical
```

## Scaled input-feature checks

```bash
python validate_scaled_features.py --channel 2mu
```

## Saved-model reproducibility

```bash
python validate_model_reload.py \
    --channel 2mu \
    --stage after
```

---

# 39. Ideal next steps

Several natural follow-up studies can be built on this framework.

## 39.1 Architecture stability scan

Automate a small scan over network sizes and check whether the AUC plateaus.

## 39.2 Feature importance

Use a manageable held-out subsample to measure permutation importance and identify which inputs are responsible for classification power.

## 39.3 Correlation diagnostics

Compare the full-feature AUC with single-feature AUCs to determine whether discrepancies are mainly one-dimensional or correlation-driven.

## 39.4 Negative-weight studies

Compare the phase-space distribution of positive- and negative-weight events and quantify whether excluding negative events from training can bias the learned correction.

## 39.5 Analysis-region extension

A C2ST can be extended to other regions, but care is required in a signal-sensitive region. If genuine signal is present, Data being distinguishable from background-only MC is not automatically a modeling failure.

Validation/control regions are therefore the natural place to establish modeling quality before interpreting a signal region.

---

# 40. A concise mental model of the complete pipeline

```text
ColumnFlow Parquet outputs
        |
        v
memory-efficient branch loader
  - DY region/channel
  - configured selections
  - requested features only
  - physical MC weights
        |
        v
positive-weight training population
        |
        +------------------------------+
        |                              |
        v                              v
  BEFORE classifier              AFTER classifier
  weight_uncorrected             DY-corrected weight
        |                              |
        +--------------+---------------+
                       |
                       v
               held-out predictions
                       |
                       v
              save compact artifacts
                       |
       +---------------+------------------+
       |               |                  |
       v               v                  v
    metrics        AUC localization    significance
       |
       +-------------------+
                           |
                           v
                  DCTR from BEFORE NN
                           |
                 p_data / (1-p_data)
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
       DY weight                        DCTR weight
          |                                 |
          +---------------+-----------------+
                          |
                          v
            Data/MC closure comparison
             shape and physical modes
```

---

# 41. Final interpretation

The C2ST is best viewed as a **high-dimensional detector of mismodeling**, not as a single-number certification procedure.

A good final analysis statement should therefore not simply say

```text
AUC = 0.60, therefore MC is bad
```

or

```text
AUC is not very far from 0.5, therefore MC is fine.
```

Instead, aim for a conclusion of the form:

> The DY correction changes the C2ST AUC from the before value to the after value. The paired bootstrap quantifies whether this change is stable. Residual separation is localized using binned AUC and conventional Data/MC comparisons. The affected regions/variables are then assessed in terms of their relevance to the physics analysis and the available modeling uncertainties.

That combination of **global sensitivity, localization, uncertainty, and physics impact** is the scientifically useful outcome of the pipeline.

