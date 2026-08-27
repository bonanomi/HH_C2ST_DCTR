# C2ST training + independent validation pipeline

## Introduction and starting point

Before diving into the actual code:

- Setup the working environment as explained [here](C2ST_DOC.md#4-environment-setup).
- Familiarize with some of the basic concepts that we are going to use throughout the project by solving the [`01_histograms_weights_datamc.ipynb`](notebooks/01_histograms_weights_datamc.ipynb) notebook. Instructions on how to set up the JupyterLab interface and run it can be found in the dedicated [`README`](notebooks/README.md).
- Start exploring the concept of Classifier Two Sample Test (C2ST) from the [toy example](toy_c2st_demo/TOY_C2ST_GUIDE.md) to have a gentle introduction to what is going to be the core part of the project.

## HTCondor submission

The training stages of the C2ST/DCTR tests described below can be submitted as jobs with `HTCondor` using [`make_condor_submit.py`](make_condor_submit.py).

Read [`CONDOR.md`](CONDOR.md) for a more detailed explanation of this script and how to use it.

## C2ST documentation

The complete C2ST/DCTR workflow is documented in [`C2ST_DOC.md`](C2ST_DOC.md).
The rest of this `README` provides only a general overview of the tool.
For a more detailed description of the problem, the individual steps of the workflow, the pitfalls, and some ideas on what to do next, consult the documentation.

### Index

- [1. Introduction to the problem](C2ST_DOC.md#1-introduction-to-the-problem)
- [2. The two main questions in this project](C2ST_DOC.md#2-the-two-main-questions-in-this-project)
  - [2.1 Does the DY correction improve Data/MC agreement?](C2ST_DOC.md#21-does-the-dy-correction-improve-datamc-agreement)
  - [2.2 How good is the overall nominal Data/MC model?](C2ST_DOC.md#22-how-good-is-the-overall-nominal-datamc-model)
- [3. Repository structure](C2ST_DOC.md#3-repository-structure)
- [4. Environment setup](C2ST_DOC.md#4-environment-setup)
- [5. `c2st_config.py`: the central user configuration](C2ST_DOC.md#5-c2st_configpy-the-central-user-configuration)
- [6. Choosing NN input features](C2ST_DOC.md#6-choosing-nn-input-features)
- [7. `VALIDATION_VARS` versus `FEATURES`](C2ST_DOC.md#7-validation_vars-versus-features)
- [8. Configurable physics selections](C2ST_DOC.md#8-configurable-physics-selections)
- [9. `LOAD_FEATURES`](C2ST_DOC.md#9-load_features)
- [10. `dyvr_lib.py`: how events are loaded](C2ST_DOC.md#10-dyvr_libpy-how-events-are-loaded)
- [11. Negative MC event weights](C2ST_DOC.md#11-negative-mc-event-weights)
- [12. Feature preprocessing (`c2st_core.py`)](C2ST_DOC.md#12-feature-preprocessing-c2st_corepy)
- [13. Train/validation/test splitting](C2ST_DOC.md#13-trainvalidationtest-splitting)
- [14. Why the MC weights are class-balanced for NN training](C2ST_DOC.md#14-why-the-mc-weights-are-class-balanced-for-nn-training)
- [15. Neural-network architecture](C2ST_DOC.md#15-neural-network-architecture)
- [16. Memory-aware training in `c2st_nn.py`](C2ST_DOC.md#16-memory-aware-training-in-c2st_nnpy)
- [17. Running the training](C2ST_DOC.md#17-running-the-training)
- [18. What is saved after training?](C2ST_DOC.md#18-what-is-saved-after-training)
- [19. `validate_metrics.py`](C2ST_DOC.md#19-validate_metricspy)
- [20. `validate_auc_bins.py`](C2ST_DOC.md#20-validate_auc_binspy)
- [21. Quantile bins versus equal-width bins](C2ST_DOC.md#21-quantile-bins-versus-equal-width-bins)
- [22. `validate_significance.py`](C2ST_DOC.md#22-validate_significancepy)
- [23. `validate_scaled_features.py`](C2ST_DOC.md#23-validate_scaled_featurespy)
- [24. `validate_model_reload.py`](C2ST_DOC.md#24-validate_model_reloadpy)
- [25. DCTR: what it is doing](C2ST_DOC.md#25-dctr-what-it-is-doing)
- [26. DCTR capping](C2ST_DOC.md#26-dctr-capping)
- [27. Plotting the DCTR-weight distribution](C2ST_DOC.md#27-plotting-the-dctr-weight-distribution)
- [28. DCTR closure modes](C2ST_DOC.md#28-dctr-closure-modes)
- [29. DCTR plot range and binning](C2ST_DOC.md#29-dctr-plot-range-and-binning)
- [30. What does a good DCTR closure mean?](C2ST_DOC.md#30-what-does-a-good-dctr-closure-mean)
- [31. Applying a saved NN/DCTR correction to new MC events](C2ST_DOC.md#31-applying-a-saved-nndctr-correction-to-new-mc-events)
- [32. Recommended study sequence for validating the DY correction](C2ST_DOC.md#32-recommended-study-sequence-for-validating-the-dy-correction)
- [33. How to interpret AUC values](C2ST_DOC.md#33-how-to-interpret-auc-values)
- [34. AUC and normalization are different questions](C2ST_DOC.md#34-auc-and-normalization-are-different-questions)
- [35. Common pitfalls](C2ST_DOC.md#35-common-pitfalls)
- [36. Memory and performance checklist](C2ST_DOC.md#36-memory-and-performance-checklist)
- [37. Reproducibility checklist before quoting a result](C2ST_DOC.md#37-reproducibility-checklist-before-quoting-a-result)
- [38. Suggested commands for a complete run](C2ST_DOC.md#28-suggested-commands-for-a-complete-run)
- [39. Ideal next steps](C2ST_DOC.md#41-ideal-next-steps)
- [40. A concise mental model of the complete pipeline](C2ST_DOC.md#41-a-concise-mental-model-of-the-complete-pipeline)
- [41. Final interpretation](C2ST_DOC.md#42-final-interpretation)

## Train/export

Run from this directory on the NAF GPU worker:

```bash
python c2st_nn.py
```

The training process works on one channel at a time and writes `c2st_artifacts/`:

- `metadata.json`: feature/stage/split configuration
- `<channel>/scaler.joblib`: fitted training-fold scaler
- `<channel>/model_before.keras`, `model_after.keras`: serialized Keras models
- `<channel>/test_fold.parquet`: raw test-fold features, label, physical before/after weights
- `<channel>/before_test.npz`, `after_test.npz`: saved NN scores and class-balanced test weights
- `<channel>/*_metrics.json`: training-time AUC/BCE checks

No full training `X`, `DATASETS`, or train/validation rows are persisted.

## Independent validations

Each command starts a fresh process and therefore cannot retain TensorFlow/training memory from the previous step.

```bash
python validate_metrics.py
python validate_auc_bins.py --var mli_ll_pt --bins 10
python validate_significance.py --bootstrap-resamples 50 --permutation-resamples 100
python validate_dctr.py --vars mli_ll_pt
python validate_scaled_features.py --channel 2e
python validate_model_reload.py --channel 2mu --stage after
```

For a fast significance-development pass, use e.g. `--subsample 1000000`. Use `--subsample 0` (the default) for the full test fold.

The permutation result is intentionally the fixed-trained-model diagnostic from the original additions code; it is not the exact C2ST null that would retrain a network for every permutation.


## Configurable phase-space selections

`c2st_config.py` contains `SELECTIONS`, which are applied branch-by-branch during parquet loading to both Data and MC. For example:

```python
SELECTIONS = {
    'mli_met_pt': (None, 200.0),
}
```

Bounds are `(inclusive_lower, exclusive_upper)` and `None` disables a bound. Non-finite values in a selected variable are rejected. Selection variables are automatically added to `LOAD_FEATURES`. Only use these cuts when they define the phase space you actually want to validate; do not cut physical tails just to accommodate preprocessing.

## Scaling

Long-tailed kinematic inputs (`*_pt`, `*_ht`, `*_lt`, masses, etc.) use `RobustScaler` only. Bounded/count/angular/score variables continue to use `MinMaxScaler`.

## C2ST weights

Training uses the physical relative MC event weights but multiplies the entire MC class by one global class-balance factor. This does not change the ideal weighted ROC AUC (a constant rescaling of all weights within one class cancels in the AUC), but it gives Data and MC equal effective class priors in the binary cross-entropy. That makes optimization better behaved and gives the classifier odds `p/(1-p)` the normalized density-ratio interpretation needed by the DCTR diagnostic. Physical, unscaled weights remain saved in `test_fold.parquet` for closure/yield plots.

## A final closure test

The code in [`c2st_final_closure`](c2st_final_closure) implements the final closure test. Consider running it only after having run the toy study and the scripts in this folder. After everything is well understood, move to this final test.
Start by reading the relevant [`README`](c2st_final_closure/README_DCTR_CLOSURE.md).
