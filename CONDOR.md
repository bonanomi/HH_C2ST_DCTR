# HTCondor submission helper

`make_condor_submit.py` generates a worker wrapper, a submission file, and a log directory:

```text
condor_jobs/
└── <task>_<timestamp>/
    ├── run.sh
    ├── job.sub
    └── logs/
```

Use `--job-dir` to choose an explicit directory.

By default, the helper only generates the files and prints the submission command. Add `--submit` to submit immediately.

## Supported tasks

| Task | Program |
|---|---|
| `training` | `dctr_c2st_generic.training` — current configurable DCTR training |
| `signed-plots` | `dctr_c2st_generic.plot_signed_physical` — full signed Data/MC plots |
| `year-plots` | `dctr_c2st_generic.plot_year_transfer` — year-comparison plots |
| `main` | `c2st_nn.py` — legacy main C2ST workflow |
| `closure` | `c2st_final_closure.train_dctr_crossfit_closure` — separate cross-fitted closure workflow |

**Use `--task training` for the current generic DCTR workflow.** `--task main` still selects the legacy program.

## Generic DCTR training

Run from the repository root:

```bash
cd /data/dust/user/bonanomm/HH_C2ST_DCTR

python make_condor_submit.py \
    --task training \
    --repo-dir "$PWD" \
    --conda-base /data/dust/user/bonanomm/miniforge3 \
    --conda-env c2st \
    --cpus 4 \
    --memory 48GB \
    --disk 10GB \
    --gpus 1 \
    --job-dir condor_jobs/data_mc24_training_full_v2 \
    --submit \
    -- \
    --config config_data_mc24_dy_only_all.py \
    --output c2st_artifacts/data_mc24_training_full_v2
```

Choose fresh job and artifact directories for a new training.

The training configuration controls:

- Inclusive versus DY-only correction.
- Input features, channels, regions, and weight definitions.
- Shape versus yield normalization.
- Fold count, model settings, and training controls.

For this task, set channels and folds in the configuration, not with the helper’s `--channels` or `--folds` options.

## Full signed Data/MC plots

Use the latest `plot_signed_physical.py` to produce before/after DCTR plots for:

- `mli_ll_pt`
- `mli_n_jet`
- `mli_mll`

```bash
python make_condor_submit.py \
    --task signed-plots \
    --repo-dir "$PWD" \
    --conda-base /data/dust/user/bonanomm/miniforge3 \
    --conda-env c2st \
    --cpus 4 \
    --memory 48GB \
    --disk 10GB \
    --gpus 0 \
    --job-dir condor_jobs/data_mc24_full_signed_mll \
    --submit \
    -- \
    --channel-dir c2st_artifacts/data_mc24_training_full/2mu \
    --output c2st_artifacts/data_mc24_training_full/2mu/signed_plots_cached_v1 \
    --cache-dir /data/dust/user/bonanomm/HH_C2ST_DCTR/c2st_cache/signed_physical
```

`--channel-dir` must point to a completed training’s channel directory containing `inference/` and the saved training artifacts. Use a fresh plotting output directory.

The script reads the saved correction scope automatically:

| Training setup | Correction applied |
|---|---|
| Inclusive | All configured MC components |
| DY-only | DY only; non-DY factors remain 1 |

Plots include the full selected physical population, including negative- and zero-weight MC events. Signed weights retain their sign after correction.

Saved factors and fold assignments are reused for classifier events. Excluded nonpositive-weight events within the corrected scope are evaluated with the exported final model. The saved normalization and capping are preserved.

`mli_mll` is loaded separately for plotting; it does not need to have been a training input. No retraining occurs.

These are full-population physical plots, not independent outer-test closure tests. Outputs include PNG/PDF figures and `signed_diagnostics.json`; the ratio panel spans 0.8–1.2.

### Plot cache

The cache stores selected Data and full signed MC tables:

- The first run reads the original inputs and builds the cache.
- Compatible subsequent runs print `Cache HIT` and reuse those tables.
- Changes to the configuration, requested columns, loader code, or input-file sizes/modification times invalidate the cache.
- `--refresh-cache` forces rebuilding.
- `--no-cache` bypasses caching.
- Existing `year_plots` caches are separate and are not reused.

Forward cache options after `--`. Use the same absolute cache directory for local and Condor runs, and use only a trusted cache directory.

Caching accelerates input loading. Provenance checks and inference for excluded corrected events still run.

`--gpus 0` runs plotting and inference on CPU. Set `--gpus 1` if GPU inference is desired; a GPU does not accelerate the parquet-loading stage.

## Year-comparison plots

The existing year-comparison plotting module is available through `--task year-plots`:

```bash
python make_condor_submit.py \
    --task year-plots \
    --repo-dir "$PWD" \
    --conda-base /data/dust/user/bonanomm/miniforge3 \
    --conda-env c2st \
    --cpus 4 \
    --memory 48GB \
    --gpus 0 \
    --job-dir condor_jobs/year_plots
```

Append the arguments required by `dctr_c2st_generic.plot_year_transfer` after `--`, using the same arguments as a working local invocation. Set plot channels in the plotting configuration.

## Legacy main C2ST training

```bash
python make_condor_submit.py \
    --task main \
    --repo-dir "$PWD" \
    --conda-base /data/dust/user/bonanomm/miniforge3 \
    --conda-env c2st \
    --memory 24GB \
    --cpus 4 \
    --gpus 1 \
    --runtime-hours 8
```

This launches `c2st_nn.py`, not the generic training module.

## Separate cross-fitted DCTR closure workflow

One channel per job is recommended. The historical inclusive target is the default:

```bash
python make_condor_submit.py \
    --task closure \
    --repo-dir "$PWD" \
    --conda-base /data/dust/user/bonanomm/miniforge3 \
    --conda-env c2st \
    --channels 2mu \
    --folds 5 \
    --memory 32GB \
    --cpus 4 \
    --gpus 1 \
    --runtime-hours 24
```

For the DY-only target in this closure workflow:

```bash
python make_condor_submit.py \
    --task closure \
    --repo-dir "$PWD" \
    --conda-base /data/dust/user/bonanomm/miniforge3 \
    --conda-env c2st \
    --channels 2mu \
    --folds 5 \
    --memory 32GB \
    --cpus 4 \
    --gpus 1 \
    --runtime-hours 24 \
    -- \
    --dctr-target dy_only
```

Repeat for `2e` if desired.

`--dctr-target` belongs to this separate closure program. For `--task training`, configure the correction scope in the generic training configuration.

## Argument forwarding and submission

Arguments before `--` configure the submission helper. Arguments after `--` are forwarded to the selected Python program.

Only forward options supported by that program. For example:

- Generic training: `--config`, `--output`.
- Signed plotting: `--channel-dir`, `--output`, `--cache-dir`, `--refresh-cache`, `--no-cache`, `--chunk-size`, `--save-events`.

Without `--submit`, inspect the generated files and then run:

```bash
condor_submit condor_jobs/YOUR_JOB_DIRECTORY/job.sub
```

The helper creates `logs/` automatically.

## Worker environment

The wrapper activates the requested Conda environment, preferring:

```bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
```

It falls back to `bin/activate` if needed.

For `training`, `year-plots`, and `signed-plots`, the helper supplies:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true
OMP_NUM_THREADS=<requested CPUs>
TF_NUM_INTRAOP_THREADS=<requested CPUs>
TF_NUM_INTEROP_THREADS=1
```

For these tasks, `--gpus 0` also sets `CUDA_VISIBLE_DEVICES=-1`.

Override defaults or add variables with repeated `--environment NAME=VALUE` arguments.

The wrapper records the hostname, Git revision, Python/TensorFlow versions, visible GPUs, and exact command in the job log.

## Shared filesystem and site-specific directives

The repository, Conda installation, input data, output paths, and cache must be accessible on the worker through the shared filesystem.

The submit file includes:

```text
initialdir = <repository path>
should_transfer_files = NO
getenv = True
```

Append site-specific submit directives with repeated `--extra-classad`.

`--runtime-hours` emits `+RequestRuntime` in seconds. Confirm that this ClassAd is supported by the local pool.
