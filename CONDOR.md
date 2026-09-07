# HTCondor submission helper

The helper creates:

```text
condor_jobs/
└── <task>_<timestamp>/
    ├── run.sh
    ├── job.sub
    └── logs/
```

## Main C2ST training

```bash
python make_condor_submit.py \
    --task main \
    --repo-dir "$PWD" \
    --memory 24GB \
    --cpus 4 \
    --gpus 1 \
    --runtime-hours 8
```

## Final cross-fitted DCTR closure

One channel per job is recommended.  The historical inclusive target is the default:

```bash
python make_condor_submit.py \
    --task closure \
    --repo-dir "$PWD" \
    --channels 2mu \
    --folds 5 \
    --memory 32GB \
    --cpus 4 \
    --gpus 1 \
    --runtime-hours 24
```

To train the process-specific `DY vs (Data - non-DY MC)` target, forward the new option to the training script after `--`:

```bash
python make_condor_submit.py \
    --task closure \
    --repo-dir "$PWD" \
    --channels 2mu \
    --folds 5 \
    --memory 32GB \
    --cpus 4 \
    --gpus 1 \
    --runtime-hours 24 \
    --conda-base /data/dust/user/bonanomm/miniforge3 \
    -- \
    --dctr-target dy_only
```

Repeat for `2e` if desired.

Use `--submit` to immediately call `condor_submit` after generating the job files.

## Extra arguments

Arguments for the underlying training program can be forwarded after `--`.  This also works for options such as `--cap-quantile` or future closure-training switches.

## Site-specific directives

Raw HTCondor submit lines can be appended with repeated `--extra-classad`.

The helper assumes the repository and input data are visible on the worker via a shared filesystem, so it writes:

```text
should_transfer_files = NO
```

The worker wrapper records the hostname, Git revision, Python/TensorFlow versions, visible GPUs, and exact training command in the job log.
