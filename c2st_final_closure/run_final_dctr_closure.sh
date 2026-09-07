#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
TARGET="${1:-inclusive}"

python -m c2st_final_closure.train_dctr_crossfit_closure \
    --folds 5 \
    --dctr-target "$TARGET"

python -m c2st_final_closure.validate_dctr_crossfit_closure \
    --dctr-target "$TARGET" \
    --vars mli_ll_pt mli_n_jet \
    --bins 60
