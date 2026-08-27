#!/usr/bin/env bash
set -euo pipefail

python train_dctr_crossfit_closure.py --folds 5
python validate_dctr_crossfit_closure.py --vars mli_ll_pt mli_n_jet --bins 60
