#!/usr/bin/env bash
set -euo pipefail
python validate_metrics.py
python validate_auc_bins.py --var mli_ll_pt
python validate_significance.py --bootstrap-resamples 50 --permutation-resamples 100
python validate_dctr.py --vars mli_ll_pt
python validate_scaled_features.py --channel 2e
python validate_model_reload.py --channel 2mu --stage after
