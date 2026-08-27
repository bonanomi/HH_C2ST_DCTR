import argparse
import json
import numpy as np
import pandas as pd

import c2st_config as cfg
from c2st_artifacts import load_test_fold, load_stage_arrays
from validation_utils import paired_bootstrap_delta_auc, fixed_model_permutation_pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bootstrap-resamples', type=int, default=50)
    ap.add_argument('--permutation-resamples', type=int, default=100)
    ap.add_argument('--subsample', type=int, default=0,
                    help='0 means full test set; use e.g. 1000000 for a faster development run')
    args = ap.parse_args()
    subsample = args.subsample or None
    rows = []

    for channel in cfg.CHANNELS:
        y = load_test_fold(cfg.ARTIFACT_DIR, channel, columns=['y'])['y'].to_numpy(dtype=np.uint8, copy=False)
        before = load_stage_arrays(cfg.ARTIFACT_DIR, channel, 'before')
        after = load_stage_arrays(cfg.ARTIFACT_DIR, channel, 'after')

        boot = paired_bootstrap_delta_auc(
            y,
            np.asarray(before['p_test']), np.asarray(before['w_test']),
            np.asarray(after['p_test']), np.asarray(after['w_test']),
            n_resamples=args.bootstrap_resamples,
            random_state=cfg.RANDOM_STATE,
            subsample=subsample,
        )
        perm = fixed_model_permutation_pvalue(
            y, np.asarray(after['p_test']), np.asarray(after['w_test']),
            n_resamples=args.permutation_resamples,
            random_state=cfg.RANDOM_STATE,
            subsample=subsample,
        )

        np.savez(
            cfg.ARTIFACT_DIR / f'significance_distributions_{channel}.npz',
            bootstrap_delta_auc=boot['bootstrap_distribution'],
            permutation_auc_after=perm['null_distribution'],
        )
        rows.append({
            'channel': channel,
            'subsample': subsample if subsample is not None else len(y),
            'auc_before': boot['auc_before'],
            'auc_after': boot['auc_after'],
            'delta_auc': boot['delta_auc_observed'],
            'delta_auc_ci_low': boot['ci_low'],
            'delta_auc_ci_high': boot['ci_high'],
            'delta_excludes_zero': boot['excludes_zero'],
            'fixed_model_perm_p_after': perm['p_value'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(cfg.ARTIFACT_DIR / 'significance_summary.csv', index=False)
    print(df.to_string(index=False))
    print('\nNote: the permutation check is the fixed-model diagnostic null, not a full retrained C2ST permutation test.')

if __name__ == '__main__':
    main()
