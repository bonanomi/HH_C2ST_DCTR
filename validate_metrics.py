from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import c2st_config as cfg
from c2st_artifacts import load_metadata, load_test_fold, load_stage_arrays
from c2st_core import weighted_bce


def main():
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for channel in cfg.CHANNELS:
        test = load_test_fold(cfg.ARTIFACT_DIR, channel, columns=['y'])
        y = test['y'].to_numpy(dtype=np.uint8, copy=False)
        del test

        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig_score, ax_score = plt.subplots(figsize=(7, 5))
        for stage in cfg.STAGES:
            arr = load_stage_arrays(cfg.ARTIFACT_DIR, channel, stage)
            p = np.asarray(arr['p_test'])
            w = np.asarray(arr['w_test'])
            auc = float(roc_auc_score(y, p, sample_weight=w))
            bce = weighted_bce(y, p, w)
            rows.append({'channel': channel, 'stage': stage, 'auc': auc, 'weighted_bce': bce, 'n_test': len(y)})

            fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
            ax.plot(fpr, tpr, label=f'{stage} (AUC={auc:.4f})')
            bins = np.linspace(0, 1, 61)
            for label, mask in [('Data', y == 1), ('MC', y == 0)]:
                ax_score.hist(p[mask], bins=bins, weights=w[mask], density=True,
                              histtype='step', label=f'{label} ({stage})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{channel}: ROC')
        ax.legend()
        fig.tight_layout()
        fig.savefig(cfg.PLOT_DIR / f'roc_{channel}.png', dpi=160)
        plt.close(fig)

        ax_score.set(xlabel='classifier output P(Data)', ylabel='density', title=f'{channel}: test scores')
        ax_score.legend()
        fig_score.tight_layout()
        fig_score.savefig(cfg.PLOT_DIR / f'scores_{channel}.png', dpi=160)
        plt.close(fig_score)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.ARTIFACT_DIR / 'validation_metrics.csv', index=False)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
