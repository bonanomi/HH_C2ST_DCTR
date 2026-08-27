import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import c2st_config as cfg
from c2st_artifacts import load_test_fold, load_stage_arrays
from validation_utils import quantile_edges


def auc_in_bins(y, p, w, values, edges, min_events_per_class=50):
    idx = np.digitize(values, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        yb = y[mask]
        n_data = int(np.sum(yb == 1)); n_mc = int(np.sum(yb == 0))
        wb = w[mask]
        wmc = wb[yb == 0]
        neff = float(wmc.sum() ** 2 / np.sum(wmc.astype(np.float64) ** 2)) if len(wmc) and np.sum(wmc.astype(np.float64) ** 2) > 0 else np.nan
        auc = float(roc_auc_score(yb, p[mask], sample_weight=wb)) if n_data >= min_events_per_class and n_mc >= min_events_per_class else np.nan
        rows.append({'bin_lo': edges[b], 'bin_hi': edges[b+1], 'bin_center': 0.5*(edges[b]+edges[b+1]),
                     'n_data': n_data, 'n_mc': n_mc, 'mc_kish_neff': neff, 'auc': auc})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--var', default='mli_ll_pt')
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--min-events-per-class', type=int, default=50)
    args = ap.parse_args()
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for channel in cfg.CHANNELS:
        test = load_test_fold(cfg.ARTIFACT_DIR, channel, columns=['y', args.var])
        y = test['y'].to_numpy(dtype=np.uint8, copy=False)
        values = test[args.var].to_numpy(dtype=np.float32, copy=False)
        edges = quantile_edges(values, args.bins)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        merged = None
        for stage in cfg.STAGES:
            arr = load_stage_arrays(cfg.ARTIFACT_DIR, channel, stage)
            df = auc_in_bins(y, np.asarray(arr['p_test']), np.asarray(arr['w_test']), values, edges, args.min_events_per_class)
            df.to_csv(cfg.ARTIFACT_DIR / f'auc_bins_{channel}_{stage}_{args.var}.csv', index=False)
            ax.plot(df['bin_center'], df['auc'], 'o-', label=stage)
            merged = df[['bin_lo','bin_hi','n_data','n_mc','mc_kish_neff']].copy() if merged is None else merged
            merged[f'auc_{stage}'] = df['auc']
        ax.axhline(0.5, color='k', ls='--', lw=1)
        ax.set(xlabel=args.var, ylabel='weighted AUC within bin', title=f'{channel}: AUC vs {args.var}')
        ax.legend(); fig.tight_layout()
        fig.savefig(cfg.PLOT_DIR / f'auc_bins_{channel}_{args.var}.png', dpi=160)
        plt.close(fig)
        print(f'\n[{channel}]\n{merged.to_string(index=False)}')

if __name__ == '__main__':
    main()
