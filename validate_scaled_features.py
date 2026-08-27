import argparse
import numpy as np
import matplotlib.pyplot as plt

import c2st_config as cfg
from c2st_artifacts import load_metadata, load_scaler, load_test_fold, load_stage_arrays
from c2st_core import apply_scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channel', choices=cfg.CHANNELS, default='2e')
    ap.add_argument('--bins-continuous', type=int, default=60)
    ap.add_argument('--bins-discrete', type=int, default=9)
    args = ap.parse_args()
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    test = load_test_fold(cfg.ARTIFACT_DIR, args.channel, columns=['y'] + cfg.FEATURES)
    y = test['y'].to_numpy(dtype=np.uint8, copy=False)
    scaler = load_scaler(cfg.ARTIFACT_DIR, args.channel)
    x = apply_scaler(test, cfg.FEATURES, scaler)  # test fold only; DATASETS/full X is unnecessary
    before = load_stage_arrays(cfg.ARTIFACT_DIR, args.channel, 'before')
    after = load_stage_arrays(cfg.ARTIFACT_DIR, args.channel, 'after')
    wb = np.asarray(before['w_test']); wa = np.asarray(after['w_test'])
    is_data = y == 1; is_mc = y == 0

    for i, feature in enumerate(cfg.FEATURES):
        nb = args.bins_continuous if any(k in feature for k in ('pt', 'mbb', 'mass', 'ht', 'lt')) else args.bins_discrete
        bins = np.linspace(0, 1, nb + 1)
        data, _ = np.histogram(x[is_data, i], bins=bins, weights=wb[is_data])
        mc_before, _ = np.histogram(x[is_mc, i], bins=bins, weights=wb[is_mc])
        mc_after, _ = np.histogram(x[is_mc, i], bins=bins, weights=wa[is_mc])
        centers = 0.5 * (bins[:-1] + bins[1:])

        fig, (ax, ratio) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
        ax.step(centers, mc_before, where='mid', label='MC before')
        ax.step(centers, mc_after, where='mid', label='MC after')
        ax.plot(centers, data, 'ko', ms=3, label='Data')
        ax.set_yscale('log'); ax.legend(title=feature)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio.plot(centers, np.divide(mc_before, data, out=np.full_like(mc_before, np.nan, dtype=float), where=data > 0), 'o', label='before')
            ratio.plot(centers, np.divide(mc_after, data, out=np.full_like(mc_after, np.nan, dtype=float), where=data > 0), 'o', label='after')
        ratio.axhline(1.0, color='k', ls='--', lw=1); ratio.set_ylim(0.5, 1.5)
        ratio.set_xlabel(f'{feature} (scaled)'); ratio.set_ylabel('MC/Data')
        fig.tight_layout(); fig.savefig(cfg.PLOT_DIR / f'scaled_{args.channel}_{feature}.png', dpi=140)
        plt.close(fig)

if __name__ == '__main__':
    main()
