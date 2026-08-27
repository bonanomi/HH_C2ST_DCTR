import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import c2st_config as cfg
from c2st_artifacts import load_test_fold, load_stage_arrays
from validation_utils import quantile_edges


def compute_dctr_weights(y, p_before, raw_weight_uncorrected, cap_quantile=0.995, eps=1e-6):
    y = np.asarray(y, dtype=np.uint8)
    p = np.clip(np.asarray(p_before, dtype=np.float32), eps, 1 - eps)
    raw = np.asarray(raw_weight_uncorrected, dtype=np.float32)
    is_mc = y == 0

    # The before-model was trained with class-balanced sample weights, so odds p/(1-p)
    # estimates the Data/MC density ratio under that balanced-prior training convention.
    dctr_raw = np.ones(len(y), dtype=np.float32)
    dctr_raw[is_mc] = p[is_mc] / (1.0 - p[is_mc])
    dctr = dctr_raw.copy()

    cap_value = None
    n_capped = 0
    frac_weight_capped = 0.0
    if cap_quantile is not None and np.any(is_mc):
        cap_value = float(np.quantile(dctr_raw[is_mc], cap_quantile))
        capped = is_mc & (dctr_raw > cap_value)
        n_capped = int(capped.sum())
        uncapped_sum = float(np.sum(raw[is_mc] * dctr_raw[is_mc], dtype=np.float64))
        capped_contribution = float(np.sum(raw[capped] * dctr_raw[capped], dtype=np.float64))
        frac_weight_capped = capped_contribution / uncapped_sum if uncapped_sum > 0 else 0.0
        dctr[capped] = cap_value

    new_weight = raw.copy()
    new_weight[is_mc] *= dctr[is_mc]
    return {
        'dctr_weight': dctr,
        'dctr_weight_raw': dctr_raw,
        'new_mc_weight': new_weight,
        'cap_value': cap_value,
        'n_capped': n_capped,
        'frac_weight_capped': frac_weight_capped,
    }


def _normalize_hist_to_data(data_counts, mc_counts):
    """Return MC histogram normalized to the Data integral and the applied scale factor.

    The normalization is computed from exactly the bins displayed in the closure plot. This
    makes the comparison explicitly shape-only and avoids differences caused by events outside
    the plotted range or non-finite values.
    """
    data_sum = float(np.sum(data_counts, dtype=np.float64))
    mc_sum = float(np.sum(mc_counts, dtype=np.float64))
    if data_sum <= 0 or mc_sum <= 0:
        return mc_counts.astype(np.float64, copy=False), np.nan
    scale = data_sum / mc_sum
    return mc_counts * scale, scale


def plot_closure(channel, var, values, y, raw_before, raw_after, new_dctr, bins,
                 normalization='shape'):
    """Compare Data with three MC weighting prescriptions.

    Parameters
    ----------
    normalization : {'shape', 'physical'}
        'shape' (recommended for this diagnostic): independently normalize MC-before,
        MC-DY-corrected and MC-DCTR to the Data integral in the displayed bins. The ratios then
        show only shape differences between the weighting prescriptions.
        'physical': preserve the physical weighted yields and therefore test normalization and
        shape simultaneously.
    """
    values = np.asarray(values)
    y = np.asarray(y)
    finite = np.isfinite(values)
    in_range = finite & (values >= bins[0]) & (values <= bins[-1])
    is_data = (y == 1) & in_range
    is_mc = (y == 0) & in_range

    data, _ = np.histogram(values[is_data], bins=bins, weights=raw_before[is_data])
    mc_before, _ = np.histogram(values[is_mc], bins=bins, weights=raw_before[is_mc])
    mc_after, _ = np.histogram(values[is_mc], bins=bins, weights=raw_after[is_mc])
    mc_dctr, _ = np.histogram(values[is_mc], bins=bins, weights=new_dctr[is_mc])

    scales = {'before': 1.0, 'dy': 1.0, 'dctr': 1.0}
    if normalization == 'shape':
        mc_before, scales['before'] = _normalize_hist_to_data(data, mc_before)
        mc_after, scales['dy'] = _normalize_hist_to_data(data, mc_after)
        mc_dctr, scales['dctr'] = _normalize_hist_to_data(data, mc_dctr)
    elif normalization != 'physical':
        raise ValueError("normalization must be 'shape' or 'physical'")

    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, (ax, ratio) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
    )
    ax.step(centers, data, where='mid', label='Data')
    ax.step(centers, mc_before, where='mid', label='MC before')
    ax.step(centers, mc_after, where='mid', label='MC DY corrected')
    ax.step(centers, mc_dctr, where='mid', label='MC DCTR')
    ax.set_ylabel('weighted events')
    suffix = 'shape-normalized' if normalization == 'shape' else 'physical normalization'
    ax.set_title(f'{channel}: closure on {var} ({suffix})')
    ax.legend()

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio.step(
            centers,
            np.divide(mc_before, data, out=np.full_like(mc_before, np.nan, dtype=float), where=data > 0),
            where='mid', label='before',
        )
        ratio.step(
            centers,
            np.divide(mc_after, data, out=np.full_like(mc_after, np.nan, dtype=float), where=data > 0),
            where='mid', label='DY corr',
        )
        ratio.step(
            centers,
            np.divide(mc_dctr, data, out=np.full_like(mc_dctr, np.nan, dtype=float), where=data > 0),
            where='mid', label='DCTR',
        )
    ratio.axhline(1.0, color='k', ls='--', lw=1)
    ratio.set_ylim(0.5, 1.5)
    ratio.set_ylabel('MC/Data')
    ratio.set_xlabel(var)
    fig.tight_layout()
    return fig, scales



def plot_dctr_weight_distribution(channel, y, dctr_raw, dctr_capped, cap_value,
                                  n_bins=80, xmax_quantile=0.999):
    """Plot the MC-only DCTR multiplicative-weight distribution.

    The x-range is limited to a high quantile of the *uncapped* distribution for readability;
    the full distribution is still used for all numerical diagnostics and for determining the
    cap. Both uncapped and capped weights are shown so the effect of tail regularization is
    explicit.
    """
    is_mc = np.asarray(y) == 0
    raw = np.asarray(dctr_raw, dtype=np.float32)[is_mc]
    capped = np.asarray(dctr_capped, dtype=np.float32)[is_mc]

    finite = np.isfinite(raw) & np.isfinite(capped) & (raw >= 0) & (capped >= 0)
    raw = raw[finite]
    capped = capped[finite]
    if len(raw) == 0:
        raise ValueError(f'No finite MC DCTR weights available for {channel}')

    xmax = float(np.quantile(raw, xmax_quantile)) if xmax_quantile is not None else float(raw.max())
    xmax = max(xmax, np.finfo(np.float32).eps)
    bins = np.linspace(0.0, xmax, n_bins + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(raw, bins=bins, histtype='step', density=True, label='DCTR raw')
    ax.hist(capped, bins=bins, histtype='step', density=True, label='DCTR capped')
    if cap_value is not None and cap_value <= xmax:
        ax.axvline(cap_value, ls='--', lw=1.2, label=f'cap = {cap_value:.3g}')

    ax.set_yscale('log')
    ax.set_xlabel('DCTR multiplicative weight')
    ax.set_ylabel('normalized MC density')
    ax.set_title(f'{channel}: DCTR weight distribution')
    ax.legend()
    fig.tight_layout()
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vars', nargs='+', default=['mli_ll_pt'])
    ap.add_argument('--bins', type=int, default=20)
    ap.add_argument('--cap-quantile', type=float, default=0.995)
    ap.add_argument('--normalization', choices=['shape', 'physical'], default='shape',
                    help='shape: normalize each MC alternative to Data in the plotted bins; '
                         'physical: preserve raw physical yields')
    ap.add_argument('--plot-weights', action='store_true',
                    help='also save an MC-only DCTR raw/capped weight-distribution plot')
    ap.add_argument('--weight-bins', type=int, default=80)
    ap.add_argument('--weight-xmax-quantile', type=float, default=0.999,
                    help='upper quantile used only for the displayed DCTR-weight x-range')
    args = ap.parse_args()
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for channel in cfg.CHANNELS:
        cols = list(dict.fromkeys(['y', 'weight_uncorrected', 'weight'] + args.vars))
        test = load_test_fold(cfg.ARTIFACT_DIR, channel, columns=cols)
        y = test['y'].to_numpy(dtype=np.uint8, copy=False)
        before = load_stage_arrays(cfg.ARTIFACT_DIR, channel, 'before')
        res = compute_dctr_weights(
            y, np.asarray(before['p_test']), test['weight_uncorrected'].to_numpy(dtype=np.float32, copy=False),
            cap_quantile=args.cap_quantile,
        )
        np.savez(cfg.ARTIFACT_DIR / f'dctr_{channel}.npz',
                 dctr_weight=res['dctr_weight'], new_mc_weight=res['new_mc_weight'])
        rows.append({'channel': channel, 'cap_value': res['cap_value'], 'n_capped': res['n_capped'],
                     'frac_weight_capped': res['frac_weight_capped']})

        if args.plot_weights:
            fig = plot_dctr_weight_distribution(
                channel, y, res['dctr_weight_raw'], res['dctr_weight'], res['cap_value'],
                n_bins=args.weight_bins, xmax_quantile=args.weight_xmax_quantile,
            )
            fig.savefig(cfg.PLOT_DIR / f'dctr_weights_{channel}.png', dpi=160)
            plt.close(fig)

        raw_before = test['weight_uncorrected'].to_numpy(dtype=np.float32, copy=False)
        raw_after = test['weight'].to_numpy(dtype=np.float32, copy=False)
        for var in args.vars:
            values = test[var].to_numpy(dtype=np.float32, copy=False)
            bins = quantile_edges(values, args.bins)
            fig, scales = plot_closure(
                channel, var, values, y, raw_before, raw_after, res['new_mc_weight'], bins,
                normalization=args.normalization,
            )
            fig.savefig(cfg.PLOT_DIR / f'dctr_closure_{channel}_{var}.png', dpi=160)
            plt.close(fig)
            print(
                f'[{channel}, {var}] normalization={args.normalization}; '
                f"MC scales: before={scales['before']:.6g}, DY={scales['dy']:.6g}, DCTR={scales['dctr']:.6g}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(cfg.ARTIFACT_DIR / 'dctr_summary.csv', index=False)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
