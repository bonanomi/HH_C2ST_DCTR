"""Physical Data/MC comparisons for 2024, 2025, and DCTR-reweighted MC2025.

Run: python -m dctr_c2st_generic.plot_year_transfer --config plot_config.py --output plots
No classifier filtering, class balancing, shape normalization or retraining is done.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import runpy
import warnings
import numpy as np
from .config import process_entries, safe_name, write_json
from .loading import load_side


def load_population(config, year, channel, region, features):
    """Use the generic loader with explicit physical weight definitions."""
    yc = config['years'][year]
    processes = {}
    for side in ('data', 'mc'):
        processes[side] = process_entries(yc[f'{side}_processes'])
        for entry in processes[side].values():
            if 'weights' in entry or 'reference_weights' in entry:
                raise ValueError('Set plot weights using data_weights/mc_weights/dy_weight, not process overrides')
            entry['weights'] = deepcopy(yc[f'{side}_weights'])
            if side == 'mc' and entry.get('is_dy', False) and yc.get('dy_weight'):
                if entry['weights']['mode'] != 'product':
                    raise ValueError('DY correction requires product weights')
                entry['weights']['fields'].append(deepcopy(yc['dy_weight']))
    source = deepcopy(yc['source'])
    cfg = dict(sources={year: source}, target_source=year, base_source=year,
               target_processes=processes['data'], base_processes=processes['mc'],
               target_weights=yc['data_weights'], base_weights=yc['mc_weights'],
               features=list(features), validation_vars=[], selections=config.get('selections', {}),
               channels=config['channels'], channel_ids=config['channel_ids'],
               regions=[region], region_ids=config['region_ids'],
               subtract_processes=[], reference_stages=[])
    data = load_side(cfg, 'target', channel)
    mc = load_side(cfg, 'base', channel)
    if set(data.event_id) & set(mc.event_id):
        raise ValueError('Data and MC input rows overlap')
    # weight_before is only the generic loader's internal name. Here the spec
    # explicitly constructs the analysis weight, INCLUDING official DY correction.
    return (data.rename(columns={'weight_before': 'weight'}),
            mc.rename(columns={'weight_before': 'weight'}))


def histogram(values, weights, edges, flow='drop'):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    edges = np.asarray(edges, float)
    if len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError('Histogram edges must be finite and strictly increasing')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError('Nonfinite histogram inputs')
    if flow == 'fold':
        values = np.clip(values, edges[0], np.nextafter(edges[-1], -np.inf))
    elif flow != 'drop':
        raise ValueError('flow must be drop or fold')
    return (np.histogram(values, edges, weights=weights)[0],
            np.histogram(values, edges, weights=weights * weights)[0])


def draw(data, mc, factor, observable, spec, config, year, channel, region, stage, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    edges = np.asarray(spec['edges'], float)
    centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    flow = spec.get('flow', 'drop')
    dh, dv = histogram(data[observable], data.weight, edges, flow)
    weights = mc.weight.to_numpy(dtype=float) * factor
    components = {}
    for label in config['years'][year]['mc_processes']:
        mask = mc.process.to_numpy() == label
        components[label] = histogram(mc.loc[mask, observable], weights[mask], edges, flow)
    mh = sum((h for h, v in components.values()), np.zeros(len(centers)))
    mv = sum((v for h, v in components.values()), np.zeros(len(centers)))
    valid = mh > 0
    ratio = np.divide(dh, mh, out=np.full_like(dh, np.nan), where=valid)
    ratio_err = np.divide(np.sqrt(dv), mh, out=np.full_like(dh, np.nan), where=valid)
    relative = np.divide(np.sqrt(mv), mh, out=np.full_like(mh, np.nan), where=valid)
    scale = 1 / widths if spec.get('density_per_unit', True) else np.ones_like(widths)
    style = {'font.size': 15, 'axes.labelsize': 19, 'axes.linewidth': 1.4,
             'xtick.direction': 'in', 'ytick.direction': 'in', 'font.family': 'sans-serif'}
    with plt.rc_context(style):
        fig, (ax, rat) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={'height_ratios': [3.5, 1], 'hspace': .04})
        palette = ['#929fa0', '#ed6900', '#a86f5b', '#ffad05', '#bf2100', '#8ed7dc', '#802bb0']
        positive, negative = np.zeros_like(mh), np.zeros_like(mh)
        handles = []
        for i, (label, (h, v)) in enumerate(components.items()):
            color = config.get('colors', {}).get(label, palette[i % len(palette)])
            shown = h * scale
            bottom = np.where(shown >= 0, positive, negative)
            ax.bar(edges[:-1], shown, widths, align='edge', bottom=bottom, color=color, linewidth=0)
            positive += np.maximum(shown, 0)
            negative += np.minimum(shown, 0)
            handles.append(Patch(facecolor=color, label=label))
        def band(axis, lower, upper):
            axis.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]],
                              step='post', facecolor='none', edgecolor='black', hatch='////', linewidth=0)
        band(ax, (mh - np.sqrt(mv)) * scale, (mh + np.sqrt(mv)) * scale)
        points = ax.errorbar(centers, dh * scale, yerr=np.sqrt(dv) * scale, xerr=widths/2,
                            fmt='o', color='black', markersize=4, linewidth=1, label='Data', zorder=5)
        uncertainty = Patch(facecolor='none', edgecolor='black', hatch='////', label='MC stat. unc.')
        ax.legend(handles=[points, *reversed(handles), uncertainty], loc='upper right',
                  ncol=2, frameon=False, fontsize=11)
        negative_bins = bool(np.any(negative < 0))
        if spec.get('log', True) and not negative_bins:
            ax.set_yscale('log')
            nonzero = np.r_[dh[dh > 0] * scale[dh > 0], mh[mh > 0] * scale[mh > 0]]
            low = max(np.min(nonzero) * .4, .01) if len(nonzero) else .1
            high = max(np.max((dh + np.sqrt(dv)) * scale), np.max(positive), low * 10)
            ax.set_ylim(low, high * 100)
        else:
            if negative_bins and spec.get('log', True):
                warnings.warn('Negative component bins: using linear axis to preserve signed predictions')
            ax.set_ylim(min(float(negative.min()) * 1.3, 0), max(float(positive.max()), float((dh*scale).max()), 1) * 1.6)
        ax.set_ylabel(spec.get('ylabel', 'Entries / GeV' if spec.get('density_per_unit', True) else 'Entries'))
        ax.text(0, 1.018, 'CMS', transform=ax.transAxes, fontsize=27, fontweight='bold')
        ax.text(.17, 1.024, config.get('cms_label', 'Work in progress'), transform=ax.transAxes, fontsize=13, style='italic')
        lumi = config['years'][year].get('luminosity_fb')
        ax.text(1, 1.025, (f'{lumi:g} fb$^{{-1}}$ (13.6 TeV)' if lumi is not None else f'{year} (13.6 TeV)'),
                transform=ax.transAxes, ha='right', fontsize=14)
        title = 'After DCTR correction' if stage == 'dctr' else 'Before DCTR correction'
        region_label = config.get('region_labels', {}).get(region, region)
        channel_label = config.get('channel_labels', {}).get(channel, channel)
        ax.text(.04, .94, f'{title}\n{year}, {region_label}\n{channel_label}',
                transform=ax.transAxes, va='top', fontsize=14)
        band(rat, 1-relative, 1+relative)
        rat.errorbar(centers[valid], ratio[valid], yerr=ratio_err[valid], xerr=widths[valid]/2,
                     fmt='o', color='black', markersize=4, linewidth=1)
        rat.axhline(1, color='gray', linestyle='--')
        rat.set_ylim(.8, 1.2)
        rat.set_yticks([.8, .9, 1, 1.1, 1.2])
        rat.set_ylabel('Data / MC')
        rat.set_xlabel(spec.get('label', observable), loc='right')
        rat.set_xlim(edges[0], edges[-1])
        for a in (ax, rat):
            a.minorticks_on()
            a.tick_params(which='both', top=True, right=True)
            a.tick_params(which='major', length=8)
            a.tick_params(which='minor', length=4)
        fig.subplots_adjust(left=.14, right=.97, top=.93, bottom=.10)
        stem = output / f'{safe_name(observable)}_{year}_{stage}'
        for extension in config.get('formats', ['png']):
            if extension not in ('png', 'pdf', 'svg'):
                raise ValueError('formats must be png, pdf or svg')
            fig.savefig(stem.with_suffix('.' + extension), dpi=160)
        plt.close(fig)
    write_json(str(stem) + '.json', dict(edges=edges, data=dh, data_sumw2=dv, mc=mh,
        mc_sumw2=mv, components={k: {'sumw': h, 'sumw2': v} for k,(h,v) in components.items()},
        ratio=ratio, ratio_data_error=ratio_err, ratio_limits=[.8, 1.2],
        undefined_ratio_bins=np.flatnonzero(~valid),
        outside_ratio_range_bins=np.flatnonzero(valid & ((ratio < .8) | (ratio > 1.2))),
        flow=flow, negative_component_bins=negative_bins))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = runpy.run_path(str(args.config.resolve()))['PLOT_CONFIG']
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'plot_config.json', config)
    from .apply import DCTRReweighter
    for channel in config['channels']:
        safe_name(channel)
        rw = DCTRReweighter(config['bundles'][channel])
        if rw.metadata.get('subtract_processes'):
            raise ValueError('This inclusive transfer script requires an inclusive Data/Data bundle')
        features = list(dict.fromkeys(rw.metadata['features'] + list(config['observables'])))
        for region in config['regions']:
            folder = args.output / channel / safe_name(region)
            folder.mkdir(parents=True)
            diagnostics = {}
            for year in ('2024', '2025'):
                print(f'Loading {year}, {channel}, {region}', flush=True)
                data, mc = load_population(config, year, channel, region, features)
                factor = np.ones(len(mc), dtype=float)
                diagnostics[year] = {'data_sumw': float(data.weight.sum()), 'mc_sumw': float(mc.weight.sum()),
                                     'data_rows': len(data), 'mc_rows': len(mc)}
                for name, spec in config['observables'].items():
                    draw(data, mc, factor, name, spec, config, year, channel, region, 'nominal', folder)
                if year == '2025':
                    # Bounded inference chunks avoid converting the full sample to a GPU tensor.
                    chunk = int(config.get('inference_chunk_size', 65536))
                    if chunk < 1:
                        raise ValueError('inference_chunk_size must be positive')
                    for start in range(0, len(mc), chunk):
                        factor[start:start+chunk] = rw.predict(mc.iloc[start:start+chunk])
                    if not np.all(np.isfinite(factor)) or np.any(factor < 0):
                        raise ValueError('Invalid DCTR factors')
                    cap = rw.metadata.get('cap')
                    weights = mc.weight.to_numpy(dtype=float)
                    per_process = {}
                    for label in config['years'][year]['mc_processes']:
                        mask = mc.process.to_numpy() == label
                        before, after = weights[mask].sum(), (weights[mask] * factor[mask]).sum()
                        per_process[label] = dict(before=before, after=after,
                            signed_weighted_mean_factor=after/before if before != 0 else None)
                    diagnostics['dctr'] = dict(bundle=str(rw.bundle), metadata=rw.metadata,
                        quantiles=dict(zip(['0', '.01', '.5', '.9', '.99', '1'], np.quantile(factor, [0,.01,.5,.9,.99,1]))),
                        fraction_at_cap=float(np.mean(factor >= np.float32(cap))) if cap is not None else 0.,
                        processes=per_process, mc_sumw_after=float(np.sum(weights*factor)),
                        signed_weighted_mean_factor=float(np.sum(weights*factor)/weights.sum()) if weights.sum() != 0 else None)
                    for name, spec in config['observables'].items():
                        draw(data, mc, factor, name, spec, config, year, channel, region, 'dctr', folder)
                del data, mc
            write_json(folder / 'yields_and_factors.json', diagnostics)


if __name__ == '__main__':
    main()
