"""Compare positive and signed MC on the same full selected sample.
Requires the latest plot_signed_physical.py with cache support.
No retraining or plot renormalization; preserves saved factor provenance.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import validate, write_json
from .plot_signed_physical import OBSERVABLES, load_signed_cached, match_rows, verify_factors, summary


def load_full(channel_dir, cache_dir, chunk_size=65536, refresh_cache=False):

    meta = json.loads((channel_dir/'inference/metadata.json').read_text())
    cfg = validate(meta['config'])
    channel = channel_dir.name
    if channel not in cfg['channels']:
        raise ValueError('Pass the channel directory, e.g. RUN/2mu')
    if cfg['compatibility'] or cfg['max_events_per_class'] is not None:
        raise ValueError('This workflow supports generic training without class truncation')
    corrected_processes = [p for p in cfg['base_processes'] if p not in cfg['subtract_processes']]
    if meta.get('corrected_processes', corrected_processes) != corrected_processes or meta.get('subtract_processes', cfg['subtract_processes']) != cfg['subtract_processes']:
        raise ValueError('Bundle/config correction scope mismatch')
    if cfg['regions'] is None or len(cfg['regions']) != 1:
        raise ValueError('This plotting workflow requires one training region')
    if meta['features'] != cfg['features']:
        raise ValueError('Bundle/config feature mismatch')
    region = cfg['regions'][0]
    print('Loading full target and signed base with saved training config', flush=True)
    # Extra plotting columns are not model inputs and do not change selections.
    # Keep the original training configuration intact for provenance checks.
    load_cfg = deepcopy(cfg)
    load_cfg['validation_vars'] = list(dict.fromkeys(
        cfg['validation_vars'] + [name for name in OBSERVABLES if name not in cfg['features']]))
    data = load_signed_cached(load_cfg, 'target', channel, cache_dir, refresh_cache)
    mc = load_signed_cached(load_cfg, 'base', channel, cache_dir, refresh_cache)
    for name in OBSERVABLES:
        if name not in mc or name not in data:
            raise ValueError(f'Required plotting column {name} was not loaded')
    if pd.Index(data.event_id).intersection(pd.Index(mc.event_id)).size:
        raise ValueError('Target/base input overlap')
    population = json.loads((channel_dir/'population.json').read_text())
    for side, full in [('target',data),('base',mc)]:
        saved = pd.read_parquet(channel_dir/f'{side}_rows.parquet')
        positions = match_rows(full, saved, side)
        old = population[side]
        if len(full) != old['full_rows'] or not np.isclose(full.weight_before.to_numpy(dtype=float).sum(), old['full_signed_sumw'], rtol=1e-6, atol=1e-5):
            raise ValueError(f'{side}: full population differs from training report')
        if side == 'base':
            base_rows, base_positions = saved, positions
    with np.load(channel_dir/'dctr_factors_base.npz', allow_pickle=False) as archive:
        saved_factor, folds, caps = (archive[n].copy() for n in ('dctr_factor','fold_id','fold_caps'))
        scope = base_rows.process.isin(corrected_processes).to_numpy()
        if not np.array_equal(archive['corrected'], scope) or not np.array_equal(base_rows.corrected, scope):
            raise ValueError('Saved correction scope disagrees with process configuration')
        if not np.array_equal(mc.iloc[base_positions].corrected.to_numpy(), scope):
            raise ValueError('Reloaded correction scope changed')
    with np.load(channel_dir/'splits.npz', allow_pickle=False) as splits:
        verify_factors(base_rows, saved_factor, folds, caps, splits, cfg['folds'])
    factor = np.ones(len(mc), dtype=np.float32)
    full_scope = mc.process.isin(corrected_processes).to_numpy()
    if not np.array_equal(mc.corrected.to_numpy(), full_scope):
        raise ValueError('Full correction scope disagrees with config')
    provenance = np.full(len(mc), -1, dtype=np.int16)
    provenance[full_scope] = cfg['folds']+1
    factor[base_positions], provenance[base_positions] = saved_factor, folds
    excluded = np.flatnonzero((mc.weight_before.to_numpy() <= 0) & full_scope)
    print(f'Reusing saved factors for {scope.sum():,} corrected classifier events; '
          f'evaluating {len(excluded):,} excluded corrected events; '
          f'keeping {np.count_nonzero(~full_scope):,} other events unchanged', flush=True)
    if len(excluded):
        from .apply import DCTRReweighter
        rw = DCTRReweighter(channel_dir/'inference')
        for start in range(0, len(excluded), chunk_size):
            indices = excluded[start:start+chunk_size]
            factor[indices] = rw.predict(mc.iloc[indices], processes=mc.iloc[indices].process.to_numpy())
        del rw
    if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
        raise ValueError('Invalid inferred factors')
    if np.any(factor[~full_scope] != 1):
        raise AssertionError('Non-corrected components were modified')
    return data, mc, factor, provenance, cfg


def make_comparison(data, mc, factor, provenance, cfg, output, channel,
                    dy_processes, ratio_range=(0.4, 2.0)):
    w = mc.weight_before.to_numpy(dtype=float)
    dw = data.weight_before.to_numpy(dtype=float)
    positive, negative = w > 0, w < 0
    dy = mc.process.isin(dy_processes).to_numpy()
    groups = [('all MC', np.ones(len(mc), bool)), ('DY', dy), ('non-DY', ~dy)]
    groups += [('process: '+label, mc.process.eq(label).to_numpy())
               for label in cfg['base_processes']]
    rows = []
    for label, mask in groups:
        for sign, sm in [('positive',positive), ('negative',negative),
                         ('zero',w == 0), ('signed',np.ones(len(mc),bool))]:
            selected = mask & sm
            rows.append(dict(group=label, sign=sign,
                             **summary(w[selected], factor[selected])))
    report = pd.DataFrame(rows).drop(columns='factor_quantiles')
    report.to_csv(output/'yields_by_process_and_sign.csv',index=False)
    print(report.loc[report.group.isin(['all MC','DY','non-DY']),
          ['group','sign','rows','before','after','signed_weighted_mean_factor']].to_string(index=False))
    print(f'Data yield (full selection): {dw.sum():.8g}')
    write_json(output/'comparison_metadata.json',dict(
        channel=channel, regions=cfg['regions'], normalization=cfg['normalization'],
        data_sumw=dw.sum(), dy_processes=dy_processes,
        corrected_processes=[p for p in cfg['base_processes'] if p not in cfg['subtract_processes']],
        population='Full selected population; common Data and MC rows for every curve',
        provenance='Saved cross-fit/final factors; excluded corrected events use final model',
        factor_sources={str(int(i)):summary(w[provenance==i],factor[provenance==i])
                        for i in np.unique(provenance)},
        errors='Histogram sumw2 only; no learned-factor uncertainty',ratio_range=ratio_range))

    def hist(x, weights, edges):
        return (np.histogram(x, edges, weights=weights)[0],
                np.histogram(x, edges, weights=weights**2)[0])

    for name, spec in OBSERVABLES.items():
        edges = np.asarray(spec['edges'])
        centers = (edges[:-1]+edges[1:])/2
        d, d2 = hist(data[name],dw,edges)
        fig, axes = plt.subplots(2,2,sharex=True,figsize=(13,8),
            gridspec_kw={'height_ratios':[3,1]},constrained_layout=True)
        binned = dict(low=edges[:-1],high=edges[1:],data=d,data_sumw2=d2)
        for col, (stage, weights) in enumerate([('before',w),('after',w*factor)]):
            ax, rat = axes[:,col]
            p,p2 = hist(mc.loc[positive,name],weights[positive],edges)
            n,n2 = hist(mc.loc[negative,name],weights[negative],edges)
            signed,signed2 = hist(mc[name],weights,edges)
            np.testing.assert_allclose(signed,p+n,rtol=1e-7,atol=1e-6)
            for key, values in [('positive',p),('negative',n),('signed',signed),
                                 ('positive_sumw2',p2),('negative_sumw2',n2),('signed_sumw2',signed2)]:
                binned[stage+'_'+key] = values
            ax.errorbar(centers,d,yerr=np.sqrt(d2),fmt='k.',label='Data')
            ax.stairs(p,edges,color='tab:blue',label='Positive MC')
            ax.stairs(-n,edges,color='tab:red',linestyle=':',label='Negative MC magnitude (subtracted)')
            ax.stairs(signed,edges,color='tab:orange',label='Signed MC = positive - magnitude')
            ax.axhline(0,color='gray',lw=.7)
            ax.set_title(stage.capitalize()+' DCTR')
            ax.set_ylabel('Weighted events / bin')
            ax.legend(fontsize=8)
            for h,h2,color,label in [(p,p2,'tab:blue','Data / positive MC'),
                                     (signed,signed2,'tab:orange','Data / signed MC')]:
                valid=h>0
                values=d[valid]/h[valid]
                err=np.sqrt(d2[valid]/h[valid]**2+d[valid]**2*h2[valid]/h[valid]**4)
                rat.errorbar(centers[valid],values,yerr=err,fmt='.',color=color,label=label)
                clipped=np.count_nonzero((values<ratio_range[0])|(values>ratio_range[1]))
                if clipped:
                    print(f'{name} {stage} {label}: {clipped} ratios outside displayed limits')
                if np.any(~valid):
                    print(f'{name} {stage} {label}: {np.count_nonzero(~valid)} bins have nonpositive denominator; ratio omitted')
            rat.axhline(1,color='gray',ls='--')
            rat.set_ylim(*ratio_range)
            rat.set_ylabel('Data / MC')
            rat.set_xlabel(spec['label'])
            rat.legend(fontsize=8)
        fig.suptitle(f"{channel}, {', '.join(cfg['regions'])} — same full selected sample\n"
                     f"Training normalization: {cfg['normalization']}; no plot renormalization")
        for extension in ['png','pdf']:
            fig.savefig(output/f'{name}_sign_comparison.{extension}',dpi=160)
        plt.close(fig)
        pd.DataFrame(binned).to_csv(output/f'{name}_bin_yields.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--channel-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,default=Path('c2st_cache/signed_physical'))
    p.add_argument('--refresh-cache',action='store_true')
    p.add_argument('--chunk-size',type=int,default=65536)
    p.add_argument('--ratio-range',type=float,nargs=2,default=(0.4,2.0))
    p.add_argument('--dy-processes',nargs='+',help='Exact DY labels; default: configured labels starting with DY')
    a=p.parse_args()
    if a.chunk_size<1 or not np.all(np.isfinite(a.ratio_range)) or a.ratio_range[0]>=a.ratio_range[1]:
        p.error('Require positive chunk size and finite, increasing ratio limits')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a fresh output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    data,mc,factor,provenance,cfg=load_full(a.channel_dir,a.cache_dir,a.chunk_size,a.refresh_cache)
    dy=a.dy_processes if a.dy_processes is not None else [n for n in cfg['base_processes'] if n.startswith('DY')]
    if not dy or set(dy)-set(cfg['base_processes']):
        p.error('Specify valid --dy-processes labels from the saved config')
    print('DY labels for yield reporting:',dy,flush=True)
    make_comparison(data,mc,factor,provenance,cfg,a.output,a.channel_dir.name,dy,a.ratio_range)
    print('Saved comparison to',a.output)


if __name__=='__main__':
    main()

