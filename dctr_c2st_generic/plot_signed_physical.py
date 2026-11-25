"""Plot Data/MC24 pt_ll, njet and mll for the full signed training selection.

Install in dctr_c2st_generic/ and run from the repository root:
python -m dctr_c2st_generic.plot_signed_physical --channel-dir RUN/2mu --output NEW_DIR

Uses saved out-of-fold/final factors for classifier events. Only excluded
nonpositive-weight events in the corrected scope are inferred with the final bundle.
Inclusive and subtraction (including DY-only) scopes are read from metadata.
Subtracted components retain their physical weights and factor one. No shape/yield
renormalization is applied. The saved physical weight definition is preserved.
Full-population cross-fit diagnostics are not an independent outer-test C2ST.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import hashlib
import os
import tempfile
import time
import joblib
import numpy as np
import pandas as pd
from .config import validate, write_json, jsonable
from .loading import load_side, branch_files, producer_path
from .plot_year_transfer import draw


OBSERVABLES = {
    'mli_ll_pt':dict(edges=np.linspace(0,200,41),label=r'$p_T^{\ell\ell}$ [GeV]',log=False,flow='drop'),
    'mli_n_jet':dict(edges=np.arange(-.5,12.5,1),label='Jet multiplicity',density_per_unit=False,log=False,flow='drop'),
    'mli_mll':dict(edges=np.linspace(0,200,41),label=r'$m_{\ell\ell}$ [GeV]',log=False,flow='drop'),
}


def signed_input_manifest(cfg, side):
    """Stat all relevant inputs, including per-process weight/source overrides."""
    paths, directories, branches = set(), {}, {}
    for entry in cfg[f'{side}_processes'].values():
        source_name = entry.get('source', cfg[f'{side}_source'])
        source = cfg['sources'][source_name]
        producers = {'reduction', source.get('feature_producer', 'dl_ml_inputs')}
        specs = [entry.get('weights', cfg[f'{side}_weights']),
                 *entry.get('reference_weights', {}).values()]
        for spec in specs:
            for field in spec.get('fields', []):
                producers.add(field.get('producer', 'event_weights')
                              if isinstance(field, dict) else 'event_weights')
        for dataset in entry['datasets']:
            if source.get('format', 'cf') == 'cf':
                for producer in sorted(producers):
                    key = (source_name, producer)
                    if key not in directories:
                        directories[key] = producer_path(source, producer)
                    directory = directories[key]
                    if directory is None:
                        continue
                    key = (str(directory), dataset, source.get('shift', 'nominal'))
                    if key not in branches:
                        branches[key] = branch_files(*key)
                    paths.update(branches[key].values())
            else:
                paths.update(Path(source['root']).glob(dataset))
    result = []
    for path in sorted(paths):
        stat = path.stat()
        result.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return result


def load_signed_cached(cfg, side, channel, cache_dir=None, refresh=False):
    """Cache selected tables only; always recheck training provenance downstream.

    Joblib caches must be owned/trusted by the user. Input paths, sizes and mtimes
    invalidate entries; use refresh if inputs were replaced preserving metadata.
    """
    start = time.monotonic()
    if cache_dir is None:
        print(f'Loading {side}/{channel} (cache disabled)', flush=True)
        result = load_side(cfg, side, channel)
    else:
        print(f'Checking input manifest for {side}/{channel}', flush=True)
        manifest = signed_input_manifest(cfg, side)
        payload = dict(version=1, config=cfg, side=side, channel=channel,
                       inputs=manifest, code={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('loading.py', 'config.py')})
        key = hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True).encode()).hexdigest()
        folder = Path(cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{side}_{key}.joblib'
        if path.exists() and not refresh:
            print(f'Cache HIT {side}/{channel}: {path}', flush=True)
            result = joblib.load(path)
        else:
            print(f'Cache MISS {side}/{channel}; reading input parquet files', flush=True)
            result = load_side(cfg, side, channel)
            if signed_input_manifest(cfg, side) != manifest:
                raise RuntimeError('Inputs changed during loading; rerun with stable inputs')
            fd, temporary = tempfile.mkstemp(prefix=key+'.', suffix='.tmp', dir=folder)
            os.close(fd)
            try:
                joblib.dump(result, temporary, compress=0)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            print(f'Cache saved: {path}', flush=True)
    print(f'Loaded {side}/{channel}: {len(result):,} rows in {time.monotonic()-start:.1f}s', flush=True)
    return result


def match_rows(full, saved, side):
    if full.event_id.duplicated().any() or saved.event_id.duplicated().any():
        raise ValueError(f'{side}: duplicate event IDs')
    index = pd.Index(full.event_id)
    positions = index.get_indexer(saved.event_id)
    if np.any(positions < 0):
        raise ValueError(f'{side}: saved events missing from reload; check paths/production')
    matched = full.iloc[positions]
    for name in ('process', 'source', 'dataset', 'branch', 'row'):
        if not np.array_equal(matched[name].to_numpy(), saved[name].to_numpy()):
            raise ValueError(f'{side}: changed {name}')
    if not np.allclose(matched.weight_before, saved.weight_before, rtol=1e-6, atol=1e-8):
        raise ValueError(f'{side}: changed physical weights')
    if not np.all(saved.weight_before.to_numpy() > 0):
        raise ValueError(f'{side}: unexpected nonpositive classifier weights')
    if np.count_nonzero(full.weight_before.to_numpy() > 0) != len(saved):
        raise ValueError(f'{side}: positive population differs; no silent inference fallback')
    return positions


def verify_factors(saved, factors, folds, caps, splits, k):
    n = len(saved)
    if factors.shape != (n,) or folds.shape != (n,) or caps.shape != (k+1,):
        raise ValueError('Factor archive dimensions do not match saved rows/config')
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError('Invalid saved factors')
    corrected = saved['corrected'].to_numpy(dtype=bool) if 'corrected' in saved else np.ones(n, bool)
    if np.any(factors[~corrected] != 1) or np.any(folds[~corrected] != -1):
        raise ValueError('Subtracted components must have factor 1 and fold_id -1')
    expected = np.full(n, -1, dtype=int)
    for j in range(k+1):
        name = f'fold_{j}' if j < k else 'final'
        hold = np.asarray(splits[f'{name}_hold_base'])
        if np.any(hold < 0) or np.any(hold >= n) or len(np.unique(hold)) != len(hold):
            raise ValueError('Invalid held-out indices')
        if np.any(~corrected[hold]):
            raise ValueError('Correction assigned to a subtracted component')
        if np.any(expected[hold] != -1):
            raise ValueError('Overlapping held-out assignments')
        for subset in ('train_base', 'val_base', 'train_subtract', 'val_subtract'):
            if np.intersect1d(hold, splits.get(f'{name}_{subset}', [])).size:
                raise ValueError('Held-out events overlap model training/validation')
        expected[hold] = j
        if np.isfinite(caps[j]) and np.any(factors[hold] > caps[j] * (1+1e-6)):
            raise ValueError('Saved factors exceed their saved cap')
    if np.any(expected[corrected] < 0) or not np.array_equal(expected, folds):
        raise ValueError('Incomplete/inconsistent cross-fit provenance')
    if not np.array_equal(np.sort(splits['final_hold_base']), np.sort(splits['base_test'][corrected[splits['base_test']]])):
        raise ValueError('Final-model assignments differ from outer test')
    for j in range(k):
        if np.intersect1d(splits[f'fold_{j}_hold_base'], splits['base_test']).size:
            raise ValueError('Cross-fit holdout overlaps outer test')


def summary(weights, factors):
    w, f = np.asarray(weights, float), np.asarray(factors, float)
    before, after = w.sum(), np.sum(w*f)
    return dict(rows=len(w), before=before, after=after,
        before_sumw2=np.sum(w*w), after_sumw2=np.sum((w*f)**2),
        signed_weighted_mean_factor=after/before if before != 0 else None,
        absolute_weighted_mean_factor=np.sum(np.abs(w)*f)/np.abs(w).sum() if np.abs(w).sum() else None,
        factor_quantiles=dict(zip(['0','.01','.5','.99','1'],np.quantile(f,[0,.01,.5,.99,1]))) if len(f) else {})


def run(channel_dir, output, chunk_size=65536, save_events=False, cache_dir=None, refresh_cache=False):
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
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
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
    w = mc.weight_before.to_numpy(dtype=float)
    report = {'population':'full signed selected population; cross-fit/final factors',
              'correction_scope':corrected_processes, 'subtract_processes':cfg['subtract_processes'],
              'corrected_mc':summary(w[full_scope],factor[full_scope]),
              'unchanged_mc':summary(w[~full_scope],factor[~full_scope]),
              'training_normalization':cfg['normalization'], 'training_population':population,
              'data_sumw':data.weight_before.to_numpy(dtype=float).sum(),
              'mc':summary(w,factor), 'processes':{}, 'signs':{}, 'factor_sources':{}}
    for label in cfg['base_processes']:
        mask = mc.process.to_numpy() == label
        report['processes'][label] = summary(w[mask], factor[mask])
    for label,mask in [('positive',w>0),('negative',w<0),('zero',w==0)]:
        report['signs'][label] = summary(w[mask], factor[mask])
    report['factor_sources']['unchanged'] = summary(w[~full_scope],factor[~full_scope])
    report['factor_sources']['unchanged']['source_id'] = -1
    report['loss_stability'] = {}
    for name in [*[f'fold_{j}' for j in range(cfg['folds'])], 'final']:
        path = channel_dir / f'dctr_{name}_history.json'
        if path.exists():
            history = json.loads(path.read_text())
            report['loss_stability'][name] = {key:value for key,value in history.items() if key != 'history'}
        else:
            report['loss_stability'][name] = {'available':False}
    for j in range(cfg['folds']+2):
        mask = provenance == j
        label = f'fold_{j}' if j < cfg['folds'] else ('outer_test_final' if j == cfg['folds'] else 'excluded_final')
        cap = caps[j] if j <= cfg['folds'] else meta['cap']
        item = summary(w[mask], factor[mask])
        item['source_id'] = j
        item['cap'] = cap
        item['fraction_at_cap'] = float(np.mean(np.isclose(factor[mask],cap,rtol=2e-6,atol=1e-8))) if cap is not None and np.isfinite(cap) and mask.any() else None
        report['factor_sources'][label] = item
    for stage in ('before','after'):
        total = report['mc'][stage]
        report[f'data_over_mc_{stage}'] = report['data_sumw']/total if total != 0 else None
    write_json(output/'signed_diagnostics.json',report)
    write_json(output/'training_config.json',cfg)
    data = data.rename(columns={'weight_before':'weight'})
    mc = mc.rename(columns={'weight_before':'weight'})
    if save_events:
        audit = mc[['event_id','process','weight']].copy()
        audit['dctr_factor'],audit['weight_dctr'],audit['factor_source_id'] = factor,w*factor,provenance
        audit.to_parquet(output/'signed_mc_factors.parquet',index=False)
    plot_cfg = dict(years={'2024':dict(mc_processes=cfg['base_processes'],luminosity_fb=None)},
                    formats=['png','pdf'],cms_label='Work in progress',
                    channel_labels={channel:f'{channel} — full signed sample'},
                    region_labels={region:region},colors={'DY (mumu)':'#e8a01a','DY (ee)':'#ffad05',
                    'DY (tautau)':'#a86f5b','ttbar DL':'#bf2100','ttbb':'#ed6900'})
    for name,spec in OBSERVABLES.items():
        for stage,f in [('nominal',np.ones(len(mc),np.float32)),('dctr',factor)]:
            draw(data,mc,f,name,spec,plot_cfg,'2024',channel,region,stage,output)
    print(json.dumps({key:report[key] for key in ('data_sumw','data_over_mc_before','data_over_mc_after')},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--channel-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=65536)
    p.add_argument('--save-events',action='store_true')
    p.add_argument('--cache-dir',type=Path,default=Path('c2st_cache/signed_physical'),
                   help='Selected signed-table cache; use only a trusted directory')
    p.add_argument('--no-cache',action='store_true',help='Read original inputs without caching')
    p.add_argument('--refresh-cache',action='store_true',help='Rebuild selected-table caches')
    a = p.parse_args()
    if a.chunk_size < 1: p.error('--chunk-size must be positive')
    run(a.channel_dir,a.output,a.chunk_size,a.save_events,
        None if a.no_cache else a.cache_dir,a.refresh_cache)

if __name__ == '__main__':
    main()

