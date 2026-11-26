"""Final-model signed DY-only validation maps; never uses outer-test events."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import validate
from .numerics import odds
from .training import transform
from .plot_signed_physical import load_signed_cached, match_rows


def divide(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)


def run(root, output, cache):
    import tensorflow as tf
    meta = json.loads((root/'inference/metadata.json').read_text())
    cfg = validate(meta['config'])
    if cfg.get('dctr_objective') != 'signed_mse' or not cfg['subtract_processes']:
        raise ValueError('This diagnostic requires signed-MSE DY-only training')
    if cfg['max_events_per_class'] is not None:
        raise ValueError('Truncated populations are unsupported')
    corrected_names = set(cfg['base_processes']) - set(cfg['subtract_processes'])
    if not corrected_names or any(not cfg['base_processes'][n].get('is_dy', False) for n in corrected_names):
        raise ValueError('Expected explicit is_dy flags on every corrected process')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(root/'inference/model.keras', compile=False)
    scaler = joblib.load(root/'inference/scaler.joblib')
    cap = json.loads((root/'dctr_final_diagnostics.json').read_text())['cap']
    features = meta['features']
    if features != cfg['features']:
        raise ValueError('Feature order mismatch')
    load_cfg = dict(cfg, validation_vars=list(dict.fromkeys(cfg['validation_vars'] + ['mli_ll_pt','mli_n_jet'])))
    xe = np.array([0,10,20,30,50,75,100,150,200,400,1000.], float)
    ye = np.arange(.5, 12., 1.)
    shape = (len(xe)-1,len(ye)-1)
    names = ['data','background','dy_before','dy_after','dy_raw','dy_sumw2_before','dy_sumw2_after',
             'data_sumw2','background_sumw2','dy_rows','capped_rows','dy_absw','capped_absw']
    h = {n:np.zeros(shape) for n in names}; audit={}
    with np.load(root/'splits.npz', allow_pickle=False) as splits:
        for side, roles in [('target',['target']),('base',['subtract','base'])]:
            frame = load_signed_cached(load_cfg, side, root.name, cache)
            saved = pd.read_parquet(root/f'{side}_rows.parquet')
            pos = match_rows(frame, saved, side, signed=True)
            for role in roles:
                tr = splits[f'final_train_{role}']; va = splits[f'final_val_{role}']; te = splits[f'final_hold_{role}']
                allidx = np.concatenate([tr,va,te])
                if len(np.unique(allidx)) != len(allidx) or np.intersect1d(va, splits[f'{side}_test']).size:
                    raise ValueError(f'Overlapping splits: {role}')
                if role=='target': expected=np.arange(len(saved))
                else:
                    corr=saved['corrected'].to_numpy(bool)
                    expected=np.flatnonzero(corr if role=='base' else ~corr)
                if not np.array_equal(np.sort(allidx),expected):
                    raise ValueError(f'Incomplete role partition: {role}')
                if not len(va): raise ValueError(f'Empty validation population: {role}')
                fraction = len(va)/len(allidx)
                audit[role]={'validation_rows':len(va),'population_rows':len(allidx),'validation_fraction':fraction,
                             'outside_map_rows':0}
                for start in range(0,len(va),131072):
                    rows=frame.iloc[pos[va[start:start+131072]]]
                    x=rows.mli_ll_pt.to_numpy(float); y=rows.mli_n_jet.to_numpy(float)
                    if not np.all(np.isfinite(x)&np.isfinite(y)):
                        raise ValueError('Nonfinite mapping features')
                    audit[role]['outside_map_rows']+=int(np.sum((x<xe[0])|(x>xe[-1])|(y<ye[0])|(y>ye[-1])))
                    w=rows.weight_before.to_numpy(float)/fraction
                    if not np.all(np.isfinite(w)): raise ValueError('Nonfinite weights')
                    def add(name, weights=None):
                        h[name][:] += np.histogram2d(x,y,bins=(xe,ye),weights=weights)[0]
                    if role!='base':
                        key='data' if role=='target' else 'background'
                        add(key,w);add(key+'_sumw2',w*w)
                    else:
                        p=model.predict(transform(rows,scaler,features),batch_size=meta['batch_size'],verbose=0).reshape(-1)
                        raw=odds(p,cfg['eps']);applied=np.minimum(raw,cap) if cap is not None else raw
                        clipped=(raw>cap) if cap is not None else np.zeros(len(raw),bool)
                        for name,weights in [('dy_before',w),('dy_after',w*applied),('dy_raw',w*raw),
                                             ('dy_sumw2_before',w*w),('dy_sumw2_after',(w*applied)**2),
                                             ('capped_rows',clipped.astype(float)),('dy_absw',abs(w)),
                                             ('capped_absw',abs(w)*clipped)]:add(name,weights)
                        add('dy_rows')
            del frame,saved,pos
    maps={
        'data_over_mc_before':divide(h['data'],h['background']+h['dy_before']),
        'data_over_mc_after':divide(h['data'],h['background']+h['dy_after']),
        'data_over_mc_uncapped':divide(h['data'],h['background']+h['dy_raw']),
        'residual_over_dy_after':divide(h['data']-h['background'],h['dy_after']),
        'cap_fraction_rows':divide(h['capped_rows'],h['dy_rows']),
        'cap_fraction_absweight':divide(h['capped_absw'],h['dy_absw']),
        'signed_dy_yield_removed_by_cap':h['dy_raw']-h['dy_after'],
        'dy_validation_rows':h['dy_rows'],
    }
    ix,iy=np.indices(shape)
    columns={'pt_low':xe[ix].ravel(),'pt_high':xe[ix+1].ravel(),'njet':((ye[iy]+ye[iy+1])/2).ravel()}
    columns.update({k:v.ravel() for k,v in {**h,**maps}.items()})
    pd.DataFrame(columns).to_csv(output/'validation_2d.csv',index=False)
    for name,z in maps.items():
        fig,ax=plt.subplots(figsize=(10,6))
        im=ax.imshow(np.ma.masked_invalid(z.T),origin='lower',aspect='auto',cmap='viridis')
        if 'over_' in name: im.set_clim(.5,1.5)
        if name.startswith('cap_fraction'): im.set_clim(0,1)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.isfinite(z[i,j]):ax.text(i,j,f'{z[i,j]:.2g}',ha='center',va='center',fontsize=6,
                                               bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=.3))
        ax.set_xticks(range(shape[0]),[f'{a:g}-{b:g}' for a,b in zip(xe[:-1],xe[1:])],rotation=45)
        ax.set_yticks(range(shape[1]),[f'{v:g}' for v in (ye[:-1]+ye[1:])/2])
        ax.set(xlabel='Dilepton pT bin [GeV] (equal display widths)',ylabel='Jet multiplicity',
               title='Final-model validation: '+name.replace('_',' '))
        fig.colorbar(im,ax=ax,extend='both' if 'over_' in name else 'neither')
        fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=170);plt.close(fig)
    audit.update(cap=cap,population='final inner validation; all physical signs retained',
                 exposure='Each role divided by its own saved validation sampling fraction',
                 caveat='Validation used for early stopping and cap selection; not an independent test',
                 bin_flow='drop; see outside_map_rows',cap_fraction='raw odds > saved final cap; denominator DY only',
                 uncertainties='CSV sumw2; no learned-model uncertainty or covariance')
    (output/'metadata.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--channel-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,default=Path('c2st_cache/signed_physical'))
    a=p.parse_args();run(a.channel_dir,a.output,a.cache_dir)
