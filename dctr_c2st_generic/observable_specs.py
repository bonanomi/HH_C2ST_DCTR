"""Plot metadata independent of event loading and model inputs."""
from copy import deepcopy
import warnings
import numpy as np


def observable_names(cfg):
    return list(dict.fromkeys(cfg['features'] + cfg.get('validation_vars', [])))


def defaults():
    result = {}
    def add(name, lo, hi, bins, label=None, unit='GeV'):
        result[name] = dict(edges=np.linspace(lo,hi,bins+1),
                            label=(label or name)+(f' [{unit}]' if unit else ''),
                            density_per_unit=bool(unit),
                            ylabel=f'Entries / {unit}' if unit else 'Entries',
                            log=False,flow='drop')
    for name,hi,label in [
        ('mli_ll_pt',200,r'$p_T^{\ell\ell}$'),('mli_mll',200,r'$m_{\ell\ell}$'),
        ('mli_met_pt',200,r'$p_T^{\mathrm{miss}}$'),('mli_ht',1000,r'$H_T$'),
        ('mli_lt',500,r'$L_T$'),('mli_mbb',500,r'$m_{bb}$'),
        ('mli_bb_pt',500,r'$p_T^{bb}$'),('mli_mllMET',600,'mllMET'),
        ('mli_mbbllMET',1200,'mbbllMET')]:add(name,0,hi,40,label)
    add('mli_n_jet',-.5,11.5,12,'Jet multiplicity','')
    add('mli_n_btag',-.5,7.5,8,'b-tagged jet multiplicity','')
    add('mli_b_score_sum',0,6,40,'Sum of b-tag scores','')
    for name in ['dr_bb','mindr_lb','dr_ll','min_dr_llbb','dr_bb_llMET']:
        add('mli_'+name,0,6,40,name,'')
    for name in ['dphi_bb','dphi_ll','dphi_bb_nu','dphi_bb_llMET']:
        add('mli_'+name,-np.pi,np.pi,40,name,'')
    for obj in ['lep','lep2','b1','b2','j1','fj']:
        add(f'mli_{obj}_pt',0,1000 if obj=='fj' else 400,40,obj+r' $p_T$')
        add(f'mli_{obj}_eta',-5,5,40,obj+r' $\eta$','')
        if obj in ['b1','b2','j1']:add(f'mli_{obj}_b_score',0,1,40,obj+' b-tag score','')
    add('mli_fj_phi',-np.pi,np.pi,40,r'Fat jet $\phi$','')
    for name in ['mass','msoftdrop']:add('mli_fj_'+name,0,500,40,'Fat jet '+name)
    return result


def resolve_observables(cfg, data, mc, overrides=None, legacy_defaults=None):
    names=observable_names(cfg)
    overrides={} if overrides is None else overrides
    if not isinstance(overrides,dict) or set(overrides)-set(names):
        raise ValueError('Overrides must map configured features/validation_vars to plot specifications')
    registry=defaults()
    # Preserve any locally customized definitions for existing observables.
    for name,spec in (legacy_defaults or {}).items():
        registry.setdefault(name,{}).update(deepcopy(spec))
    output={};flows={}
    allowed={'edges','label','density_per_unit','ylabel','log','flow'}
    for name in names:
        spec=deepcopy(registry.get(name,dict(label=name,density_per_unit=False,ylabel='Entries',log=False,flow='drop')))
        if name in cfg.get('plot_edges',{}):spec['edges']=cfg['plot_edges'][name]
        custom=overrides.get(name,{})
        if not isinstance(custom,dict) or set(custom)-allowed:
            raise ValueError(f'Invalid observable override for {name}')
        spec.update(deepcopy(custom))
        if 'edges' not in spec:
            bounds=[]
            for frame in (data,mc):
                vals=frame[name].to_numpy(dtype=float)
                finite=vals[np.isfinite(vals)]
                if len(finite):bounds.append((finite.min(),finite.max()))
            if not bounds:raise ValueError(f'No finite values for {name}')
            lo=min(v[0] for v in bounds);hi=max(v[1] for v in bounds)
            if lo==hi:lo,hi=lo-.5,hi+.5
            spec['edges']=np.linspace(lo,hi,int(cfg.get('plot_bins',40))+1)
            warnings.warn(f'{name}: no predefined binning; using full finite range [{lo}, {hi}]. Supply an override if outliers dominate.')
        edges=np.asarray(spec['edges'],dtype=float)
        if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or not (np.diff(edges)>0).all():
            raise ValueError(f'Invalid edges for {name}')
        if spec['flow'] not in ('drop','fold'):raise ValueError(f'Invalid flow for {name}')
        spec['edges']=edges
        output[name]=spec;flows[name]={}
        for side,frame in [('data',data),('mc',mc)]:
            vals=frame[name].to_numpy(dtype=float)
            masks={'nonfinite':~np.isfinite(vals),'underflow':np.isfinite(vals)&(vals<edges[0]),
                   'overflow':np.isfinite(vals)&(vals>edges[-1])}
            weights=frame['weight_before'].to_numpy(dtype=float)
            flows[name][side]={key:dict(rows=int(mask.sum()),signed_sumw=float(weights[mask].sum())) for key,mask in masks.items()}
            if masks['nonfinite'].any():raise ValueError(f'Nonfinite {name} in {side}')
    return output,flows
