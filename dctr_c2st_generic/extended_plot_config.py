"""Extend an existing plot config while preserving sources, samples and selections."""
from copy import deepcopy
import numpy as np
from .examples.legacy_inclusive import NOMINAL_FIELDS


def extend_plots(original):
    cfg = deepcopy(original)
    obs = {}
    def add(name, low, high, bins, label, unit='GeV'):
        obs[name] = dict(edges=np.linspace(low, high, bins+1),
                         label=label + (f' [{unit}]' if unit else ''),
                         density_per_unit=bool(unit),
                         ylabel=f'Entries / {unit}' if unit else 'Entries',
                         log=True, flow='drop')
    scalar = {
        'mli_ht':(0,1000,40,r'$H_T$'), 'mli_lt':(0,500,40,r'$L_T$'),
        'mli_mll':(0,200,40,r'$m_{\ell\ell}$'),
        'mli_mbb':(0,500,40,r'$m_{bb}$'), 'mli_ll_pt':(0,200,40,r'$p_T^{\ell\ell}$'),
        'mli_bb_pt':(0,500,40,r'$p_T^{bb}$'), 'mli_met_pt':(0,200,40,r'$p_T^{miss}$'),
        'mli_mllMET':(0,600,40,'mllMET'), 'mli_mbbllMET':(0,1200,40,'mbbllMET'),
    }
    for name,args in scalar.items(): add(name,*args)
    for name in ['mli_dr_bb','mli_mindr_lb','mli_dr_ll','mli_min_dr_llbb','mli_dr_bb_llMET']:
        add(name,0,6,40,name.removeprefix('mli_'),unit='')
    for name in ['mli_dphi_bb','mli_dphi_ll','mli_dphi_bb_nu','mli_dphi_bb_llMET']:
        add(name,-np.pi,np.pi,40,name.removeprefix('mli_'),unit='')
    for name,high in [('mli_n_jet',12),('mli_n_btag',8)]:
        add(name,-.5,high-.5,high,name.removeprefix('mli_'),unit='')
    add('mli_b_score_sum',0,6,40,'Sum of b scores',unit='')
    for obj,label in [('lep','Leading lepton'),('lep2','Subleading lepton'),
                      ('b1','b1'),('b2','b2'),('j1','j1'),('fj','Fat jet')]:
        add(f'mli_{obj}_pt',0,1000 if obj=='fj' else 400,40,label+r' $p_T$')
        add(f'mli_{obj}_eta',-5,5,40,label+r' $\eta$',unit='')
        if obj in ('b1','b2','j1'):
            add(f'mli_{obj}_b_score',0,1,40,label+' b score',unit='')
    add('mli_fj_phi',-np.pi,np.pi,40,r'Fat jet $\phi$',unit='')
    for name in ['mass','msoftdrop']:
        add(f'mli_fj_{name}',0,500,40,'Fat jet '+name)
    # User's previous binning/labels and any additional observables win.
    obs.update(deepcopy(cfg.get('observables', {})))
    cfg['observables'] = obs
    for year,lumi in [('2024',109.8),('2025',110.6)]:
        cfg['years'][year]['mc_weights'] = {'mode':'product','fields':[
            {'producer':'event_weights','column':f,'optional':True} for f in NOMINAL_FIELDS]}
        cfg['years'][year]['dy_weight'] = None
        cfg['years'][year]['luminosity_fb'] = lumi
    cfg['mc_comparison'] = dict(enabled=True, modes=['raw','luminosity','shape'],
        per_process=True, stages=['nominal','dctr'],ratio_limits=[.8,1.2],weight_label='Full nominal, no DY')
    cfg['mc_weight_variations'] = {
        'stitching_only': {'mode':'product','fields':[
            {'producer':'event_weights','column':'stitched_normalization_weight','optional':False}]},
    }
    return cfg

