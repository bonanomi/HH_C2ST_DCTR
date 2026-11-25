"""MC25/MC24 comparisons from the plotting script's histogram sidecars.

Can also run without parquet loading or TensorFlow:
python -m dctr_c2st_generic.mc_comparison --input EXISTING_RUN --output NEW_DIR
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
from .config import safe_name, write_json


def normalized(h, v, mode, lumi=None):
    h, v = np.asarray(h, float), np.asarray(v, float)
    if mode == 'raw':
        return h, v
    if mode == 'luminosity':
        if lumi is None or not np.isfinite(lumi) or lumi <= 0:
            raise ValueError('Set positive luminosity_fb for BOTH years')
        return h / lumi, v / lumi**2
    if mode == 'shape':
        total = h.sum()
        if total <= 0:
            raise ValueError('Shape normalization requires positive signed integral')
        # Delta-method variance includes correlation of each bin with its total.
        variance = v / total**2 + h**2 * v.sum() / total**4 - 2*h*v / total**3
        return h / total, np.maximum(variance, 0)
    raise ValueError('MC comparison modes: raw, luminosity, shape')


def compare_folder(source, output, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    options = config.get('mc_comparison', {})
    if not options.get('enabled', False):
        return
    output.mkdir(parents=True, exist_ok=True)
    for observable, spec in config['observables'].items():
        reference = json.loads((source / f'{observable}_2024_nominal.json').read_text())
        edges = np.asarray(reference['edges'], float)
        centers, widths = (edges[1:] + edges[:-1])/2, np.diff(edges)
        for stage in options.get('stages', ['nominal']):
            if stage not in ('nominal', 'dctr'):
                raise ValueError('MC comparison stages: nominal, dctr')
            candidate = json.loads((source / f'{observable}_2025_{stage}.json').read_text())
            if candidate['edges'] != reference['edges'] or candidate.get('flow') != reference.get('flow'):
                raise ValueError('MC comparisons require identical bins and flow handling')
            groups = {'total': None}
            if options.get('per_process', False):
                if set(reference['components']) != set(candidate['components']):
                    raise ValueError('Per-process comparisons require matching process labels across years')
                groups.update({f'process_{i:02d}': label for i,label in enumerate(reference['components'])})
            for mode in options.get('modes', ['luminosity']):
                for slug, label in groups.items():
                    if mode == 'shape':
                        totals = [sum(item['mc'] if label is None else item['components'][label]['sumw'])
                                  for item in (reference,candidate)]
                        if min(totals) <= 0:
                            warnings.warn(f'Skipping undefined shape normalization: {observable}, {label}, {stage}')
                            write_json(output/f'{safe_name(observable)}_{stage}_shape_{slug}_skipped.json',
                                       {'reason':'Nonpositive signed integral', 'integrals':totals})
                            continue
                    arrays = []
                    for year, item in [('2024',reference), ('2025',candidate)]:
                        h, v = (item['mc'], item['mc_sumw2']) if label is None else (item['components'][label]['sumw'], item['components'][label]['sumw2'])
                        arrays.append(normalized(h,v,mode,config['years'][year].get('luminosity_fb')))
                    (a,va),(b,vb)=arrays
                    valid = a > 0
                    ratio = np.divide(b,a,out=np.full_like(b,np.nan),where=valid)
                    numerr = np.divide(np.sqrt(vb),a,out=np.full_like(b,np.nan),where=valid)
                    denerr = np.divide(np.sqrt(va),a,out=np.full_like(a,np.nan),where=valid)
                    limits = options.get('ratio_limits',[.8,1.2])
                    if len(limits)!=2 or limits[0]>=limits[1]:
                        raise ValueError('Invalid MC comparison ratio_limits')
                    scale = 1/widths if spec.get('density_per_unit',True) else np.ones_like(widths)
                    with plt.rc_context({'font.size':13,'axes.labelsize':16,'xtick.direction':'in','ytick.direction':'in'}):
                        fig,(ax,rat)=plt.subplots(2,1,figsize=(9,8),sharex=True,gridspec_kw={'height_ratios':[3,1],'hspace':.05})
                        ax.stairs(a*scale,edges,color='#2864a0',linewidth=2,label='MC 2024')
                        def band(axis,lo,hi):
                            axis.fill_between(edges,np.r_[lo,lo[-1]],np.r_[hi,hi[-1]],step='post',facecolor='none',edgecolor='#2864a0',hatch='////',linewidth=0)
                        band(ax,(a-np.sqrt(va))*scale,(a+np.sqrt(va))*scale)
                        ax.errorbar(centers,b*scale,yerr=np.sqrt(vb)*scale,fmt='o',color='#c43b30',markersize=3,
                                    label='MC 2025' + (' × DCTR' if stage=='dctr' else ''))
                        if spec.get('log',True) and np.all(a>=0) and np.all(b>=0) and np.any(a+b>0):
                            ax.set_yscale('log')
                            positive=np.r_[a[a>0]*scale[a>0],b[b>0]*scale[b>0]]
                            ax.set_ylim(max(positive.min()*.3,1e-12),positive.max()*15)
                        else:
                            ax.margins(y=.25)
                        ylabel={'raw':'Weighted entries','luminosity':'Weighted entries / fb$^{-1}$','shape':'Unit-area distribution'}[mode]
                        if spec.get('density_per_unit',True): ylabel+=' / bin width'
                        ax.set_ylabel(ylabel)
                        ax.text(0,1.025,'CMS',fontweight='bold',fontsize=24,transform=ax.transAxes)
                        ax.text(.2,1.035,config.get('cms_label','Work in progress'),fontsize=12,transform=ax.transAxes)
                        ax.text(.03,.94,f'{label or "Total MC"}\n{mode}; {options.get("context_label", str(source.parent.name) + ", " + source.name)}\n{options.get("weight_label", "")}',va='top',transform=ax.transAxes)
                        ax.legend(loc='upper right',frameon=False)
                        band(rat,1-denerr,1+denerr)
                        rat.errorbar(centers[valid],ratio[valid],yerr=numerr[valid],fmt='o',color='#c43b30',markersize=3)
                        rat.axhline(1,color='gray',linestyle='--')
                        rat.set_ylim(*limits);rat.set_ylabel('2025 / 2024')
                        rat.set_xlabel(spec.get('label',observable));rat.set_xlim(edges[0],edges[-1])
                        for axis in (ax,rat):
                            axis.minorticks_on();axis.tick_params(which='both',top=True,right=True)
                        fig.subplots_adjust(left=.16,right=.97,top=.92,bottom=.1)
                        stem=output/f'{safe_name(observable)}_mc25_over_mc24_{stage}_{mode}_{slug}'
                        for extension in config.get('formats',['png']):
                            if extension not in ('png','pdf','svg'): raise ValueError('Unsupported figure format')
                            fig.savefig(str(stem)+'.'+extension,dpi=160)
                        plt.close(fig)
                    write_json(str(stem)+'.json',dict(edges=edges,process=label or 'total',mode=mode,stage=stage,
                        mc24=a,mc25=b,mc24_variance=va,mc25_variance=vb,ratio=ratio,
                        numerator_error_over_denominator=numerr,denominator_relative_error=denerr,
                        ratio_limits=limits,undefined_ratio_bins=np.flatnonzero(~valid),
                        outside_ratio_range_bins=np.flatnonzero(valid & ((ratio<limits[0]) | (ratio>limits[1]))),
                        uncertainty_note='Separate marginal statistical errors; cross-year covariance and DCTR uncertainty not included.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--modes',nargs='+',choices=['raw','luminosity','shape'],default=['raw','luminosity','shape'])
    p.add_argument('--lumi24',type=float)
    p.add_argument('--lumi25',type=float)
    p.add_argument('--per-process',action='store_true')
    p.add_argument('--include-dctr',action='store_true')
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a fresh output directory')
    cfg=json.loads((args.input/'plot_config.json').read_text())
    for year,lumi in [('2024',args.lumi24),('2025',args.lumi25)]:
        if lumi is not None: cfg['years'][year]['luminosity_fb']=lumi
    cfg['mc_comparison']=dict(enabled=True,modes=args.modes,per_process=args.per_process,
                              stages=['nominal','dctr'] if args.include_dctr else ['nominal'])
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'plot_config.json',cfg)
    for channel in cfg['channels']:
        for region in cfg['regions']:
            compare_folder(args.input/channel/region,args.output/channel/region,cfg)


if __name__=='__main__':
    main()

