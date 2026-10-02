"""Scientific figure of already-scored development paths, never checkpoint selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def curves(method):
    if method.get('status')!='ok':raise ValueError('successful scored trajectory required')
    t=method['trajectory'];s=t['stages']
    if len(s)!=10 or [r['cumulative_gradients'] for r in s]!=list(range(30,301,30)):
        raise ValueError('ten declared equal-gradient prefixes required')
    # Recover the recorded initial objective from E1 prefix when possible is NOT
    # safe: stage0 may already have improved it. Plot absolute FULL E1 instead.
    return dict(x=[r['cumulative_gradients'] for r in s],
        energy=[r['accepted_full_total'] for r in s],
        mean=[r['accepted']['mean_canvas_px'] for r in s],
        prefix_mean=[r['image_selected_prefix']['mean_canvas_px'] for r in s],
        p90=[r['accepted']['p90_canvas_px'] for r in s],
        prefix_p90=[r['image_selected_prefix']['p90_canvas_px'] for r in s])


def run(source,output):
    if output.suffix!='.png':raise ValueError('PNG output required')
    svg=output.with_suffix('.svg')
    if output.exists() or svg.exists():raise FileExistsError('new scientific figure targets required')
    result=json.loads(source.read_text(encoding='utf-8'))
    if result.get('posthoc_only') is not True or len(result['rows'])!=3:
        raise ValueError('three-pair posthoc score required')
    data=[{name:curves(row['methods'][name]) for name in ('analytic','f2')} for row in result['rows']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True)
    colors={'analytic':'#1f77b4','f2':'#d97706'}
    for col,(row,values) in enumerate(zip(result['rows'],data,strict=True)):
        for name,c in values.items():
            for r,key in enumerate(('energy','mean','p90')):
                axes[r,col].plot(c['x'],c[key],color=colors[name],marker='.',label=name+' accepted')
                if r:
                    axes[r,col].plot(c['x'],c['prefix_'+key],color=colors[name],linestyle='--',
                                     label=name+' E1-selected prefix')
        axes[0,col].set_title(row['name'].replace('miit_','')+f"; n={row['required_available']}")
        for r in range(3):axes[r,col].grid(alpha=.2)
        axes[2,col].set_xlabel('Cumulative gradients (not elapsed time)')
    for r,label in enumerate(('Full-resolution E1 (lower is better)','Mean TRE (512 canvas pixels)',
                              'Pair-p90 TRE (512 canvas pixels)')):
        axes[r,0].set_ylabel(label)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.012))
    fig.suptitle('Unchanged recipes: image-objective progress versus anatomical error',fontsize=13)
    fig.tight_layout(rect=(0,.075,1,.965))
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);fig.savefig(svg);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)


if __name__=='__main__':main()
