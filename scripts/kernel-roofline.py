#!/usr/bin/env python3
"""A30 per-kernel roofline: NCU scalar FLOPs/DRAM bytes, measured triad and cuBLAS peak.

This is separate from the application W4 plot. Run on an idle analysis host.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ncu',type=Path,required=True)
    ap.add_argument('--stream',type=Path,required=True)
    ap.add_argument('--peaks',type=Path,default=Path('results/hpc-a30/2026-09-17/peaks/peak.jsonl'))
    ap.add_argument('--out',type=Path,default=Path('results/analysis/figures/hpc-a30-kernel-2026-11-03'))
    a=ap.parse_args()
    rows=json.loads((a.ncu/'kernels.json').read_text())
    stream=json.loads((a.stream/'summary.json').read_text())
    bandwidth=next(r['median_GB_s'] for r in stream if r['operation']=='triad')
    peaks=[json.loads(l) for l in a.peaks.read_text().splitlines() if l.strip()]
    peaks=[r for r in peaks if r.get('backend')=='cudann' and r.get('blas')=='cublas'
           and r.get('dtype')=='float' and r.get('device',{}).get('name')=='NVIDIA A30']
    if not peaks: raise ValueError('no native cuBLAS FP32 A30 peak')
    sizes=defaultdict(list)
    for r in peaks: sizes[r['n']].append(r['gemm_gflops'])
    size,peak=max(((n,st.median(v)) for n,v in sizes.items()),key=lambda x:x[1])
    wanted={('NN',512,256,784),('TN',512,256,256),('NT',512,784,256),('NN',4096,4096,4096)}
    points=[r for r in rows if r.get('shape_verified') and tuple(r.get('shape') or []) in wanted
            and r.get('intensity') and r.get('gflops')]
    if not points: raise ValueError('no verified GEMM kernels with measured FLOPs, duration and DRAM bytes')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5),dpi=150)
    ridge=peak/bandwidth
    xs=[min(min(r['intensity'] for r in points)/2,ridge/10),ridge,max(max(r['intensity'] for r in points)*2,ridge*10)]
    ax.plot(xs,[min(peak,bandwidth*x) for x in xs],'k--',label=f'cuBLAS n={size}: {peak/1000:.2f} TFLOP/s; triad median {bandwidth:.1f} GB/s')
    colors={'cudann':'#2ca02c','syclnn':'#1f77b4','ompnn':'#d62728'}
    markers={'NN':'o','TN':'^','NT':'s'}
    seen=set()
    for r in points:
        shape=r['shape']; label=r['label']+' '+r['blas']
        ax.scatter(r['intensity'],r['gflops'],color='black' if r['blas']=='auto' else colors[r['backend']],
                   marker=markers[shape[0]],s=45,label=label if label not in seen else None)
        ax.annotate(shape[0]+(' 4096' if shape[1]==4096 else ''),(r['intensity'],r['gflops']),xytext=(4,4),textcoords='offset points',fontsize=7)
        seen.add(label)
    ax.set(xscale='log',yscale='log',xlabel='NCU scalar FLOPs / measured DRAM bytes',ylabel='NCU scalar GFLOP/s',
           title='A30 FP32 GEMM kernel roofline\nNCU kernel replay, cold caches; application W4 shown separately')
    ax.grid(True,alpha=.25); ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1))
    a.out.mkdir(parents=True,exist_ok=True); fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(a.out/f'kernel-roofline.{suffix}',bbox_inches='tight')
    plt.close(fig)
    (a.out/'reference.json').write_text(json.dumps({'bandwidth_GB_s':bandwidth,'bandwidth_statistic':'median triad',
         'peak_GFLOP_s':peak,'peak_n':size,'peak_source':str(a.peaks),'ncu_source':str(a.ncu),
         'stream_source':str(a.stream),'points':points},indent=2)+'\n')


if __name__=='__main__': main()
