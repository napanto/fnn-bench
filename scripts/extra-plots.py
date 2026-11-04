#!/usr/bin/env python3
"""Figures for new workstation experiments. Run only on an idle analysis host."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import runpy
import statistics as st
from gemm_shapes import TARGETS

COLORS={'cudann':'#2ca02c','syclnn':'#1f77b4','ompnn':'#d62728'}
MODEL_LABELS={'hipcc':'HIP (cudann)','adaptivecpp':'SYCL (AdaptiveCpp)','amdclang':'OpenMP (amdclang)'}


def save(plt,fig,out,name):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)
    return name


def stream(plt,out):
    rows=json.loads((out/'summary.json').read_text())
    fig,ax=plt.subplots(figsize=(6.5,3.8))
    xs=list(range(len(rows)))
    ax.bar([x-.18 for x in xs],[r['best_GB_s'] for r in rows],.36,label='best')
    ax.bar([x+.18 for x in xs],[r['median_GB_s'] for r in rows],.36,label='median')
    ax.set_xticks(xs,[r['operation'] for r in rows]);ax.set_ylabel('effective GB/s (decimal)')
    ax.set_title(out.parent.name+' device-resident bandwidth');ax.legend()
    return [save(plt,fig,out,'stream-bandwidth')]


def sanitizer(plt,out):
    records={}
    for path in out.glob('cases-*.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip():
                r=json.loads(line);records[r['key']]=r
    labels=sorted({(r['case'],r['dtype'],r['options']) for r in records.values()})
    if not labels: return []
    tools=['memcheck','racecheck','synccheck','initcheck']
    data=[[-1 for t in tools] for l in labels]
    for r in records.values():
        data[labels.index((r['case'],r['dtype'],r['options']))][tools.index(r['tool'])]=1 if r['result']=='pass' else 0
    from matplotlib.colors import ListedColormap,BoundaryNorm
    fig,ax=plt.subplots(figsize=(7,max(3,len(labels)*.3+1.2)))
    ax.imshow(data,cmap=ListedColormap(['#dddddd','#cf5555','#68ad76']),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),aspect='auto')
    ax.set_xticks(range(4),tools)
    ax.set_yticks(range(len(labels)),[f"{c.replace('test_parity_with_reference','parity').replace('test_remainder_batch','remainder')} / {d} / {o}" for c,d,o in labels],fontsize=7)
    for i,row in enumerate(data):
        for j,v in enumerate(row): ax.text(j,i,{-1:'pending',0:'see log',1:'pass'}[v],ha='center',va='center',fontsize=7)
    ax.set_title('Compute Sanitizer + parity assertions (48 requested combinations)')
    return [save(plt,fig,out,'sanitizer-matrix')]


def stat(plt,out):
    rows=json.loads((out/'summary.json').read_text()); names=[]
    for family in ('mnist','cup'):
        selected=[r for r in rows if family in r['configuration']]
        if not selected: continue
        fig,ax=plt.subplots(figsize=(7,4))
        for i,r in enumerate(selected):
            ys=r['process_medians_ms'];dx=[i+(j-(len(ys)-1)/2)*.12 for j in range(len(ys))]
            color=COLORS['syclnn' if r['configuration'].startswith('syclnn') else 'cudann']
            ax.scatter(dx,ys,color=color,s=35,label='independent process median' if i==0 else None)
            ax.plot([i-.25,i+.25],[st.median(ys)]*2,'k-',lw=1.5,label='median across processes' if i==0 else None)
        ax.set_xticks(range(len(selected)),[r['configuration'] for r in selected],rotation=25,ha='right',fontsize=8)
        ax.set_ylabel('steady epoch ms');ax.set_yscale('log');ax.legend(fontsize=7)
        ax.set_title('Independent-process confirmation: '+family+' (within-process IQRs in table)')
        names.append(save(plt,fig,out,'stat-'+family))
    return names


def nsys(plt,out):
    rows=json.loads((out/'summary.json').read_text());names=[]
    if not rows: return names
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
    x=list(range(len(rows)));labels=[r['name'] for r in rows]
    busy=[r['gpu_busy_ns']/1e6 for r in rows];gap=[r['gpu_idle_ns']/1e6 for r in rows]
    axes[0].bar(x,busy,label='target GPU-work union')
    axes[0].bar(x,gap,bottom=busy,label='gaps in target GPU work')
    axes[0].set_ylabel('range ms');axes[0].legend(fontsize=7)
    axes[0].set_title('Captured interval: target GPU work and gaps (background work may exist)')
    classes=['GEMM','BLAS vector/reduction','hand-written','other kernels','H2D','D2H','memset','other copy'];bottom=[0.]*len(rows)
    for kind in classes:
        values=[r['gpu_ns'].get(kind,0)/1e6 for r in rows]
        if any(values): axes[1].bar(x,values,bottom=bottom,label=kind)
        bottom=[a+b for a,b in zip(bottom,values)]
    axes[1].plot(x,[r['range_ns']/1e6 for r in rows],'k_',ms=14,label='range wall')
    axes[1].set_ylabel('sum of GPU intervals, ms');axes[1].set_title('Execution classes (sums can include overlap)')
    axes[1].set_xticks(x,labels,rotation=25,ha='right');axes[1].legend(fontsize=7,ncol=2)
    names.append(save(plt,fig,out,'nsys-breakdown'))
    helper=runpy.run_path(str(Path(__file__).with_name('nsys-epoch-table.py')))
    fig,axes=plt.subplots(len(rows),1,figsize=(10,max(3,len(rows)*1.5)),squeeze=False)
    palette={'GEMM':'#4c72b0','BLAS vector/reduction':'#937860','hand-written':'#dd8452','other kernels':'#999999',
             'H2D':'#55a868','D2H':'#c44e52','memset':'#8172b3','other copy':'#64b5cd'}
    for ax,summary in zip(axes[:,0],rows):
        prefix=out/summary['name'];(lo,hi),source,apis,nv=helper['ranges'](Path(str(prefix)+'.sqlite'))
        trace=helper['csv_rows'](Path(str(prefix)+'-gpu_cuda_gpu_trace.csv'),'Start (ns)')
        intervals=defaultdict(list)
        for r in trace:
            a=float(r['Start (ns)'].replace(',',''));b=a+float(r['Duration (ns)'].replace(',',''))
            if b<=lo or a>=hi: continue
            stream_id=r.get('Stream') or r.get('Strm') or r.get('Strm ID') or r.get('Stream ID') or 'GPU'
            kind=helper['gpu_class'](r['Name'])
            intervals[(stream_id,kind)].append(((max(a,lo)-lo)/1e6,(min(b,hi)-max(a,lo))/1e6))
        streams=sorted({s for s,k in intervals},key=lambda s:int(s) if s.isdigit() else float('inf'))
        for (s,kind),values in intervals.items(): ax.broken_barh(values,(streams.index(s)-.4,.8),facecolors=palette[kind])
        ax.set_yticks(range(len(streams)),streams,fontsize=6);ax.set_ylabel('stream',fontsize=7)
        ax.set_title(summary['name'],fontsize=8,loc='left');ax.set_xlim(0,(hi-lo)/1e6)
    axes[-1,0].set_xlabel('ms from delimited range start')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=k) for k,c in palette.items()],loc='upper center',
               bbox_to_anchor=(.5,1.035),ncol=4,fontsize=7,frameon=False)
    names.append(save(plt,fig,out,'nsys-timelines'))
    return names


def counters(plt,out):
    rows=[r for r in json.loads((out/'kernels.json').read_text()) if tuple(r.get('shape') or []) in TARGETS]
    if not rows:return []
    labels=sorted({(r['label'],r['blas']) for r in rows})
    shapes=sorted({tuple(r['shape']) for r in rows},key=lambda s:(s[1]==4096,{'NN':0,'TN':1,'NT':2}[s[0]]))
    def field(r,name):
        if name=='VGPR_Count':return r.get('vgprs')
        if name=='time_us':return r.get('duration_ns',0)/1e3
        c=r.get('counters',{})
        fallbacks={'LDSBankConflict':'SQ_LDS_BANK_CONFLICT','LDSInsts':'FNN_LDS_BASE','Wavefronts':'SQ_WAVES'}
        return c.get(name,c.get(fallbacks.get(name,'')))
    metrics=['VGPR_Count','LDSBankConflict','LDSInsts','VALUInsts','OccupancyPercent','time_us']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    width=.8/max(1,len(labels))
    for ax,metric in zip(axes.flat,metrics):
        for j,(label,blas) in enumerate(labels):
            points={tuple(r['shape']):r for r in rows if (r['label'],r['blas'])==(label,blas)}
            pairs=[(i,field(points[s],metric)) for i,s in enumerate(shapes) if s in points]
            pairs=[(i,v) for i,v in pairs if v is not None]
            if pairs:
                first=next(iter(points.values()))
                color='black' if blas=='auto' else COLORS.get(first['backend'])
                xs=[i+(j-(len(labels)-1)/2)*width for i,v in pairs];ys=[v for i,v in pairs]
                name=MODEL_LABELS.get(label,label)+' '+('rocBLAS' if blas=='auto' else blas)
                if metric in ('LDSInsts','VALUInsts','time_us'):
                    ax.scatter(xs,ys,s=25,marker='D' if blas=='auto' else 'o',label=name,color=color)
                else:
                    ax.bar(xs,ys,width,label=name,color=color)
        ax.set_title(metric+' (tool value)' if metric!='time_us' else 'kernel time us (exploratory)')
        if metric in ('LDSInsts','VALUInsts','time_us'): ax.set_yscale('log')
        ax.set_xticks(range(len(shapes)),[s[0]+(' 4096' if s[1]==4096 else '') for s in shapes],fontsize=8)
    axes[0,0].legend(fontsize=6)
    fig.suptitle('GEMM counters: '+out.parent.name+', current profiling conditions')
    return [save(plt,fig,out,'gemm-counters')]


def roofline(plt,out,stream_dir):
    rows=[r for r in json.loads((out/'kernels.json').read_text()) if r.get('intensity') and r.get('gflops') and tuple(r.get('shape') or []) in TARGETS]
    if not rows: return []
    calibration=stream_dir/'calibration.json'
    if not calibration.exists():
        raise ValueError('GL2 request bytes require a matching interface calibration, not the large-buffer DRAM STREAM rate')
    reference=json.loads(calibration.read_text())
    if reference.get('counter_boundary')!='GL2 external interface':
        raise ValueError('counter and bandwidth reference boundaries differ')
    bandwidth=reference['counter_bandwidth_median_GB_s']
    if bandwidth<=0: raise ValueError('bandwidth reference must be positive')
    peaks=[]
    for path in (out/'peaks').glob('*.jsonl'):
        peaks += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    peaks=[r for r in peaks if r.get('dtype')=='float' and r.get('backend')=='cudann' and r.get('blas') not in ('tiled','handwritten')]
    if not peaks: return []
    peak=max(r['gemm_gflops'] for r in peaks);ridge=peak/bandwidth
    xs=[min(min(r['intensity'] for r in rows)/2,ridge/10),ridge,max(max(r['intensity'] for r in rows)*2,ridge*5)]
    fig,ax=plt.subplots(figsize=(8,5));ax.loglog(xs,[min(peak,bandwidth*x) for x in xs],'k--',label=f'vendor reference {peak/1000:.2f} TFLOP/s; GL2 triad {bandwidth:.1f} GB/s')
    seen=set();markers={'NN':'o','TN':'^','NT':'s'}
    for r in rows:
        label=MODEL_LABELS.get(r['label'],r['label'])+' '+('rocBLAS' if r['blas']=='auto' else r['blas']);shape=r['shape']
        ax.scatter(r['intensity'],r['gflops'],s=40,marker=markers[shape[0]],color='black' if r['blas']=='auto' else COLORS.get(r['backend']),label=label if label not in seen else None)
        ax.annotate(shape[0]+(' 4096' if shape[1]==4096 else ''),(r['intensity'],r['gflops']),xytext=(4,3),textcoords='offset points',fontsize=7)
        seen.add(label)
    ax.set_xlabel('mathematical GEMM FLOPs / measured GL2 external bytes')
    ax.set_ylabel('mathematical GEMM GFLOP/s (memory-pass median)')
    ax.set_title(out.parent.name+' per-kernel GL2 interface roofline\nCache-resident bandwidth reference; current profiling conditions')
    ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1))
    (out/'figures').mkdir(exist_ok=True)
    (out/'figures/kernel-roofline-reference.json').write_text(json.dumps({'counter_boundary':reference['counter_boundary'],
        'bandwidth_GB_s':bandwidth,'bandwidth_source':str(calibration),'peak_GFLOP_s':peak,
        'peak_source':str(out/'peaks'),'points_source':str(out/'kernels.json')},indent=2)+'\n')
    return [save(plt,fig,out,'kernel-roofline')]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('kind',choices=('stream','sanitizer','stat','nsys','counters','roofline'))
    ap.add_argument('directory',type=Path);ap.add_argument('--stream',type=Path)
    a=ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'font.size':9,'axes.grid':True,'grid.alpha':.2})
    if a.kind=='roofline': names=roofline(plt,a.directory,a.stream)
    else: names=globals()[a.kind](plt,a.directory)
    if names:
        p=a.directory/'README.md';text=p.read_text() if p.exists() else ''
        for name in names:
            line=f'![{name}](figures/{name}.png)'
            if line not in text: text+='\n'+line+'\n'
        p.write_text(text)
    print('figures:',', '.join(names) if names else 'no usable data for this figure')


if __name__=='__main__': main()
