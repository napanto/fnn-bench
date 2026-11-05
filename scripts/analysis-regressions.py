#!/usr/bin/env python3
"""Small, single-threaded regressions for analysis semantics; no benchmark or plots."""
import copy
import importlib.util
import math
import os
import csv
import json
import runpy
import sqlite3
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'bench'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnnbench import results as R, plot as P


def row(**updates):
    r = dict(backend='syclnn', device={'name':'NVIDIA A30','type':'gpu'}, dtype='float',
             build_info={'compiler':'clang', 'sycl_implementation':'DPC++ (intel/llvm)', 'version':'1'},
             blas='auto', mode='train', workload='mnist-512-256', batch=256, n_samples=8192,
             epochs=3, repeat=3, warmup=1, timestamp='1', options={}, effective_options={'queue':'InOrder','memory':'Device'},
             check={'ok':True}, results={'median_epoch_s':1., 'steady_epoch_s':.8, 'losses_finite':True,'losses':[[1.]],
                                       'flops_per_epoch':{'gemm':1e9}, 'intensity_flop_per_byte':10})
    r.update(updates)
    return r


class Ax:
    def __init__(self): self.lines=[]; self.bars=[]; self.title=''
    def plot(self,x,y,*args,**kwargs): self.lines.append((list(x),list(y),kwargs))
    def bar(self,x,y,*args,**kwargs): self.bars.append(list(y))
    def set_title(self,s): self.title=s
    def __getattr__(self,name): return lambda *args,**kwargs: None


class Plt:
    def __init__(self): self.axes=[]
    def subplots(self,**kwargs):
        ax=Ax(); self.axes.append(ax); return object(),ax


class AnalysisTests(unittest.TestCase):
    def test_replicates_supersession_and_copies(self):
        a=row(); b=copy.deepcopy(a); b['timestamp']='2'; b['results'].update(steady_epoch_s=1.2,median_epoch_s=2.)
        self.assertEqual(len(R.dedupe([a,b,copy.deepcopy(a)])),2)
        self.assertIs(R.dedupe([a,b],supersede=True)[0],b)
        out=R.aggregate([a,b,copy.deepcopy(a)])
        self.assertEqual(out[0]['aggregate_count'],2)
        self.assertEqual(R.epoch_s(out[0]),1.)

    def test_configuration_boundaries(self):
        a=row()
        for change in ('effective','version','package','flags','library','protocol'):
            b=copy.deepcopy(a)
            if change=='effective': b['effective_options']['memory']='Shared'
            if change=='version': b['build_info']['version']='2'
            if change=='package': b['sysinfo']={'packages':{'syclnn':'2'}}
            if change=='flags': b['build_info']['flags']='-fast'
            if change=='library': b['sysinfo']={'libs':['libcublas.so.13']}
            if change=='protocol': b['epochs']=5
            self.assertNotEqual(R.configuration_key(a),R.configuration_key(b),change)
            self.assertEqual(len(R.aggregate([a,b])),2)

    def test_shared_validity_and_inference_clock(self):
        a=row(mode='infer'); a['results'].update(median_epoch_s=10.,steady_epoch_s=1.,steady_epochs_s=[.1,.2])
        self.assertEqual(P._epoch_s(a),10.)
        b=copy.deepcopy(a); b['check']['ok']=False
        c=copy.deepcopy(a); c['results']['losses']=[[float('nan')]]
        d=copy.deepcopy(a); d['results']['median_epoch_s']=float('inf')
        self.assertEqual(len(P._good([a,b,c,d])),1)
        p=Plt()
        with patch.object(P,'_save'):
            P.plot_inference(p,[a],Path('unused'))
        self.assertEqual(p.axes[0].bars,[[10000.]])
        self.assertEqual(p.axes[0].lines[0][1],[1000.])

    def test_new_steady_samples(self):
        a=row(); a['results']['steady_epochs_s']=[1.,3.,2.]
        self.assertEqual(R.epoch_s(a),2.)
        self.assertEqual(R.epoch_s(a,'median'),1.)

    def test_reference_selection_and_ridge(self):
        a=row(backend='cudann',blas='cublas',n=8192,gemm_gflops=9723.,act_bandwidth_gbs=700.)
        b=copy.deepcopy(a); b.update(blas='tiled',gemm_gflops=309.)
        c=copy.deepcopy(a); c.update(n=1024,gemm_gflops=5000.)
        refs=P._select_peaks([a,b,c])
        self.assertEqual(sorted(r['gemm_gflops'] for r in refs),[309.,9723.])
        self.assertEqual(sorted(r['gemm_gflops'] for r in P._select_peaks([c,b,a])),[309.,9723.])
        xs,ys=P._roofline_points(100.,10.)
        self.assertEqual(xs[1],10.)
        self.assertEqual(ys,[1.,100.,100.])

    def test_ablation_names_and_titles(self):
        rs=[]
        for impl in ('DPC++ (intel/llvm)','AdaptiveCpp'):
            for batch in (256,1024):
                for mode in ('train','infer'):
                    a=row(batch=batch,mode=mode)
                    a['build_info']['sycl_implementation']=impl
                    rs.append(a)
                    for opt,value in (('queue','OutOfOrder'),('memory','Shared')):
                        b=copy.deepcopy(a); b['options']={opt:value}; b['effective_options'][opt]=value
                        rs.append(b)
        saved=[]; p=Plt()
        with patch.object(P,'_save',side_effect=lambda plt,fig,out,name:saved.append(name)):
            P.plot_ablation(p,rs,Path('unused'))
        self.assertEqual(len(saved),8)
        self.assertEqual(len(set(saved)),8)
        for name in saved:
            self.assertTrue('dpcpp' in name or 'adaptivecpp' in name)
            self.assertTrue('-b256-' in name or '-b1024-' in name)
            self.assertTrue('-train-' in name or '-infer-' in name)
        self.assertTrue(any('DPC++' in ax.title for ax in p.axes))
        self.assertTrue(any('AdaptiveCpp' in ax.title for ax in p.axes))

    def test_ablation_protocol_difference_is_labelled(self):
        a=row(epochs=5,repeat=5)
        variants=[]
        for opt,value in (('queue','OutOfOrder'),('memory','Shared')):
            b=copy.deepcopy(a); b.update(epochs=3,repeat=3)
            b['options']={opt:value}; b['effective_options'][opt]=value; variants.append(b)
        p=Plt()
        with patch.object(P,'_save') as save:
            P.plot_ablation(p,[a,*variants],Path('unused'))
        self.assertEqual(save.call_count,1)
        self.assertIn('baseline 5 epochs / 5 repeats',p.axes[0].title)

    def test_delimited_trace_union_and_nested_host_apis(self):
        ns=runpy.run_path(str(Path(__file__).with_name('nsys-epoch-table.py')))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent/'results/analysis') as tmp:
            prefix=Path(tmp)/'cudann-off'
            Path(str(prefix)+'.json').write_text(json.dumps(dict(backend='cudann',profile_enabled=False,
                train_call_ns=100,internal_profile={'epoch_wall_ns':[45,45]},epochs=2)))
            db=sqlite3.connect(str(prefix)+'.sqlite')
            db.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT); '
                'CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT); '
                'CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER); '
                'CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER);')
            db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'cudaProfilerStart'),(2,'cudaProfilerStop'),
                (3,'cudaLaunchKernel'),(4,'cuLaunchKernel'),(5,'cudaDeviceSynchronize')])
            db.execute("INSERT INTO NVTX_EVENTS VALUES (100,200,'fnn.train.capture')")
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)',
                [(80,100,1,1),(200,205,2,1),(115,135,3,1),(175,195,5,1)])
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (120,130,4,1)'); db.commit(); db.close()
            with Path(str(prefix)+'-gpu_cuda_gpu_trace.csv').open('w') as fh:
                w=csv.writer(fh); w.writerow(['Start (ns)','Duration (ns)','Name'])
                w.writerows([(90,20,'[CUDA memcpy Device-to-Host]'),(110,30,'cublasLt::splitKreduce_kernel'),
                             (120,30,'[CUDA memcpy Host-to-Device]'),(180,10,'[CUDA memset]'),(210,10,'outside capture')])
            r=ns['analyze'](prefix)
            self.assertEqual((r['range_ns'],r['gpu_busy_ns'],r['gpu_overlap_ns'],r['gpu_idle_ns']),(100,60,20,40))
            self.assertEqual(r['host_api_thread_union_ns']['launch'],20)
            self.assertEqual(r['gpu_ns']['GEMM'],30)
            self.assertEqual(r['gpu_ns']['D2H'],10)

    def test_ncu_measured_flops_and_bytes(self):
        ncu=runpy.run_path(str(Path(__file__).with_name('ncu-table.py')))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent/'results/analysis') as tmp:
            p=Path(tmp)/'kernel.csv'
            p.with_suffix('.command.json').write_text(json.dumps(dict(backend='cudann',label='nvcc',blas='tiled',
                shapes_in_submission_order=[['NN',512,256,784]])))
            metrics={'gpu__time_duration.sum':(20,'nsecond'),'dram__bytes_read.sum':(100,'byte'),
                     'dram__bytes_write.sum':(100,'byte')}
            for op,value in [('ffma',20),('fadd',1),('fmul',2)]:
                metrics[f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum']=(value,'inst')
            with p.open('w') as fh:
                w=csv.writer(fh,quoting=csv.QUOTE_ALL); w.writerow(['ID','Kernel Name','Metric Name','Metric Unit','Metric Value'])
                for name,(value,unit) in metrics.items(): w.writerow([0,'gemm_k<float>',name,unit,value])
            r=ncu['summarize'](p)[0]
            self.assertEqual(r['measured_scalar_flops'],43)
            self.assertEqual(r['dram_bytes'],200)
            self.assertAlmostEqual(r['intensity'],.215)
            self.assertAlmostEqual(r['gflops'],2.15)

    def test_rocprof_range_units_and_incomplete_pass_alignment(self):
        module=runpy.run_path(str(Path(__file__).with_name('rocprof-table.py')))
        from gemm_shapes import MNIST
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent/'results/analysis') as tmp:
            root=Path(tmp)
            (root/'counter-units.json').write_text(json.dumps({'FETCH_SIZE':'KiB','WRITE_SIZE':'KiB'}))
            for group,count in [('memory',8),('lds-conflicts',7)]:
                p=root/'hipcc-tiled-mnist'/group;p.mkdir(parents=True)
                (p/'command.json').write_text(json.dumps(dict(backend='cudann',label='hipcc',blas='tiled',size='mnist',
                    group=group,shapes_in_submission_order=MNIST)))
                (p/'exit.txt').write_text('0\n')
                (p/'driver.json').write_text(json.dumps(dict(losses_finite=True,build_info={'version':'test'})))
                with (p/'run_marker_api_trace.csv').open('w') as fh:
                    w=csv.writer(fh);w.writerow(['Function','Start_Timestamp','End_Timestamp'])
                    w.writerow(['fnn.train.capture',1000000,1000000000])
                with (p/'run_counter_collection.csv').open('w') as fh:
                    w=csv.writer(fh);w.writerow(['Agent_Id','Dispatch_Id','Kernel_Name','Counter_Name','Counter_Value',
                        'Start_Timestamp','End_Timestamp','VGPR_Count','LDS_Block_Size'])
                    # The setup dispatch must not leak into the captured eight.
                    w.writerow([0,0,'gemm_k','FETCH_SIZE',999,1,100,32,2048])
                    for i in range(count):
                        counters=[('FETCH_SIZE',1),('WRITE_SIZE',2)] if group=='memory' else [('LDSBankConflict',3)]
                        for metric,value in counters:
                            w.writerow([0,i+1,'gemm_k',metric,value,2000000+i*1000000,2100000+i*1000000,32,2048])
            rows,failures,units=module['analyze'](root)
            mapped=[r for r in rows if r['shape_order_matched']]
            self.assertEqual(len(mapped),8)
            self.assertEqual(len(rows),15)
            self.assertTrue(failures)
            for r in mapped:
                self.assertEqual(r['measured_memory_bytes'],3072)
                self.assertNotIn('LDSBankConflict',r['counters'])
                self.assertEqual(r['duration_ns'],100000)
            self.assertIsNone(module['number']('NaN'))

    def test_workstation_gate_precedes_remote_actions(self):
        spec=importlib.util.spec_from_file_location('extra_control',Path(__file__).with_name('workstation-extras-control.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with patch.object(m.subprocess,'run',return_value=type('Result',(),{'returncode':0})()), patch.object(m,'ssh') as ssh:
            with self.assertRaises(RuntimeError): m.gate()
            ssh.assert_not_called()

    def test_public_conditions_preserve_policy_without_service_inventory(self):
        from extra_conditions import footer, public_bench_status, BACKGROUND_NOTE
        status = ('private-server 2026-11-04T01:51:19+01:00 boot_id=private-id\n'
                  'cpufreq: driver=intel_cpufreq no_turbo=1\n'
                  '  policies: 24 governor=performance min=3500000 max=3500000\n'
                  'kernel.numa_balancing=0 ksm.run=0\n'
                  'private-guest: running (booted)  hugepage guard: inactive\n'
                  'private-service running\n')
        safe = public_bench_status(status)
        self.assertNotIn('private-', safe)
        self.assertIn('min=3500000 max=3500000', safe)
        self.assertIn('kernel.numa_balancing=0', safe)
        self.assertEqual(public_bench_status(safe), safe)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent/'results/analysis') as tmp:
            p = Path(tmp)
            (p/'bench-mode-status.txt').write_text(status)
            (p/'background-host.txt').write_text('private-service private-image running\n')
            text = footer(p)
            self.assertNotIn('private-', text)
            self.assertIn(BACKGROUND_NOTE, text)
            self.assertIn('not dedicated to the benchmark', text)

    def test_public_host_identifier_and_csv_export(self):
        from fnnbench import sysinfo
        with patch.dict(os.environ, {'HOSTNAME':'private-server'}, clear=True), patch.object(sysinfo, '_read', return_value=None):
            self.assertEqual(sysinfo.machine_id(), 'host')
            self.assertEqual(sysinfo.container_info()['HOSTNAME'], 'host')
            os.environ['FNN_MACHINE_ID'] = 'ws-amd'
            self.assertEqual(sysinfo.container_info()['HOSTNAME'], 'ws-amd')
        r = row(sysinfo={'hostname':'private-server'})
        self.assertEqual(R.flatten(r)['host'], 'host')
        r['_machine'] = 'ws-amd'
        self.assertEqual(R.flatten(r)['host'], 'ws-amd')

    def test_dynamic_lds_allocation_is_not_static_symbol_size(self):
        m=runpy.run_path(str(Path(__file__).with_name('rocprof-table.py')))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent/'results/analysis') as tmp:
            p=Path(tmp)
            (p/'run_marker_api_trace.csv').write_text('Function,Start_Timestamp,End_Timestamp\nfnn.train.capture,100,200\n')
            (p/'run_counter_collection.csv').write_text('Agent_Id,Process_Id,Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp,VGPR_Count,LDS_Block_Size\n0,7,1,syclnn_gemm,Wavefronts,64,110,190,32,0\n')
            (p/'run_results.json').write_text(json.dumps({'rocprofiler-sdk-tool':[{'metadata':{'pid':7},
                'callback_records':{'counter_collection':[{'dispatch_data':{'dispatch_info':{'dispatch_id':1,'group_segment_size':2048}}}]}}]}))
            r=m['dispatches'](p)[0]
            self.assertEqual(r['static_lds_bytes'],0)
            self.assertEqual(r['lds_bytes'],2048)

    def test_roofline_rejects_unmatched_bandwidth_boundary(self):
        m=runpy.run_path(str(Path(__file__).with_name('extra-plots.py')))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent/'results/analysis') as tmp:
            out=Path(tmp)/'gemm';stream=Path(tmp)/'stream';out.mkdir();stream.mkdir()
            (out/'kernels.json').write_text(json.dumps([{'shape':['NN',512,256,784],'intensity':10,'gflops':1000}]))
            with self.assertRaises(ValueError):m['roofline'](Plt(),out,stream)
            (stream/'calibration.json').write_text(json.dumps({'counter_boundary':'DRAM','counter_bandwidth_median_GB_s':100}))
            with self.assertRaises(ValueError):m['roofline'](Plt(),out,stream)

    def test_sanitizer_uses_errors_not_display_limit(self):
        outcome=runpy.run_path(str(Path(__file__).with_name('sanitizer-cases.py')))['outcome']
        self.assertEqual(outcome('1 passed\nRACECHECK SUMMARY: 100 hazards displayed (122 errors, 1 warning)',86),('errors',122))
        self.assertEqual(outcome('1 passed\nRACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warning)',86),('warnings',0))


if __name__=='__main__':
    unittest.main()
