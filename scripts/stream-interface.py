#!/usr/bin/env python3
"""Calibrate the GL2 external byte/time boundary with cache-resident triad kernels."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    a=ap.parse_args();p=a.directory;groups={}
    for metric in ('FETCH_SIZE','WRITE_SIZE'):
        rows=[]
        for f in (p/metric).rglob('*counter_collection.csv'):
            with f.open() as h:rows+=list(csv.DictReader(h))
        rows.sort(key=lambda r:int(r['Dispatch_Id']))
        if len(rows)!=8 or not all('stream_op<3>' in r['Kernel_Name'] for r in rows):
            raise ValueError('expected five warm-ups and three triad samples')
        groups[metric]=[{'bytes':float(r['Counter_Value'])*1024,
                        'duration_ns':int(r['End_Timestamp'])-int(r['Start_Timestamp'])} for r in rows[5:]]
    read=st.median(r['bytes'] for r in groups['FETCH_SIZE']);write=st.median(r['bytes'] for r in groups['WRITE_SIZE'])
    duration=st.median([st.median(r['duration_ns'] for r in groups[k]) for k in groups])
    if min(read,write,duration)<=0:
        raise ValueError('zero/invalid read, write or duration cannot calibrate this interface')
    data={'counter_boundary':'GL2 external interface','working_set_MiB':72,
          'read_bytes':read,'write_bytes':write,'kernel_duration_ns':duration,
          'counter_bandwidth_median_GB_s':(read+write)/duration,
          'logical_triad_bandwidth_median_GB_s':next(r['median_GB_s'] for r in json.loads((p/'summary.json').read_text()) if r['operation']=='triad'),
          'method':'Five warm-ups then three recorded triad kernels per counter group. Read/write groups are separate deterministic repeats, as for the GEMM points. The reference uses kernel timestamps, not bracketing event durations.',
          'limits':'Cache-resident reference for GL2 request bytes, not a DRAM bandwidth ceiling. 384 MiB STREAM is separate. Firmware-selected clocks and other GPU contexts can affect results.',
          'samples':groups}
    (p/'calibration.json').write_text(json.dumps(data,indent=2)+'\n')
    with (p/'README.md').open('a') as f:
        f.write(f"\n## Interface-matched roofline calibration\n\nThe 72 MiB working set exceeds L2 and fits within the RX 7900 XTX's 96 MiB Infinity Cache. "
                f"The same FETCH_SIZE/WRITE_SIZE counters as the GEMM runs report median read/write traffic {read/2**20:.3f}/{write/2**20:.3f} MiB; "
                f"the median of the two groups' kernel-duration medians is {duration/1e3:.3f} us. "
                f"Their matched reference is **{data['counter_bandwidth_median_GB_s']:.2f} GB/s**. "
                f"The ordinary event-timed logical-byte triad result is {data['logical_triad_bandwidth_median_GB_s']:.2f} GB/s. "
                "These are different measurement boundaries. This calibration supports a GL2 external-interface roofline, not a DRAM-only roofline; "
                "the independent 384 MiB probe remains the large-buffer bandwidth experiment. Raw samples and the formula are in calibration.json.\n")


if __name__=='__main__':main()
