#!/usr/bin/env python3
"""Summarise rocprofv3 dispatch counters within the named post-warm-up interval."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as st
from extra_conditions import write_report
from gemm_shapes import is_gemm, TARGETS


def number(value):
    try:
        value = float(str(value).replace(",", ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def interval(directory):
    ranges = []
    for path in directory.rglob("*marker_api_trace.csv"):
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("Function") == "fnn.train.capture":
                    start, end = number(r.get("Start_Timestamp")), number(r.get("End_Timestamp"))
                    if start is not None and end is not None and end > start:
                        ranges.append((start, end))
    if len(ranges) != 1:
        raise ValueError(f"expected one post-warm-up ROCTx range, found {len(ranges)}")
    return ranges[0]


def dispatches(directory):
    lo, hi = interval(directory)
    dispatch_info = {}
    for path in directory.glob('*_results.json'):
        for data in json.loads(path.read_text()).get('rocprofiler-sdk-tool', []):
            pid = data.get('metadata', {}).get('pid')
            for entry in data.get('callback_records', {}).get('counter_collection', []):
                info = entry.get('dispatch_data', {}).get('dispatch_info') or {}
                if 'dispatch_id' in info:
                    dispatch_info[(str(pid), int(info['dispatch_id']))] = info
    kernels = {}
    for path in directory.rglob("*counter_collection.csv"):
        with path.open() as fh:
            for r in csv.DictReader(fh):
                start, end = number(r.get("Start_Timestamp")), number(r.get("End_Timestamp"))
                if start is None or end is None or end <= start or not lo <= start < hi or not is_gemm(r.get("Kernel_Name", "")):
                    continue
                key = (r.get("Agent_Id"), int(r["Dispatch_Id"]))
                info = dispatch_info.get((r.get('Process_Id'), int(r['Dispatch_Id'])), {})
                k = kernels.setdefault(key, dict(kernel=r["Kernel_Name"], dispatch=int(r["Dispatch_Id"]),
                    duration_ns=end-start, counters={}, counter_instances={}, vgprs=number(r.get("VGPR_Count")), sgprs=number(r.get("SGPR_Count")),
                    lds_bytes=info.get('group_segment_size', number(r.get("LDS_Block_Size"))),
                    static_lds_bytes=number(r.get("LDS_Block_Size")), grid=number(r.get("Grid_Size")),
                    grid_dimensions=info.get('grid_size'), workgroup_dimensions=info.get('workgroup_size'),
                    workgroup=number(r.get("Workgroup_Size")), start_ns=start, end_ns=end))
                metric = r.get("Counter_Name")
                value = number(r.get("Counter_Value"))
                if metric and value is not None:
                    # Do not sum percentages or silently choose a hardware instance.
                    instances = k["counter_instances"].setdefault(metric, [])
                    instances.append({"value": value, "dimensions": {key: val for key, val in r.items() if "dim" in key.lower()}})
                    k["counters"][metric] = value if len(instances) == 1 else None
    return sorted(kernels.values(), key=lambda k: k["dispatch"])


def analyze(root):
    grouped = defaultdict(list); failures = []
    for manifest in sorted(root.glob("*/*/command.json")):
        directory = manifest.parent; meta = json.loads(manifest.read_text())
        status = directory / "exit.txt"
        if not status.exists() or status.read_text().strip() != "0":
            failures.append(f"{directory.relative_to(root)}: incomplete/nonzero exit; see run.log")
            continue
        try:
            driver = json.loads((directory/"driver.json").read_text())
            if not driver.get("losses_finite"):
                raise ValueError("nonfinite losses")
            kernels = dispatches(directory)
            shapes = meta["shapes_in_submission_order"]
            matched = len(kernels) == len(shapes)
            if matched and meta['blas'] == 'tiled' and meta['backend'] in ('cudann', 'syclnn'):
                matched = all(k['grid'] is None or k['grid'] == ((s[1]+15)//16)*((s[2]+15)//16)*256
                              for k, s in zip(kernels, shapes))
            if not matched:
                failures.append(f"{directory.relative_to(root)}: GEMM count/order or tiled launch geometry mismatch ({len(kernels)} main GEMMs, expected {len(shapes)}); shape labels withheld")
            for i, k in enumerate(kernels):
                k.update(source=str(directory.relative_to(root)), backend=meta["backend"], label=meta["label"],
                         blas=meta["blas"], size=meta["size"], group=meta["group"],
                         shape=shapes[i] if matched else None, shape_order_matched=matched,
                         build_info=driver.get("build_info"))
                build = json.dumps(driver.get("build_info"), sort_keys=True)
                identity = str(i) if matched else str(directory.relative_to(root))+f":{i}"
                grouped[(meta["label"], meta["blas"], meta["size"], build, identity)].append(k)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            failures.append(f"{directory.relative_to(root)}: {exc}")
    units_file = root/"counter-units.json"
    units = json.loads(units_file.read_text()) if units_file.exists() else {}
    output = []
    for key, runs in sorted(grouped.items()):
        signatures={(r['kernel'],r['grid'],r['workgroup'],r['lds_bytes'],r['vgprs']) for r in runs}
        if len(signatures)>1:
            failures.append(f"{key[0]}/{key[1]}/{key[2]} dispatch slot {key[-1]}: counter passes used different kernels or launch resources; not joined")
            continue
        memory = next((r for r in runs if r["group"] == "memory"), None)
        memory_read = next((r for r in runs if r["group"] == "memory-read"), None)
        memory_write = next((r for r in runs if r["group"] == "memory-write"), None)
        representative = memory or memory_read or runs[0]
        counters = {name: value for r in runs for name, value in r["counters"].items()}
        row = {k: v for k, v in representative.items() if k not in ("counters", "group", "source")}
        row.update(counters=counters, passes=runs, duration_source=representative["source"],
                   flop_basis="mathematical GEMM 2*M*N*K, not retired instructions")
        shape = row.get("shape")
        flops = 2*shape[1]*shape[2]*shape[3] if shape else None
        traffic = None
        if memory and units.get("FETCH_SIZE") == units.get("WRITE_SIZE") == "KiB":
            vals = memory["counters"]
            if vals.get("FETCH_SIZE") is not None and vals.get("WRITE_SIZE") is not None:
                traffic = 1024*(vals["FETCH_SIZE"]+vals["WRITE_SIZE"])
        elif memory_read and memory_write and units.get("FETCH_SIZE") == units.get("WRITE_SIZE") == "KiB":
            read = memory_read["counters"].get("FETCH_SIZE")
            write = memory_write["counters"].get("WRITE_SIZE")
            if read is not None and write is not None:
                traffic = 1024*(read+write)
                row["duration_ns"] = st.median([memory_read["duration_ns"], memory_write["duration_ns"]])
                row["duration_source"] = [memory_read["source"], memory_write["source"]]
                row["memory_collection"] = "two deterministic repeats (read and write); median of their kernel durations"
        row.update(mathematical_flops=flops, measured_memory_bytes=traffic,
                   intensity=flops/traffic if flops and traffic else None,
                   gflops=flops/row["duration_ns"] if flops and row["duration_ns"] else None)
        output.append(row)
    return output, failures, units


def fmt(value):
    return f"{value:.4g}" if value is not None else "not available"


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("directory", type=Path)
    a = ap.parse_args(); rows, failures, units = analyze(a.directory)
    (a.directory/"kernels.json").write_text(json.dumps(rows, indent=2)+"\n")
    out = ["# RX 7900 XTX GEMM counters\n\n",
           "Existing cudann HIP, AdaptiveCpp and amdclang wheels, float, blas=tiled; cudann/rocBLAS comparison. "
           "MNIST one batch of 256 and a large W4 batch after warm-up. Commands/builds are in each command.json and driver.json. "
           "Counter names were selected from this device's saved rocprofv3 --list-avail output. "
           "Each counter group is a separate process. The global ROCTx pause/resume and named range exclude setup/warm-up.\n\n",
           "Shape labels follow the source submission order only when eight main GEMMs are observed. "
           "A changed vendor decomposition remains unlabelled. HIP/SYCL tiled grid sizes also check the association. "
           "Registers come from kernel metadata; LDS uses the JSON dispatch allocation, including dynamic local memory. "
           "The CSV LDS_Block_Size alone is static and reads zero for AdaptiveCpp even when its launch allocates local memory. "
           "Read/write traffic requires separate deterministic repeats because the combined counter request exceeds hardware slots. "
           "The roofline uses their summed traffic and median kernel duration; both original durations remain in kernels.json. "
           "FLOPs are the mathematical 2MNK count, not an instruction counter. FETCH_SIZE/WRITE_SIZE are converted only "
           "when counter-units.json records verified units and provenance. Counter boundary is described there.\n\n",
           "| model | BLAS | shape (op M N K) | VGPRs | LDS bytes | LDS conflicts % | SQ LDS instructions | VALUInsts | occupancy % | waves | GL2 external bytes | time us | FLOP/byte | GFLOP/s |\n",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    validity_path = a.directory/'counter-validity.json'
    if validity_path.exists():
        validity = json.loads(validity_path.read_text())
        out.insert(1, "**Diagnostic capture, not the final counter result.** "+validity['reason']+
                   " The zero fields listed in counter-validity.json are unusable readouts, not physical zeros. "
                   "Use the separately recorded stable-power-state experiment for those metrics.\n\n")
    for r in rows:
        if tuple(r.get('shape') or []) not in TARGETS: continue
        shape = " ".join(map(str,r["shape"])) if r["shape"] else "unverified"
        c = r["counters"]
        pick = lambda *names: next((c[n] for n in names if n in c), None)
        values = [r["vgprs"], r["lds_bytes"], pick("LDSBankConflict","SQ_LDS_BANK_CONFLICT"),
                  pick("FNN_LDS_BASE","LDSInsts","FNN_LDS_INSTS","SQ_INSTS_LDS"), pick("VALUInsts"), pick("OccupancyPercent"), pick("Wavefronts","SQ_WAVES"),
                  r["measured_memory_bytes"], r["duration_ns"]/1e3, r["intensity"], r["gflops"]]
        out.append(f"| {r['label']} | {r['blas']} | {shape} | "+" | ".join(fmt(v) for v in values)+" |\n")
    out += ["\nLaunch geometry for the requested kernels (OpenMP teams may iterate over multiple output tiles):\n\n",
            "| model | BLAS | shape | workgroups | threads/group | static LDS B | dispatch LDS B |\n|---|---|---|---|---|---|---|\n"]
    for r in rows:
        if tuple(r.get('shape') or []) not in TARGETS: continue
        groups = r['grid']/r['workgroup'] if r.get('grid') and r.get('workgroup') else None
        out.append(f"| {r['label']} | {r['blas']} | {' '.join(map(str,r['shape']))} | {fmt(groups)} | {fmt(r['workgroup'])} | {fmt(r['static_lds_bytes'])} | {fmt(r['lds_bytes'])} |\n")
    out += [f"\nThe main table selects the four requested shapes; all {len(rows)} main-GEMM records and their passes are in kernels.json. "
            "VALU utilisation/busy is not exposed on this gfx1100 setup. VALUInsts is the tool's average instruction count per work-item, not a utilisation percentage. All raw counters are retained in kernels.json and the CSVs. "
            "Do not infer occupancy from a waves count alone. Kernel-7.3 timing regressions and background display activity "
            "limit timing interpretation; use this experiment primarily to compare counter patterns.\n"]
    out += ["\nOpenMP captures set OMP_TOOL=disabled to avoid an OMPT initialisation re-entry from OpenBLAS; "
            "hardware dispatch tracing/counting remains enabled. The compiled kernel and library are unchanged.\n"]
    if units: out += ["\nMemory counter definitions/units:\n\n```json\n",json.dumps(units,indent=2),"\n```\n"]
    if failures: out += ["\nIncomplete or ambiguous captures:\n\n"]+["- "+s+"\n" for s in failures]
    write_report(a.directory, "".join(out))


if __name__ == "__main__": main()
