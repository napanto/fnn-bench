#!/usr/bin/env python3
"""Extract matched nvprof or legacy Pascal-capable NCU without installing packages."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

PACKAGES = {
    "nvprof": ("ubuntu2404", "cuda-nvprof-12-9", "12.9.79-1", "nvprof"),
    "ncu-legacy": ("ubuntu1804", "nsight-compute-2019.5.0", "2019.5.0.14-1", "nv-nsight-cu-cli"),
}


def download(url, path, deadline):
    with urllib.request.urlopen(url, timeout=30) as response, path.open("wb") as output:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("tool acquisition time budget exhausted")
            block = response.read(1024*1024)
            if not block: break
            output.write(block)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tool", choices=PACKAGES); ap.add_argument("out", type=Path)
    ap.add_argument("--budget-seconds", type=int, default=1200)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    release,package,version,exe=PACKAGES[a.tool]
    base=f"https://developer.download.nvidia.com/compute/cuda/repos/{release}/x86_64/"
    deadline=time.monotonic()+a.budget_seconds
    index=a.out/f"{release}-Packages.gz"
    if not index.exists(): download(base+"Packages.gz",index,deadline)
    matches=[]
    for record in gzip.decompress(index.read_bytes()).decode().split("\n\n"):
        if not record.startswith("Package: "+package+"\n"): continue
        fields=dict(line.split(": ",1) for line in record.splitlines() if ": " in line and not line.startswith(" "))
        if fields["Version"]==version: matches.append((fields,record))
    if len(matches)!=1: raise RuntimeError(f"Requested official package unavailable: {package} {version}")
    fields,record=matches[0]
    path=a.out/Path(fields["Filename"]).name
    if not path.exists(): download(base+fields["Filename"].removeprefix("./"),path,deadline)
    digest=hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda:fh.read(1024*1024),b""): digest.update(block)
    if digest.hexdigest()!=fields["SHA256"]: raise ValueError("download SHA256 mismatch")
    target=a.out/a.tool
    if not target.exists(): subprocess.run(["dpkg-deb","-x",str(path),str(target)],check=True,timeout=120)
    files=sorted({p.resolve() for p in target.rglob(exe) if p.is_file()})
    if not files: raise RuntimeError("profiler executable not present after extraction")
    candidates=[p for p in files if 'x64' in str(p)] or files
    binary=candidates[0]
    (a.out/f"{a.tool}-path.txt").write_text(str(binary)+"\n")
    metadata=dict(package=package,version=version,url=base+fields["Filename"].removeprefix("./"),
                  sha256=digest.hexdigest(),index_url=base+"Packages.gz",executable=str(binary),
                  action="private dpkg-deb -x; no installation scripts or host configuration changes")
    (a.out/f"{a.tool}-acquisition.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (a.out/f"{a.tool}-package-record.txt").write_text(record+"\n")
    result=subprocess.run([str(binary),"--version"],capture_output=True,text=True,timeout=30)
    (a.out/f"{a.tool}-version.txt").write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr,end="")
    raise SystemExit(result.returncode)


if __name__=="__main__": main()
