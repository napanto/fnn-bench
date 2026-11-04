#!/usr/bin/env python3
"""Extract CUDA 12.9 Compute Sanitizer into the experiment directory, without installing it.

Use only when the fnn-cuda sandbox lacks the tool. Verify the NVIDIA package's
SHA256 against its HTTPS package index; no package scripts or root operation run.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

BASE='https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/'
VERSION='12.9.79-1'


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('directory',type=Path)
    out=ap.parse_args().directory; out.mkdir(parents=True,exist_ok=True)
    index=out/'nvidia-Packages.gz'
    if not index.exists():
        with urllib.request.urlopen(BASE+'Packages.gz',timeout=30) as r: index.write_bytes(r.read())
    for record in gzip.decompress(index.read_bytes()).decode().split('\n\n'):
        if not record.startswith('Package: cuda-sanitizer-12-9\n'): continue
        fields=dict(line.split(': ',1) for line in record.splitlines() if ': ' in line and not line.startswith(' '))
        if fields['Version']!=VERSION: continue
        (out/'sanitizer-package-record.txt').write_text(record+'\n')
        path=out/Path(fields['Filename']).name
        if not path.exists():
            with urllib.request.urlopen(BASE+fields['Filename'].removeprefix('./'),timeout=60) as r: path.write_bytes(r.read())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=fields['SHA256']: raise ValueError('package SHA256 mismatch')
        subprocess.run(['dpkg-deb','-x',str(path),str(out/'toolkit')],check=True)
        metadata={'package':fields['Package'],'version':VERSION,'url':BASE+fields['Filename'].removeprefix('./'),
                  'sha256':digest,'index':BASE+'Packages.gz','reason':'compute-sanitizer absent from existing fnn-cuda sandbox',
                  'action':'dpkg-deb -x only, no installation or package scripts'}
        (out/'tool-acquisition.json').write_text(json.dumps(metadata,indent=2)+'\n')
        binary=out/'toolkit/usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer'
        if not binary.is_file(): raise RuntimeError('Compute Sanitizer executable absent from package')
        # bin/compute-sanitizer is a launcher; use the executable beside its libraries.
        (out/'sanitizer-path.txt').write_text(str(binary.resolve())+'\n')
        version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True)
        (out/'sanitizer-version.txt').write_text(version.stdout+version.stderr)
        print(version.stdout,end='')
        return
    raise RuntimeError('requested CUDA 12.9 package absent from NVIDIA index')


if __name__=='__main__': main()
