#!/usr/bin/env python3
"""Download, verify and extract the immutable recovered-expert supplement."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

NAME='SCFusNet-recovered-experts-20261018.zip'
SIZE=37467351
SHA256='1fa2f1978a6d7182022faab2a469e4f39de55d510e3fed2489c1adc4e53aa618'
URLS={
    'zenodo':'https://zenodo.org/records/23446670/files/'+NAME+'?download=1',
    'github':'https://github.com/S1reN-hub/SCFusNet/releases/download/experts-20261018/'+NAME,
}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source',choices=tuple(URLS),default='zenodo')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    archive=args.output_dir/NAME
    if not archive.exists() or archive.stat().st_size!=SIZE or digest(archive)!=SHA256:
        temporary=archive.with_suffix('.zip.part')
        request=urllib.request.Request(URLS[args.source],headers={'User-Agent':'SCFusNet-reproduction/20261018'})
        with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as out:
            while block:=response.read(1024*1024):out.write(block)
        if temporary.stat().st_size!=SIZE or digest(temporary)!=SHA256:
            raise ValueError('Downloaded model archive failed size/SHA-256 verification; it was not extracted.')
        temporary.replace(archive)
    root=args.output_dir.resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (root/item.filename).resolve().is_relative_to(root):
                raise ValueError('Unsafe archive member path')
        bad=z.testzip()
        if bad is not None:raise ValueError('Archive CRC failure: '+bad)
        z.extractall(root)
    print('Verified model bundle: '+str(root/'SCFusNet-recovered-experts'))

if __name__=='__main__':main()
