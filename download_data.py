"""Download SATSEG colour images and manual masks from its official Borealis release."""
import concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
API='https://borealisdata.ca'
def main():
    out=ROOT/'data/raw';out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/'data/source_metadata.json'
    if meta.exists(): d=json.loads(meta.read_text())
    else:
        d=json.load(urllib.request.urlopen(API+'/api/datasets/:persistentId/?persistentId=doi:10.5683/SP3/VDAN02',timeout=60))
        meta.write_text(json.dumps(d,indent=2))
    files=[f['dataFile'] for f in d['data']['latestVersion']['files'] if f['dataFile']['filename'].startswith(('clr_','mask_'))]
    def fetch(f):
        dest=out/f['filename']
        if dest.exists() and dest.stat().st_size==f['filesize']:return
        for attempt in range(4):
            try:
                with urllib.request.urlopen(API+'/api/access/datafile/'+str(f['id']),timeout=60) as r: b=r.read()
                if len(b)!=f['filesize']:raise ValueError('Incomplete file '+f['filename'])
                checksum=f.get('checksum',{})
                if checksum.get('type')=='MD5' and hashlib.md5(b).hexdigest()!=checksum['value']:raise ValueError('Checksum mismatch')
                dest.write_bytes(b);return
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        for i,_ in enumerate(ex.map(fetch,files),1):
            if i%20==0:print(f'{i}/{len(files)} files ready',flush=True)
    print('Downloaded',len(files),'files;',sum(f['filesize'] for f in files),'bytes')
if __name__=='__main__':main()
