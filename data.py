"""Validate official pairs and create a documented spacecraft-type-held-out split."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parent
GROUPS={'train':['ISS','cubesat','cygnus','dragon','htv','hubble','shenzhou','tesla'], 'val':['radarsat'], 'test':['mir','orbitalexpress','sentinel','shuttle','soyuz']}
def canonical(s):return s.replace('soyouz','soyuz')
def prepare(size=128):
    raw=ROOT/'data/raw';processed=ROOT/'data/processed';processed.mkdir(exist_ok=True)
    masks={canonical(p.stem[5:]):p for p in raw.glob('mask_*.png')}
    split={k:[] for k in GROUPS};audit=[]
    for p in sorted(raw.glob('clr_*.jpg')):
        key=canonical(p.stem[4:]);group=key.rsplit('_',1)[0]
        target=[s for s,gs in GROUPS.items() if group in gs]
        if len(target)!=1:raise ValueError('Unknown target '+key)
        mpath=masks[key]
        with Image.open(p) as im:im=im.convert('RGB')
        with Image.open(mpath) as m:m=m.convert('L')
        if im.size!=m.size:raise ValueError('Misaligned dimensions '+key)
        vals=np.unique(np.asarray(m))
        if not set(vals).issubset({0,255}):raise ValueError('Unexpected mask IDs '+str(vals))
        w,h=im.size;scale=size/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));left,top=(size-nw)//2,(size-nh)//2
        img=Image.new('RGB',(size,size));img.paste(im.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
        label=Image.fromarray((np.asarray(m)>0).astype('uint8')).resize((nw,nh),Image.Resampling.NEAREST)
        mask=Image.new('L',(size,size),255);mask.paste(label,(left,top))
        img.save(processed/(key+'_image.png'));mask.save(processed/(key+'_mask.png'))
        split[target[0]].append(key)
        audit.append({'id':key,'split':target[0],'original_size':[w,h],'raw_values':vals.tolist(),'foreground_fraction':float((np.asarray(label)==1).mean()),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256(mpath.read_bytes()).hexdigest()})
    assert len(audit)==100, f'Expected 100 unique colour scenes, got {len(audit)}'
    for s,ids in split.items():(ROOT/'data'/f'{s}.txt').write_text('\n'.join(ids)+'\n')
    (ROOT/'data/audit.json').write_text(json.dumps({'size':size,'groups':GROUPS,'records':audit},indent=2))
    return split
class Scenes(Dataset):
    def __init__(self,split):
        self.ids=(ROOT/'data'/f'{split}.txt').read_text().splitlines();self.items=[]
        for key in self.ids:
            x=np.asarray(Image.open(ROOT/'data/processed'/f'{key}_image.png').convert('RGB'),dtype=np.float32)/255
            y=np.asarray(Image.open(ROOT/'data/processed'/f'{key}_mask.png'),dtype=np.int64)
            self.items.append((torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y.copy())))
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return *self.items[i],self.ids[i]
if __name__=='__main__':print({k:len(v) for k,v in prepare().items()})
