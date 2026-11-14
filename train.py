"""Train two paired experiments; test data are never used for model selection."""
import argparse,csv,json,os,platform,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import ROOT,Scenes,prepare
from model import Segmenter
from metrics import update,summarize

def loss_fn(logits,y):
    ce=F.cross_entropy(logits,y,ignore_index=255)
    valid=y!=255;p=logits.softmax(1)[:,1]*valid;t=(y==1).float()
    dice=(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)
    return ce+0.5*(1-dice)

def evaluate(model,loader,device):
    model.eval();cm=np.zeros((2,2),dtype=np.int64);loss=0;n=0
    with torch.inference_mode():
        for x,y,_ in loader:
            x,y=x.to(device),y.to(device);z=model(x)
            loss+=float(loss_fn(z,y))*len(x);n+=len(x);update(cm,z.argmax(1),y)
    return {'loss':loss/n,**summarize(cm)}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=40);p.add_argument('--seed',type=int,default=42);p.add_argument('--size',type=int,default=128);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='auto');a=p.parse_args()
    torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    split=prepare(a.size);print('Split counts', {s:len(v) for s,v in split.items()},flush=True)
    config={**vars(a),'device':device,'learning_rate':0.001,'optimizer':'Adam','loss':'cross entropy + 0.5 foreground soft Dice loss','flip_probability':0.5,'checkpoint':'highest validation dataset-level mIoU; earliest tie','stopping':'fixed epoch budget','pretrained':False,'split_protocol':'spacecraft type held out; fixed groups in data.py'}
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'device':device,'cpu':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor(),'cpu_count':os.cpu_count(),'threads':a.threads,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    train=Scenes('train');val=Scenes('val')
    for name,use_skip in [('baseline',False),('skip',True)]:
        seed_all(a.seed);model=Segmenter(use_skip).to(device);opt=torch.optim.Adam(model.parameters(),lr=0.001)
        loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),num_workers=0)
        valid=DataLoader(val,batch_size=a.batch_size,num_workers=0)
        aug=torch.Generator().manual_seed(a.seed+1000);best=-1;rows=[];start=time.perf_counter()
        for epoch in range(1,a.epochs+1):
            model.train();total=0
            for x,y,_ in loader:
                flips=torch.rand(len(x),generator=aug)<0.5
                x[flips]=x[flips].flip(-1);y[flips]=y[flips].flip(-1)
                x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);loss=loss_fn(model(x),y);loss.backward();opt.step();total+=float(loss.detach())*len(x)
            ev=evaluate(model,valid,device)
            row={'epoch':epoch,'train_loss':total/len(train),'val_loss':ev['loss'],'val_miou':ev['miou'],'elapsed_seconds':time.perf_counter()-start};rows.append(row)
            if ev['miou']>best:
                best=ev['miou'];torch.save({'state_dict':model.state_dict(),'skip':use_skip,'epoch':epoch,'val_miou':best,'config':config},out/f'{name}.pt')
            with (out/f'{name}_log.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
            print(f'{name} epoch {epoch}/{a.epochs} train={row["train_loss"]:.4f} val_mIoU={ev["miou"]:.4f} elapsed={row["elapsed_seconds"]:.1f}s',flush=True)
        (out/f'{name}_training.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'best_val_miou':best,'parameters':sum(p.numel() for p in model.parameters())},indent=2))
if __name__=='__main__':main()
