"""Exact held-out evaluation, model-only latency, plots, and standalone visual report."""
import argparse,csv,json,time,base64
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from data import ROOT,Scenes
from model import Segmenter
from metrics import update,summarize
COLORS=['#64748b','#2563eb']
def load_model(name,device):
    c=torch.load(ROOT/'results'/f'{name}.pt',map_location=device,weights_only=True)
    m=Segmenter(c['skip']).to(device);m.load_state_dict(c['state_dict']);m.eval();return m,c

def architecture(out):
    fig,ax=plt.subplots(figsize=(9,9));ax.set(xlim=(0,10),ylim=(-.5,10));ax.axis('off')
    labels=['RGB input  128 x 128 x 3','E1  two Conv 3x3 + GN + ReLU\n128 x 128 x 16','E2  MaxPool 2 + double conv\n64 x 64 x 32','Bottleneck  MaxPool 2 + double conv\n32 x 32 x 64','D2  bilinear upsample + Conv 1x1\noptional ADD E2, then double conv\n64 x 64 x 32','D1  bilinear upsample + Conv 1x1\noptional ADD E1, then double conv\n128 x 128 x 16','Conv 1x1  16 to 2\nargmax gives spacecraft/background']
    ys=[9,7.65,6.3,4.95,3.55,2.05,.55]
    for i,(label,y) in enumerate(zip(labels,ys)):
        ax.add_patch(FancyBboxPatch((2.3,y-.4),5.4,.82,boxstyle='round,pad=.12',fc='#eff6ff' if i not in [0,6] else '#e2e8f0',ec='#334155'))
        ax.text(5,y,label,ha='center',va='center',fontsize=10)
        if i:ax.annotate('',xy=(5,y+.54),xytext=(5,ys[i-1]-.54),arrowprops={'arrowstyle':'->','color':'#334155'})
    for x,y1,y2,label in [(8.5,6.3,3.55,'E2 skip'),(1.1,7.65,2.05,'E1 skip')]:
        side=7.8 if x>5 else 2.2
        ax.plot([side,x,x,side],[y1,y1,y2,y2],'--',color='#2563eb',lw=1.8)
        ax.annotate('',xy=(side,y2),xytext=(x,y2),arrowprops={'arrowstyle':'->','color':'#2563eb'})
        ax.text(x+.15 if x>5 else x-.15,(y1+y2)/2,label,rotation=90,va='center',ha='left' if x>5 else 'right',color='#2563eb')
    ax.set_title('Parameter-matched encoder-decoder\nDashed additions enabled only in the skip model',fontsize=15,pad=16)
    fig.savefig(out/'architecture.png',dpi=150,bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(a.threads);device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    torch.use_deterministic_algorithms(True);out=ROOT/'results';data=Scenes('test');metrics={};preds={};per=[]
    for name in ['baseline','skip']:
        model,c=load_model(name,device);cm=np.zeros((2,2),np.int64);preds[name]=[]
        with torch.inference_mode():
            for x,y,key in data:
                pred=model(x[None].to(device)).argmax(1)[0].cpu();preds[name].append(pred.numpy());update(cm,pred,y)
                one=np.zeros((2,2),np.int64);update(one,pred,y);s=summarize(one);per.append({'model':name,'id':key,**{k:v for k,v in s.items() if k!='confusion_matrix'}})
            sample=data[0][0][None].to(device)
            for _ in range(20):model(sample)
            durations=[]
            for _ in range(100):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();model(sample)
                if device=='cuda':torch.cuda.synchronize()
                durations.append((time.perf_counter()-start)*1000)
        metrics[name]={**summarize(cm),'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_epoch':c['epoch'],'validation_miou':c['val_miou'],'latency_ms_mean':float(np.mean(durations)),'latency_ms_median':float(np.median(durations)),'latency_ms_p95':float(np.percentile(durations,95)),'training_seconds':json.loads((out/f'{name}_training.json').read_text())['seconds']}
    protocol={'device':device,'threads':a.threads,'batch_size':1,'input_shape':list(sample.shape),'precision':'float32','mode':'eval + inference_mode','warmup':20,'repetitions':100,'includes':'model forward pass only; excludes disk IO, preprocessing, transfer, argmax and rendering','test_images':len(data),'metric_aggregation':'sum confusion matrices across all valid test pixels; exclude padding ID 255','threshold':'argmax of 2 logits'}
    (out/'metrics.json').write_text(json.dumps({'models':metrics,'latency_protocol':protocol},indent=2))
    with (out/'per_image_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model']+[k for k in metrics['baseline'] if k!='confusion_matrix']);w.writeheader();w.writerows({'model':n,**{k:v for k,v in m.items() if k!='confusion_matrix'}} for n,m in metrics.items())
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':11})
    fig,axs=plt.subplots(1,2,figsize=(11,3.8))
    for name,color in zip(metrics,COLORS):
        rows=list(csv.DictReader((out/f'{name}_log.csv').open()));ep=[int(r['epoch']) for r in rows]
        axs[0].plot(ep,[float(r['train_loss']) for r in rows],label=name+' train',color=color)
        axs[0].plot(ep,[float(r['val_loss']) for r in rows],label=name+' validation',color=color,ls='--')
        axs[1].plot(ep,[float(r['val_miou'])*100 for r in rows],label=name,color=color)
    axs[0].set(title='Training and validation loss',xlabel='Epoch',ylabel='CE + 0.5 Dice loss');axs[1].set(title='Checkpoint selection on validation only',xlabel='Epoch',ylabel='Validation mIoU (%)')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    keys=['background_iou','spacecraft_iou','miou'];labels=['Background IoU','Spacecraft IoU','mIoU'];fig,ax=plt.subplots(figsize=(9,4))
    for i,(n,c) in enumerate(zip(metrics,COLORS)):
        bars=ax.bar(np.arange(3)+(i-.5)*.36,[metrics[n][k]*100 for k in keys],width=.36,label=n,color=c)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
    ax.set(xticks=np.arange(3),xticklabels=labels,ylim=(0,110),ylabel='Test score (%)',title='Held-out spacecraft types');ax.legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8,3.8))
    for ax,n in zip(axs,metrics):
        cm=np.array(metrics[n]['confusion_matrix']);norm=cm/np.maximum(1,cm.sum(1,keepdims=True));ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{cm[i,j]:,}\n{norm[i,j]:.1%}',ha='center',va='center',color='white' if norm[i,j]>.6 else 'black',fontsize=9)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Background','Spacecraft'],yticklabels=['Background','Spacecraft'],xlabel='Predicted',ylabel='True',title=n)
    fig.tight_layout();fig.savefig(out/'confusion_matrices.png',dpi=160);plt.close(fig)
    scores={r['id']:r['spacecraft_iou'] for r in per if r['model']=='skip'}
    order=sorted(range(len(data)),key=lambda i:scores[data.ids[i]])
    chosen=order[:3]+list(reversed(order[-3:]));fig,axs=plt.subplots(6,4,figsize=(10,14))
    for row,i in enumerate(chosen):
        x,y,key=data[i];valid=y.numpy()!=255;rgb=x.permute(1,2,0).numpy();truth=np.ma.masked_where(~valid,y.numpy())
        axs[row,0].imshow(rgb);axs[row,0].set_title(key+'\n'+('Lowest skip IoU' if row<3 else 'Highest skip IoU'),fontsize=9)
        axs[row,1].imshow(truth,vmin=0,vmax=1,cmap='gray');axs[row,1].set_title('Manual mask',fontsize=9)
        for col,n in [(2,'baseline'),(3,'skip')]:
            axs[row,col].imshow(np.ma.masked_where(~valid,preds[n][i]),vmin=0,vmax=1,cmap='gray');s=next(r['spacecraft_iou'] for r in per if r['id']==key and r['model']==n);axs[row,col].set_title(f'{n}: IoU {s:.3f}',fontsize=9)
        for ax in axs[row]:ax.axis('off');ax.set_facecolor('#cbd5e1')
    fig.tight_layout();fig.savefig(out/'predictions.png',dpi=130,bbox_inches='tight');plt.close(fig)
    architecture(out)
    delta=(metrics['skip']['miou']-metrics['baseline']['miou'])*100
    interpretation=f'Additive skips changed test mIoU by {delta:+.2f} percentage points. This is a single-seed, small-data comparison on held-out spacecraft types; the result does not establish general superiority. Selected checkpoints use validation only.'
    rows=''.join('<tr><td>'+n+'</td>'+''.join(f'<td>{m[k]*100:.2f}%</td>' for k in keys)+f'<td>{m["parameters"]:,}</td><td>{m["latency_ms_median"]:.2f} ms</td></tr>' for n,m in metrics.items())
    def embed(fn):return 'data:image/png;base64,'+base64.b64encode((out/fn).read_bytes()).decode()
    images=''.join(f'<h2>{title}</h2><img src="{embed(fn)}" alt="{title}">' for title,fn in [('Architecture','architecture.png'),('Training evidence','learning_curves.png'),('Test comparison','comparison.png'),('Errors by class','confusion_matrices.png'),('Failures and successes','predictions.png')])
    html=f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SATSEG experiment results</title><style>body{{font:17px/1.6 system-ui;background:#f1f5f9;color:#172033;margin:0}}main{{max-width:1000px;margin:auto;background:white;padding:45px}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:45px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;border-bottom:1px solid #cbd5e1;text-align:left}}th{{background:#e2e8f0}}small{{color:#475569}}@media(max-width:600px){{main{{padding:18px}}table{{font-size:11px}}td,th{{padding:5px}}}}</style><main><small>ADVANCED COMPUTER VISION · REPRODUCIBLE SEGMENTATION</small><h1>Do skip connections help segment spacecraft?</h1><p>{interpretation}</p><p>100 SATSEG colour scenes. 65 training / 19 validation / 16 test. RGB and grayscale duplicates are not mixed across splits. Whole spacecraft types are held out.</p><table><tr><th>Model</th><th>Background IoU</th><th>Spacecraft IoU</th><th>mIoU</th><th>Parameters</th><th>Median latency</th></tr>{rows}</table>{images}<h2>Measurement conditions and limitations</h2><p>{json.dumps(protocol)}</p><p>One seed; small test set; a single validation spacecraft type; possible unrecognised related source photographs; letterboxing to 128 pixels may erase thin structures. Latency is measured on this execution environment and is not a flight-computer benchmark.</p><p>Data: Shi and Ulrich, SATSEG, <a href="https://doi.org/10.5683/SP3/VDAN02">official dataset</a>. CC BY-NC-SA 4.0. Image IDs appear above. Failures and successes are selected by skip-model foreground IoU, not manually cherry-picked.</p></main></html>'''
    (out/'dashboard.html').write_text(html);(out/'interpretation.txt').write_text(interpretation)
    print(json.dumps(metrics,indent=2));print(interpretation)
if __name__=='__main__':main()
