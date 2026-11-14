"""Check pairing, valid labels, identical initialisation, known IoU and tiny-data learning."""
import json
import numpy as np
import torch
from data import prepare,Scenes,ROOT
from model import Segmenter
from train import loss_fn,seed_all
from metrics import update,summarize

def main():
    torch.set_num_threads(2);split=prepare();seed_all(42);a=Segmenter(False);seed_all(42);b=Segmenter(True)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    cm=np.zeros((2,2),np.int64);update(cm,torch.tensor([0,1,0,1]),torch.tensor([0,1,1,255]));assert cm.tolist()==[[1,0],[1,1]];assert summarize(cm)['miou']==.5
    d=Scenes('train');x=torch.stack([d[i][0] for i in range(4)]);y=torch.stack([d[i][1] for i in range(4)])
    opt=torch.optim.Adam(b.parameters(),lr=.001);start=float(loss_fn(b(x),y).detach())
    for _ in range(40):
        opt.zero_grad();loss=loss_fn(b(x),y);loss.backward();opt.step()
    end=float(loss_fn(b(x),y).detach());assert end<start*.8,(start,end)
    result={'pair_count':sum(map(len,split.values())),'split_counts':{s:len(v) for s,v in split.items()},'initial_weights_identical':True,'known_iou_test':'passed','tiny_training_images':4,'tiny_training_steps':40,'initial_loss':start,'final_loss':end,'passed':True}
    (ROOT/'results/sanity_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
