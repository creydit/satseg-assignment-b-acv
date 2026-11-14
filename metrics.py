import numpy as np
import torch
def update(cm,pred,target):
    valid=target!=255
    cm+=torch.bincount((2*target[valid]+pred[valid]).cpu(),minlength=4).reshape(2,2).numpy()
def summarize(cm):
    cm=np.asarray(cm);tp=cm.diagonal();union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(2,np.nan),where=union>0)
    precision=float(tp[1]/max(1,cm[:,1].sum()));recall=float(tp[1]/max(1,cm[1,:].sum()))
    return {'background_iou':float(iou[0]),'spacecraft_iou':float(iou[1]),'miou':float(np.nanmean(iou)),'pixel_accuracy':float(tp.sum()/max(1,cm.sum())),'precision':precision,'recall':recall,'dice':float(2*tp[1]/max(1,cm[:,1].sum()+cm[1,:].sum())),'confusion_matrix':cm.tolist()}
