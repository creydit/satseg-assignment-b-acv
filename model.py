"""Parameter-matched encoder-decoder with optional additive skip fusion."""
import torch
from torch import nn
from torch.nn import functional as F
class Block(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.ReLU(),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.ReLU())
class Segmenter(nn.Module):
    def __init__(self,skip=False):
        super().__init__();self.skip=skip
        self.e1=Block(3,16);self.e2=Block(16,32);self.b=Block(32,64)
        self.p2=nn.Conv2d(64,32,1);self.d2=Block(32,32)
        self.p1=nn.Conv2d(32,16,1);self.d1=Block(16,16);self.head=nn.Conv2d(16,2,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));z=self.b(F.max_pool2d(b,2))
        z=self.p2(F.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False))
        z=self.d2(z+b if self.skip else z)
        z=self.p1(F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False))
        return self.head(self.d1(z+a if self.skip else z))
