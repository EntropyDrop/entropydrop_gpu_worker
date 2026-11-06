from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F
from layout import REGION, FACE, LOCAL, INNER, VALID
BG=np.array([97,2,101],np.uint8)
TOR=REGION==7
IDS=np.flatnonzero((REGION>=0)&(REGION%6==1))
OUTER=(REGION.flatten()[IDS]==7)
OUTER_IDS=IDS[OUTER]
FB=np.zeros((64,64),bool);FB[36:48,20:28]=1;FB[36:48,32:40]=1
FB_MASK=FB.flatten()[OUTER_IDS]
FB_IDS=OUTER_IDS[FB_MASK]
FACE_IDS=np.array([0 if x<28 else 1 for x in FB_IDS%64],np.int64)
VIEWS=['front_left','back_left']
ROOT=Path(__file__).resolve().parents[1]
def quantize(uv):
    a=uv.detach().float().cpu()[0].clamp(0,1);a[3]=(a[3]>=.5).float();a[3,INNER]=1;a[:,~VALID]=0;a[:3]*=a[3:4]
    return np.rint(a.permute(1,2,0).numpy()*255).astype(np.uint8)

class Normalizer:
    """Never changes the protected body envelope or uncertain background/boundary pixels."""
    def __init__(self,path=None):
        z=np.load(path or ROOT/'background_guard.npz');self.guard=z['guard'].astype(bool);self.silhouette=z['silhouette'].astype(bool)
    def mask(self,a):
        mask=np.zeros((512,512),bool);audit=[]
        for vi in range(2):
            x0=vi*256;v=a[:,x0:x0+256];guard=self.guard[:,x0:x0+256]
            border=np.zeros((512,256),bool);border[:6]=1;border[-6:]=1;border[:,:6]=1;border[:,-6:]=1
            vals=v[border&~guard].astype(float)
            if len(vals)<100:audit.append({'fallback':'insufficient_border'});continue
            # Explicit expected edited background family; arbitrary backgrounds safely fall back.
            allowed=np.linalg.norm(vals-BG.astype(float),axis=1)<55
            if allowed.mean()<.65:audit.append({'fallback':'untrusted_background_colour'});continue
            vals=vals[allowed];bins=(vals//8).astype(int);u,cnt=np.unique(bins,axis=0,return_counts=True);mode=u[cnt.argmax()]
            cluster=vals[(np.abs(bins-mode)<=1).all(1)];center=np.median(cluster,axis=0)
            distances=np.linalg.norm(cluster-center,axis=1);radius=float(np.clip(np.quantile(distances,.95)+5,14,30))
            candidate=np.linalg.norm(v.astype(float)-center,axis=2)<=radius
            seed=np.zeros(candidate.shape,bool);seed[[0,-1]]=1;seed[:,[0,-1]]=1
            connected=ndimage.binary_propagation(seed&candidate,mask=candidate)
            # No island removal, no hole filling, and a three-pixel uncertain edge band.
            safe=connected&~guard&(ndimage.distance_transform_edt(connected)>3)
            if safe.sum()<256:audit.append({'fallback':'insufficient_safe_area'});continue
            mask[:,x0:x0+256]=safe;audit.append(dict(background_center=center.tolist(),radius=radius,safe_pixels=int(safe.sum())))
        assert not (mask&self.guard).any()
        return mask,audit
    def __call__(self,a):
        mask,audit=self.mask(a);out=a.copy();out[mask]=BG
        assert np.array_equal(out[self.guard],a[self.guard]);return out,mask,audit
    def field(self,a):
        mask,audit=self.mask(a)
        if mask.sum()<512:return np.broadcast_to(BG,a.shape).copy(),dict(fallback=True,audit=audit)
        # Only trusted exterior colour enters the bank; no figure texture is extracted.
        ix=ndimage.distance_transform_edt(~mask,return_distances=False,return_indices=True)
        return a[ix[0],ix[1]],dict(fallback=False,safe_pixels=int(mask.sum()),audit=audit)

class TorsoAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(198,128);self.position=nn.Linear(2,128);self.face=nn.Embedding(12,128)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(128,4,384,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.norm=nn.LayerNorm(128);self.delta=nn.Linear(128,1);nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        self.register_buffer('local_uv',torch.from_numpy(LOCAL.reshape(-1,2)[IDS]));self.register_buffer('face_ids',torch.from_numpy((REGION.flatten()[IDS]//6)*6+FACE.flatten()[IDS]%6))
        self.register_buffer('outer',torch.from_numpy(OUTER))
    def forward(self,f,z):
        x=self.input(f.float())+self.position(self.local_uv)[None]+self.face(self.face_ids)[None]
        for layer in self.layers:x=layer(x)
        delta=self.delta(self.norm(x)).float()[:,self.outer,0];return z.float()+delta,delta

def fuse(base_uv,original_rgb,logits):
    a=base_uv.numpy().copy() if isinstance(base_uv,torch.Tensor) else base_uv.copy()
    mask=np.zeros((64,64),bool);mask.flat[OUTER_IDS]=logits.detach().float().cpu().sigmoid().numpy()>=.5
    colors=np.rint(original_rgb.permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    a[...,3][TOR]=mask[TOR]*255;a[...,:3][TOR]=colors[TOR];a[...,:3][a[...,3]==0]=0
    old=base_uv.numpy() if isinstance(base_uv,torch.Tensor) else base_uv
    assert np.array_equal(a[~TOR],old[~TOR]);common=TOR&(a[...,3]>127)&(old[...,3]>127);assert np.array_equal(a[common],old[common])
    return a
def crop_image(a):
    from PIL import Image
    return torch.stack([torch.from_numpy(np.array(Image.fromarray(a).crop((v*256+30,125,v*256+232,375)).resize((128,160),Image.Resampling.BILINEAR)).copy()).permute(2,0,1) for v in range(2)])

class FlatJudge(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Sequential(nn.Linear(597,96),nn.LayerNorm(96),nn.GELU());self.position=nn.Linear(2,96,bias=False);self.face=nn.Embedding(2,96)
        layers=[];last=3
        for c in [24,48,96,96]:layers.extend([nn.Conv2d(last,c,3,stride=2,padding=1),nn.GroupNorm(8,c),nn.GELU()]);last=c
        self.cnn=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((8,6)),nn.Flatten(),nn.Linear(96*8*6,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,128),nn.GELU())
        self.pixel=nn.Sequential(nn.Linear(224,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,32),nn.GELU(),nn.Linear(32,1));self.empty=nn.Sequential(nn.Linear(224,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        nn.init.zeros_(self.pixel[-1].weight);nn.init.constant_(self.pixel[-1].bias,-3.);nn.init.zeros_(self.empty[-1].weight);nn.init.constant_(self.empty[-1].bias,-4.)
        self.register_buffer('face_id',torch.from_numpy(FACE_IDS));self.register_buffer('local',torch.from_numpy(LOCAL.reshape(-1,2)[FB_IDS]));self.register_buffer('counts',torch.bincount(torch.from_numpy(FACE_IDS),minlength=2).float())
    def forward(self,bundle,z):
        f,images=bundle;r,n=f[:,0].float(),f[:,1].float();im=images.float()/255
        if self.training:
            # Shared photometric changes across both views, preserving geometric labels.
            gain=torch.empty(len(im),1,3,1,1,device=im.device).uniform_(.65,1.35);bias=torch.empty(len(im),1,3,1,1,device=im.device).uniform_(-.08,.08)
            im=(im*gain+bias).clamp(0,1)
            if torch.rand((),device=im.device)<.25:im=im.mean(2,keepdim=True).expand(-1,-1,3,-1,-1)
        ci=self.cnn(im.flatten(0,1)*2-1).reshape(len(f),2,128)
        x=self.input(torch.cat([r,n,(r-n).abs(),z.transpose(1,2).float()],-1))+self.position(self.local)[None]+self.face(self.face_id)[None]
        ctx=x.new_zeros(len(f),2,96);ctx.scatter_add_(1,self.face_id[None,:,None].expand_as(x),x);ctx=ctx/self.counts[None,:,None]
        return self.pixel(torch.cat([x,ci[:,self.face_id]],-1)).squeeze(-1).float(),self.empty(torch.cat([ctx,ci],-1)).squeeze(-1).float()
