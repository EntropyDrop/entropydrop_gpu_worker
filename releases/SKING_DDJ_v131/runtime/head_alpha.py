from pathlib import Path

import json,hashlib

import torch

from torch import nn

import torch.nn.functional as F

from layout import REGION,FACE,LOCAL,VALID,INNER,OUTER





class HeadAlphaRefiner(nn.Module):
    def __init__(self,width=128):
        super().__init__();ids=torch.from_numpy(((REGION>=0)&(REGION%6==0)).flatten()).nonzero().flatten();self.register_buffer('ids',ids)
        region=torch.from_numpy(REGION.flatten())[ids];face=torch.from_numpy(FACE.flatten())[ids];local=torch.from_numpy(LOCAL.reshape(-1,2))[ids]
        self.register_buffer('outer',region==6);self.register_buffer('local_uv',local);self.register_buffer('face_id',(region//6)*6+face%6)
        self.input=nn.Linear(192+4+2,width);self.face_embedding=nn.Embedding(12,width);self.position=nn.Linear(2,width)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(width,4,width*3,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.norm=nn.LayerNorm(width);self.delta=nn.Linear(width,1);nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        # Only within-face horizontal mirror pairs for top/front/back. Labels decide
        # whether each pair should agree; asymmetric accessories are never mirrored.
        outer_ids=ids[self.outer].tolist();index={q:i for i,q in enumerate(outer_ids)};pairs=[];edges=[]
        for flat in outer_ids:
            y,x=divmod(flat,64);f=int(FACE[y,x])%6
            if f in [0,3,5]:
                x0=40 if f in [0,3] else 56;mirror=y*64+x0+7-(x-x0)
                if flat<mirror:pairs.append([index[flat],index[mirror]])
            for nx,ny in [(x+1,y),(x,y+1)]:
                other=ny*64+nx
                if nx<64 and ny<64 and other in index and FACE[ny,nx]==FACE[y,x]:edges.append([index[flat],index[other]])
        self.register_buffer('mirror_pairs',torch.tensor(pairs));self.register_buffer('edge_pairs',torch.tensor(edges))
    def forward(self,evidence):
        uv=evidence['uv_v2'].flatten(2)[:,:,self.ids].transpose(1,2)
        x=self.input(torch.cat([evidence['features'].float(),uv,evidence['view_evidence'].float()],-1))
        x=x+self.face_embedding(self.face_id)[None]+self.position(self.local_uv)[None]
        for layer in self.layers:x=layer(x)
        delta=self.delta(self.norm(x)).float()[...,0][:,self.outer]
        return evidence['alpha_logits'][:,self.outer].float()+delta,delta

def assemble(evidence,refiner,logits=None):
    # Body colour is an independently frozen prediction, never rescaled by a head gate.
    out=evidence['uv_v2'].float().clone();body=torch.as_tensor((REGION>=0)&(REGION%6!=0),device=out.device)
    assert 'uv_v1' in evidence,'Body preservation requires the raw v1 prediction, including its hidden RGB.'
    out[:,:3,body]=evidence['uv_v1'][:,:3,body]
    if logits is not None:
        flat=out.flatten(2);flat[:,3,refiner.ids[refiner.outer]]=logits.sigmoid()
    return out
