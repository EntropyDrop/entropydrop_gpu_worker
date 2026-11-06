from pathlib import Path

import torch

from torch import nn

import torch.nn.functional as F

from layout import REGION,FACE,LOCAL,VALID,INNER

class Conv(nn.Sequential):
    def __init__(self,cin,cout,stride=1):
        super().__init__(nn.Conv2d(cin,cout,3,stride,1),nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,1,1),nn.GroupNorm(8,cout),nn.SiLU())

class DecoderBlock(nn.Module):
    def __init__(self,width=192):
        super().__init__();self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width);self.n3=nn.LayerNorm(width)
        self.self_attention=nn.MultiheadAttention(width,4,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(width,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x,m):
        q=self.n1(x);x=x+self.self_attention(q,q,q,need_weights=False)[0]
        x=x+self.cross_attention(self.n2(x),m,m,need_weights=False)[0]
        return x+self.ff(self.n3(x))

class UVModel(nn.Module):
    def __init__(self,geometry,width=192,blocks=4):
        super().__init__();ids=torch.from_numpy(VALID.flatten()).nonzero().flatten();self.register_buffer('ids',ids)
        self.register_buffer('inner',torch.from_numpy(INNER));self.register_buffer('valid',torch.from_numpy(VALID))
        self.register_buffer('xy',geometry['xy'].reshape(2,4096,2)[:,ids]);self.register_buffer('support',geometry['support'].reshape(2,4096)[:,ids])
        self.register_buffer('region',torch.from_numpy(REGION.flatten())[ids]);self.register_buffer('face',torch.from_numpy(FACE.flatten())[ids]);self.register_buffer('local_uv',torch.from_numpy(LOCAL.reshape(4096,2))[ids])
        self.encoder1=Conv(3,32,2);self.encoder2=Conv(32,64,2);self.encoder3=Conv(64,128,2);self.encoder4=Conv(128,192,2)
        self.image_parts=nn.Sequential(nn.Conv2d(64+192+2,64,1),nn.SiLU(),nn.Conv2d(64,13,1))
        self.dino_projection=nn.Linear(1024,width);self.image_projection=nn.Conv2d(192,width,1)
        self.local_projection=nn.Sequential(nn.Linear(18*64,width),nn.LayerNorm(width),nn.GELU())
        self.query=nn.Parameter(torch.randn(len(ids),width)*.02);self.part_embedding=nn.Embedding(12,width);self.face_embedding=nn.Embedding(72,width);self.uv_position=nn.Linear(2,width)
        self.memory_position=nn.Parameter(torch.randn(1,512,width)*.01)
        self.blocks=nn.ModuleList([DecoderBlock(width) for _ in range(blocks)])
        self.norm=nn.LayerNorm(width);self.rgb=nn.Linear(width,3);self.alpha=nn.Linear(width,1);self.visibility=nn.Linear(width,1)
        nn.init.constant_(self.alpha.bias,-1.5)
        d=torch.tensor([-4.,0.,4.]);yy,xx=torch.meshgrid(d,d,indexing='ij');self.register_buffer('patch',torch.stack([xx,yy],-1).reshape(9,2))
    def forward(self,image,semantic_tokens):
        b=len(image);f=self.encoder2(self.encoder1(image*2-1));deep=self.encoder4(self.encoder3(f))
        positions=self.xy[:,None]+self.patch[None,:,None]
        grid=(positions/511*2-1).reshape(1,2*9,-1,2).expand(b,-1,-1,-1)
        local=F.grid_sample(f,grid,align_corners=True,padding_mode='zeros').reshape(b,64,2,9,-1)
        local=local*self.support[None,None,:,None]
        local=local.permute(0,4,2,3,1).reshape(b,len(self.ids),-1)
        q=self.query+self.part_embedding(self.region)+self.face_embedding(self.face)+self.uv_position(self.local_uv)
        x=q[None]+self.local_projection(local)
        memory=torch.cat([self.dino_projection(semantic_tokens),F.adaptive_avg_pool2d(self.image_projection(deep),16).flatten(2).transpose(1,2)],1)+self.memory_position
        for block in self.blocks:x=block(x,memory)
        x=self.norm(x);rgb=self.rgb(x).float().sigmoid();alogit=self.alpha(x).float();vlogit=self.visibility(x).float()
        dense=torch.zeros(b,4,4096,device=x.device,dtype=torch.float32)
        dense[:,:3,self.ids]=rgb.transpose(1,2);dense[:,3:4,self.ids]=alogit.sigmoid().transpose(1,2)
        uv=dense.reshape(b,4,64,64);uv[:,3]=torch.where(self.inner,1.,uv[:,3])*self.valid
        yy,xx=torch.meshgrid(torch.linspace(-1,1,f.shape[2],device=f.device),torch.linspace(-1,1,f.shape[3],device=f.device),indexing='ij')
        coords=torch.stack([xx,yy])[None].expand(b,-1,-1,-1)
        part_features=torch.cat([f,F.interpolate(deep,f.shape[-2:],mode='bilinear',align_corners=False),coords],1)
        return uv,dict(alpha_logits=alogit[:,:,0],visibility_logits=vlogit[:,:,0],image_part_logits=self.image_parts(part_features).float())

def semantics(encoder,image):
    x=F.interpolate(image.float(),(224,224),mode='bilinear',align_corners=False)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=image.is_cuda):t=encoder.forward_features((x-mean)/std)['x_norm_patchtokens']
    assert t.shape[1:]==(256,1024)
    return t
