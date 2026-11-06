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
    def forward(self,x,m,head_memory,gates,head_gate,part_indices):
        out=torch.zeros_like(x)
        for part,ids in enumerate(part_indices):
            z=x[:,ids];q=self.n1(z);z=z+self.self_attention(q,q,q,need_weights=False)[0]
            memory=head_memory if part==0 else m;gate=head_gate if part==0 else gates[:,part]
            bias=gate.clamp_min(.02).log().to(z.dtype)[:,None,None,:].expand(-1,4,len(ids),-1).reshape(len(x)*4,len(ids),-1)
            z=z+self.cross_attention(self.n2(z),memory,memory,attn_mask=bias,need_weights=False)[0]
            out[:,ids]=z+self.ff(self.n3(z))
        return out

class UVModel(nn.Module):
    def __init__(self,geometry,width=192,blocks=4):
        super().__init__();ids=torch.from_numpy(VALID.flatten()).nonzero().flatten();self.register_buffer('ids',ids)
        self.register_buffer('inner',torch.from_numpy(INNER));self.register_buffer('valid',torch.from_numpy(VALID))
        self.register_buffer('xy',geometry['xy'].reshape(2,4096,2)[:,ids]);self.register_buffer('support',geometry['support'].reshape(2,4096)[:,ids])
        self.register_buffer('region',torch.from_numpy(REGION.flatten())[ids]);self.register_buffer('face',torch.from_numpy(FACE.flatten())[ids]);self.register_buffer('local_uv',torch.from_numpy(LOCAL.reshape(4096,2))[ids])
        self.register_buffer('part_priors',geometry['part_priors']);self.register_buffer('head_boxes',geometry['head_boxes'])
        for part in range(6):self.register_buffer(f'part_{part}',(self.region%6==part).nonzero().flatten(),persistent=False)
        self.encoder1=Conv(3,32,2);self.encoder2=Conv(32,64,2);self.encoder3=Conv(64,128,2);self.encoder4=Conv(128,192,2)
        self.image_parts=nn.Sequential(nn.Conv2d(64+192+2,64,1),nn.SiLU(),nn.Conv2d(64,13,1))
        self.dino_projection=nn.Linear(1024,width);self.image_projection=nn.Conv2d(192,width,1)
        self.local_projection=nn.Sequential(nn.Linear(18*64,width),nn.LayerNorm(width),nn.GELU())
        self.query=nn.Parameter(torch.randn(len(ids),width)*.02);self.part_embedding=nn.Embedding(12,width);self.face_embedding=nn.Embedding(72,width);self.uv_position=nn.Linear(2,width)
        self.memory_position=nn.Parameter(torch.randn(1,512,width)*.01)
        self.head_memory_position=nn.Parameter(torch.randn(1,640,width)*.01)
        self.view_evidence=nn.Sequential(nn.Linear(width+64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.view_evidence[-1].weight);nn.init.constant_(self.view_evidence[-1].bias,2.)
        self.blocks=nn.ModuleList([DecoderBlock(width) for _ in range(blocks)])
        self.norm=nn.LayerNorm(width);self.rgb=nn.Linear(width,3);self.alpha=nn.Linear(width,1);self.visibility=nn.Linear(width,1)
        d=torch.tensor([-4.,0.,4.]);yy,xx=torch.meshgrid(d,d,indexing='ij');self.register_buffer('patch',torch.stack([xx,yy],-1).reshape(9,2))

    def forward(self,image,semantic_tokens=None,encoder=None):
        assert encoder is not None,'Frozen encoder is required for isolated head features'
        b=len(image);f=self.encoder2(self.encoder1(image*2-1));deep=self.encoder4(self.encoder3(f))
        yy,xx=torch.meshgrid(torch.linspace(-1,1,f.shape[2],device=f.device),torch.linspace(-1,1,f.shape[3],device=f.device),indexing='ij')
        coords=torch.stack([xx,yy])[None].expand(b,-1,-1,-1)
        part_features=torch.cat([f,F.interpolate(deep,f.shape[-2:],mode='bilinear',align_corners=False),coords],1)
        part_logits=self.image_parts(part_features).float();probs=part_logits.softmax(1)
        # Routing is supervised through segmentation, not rewarded for hiding hard pixels.
        physical=(probs[:,1:7]+probs[:,7:13]).detach()
        physical=F.interpolate(physical,(512,512),mode='bilinear',align_corners=False)*self.part_priors[None]
        head_mask=((physical[:,0:1]-.10)/.60).clamp(0,1)
        crops=[];crop_gates=[]
        for box in self.head_boxes:
            x0,y0,x1,y1=box;cy,cx=torch.meshgrid(torch.linspace(y0,y1,224,device=image.device),torch.linspace(x0,x1,224,device=image.device),indexing='ij')
            grid=torch.stack([cx,cy],-1)[None].expand(b,-1,-1,-1)/511*2-1
            gate=F.grid_sample(head_mask,grid,align_corners=True)
            crop=F.grid_sample(image,grid,align_corners=True)
            crops.append(crop*gate+.5*(1-gate));crop_gates.append(gate)
        head_image=torch.stack(crops,1).reshape(b*2,3,224,224);head_gate_image=torch.stack(crop_gates,1).reshape(b*2,1,224,224)
        hf=self.encoder2(self.encoder1(head_image*2-1));hd=self.encoder4(self.encoder3(hf))
        global_semantics=semantics(encoder,image);head_semantics=semantics(encoder,head_image)
        positions=self.xy[:,None]+self.patch[None,:,None]
        grid=(positions/511*2-1).reshape(1,2*9,-1,2).expand(b,-1,-1,-1)
        local=F.grid_sample(f,grid,align_corners=True,padding_mode='zeros').reshape(b,64,2,9,-1)
        gate=F.grid_sample(physical,grid,align_corners=True).reshape(b,6,2,9,-1)
        gi=(self.region%6)[None,None,None,None,:].expand(b,1,2,9,-1);gate=gate.gather(1,gi)[:,0]
        boxes=self.head_boxes[:,None,None];hg=(positions-boxes[...,:2])/(boxes[...,2:]-boxes[...,:2])*2-1
        hg=hg[None].expand(b,-1,-1,-1,-1).reshape(b*2,9,-1,2)
        head_local=F.grid_sample(hf,hg,align_corners=True).reshape(b,2,64,9,-1).permute(0,2,1,3,4)
        local=torch.where((self.region%6==0)[None,None,None,None],head_local,local)
        local=local*gate[:,None]*self.support[None,None,:,None]
        q=self.query+self.part_embedding(self.region)+self.face_embedding(self.face)+self.uv_position(self.local_uv)
        ev=torch.cat([local.mean(3).permute(0,2,3,1),q[None,None].expand(b,2,-1,-1)],-1)
        view_logits=self.view_evidence(ev).float()[...,0]
        local=local*view_logits.sigmoid()[:,None,:,None]
        local=local.permute(0,4,2,3,1).reshape(b,len(self.ids),-1);x=q[None]+self.local_projection(local)
        memory=torch.cat([self.dino_projection(global_semantics),F.adaptive_avg_pool2d(self.image_projection(deep),16).flatten(2).transpose(1,2)],1)+self.memory_position
        hsem=self.dino_projection(head_semantics).reshape(b,512,-1);hcnn=F.adaptive_avg_pool2d(self.image_projection(hd),8).flatten(2).transpose(1,2).reshape(b,128,-1)
        head_memory=torch.cat([hsem,hcnn],1)+self.head_memory_position
        memgate=F.adaptive_avg_pool2d(physical,16).flatten(2).repeat(1,1,2)
        hgates=torch.cat([F.adaptive_avg_pool2d(head_gate_image,16).reshape(b,512),F.adaptive_avg_pool2d(head_gate_image,8).reshape(b,128)],1)
        for block in self.blocks:x=block(x,memory,head_memory,memgate,hgates,[getattr(self,f'part_{p}') for p in range(6)])
        x=self.norm(x);rgb=self.rgb(x).float().sigmoid();alogit=self.alpha(x).float();vlogit=self.visibility(x).float()
        dense=torch.zeros(b,4,4096,device=x.device,dtype=torch.float32);dense[:,:3,self.ids]=rgb.transpose(1,2);dense[:,3:4,self.ids]=alogit.sigmoid().transpose(1,2)
        uv=dense.reshape(b,4,64,64);uv[:,3]=torch.where(self.inner,1.,uv[:,3])*self.valid
        return uv,dict(alpha_logits=alogit[:,:,0],visibility_logits=vlogit[:,:,0],view_visibility_logits=view_logits,image_part_logits=part_logits)

def semantics(encoder,image):
    x=F.interpolate(image.float(),(224,224),mode='bilinear',align_corners=False);mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=image.is_cuda):t=encoder.forward_features((x-mean)/std)['x_norm_patchtokens']
    return t
