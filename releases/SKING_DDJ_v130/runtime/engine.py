"""SKING_DDJ_v130: fixed-camera edited render to a classic 64x64 skin atlas."""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from layout import REGION, INNER, VALID
import decoder_body as body_model
import decoder_head as head_model
from head_alpha import HeadAlphaRefiner, assemble

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

class UVEngine:
    def __init__(self, root=None):
        self.root=Path(root or Path(__file__).resolve().parents[1]).resolve()
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        self.release=json.loads((self.root/'release.json').read_text())
        if self.release.get('model_version')!='SKING_DDJ_v130' or self.release.get('format')!=2:
            raise ValueError('SKING_DDJ_v130 release identity mismatch')
        for name,expected in self.release['sha256'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or sha(path)!=expected:
                raise ValueError('SKING_DDJ_v130 integrity failure: '+name)
        self.pipeline=json.loads((self.root/'pipeline.json').read_text())
        assert self.pipeline['head_outer_threshold']==.25
        state=torch.load(self.root/'parser.pt',map_location='cpu',weights_only=True)
        assert state['model_version']=='SKING_DDJ_v130'
        self.encoder=torch.hub.load(str(self.root/'vendor/dinov2'),'dinov2_vitl14_reg',source='local',pretrained=False)
        self.encoder.load_state_dict(torch.load(self.root/'semantic_encoder.pth',map_location='cpu',weights_only=True),strict=True)
        self.encoder=self.encoder.cuda().eval().requires_grad_(False)
        self.body=body_model.UVModel(torch.load(self.root/'mappings/geometry.pt',weights_only=True)).cuda().eval().requires_grad_(False)
        self.head=head_model.UVModel(torch.load(self.root/'mappings/routing_geometry.pt',weights_only=True)).cuda().eval().requires_grad_(False)
        self.body.load_state_dict(state['body'],strict=True)
        self.head.load_state_dict(state['head'],strict=True)
        self.head_query=(self.head.region%6==0).nonzero().flatten()
        self.features=None
        self.hook=self.head.norm.register_forward_hook(self.capture)
        self.refiner=HeadAlphaRefiner().cuda().eval()
        self.refiner.load_state_dict(state['refiner'],strict=True)
        self.calls=0
        torch.cuda.synchronize()

    def capture(self,module,inputs,output):
        self.features=output[:,self.head_query].detach()

    @torch.no_grad()
    def extract(self,image):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=image.is_cuda):
            uv,aux=self.head(image,encoder=self.encoder)
            evidence={'uv_v2':uv,'features':self.features,'alpha_logits':aux['alpha_logits'][:,self.head_query],
                      'view_evidence':aux['view_visibility_logits'][:,:,self.head_query].sigmoid().transpose(1,2)}
            evidence['uv_v1']=self.body(image,body_model.semantics(self.encoder,image))[0]
        return evidence

    def infer(self,input_path,output_path):
        started=time.monotonic()
        with Image.open(input_path) as im:
            im=im.convert('RGBA')
            if im.size!=(512,512):raise ValueError('SKING_DDJ_v130 expects a 512x512 front_left + back_left edited render')
            bg=Image.new('RGBA',im.size,(97,2,101,255));bg.alpha_composite(im)
        image=torch.from_numpy(np.asarray(bg.convert('RGB')).copy()).permute(2,0,1).float()/255
        image=image[None].cuda()
        with torch.no_grad():
            evidence=self.extract(image)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=self.refiner(evidence)
            uv=assemble(evidence,self.refiner,logits)
            uv.flatten(2)[:,3,self.refiner.ids[self.refiner.outer]]=(logits.sigmoid()>=self.pipeline['head_outer_threshold']).float()
        # This is the calibrated release's original quantization/export order.
        a=uv.detach().float().cpu()[0].clamp(0,1)
        a[3]=(a[3]>=.5).float();a[3,INNER]=1;a[:,~VALID]=0;a[:3]*=a[3:4]
        pixels=(a.permute(1,2,0).numpy()*255).round().astype(np.uint8)
        path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(path)
        self.calls+=1
        record=dict(complete=True,model_version='SKING_DDJ_v130',input_sha256=sha(input_path),output_sha256=sha(path),
                    release_sha256=sha(self.root/'release.json'),checkpoint_sha256=self.release['sha256']['parser.pt'],
                    pipeline=self.pipeline,worker_pid=os.getpid(),generation_index=self.calls,
                    seconds=time.monotonic()-started)
        path.with_suffix(path.suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
