"""Export inference tensors and local runtime assets; no training data or optimizer state."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ['base-release','torso-checkpoint','face-checkpoint','background-guard','renderer-source','runtime-dir','destination']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();dest=a.destination
    assert not (dest/'release.json').exists(),'Finalized releases are immutable'
    base=json.loads((a.base_release/'release.json').read_text())
    assert base['model_version']=='SKING_DDJ_v130'
    for n,d in base['sha256'].items():assert sha(a.base_release/n)==d
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['mappings','vendor']:
        shutil.copytree(a.base_release/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    shutil.copyfile(a.base_release/'semantic_encoder.pth',dest/'semantic_encoder.pth')
    shutil.copytree(a.runtime_dir,dest/'runtime',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['layout.py','decoder_body.py','decoder_head.py','head_alpha.py']:
        shutil.copy2(a.base_release/'runtime'/name,dest/'runtime'/name)
    code=(a.base_release/'runtime/engine.py').read_text().replace('SKING_DDJ_v130','SKING_DDJ_v131')
    (dest/'runtime/base_engine.py').write_text(code)
    shutil.copy2(a.renderer_source,dest/'runtime/renderer.py')
    shutil.copy2(a.background_guard,dest/'background_guard.npz')
    b=torch.load(a.base_release/'parser.pt',map_location='cpu',weights_only=True)
    c=torch.load(a.torso_checkpoint,map_location='cpu',weights_only=True)
    f=torch.load(a.face_checkpoint,map_location='cpu',weights_only=True)
    weights={k:b[k] for k in ['body','head','refiner']}
    weights.update(torso_adapter=c['adapter'],face_judge=f['model'])
    for state in weights.values():assert all(isinstance(v,torch.Tensor) for v in state.values())
    clean={k:{n:v.detach().cpu().contiguous().clone() for n,v in state.items()} for k,state in weights.items()}
    torch.save(dict(model_version='SKING_DDJ_v131',**clean),dest/'parser.pt')
    restored=torch.load(dest/'parser.pt',map_location='cpu',weights_only=True)
    for key,state in weights.items():
        assert state.keys()==restored[key].keys() and all(torch.equal(v,restored[key][n]) for n,v in state.items())
    pipeline=dict(model_version='SKING_DDJ_v131',input_size=[512,512],views=['front_left','back_left'],input_background=[97,2,101],head_outer_threshold=.25,other_outer_threshold=.5,torso_empty_threshold=.999,torso_geometry=dict(max_component=3,max_face=8,max_relative_error=.4),output_size=[64,64],output_mode='RGBA',body_rgb_preserved=True)
    (dest/'pipeline.json').write_text(json.dumps(pipeline,indent=2)+'\n')
    audit=dict(model_version='SKING_DDJ_v131',inference_only=True,tensor_values_unchanged=True,base_release_sha256=sha(a.base_release/'release.json'),source_checkpoint_sha256={'base':sha(a.base_release/'parser.pt'),'torso_adapter':sha(a.torso_checkpoint),'face_judge':sha(a.face_checkpoint)},exported_checkpoint_sha256=sha(dest/'parser.pt'))
    (dest/'weights_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    manifest=dict(format=2,model_version='SKING_DDJ_v131',python='/opt/conda/envs/uv-runtime/bin/python',execution='resident',input_size=[512,512],output_size=[64,64],views=['front_left','back_left'],mappings_dir='mappings',framework=base['framework'],sha256={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
    (dest/'release.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'complete':True,'model_version':'SKING_DDJ_v131','tensor_values_unchanged':True,'release_sha256':sha(dest/'release.json'),'files':len(manifest['sha256'])}))

if __name__=='__main__':main()
