import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from PIL import Image,ImageDraw
import asset_paths
from asset_paths import dense_uv_asset_paths
from bundled_dense_uv_runtime import create_dense_uv_runtime
from semantic_uv_runtime import SemanticUVRuntime,sha,validate_skin

@pytest.fixture(params=['SKING_DDJ_v130','SKING_DDJ_v131'])
def bundle(tmp_path,request):
    root=tmp_path/request.param;root.mkdir()
    names=['parser.pt','semantic_encoder.pth','runtime/engine.py','runtime/run.py','runtime/layout.py',
           'runtime/decoder_body.py','runtime/decoder_head.py','runtime/head_alpha.py',
           'mappings/geometry.pt','mappings/routing_geometry.pt']
    if request.param == 'SKING_DDJ_v131':
        names += ['background_guard.npz','runtime/base_engine.py','runtime/torso_models.py','runtime/torso_geometry.py','runtime/renderer.py','mappings/front_left_mapping.pt','mappings/back_left_mapping.pt']
    for name in names:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'pinned')
    pipeline=dict(model_version=request.param,head_outer_threshold=.25)
    (root/'pipeline.json').write_text(json.dumps(pipeline))
    release=dict(format=2,model_version=request.param,python=sys.executable,
                 sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    (root/'release.json').write_text(json.dumps(release))
    return root

def make_runtime(root,mode='resident'):
    return SemanticUVRuntime(root/'parser.pt',root/'mappings',root.name,execution_mode=mode)

def input_bytes(size=(512,512)):
    f=io.BytesIO();Image.new('RGB',size,(97,2,101)).save(f,format='PNG');return f.getvalue()

def skin_bytes():
    im=Image.new('RGBA',(64,64));d=ImageDraw.Draw(im)
    for u,v,w,dep,h in [(0,0,8,8,8),(16,16,8,4,12),(40,16,4,4,12),(32,48,4,4,12),(0,16,4,4,12),(16,48,4,4,12)]:
        d.rectangle((u+dep,v,u+dep+2*w-1,v+dep-1),fill=(10,20,30,255))
        d.rectangle((u,v+dep,u+2*(w+dep)-1,v+dep+h-1),fill=(10,20,30,255))
    f=io.BytesIO();im.save(f,format='PNG');return f.getvalue()

def test_v130_paths_are_bundled_and_old_roots_unchanged():
    p,m=dense_uv_asset_paths('/old','/renderer','SKING_DDJ_v130/parser.pt','SKING_DDJ_v130/mappings',model_version='SKING_DDJ_v130')
    root=Path(asset_paths.__file__).resolve().parent/'releases/SKING_DDJ_v130'
    assert p==root/'parser.pt' and m==root/'mappings'
    assert dense_uv_asset_paths('/old','/renderer','SKING_DDJ_v66.pt','mappings_256x512')==(Path('/old/SKING_DDJ_v66.pt'),Path('/renderer/mappings_256x512'))
    for path in ['../escape','/escape','SKING_DDJ_v101c/parser.pt']:
        with pytest.raises(ValueError):dense_uv_asset_paths('/old','/renderer',path,'SKING_DDJ_v130/mappings',model_version='SKING_DDJ_v130')
    with pytest.raises(ValueError):dense_uv_asset_paths('/old','/renderer','SKING_DDJ_v130/parser.pt','mappings_256x512',model_version='SKING_DDJ_v130')

def test_v130_dispatch_never_falls_back_to_legacy(bundle,monkeypatch):
    old=Mock(return_value='legacy');monkeypatch.setitem(sys.modules,'dense_uv_runtime',SimpleNamespace(DenseUVInferenceRuntime=old))
    args=dict(toolkit_root='/unused',checkpoint_path=str(bundle/'parser.pt'),mappings_dir=str(bundle/'mappings'),device='cuda')
    assert isinstance(create_dense_uv_runtime(bundle.name,**args),SemanticUVRuntime)
    (bundle/'release.json').unlink()
    with pytest.raises(FileNotFoundError):create_dense_uv_runtime(bundle.name,**args)
    old.assert_not_called()
    assert create_dense_uv_runtime('SKING_DDJ_v61',**args)=='legacy'

def test_integrity_and_mapping_identity(bundle):
    make_runtime(bundle)
    with pytest.raises(ValueError,match='mappings mismatch'):
        SemanticUVRuntime(bundle/'parser.pt',bundle/'other',bundle.name)
    (bundle/'parser.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='integrity'):make_runtime(bundle)

@pytest.mark.parametrize('mode',['resident','oneshot'])
@pytest.mark.parametrize('corrupt',[False,True])
def test_result_audit_alpha_and_temporary_cleanup(bundle,monkeypatch,mode,corrupt):
    runtime=make_runtime(bundle,mode);data=input_bytes();output=skin_bytes();folders=[]
    process=Mock();runtime._process=process
    monkeypatch.setattr(runtime,'_ensure',lambda deadline:None)
    def produce(input_path,output_path):
        folders.append(output_path.parent)
        assert input_path.read_bytes()==data
        output_path.write_bytes(output)
        meta=dict(complete=True,model_version=bundle.name,input_sha256='wrong' if corrupt else hashlib.sha256(data).hexdigest(),
                  output_sha256=hashlib.sha256(output).hexdigest(),release_sha256=runtime.release_hash,
                  checkpoint_sha256=runtime.release['sha256']['parser.pt'],pipeline=runtime.pipeline)
        output_path.with_suffix('.png.json').write_text(json.dumps(meta))
    process.request.side_effect=lambda req,deadline:produce(Path(req['input']),Path(req['output']))
    monkeypatch.setattr(runtime,'_oneshot',lambda inp,out,work:produce(inp,out))
    if corrupt:
        with pytest.raises(RuntimeError,match='provenance'):runtime.infer_png(data)
        process.close.assert_called_once()
    else:
        assert runtime.infer_png(data)==output;process.close.assert_not_called()
    assert folders and all(not f.exists() for f in folders)
    runtime.close()

def test_invalid_input_preserves_resident_and_timeout_cleans_it(bundle,monkeypatch):
    runtime=make_runtime(bundle);process=Mock();runtime._process=process
    with pytest.raises(ValueError,match='512x512'):runtime.infer_png(input_bytes((1024,1024)))
    process.close.assert_not_called()
    monkeypatch.setattr(runtime,'_ensure',Mock(side_effect=TimeoutError('deadline')))
    with pytest.raises(TimeoutError):runtime.infer_png(input_bytes())
    process.close.assert_called_once();assert runtime.resident_pids=={}

def test_rejects_opaque_unused_pixels_and_hidden_rgb():
    validate_skin(skin_bytes())
    im=Image.open(io.BytesIO(skin_bytes()));im.putpixel((0,0),(10,20,30,255));out=io.BytesIO();im.save(out,format='PNG')
    with pytest.raises(RuntimeError,match='atlas'):validate_skin(out.getvalue())
    im.putpixel((0,0),(10,0,0,0));out=io.BytesIO();im.save(out,format='PNG')
    with pytest.raises(RuntimeError,match='atlas'):validate_skin(out.getvalue())

@pytest.mark.parametrize('version',['SKING_DDJ_v130','SKING_DDJ_v131'])
def test_version_paths_cannot_cross(version):
    other='SKING_DDJ_v131' if version.endswith('130') else 'SKING_DDJ_v130'
    p,m=dense_uv_asset_paths('/old','/renderer',version+'/parser.pt',version+'/mappings',model_version=version)
    assert p.parent.name==version and m.parent==p.parent
    with pytest.raises(ValueError):dense_uv_asset_paths('/old','/renderer',other+'/parser.pt',version+'/mappings',model_version=version)

def test_release_version_must_match_request(bundle):
    other='SKING_DDJ_v131' if bundle.name.endswith('130') else 'SKING_DDJ_v130'
    with pytest.raises(ValueError,match='identity'):
        SemanticUVRuntime(bundle/'parser.pt',bundle/'mappings',other)
