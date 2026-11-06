"""Versioned semantic UV worker adapter; imports no GPU libraries in the RQ parent."""
import atexit
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import weakref
from PIL import Image
from dense_uv_resident import ResidentProcess

_runtimes=weakref.WeakSet()

@atexit.register
def _shutdown():
    for runtime in list(_runtimes):runtime.close()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def validate_skin(data):
    with Image.open(io.BytesIO(data)) as im:
        if im.size!=(64,64) or im.mode!='RGBA':raise RuntimeError('Semantic UV returned invalid RGBA dimensions')
        pixels=list(im.getdata())
    inner=set();valid=set()
    for u,v,w,d,h,base in [(0,0,8,8,8,True),(32,0,8,8,8,False),(16,16,8,4,12,True),
                          (16,32,8,4,12,False),(40,16,4,4,12,True),(40,32,4,4,12,False),
                          (32,48,4,4,12,True),(48,48,4,4,12,False),(0,16,4,4,12,True),
                          (0,32,4,4,12,False),(16,48,4,4,12,True),(0,48,4,4,12,False)]:
        cells={(x,y) for x in range(u+d,u+d+2*w) for y in range(v,v+d)}
        cells|={(x,y) for x in range(u,u+2*(w+d)) for y in range(v+d,v+d+h)}
        valid|=cells
        if base:inner|=cells
    for i,(r,g,b,a) in enumerate(pixels):
        xy=(i%64,i//64)
        if a not in (0,255) or (xy in inner and a!=255) or (xy not in valid and a!=0) or (a==0 and (r or g or b)):
            raise RuntimeError('Semantic UV returned invalid atlas alpha/RGB')

class SemanticUVRuntime:
    def __init__(self,checkpoint_path,mappings_dir,model_version,device='cuda',execution_mode=None):
        if device!='cuda':raise ValueError('Semantic UV requires CUDA')
        self.model_version=model_version
        self.root=Path(checkpoint_path).resolve().parent
        self.release=json.loads((self.root/'release.json').read_text())
        if (model_version not in {'SKING_DDJ_v130','SKING_DDJ_v131'} or self.release.get('model_version')!=model_version or
            self.release.get('format')!=2 or Path(checkpoint_path).resolve()!=self.root/'parser.pt'):
            raise ValueError('Semantic UV release identity mismatch')
        if Path(mappings_dir).resolve()!=self.root/'mappings':raise ValueError('Semantic UV mappings mismatch')
        required={'parser.pt','semantic_encoder.pth','pipeline.json','runtime/engine.py','runtime/run.py',
                  'runtime/layout.py','runtime/decoder_body.py','runtime/decoder_head.py','runtime/head_alpha.py',
                  'mappings/geometry.pt','mappings/routing_geometry.pt'}
        if model_version == 'SKING_DDJ_v131':
            required.update({'background_guard.npz','runtime/base_engine.py','runtime/torso_models.py',
                             'runtime/torso_geometry.py','runtime/renderer.py',
                             'mappings/front_left_mapping.pt','mappings/back_left_mapping.pt'})
        if not required.issubset(self.release['sha256']):raise ValueError('Semantic UV release is incomplete')
        for name,expected in self.release['sha256'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or sha(path)!=expected:
                raise ValueError('Semantic UV release integrity failed: '+name)
        self.python=os.getenv(model_version.upper()+'_PYTHON',self.release['python'])
        if not os.access(self.python,os.X_OK):raise ValueError('Semantic UV Python runtime is unavailable')
        self.pipeline=json.loads((self.root/'pipeline.json').read_text())
        self.release_hash=sha(self.root/'release.json')
        self.execution_mode=execution_mode or os.getenv('DENSE_UV_EXECUTION_MODE','resident')
        if self.execution_mode not in {'resident','oneshot'}:raise ValueError('Invalid Dense UV execution mode')
        self.timeout_seconds=540
        self._process=None;self._lock=threading.RLock();_runtimes.add(self)

    def _environment(self):
        return {**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1',
                'PYTHONPATH':str(self.root/'runtime')}

    def _ensure(self,deadline):
        if self._process is None:
            entry=Path(__file__).with_name('dense_uv_resident.py')
            self._process=ResidentProcess([self.python,str(entry),'--release-root',str(self.root),'--stage','semantic'],
                                           self.root/'runtime',self._environment())
        self._process.start(deadline)

    @property
    def resident_pids(self):return {'semantic':self._process.pid} if self._process is not None else {}

    def warmup(self):
        if self.execution_mode=='resident':
            with self._lock:
                try:self._ensure(time.monotonic()+self.timeout_seconds)
                except BaseException:self.close();raise

    def close(self):
        with self._lock:
            if self._process is not None:self._process.close();self._process=None

    def _oneshot(self,input_path,output_path,work):
        command=[self.python,str(self.root/'runtime/run.py'),'--input',str(input_path),'--output',str(output_path)]
        with (work/'runtime.log').open('wb') as log:
            proc=subprocess.Popen(command,cwd=self.root/'runtime',env=self._environment(),stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            try:
                if proc.wait(timeout=self.timeout_seconds):raise RuntimeError('Semantic UV inference failed: '+(work/'runtime.log').read_text(errors='replace')[-2000:])
            finally:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()

    def infer_png(self,image_bytes):
        if not image_bytes or len(image_bytes)>20*1024**2:raise ValueError('Semantic UV input size must be within 20 MiB')
        with Image.open(io.BytesIO(image_bytes)) as im:
            if im.size!=(512,512):raise ValueError('Semantic UV expects a 512x512 edited render')
            im.verify()
        with self._lock:
            try:
                with tempfile.TemporaryDirectory(prefix=self.model_version.lower()+'-') as tmp:
                    work=Path(tmp);input_path=work/'input.png';output_path=work/'uv.png';input_path.write_bytes(image_bytes)
                    if self.execution_mode=='resident':
                        deadline=time.monotonic()+self.timeout_seconds;self._ensure(deadline)
                        self._process.request(dict(input=str(input_path),output=str(output_path)),deadline)
                    else:self._oneshot(input_path,output_path,work)
                    meta=json.loads(output_path.with_suffix('.png.json').read_text());result=output_path.read_bytes()
                    expected=dict(complete=True,model_version=self.model_version,input_sha256=hashlib.sha256(image_bytes).hexdigest(),
                                  output_sha256=hashlib.sha256(result).hexdigest(),release_sha256=self.release_hash,
                                  checkpoint_sha256=self.release['sha256']['parser.pt'],pipeline=self.pipeline)
                    if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError('Semantic UV output provenance mismatch')
                    validate_skin(result)
                    return result
            except BaseException:
                self.close();raise
