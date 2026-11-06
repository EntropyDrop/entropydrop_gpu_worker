"""RQ deployment probe on local fixtures: no billing, uploads or callbacks."""
import hashlib
from pathlib import Path
import time

def run(model_version, input_path, expected_sha256, output_path):
    from worker_tasks import init_dense_uv_pipeline
    if model_version not in {'SKING_DDJ_v130','SKING_DDJ_v131'}:
        raise ValueError('Unsupported probe version')
    data=Path(input_path).read_bytes()
    assert hashlib.sha256(data).hexdigest()==expected_sha256
    started=time.monotonic()
    runtime=init_dense_uv_pipeline(model_version,model_version+'/parser.pt',model_version+'/mappings')
    png=runtime.infer_png(data)
    Path(output_path).write_bytes(png)
    return dict(model_version=model_version,output_sha256=hashlib.sha256(png).hexdigest(),
                resident_pids=runtime.resident_pids,seconds=time.monotonic()-started)
