"""Private RQ deployment check using a local fixture; no billing, S3, or callbacks."""
import hashlib
from pathlib import Path
import time


def run_v104(fixture_path, expected_sha256):
    from worker_tasks import init_dense_uv_pipeline
    data = Path(fixture_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Smoke fixture changed')
    start = time.monotonic()
    runtime = init_dense_uv_pipeline('SKING_DDJ_v104', 'SKING_DDJ_v104/parser.pt', 'mappings_256x512')
    result = runtime.infer_png(data)
    return {'model_version': 'SKING_DDJ_v104', 'seconds': round(time.monotonic() - start, 2),
            'output_sha256': hashlib.sha256(result).hexdigest(), 'bytes': len(result)}
