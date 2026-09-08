"""Private RQ deployment check using a local fixture; no billing, S3, or callbacks."""
import hashlib
from pathlib import Path
import time


def run_v104(fixture_path, expected_sha256, reference_path=None):
    from worker_tasks import init_dense_uv_pipeline
    data = Path(fixture_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Smoke fixture changed')
    start = time.monotonic()
    runtime = init_dense_uv_pipeline('SKING_DDJ_v104', 'SKING_DDJ_v104/parser.pt', 'mappings_256x512')
    result = runtime.infer_png(data)
    report = {'model_version': 'SKING_DDJ_v104', 'seconds': round(time.monotonic() - start, 2),
            'output_sha256': hashlib.sha256(result).hexdigest(), 'bytes': len(result)}
    if reference_path is not None:
        import io
        import numpy as np
        from PIL import Image
        reference_bytes = Path(reference_path).read_bytes()
        expected = np.asarray(Image.open(io.BytesIO(reference_bytes)).convert('RGBA'))
        actual = np.asarray(Image.open(io.BytesIO(result)))
        if actual.shape != expected.shape:
            raise ValueError('Smoke reference dimensions differ')
        visible = (actual[:, :, 3] > 0) | (expected[:, :, 3] > 0)
        report.update(reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                      alpha_exact=bool(np.array_equal(actual[:, :, 3], expected[:, :, 3])),
                      visible_rgb_max=int(np.abs(actual.astype(int) - expected.astype(int))[:, :, :3][visible].max()))
    return report
