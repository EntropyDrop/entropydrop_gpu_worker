"""Run an immutable full UV release without mixing its imports with legacy v61."""
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from PIL import Image

logger = logging.getLogger(__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class BundledDenseUVRuntime:
    """The trained foreground + v104 parser + final material fitting pipeline."""

    def __init__(self, checkpoint_path, mappings_dir, model_version, device='cuda'):
        if device != 'cuda':
            raise ValueError('The bundled pipeline requires CUDA')
        self.root = Path(checkpoint_path).resolve().parent
        self.release = json.loads((self.root / 'release.json').read_text())
        release_model_version = self.release.get('model_version')
        is_matching_version = (
            release_model_version == model_version
            or (release_model_version == 'SKING_DDJ_v104' and model_version == 'SKING_DDJ_v104b')
        )
        if (self.release.get('format') != 1
                or not is_matching_version
                or Path(checkpoint_path).resolve() != self.root / 'parser.pt'):
            raise ValueError('Dense UV release identity mismatch')
        if Path(mappings_dir).resolve() != Path(self.release['mappings_dir']).resolve():
            raise ValueError('Dense UV mappings mismatch')
        # Verify model/code/cache bytes once per worker, before accepting traffic.
        for name, expected in self.release['sha256'].items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or file_sha256(path) != expected:
                raise ValueError(f'Dense UV release integrity failed: {name}')
        for name, expected in self.release['mappings_sha256'].items():
            path = (Path(mappings_dir) / name).resolve()
            if not path.is_relative_to(Path(mappings_dir).resolve()) or file_sha256(path) != expected:
                raise ValueError(f'Dense UV mappings integrity failed: {name}')
        for key in ('parser_python', 'foreground_python'):
            if not os.access(self.release[key], os.X_OK):
                raise ValueError(f'Dense UV runtime unavailable: {key}')
        self.timeout_seconds = 540

    def _run(self, command, cwd, env, log_path):
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=self.timeout_seconds)
                if code:
                    logger.error('Bundled UV subprocess failed: %s', log_path.read_text(errors='replace')[-4000:])
                    raise RuntimeError(f'Dense UV subprocess exited with code {code}')
            finally:
                # RQ's timeout raises in the parent. Kill the complete process group,
                # including foreground/parser grandchildren, before removing inputs.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    def infer_png(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as source:
            width, height = source.size
            if width != height or width < 256 or width > 4096 or width % 2:
                raise ValueError('Dense UV expects an even square combined render (256–4096px)')
            source.verify()
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='dense-uv-v104-') as temporary:
            work = Path(temporary)
            input_path = work / 'input.png'
            input_path.write_bytes(image_bytes)
            entry = self.root / 'runtime/dense_uv_parser/run_local.py'
            command = [self.release['parser_python'], str(entry), 'foreground_batch',
                       '--checkpoint', str(self.root / 'parser.pt'),
                       '--foreground-checkpoint', str(self.root / 'foreground.pt'),
                       '--foreground-model-dir', str(self.root / 'foreground_base'),
                       '--foreground-python', self.release['foreground_python'],
                       '--parser-python', self.release['parser_python'],
                       '--pipeline', str(self.root / 'pipeline.json'),
                       '--probability-dir', str(work / 'foreground'),
                       '--output-dir', str(work / 'output'), '--inputs', str(input_path)]
            env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                   'HF_HOME': str(self.root / 'hf_cache'),
                   'HF_HUB_CACHE': str(self.root / 'hf_cache/hub'),
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                   'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TMPDIR': str(work)}
            self._run(command, self.root / 'runtime', env, work / 'inference.log')
            manifests = list((work / 'output').glob('*/manifest.json'))
            if len(manifests) != 1:
                raise RuntimeError('Dense UV did not produce exactly one completed output')
            manifest = json.loads(manifests[0].read_text())
            expected_sources = {Path(k).name: v for k, v in self.release['sha256'].items()
                                if Path(k).parent == Path('runtime/dense_uv_parser') and k.endswith('.py')}
            if (manifest.get('complete') is not True
                    or manifest.get('input_sha256') != hashlib.sha256(image_bytes).hexdigest()
                    or manifest.get('checkpoint_sha256') != self.release['sha256']['parser.pt']
                    or manifest.get('pipeline') != json.loads((self.root / 'pipeline.json').read_text())
                    or manifest.get('source_sha256') != expected_sources
                    or manifest.get('foreground_provider', {}).get('adaptation_sha256') != self.release['sha256']['foreground.pt']
                    or manifest.get('foreground_provider', {}).get('model_sha256') != self.release['sha256']['foreground_base/model.safetensors']):
                raise RuntimeError('Dense UV output provenance mismatch')
            result = (manifests[0].parent / 'pred_uv.png').read_bytes()
            with Image.open(io.BytesIO(result)) as uv:
                if uv.size != (64, 64) or uv.mode != 'RGBA' or not uv.getchannel('A').getbbox():
                    raise RuntimeError('Dense UV returned an invalid skin image')
            logger.info('Dense UV %s completed in %.2fs', self.release['model_version'], time.monotonic() - start)
            return result


def create_dense_uv_runtime(model_version, toolkit_root, checkpoint_path, mappings_dir, device):
    if model_version in ('SKING_DDJ_v104', 'SKING_DDJ_v104b'):
        return BundledDenseUVRuntime(checkpoint_path, mappings_dir, model_version, device)
    from dense_uv_runtime import DenseUVInferenceRuntime
    return DenseUVInferenceRuntime(toolkit_root=toolkit_root, checkpoint_path=checkpoint_path,
                                   mappings_dir=mappings_dir, device=device)
