import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from bundled_dense_uv_runtime import BundledDenseUVRuntime, create_dense_uv_runtime, file_sha256


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'release'
    root.mkdir()
    for name in ['parser.pt', 'foreground.pt', 'foreground_base/model.safetensors',
                 'runtime/dense_uv_parser/run_local.py']:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'pinned')
    (root / 'pipeline.json').write_text('{"version":"v104"}')
    maps = tmp_path / 'maps'
    maps.mkdir()
    (maps / 'view.pt').write_bytes(b'mapping')
    release = dict(format=1, model_version='SKING_DDJ_v104', mappings_dir=str(maps),
                   mappings_sha256={'view.pt': file_sha256(maps / 'view.pt')},
                   parser_python=sys.executable, foreground_python=sys.executable,
                   sha256={str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()})
    (root / 'release.json').write_text(json.dumps(release))
    return root, maps


def runtime(bundle):
    root, maps = bundle
    return BundledDenseUVRuntime(root / 'parser.pt', maps, 'SKING_DDJ_v104')


def input_bytes():
    out = io.BytesIO()
    Image.new('RGB', (256, 256)).save(out, format='PNG')
    return out.getvalue()


def test_rejects_changed_release_or_mappings(bundle):
    root, maps = bundle
    runtime(bundle)
    (maps / 'view.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='mappings integrity'):
        runtime(bundle)
    (maps / 'view.pt').write_bytes(b'mapping')
    (root / 'parser.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='release integrity'):
        runtime(bundle)


@pytest.mark.parametrize('corrupt', [False, True])
def test_output_provenance_and_temporary_cleanup(bundle, monkeypatch, corrupt):
    r = runtime(bundle)
    paths = []
    data = input_bytes()
    def fake_run(command, cwd, env, log_path):
        work = log_path.parent
        paths.append(work)
        assert (work / 'input.png').read_bytes() == data
        assert 'foreground_batch' in command and '--foreground-checkpoint' in command
        assert env['HF_HUB_OFFLINE'] == '1'
        out = work / 'output/sample'
        out.mkdir(parents=True)
        manifest = dict(complete=True, input_sha256='wrong' if corrupt else hashlib.sha256(data).hexdigest(),
                        checkpoint_sha256=r.release['sha256']['parser.pt'], pipeline={'version':'v104'},
                        source_sha256={'run_local.py':r.release['sha256']['runtime/dense_uv_parser/run_local.py']},
                        foreground_provider=dict(adaptation_sha256=r.release['sha256']['foreground.pt'],
                                                 model_sha256=r.release['sha256']['foreground_base/model.safetensors']))
        (out / 'manifest.json').write_text(json.dumps(manifest))
        Image.new('RGBA', (64, 64), (1, 2, 3, 255)).save(out / 'pred_uv.png')
    monkeypatch.setattr(r, '_run', fake_run)
    if corrupt:
        with pytest.raises(RuntimeError, match='provenance'):
            r.infer_png(data)
    else:
        assert Image.open(io.BytesIO(r.infer_png(data))).size == (64, 64)
    assert paths and not paths[0].exists()


@pytest.mark.skipif(sys.platform != 'linux', reason='Production process groups run on Linux')
def test_timeout_kills_parser_grandchildren(bundle, tmp_path):
    r = runtime(bundle)
    r.timeout_seconds = 1
    pidfile = tmp_path / 'child.pid'
    script = ('import subprocess,sys,time; from pathlib import Path; '
              'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
              f'Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)')
    with pytest.raises(subprocess.TimeoutExpired):
        r._run([sys.executable, '-c', script], tmp_path, os.environ.copy(), tmp_path / 'log')
    pid = int(pidfile.read_text())
    for _ in range(20):
        status = Path(f'/proc/{pid}/stat')
        if not status.exists() or status.read_text().split()[2] == 'Z':
            break
        time.sleep(.05)
    else:
        pytest.fail('GPU grandchild survived the request timeout')


def test_v104_never_falls_back_to_legacy_and_old_models_still_work(bundle, monkeypatch):
    old = Mock(return_value='legacy')
    monkeypatch.setitem(sys.modules, 'dense_uv_runtime', SimpleNamespace(DenseUVInferenceRuntime=old))
    root, maps = bundle
    args = dict(toolkit_root='old', checkpoint_path=str(root / 'parser.pt'), mappings_dir=str(maps), device='cuda')
    assert create_dense_uv_runtime('SKING_DDJ_v61b', **args) == 'legacy'
    (root / 'release.json').unlink()
    with pytest.raises(FileNotFoundError):
        create_dense_uv_runtime('SKING_DDJ_v104', **args)
    assert old.call_count == 1
