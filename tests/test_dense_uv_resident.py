"""Real Linux IPC/process lifecycle tests, without loading CUDA or Redis."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from dense_uv_resident import ResidentProcess
from dense_uv_cache import DenseUVRuntimeCache


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Production worker uses Linux pidfds')


@pytest.fixture
def server(tmp_path):
    script = tmp_path / 'fake_stage.py'
    script.write_text('''
import json, os, subprocess, sys, time
from pathlib import Path
import dense_uv_resident as resident
def load(root, stage):
    calls = 0
    def infer(request):
        nonlocal calls
        calls += 1
        print('Library output is deliberately not part of the IPC protocol', flush=True)
        if request.get('fail'):
            raise ValueError('injected inference failure')
        if request.get('hang'):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            Path(request['output']).write_text(str(child.pid))
            time.sleep(60)
        Path(request['output']).write_text(json.dumps({'pid': os.getpid(), 'calls': calls}))
    return infer
resident._load_handler = load
resident.main()
''')
    env = {**os.environ, 'PYTHONPATH': str(Path.cwd())}
    command = [sys.executable, str(script), '--release-root', str(tmp_path), '--stage', 'parser']
    return command, env


def is_running(pid):
    status = Path(f'/proc/{pid}/stat')
    try:
        return status.read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def assert_stopped(pid):
    for _ in range(50):
        if not is_running(pid):
            return
        time.sleep(.02)
    pytest.fail(f'Resident process {pid} survived cleanup')


def test_requests_reuse_same_process_and_ignore_stdout(server, tmp_path):
    command, env = server
    process = ResidentProcess(command, tmp_path, env)
    output = tmp_path / 'result'
    try:
        process.request({'output': str(output)}, time.monotonic() + 5)
        first = json.loads(output.read_text())
        process.request({'output': str(output)}, time.monotonic() + 5)
        second = json.loads(output.read_text())
        assert second == {'pid': first['pid'], 'calls': 2}
        directory = Path(process.directory.name)
    finally:
        process.close()
    assert_stopped(first['pid'])
    assert not directory.exists()


def test_failed_request_terminates_process_and_next_request_recovers(server, tmp_path):
    command, env = server
    process = ResidentProcess(command, tmp_path, env)
    try:
        process.start(time.monotonic() + 5)
        old_pid = process.pid
        with pytest.raises(RuntimeError, match='injected inference failure'):
            process.request({'fail': True}, time.monotonic() + 5)
        assert process.pid is None
        assert_stopped(old_pid)
        output = tmp_path / 'result'
        process.request({'output': str(output)}, time.monotonic() + 5)
        assert process.pid != old_pid
        assert json.loads(output.read_text())['calls'] == 1
    finally:
        process.close()


def test_models_survive_their_spawning_thread_exiting(server, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    command, env = server
    process = ResidentProcess(command, tmp_path, env)
    output = tmp_path / 'result'
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(process.request, {'output': str(output)}, time.monotonic() + 5).result()
            original_pid = process.pid
        time.sleep(.1)
        assert is_running(original_pid)
        process.request({'output': str(output)}, time.monotonic() + 5)
        assert json.loads(output.read_text()) == {'pid': original_pid, 'calls': 2}
    finally:
        process.close()


def test_request_deadline_kills_stage_and_grandchild(server, tmp_path):
    command, env = server
    process = ResidentProcess(command, tmp_path, env)
    output = tmp_path / 'grandchild'
    try:
        process.start(time.monotonic() + 5)
        stage_pid = process.pid
        with pytest.raises(TimeoutError):
            process.request({'output': str(output), 'hang': True}, time.monotonic() + .5)
        assert_stopped(stage_pid)
        assert_stopped(int(output.read_text()))
    finally:
        process.close()


def test_parent_sigkill_does_not_orphan_resident_stage(server, tmp_path):
    command, env = server
    owner = tmp_path / 'owner.py'
    info = tmp_path / 'owner-info.json'
    owner.write_text(
        'import json, time\nfrom pathlib import Path\n'
        'from dense_uv_resident import ResidentProcess\n'
        f'p=ResidentProcess({command!r}, {str(tmp_path)!r}, {env!r})\n'
        'p.start(time.monotonic()+5)\n'
        f'Path({str(info)!r}).write_text(json.dumps(dict(pid=p.pid, directory=p.directory.name)))\n'
        'time.sleep(60)\n'
    )
    parent = subprocess.Popen([sys.executable, str(owner)], env=env)
    record = None
    try:
        deadline = time.monotonic() + 5
        while not info.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert info.exists()
        record = json.loads(info.read_text())
        parent.kill()
        parent.wait(timeout=5)
        assert_stopped(record['pid'])
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if record:
            import shutil
            shutil.rmtree(record['directory'], ignore_errors=True)


def test_cache_reuses_version_and_releases_before_loading_another():
    events = []
    class Runtime:
        def close(self):
            events.append('closed')
    cache = DenseUVRuntimeCache()
    first = cache.get_or_create('one', Runtime)
    assert cache.get_or_create('one', lambda: pytest.fail('loaded twice')) is first
    def create_second():
        assert events == ['closed']
        return Runtime()
    second = cache.get_or_create('two', create_second)
    assert second is not first
    cache.close()
    assert events == ['closed', 'closed']
    assert cache.runtime is None


def test_failed_model_switch_leaves_no_stale_runtime():
    cache = DenseUVRuntimeCache()
    cache.get_or_create('one', object)
    def fail():
        raise ValueError('bad release')
    with pytest.raises(ValueError, match='bad release'):
        cache.get_or_create('two', fail)
    assert cache.runtime is None and cache.key is None
