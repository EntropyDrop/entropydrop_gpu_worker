"""Exercise the reconnect loop without importing GPU models or live Redis settings."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('busy_shutdown', [False, True])
def test_warm_shutdown_exits_outer_reconnect_loop(monkeypatch, busy_shutdown):
    tree = ast.parse(Path('run_worker.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_worker')
    worker = SimpleNamespace(_stop_requested=busy_shutdown,
                             _shutdown_requested_date=object(), work=Mock())
    factory = Mock(return_value=worker)
    sleep = Mock(side_effect=AssertionError('A requested shutdown must not reconnect'))
    monkeypatch.setitem(sys.modules, 'worker_tasks', SimpleNamespace())
    namespace = dict(Worker=factory, GPUWorker=factory, listen=['queue_render_to_uv'],
                     get_next_redis_connection=lambda: (object(), 'test-redis'),
                     Queue=lambda *a, **kw: object(), load_redis_urls=lambda: ['test-redis'],
                     current_redis_index=0, time=SimpleNamespace(sleep=sleep), WORKER_RESTART_DELAY_SECONDS=5)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'run_worker.py', 'exec'), namespace)
    namespace['run_worker']()
    assert factory.call_count == 1
    worker.work.assert_called_once_with(with_scheduler=True)
    sleep.assert_not_called()
