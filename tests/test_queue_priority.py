"""Test the real stage handoff without loading GPU weights or live settings."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from queue_priority import effective_queue_prefix


@pytest.mark.parametrize("original,promoted,expected", [
    ("", None, ""), ("", b"1", "high_"), ("", "1", "high_"),
    ("high_", None, "high_"),
])
def test_effective_priority(original, promoted, expected):
    connection = SimpleNamespace(get=Mock(return_value=promoted))
    assert effective_queue_prefix(connection, "skin-id", original) == expected
    if original == "high_":
        connection.get.assert_not_called()
    else:
        connection.get.assert_called_once_with("generation:pro_priority:skin-id")


@pytest.mark.parametrize("existing_lane", [None, "queue_image_to_skin", "high_queue_image_to_skin"])
def test_promoted_stage_handoff_uses_high_queue_and_checks_both_lanes(existing_lane):
    tree = ast.parse((Path(__file__).parents[1] / "worker_tasks.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "enqueue_image_to_skin_once")
    enqueue = Mock(return_value=object())
    existing = object()
    checked = []

    def fetch(queue, job_id):
        checked.append(queue.name)
        assert job_id == "generation_skin_image_to_skin"
        return existing if queue.name == existing_lane else None

    class FakeQueue:
        def __init__(self, name, connection):
            self.name = name

        def enqueue(self, *args, **kwargs):
            assert self.name == "high_queue_image_to_skin"
            assert kwargs["job_id"] == "generation_skin_image_to_skin"
            return enqueue(*args, **kwargs)

    namespace = dict(
        Queue=FakeQueue, effective_queue_prefix=effective_queue_prefix,
        redis_conn=SimpleNamespace(get=lambda key: b"1"),
        make_generation_job_id=lambda log_id, stage: f"generation_{log_id}_{stage}",
        fetch_active_job=fetch, IMAGE_TO_SKIN_JOB_TIMEOUT=-1,
        retry_policy=object(), RECOVERABLE_MODEL_VERSIONS={"legacy-model"},
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "worker_tasks.py", "exec"), namespace)
    result, created = namespace[function.name]("skin", True, "intermediate.png", "prompt", 4.0, "legacy-model", "aux", 1, 28, "")
    if existing_lane:
        assert result is existing and not created
        enqueue.assert_not_called()
    else:
        assert created
        enqueue.assert_called_once()
    assert checked == (["queue_image_to_skin"] if existing_lane == "queue_image_to_skin" else ["queue_image_to_skin", "high_queue_image_to_skin"])
