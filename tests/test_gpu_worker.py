from types import SimpleNamespace

from rq import SimpleWorker

from gpu_worker import (
    GPUWorker,
    IMAGE_TO_SKIN_JOB_TIMEOUT,
    enforce_image_to_skin_timeout,
)


def test_enforces_infinite_timeout_for_old_image_to_skin_job():
    job = SimpleNamespace(
        func_name="worker_tasks.task_image_to_skin",
        timeout=120,
    )

    assert enforce_image_to_skin_timeout(job) is True
    assert job.timeout == IMAGE_TO_SKIN_JOB_TIMEOUT == -1


def test_preserves_timeout_for_other_gpu_jobs():
    job = SimpleNamespace(
        func_name="worker_tasks.task_render_to_uv",
        timeout=120,
    )

    assert enforce_image_to_skin_timeout(job) is False
    assert job.timeout == 120


def test_gpu_worker_overrides_timeout_before_rq_prepares_execution(monkeypatch):
    observed_timeouts = []

    def fake_execute_job(self, job, queue):
        observed_timeouts.append(job.timeout)

    monkeypatch.setattr(SimpleWorker, "execute_job", fake_execute_job)
    worker = object.__new__(GPUWorker)
    worker.name = "test-worker"
    worker.log = SimpleNamespace(warning=lambda *args: None)
    job = SimpleNamespace(
        id="old-job",
        func_name="worker_tasks.task_image_to_skin",
        timeout=120,
    )

    worker.execute_job(job, SimpleNamespace())

    assert observed_timeouts == [-1]
