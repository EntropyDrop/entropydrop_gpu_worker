from rq import SimpleWorker, get_current_job


IMAGE_TO_SKIN_TASK_NAME = "worker_tasks.task_image_to_skin"
IMAGE_TO_SKIN_JOB_TIMEOUT = -1


def current_job_will_retry() -> bool:
    job = get_current_job()
    return bool(job and (getattr(job, "retries_left", 0) or 0) > 0)


def enforce_image_to_skin_timeout(job) -> bool:
    """Disable RQ's deadline for image_to_skin jobs, including old queued jobs."""
    if getattr(job, "func_name", None) != IMAGE_TO_SKIN_TASK_NAME:
        return False

    job.timeout = IMAGE_TO_SKIN_JOB_TIMEOUT
    return True


class GPUWorker(SimpleWorker):
    """SimpleWorker that protects image_to_skin's internal infinite S3 retry."""

    def execute_job(self, job, queue):
        previous_timeout = getattr(job, "timeout", None)
        if enforce_image_to_skin_timeout(job) and previous_timeout != job.timeout:
            self.log.warning(
                "Worker %s: overriding timeout for job %s from %s to infinite",
                self.name,
                job.id,
                previous_timeout,
            )
        return super().execute_job(job, queue)
