# EntropyDrop GPU Worker

## SKING_DDJ series

`SKING_DDJ` uses a dedicated GPU process that consumes `queue_render_to_uv`.
The process loads the Dense UV parser checkpoint, SigLIP2, and renderer mappings
from the first task and retains each distinct asset combination for later jobs.

Required runtime files:

```text
/root/SkingToolkit
/root/Sking/SKING_DDJ_v66.pt
/root/differentiable_minecraft_renderer/mappings_256x512
```

The backend sends `dense_uv_checkpoint_file` and `DMR_mappings_dir` as task
parameters. This worker has no model-to-pipeline registry. It joins those
relative values to `SKING_ROOT_DIR` and `DMR_ROOT_DIR`; `SKING_TOOLKIT_ROOT` and
`DENSE_UV_DEVICE` also remain deployment settings.

Start exactly one Dense UV process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python run_worker.py queue_render_to_uv
```

Alternatively, set `GPU_WORKER_QUEUE=queue_render_to_uv` and run
`python run_worker.py` without queue arguments.

Do not combine `queue_render_to_uv` with the Flux queues in the same process.
The worker refuses that configuration so Dense UV cannot accidentally share a
GPU process with a separately loaded Flux pipeline.

Dense UV jobs are enqueued with five bounded retries by both the stage-1 worker
and backend recovery path. A transient attempt reports `processing_skin`; only
the final exhausted attempt reports `failed`, so a paid Provider result is not
discarded because of a temporary S3, Redis, or GPU-worker interruption.

The Dense UV stage reports its normalized real-to-render input as
`image_to_skin_edited_result`. `edited_result` remains reserved for the
text-to-skin or image-edit-to-skin first-stage artifact.
