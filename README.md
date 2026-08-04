# EntropyDrop GPU Worker

## SKING_DDJ_v54

`SKING_DDJ_v54` uses a dedicated GPU process that consumes
`queue_render_to_uv`. The process loads the Dense UV parser checkpoint,
SigLIP2, and renderer mappings once at startup and retains them for all jobs.

Required runtime files:

```text
/root/SkingToolkit
/root/Sking/SKING_DDJ_v54.pt
/root/differentiable_minecraft_renderer/mappings_256x512
```

The paths can be overridden with `SKING_TOOLKIT_ROOT`,
`DENSE_UV_CHECKPOINT_PATH`, and `DENSE_UV_MAPPINGS_DIR`.

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
