# EntropyDrop GPU Worker

## SKING_DDJ series

`SKING_DDJ` uses a dedicated GPU process that consumes `queue_render_to_uv`.
Each process retains one active model version. Switching versions first releases
the previous runtime, so old releases cannot accumulate in GPU memory.

Immutable bundled releases use two resident subprocesses: foreground inference
in the release's foreground Python environment, and parser/SigLIP2/renderer
inference in its parser environment. The original release entry points still
produce and validate their provenance manifests; release files are not edited.
Models and renderers are loaded once per subprocess and reused across jobs.
Request deadlines kill the subprocess groups, a crashed stage is recreated on
the next request, and a Linux pidfd monitor prevents orphaned CUDA processes
without tying model lifetime to a temporary launcher thread.

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

Start a dedicated Dense UV worker:

```bash
CUDA_VISIBLE_DEVICES=0 python run_worker.py queue_render_to_uv
```

DS defaults to three Dense UV workers on GPU 0, alongside five CPU/I/O
real-to-render workers. `/home/ds/start_sking_services.sh` manages all instances;
the enabled `sking-generation.service` invokes it at boot. Use its
`restart-dense` action to reload only the three Dense UV workers with a warm
shutdown timeout, or `add-dense` to create missing instances without restarting
existing ones.

The DS launcher preloads the current v101c release before listening for jobs:

```bash
export DENSE_UV_EXECUTION_MODE=resident
export DENSE_UV_PRELOAD_MODEL_VERSION=SKING_DDJ_v101c
export DENSE_UV_PRELOAD_CHECKPOINT_FILE=SKING_DDJ_v101c/parser.pt
export DENSE_UV_PRELOAD_MAPPINGS_DIR=mappings_256x512
```

Without preload variables, the first job loads the requested model and keeps it
resident. `DENSE_UV_EXECUTION_MODE=oneshot` retains the original subprocess-per-job
implementation for output comparisons or rollback. On DS, apply that override
with `DENSE_UV_EXECUTION_MODE=oneshot /home/ds/start_sking_services.sh restart-dense`.
Concurrency shares a GPU with Flux and must be sized against peak inference
memory, including the allocator caches retained after warmup.

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
