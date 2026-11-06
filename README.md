# EntropyDrop GPU Worker

## SKING_DDJ series

`SKING_DDJ` uses a dedicated GPU process that consumes `queue_render_to_uv`.
Each process retains one active model version. Switching versions first releases
the previous runtime, so old releases cannot accumulate in GPU memory.

Earlier immutable bundled releases use two resident subprocesses: foreground inference
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
parameters. Legacy versions have no model-to-pipeline registry. The worker joins their
relative values to `SKING_ROOT_DIR` and `DMR_ROOT_DIR`; `SKING_TOOLKIT_ROOT` and
`DENSE_UV_DEVICE` also remain deployment settings.

Start a dedicated Dense UV worker:

```bash
CUDA_VISIBLE_DEVICES=0 python run_worker.py queue_render_to_uv
```

DS defaults to one Dense UV worker on GPU 0, alongside five CPU/I/O
real-to-render workers. `/home/ds/start_sking_services.sh` manages all instances;
the enabled `sking-generation.service` invokes it at boot. The launcher's
`DENSE_UV_INSTANCES=1` sets the persistent Dense UV instance count. Use its
`restart-dense` action to reload the configured Dense UV workers with a warm
shutdown timeout, or `add-dense` to create missing instances without restarting
existing ones. When reducing the count, also delete the surplus
`sking-v61-render-to-uv-N` entries from PM2 and run `pm2 save`; changing the
launcher alone does not remove already running instances.

The DS launcher preloads the v130 release before listening for jobs:

```bash
export DENSE_UV_EXECUTION_MODE=resident
export DENSE_UV_PRELOAD_MODEL_VERSION=SKING_DDJ_v130
export DENSE_UV_PRELOAD_CHECKPOINT_FILE=SKING_DDJ_v130/parser.pt
export DENSE_UV_PRELOAD_MAPPINGS_DIR=SKING_DDJ_v130/mappings
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

## SKING_DDJ_v130

The new version is packaged in `releases/SKING_DDJ_v130` and uses one resident
semantic UV subprocess. It accepts a 512x512 `front_left + back_left` edited image
and returns a 64x64 RGBA skin, with head outer-alpha threshold 0.25. Its logical
asset paths resolve inside this repository; existing versions keep their original
asset roots and runtime behavior. See [SKING_DDJ_v130.md](SKING_DDJ_v130.md) for
task parameters, deployment requirements, and validation. The optional preload
settings are in `SKING_DDJ_v130.env.example`.

Production cutover (2026-11-01): only `SKING_DDJ_v130` is enabled for new skin tasks. The dedicated UV worker preloads this release and sets `DENSE_UV_ALLOWED_MODEL_VERSIONS=SKING_DDJ_v130`. The old image-to-skin worker is removed from PM2 and boot startup. Historical model assets remain on disk. Image-to-skin generation is now enabled for SKING_DDJ_v130; text-to-skin and image-edit-to-skin remain paused.

## SKING_DDJ_v131 (parallel release)

The new provider-independent edited-to-UV release is available alongside v130. It adds background-invariant torso inference and conservative front/back outer-layer correction. Keep both versions in `DENSE_UV_ALLOWED_MODEL_VERSIONS`; v130 is not retired by this update. Upstream Nano Banana request parameters stay unchanged. See [SKING_DDJ_v131.md](SKING_DDJ_v131.md) and `SKING_DDJ_v131.env.example`.
