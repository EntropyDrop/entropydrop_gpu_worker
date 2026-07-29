import io
import uuid
import time
import httpx
import base64
from PIL import Image
import sys
import os
from config import settings, load_redis_urls
from s3_utils import (
    S3ObjectNotFoundError,
    S3OperationAbortedError,
    download_from_s3,
    upload_to_s3,
)
from mc_voxel_texture_resolver import resolve_voxel_consistency
import asyncio
import json
import redis
from rq import Queue, Retry, get_current_job
from rq.job import Job
from rq.registry import StartedJobRegistry, DeferredJobRegistry, ScheduledJobRegistry
from gpu_worker import IMAGE_TO_SKIN_JOB_TIMEOUT
from diffusers import Flux2KleinPipeline
import torch
# Select a working Redis connection dynamically at import time for worker_tasks
def _init_redis_conn():
    urls = load_redis_urls()
    for url in urls:
        try:
            conn = redis.from_url(
                url,
                health_check_interval=10,
                socket_timeout=60,
                socket_connect_timeout=60,
                retry_on_timeout=True
            )
            conn.ping()
            return conn
        except Exception:
            pass
    # Fallback to first if all fail
    return redis.from_url(
        urls[0],
        health_check_interval=10,
        socket_timeout=60,
        socket_connect_timeout=60,
        retry_on_timeout=True
    )

redis_conn = _init_redis_conn()

q_t2i = Queue('queue_text_to_image', connection=redis_conn)
q_edit = Queue('queue_image_edit', connection=redis_conn)
q_skin = Queue('queue_image_to_skin', connection=redis_conn)
retry_policy = Retry(max=99999, interval=[5, 10, 30, 60])
RESULT_QUEUE_KEY = os.getenv("GENERATE_RESULT_QUEUE_KEY", "generate_results")
GENERATION_CANCELLATION_KEY_PREFIX = os.getenv(
    "GENERATION_CANCELLATION_KEY_PREFIX",
    "generation:cancelled:",
)
RECOVERABLE_MODEL_VERSIONS = {
    version.strip()
    for version in os.getenv(
        "RECOVERABLE_MODEL_VERSIONS",
        "sking_v73_flux_4b_000027000",
    ).split(",")
    if version.strip()
}

# Pipeline variable initialized by run_worker.py
img_to_skin_pipe = None
img_edit_pipe = None
text_to_img_pipe = None
skin_gen_prompt_embeds = None
dense_uv_pipe = None

# Track currently loaded LoRA for img_to_skin_pipe
current_lora_name = None


def init_dense_uv_pipeline():
    global dense_uv_pipe
    if dense_uv_pipe is not None:
        return dense_uv_pipe

    from dense_uv_runtime import DenseUVInferenceRuntime

    print(
        "[*] Loading Dense UV pipeline "
        f"from {settings.DENSE_UV_CHECKPOINT_PATH}..."
    )
    dense_uv_pipe = DenseUVInferenceRuntime(
        toolkit_root=settings.SKING_TOOLKIT_ROOT,
        checkpoint_path=settings.DENSE_UV_CHECKPOINT_PATH,
        mappings_dir=settings.DENSE_UV_MAPPINGS_DIR,
        device=settings.DENSE_UV_DEVICE,
    )
    print("[*] Dense UV checkpoint, SigLIP2, and mappings loaded.")
    return dense_uv_pipe

def init_text_to_img_pipeline():
    global text_to_img_pipe
    if text_to_img_pipe is not None:
        return text_to_img_pipe
    import torch
    from diffusers import ZImagePipeline
    from diffusers.utils import logging as diffusers_logging
    import transformers
    import os
    
    # Disable all related progress bars to completely prevent Broken Pipe errors caused by progress bar output to sys.stderr
    diffusers_logging.disable_progress_bar()
    transformers.utils.logging.disable_progress_bar()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    text_to_img_pipe = ZImagePipeline.from_pretrained(
        settings.ZIMAGE_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage = False,
    )
    text_to_img_pipe.to("cuda")
    text_to_img_pipe.set_progress_bar_config(disable=True)
    print("[*] Model loaded.")
    return text_to_img_pipe
def init_img_edit_pipeline():
    global img_edit_pipe
    if img_edit_pipe is not None:
        return img_edit_pipe
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.utils import logging as diffusers_logging
    import transformers
    import os
    
    # Disable all related progress bars to completely prevent Broken Pipe errors caused by progress bar output to sys.stderr
    diffusers_logging.disable_progress_bar()
    transformers.utils.logging.disable_progress_bar()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    print("[*] Loading Flux2KleinPipeline (Progress bar disabled)...")
    img_edit_pipe = Flux2KleinPipeline.from_pretrained(
        settings.FLUX_MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    img_edit_pipe.to("cuda")
    img_edit_pipe.set_progress_bar_config(disable=True)
    print("[*] Model loaded.")
    return img_edit_pipe

def init_img_to_skin_pipeline():
    global img_to_skin_pipe, current_lora_name, skin_gen_prompt_embeds
    if img_to_skin_pipe is not None:
        return img_to_skin_pipe
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.utils import logging as diffusers_logging
    import transformers
    import os
    
    # Disable all related progress bars to completely prevent Broken Pipe errors caused by progress bar output to sys.stderr
    diffusers_logging.disable_progress_bar()
    transformers.utils.logging.disable_progress_bar()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    print("[*] Loading Flux2KleinPipeline (Progress bar disabled)...")
    img_to_skin_pipe = Flux2KleinPipeline.from_pretrained(
        settings.FLUX_MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    img_to_skin_pipe.to("cuda")
    #img_to_skin_pipe.load_lora_weights(settings.FLUX_LORA_PATH)
    img_to_skin_pipe.set_progress_bar_config(disable=True)

    prompt_encoding = img_to_skin_pipe.encode_prompt(
        prompt="",
        device="cuda",
        num_images_per_prompt=1
    )
    skin_gen_prompt_embeds = prompt_encoding[0] if isinstance(prompt_encoding, tuple) else prompt_encoding
    print("[*] Model loaded.")
    return img_to_skin_pipe
# Pipeline variable for lazy initialization
from utils import remove_bg

def make_generation_job_id(log_id: str, stage: str) -> str:
    return f"generation_{log_id}_{stage}"


def generation_cancellation_key(log_id: str) -> str:
    return f"{GENERATION_CANCELLATION_KEY_PREFIX}{log_id}"


def is_generation_cancelled(log_id: str) -> bool:
    try:
        return bool(redis_conn.exists(generation_cancellation_key(log_id)))
    except Exception as exc:
        # A Redis outage must not be mistaken for a user cancellation.
        print(f"[!] [{log_id}] Failed to check cancellation state: {exc}")
        return False


def ensure_generation_active(log_id: str) -> None:
    if is_generation_cancelled(log_id):
        raise S3OperationAbortedError(
            f"Generation {log_id} was cancelled"
        )


def get_registry_job_ids(registry):
    try:
        return registry.get_job_ids(cleanup=False)
    except TypeError:
        return registry.get_job_ids()


def fetch_active_job(queue: Queue, job_id: str):
    try:
        if job_id in queue.get_job_ids():
            return Job.fetch(job_id, connection=redis_conn)
    except Exception:
        pass

    try:
        intermediate_queue = getattr(queue, "intermediate_queue", None)
        if intermediate_queue and job_id in intermediate_queue.get_job_ids():
            return Job.fetch(job_id, connection=redis_conn)
    except Exception:
        pass

    for RegistryCls in [StartedJobRegistry, DeferredJobRegistry, ScheduledJobRegistry]:
        try:
            registry = RegistryCls(name=queue.name, connection=redis_conn)
            if job_id in get_registry_job_ids(registry):
                return Job.fetch(job_id, connection=redis_conn)
        except Exception:
            pass
    return None


def get_job_intermediate_filename(job, fallback: str) -> str:
    try:
        if job and job.kwargs and job.kwargs.get("intermediate_filename"):
            return job.kwargs["intermediate_filename"]
        if job and job.args and len(job.args) >= 3:
            return job.args[2]
    except Exception:
        pass
    return fallback


def enqueue_image_to_skin_once(log_id: str, is_public: bool, intermediate_filename: str, prompt: str, guidance: float, model_version: str, aux_model_version: str, seed: int, n_step: int, queue_prefix: str):
    target_q_skin = Queue(f"{queue_prefix}queue_image_to_skin", connection=redis_conn)
    job_id = make_generation_job_id(log_id, "image_to_skin")
    active_job = fetch_active_job(target_q_skin, job_id)
    if active_job:
        print(f"[*] [{log_id}] image_to_skin job already active: {job_id}")
        return active_job, False

    job = target_q_skin.enqueue(
        "worker_tasks.task_image_to_skin",
        args=(log_id, is_public, intermediate_filename, "image/jpeg", prompt),
        kwargs={"intermediate_filename": intermediate_filename, "guidance": guidance, "model_version": model_version, "aux_model_version": aux_model_version, "seed": seed, "n_step": n_step},
        job_timeout=IMAGE_TO_SKIN_JOB_TIMEOUT,
        retry=retry_policy if model_version in RECOVERABLE_MODEL_VERSIONS else None,
        result_ttl=10,
        job_id=job_id
    )
    return job, True


def report_status(log_id: str, status: str, result: str = None, edited_result: str = None, error_msg: str = None, source: str = None, stage: str = None, pipeline_version: str = None):
    """Push status report to Redis"""
    payload = {"log_id": log_id, "status": status}
    if result is not None: payload["result"] = result
    if edited_result is not None: payload["edited_result"] = edited_result
    if error_msg is not None: payload["error_msg"] = error_msg
    if source is not None: payload["source"] = source
    if stage is not None: payload["stage"] = stage
    if pipeline_version is not None: payload["pipeline_version"] = pipeline_version
    redis_conn.lpush(RESULT_QUEUE_KEY, json.dumps(payload))

def process_and_upload_final_skin(
    img_data_bytes: bytes,
    s3id_result: str,
    is_public: bool,
    abort_if=None,
) -> str:
    t_start = time.time()
    img = Image.open(io.BytesIO(img_data_bytes))
    img = img.crop((0, 0, img.width // 2, img.height // 2))
    
    t_bg = time.time()
    img = remove_bg(img)
    
    t_voxel = time.time()
    img = resolve_voxel_consistency(img)
    
    t_post = time.time()
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    filename = f"generations/{s3id_result}.png"
    
    t_up = time.time()
    upload_to_s3(
        img_io.getvalue(),
        filename,
        is_public,
        "image/png",
        abort_if=abort_if,
    )
    t_end = time.time()
    
    print(f"[*] Crop/Format: {t_bg - t_start:.2f}s, RemoveBG: {t_voxel - t_bg:.2f}s, Voxel: {t_post - t_voxel:.2f}s, S3 Upload: {t_end - t_up:.2f}s, Total post-process: {t_end - t_start:.2f}s")
    return filename

async def task_text_to_image_async(log_id: str, is_public: bool, prompt: str, model_version: str, aux_model_version: str = None, seed: int = None, n_step: int = None, guidance: float = None):
    abort_if = lambda: is_generation_cancelled(log_id)
    try:
        ensure_generation_active(log_id)
        report_status(log_id, "processing", stage="text_to_image")
        
        global text_to_img_pipe
        if text_to_img_pipe is None:
            print("[*] Lazy loading ZImagePipeline (Fallback)...")
            init_text_to_img_pipeline()
        assert text_to_img_pipe is not None
            
        # Use local ZImagePipeline to generate image
        # Force-disable progress bar before each call and redirect tqdm output to devnull
        import os as _os
        text_to_img_pipe.set_progress_bar_config(disable=True, file=open(_os.devnull, 'w'))
        
        t_pipe_start = time.time()
        pipeline_output = text_to_img_pipe(
            prompt=prompt or "",
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            num_images_per_prompt=1,
            generator=torch.Generator("cuda").manual_seed(seed if seed is not None else 42),
        )
        images = pipeline_output.images
        t_pipe_end = time.time()
        print(f"[*] [{log_id}] ZImagePipeline inference took {t_pipe_end - t_pipe_start:.2f}s")
        ensure_generation_active(log_id)
        
        # Convert generated image to JPEG bytes
        img_io = io.BytesIO()

        # resize to 768x768
        images[0] = images[0].resize((768, 768))
        images[0].convert("RGB").save(img_io, format="JPEG", quality=95)
        img_data = img_io.getvalue()
            
        # upload intermediate
        s3id_result = log_id
        intermediate_filename = f"text_to_image_intermediate/{s3id_result}.jpg"
        
        t_up_start = time.time()
        upload_to_s3(
            img_data,
            intermediate_filename,
            is_public,
            "image/jpeg",
            abort_if=abort_if,
        )
        t_up_end = time.time()
        print(f"[*] [{log_id}] Upload intermediate to S3 took {t_up_end - t_up_start:.2f}s")
        ensure_generation_active(log_id)
        
        # Dispatch to the specialized queue for the next stage: image_to_skin
        job = get_current_job()
        current_queue_name = job.origin if job else ""
        prefix = "high_" if current_queue_name.startswith("high_") else ""

        skin_job, _ = enqueue_image_to_skin_once(
            log_id,
            is_public,
            intermediate_filename,
            prompt,
            guidance,
            model_version,
            aux_model_version,
            seed,
            n_step,
            prefix,
        )

        # Report after stage 2 is durably enqueued so a crash cannot leave
        # the DB in pending_skin without a follow-up Redis job.
        report_status(
            log_id,
            "pending_skin",
            edited_result=get_job_intermediate_filename(skin_job, intermediate_filename),
            stage="text_to_image",
        )
    except S3OperationAbortedError as exc:
        print(f"[*] [{log_id}] Text-to-image task cancelled: {exc}")
        return
    except S3ObjectNotFoundError as exc:
        print(f"[!] [{log_id}] Text-to-image input is missing: {exc}")
        report_status(
            log_id,
            "failed",
            error_msg=str(exc),
            stage="text_to_image",
        )
        return
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"[{log_id}] Text-to-image Task failed with exception:\n{err_detail}")
        report_status(log_id, "failed", error_msg=f"{str(e)}\n\n{err_detail}", stage="text_to_image")
        raise e

async def task_image_edit_async(log_id: str, is_public: bool, source: str, content_type: str, prompt: str, model_version: str, aux_model_version: str = None, seed: int = None, n_step: int = None, guidance: float = None):
    abort_if = lambda: is_generation_cancelled(log_id)
    try:
        ensure_generation_active(log_id)
        report_status(log_id, "processing", stage="image_edit")
        
        t_dl_start = time.time()
        file_content = download_from_s3(
            source,
            is_public,
            abort_if=abort_if,
        )
        t_dl_end = time.time()
        print(f"[*] [{log_id}] Download image from S3 took {t_dl_end - t_dl_start:.2f}s")

        global img_edit_pipe
        if img_edit_pipe is None:
            print("[*] Lazy loading Flux2KleinPipeline (Fallback)...")
            init_img_edit_pipeline()
        assert img_edit_pipe is not None
            
        # Use local Flux2KleinPipeline to generate edited image
        img = Image.open(io.BytesIO(file_content)).convert("RGB")
        # Force-disable progress bar before each call and redirect tqdm output to devnull
        # Prevents BrokenPipeError caused by tqdm flushing stderr after SSH disconnection
        import os as _os
        img_edit_pipe.set_progress_bar_config(disable=True, file=open(_os.devnull, 'w'))
        
        t_pipe_start = time.time()
        pipeline_output = img_edit_pipe(
            image=img,
            prompt=prompt or "",
            height=768,
            width=768,
            num_inference_steps=n_step if n_step is not None else 30,
            guidance_scale=guidance if guidance is not None else 4.0,
            num_images_per_prompt=1,
            generator=torch.Generator("cuda").manual_seed(seed if seed is not None else 42),
        )
        images = pipeline_output.images
        t_pipe_end = time.time()
        print(f"[*] [{log_id}] Flux2KleinPipeline (edit) inference took {t_pipe_end - t_pipe_start:.2f}s")
        ensure_generation_active(log_id)
        
        # Convert generated image to JPEG bytes
        img_io = io.BytesIO()
        images[0].convert("RGB").save(img_io, format="JPEG", quality=95)
        img_data = img_io.getvalue()
            
        s3id_result = log_id
        intermediate_filename = f"image_edit/{s3id_result}.jpg"
        
        t_up_start = time.time()
        upload_to_s3(
            img_data,
            intermediate_filename,
            is_public,
            "image/jpeg",
            abort_if=abort_if,
        )
        t_up_end = time.time()
        print(f"[*] [{log_id}] Upload intermediate to S3 took {t_up_end - t_up_start:.2f}s")
        ensure_generation_active(log_id)
        
        # Dispatch to the specialized queue for the next stage: image_to_skin
        job = get_current_job()
        current_queue_name = job.origin if job else ""
        prefix = "high_" if current_queue_name.startswith("high_") else ""

        skin_job, _ = enqueue_image_to_skin_once(
            log_id,
            is_public,
            intermediate_filename,
            prompt,
            guidance,
            model_version,
            aux_model_version,
            seed,
            n_step,
            prefix,
        )

        # Report after stage 2 is durably enqueued so a crash cannot leave
        # the DB in pending_skin without a follow-up Redis job.
        report_status(
            log_id,
            "pending_skin",
            edited_result=get_job_intermediate_filename(skin_job, intermediate_filename),
            stage="image_edit",
        )
    except S3OperationAbortedError as exc:
        print(f"[*] [{log_id}] Image-edit task cancelled: {exc}")
        return
    except S3ObjectNotFoundError as exc:
        print(f"[!] [{log_id}] Image-edit input is missing: {exc}")
        report_status(
            log_id,
            "failed",
            error_msg=str(exc),
            stage="image_edit",
        )
        return
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"[{log_id}] Image-edit Task failed with exception:\n{err_detail}")
        report_status(log_id, "failed", error_msg=f"{str(e)}\n\n{err_detail}", stage="image_edit")
        raise e

async def task_image_to_skin_async(log_id: str, is_public: bool, source: str, content_type: str, prompt: str, model_version: str = None, aux_model_version: str = None, seed: int = None, n_step: int = None, guidance: float = None, intermediate_filename: str = None):
    abort_if = lambda: is_generation_cancelled(log_id)
    try:
        ensure_generation_active(log_id)
        report_status(log_id, "processing_skin", stage="image_to_skin")
        
        t_dl_start = time.time()
        file_content = download_from_s3(
            source,
            is_public,
            abort_if=abort_if,
        )
        t_dl_end = time.time()
        print(f"[*] [{log_id}] Download image from S3 took {t_dl_end - t_dl_start:.2f}s")

        global img_to_skin_pipe, current_lora_name, skin_gen_prompt_embeds
        if img_to_skin_pipe is None:
            print("[*] Lazy loading Flux2KleinPipeline (Fallback)...")
            init_img_to_skin_pipeline()
        
        # Dynamic LoRA loading based on model_version
        requested_lora = model_version + '.safetensors'
        if current_lora_name != requested_lora:
            print(f"[*] Switching LoRA: {current_lora_name} -> {requested_lora}")
            lora_dir = settings.FLUX_LORA_DIR
            if os.path.exists(os.path.join(lora_dir, requested_lora)):
                try:
                    t0 = time.time()
                    if current_lora_name is not None and hasattr(img_to_skin_pipe, "unload_lora_weights"):
                        img_to_skin_pipe.unload_lora_weights()
                    t1 = time.time()
                    img_to_skin_pipe.load_lora_weights(lora_dir, weight_name=requested_lora)
                    t2 = time.time()
                    current_lora_name = requested_lora
                    print(f"[*] LoRA switched to: {requested_lora} (unload: {t1-t0:.2f}s, load: {t2-t1:.2f}s, total: {t2-t0:.2f}s)")
                except Exception as le:
                    print(f"[*] Failed to switch LoRA to {requested_lora}: {le}")
                    # Raise the exception to terminate the task immediately
                    raise le
            else:
                raise FileNotFoundError(f"Requested LoRA file not found: {requested_lora}")
            
        # Use local Flux2KleinPipeline to generate skin image
        img = Image.open(io.BytesIO(file_content)).convert("RGBA")
        # Force-disable progress bar before each call and redirect tqdm output to devnull
        # Prevents BrokenPipeError caused by tqdm flushing stderr after SSH disconnection
        import os as _os
        img_to_skin_pipe.set_progress_bar_config(disable=True, file=open(_os.devnull, 'w'))
        
        t_pipe_start = time.time()
        pipeline_output = img_to_skin_pipe(
            image=img,
            prompt=None,
            prompt_embeds = skin_gen_prompt_embeds,
            height=768,
            width=768,
            num_inference_steps=n_step if n_step is not None else 100,
            guidance_scale=guidance if guidance is not None else 4.0,
            num_images_per_prompt=1,
            generator=torch.Generator("cuda").manual_seed(seed if seed is not None else 42),
        )
        images = pipeline_output.images
        t_pipe_end = time.time()
        print(f"[*] [{log_id}] Flux2KleinPipeline (skin) inference took {t_pipe_end - t_pipe_start:.2f}s")
        ensure_generation_active(log_id)
        
        # Convert generated image to PNG bytes
        img_io = io.BytesIO()
        images[0].save(img_io, format="PNG")
        img_data = img_io.getvalue()

        # Post process
        s3id_result = log_id
        
        t_post_start = time.time()
        final_filename = process_and_upload_final_skin(
            img_data,
            s3id_result,
            is_public,
            abort_if=abort_if,
        )
        t_post_end = time.time()
        print(f"[*] [{log_id}] Complete post-processing and final upload took {t_post_end - t_post_start:.2f}s")

        report_status(log_id, "success", result=final_filename, edited_result=intermediate_filename, stage="image_to_skin")
    except S3OperationAbortedError as exc:
        print(f"[*] [{log_id}] Image-to-skin task cancelled: {exc}")
        return
    except S3ObjectNotFoundError as exc:
        print(f"[!] [{log_id}] Image-to-skin source is missing: {exc}")
        report_status(
            log_id,
            "failed",
            error_msg=str(exc),
            stage="image_to_skin",
        )
        return
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"[{log_id}] Task failed with exception:\n{err_detail}")
        report_status(log_id, "failed", error_msg=f"{str(e)}\n\n{err_detail}", stage="image_to_skin")
        raise e

def task_text_to_image(*args, **kwargs):
    asyncio.run(task_text_to_image_async(*args, **kwargs))

def task_image_edit(*args, **kwargs):
    asyncio.run(task_image_edit_async(*args, **kwargs))

def task_image_to_skin(*args, **kwargs):
    asyncio.run(task_image_to_skin_async(*args, **kwargs))


def task_render_to_uv(
    log_id: str,
    is_public: bool,
    source: str,
    content_type: str,
    pipeline_version: str,
):
    del content_type
    abort_if = lambda: is_generation_cancelled(log_id)
    try:
        ensure_generation_active(log_id)
        if pipeline_version != settings.DENSE_UV_PIPELINE_VERSION:
            raise ValueError(
                "Dense UV pipeline version mismatch: "
                f"task={pipeline_version!r}, "
                f"worker={settings.DENSE_UV_PIPELINE_VERSION!r}"
            )

        report_status(
            log_id,
            "processing_skin",
            stage="render_to_uv",
            pipeline_version=pipeline_version,
        )
        combined_render = download_from_s3(
            source,
            is_public,
            abort_if=abort_if,
        )
        skin_png = init_dense_uv_pipeline().infer_png(combined_render)
        ensure_generation_active(log_id)
        final_filename = f"generations/{log_id}.png"
        upload_to_s3(
            skin_png,
            final_filename,
            is_public,
            "image/png",
            abort_if=abort_if,
        )
        report_status(
            log_id,
            "success",
            result=final_filename,
            edited_result=source,
            stage="render_to_uv",
            pipeline_version=pipeline_version,
        )
    except S3OperationAbortedError as exc:
        print(f"[*] [{log_id}] Dense UV task cancelled: {exc}")
        return
    except S3ObjectNotFoundError as exc:
        print(f"[!] [{log_id}] Dense UV input is missing: {exc}")
        report_status(
            log_id,
            "failed",
            error_msg=str(exc),
            stage="render_to_uv",
            pipeline_version=pipeline_version,
        )
        return
    except Exception as exc:
        import traceback

        error_detail = traceback.format_exc()
        print(
            f"[{log_id}] Dense UV task failed with exception:\n"
            f"{error_detail}"
        )
        report_status(
            log_id,
            "failed",
            error_msg=f"{exc}\n\n{error_detail}",
            stage="render_to_uv",
            pipeline_version=pipeline_version,
        )
        raise
