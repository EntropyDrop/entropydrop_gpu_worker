import os
import sys

# Fix macOS crash due to Objective-C runtime safety checks when Python RQ forks child processes.
# To ensure this takes effect before all system-level dependencies are loaded, we force-restart the current Python process with this environment variable.
if os.environ.get("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Replace default sys.stderr with a safe stderr wrapper.
# When an SSH connection is disconnected, tqdm unconditionally calls sys.stderr.flush(), which causes a BrokenPipeError.
# This wrapper swallows those errors, ensuring the worker runs stably even in headless/no-terminal environments.
class SafeStream:
    def __init__(self, stream):
        self._stream = stream
    def write(self, data):
        try:
            self._stream.write(data)
        except BrokenPipeError:
            pass
    def flush(self):
        try:
            self._stream.flush()
        except BrokenPipeError:
            pass
    def __getattr__(self, name):
        return getattr(self._stream, name)

sys.stderr = SafeStream(sys.stderr)
sys.stdout = SafeStream(sys.stdout)

from redis import Redis
from rq import Worker, Queue, SimpleWorker
from config import settings

# Get central Redis URL (read from configuration/environment variables)
REDIS_URL = settings.REDIS_URL

# Bind to a specific GPU (e.g., "0" to use the first card).
# Can also be set before the startup command: CUDA_VISIBLE_DEVICES=0 python run_worker.py
GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0")

print(f"[*] Starting GPU Worker binding to GPU: {GPU_ID}")
print(f"[*] Connecting to Redis: {REDIS_URL}")

# Read queues to listen to from command-line arguments. If no arguments are provided, listen to all queues by default!
requested_queues = sys.argv[1:] if len(sys.argv) > 1 else ['queue_text_to_image', 'queue_image_edit', 'queue_image_to_skin']

# Add a high-priority version (prefixed with high_) for each queue.
# RQ processes queues in list order, so we place all high-priority queues first.
listen = [f"high_{q}" for q in requested_queues] + requested_queues

print(f"[*] Listening on queues: {listen}")

# Establish Redis connection.
# Include health_check_interval and retry_on_timeout to prevent Broken pipe errors caused by SSH tunnel disconnection or idle timeouts.
conn = Redis.from_url(REDIS_URL, health_check_interval=10, retry_on_timeout=True)

def run_worker():
    # Initialize Worker with connection
    queues = [Queue(name, connection=conn) for name in listen]
    
    worker_cls = Worker
    # Force the use of SimpleWorker if listening to any queue that requires a GPU to prevent CUDA re-initialization and model reloading caused by fork()
    gpu_queues = ['queue_text_to_image', 'queue_image_edit', 'queue_image_to_skin']
    if any(q in listen for q in gpu_queues) or any(f"high_{q}" in listen for q in gpu_queues):
        import worker_tasks
        
        if 'queue_text_to_image' in listen or 'high_queue_text_to_image' in listen:
            print("[*] Pre-loading ZImagePipeline for text_to_image tasks...")
            worker_tasks.init_text_to_img_pipeline()
            
        if 'queue_image_edit' in listen or 'high_queue_image_edit' in listen:
            print("[*] Pre-loading Flux2KleinPipeline for image_edit tasks...")
            worker_tasks.init_img_edit_pipeline()
            
        if 'queue_image_to_skin' in listen or 'high_queue_image_to_skin' in listen:
            print("[*] Pre-loading Flux2KleinPipeline for image_to_skin tasks...")
            worker_tasks.init_img_to_skin_pipeline()
            
        print("[*] All requested models loaded. Using SimpleWorker to maintain GPU memory.")
        worker_cls = SimpleWorker
        
    worker = worker_cls(queues, connection=conn)
    
    # Start the work loop.
    # It blocks and continuously waits for new tasks to arrive.
    worker.work(with_scheduler=True)

if __name__ == '__main__':
    try:
        run_worker()
    except KeyboardInterrupt:
        print("\n[*] Worker stopped.")
        sys.exit(0)
