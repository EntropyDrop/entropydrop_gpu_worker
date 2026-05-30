import os
import sys
import time

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
from config import settings, load_redis_urls

current_redis_index = 0

def get_next_redis_connection():
    global current_redis_index
    attempts = 0
    while True:
        # Hot-reload the URLs list dynamically from disk
        urls_list = load_redis_urls()
        
        # Guard index in case list size changed dynamically
        current_redis_index = current_redis_index % len(urls_list)
        url = urls_list[current_redis_index]
        
        print(f"[*] Attempting to connect to Redis URL ({current_redis_index + 1}/{len(urls_list)}): {url}")
        try:
            conn = Redis.from_url(url, health_check_interval=10, retry_on_timeout=True)
            conn.ping()
            print(f"[✓] Successfully connected to Redis!")
            return conn, url
        except Exception as e:
            attempts += 1
            print(f"[!] Failed to connect to Redis at {url}: {e}")
            current_redis_index = (current_redis_index + 1) % len(urls_list)
            
            # Backoff sleep to avoid Hammering CPU during long outages
            sleep_time = min(5, attempts)
            print(f"  [*] Retrying Redis connection loop in {sleep_time}s...")
            time.sleep(sleep_time)

# Bind to a specific GPU (e.g., "0" to use the first card).
GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0")
WORKER_RESTART_DELAY_SECONDS = int(os.getenv("WORKER_RESTART_DELAY_SECONDS", "5"))

print(f"[*] Starting GPU Worker binding to GPU: {GPU_ID}")
print(f"[*] Configuring Redis Cluster lists: Prioritizing redis_urls.txt file.")

# Read queues to listen to from command-line arguments. If no arguments are provided, listen to all queues by default!
requested_queues = sys.argv[1:] if len(sys.argv) > 1 else ['queue_text_to_image', 'queue_image_edit', 'queue_image_to_skin']

# Add a high-priority version (prefixed with high_) for each queue.
listen = [f"high_{q}" for q in requested_queues] + requested_queues

print(f"[*] Listening on queues: {listen}")

def run_worker():
    global current_redis_index
    worker_cls = Worker
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

    while True:
        # 1. Get a working Redis connection dynamically
        conn, active_url = get_next_redis_connection()
        
        # 2. Update the import-time connection in worker_tasks to match our active connection dynamically
        try:
            import worker_tasks
            worker_tasks.redis_conn = conn
        except Exception as e:
            print(f"[!] Warning syncing connection to worker_tasks: {e}")
            
        queues = [Queue(name, connection=conn) for name in listen]
        worker = worker_cls(queues, connection=conn)

        try:
            print(f"[*] Worker starting work loop on: {active_url}")
            worker.work(with_scheduler=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[!] Worker loop crashed on {active_url}: {exc}")

        # Cycle to the next connection on exit to try another host if this crashed
        urls_list = load_redis_urls()
        current_redis_index = (current_redis_index + 1) % len(urls_list)
        print(f"[*] Worker loop exited. Reconnecting in {WORKER_RESTART_DELAY_SECONDS}s...")
        time.sleep(WORKER_RESTART_DELAY_SECONDS)

if __name__ == '__main__':
    try:
        run_worker()
    except KeyboardInterrupt:
        print("\n[*] Worker stopped.")
        sys.exit(0)
