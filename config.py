import os
import time

# Robustly try to load .env file using python-dotenv if available
try:
    from dotenv import load_dotenv
    dotenv_path = os.getenv("ENV_FILE", ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        print(f"[*] Warning: Environment file '{dotenv_path}' not found. Relying on system environment variables.")
except ImportError:
    print("[!] Warning: python-dotenv is not installed. Environment variables from .env will not be loaded!")

class WorkerSettings:
    # AWS S3 / CDN Config (Dedicated S3 Keys to avoid clash with EC2 discovery key)
    AWS_S3_ACCESS_KEY_ID: str = os.getenv("AWS_S3_ACCESS_KEY_ID", "")
    AWS_S3_SECRET_ACCESS_KEY: str = os.getenv("AWS_S3_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-2")
    AWS_BUCKET_NAME: str = os.getenv("AWS_BUCKET_NAME", "")
    AWS_PRIVATE_BUCKET_NAME: str = os.getenv("AWS_PRIVATE_BUCKET_NAME", "")
    AWS_CDN_DOMAIN: str = os.getenv("AWS_CDN_DOMAIN", "")

    def __init__(self):
        if not self.AWS_S3_ACCESS_KEY_ID or not self.AWS_S3_SECRET_ACCESS_KEY:
            print("[!] CRITICAL WARNING: AWS_S3_ACCESS_KEY_ID or AWS_S3_SECRET_ACCESS_KEY is empty in WorkerSettings!")

    # Redis Password Configuration
    AWS_REDIS_PASSWORD: str = os.getenv("AWS_REDIS_PASSWORD", "")

    # Cloudflare Tunnel Configuration
    CF_TUNNEL_HOSTNAME: str = os.getenv("CF_TUNNEL_HOSTNAME", "")
    CF_ACCESS_CLIENT_ID: str = os.getenv("CF_ACCESS_CLIENT_ID", "")
    CF_ACCESS_CLIENT_SECRET: str = os.getenv("CF_ACCESS_CLIENT_SECRET", "")

    # FLUX / AI Model Directory Settings
    FLUX_MODEL_DIR: str = os.getenv("FLUX_MODEL_DIR", "")
    FLUX_LORA_DIR: str = os.getenv("FLUX_LORA_DIR", "")
    ZIMAGE_MODEL_DIR: str = os.getenv("ZIMAGE_MODEL_DIR", "")

settings = WorkerSettings()

def load_redis_urls():
    """Robustly loads the Redis URLs list, prioritizing the dynamic redis_urls.txt file for hot-reloads.
    If the file exists but reading fails, it infinitely retries reading with a delay.
    """
    urls_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "redis_urls.txt")
    if os.path.exists(urls_file):
        attempts = 0
        while True:
            try:
                with open(urls_file, "r") as f:
                    content = f.read().strip()
                break  # Successful read
            except Exception as e:
                attempts += 1
                print(f"[!] Error reading redis_urls.txt (Attempt {attempts}): {e}. Retrying in 1s...")
                time.sleep(1)
        
        # Split by lines to filter out comments first
        lines = [line.strip() for line in content.splitlines()]
        clean_lines = [line for line in lines if line and not line.startswith("#")]
        
        urls = []
        for line in clean_lines:
            for part in line.split(","):
                part_stripped = part.strip()
                if part_stripped:
                    urls.append(part_stripped)
        if urls:
            return urls
            
    # Default fallback when file is not set or empty
    return ["redis://localhost:6379/0"]

def load_proxies():
    """Robustly loads the Proxies list, prioritizing the dynamic proxies.txt file for hot-reloads.
    If the file exists but reading fails, it infinitely retries reading with a delay.
    """
    proxies_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxies.txt")
    if os.path.exists(proxies_file):
        attempts = 0
        while True:
            try:
                with open(proxies_file, "r") as f:
                    content = f.read().strip()
                break  # Successful read
            except Exception as e:
                attempts += 1
                print(f"[!] Error reading proxies.txt (Attempt {attempts}): {e}. Retrying in 1s...")
                time.sleep(1)
        
        # Split by lines to filter out comments first
        lines = [line.strip() for line in content.splitlines()]
        clean_lines = [line for line in lines if line and not line.startswith("#")]
        
        proxies = []
        for line in clean_lines:
            for part in line.split(","):
                part_stripped = part.strip()
                if part_stripped:
                    proxies.append(part_stripped)
        if proxies:
            return proxies
            
    # Default fallback when file is not set or empty (no proxies)
    return []

