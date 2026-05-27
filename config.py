import os

# Robustly try to load .env file using python-dotenv if available
try:
    from dotenv import load_dotenv
    dotenv_path = os.getenv("ENV_FILE", ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except ImportError:
    pass

class WorkerSettings:
    # Redis Config
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # AWS S3 / CDN Config
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-2")
    AWS_BUCKET_NAME: str = os.getenv("AWS_BUCKET_NAME", "")
    AWS_PRIVATE_BUCKET_NAME: str = os.getenv("AWS_PRIVATE_BUCKET_NAME", "")
    AWS_CDN_DOMAIN: str = os.getenv("AWS_CDN_DOMAIN", "")

    # FLUX / AI Model Directory Settings
    FLUX_MODEL_DIR: str = os.getenv("FLUX_MODEL_DIR", "")
    FLUX_LORA_DIR: str = os.getenv("FLUX_LORA_DIR", "")
    ZIMAGE_MODEL_DIR: str = os.getenv("ZIMAGE_MODEL_DIR", "")

settings = WorkerSettings()
