import time
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from config import settings, load_proxies

# Dynamic proxy tracking variables
current_proxy_index = 0
active_s3_client = None
last_used_proxy = None

def get_s3_client(force_new=False):
    """Dynamically creates or retrieves the active S3 client configured with the current proxy."""
    global active_s3_client, current_proxy_index, last_used_proxy
    
    # Hot-reload proxies list from disk/environment
    proxies_list = load_proxies()
    
    # Guard index in case list size changed dynamically
    if proxies_list:
        current_proxy_index = current_proxy_index % len(proxies_list)
        target_proxy = proxies_list[current_proxy_index]
    else:
        target_proxy = None
        
    if active_s3_client is not None and not force_new and last_used_proxy == target_proxy:
        return active_s3_client
        
    client_args = {
        'service_name': 's3',
        'aws_access_key_id': settings.AWS_S3_ACCESS_KEY_ID,
        'aws_secret_access_key': settings.AWS_S3_SECRET_ACCESS_KEY,
        'region_name': settings.AWS_REGION,
        'endpoint_url': f"https://s3.{settings.AWS_REGION}.amazonaws.com"
    }
    
    # Configure proxy if target proxy is active
    if target_proxy:
        print(f"[*] Initializing S3 client using proxy ({current_proxy_index + 1}/{len(proxies_list)}): {target_proxy}")
        config = Config(
            proxies={
                'http': target_proxy,
                'https': target_proxy
            },
            connect_timeout=15,
            read_timeout=15
        )
        client_args['config'] = config
    else:
        # Standard configuration with high resilience timeouts
        client_args['config'] = Config(connect_timeout=15, read_timeout=15)
        
    active_s3_client = boto3.client(**client_args)
    last_used_proxy = target_proxy
    return active_s3_client

def execute_s3_with_failover(operation_name, action_func, *args, **kwargs):
    """Executes an S3 request with infinite proxy-rotation failover and direct connection fallback."""
    global current_proxy_index
    attempts = 0
    
    while True:
        client = get_s3_client()
        try:
            return action_func(client, *args, **kwargs)
        except Exception as e:
            attempts += 1
            proxies_list = load_proxies()
            print(f"[!] S3 operation '{operation_name}' failed (Attempt {attempts}) using proxy index {current_proxy_index}: {e}")
            
            # Rotate to the next proxy
            if proxies_list:
                current_proxy_index = (current_proxy_index + 1) % len(proxies_list)
                print(f"  [*] Switching to next proxy ({current_proxy_index + 1}/{len(proxies_list)})...")
                get_s3_client(force_new=True)
            
            # Try a direct connection check periodically (e.g. after cycling through all proxies)
            if proxies_list and attempts % len(proxies_list) == 0:
                print("[!] Full proxy cycle completed. Attempting direct S3 connection check...")
                try:
                    direct_args = {
                        'service_name': 's3',
                        'aws_access_key_id': settings.AWS_S3_ACCESS_KEY_ID,
                        'aws_secret_access_key': settings.AWS_S3_SECRET_ACCESS_KEY,
                        'region_name': settings.AWS_REGION,
                        'endpoint_url': f"https://s3.{settings.AWS_REGION}.amazonaws.com",
                        'config': Config(connect_timeout=10, read_timeout=10)
                    }
                    direct_client = boto3.client(**direct_args)
                    return action_func(direct_client, *args, **kwargs)
                except Exception as direct_err:
                    print(f"[!] Direct S3 check failed: {direct_err}")
            
            # Progressive backoff to avoid hammering
            sleep_time = min(5, attempts)
            print(f"  [*] Retrying S3 operation in {sleep_time}s...")
            time.sleep(sleep_time)

def generate_presigned_url_get(object_name, bucket=None, expiration=3600):
    """Generate a presigned URL to share an S3 object (GET)"""
    bucket = bucket or settings.AWS_BUCKET_NAME
    try:
        client = get_s3_client()
        response = client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        print(e)
        return None
    return response

def generate_presigned_url(object_name, bucket=None, expiration=3600):
    """Generate a presigned URL for uploading to S3 (PUT)"""
    bucket = bucket or settings.AWS_BUCKET_NAME
    try:
        client = get_s3_client()
        response = client.generate_presigned_url('put_object',
                                                    Params={'Bucket': bucket,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        print(e)
        return None
    return response

def get_cdn_url(object_name, bucket=None):
    bucket = bucket or settings.AWS_BUCKET_NAME
    if settings.AWS_CDN_DOMAIN:
        return f"https://{settings.AWS_CDN_DOMAIN}/{object_name}"
    return f"https://{bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{object_name}"

def get_s3_url(key: str, is_public: bool):
    if not key:
        return None
    if key.startswith("http"):
        return key
    if is_public:
        return get_cdn_url(key, bucket=settings.AWS_BUCKET_NAME)
    return generate_presigned_url_get(key, bucket=settings.AWS_PRIVATE_BUCKET_NAME)

def delete_from_s3(key: str, is_public: bool):
    """
    Delete object from S3
    """
    if not key:
        return
    bucket = settings.AWS_BUCKET_NAME if is_public else settings.AWS_PRIVATE_BUCKET_NAME
    
    def _action(client):
        client.delete_object(Bucket=bucket, Key=key)
        
    try:
        execute_s3_with_failover("delete_from_s3", _action)
    except Exception as e:
        print(f"Error deleting {key} from S3: {e}")

def upload_to_s3(file_content: bytes, key: str, is_public: bool, content_type: str = "image/png") -> str:
    """
    Upload file to S3 and return the Key
    """
    bucket = settings.AWS_BUCKET_NAME if is_public else settings.AWS_PRIVATE_BUCKET_NAME
    upload_args = {
        "Bucket": bucket,
        "Key": key,
        "Body": file_content,
        "ContentType": content_type
    }
    if is_public:
        upload_args["ACL"] = 'public-read'
        
    def _action(client):
        client.put_object(**upload_args)
        return key
        
    return execute_s3_with_failover("upload_to_s3", _action)

def download_from_s3(key: str, is_public: bool) -> bytes:
    """
    Download file content from S3
    """
    bucket = settings.AWS_BUCKET_NAME if is_public else settings.AWS_PRIVATE_BUCKET_NAME
    
    def _action(client):
        response = client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
        
    return execute_s3_with_failover("download_from_s3", _action)
