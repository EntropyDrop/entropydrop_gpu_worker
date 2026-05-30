#!/usr/bin/env python3
"""
AWS EC2 Dynamic Jump Host Discovery Script
Reads GPU worker dynamic discovery AWS keys from `.env` and scans
multi-region EC2 instances to retrieve active jump hosts.
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def discover_jump_hosts():
    aws_id = os.getenv("AWS_GPU_WORKER_ACCESS_KEY_ID")
    aws_key = os.getenv("AWS_GPU_WORKER_SECRET_ACCESS_KEY")
    
    if not aws_id or not aws_key:
        print("\n" + "=" * 80)
        print("   ⚠️  AWS DYNAMIC DISCOVERY KEYS MISSING")
        print("=" * 80)
        print(" Please configure the following in your GPU worker's .env file:")
        print("   AWS_GPU_WORKER_ACCESS_KEY_ID=\"your-least-privilege-discovery-key\"")
        print("   AWS_GPU_WORKER_SECRET_ACCESS_KEY=\"your-least-privilege-discovery-secret\"")
        print("=" * 80 + "\n")
        return []

    # Standard region pool for multi-country high-availability jump hosts
    regions = ["us-east-1", "ap-northeast-1", "ap-southeast-1"]
    discovered = []

    print("\n" + "═" * 80)
    print(" 📡  SCANNING AWS REGIONS FOR ACTIVE JUMP HOSTS")
    print("═" * 80)

    for region in regions:
        print(f"[*] Scanning region: {region:15} ... ", end="", flush=True)
        try:
            ec2 = boto3.client(
                "ec2",
                aws_access_key_id=aws_id,
                aws_secret_access_key=aws_key,
                region_name=region
            )
            
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Group", "Values": ["ed-jump-host-pool"]},
                    {"Name": "instance-state-name", "Values": ["running"]}
                ]
            )
            
            found_in_region = 0
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    public_ip = instance.get("PublicIpAddress")
                    if not public_ip:
                        continue
                        
                    name = "unknown"
                    for tag in instance.get("Tags", []):
                        if tag.get("Key") == "Name":
                            name = tag.get("Value")
                            break
                            
                    discovered.append({
                        "region": region,
                        "name": name,
                        "ip": public_ip,
                        "id": instance.get("InstanceId"),
                        "type": instance.get("InstanceType")
                    })
                    found_in_region += 1
            
            if found_in_region > 0:
                print(f"[✓] Found {found_in_region} host(s)")
            else:
                print("[ ] None")
                
        except ClientError as e:
            print(f"[!] API Error: {e.response['Error']['Code']}")
        except Exception as e:
            print(f"[!] Error: {e}")
            
    print("═" * 80)
    if discovered:
        print(f"\n[✓] Successfully Discovered {len(discovered)} Active Jump Host(s):\n")
        
        # Print a beautiful formatted markdown table
        header = f"| {'#':3} | {'Jump Host Name':25} | {'Public IP':18} | {'AWS Region':15} | {'Instance ID':20} |"
        divider = f"|{'-'*5}|{'-'*27}|{'-'*20}|{'-'*17}|{'-'*22}|"
        print(header)
        print(divider)
        for idx, host in enumerate(discovered, 1):
            row = f"| {idx:<3} | {host['name']:25} | {host['ip']:18} | {host['region']:15} | {host['id']:20} |"
            print(row)
        print()
    else:
        print("\n[!] No active jump hosts found. Make sure jump hosts are running and tagged with 'Group=ed-jump-host-pool'.\n")
        
    return discovered

if __name__ == "__main__":
    try:
        discover_jump_hosts()
    except KeyboardInterrupt:
        print("\n[*] Aborted.")
        sys.exit(0)
