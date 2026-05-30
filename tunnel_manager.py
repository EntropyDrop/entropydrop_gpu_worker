#!/usr/bin/env python3
"""
AWS EC2 AutoSSH Tunnel Manager (HA Reconciler)
Periodically polls running jump hosts, maps them to deterministic indexes,
and dynamically provisions/tears down autossh tunnels, rewriting local configs.
"""

import os
import sys
import time
import subprocess
import signal
from config import settings


# Base ports (single entry configuration)
REDIS_PORT = 6380
SOCKS_PORT = 9000
REMOTE_REDIS_PORT = 6379

# Dynamic config files to write
REDIS_FILE = "redis_urls.txt"
PROXIES_FILE = "proxies.txt"

def get_running_autossh_processes():
    """
    Parses active system processes to find running autossh tunnels managed by us.
    Returns a dict mapping: local_port (int) -> {
        "pid": int,
        "type": "redis" | "socks",
        "ip": str
    }
    """
    active_tunnels = {}
    try:
        # Run 'ps aux' to capture running processes
        res = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            if "autossh" not in line or "python" in line:
                continue
                
            parts = line.split()
            if len(parts) < 11:
                continue
                
            pid = int(parts[1])
            # Combine the command arguments
            cmd_args = " ".join(parts[10:])
            
            # Identify target IP (e.g. ubuntu@54.89.23.45)
            target_ip = None
            for arg in parts[10:]:
                if "@" in arg:
                    target_ip = arg.split("@")[-1]
                    break
                    
            if not target_ip:
                continue
                
            # Parse Redis Local Forwarding (-L 6380:127.0.0.1:6379)
            if "-L" in cmd_args:
                for idx, arg in enumerate(parts[10:]):
                    if arg == "-L" and idx + 1 < len(parts[10:]):
                        forward_spec = parts[10:][idx + 1]
                        local_port = int(forward_spec.split(":")[0])
                        active_tunnels[local_port] = {
                            "pid": pid,
                            "type": "redis",
                            "ip": target_ip
                        }
                        
            # Parse SOCKS Proxy Forwarding (-D 9000)
            elif "-D" in cmd_args:
                for idx, arg in enumerate(parts[10:]):
                    if arg == "-D" and idx + 1 < len(parts[10:]):
                        local_port = int(parts[10:][idx + 1])
                        active_tunnels[local_port] = {
                            "pid": pid,
                            "type": "socks",
                            "ip": target_ip
                        }
    except Exception as e:
        print(f"[!] Error scanning running processes: {e}")
        
    return active_tunnels

def kill_process(pid, label):
    """Kills a process with SIGTERM (and SIGKILL fallback)."""
    print(f"[*] Terminating {label} (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # Give it a moment to exit
        for _ in range(10):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                return # Process successfully exited!
        # Force kill if still alive
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

def spawn_autossh_tunnel(tunnel_type, local_port, hostname, username="ubuntu"):
    """Spawns a background autossh process with optimized configs and cloudflared ProxyCommand."""
    cf_client_id = settings.CF_ACCESS_CLIENT_ID
    cf_client_secret = settings.CF_ACCESS_CLIENT_SECRET

    common_opts = [
        "autossh", "-M", "0",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=1",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10"
    ]

    # If Cloudflare service token credentials are configured, use cloudflared ProxyCommand
    if cf_client_id and cf_client_secret:
        # Standardize SSH options to use Key=Value format for copy-paste safety
        proxy_cmd = f"cloudflared access ssh --hostname {hostname} --id {cf_client_id} --secret {cf_client_secret}"
        common_opts += ["-o", f"ProxyCommand={proxy_cmd}"]
        print(f"[*] Authenticating to Cloudflare Access with Service Token for {hostname}")
    else:
        # Fall back to standard browser-based or unauthenticated cloudflared access proxy
        proxy_cmd = f"cloudflared access ssh --hostname {hostname}"
        common_opts += ["-o", f"ProxyCommand={proxy_cmd}"]
        print(f"[*] Warning: Service Token credentials missing. Falling back to browser-based cloudflared proxy.")

    # Dynamically detect if ed-jump-host-key.pem private key exists in the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(script_dir, "ed-jump-host-key.pem")
    if os.path.exists(key_file):
        common_opts += ["-i", key_file]
        print(f"[*] Found local private key {key_file}, appending -i option.")

    if tunnel_type == "redis":
        cmd = common_opts + [
            "-f", "-N",
            "-L", f"{local_port}:127.0.0.1:{REMOTE_REDIS_PORT}",
            f"{username}@{hostname}"
        ]
        label = f"Redis Tunnel {local_port} -> {hostname}:{REMOTE_REDIS_PORT}"
    else:
        cmd = common_opts + [
            "-D", str(local_port),
            "-f", "-N",
            f"{username}@{hostname}"
        ]
        label = f"SOCKS Proxy {local_port} -> {hostname}"

    print(f"[*] Launching: {' '.join(cmd)}")
    try:
        # Run with environmental AUTOSSH_GATETIME=0 to guarantee persistent retries
        env = os.environ.copy()
        env["AUTOSSH_GATETIME"] = "0"
        
        subprocess.run(cmd, env=env, check=True)
        print(f"[✓] Spawned {label} successfully in background.")
        
        # --- Port Occupancy Verification Checkpoint using lsof ---
        print(f"[*] Verifying port {local_port} binding using lsof (waiting infinitely)...")
        attempts = 0
        while True:
            attempts += 1
            # Run: lsof -i :<port> to check if the port is occupied
            res = subprocess.run(["lsof", "-i", f":{local_port}"], capture_output=True)
            if res.returncode == 0:
                break
            
            # Print a progress indicator every 10 attempts (5 seconds)
            if attempts % 10 == 0:
                print(f"[*] Still waiting for port {local_port} to bind (elapsed: {attempts * 0.5:.1f}s)...")
                
            time.sleep(0.5)
                
        print(f"[✓] Verified: Port {local_port} is successfully open and active!")
        
        # --- youtube.com SOCKS5 Connectivity Verification Checkpoint ---
        if tunnel_type == "socks":
            print(f"[*] Testing SOCKS5 proxy connectivity to youtube.com via port {local_port}...")
            socks_ok = False
            for test_attempt in range(1, 11):
                # We use --socks5-hostname so that DNS resolution happens through the SOCKS5 proxy
                res = subprocess.run([
                    "curl", "-s", "-o", "/dev/null",
                    "-w", "%{http_code}",
                    "--socks5-hostname", f"localhost:{local_port}",
                    "https://www.youtube.com",
                    "--connect-timeout", "5"
                ], capture_output=True, text=True)
                
                http_code = res.stdout.strip()
                if res.returncode == 0 and http_code in ("200", "301", "302"):
                    print(f"[✓] SOCKS5 proxy connectivity test succeeded: successfully connected to youtube.com (HTTP {http_code})!")
                    socks_ok = True
                    break
                else:
                    print(f"[*] Attempt {test_attempt}/10: SOCKS5 youtube.com connectivity check failed (HTTP {http_code}, code {res.returncode}). Retrying in 1s...")
                    time.sleep(1)
            
            if not socks_ok:
                print(f"[!] Warning: SOCKS5 proxy port {local_port} is bound, but failed to connect to youtube.com after 10 attempts.")
            
    except Exception as e:
        print(f"[!] Failed to spawn {label}: {e}")
        raise

def update_config_files():
    """Writes the active tunnel endpoints into redis_urls.txt and proxies.txt."""
    redis_password = settings.AWS_REDIS_PASSWORD
    
    # Single entry-point endpoints
    redis_url = f"redis://:{redis_password}@127.0.0.1:{REDIS_PORT}/0"
    proxy_url = f"socks5h://localhost:{SOCKS_PORT}"
    
    # Write redis_urls.txt
    print(f"[*] Re-writing {REDIS_FILE}...")
    with open(REDIS_FILE, "w") as f:
        f.write("# Dynamic Redis URLs managed by tunnel_manager.py\n")
        f.write(f"{redis_url}\n")
        
    # Write proxies.txt
    print(f"[*] Re-writing {PROXIES_FILE}...")
    with open(PROXIES_FILE, "w") as f:
        f.write("# Dynamic SOCKS5 Proxies managed by tunnel_manager.py\n")
        f.write(f"{proxy_url}\n")

def reconcile():
    """Main synchronization loop: conforms local tunnels to match the configured CF entry point."""
    cf_hostname = settings.CF_TUNNEL_HOSTNAME
    
    # Scan active system autossh processes
    active_tunnels = get_running_autossh_processes()
    
    # Keep track of expected ports
    expected_ports = {REDIS_PORT, SOCKS_PORT}
    
    # --- Reconcile Redis Tunnel on REDIS_PORT ---
    if REDIS_PORT in active_tunnels:
        current_tunnel = active_tunnels[REDIS_PORT]
        if current_tunnel["ip"] != cf_hostname:
            print(f"[!] Redis Port {REDIS_PORT} hostname mismatch (Expected: {cf_hostname}, Active: {current_tunnel['ip']})")
            kill_process(current_tunnel["pid"], f"Redis Tunnel {REDIS_PORT}")
            spawn_autossh_tunnel("redis", REDIS_PORT, cf_hostname)
        else:
            # Active and hostname matches, keep running
            pass
    else:
        # Tunnel not running, spawn it
        spawn_autossh_tunnel("redis", REDIS_PORT, cf_hostname)
        
    # --- Reconcile SOCKS Tunnel on SOCKS_PORT ---
    if SOCKS_PORT in active_tunnels:
        current_tunnel = active_tunnels[SOCKS_PORT]
        if current_tunnel["ip"] != cf_hostname:
            print(f"[!] SOCKS Port {SOCKS_PORT} hostname mismatch (Expected: {cf_hostname}, Active: {current_tunnel['ip']})")
            kill_process(current_tunnel["pid"], f"SOCKS Proxy {SOCKS_PORT}")
            spawn_autossh_tunnel("socks", SOCKS_PORT, cf_hostname)
        else:
            # Active and hostname matches, keep running
            pass
    else:
        # Tunnel not running, spawn it
        spawn_autossh_tunnel("socks", SOCKS_PORT, cf_hostname)
        
    # --- Clean up any leftover tunnels beyond our single entry ports ---
    for local_port, tunnel in active_tunnels.items():
        if local_port not in expected_ports:
            print(f"[!] Cleaning up obsolete tunnel on port {local_port}...")
            kill_process(tunnel["pid"], f"Obsolete Tunnel {local_port}")
            
    # Update configuration text files for hot-reloading
    update_config_files()
    print("[✓] Reconcile sync loop completed successfully.\n")

def main():
    print("=" * 80)
    print("   🚀  CLOUDFLARE AUTOSSH TUNNEL SETUP STARTING (SINGLE ENTRY POINT)")
    print("=" * 80)
    try:
        reconcile()
        print("[✓] Tunnel configuration and startup completed successfully.")
    except Exception as e:
        print(f"[!] Critical error during tunnel setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
