#!/usr/bin/env python3
import socket
import select
import threading
import struct
import sys
from urllib.parse import urlparse

def socks5_connect(socks_host, socks_port, target_host, target_port):
    """
    Establishes a connection to target_host:target_port through a SOCKS5 proxy.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect((socks_host, socks_port))
    
    # 1. Handshake: Send SOCKS5 Greeting (No Authentication Required)
    s.sendall(b'\x05\x01\x00')
    
    # Read response (2 bytes)
    resp = s.recv(2)
    if len(resp) < 2 or resp[0] != 0x05 or resp[1] != 0x00:
        s.close()
        raise Exception("SOCKS5 greeting/auth negotiation failed")
        
    # 2. Request: Send CONNECT command
    # VER=5, CMD=1 (CONNECT), RSV=0
    req = b'\x05\x01\x00'
    
    # Address Type: ATYP
    try:
        # Check if target_host is a valid IPv4
        ip_bytes = socket.inet_aton(target_host)
        req += b'\x01' + ip_bytes
    except socket.error:
        # Domain name
        host_bytes = target_host.encode('idna')
        req += b'\x03' + bytes([len(host_bytes)]) + host_bytes
        
    req += struct.pack('>H', target_port)
    s.sendall(req)
    
    # Read response
    resp = s.recv(4096)
    if len(resp) < 4 or resp[1] != 0x00:
        s.close()
        raise Exception(f"SOCKS5 connection request failed (reply code: {resp[1] if len(resp) > 1 else 'unknown'})")
        
    return s

def pipe_sockets(s1, s2):
    """
    Pipes data bidirectionally between s1 and s2.
    """
    def forward(source, dest):
        try:
            while True:
                data = source.recv(8192)
                if not data:
                    break
                dest.sendall(data)
        except Exception:
            pass
        finally:
            try:
                source.close()
            except:
                pass
            try:
                dest.close()
            except:
                pass

    t1 = threading.Thread(target=forward, args=(s1, s2), daemon=True)
    t2 = threading.Thread(target=forward, args=(s2, s1), daemon=True)
    t1.start()
    t2.start()

def handle_client(client_conn, socks_host, socks_port):
    try:
        # Read the request line and headers (up to \r\n\r\n)
        request_data = b""
        while b"\r\n\r\n" not in request_data:
            chunk = client_conn.recv(4096)
            if not chunk:
                break
            request_data += chunk
            if len(request_data) > 65536:  # Safety ceiling
                break
                
        if b"\r\n" not in request_data:
            client_conn.close()
            return
            
        first_line = request_data.split(b"\r\n")[0].decode('utf-8', errors='ignore')
        parts = first_line.split()
        if len(parts) < 2:
            client_conn.close()
            return
            
        method = parts[0].upper()
        url = parts[1]
        
        host = None
        port = None
        
        if method == 'CONNECT':
            # CONNECT host:port HTTP/1.1
            if ':' in url:
                host, port_str = url.split(':', 1)
                port = int(port_str)
            else:
                host = url
                port = 443
        else:
            # Standard HTTP (GET, POST, etc.)
            parsed = urlparse(url)
            host = parsed.hostname
            if parsed.port:
                port = parsed.port
            elif parsed.scheme == 'https':
                port = 443
            elif parsed.scheme == 'http':
                port = 80
                
            # If not absolute URL, extract from Host header
            if not host:
                for line in request_data.split(b"\r\n"):
                    if line.lower().startswith(b"host:"):
                        host_val = line.split(b":", 1)[1].strip().decode('utf-8')
                        if ":" in host_val:
                            host, port_str = host_val.split(":", 1)
                            port = int(port_str)
                        else:
                            host = host_val
                            port = 80
                        break
                        
        if not host or port is None:
            client_conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            client_conn.close()
            return
            
        # Connect to upstream SOCKS5
        try:
            upstream_conn = socks5_connect(socks_host, socks_port, host, port)
        except Exception as e:
            client_conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            client_conn.close()
            return
            
        if method == 'CONNECT':
            # Send HTTP 200 Connection Established
            client_conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            pipe_sockets(client_conn, upstream_conn)
        else:
            # For standard HTTP, send the already read request to upstream
            upstream_conn.sendall(request_data)
            pipe_sockets(client_conn, upstream_conn)
            
    except Exception:
        try:
            client_conn.close()
        except:
            pass

def start_http_to_socks_proxy(host, http_port, socks_host, socks_port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((host, http_port))
    except Exception as e:
        print(f"[!] Bind to {host}:{http_port} failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    server.listen(128)
    print(f"[✓] HTTP-to-SOCKS5 Proxy listening on {host}:{http_port} forwarding to SOCKS5 at {socks_host}:{socks_port}")
    
    while True:
        try:
            client_conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(client_conn, socks_host, socks_port), daemon=True)
            t.start()
        except KeyboardInterrupt:
            print("[*] HTTP-to-SOCKS5 Proxy stopping...")
            break
        except Exception as e:
            print(f"[-] Error accepting connection: {e}", file=sys.stderr)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert SOCKS5 proxy to HTTP proxy.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the HTTP proxy to")
    parser.add_argument("--port", type=int, default=9100, help="Port to bind the HTTP proxy to")
    parser.add_argument("--socks-host", default="127.0.0.1", help="Host of the SOCKS5 proxy")
    parser.add_argument("--socks-port", type=int, default=9000, help="Port of the SOCKS5 proxy")
    args = parser.parse_args()
    
    start_http_to_socks_proxy(args.host, args.port, args.socks_host, args.socks_port)
