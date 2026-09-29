"""Private IPC workers that retain the models of an immutable Dense UV release.

The release's own CLI entry points still perform all image processing and write
their original provenance manifests. Only their model/renderer loaders are
cached, inside a separate process for each Python environment and release.
"""
import argparse
import contextlib
import ctypes
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 65536


class ResidentProcess:
    def __init__(self, command, cwd, env):
        self.command = list(command)
        self.cwd = str(cwd)
        self.env = dict(env)
        self.process = None
        self.socket = None
        self.directory = None
        self.buffer = b''
        self.ready_info = None

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    def _set_timeout(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Dense UV resident request timed out')
        self.socket.settimeout(remaining)

    def _receive(self, deadline):
        while b'\n' not in self.buffer:
            self._set_timeout(deadline)
            data = self.socket.recv(8192)
            if not data:
                raise RuntimeError('Dense UV resident process disconnected')
            self.buffer += data
            if len(self.buffer) > MAX_MESSAGE_BYTES:
                raise RuntimeError('Dense UV resident response is too large')
        line, self.buffer = self.buffer.split(b'\n', 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError('Invalid Dense UV resident response')
        return response

    def _log_tail(self):
        if self.directory is None:
            return ''
        try:
            with (Path(self.directory.name) / 'server.log').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 4000))
                return stream.read().decode(errors='replace')
        except OSError:
            return ''

    def start(self, deadline):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        parent, child = socket.socketpair()
        self.socket = parent
        self.directory = tempfile.TemporaryDirectory(prefix='dense-uv-resident-')
        stage_env = {**self.env, 'TMPDIR': self.directory.name}
        try:
            with (Path(self.directory.name) / 'server.log').open('wb') as log:
                self.process = subprocess.Popen(
                    self.command + ['--ipc-fd', str(child.fileno()),
                                    '--parent-pid', str(os.getpid())],
                    cwd=self.cwd, env=stage_env, pass_fds=(child.fileno(),),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            child.close()
            ready = self._receive(deadline)
            if ready.get('ready') is not True or ready.get('protocol') != PROTOCOL_VERSION:
                raise RuntimeError('Dense UV resident initialization failed: ' + str(ready.get('error', ready)))
            self.ready_info = ready
        except BaseException as exc:
            tail = self._log_tail()
            self.close()
            if isinstance(exc, Exception) and tail:
                exc.add_note(tail)
            raise
        finally:
            child.close()

    def request(self, payload, deadline):
        try:
            self.start(deadline)
            request_id = uuid.uuid4().hex
            self._set_timeout(deadline)
            self.socket.sendall(json.dumps({**payload, 'id': request_id}).encode() + b'\n')
            response = self._receive(deadline)
            if response.get('id') != request_id:
                raise RuntimeError('Dense UV resident response identity mismatch')
            if response.get('ok') is not True:
                raise RuntimeError('Dense UV resident inference failed: ' + str(response.get('error')))
            return response
        except BaseException as exc:
            tail = self._log_tail()
            self.close()
            if isinstance(exc, Exception) and tail:
                exc.add_note(tail)
            raise

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
            self.process = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
        self.buffer = b''
        self.ready_info = None


def _bind_release(root):
    # Use the pinned checkout's existing package binder; never import the live
    # SkingToolkit checkout into either resident model process.
    entry = root / 'runtime/dense_uv_parser/run_local.py'
    spec = importlib.util.spec_from_file_location('_dense_uv_release_entry', entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bind_checkout()


@contextlib.contextmanager
def _arguments(module, arguments):
    previous = sys.argv
    sys.argv = [module.__file__, *map(str, arguments)]
    try:
        yield
    finally:
        sys.argv = previous


def _load_handler(root, stage):
    from functools import lru_cache
    import torch

    torch.set_num_threads(4)
    _bind_release(root)
    if stage == 'foreground':
        loader = importlib.import_module('SkingToolkit.dense_uv_parser.foreground_model')
        entry = importlib.import_module('SkingToolkit.dense_uv_parser.foreground_provider')
        model_dir, checkpoint = root / 'foreground_base', root / 'foreground.pt'
        loaded = loader.load_foreground_model(model_dir, checkpoint)

        def cached_foreground(requested_dir, requested_checkpoint):
            if (Path(requested_dir).resolve() != model_dir
                    or Path(requested_checkpoint).resolve() != checkpoint):
                raise ValueError('Resident foreground model identity mismatch')
            return loaded

        loader.load_foreground_model = cached_foreground
        entry.sha = lru_cache(maxsize=512)(entry.sha)

        def infer(request):
            arguments = ['--inputs', request['input'], '--output-dir', request['output'],
                         '--model-dir', model_dir, '--checkpoint', checkpoint,
                         '--result-manifest', request['manifest']]
            with _arguments(entry, arguments):
                entry.main()
    else:
        entry = importlib.import_module('SkingToolkit.dense_uv_parser.batch_accessories')
        checkpoint = root / 'parser.pt'
        checkpoint_bytes = checkpoint.read_bytes()
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        model, arguments = entry.load_parser(io.BytesIO(checkpoint_bytes), torch.device('cuda'))
        del checkpoint_bytes
        mappings = Path(arguments['mappings_dir']).resolve()
        renderer = entry.DifferentiableRenderer(str(mappings)).cuda()

        def cached_parser(snapshot, device):
            if (hashlib.sha256(snapshot.getvalue()).hexdigest() != checkpoint_hash
                    or torch.device(device).type != 'cuda'):
                raise ValueError('Resident parser model identity mismatch')
            return model, arguments

        def cached_renderer(requested_mappings):
            if Path(requested_mappings).resolve() != mappings:
                raise ValueError('Resident renderer mappings mismatch')
            return renderer

        entry.load_parser = cached_parser
        entry.DifferentiableRenderer = cached_renderer
        entry.sha = lru_cache(maxsize=512)(entry.sha)

        def infer(request):
            args = ['--checkpoint', checkpoint, '--pipeline', root / 'pipeline.json',
                    '--inputs', request['input'], '--output-dir', request['output'],
                    '--foreground-probabilities', request['probability']]
            with _arguments(entry, args):
                entry.main()
    torch.cuda.synchronize()
    return infer


def _open_parent_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # DS's Conda Python was built against older headers even though its Linux
    # 6.8 kernel supports pidfds. syscall 434 is pidfd_open on x86-64 Linux.
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise RuntimeError('Resident Dense UV requires Linux pidfd support')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('foreground', 'parser'), required=True)
    parser.add_argument('--ipc-fd', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    # Monitor the owning *process*, not its spawning thread. PR_SET_PDEATHSIG
    # also fires when a short-lived launcher thread exits, which would discard
    # healthy models after parallel warmup. A pidfd also prevents PID reuse.
    try:
        parent_fd = _open_parent_pidfd(args.parent_pid)
    except ProcessLookupError:
        return
    if os.getppid() != args.parent_pid:
        os.close(parent_fd)
        return
    def watch_parent():
        select.select([parent_fd], [], [])
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpgrp(), signal.SIGKILL)
        else:
            os.kill(os.getpid(), signal.SIGKILL)
    threading.Thread(target=watch_parent, daemon=True, name='dense-uv-parent-watch').start()
    with socket.socket(fileno=args.ipc_fd) as channel:
        def send(response):
            channel.sendall(json.dumps(response).encode() + b'\n')

        try:
            infer = _load_handler(args.release_root.resolve(), args.stage)
            send({'ready': True, 'protocol': PROTOCOL_VERSION, 'stage': args.stage, 'pid': os.getpid()})
        except BaseException as exc:
            traceback.print_exc()
            send({'ready': False, 'protocol': PROTOCOL_VERSION, 'error': str(exc)})
            return
        with channel.makefile('rb') as requests:
            while True:
                line = requests.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b'\n'):
                    raise ValueError('Invalid Dense UV resident request length')
                request = json.loads(line)
                try:
                    infer(request)
                    send({'id': request['id'], 'ok': True})
                except BaseException as exc:
                    traceback.print_exc()
                    send({'id': request.get('id'), 'ok': False, 'error': str(exc)})
                    return


if __name__ == '__main__':
    main()
