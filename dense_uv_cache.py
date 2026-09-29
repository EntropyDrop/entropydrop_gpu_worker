"""One resident model version per GPU worker, with explicit eviction."""
import gc
import sys


class DenseUVRuntimeCache:
    def __init__(self):
        self.key = None
        self.runtime = None

    def get_or_create(self, key, factory):
        if self.key == key and self.runtime is not None:
            return self.runtime
        # Evict before loading: two complete releases must never transiently
        # double a worker's model memory during a version switch.
        self.close()
        runtime = factory()
        self.key, self.runtime = key, runtime
        return runtime

    def close(self):
        runtime, self.runtime = self.runtime, None
        self.key = None
        if runtime is None:
            return
        close = getattr(runtime, 'close', None)
        if close is not None:
            close()
        del close, runtime
        gc.collect()
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
