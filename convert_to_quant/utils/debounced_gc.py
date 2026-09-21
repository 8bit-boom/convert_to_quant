"""
Debounced garbage collection for per-tensor hot loops.

gc.collect() costs tens of milliseconds and appears at several per-tensor call
sites. CPython refcounting already frees tensors immediately when their last
reference drops; gc.collect() only breaks reference cycles, and
torch.cuda.empty_cache() only returns cached blocks to the driver (it does not
prevent fragmentation, but it does force a device synchronization).

Calling both on every tensor therefore stalls the pipeline for little benefit.
Debouncing to every Nth call keeps memory growth bounded (worst case: N-1
tensors of extra cycles/blocks between collections) while removing the stall.

OOM-recovery and end-of-run paths should use collect_now(), not maybe_collect().
"""

import gc
from typing import Optional

import torch


class DebouncedGC:
    """Rate-limits gc.collect()/torch.cuda.empty_cache() calls."""

    def __init__(self, interval: int = 8):
        self.interval = max(1, int(interval))
        self._counter = 0

    def maybe_collect(self) -> None:
        """Run a full collection only every `interval`-th call."""
        self._counter += 1
        if self._counter % self.interval == 0:
            self.collect_now()

    def collect_now(self) -> None:
        """Immediate full collection (OOM recovery, end of run)."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self) -> None:
        self._counter = 0


# Shared instance used by per-tensor cleanup sites in the format loops.
# Tests that need deterministic collection can reset() or call collect_now().
default_gc_debouncer = DebouncedGC()
