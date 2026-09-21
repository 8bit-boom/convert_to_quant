"""Tests for the debounced GC utility."""
import torch
import pytest

from convert_to_quant.utils.debounced_gc import DebouncedGC, default_gc_debouncer


class TestDebouncedGC:
    def test_collects_every_nth_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr("convert_to_quant.utils.debounced_gc.gc.collect", lambda: calls.append("gc"))
        d = DebouncedGC(interval=3)
        for _ in range(3):
            d.maybe_collect()
        assert len(calls) == 1  # only the 3rd call triggers

    def test_interval_1_collects_every_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr("convert_to_quant.utils.debounced_gc.gc.collect", lambda: calls.append("gc"))
        d = DebouncedGC(interval=1)
        d.maybe_collect()
        d.maybe_collect()
        assert len(calls) == 2

    def test_interval_clamped_to_at_least_1(self):
        assert DebouncedGC(interval=0).interval == 1
        assert DebouncedGC(interval=-5).interval == 1

    def test_collect_now_always_collects(self, monkeypatch):
        calls = []
        monkeypatch.setattr("convert_to_quant.utils.debounced_gc.gc.collect", lambda: calls.append("gc"))
        d = DebouncedGC(interval=100)
        d.collect_now()
        assert len(calls) == 1

    def test_reset(self, monkeypatch):
        calls = []
        monkeypatch.setattr("convert_to_quant.utils.debounced_gc.gc.collect", lambda: calls.append("gc"))
        d = DebouncedGC(interval=2)
        d.maybe_collect()  # counter=1, no collect
        d.reset()
        d.maybe_collect()  # counter=1 again, no collect
        assert len(calls) == 0
        d.maybe_collect()  # counter=2 -> collect
        assert len(calls) == 1

    def test_shared_default_instance(self):
        assert isinstance(default_gc_debouncer, DebouncedGC)
        # Default interval gives bounded memory growth between collections
        assert 1 <= default_gc_debouncer.interval <= 64
