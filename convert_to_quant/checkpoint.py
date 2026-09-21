"""
Resumable checkpoints for tensor-by-tensor conversions.

A checkpoint is a directory:

    <checkpoint_dir>/
        manifest.json          # run identity, seed, per-tensor entries, accumulated state
        shards/000000.safetensors
        shards/000001.safetensors
        ...

Each entry in the manifest corresponds to one processed weight key and
points at a shard holding every output tensor that key contributed (the
quantized weight plus its scale / comfy_quant / bias companions). Entries
with "shard": null produced no output tensors (e.g. removed T5XXL decoder
tensors) but are still recorded so resume skips them correctly.

Crash safety: the shard is written first, the manifest second, both via
temp-file + os.replace, so the manifest never references a shard that is
not fully on disk.

Stop semantics: a run is asked to stop via a stop file (--stop-file) or
SIGINT/SIGTERM (both install flag-setting handlers while a checkpoint is
active). The loop checks between tensors; a completed shard is always
valid, so stopping at any moment leaves a consistent checkpoint.
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import threading
import time
from typing import Any, Dict, Optional

MANIFEST_NAME = "manifest.json"
SHARDS_DIRNAME = "shards"
FORMAT_VERSION = 1

EXIT_STOPPED = 3  # process exit code: stopped cleanly, resumable


class CheckpointMismatchError(Exception):
    """Raised when a checkpoint dir does not match the current run."""


class StopRequested(Exception):
    """Raised in the conversion loop when a checkpointed stop was requested."""


# --- stop signalling -------------------------------------------------------

_stop_flag = threading.Event()
_old_handlers: Dict[int, Any] = {}


def _flag_handler(signum, frame):
    _stop_flag.set()


def install_stop_signal_handlers() -> None:
    """Make SIGINT/SIGTERM request a checkpointed stop instead of killing us."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        if sig in _old_handlers:
            continue
        try:
            _old_handlers[sig] = signal.signal(sig, _flag_handler)
        except (ValueError, OSError, RuntimeError):
            # Not in main thread / unsupported platform: stop-file polling
            # still works.
            pass


def restore_stop_signal_handlers() -> None:
    for sig, old in list(_old_handlers.items()):
        try:
            signal.signal(sig, old)
        except (ValueError, OSError, RuntimeError):
            pass
        del _old_handlers[sig]
    _stop_flag.clear()


def should_stop(stop_file: Optional[str]) -> bool:
    """True when a stop was requested (signal) or the stop file exists."""
    if _stop_flag.is_set():
        return True
    if stop_file and os.path.exists(stop_file):
        return True
    return False


def clear_stop_request() -> None:
    _stop_flag.clear()


# --- checkpoint ------------------------------------------------------------


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)


def _replace_with_retry(src: str, dst: str, retries: int = 10, delay: float = 0.02) -> None:
    """os.replace, tolerating transient Windows sharing violations.

    A concurrent reader (progress watcher, GUI) may hold the destination
    open for a moment; on Windows that makes os.replace fail with
    PermissionError, so retry briefly before giving up.
    """
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
        except OSError:
            raise


class TensorCheckpoint:
    """Incremental, crash-safe checkpoint of a per-tensor conversion run."""

    def __init__(self, dir_path: str, manifest: Dict[str, Any]):
        self.dir = dir_path
        self.manifest = manifest
        self._entries: list = manifest.setdefault("entries", [])
        self._done_keys = {e["key"] for e in self._entries}

    # -- creation / open --

    @staticmethod
    def exists(dir_path: str) -> bool:
        return os.path.isfile(os.path.join(dir_path, MANIFEST_NAME))

    @classmethod
    def create(
        cls,
        dir_path: str,
        *,
        input_file: str,
        seed: int,
        target_format: str,
        weight_keys: list,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "TensorCheckpoint":
        os.makedirs(os.path.join(dir_path, SHARDS_DIRNAME), exist_ok=True)
        manifest = {
            "version": FORMAT_VERSION,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "input_file": os.path.abspath(input_file),
            "seed": int(seed),
            "target_format": target_format,
            "weight_keys": list(weight_keys),
            "total": len(weight_keys),
            "finished": False,
            "extra": extra or {},
            "entries": [],
        }
        cp = cls(dir_path, manifest)
        cp._write_manifest()
        return cp

    @classmethod
    def open(cls, dir_path: str) -> "TensorCheckpoint":
        path = os.path.join(dir_path, MANIFEST_NAME)
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("version") != FORMAT_VERSION:
            raise CheckpointMismatchError(
                f"Checkpoint at '{dir_path}' has unsupported version "
                f"{manifest.get('version')} (expected {FORMAT_VERSION})."
            )
        return cls(dir_path, manifest)

    # -- identity / validation --

    @property
    def input_file(self) -> str:
        return self.manifest["input_file"]

    @property
    def seed(self) -> int:
        return int(self.manifest["seed"])

    @property
    def target_format(self) -> str:
        return self.manifest["target_format"]

    @property
    def finished(self) -> bool:
        return bool(self.manifest.get("finished"))

    @property
    def completed_count(self) -> int:
        return len(self._entries)

    @property
    def total(self) -> int:
        return int(self.manifest["total"])

    @property
    def completed_keys(self) -> set:
        return set(self._done_keys)

    @property
    def extra(self) -> Dict[str, Any]:
        return self.manifest.setdefault("extra", {})

    def validate_for(self, *, input_file: str, target_format: str, weight_keys: list) -> None:
        if os.path.abspath(input_file) != self.input_file:
            raise CheckpointMismatchError(
                f"Checkpoint input '{self.input_file}' does not match '{os.path.abspath(input_file)}'."
            )
        if target_format != self.target_format:
            raise CheckpointMismatchError(
                f"Checkpoint format '{self.target_format}' does not match '{target_format}'."
            )
        if list(weight_keys) != list(self.manifest["weight_keys"]):
            raise CheckpointMismatchError(
                "Checkpoint weight tensor list does not match the input file "
                "(model changed since the checkpoint was created?)."
            )

    # -- recording --

    def is_done(self, key: str) -> bool:
        return key in self._done_keys

    def record(
        self,
        key: str,
        tensors: Dict[str, Any],
        *,
        lora: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the output tensors for one processed weight key.

        `tensors` may be empty (key produced no output, e.g. removed).
        `lora` (optional) holds LoRA adapter tensors extracted for this key,
        stored in a separate shard so replay never mixes them into the model
        output. `state` (must be JSON-serializable) is merged into the
        manifest so mid-run accumulators (quant metadata, processed-key
        sets) survive.
        """
        if key in self._done_keys:
            return
        index = len(self._entries)
        entry: Dict[str, Any] = {"key": key, "shard": None, "n_tensors": len(tensors)}
        if tensors:
            shard_name = f"{index:06d}.safetensors"
            self._write_shard(os.path.join(self.dir, SHARDS_DIRNAME, shard_name), tensors)
            entry["shard"] = shard_name
        if lora:
            lora_name = f"{index:06d}.lora.safetensors"
            self._write_shard(os.path.join(self.dir, SHARDS_DIRNAME, lora_name), lora)
            entry["lora_shard"] = lora_name
            entry["n_lora_tensors"] = len(lora)
        self._entries.append(entry)
        self._done_keys.add(key)
        if state is not None:
            self.manifest["state"] = state
        self._write_manifest()

    def load_all(self) -> Dict[str, Any]:
        """Replay every recorded shard into a single tensors dict, in order."""
        from safetensors.torch import load_file

        out: Dict[str, Any] = {}
        for entry in self._entries:
            shard = entry.get("shard")
            if shard is None:
                continue
            path = os.path.join(self.dir, SHARDS_DIRNAME, shard)
            if not os.path.isfile(path):
                raise CheckpointMismatchError(
                    f"Checkpoint shard missing: {path}. Delete the checkpoint and restart."
                )
            out.update(load_file(path))
        return out

    def load_lora(self) -> Dict[str, Any]:
        """Replay every recorded LoRA shard into a single tensors dict, in order."""
        from safetensors.torch import load_file

        out: Dict[str, Any] = {}
        for entry in self._entries:
            shard = entry.get("lora_shard")
            if shard is None:
                continue
            path = os.path.join(self.dir, SHARDS_DIRNAME, shard)
            if not os.path.isfile(path):
                raise CheckpointMismatchError(
                    f"Checkpoint LoRA shard missing: {path}. Delete the checkpoint and restart."
                )
            out.update(load_file(path))
        return out

    @property
    def state(self) -> Dict[str, Any]:
        return self.manifest.setdefault("state", {})

    def mark_finished(self) -> None:
        self.manifest["finished"] = True
        self.manifest["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._write_manifest()

    # -- io --

    @staticmethod
    def _write_shard(path: str, tensors: Dict[str, Any]) -> None:
        from safetensors.torch import save_file
        import io
        import tempfile

        # Save via a temp file on the same filesystem, then atomic-rename, so
        # a crash mid-save never leaves a shard the manifest could reference.
        fd, tmp = tempfile.mkstemp(prefix="shard-", suffix=".tmp", dir=os.path.dirname(path))
        os.close(fd)
        try:
            save_file(tensors, tmp)
            _replace_with_retry(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write_manifest(self) -> None:
        payload = json.dumps(self.manifest, indent=1).encode("utf-8")
        _atomic_write_bytes(os.path.join(self.dir, MANIFEST_NAME), payload)
