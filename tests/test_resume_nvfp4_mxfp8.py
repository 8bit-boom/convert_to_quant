"""End-to-end stop & resume tests for the dedicated NVFP4 and MXFP8 paths.

NVFP4 simple mode runs real quantization on CPU. MXFP8's real kernel needs
the comfy_kitchen NVIDIA fork (unavailable here), so its converter is
replaced with a deterministic fake — these tests exercise the checkpoint
loop mechanics, not kernel numerics.
"""

import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert_to_quant import formats
from convert_to_quant.checkpoint import StopRequested, TensorCheckpoint
from convert_to_quant.cli.main import get_parser, run_conversion


def _make_model(path: Path, n_blocks: int = 6) -> list:
    g = torch.Generator().manual_seed(0)
    tensors = {}
    weight_keys = []
    for b in range(n_blocks):
        prefix = f"blocks.{b}.attn"
        tensors[f"{prefix}.wq.weight"] = torch.randn(16, 32, generator=g, dtype=torch.bfloat16)
        tensors[f"{prefix}.wq.bias"] = torch.randn(16, generator=g, dtype=torch.bfloat16)
        tensors[f"{prefix}.norm.weight"] = torch.randn(32, generator=g, dtype=torch.bfloat16)
        weight_keys.append(f"{prefix}.wq.weight")
    save_file(tensors, str(path))
    return sorted(weight_keys)


def _run(argv):
    run_conversion(get_parser().parse_args(argv))


def _assert_outputs_identical(path_a: Path, path_b: Path):
    a, b = load_file(str(path_a)), load_file(str(path_b))
    assert a.keys() == b.keys(), f"key mismatch: {set(a) ^ set(b)}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor '{k}' differs"
    with safe_open(str(path_a), framework="pt") as fa, safe_open(str(path_b), framework="pt") as fb:
        assert fa.metadata() == fb.metadata(), "file metadata (incl. _quantization_metadata) differs"


def _stop_after(checkpoint_dir: Path, n_entries: int, stop_file: Path) -> threading.Thread:
    done = threading.Event()

    def watchdog():
        while not done.is_set():
            if TensorCheckpoint.exists(str(checkpoint_dir)):
                try:
                    completed = TensorCheckpoint.open(str(checkpoint_dir)).completed_count
                except (PermissionError, OSError):
                    continue  # manifest being replaced; retry shortly
                if completed >= n_entries:
                    stop_file.write_text("")
                    done.set()
                    return
            time.sleep(0.001)

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    return t


def _nvfp4_argv(src, out, cp_dir=None, stop_file=None):
    argv = ["-i", str(src), "-o", str(out), "--nvfp4", "--simple", "--manual-seed", "7", "--calib-samples", "64"]
    if cp_dir:
        argv += ["--checkpoint-dir", str(cp_dir)]
    if stop_file:
        argv += ["--stop-file", str(stop_file)]
    return argv


def test_nvfp4_stopped_run_resumes_byte_identical(tmp_path):
    src = tmp_path / "model.safetensors"
    weight_keys = _make_model(src)

    out_ref = tmp_path / "ref.safetensors"
    _run(_nvfp4_argv(src, out_ref))

    out_res = tmp_path / "resumed.safetensors"
    cp_dir = tmp_path / "cp"
    stop_file = tmp_path / "stop"
    watcher = _stop_after(cp_dir, 2, stop_file)

    with pytest.raises(StopRequested):
        _run(_nvfp4_argv(src, out_res, cp_dir, stop_file))
    watcher.join(timeout=5)

    cp = TensorCheckpoint.open(str(cp_dir))
    assert 2 <= cp.completed_count < len(weight_keys), "stop should land mid-run"

    stop_file.unlink()
    _run(_nvfp4_argv(src, out_res, cp_dir, stop_file))

    assert TensorCheckpoint.open(str(cp_dir)).finished
    _assert_outputs_identical(out_ref, out_res)


def test_nvfp4_uninterrupted_checkpointed_run_matches_reference(tmp_path):
    src = tmp_path / "model.safetensors"
    _make_model(src)

    out_ref = tmp_path / "ref.safetensors"
    _run(_nvfp4_argv(src, out_ref))

    out_cp = tmp_path / "cp_out.safetensors"
    _run(_nvfp4_argv(src, out_cp, tmp_path / "cp"))
    _assert_outputs_identical(out_ref, out_cp)


class _FakeMXFP8Converter:
    """Deterministic stand-in for MXFP8Converter (quantize/dequantize only)."""

    def __init__(self, block_size=32, pad_to_32x=True):
        self.block_size = block_size

    def quantize(self, w):
        q = (w.float() * 16).round().clamp(-448, 448).to(torch.float8_e4m3fn)
        scales = torch.ones(w.shape[0], max(1, w.shape[1] // self.block_size), dtype=torch.uint8)
        return q, scales

    def dequantize(self, qdata, block_scales, output_dtype=torch.float32):
        return qdata.float().to(output_dtype)


def _mxfp8_argv(src, out, cp_dir=None, stop_file=None):
    argv = ["-i", str(src), "-o", str(out), "--mxfp8", "--simple", "--manual-seed", "7", "--calib-samples", "64"]
    if cp_dir:
        argv += ["--checkpoint-dir", str(cp_dir)]
    if stop_file:
        argv += ["--stop-file", str(stop_file)]
    return argv


def test_mxfp8_stopped_run_resumes_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(formats.mxfp8_conversion, "MXFP8Converter", _FakeMXFP8Converter)

    src = tmp_path / "model.safetensors"
    weight_keys = _make_model(src)

    out_ref = tmp_path / "ref.safetensors"
    _run(_mxfp8_argv(src, out_ref))

    out_res = tmp_path / "resumed.safetensors"
    cp_dir = tmp_path / "cp"
    stop_file = tmp_path / "stop"
    watcher = _stop_after(cp_dir, 2, stop_file)

    with pytest.raises(StopRequested):
        _run(_mxfp8_argv(src, out_res, cp_dir, stop_file))
    watcher.join(timeout=5)

    cp = TensorCheckpoint.open(str(cp_dir))
    assert 2 <= cp.completed_count < len(weight_keys), "stop should land mid-run"

    stop_file.unlink()
    _run(_mxfp8_argv(src, out_res, cp_dir, stop_file))

    assert TensorCheckpoint.open(str(cp_dir)).finished
    _assert_outputs_identical(out_ref, out_res)


def test_mxfp8_resume_rejects_changed_format(tmp_path, monkeypatch):
    monkeypatch.setattr(formats.mxfp8_conversion, "MXFP8Converter", _FakeMXFP8Converter)

    src = tmp_path / "model.safetensors"
    _make_model(src)

    out_res = tmp_path / "resumed.safetensors"
    cp_dir = tmp_path / "cp"
    stop_file = tmp_path / "stop"
    stop_file.write_text("")

    with pytest.raises(StopRequested):
        _run(_mxfp8_argv(src, out_res, cp_dir, stop_file))
    stop_file.unlink()

    # An NVFP4 resume attempt against an MXFP8 checkpoint must refuse.
    from convert_to_quant.checkpoint import CheckpointMismatchError

    nv_argv = ["-i", str(src), "-o", str(out_res), "--nvfp4", "--simple", "--checkpoint-dir", str(cp_dir)]
    with pytest.raises(CheckpointMismatchError):
        _run(nv_argv)
