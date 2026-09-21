"""End-to-end stop & resume tests for the FP8/INT8 conversion path.

A conversion stopped via stop-file (or signal) must resume into output that
is byte-identical to an uninterrupted run of the same command.
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

from convert_to_quant.checkpoint import CheckpointMismatchError, StopRequested, TensorCheckpoint
from convert_to_quant.cli.main import get_parser, run_conversion


def _make_model(path: Path, n_blocks: int = 8, in_features: int = 32, out_features: int = 16) -> list:
    """Small synthetic model; returns the sorted 2D weight keys."""
    g = torch.Generator().manual_seed(0)
    tensors = {}
    weight_keys = []
    for b in range(n_blocks):
        prefix = f"blocks.{b}.attn"
        w = torch.randn(out_features, in_features, generator=g, dtype=torch.bfloat16)
        tensors[f"{prefix}.wq.weight"] = w
        tensors[f"{prefix}.wq.bias"] = torch.randn(out_features, generator=g, dtype=torch.bfloat16)
        tensors[f"{prefix}.norm.weight"] = torch.randn(in_features, generator=g, dtype=torch.bfloat16)
        weight_keys.append(f"{prefix}.wq.weight")
    save_file(tensors, str(path))
    return sorted(weight_keys)


def _run(argv):
    args = get_parser().parse_args(argv)
    run_conversion(args)
    return args


def _argv(input_path, output_path, checkpoint_dir=None, stop_file=None):
    argv = [
        "-i", str(input_path),
        "-o", str(output_path),
        "--simple",               # no learned rounding: fast + CPU-deterministic
        "--device", "cpu",
        "--manual-seed", "7",
        "--calib-samples", "64",
        "--save-quant-metadata",
    ]
    if checkpoint_dir:
        argv += ["--checkpoint-dir", str(checkpoint_dir)]
    if stop_file:
        argv += ["--stop-file", str(stop_file)]
    return argv


def _assert_outputs_identical(path_a: Path, path_b: Path):
    a, b = load_file(str(path_a)), load_file(str(path_b))
    assert a.keys() == b.keys(), f"key mismatch: {set(a) ^ set(b)}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor '{k}' differs"
    with safe_open(str(path_a), framework="pt") as fa, safe_open(str(path_b), framework="pt") as fb:
        assert fa.metadata() == fb.metadata(), "file metadata (incl. _quantization_metadata) differs"


def test_stopped_run_resumes_byte_identical(tmp_path):
    src = tmp_path / "model.safetensors"
    weight_keys = _make_model(src)

    # Reference: uninterrupted run.
    out_ref = tmp_path / "ref.safetensors"
    _run(_argv(src, out_ref))

    # Checkpointed run, stopped after >= 3 tensors, then resumed.
    out_res = tmp_path / "resumed.safetensors"
    cp_dir = tmp_path / "cp"
    stop_file = tmp_path / "stop"

    stop_written = threading.Event()

    def watchdog():
        # Write the stop file as soon as the checkpoint has 3 entries.
        while not stop_written.is_set():
            if TensorCheckpoint.exists(str(cp_dir)):
                try:
                    completed = TensorCheckpoint.open(str(cp_dir)).completed_count
                except (PermissionError, OSError):
                    continue  # manifest being replaced; retry shortly
                if completed >= 3:
                    Path(stop_file).write_text("")
                    stop_written.set()
                    return
            time.sleep(0.001)

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()

    with pytest.raises(StopRequested):
        _run(_argv(src, out_res, cp_dir, stop_file))
    watcher.join(timeout=5)

    cp = TensorCheckpoint.open(str(cp_dir))
    assert 3 <= cp.completed_count < len(weight_keys), "stop should land mid-run"
    assert not cp.finished

    Path(stop_file).unlink()  # the requester consumes the stop request
    _run(_argv(src, out_res, cp_dir, stop_file))

    assert TensorCheckpoint.open(str(cp_dir)).finished
    assert out_res.is_file()
    _assert_outputs_identical(out_ref, out_res)


def test_uninterrupted_checkpointed_run_matches_reference(tmp_path):
    src = tmp_path / "model.safetensors"
    _make_model(src)

    out_ref = tmp_path / "ref.safetensors"
    _run(_argv(src, out_ref))

    out_cp = tmp_path / "cp_out.safetensors"
    cp_dir = tmp_path / "cp"
    _run(_argv(src, out_cp, cp_dir))
    assert TensorCheckpoint.open(str(cp_dir)).finished
    _assert_outputs_identical(out_ref, out_cp)


def test_resume_rejects_changed_input(tmp_path):
    src = tmp_path / "model.safetensors"
    _make_model(src)
    other = tmp_path / "other.safetensors"
    _make_model(other)

    out_res = tmp_path / "resumed.safetensors"
    cp_dir = tmp_path / "cp"
    stop_file = tmp_path / "stop"
    stop_file.write_text("")

    with pytest.raises(StopRequested):
        _run(_argv(src, out_res, cp_dir, stop_file))
    stop_file.unlink()

    with pytest.raises(CheckpointMismatchError):
        _run(_argv(other, out_res, cp_dir))


def test_resume_with_zero_completed_tensors(tmp_path):
    # Stop file present from the very start: stops before tensor 1, yet the
    # checkpoint must still resume into a complete, identical output.
    src = tmp_path / "model.safetensors"
    _make_model(src)

    out_ref = tmp_path / "ref.safetensors"
    _run(_argv(src, out_ref))

    out_res = tmp_path / "resumed.safetensors"
    cp_dir = tmp_path / "cp"
    stop_file = tmp_path / "stop"
    stop_file.write_text("")

    with pytest.raises(StopRequested):
        _run(_argv(src, out_res, cp_dir, stop_file))
    assert TensorCheckpoint.open(str(cp_dir)).completed_count == 0

    stop_file.unlink()
    _run(_argv(src, out_res, cp_dir))
    _assert_outputs_identical(out_ref, out_res)
