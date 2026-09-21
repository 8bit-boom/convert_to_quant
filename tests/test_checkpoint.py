"""Unit tests for convert_to_quant.checkpoint."""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert_to_quant.checkpoint import (
    FORMAT_VERSION,
    CheckpointMismatchError,
    TensorCheckpoint,
    clear_stop_request,
    should_stop,
)


def _make_checkpoint(tmp_path, weight_keys=("a.weight", "b.weight", "c.weight"), **kw):
    defaults = dict(
        input_file=str(tmp_path / "in.safetensors"),
        seed=1234,
        target_format="fp8",
        weight_keys=list(weight_keys),
    )
    defaults.update(kw)
    return TensorCheckpoint.create(str(tmp_path / "cp"), **defaults)


def test_create_writesManifest(tmp_path):
    cp = _make_checkpoint(tmp_path)
    assert TensorCheckpoint.exists(cp.dir)
    m = json.loads((Path(cp.dir) / "manifest.json").read_text())
    assert m["version"] == FORMAT_VERSION
    assert m["seed"] == 1234
    assert m["target_format"] == "fp8"
    assert m["total"] == 3
    assert m["finished"] is False
    assert m["entries"] == []


def test_recordWritesShardThenManifestAndReplays(tmp_path):
    cp = _make_checkpoint(tmp_path)
    t1 = torch.randn(4, 8)
    s1 = torch.randn(4)
    cp.record("a.weight", {"a.weight": t1, "a.weight_scale": s1})
    cp.record("b.weight", {})  # removed-tensor style: no output

    assert cp.completed_count == 2
    assert cp.is_done("a.weight") and cp.is_done("b.weight")
    assert not cp.is_done("c.weight")

    replayed = cp.load_all()
    assert set(replayed) == {"a.weight", "a.weight_scale"}
    assert torch.equal(replayed["a.weight"], t1)
    assert torch.equal(replayed["a.weight_scale"], s1)

    # Shard file exists on disk and manifest references it
    entries = {e["key"]: e for e in cp.manifest["entries"]}
    assert entries["a.weight"]["shard"] is not None
    assert entries["b.weight"]["shard"] is None
    assert (Path(cp.dir) / "shards" / entries["a.weight"]["shard"]).is_file()


def test_recordIsIdempotentPerKey(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cp.record("a.weight", {"a.weight": torch.randn(2, 2)})
    cp.record("a.weight", {"a.weight": torch.randn(2, 2)})  # second call ignored
    assert cp.completed_count == 1
    assert len(list((Path(cp.dir) / "shards").glob("*.safetensors"))) == 1


def test_stateRoundTrips(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cp.record(
        "a.weight",
        {"a.weight": torch.randn(2, 2)},
        state={"quant_metadata_layers": {"a": {"format": "float8_e4m3fn"}}, "quantized_weight_keys": ["a.weight"]},
    )
    cp2 = TensorCheckpoint.open(cp.dir)
    assert cp2.state["quant_metadata_layers"]["a"]["format"] == "float8_e4m3fn"
    assert cp2.state["quantized_weight_keys"] == ["a.weight"]


def test_openRestoresSeedAndIdentity(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cp.record("a.weight", {"a.weight": torch.randn(2, 2)})
    cp2 = TensorCheckpoint.open(cp.dir)
    assert cp2.seed == 1234
    assert cp2.target_format == "fp8"
    assert cp2.completed_count == 1
    assert not cp2.finished


def test_markFinishedPersists(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cp.mark_finished()
    assert TensorCheckpoint.open(cp.dir).finished


def test_validateForRejectsMismatches(tmp_path):
    src = tmp_path / "in.safetensors"
    src.write_bytes(b"x")
    cp = _make_checkpoint(tmp_path, weight_keys=["a.weight", "b.weight"])
    cp.record("a.weight", {"a.weight": torch.randn(2, 2)})

    cp.validate_for(input_file=str(src), target_format="fp8", weight_keys=["a.weight", "b.weight"])  # ok

    with pytest.raises(CheckpointMismatchError):
        cp.validate_for(input_file=str(tmp_path / "other.st"), target_format="fp8", weight_keys=["a.weight", "b.weight"])
    with pytest.raises(CheckpointMismatchError):
        cp.validate_for(input_file=str(src), target_format="int8", weight_keys=["a.weight", "b.weight"])
    with pytest.raises(CheckpointMismatchError):
        cp.validate_for(input_file=str(src), target_format="fp8", weight_keys=["a.weight", "zzz.weight"])


def test_openRejectsBadVersion(tmp_path):
    cp = _make_checkpoint(tmp_path)
    manifest_path = Path(cp.dir) / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["version"] = 999
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(CheckpointMismatchError):
        TensorCheckpoint.open(cp.dir)


def test_missingShardRaisesOnReplay(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cp.record("a.weight", {"a.weight": torch.randn(2, 2)})
    entry = cp.manifest["entries"][0]
    os.unlink(Path(cp.dir) / "shards" / entry["shard"])
    with pytest.raises(CheckpointMismatchError):
        cp.load_all()


def test_crashMidShardLeavesConsistentManifest(tmp_path):
    # Simulate a crash during shard write: orphan tmp file, manifest must not
    # reference any shard, and a fresh record still indexes correctly.
    cp = _make_checkpoint(tmp_path)
    shards_dir = Path(cp.dir) / "shards"
    (shards_dir / "orphan.tmp").write_bytes(b"garbage")
    cp2 = TensorCheckpoint.open(cp.dir)
    cp2.record("a.weight", {"a.weight": torch.randn(2, 2)})
    assert cp2.load_all()["a.weight"] is not None
    assert TensorCheckpoint.open(cp.dir).completed_count == 1


def test_shouldStopViaStopFile(tmp_path):
    clear_stop_request()
    flag = tmp_path / "stop"
    assert not should_stop(str(flag))
    flag.write_text("")
    assert should_stop(str(flag))
    assert not should_stop(None)
