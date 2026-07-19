"""Tests for adaptive learned-rounding convergence control."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from convert_to_quant.converters.base_converter import BaseLearnedConverter
from convert_to_quant.converters.convergence import AdaptiveConvergenceController, TuningReportCollector, convergence_window
from convert_to_quant.converters.learned_mxfp8 import LearnedMXFP8Converter
from convert_to_quant.converters.learned_nvfp4 import LearnedNVFP4Converter
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter


class _RetryConverter(BaseLearnedConverter):
    def convert(self, W_orig, key=None, depth=-1, **kwargs):
        self._active_layer_key = key
        return self._run_selected_optimizer(W_orig)

    def _optimize_original(self, weight):
        self.seen_schedules = getattr(self, "seen_schedules", [])
        self.seen_schedules.append(self.lr_schedule)
        best = math.inf
        result = weight.clone()
        for iteration in range(self.num_iter):
            loss = float("nan") if self.lr > 0.2 else 1.0 / (iteration + 1)
            improved = self._check_improvement(loss, best)
            if improved:
                best = loss
                result = torch.full_like(weight, self.lr)
            if self._active_auto_controller and self._active_auto_controller.should_stop:
                break
        return result


@pytest.mark.unit
def test_convergence_window_is_shape_aware_and_bounded():
    square = convergence_window((128, 128), rank=128)
    wide = convergence_window((128, 2048), rank=128)
    large_rank = convergence_window((2048, 2048), rank=1024)

    assert 16 <= square <= 128
    assert square < wide <= 128
    assert square < large_rank <= 128


@pytest.mark.unit
def test_controller_stops_a_quiet_plateau_without_owning_the_scheduler():
    controller = AdaptiveConvergenceController(
        shape=(16, 16), rank=16, optimizer="adamw", initial_lr=0.1, budget=128, kind="selected"
    )
    best = math.inf
    for _ in range(96):
        improved = controller.observe(1.0, best)
        if improved:
            best = 1.0
        if controller.should_stop:
            break

    assert controller.should_stop
    assert controller.summary.stop_reason == "converged_plateau"
    assert controller.summary.iterations < controller.summary.budget


@pytest.mark.unit
def test_controller_marks_nonfinite_loss_for_retry():
    controller = AdaptiveConvergenceController(
        shape=(64, 64), rank=32, optimizer="original", initial_lr=1.0, budget=100, kind="selected"
    )
    controller.observe(1.0, math.inf)
    controller.observe(float("nan"), 1.0)

    assert controller.should_stop
    assert controller.retry_recommended
    assert controller.summary.stop_reason == "nonfinite"
    assert controller.summary.retry_reason == "nonfinite_loss"


@pytest.mark.unit
def test_dispatch_uses_one_bounded_retry_and_keeps_it_inside_budget():
    converter = _RetryConverter(optimizer="original", num_iter=96, lr=1.0, device="cpu", auto_tune=True)

    result = converter.convert(torch.zeros(4, 4), key="retry.weight")
    record = converter.get_tuning_report()["layers"][0]

    assert record["retried"]
    assert record["iterations"] <= record["budget"]
    assert len([attempt for attempt in record["attempts"] if attempt["kind"] == "retry"]) == 1
    assert torch.allclose(result, torch.full_like(result, 0.1))


@pytest.mark.unit
def test_auto_tuning_preserves_the_selected_scheduler():
    converter = _RetryConverter(
        optimizer="original", num_iter=8, lr=0.1, lr_schedule="plateau", device="cpu", auto_tune=True
    )

    converter.convert(torch.zeros(4, 4), key="scheduler.weight")
    record = converter.get_tuning_report()["layers"][0]

    assert converter.seen_schedules == ["plateau"]
    assert record["scheduler"] == "plateau"
    assert converter.lr_schedule == "plateau"


@pytest.mark.unit
def test_tuning_report_collector_writes_versioned_json():
    report_path = Path("test_auto_tuning_collector.json")
    try:
        collector = TuningReportCollector(str(report_path))
        collector.add({"layer": "layer.weight", "stop_reason": "budget", "retried": False})

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["version"] == 2
        assert payload["mode"] == "auto"
        assert payload["summary"]["layers"] == 1
        assert payload["layers"][0]["layer"] == "layer.weight"
    finally:
        report_path.unlink(missing_ok=True)


@pytest.mark.integration
def test_auto_tuning_respects_budget_restores_configuration_and_reports():
    torch.manual_seed(123)
    weight = torch.randn(16, 16)
    report_path = Path("test_auto_tuning_integration.json")
    try:
        converter = LearnedRoundingConverter(
            target_format="fp8",
            scaling_mode="tensor",
            optimizer="original",
            num_iter=96,
            lr=0.1,
            lr_schedule="plateau",
            top_p=0.5,
            min_k=2,
            max_k=4,
            device="cpu",
            auto_tune=True,
            auto_tune_report=str(report_path),
        )

        qdata, scale, dequantized, extra = converter.convert(weight, key="layer.weight", has_bias=False)
        record = converter.get_tuning_report()["layers"][0]

        assert qdata.shape == weight.shape
        assert dequantized.shape == weight.shape
        assert scale.ndim == 0
        assert isinstance(extra, dict)
        assert record["iterations"] <= 96
        assert len([attempt for attempt in record["attempts"] if attempt["kind"] == "probe"]) == 0
        assert record["selected_lr"] == 0.1
        assert record["effective_initial_lr"] == 0.1
        assert record["schedule_horizon"] == 2000
        assert record["best_loss"] is not None
        assert converter.num_iter == 96
        assert converter.lr == 0.1
        assert converter.lr_schedule == "plateau"
        assert json.loads(report_path.read_text(encoding="utf-8"))["summary"]["layers"] == 1
    finally:
        report_path.unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", ("original", "adamw", "radam", "prodigy"))
def test_auto_tuning_runs_through_every_optimizer(optimizer):
    torch.manual_seed(456)
    weight = torch.randn(8, 8)
    converter = LearnedRoundingConverter(
        target_format="fp8",
        scaling_mode="tensor",
        optimizer=optimizer,
        num_iter=4,
        lr=0.1,
        top_p=0.5,
        min_k=2,
        max_k=4,
        device="cpu",
        auto_tune=True,
    )

    qdata, _, dequantized, _ = converter.convert(weight, key=f"{optimizer}.weight", has_bias=False)

    assert qdata.shape == weight.shape
    assert dequantized.shape == weight.shape
    assert converter.get_tuning_report()["summary"]["layers"] == 1
    record = converter.get_tuning_report()["layers"][0]
    assert record["attempts"][0]["kind"] == "selected"
    assert not any(attempt["kind"] == "probe" for attempt in record["attempts"])
    if optimizer == "prodigy":
        assert converter.configured_lr == 0.1
        assert converter.lr == 1.0
        assert record["configured_lr"] == 0.1
        assert record["effective_initial_lr"] == 1.0


@pytest.mark.unit
def test_auto_controller_captures_small_strict_improvements():
    controller = AdaptiveConvergenceController(
        shape=(16, 16), rank=16, optimizer="adamw", initial_lr=0.1, budget=64, kind="selected"
    )

    assert controller.observe(1.0, math.inf)
    assert controller.observe(1.0 - 5e-7, 1.0)
    assert controller.best_loss == pytest.approx(1.0 - 5e-7)


@pytest.mark.unit
def test_internal_schedule_clock_is_independent_of_maximum_budget():
    short = _RetryConverter(optimizer="original", num_iter=500, lr=0.1, device="cpu")
    long = _RetryConverter(optimizer="original", num_iter=8000, lr=0.1, device="cpu")

    for iteration in (0, 50, 500, 1999, 2000, 4000):
        assert short._optimization_schedule_progress(iteration) == long._optimization_schedule_progress(iteration)
    assert short._optimization_schedule_interval(4) == long._optimization_schedule_interval(4) == 500


@pytest.mark.unit
def test_controller_trajectory_is_independent_of_maximum_budget():
    short = AdaptiveConvergenceController(
        shape=(16, 16), rank=16, optimizer="adamw", initial_lr=0.1, budget=64, kind="selected"
    )
    long = AdaptiveConvergenceController(
        shape=(16, 16), rank=16, optimizer="adamw", initial_lr=0.1, budget=256, kind="selected"
    )
    for iteration in range(64):
        loss = 1.0 if iteration < 40 else 0.9
        short_improved = short.observe(loss, short.best_loss)
        long_improved = long.observe(loss, long.best_loss)

        assert short_improved == long_improved
        assert short.should_stop == long.should_stop

    assert short.losses == long.losses
    assert short.summary.windows == long.summary.windows


@pytest.mark.unit
def test_prodigy_controller_does_not_stop_during_adaptation():
    controller = AdaptiveConvergenceController(
        shape=(8, 8), rank=8, optimizer="prodigy", initial_lr=1.0, budget=128, kind="selected"
    )
    for _ in range(49):
        controller.observe(1.0, controller.best_loss)
        assert not controller.should_stop


@pytest.mark.integration
@pytest.mark.parametrize(
    ("converter_type", "shape"),
    ((LearnedMXFP8Converter, (32, 32)), (LearnedNVFP4Converter, (16, 16))),
)
def test_auto_tuning_covers_block_float_formats(converter_type, shape):
    torch.manual_seed(789)
    converter = converter_type(
        optimizer="original",
        num_iter=2,
        lr=0.1,
        top_p=0.5,
        min_k=2,
        max_k=4,
        device="cpu",
        auto_tune=True,
    )

    converter.convert(torch.randn(*shape), key=f"{converter_type.__name__}.weight")
    record = converter.get_tuning_report()["layers"][0]

    assert record["converter"] == converter_type.__name__
    assert record["iterations"] == 2
    assert record["rank"] == 4


@pytest.mark.integration
def test_auto_tuning_covers_int8_convrot_adaround():
    torch.manual_seed(246)
    weight = torch.randn(8, 16)
    calibration = torch.randn(4, 16)
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=16,
        optimizer="adamw",
        num_iter=4,
        lr=0.1,
        top_p=0.5,
        min_k=2,
        max_k=4,
        device="cpu",
        auto_tune=True,
    )

    qdata, scale, dequantized, _ = converter.convert(
        weight,
        key="convrot.weight",
        calibration_data=calibration,
        has_bias=False,
    )
    record = converter.get_tuning_report()["layers"][0]

    assert qdata.dtype == torch.int8
    assert scale.shape == (8, 1)
    assert dequantized.shape == weight.shape
    assert record["method"] == "_optimize_int8_adaround"
