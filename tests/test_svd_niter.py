"""Tests for the configurable svd_niter parameter (perf: niter=1 is much faster)."""
import torch
import pytest

from convert_to_quant.converters.base_converter import BaseLearnedConverter
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter


class TestSvdNiterConfig:
    def test_default_is_4(self):
        conv = LearnedRoundingConverter(device="cpu", optimizer="adamw", num_iter=2)
        assert conv.svd_niter == 4

    def test_custom_value(self):
        conv = LearnedRoundingConverter(device="cpu", optimizer="adamw", num_iter=2, svd_niter=1)
        assert conv.svd_niter == 1

    def test_clamped_to_minimum_1(self):
        conv = LearnedRoundingConverter(device="cpu", optimizer="adamw", num_iter=2, svd_niter=0)
        assert conv.svd_niter == 1
        conv = LearnedRoundingConverter(device="cpu", optimizer="adamw", num_iter=2, svd_niter=-3)
        assert conv.svd_niter == 1

    def test_niter_1_produces_valid_components(self):
        """svd_niter=1 must still yield a usable low-rank subspace."""
        conv = LearnedRoundingConverter(device="cpu", optimizer="adamw", num_iter=2, svd_niter=1, top_p=0.5)
        W = torch.randn(256, 256)
        U_k, Vh_k, k = conv._compute_svd_components(W, verbose=False)
        assert U_k.shape[0] == 256
        assert Vh_k.shape[1] == 256
        assert k == U_k.shape[1] == Vh_k.shape[0]
        assert torch.isfinite(U_k).all()
        assert torch.isfinite(Vh_k).all()

    def test_niter_1_vs_4_subspace_quality(self):
        """niter=1 subspace should capture most of the spectral energy of niter=4."""
        torch.manual_seed(0)
        conv = LearnedRoundingConverter(device="cpu", optimizer="adamw", num_iter=2, top_p=0.5)
        W = torch.randn(256, 256)
        U4, Vh4, _ = conv._compute_svd_components(W, verbose=False)

        conv1 = LearnedRoundingConverter(device="cpu", optimizer="adamw", num_iter=2, svd_niter=1, top_p=0.5)
        U1, Vh1, _ = conv1._compute_svd_components(W, verbose=False)

        # Projection energy of the leading component direction should be close
        u4 = U4[:, 0]
        u1 = U1[:, 0]
        cos = torch.dot(u4, u1).abs() / (u4.norm() * u1.norm())
        assert cos > 0.8, f"Leading singular vectors diverged: cos={cos:.3f}"
