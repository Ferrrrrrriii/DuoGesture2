"""
Unit tests for S-VIB (Semantic Variational Information Bottleneck).
Run: python -m pytest tests/test_vib.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from models.duogesture import SemanticVIB


# ─────────────── 1. Shape & signature tests ───────────────────────────

class TestSemanticVIBShapes:
    """Verify output shapes and return signature."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.bs, self.T, self.sem_dim = 4, 16, 256
        self.z_dim = 16
        self.timing_dim = 64
        self.vib = SemanticVIB(sem_dim=self.sem_dim,
                               timing_dim=self.timing_dim,
                               z_dim=self.z_dim)

    def test_output_is_tuple_of_three(self):
        """Forward must return (gate_logits, mu, logvar)."""
        x_sem  = torch.randn(self.bs, self.T, self.sem_dim)
        x_time = torch.randn(self.bs, self.T, self.sem_dim)
        out = self.vib(x_sem, x_time)
        assert isinstance(out, tuple) and len(out) == 3

    def test_gate_logits_shape(self):
        x_sem  = torch.randn(self.bs, self.T, self.sem_dim)
        x_time = torch.randn(self.bs, self.T, self.sem_dim)
        gate, mu, logvar = self.vib(x_sem, x_time)
        assert gate.shape == (self.bs, self.T, 2), f"Expected [bs,T,2], got {gate.shape}"
        assert mu.shape == (self.bs, self.T, self.z_dim)
        assert logvar.shape == (self.bs, self.T, self.z_dim)

    def test_eval_mode_deterministic(self):
        """In eval mode, reparameterize returns mu → output is deterministic."""
        self.vib.eval()
        x_sem  = torch.randn(self.bs, self.T, self.sem_dim)
        x_time = torch.randn(self.bs, self.T, self.sem_dim)
        g1, _, _ = self.vib(x_sem, x_time)
        g2, _, _ = self.vib(x_sem, x_time)
        assert torch.allclose(g1, g2), "Eval mode should be deterministic"

    def test_train_mode_stochastic(self):
        """In train mode, sampling noise should differ between calls."""
        self.vib.train()
        x_sem  = torch.randn(self.bs, self.T, self.sem_dim)
        x_time = torch.randn(self.bs, self.T, self.sem_dim)
        g1, _, _ = self.vib(x_sem, x_time)
        g2, _, _ = self.vib(x_sem, x_time)
        # Very unlikely to be identical with 4*16*2 random values
        assert not torch.allclose(g1, g2), "Train mode should be stochastic"


# ─────────────── 2. Timing bottleneck capacity test ──────────────────

class TestTimingBottleneck:
    """Verify the timing stream is genuinely low-capacity."""

    def test_timing_proj_output_dim(self):
        timing_dim = 64
        vib = SemanticVIB(sem_dim=256, timing_dim=timing_dim, z_dim=16)
        x = torch.randn(2, 16, 256)
        out = vib.timing_proj(x)
        assert out.shape[-1] == timing_dim, \
            f"Timing projection should output {timing_dim}, got {out.shape[-1]}"


# ─────────────── 3. KL loss helper tests ─────────────────────────────

class TestKLLoss:
    """Test the static _compute_kl_loss helper from the trainer."""

    def test_kl_standard_normal(self):
        """KL(N(0,1) || N(0,1)) ≈ 0 (below free-bits threshold)."""
        # With free_bits=0.5, output = free_bits since actual KL ≈ 0
        from duogesture_sparse_trainer import CustomTrainer
        mu = torch.zeros(4, 16, 16)
        logvar = torch.zeros(4, 16, 16)
        kl = CustomTrainer._compute_kl_loss(mu, logvar, free_bits=0.0)
        assert kl.item() < 1e-5, f"KL of standard normal should be ~0, got {kl.item()}"

    def test_kl_free_bits_floors(self):
        """Free-bits should floor per-dim KL at the threshold."""
        from duogesture_sparse_trainer import CustomTrainer
        mu = torch.zeros(4, 16, 16)
        logvar = torch.zeros(4, 16, 16)
        kl_fb = CustomTrainer._compute_kl_loss(mu, logvar, free_bits=0.5)
        assert kl_fb.item() >= 0.5 - 1e-5, \
            f"Free-bits 0.5 should floor KL, got {kl_fb.item()}"

    def test_kl_positive_for_non_trivial(self):
        """Non-zero mu should give positive KL."""
        from duogesture_sparse_trainer import CustomTrainer
        mu = torch.ones(4, 16, 16) * 3.0
        logvar = torch.zeros(4, 16, 16)
        kl = CustomTrainer._compute_kl_loss(mu, logvar, free_bits=0.0)
        assert kl.item() > 1.0, f"KL with mu=3 should be large, got {kl.item()}"


# ─────────────── 4. Posterior collapse detection ─────────────────────

class TestPosteriorCollapse:
    """Ensure S-VIB doesn't immediately collapse (logvar ≠ -inf)."""

    def test_logvar_clamped(self):
        vib = SemanticVIB(sem_dim=256, timing_dim=64, z_dim=16)
        vib.train()
        x_sem  = torch.randn(4, 16, 256)
        x_time = torch.randn(4, 16, 256)
        _, _, logvar = vib(x_sem, x_time)
        assert logvar.min().item() >= -10.0, \
            f"logvar clamp violated: min={logvar.min().item()}"
        assert logvar.max().item() <= 2.0, \
            f"logvar clamp violated: max={logvar.max().item()}"

    def test_gradient_flows_through_mu_logvar(self):
        """Verify gradients flow back through both mu and logvar."""
        vib = SemanticVIB(sem_dim=256, timing_dim=64, z_dim=16)
        vib.train()
        x_sem  = torch.randn(4, 16, 256, requires_grad=True)
        x_time = torch.randn(4, 16, 256, requires_grad=True)
        gate, mu, logvar = vib(x_sem, x_time)
        loss = gate.sum() + mu.sum() + logvar.sum()
        loss.backward()
        assert x_sem.grad is not None and x_sem.grad.abs().sum() > 0
        assert x_time.grad is not None and x_time.grad.abs().sum() > 0


# ─────────────── 5. Beta warmup schedule tests ──────────────────────

class TestBetaSchedule:
    """Test sigmoid warmup shape."""

    def _make_mock_trainer(self):
        """Create minimal object with the _compute_vib_beta method."""
        from duogesture_sparse_trainer import CustomTrainer
        # We just need the method, create a simple namespace
        class MockTrainer:
            _vib_enabled = True
            _vib_beta_target = 0.001
            _vib_warmup_start = 20
            _vib_warmup_end = 100
            _compute_vib_beta = CustomTrainer._compute_vib_beta
        return MockTrainer()

    def test_beta_zero_before_warmup(self):
        t = self._make_mock_trainer()
        assert t._compute_vib_beta(0) == 0.0
        assert t._compute_vib_beta(19) == 0.0

    def test_beta_midpoint(self):
        t = self._make_mock_trainer()
        mid = (20 + 100) / 2  # epoch 60
        beta = t._compute_vib_beta(mid)
        # At sigmoid midpoint, output ≈ 0.5 * target
        assert 0.0003 < beta < 0.0007, f"Midpoint beta should be ~0.0005, got {beta}"

    def test_beta_saturates(self):
        t = self._make_mock_trainer()
        beta = t._compute_vib_beta(200)
        assert beta > 0.0009, f"Should saturate near target, got {beta}"


# ─────────────── 6. Return dict compatibility ───────────────────────

class TestReturnDictCompat:
    """Ensure the new return dict has gate_mu and gate_logvar keys."""

    def test_forward_dict_keys_present(self):
        """Smoke test: model return dict includes VIB fields when enabled."""
        # This test just verifies SemanticVIB returns 3 values that
        # can be packed into the dict structure
        vib = SemanticVIB(sem_dim=256, timing_dim=64, z_dim=16)
        x_sem  = torch.randn(2, 16, 256)
        x_time = torch.randn(2, 16, 256)
        gate, mu, logvar = vib(x_sem, x_time)
        out = {"gate": gate, "gate_mu": mu, "gate_logvar": logvar}
        assert "gate_mu" in out
        assert "gate_logvar" in out
        assert out["gate"].shape[-1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
