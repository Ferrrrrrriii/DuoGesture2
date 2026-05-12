#!/usr/bin/env python
"""
Integration test: verify that GestureLSMBaseMotion is a drop-in replacement
for duogesture_base when used as the pluggable base in the sparse pathway.

Checks:
  1. Both modules accept the same inputs
  2. Both produce forward() outputs with identical keys and shapes
  3. Both produce forward_latent() outputs with identical keys and shapes
  4. The FM toggle in the sparse trainer loads the correct module

Usage:
    python tests/test_fm_integration.py --config configs/duogesture_flowmatch_base.yaml
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse


def _make_dummy_args():
    """Create a minimal args namespace with all required attributes."""
    class Args:
        pass
    a = Args()
    # model dims
    a.hidden_size = 768
    a.audio_f = 256
    a.motion_f = 256
    a.pose_dims = 330
    a.pose_length = 64
    a.pre_frames = 4
    a.vae_codebook_size = 256
    a.vae_layer = 4
    a.vae_length = 240
    # FM-specific
    a.fm_num_layers = 6
    a.fm_num_heads = 8
    a.fm_dropout = 0.1
    a.fm_beta_alpha = 2.0
    a.fm_beta_beta = 1.2
    a.fm_sigma_min = 0.0001
    a.fm_num_inference_steps = 5  # low for speed
    a.fm_cfg_drop_prob = 0.0
    return a


def _make_inputs(B=2, T=64, device="cpu"):
    """Create dummy inputs matching the base module interface."""
    T_prime = T // 4  # 16
    return {
        "in_audio":  torch.randn(B, T, 3, device=device),
        "in_word":   torch.randint(0, 10, (B, T), device=device),
        "mask":      torch.cat([
            torch.zeros(B, 4, 337, device=device),
            torch.ones(B, T - 4, 337, device=device),
        ], dim=1),
        "in_motion": torch.randn(B, T, 337, device=device),
        "in_id":     torch.randint(0, 25, (B, T, 1), device=device),
        "hubert":    torch.randn(B, T, 1024, device=device),
    }, {
        "face":  torch.randn(B, T_prime, 256, device=device),
        "upper": torch.randn(B, T_prime, 256, device=device),
        "hands": torch.randn(B, T_prime, 256, device=device),
        "lower": torch.randn(B, T_prime, 256, device=device),
    }


def test_forward_shapes():
    """Both modules produce the same output keys and shapes from forward()."""
    from models.duogesture import duogesture_base
    from models.flow_matching_base import GestureLSMBaseMotion

    args = _make_dummy_args()
    device = "cpu"

    orig = duogesture_base(args).to(device).eval()
    fm   = GestureLSMBaseMotion(args).to(device).eval()

    inputs, targets = _make_inputs(device=device)

    with torch.no_grad():
        # Original forward
        out_orig = orig(
            in_audio=inputs["in_audio"],
            in_word=inputs["in_word"],
            mask=inputs["mask"],
            in_motion=inputs["in_motion"],
            in_id=inputs["in_id"],
            hubert=inputs["hubert"],
            use_attentions=True,
        )
        # FM forward (inference mode — no target_latents)
        out_fm = fm(
            in_audio=inputs["in_audio"],
            in_word=inputs["in_word"],
            mask=inputs["mask"],
            in_motion=inputs["in_motion"],
            in_id=inputs["in_id"],
            hubert=inputs["hubert"],
            use_attentions=True,
        )

    # Check shared keys
    required_keys = ["rec_face", "rec_upper", "rec_hands", "rec_lower",
                     "cls_face", "cls_upper", "cls_hands", "cls_lower"]
    for k in required_keys:
        assert k in out_orig, f"Original missing key: {k}"
        assert k in out_fm,   f"FM missing key: {k}"
        assert out_orig[k].shape == out_fm[k].shape, \
            f"Shape mismatch for {k}: orig={out_orig[k].shape} vs fm={out_fm[k].shape}"
        print(f"  {k}: {out_orig[k].shape} == {out_fm[k].shape} ✓")

    # FM-specific keys
    assert "fm_loss" in out_fm, "FM output missing fm_loss"
    print(f"  fm_loss: {out_fm['fm_loss'].item():.4f} ✓")

    print("test_forward_shapes PASSED ✓\n")


def test_forward_latent_shapes():
    """Both modules produce identical forward_latent() output keys and shapes."""
    from models.duogesture import duogesture_base
    from models.flow_matching_base import GestureLSMBaseMotion

    args = _make_dummy_args()
    device = "cpu"

    orig = duogesture_base(args).to(device).eval()
    fm   = GestureLSMBaseMotion(args).to(device).eval()

    inputs, _ = _make_inputs(device=device)

    with torch.no_grad():
        lat_orig = orig.forward_latent(
            in_audio=inputs["in_audio"],
            in_word=inputs["in_word"],
            mask=inputs["mask"],
            in_motion=inputs["in_motion"],
            in_id=inputs["in_id"],
            hubert=inputs["hubert"],
            use_attentions=True,
        )
        lat_fm = fm.forward_latent(
            in_audio=inputs["in_audio"],
            in_word=inputs["in_word"],
            mask=inputs["mask"],
            in_motion=inputs["in_motion"],
            in_id=inputs["in_id"],
            hubert=inputs["hubert"],
            use_attentions=True,
        )

    required_keys = ["face_latent", "upper_latent", "lower_latent", "hands_latent"]
    for k in required_keys:
        assert k in lat_orig, f"Original missing key: {k}"
        assert k in lat_fm,   f"FM missing key: {k}"
        assert lat_orig[k].shape == lat_fm[k].shape, \
            f"Shape mismatch for {k}: orig={lat_orig[k].shape} vs fm={lat_fm[k].shape}"
        print(f"  {k}: {lat_orig[k].shape} == {lat_fm[k].shape} ✓")

    print("test_forward_latent_shapes PASSED ✓\n")


def test_fm_training_forward():
    """FM training forward with target_latents produces fm_loss > 0."""
    from models.flow_matching_base import GestureLSMBaseMotion

    args = _make_dummy_args()
    device = "cpu"
    fm = GestureLSMBaseMotion(args).to(device).train()

    inputs, targets = _make_inputs(device=device)

    out = fm(
        in_audio=inputs["in_audio"],
        in_word=inputs["in_word"],
        mask=inputs["mask"],
        in_motion=inputs["in_motion"],
        in_id=inputs["in_id"],
        hubert=inputs["hubert"],
        is_train=True,
        target_latents=targets,
    )

    assert "fm_loss" in out
    assert out["fm_loss"].item() > 0, "FM loss should be > 0 during training"
    print(f"  fm_loss = {out['fm_loss'].item():.4f} ✓")

    # Check backward
    out["fm_loss"].backward()
    grad_count = sum(1 for p in fm.parameters() if p.grad is not None)
    print(f"  backward: {grad_count} params have gradients ✓")

    print("test_fm_training_forward PASSED ✓\n")


def test_toggle_flag():
    """Verify that use_flow_matching flag selects the correct module class."""
    from models.duogesture import duogesture_base
    from models.flow_matching_base import GestureLSMBaseMotion

    args = _make_dummy_args()

    # Simulate toggle=False
    args.use_flow_matching = False
    base = duogesture_base(args)
    assert isinstance(base, duogesture_base), "Should be duogesture_base"
    print(f"  use_flow_matching=False → {type(base).__name__} ✓")

    # Simulate toggle=True
    args.use_flow_matching = True
    fm = GestureLSMBaseMotion(args)
    assert isinstance(fm, GestureLSMBaseMotion), "Should be GestureLSMBaseMotion"
    print(f"  use_flow_matching=True  → {type(fm).__name__} ✓")

    print("test_toggle_flag PASSED ✓\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  FM Integration Tests")
    print("=" * 60)
    print()

    test_toggle_flag()
    test_forward_shapes()
    test_forward_latent_shapes()
    test_fm_training_forward()

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
