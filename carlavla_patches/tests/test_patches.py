"""Tests for the CarlaVLA drop-in patches.

The contract these must protect: existing checkpoints keep loading. Every patch
preserves parameter names and shapes, so a fix cannot cost the trained weights.
"""
from __future__ import annotations

import pytest
import torch

from carlavla_patches.diffusion_decoder import (
    CollisionHead, DiffusionHead, coord_range_report, fit_coord_range,
)
from carlavla_patches.flash_mha import FlashMultiheadAttention

# The real CarlaVLA coordinate range, from the original diffusion_decoder docstring.
REAL_WP = torch.stack([torch.linspace(2.5, 19.3, 10),
                       torch.linspace(-0.01, 4.8, 10)], 1)[None]


# ------------------------------------------------------------ flash MHA
def test_flash_mha_is_state_dict_compatible_with_nn_multiheadattention():
    """The whole point of keeping in_proj_weight rather than separate q/k/v
    Linears: an existing checkpoint must load with no key renaming."""
    ref = torch.nn.MultiheadAttention(896, 8, batch_first=True)
    mine = FlashMultiheadAttention(896, 8)
    assert sorted(ref.state_dict()) == sorted(mine.state_dict())
    for k, v in ref.state_dict().items():
        assert mine.state_dict()[k].shape == v.shape, k
    mine.load_state_dict(ref.state_dict(), strict=True)


def test_flash_mha_output_matches_nn_multiheadattention_exactly():
    torch.manual_seed(0)
    ref = torch.nn.MultiheadAttention(128, 8, batch_first=True).eval()
    mine = FlashMultiheadAttention(128, 8).eval()
    mine.load_state_dict(ref.state_dict())
    q, kv = torch.randn(2, 32, 128), torch.randn(2, 48, 128)
    with torch.no_grad():
        a, _ = ref(q, kv, kv, need_weights=False)
        b, _ = mine(q, kv, kv)
        c, w = mine(q, kv, kv, need_weights=True)
    assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
    assert torch.allclose(a, c, atol=1e-5)
    assert w.shape == (2, 32, 48)


def test_flash_mha_defaults_to_not_returning_weights():
    """nn.MultiheadAttention defaults need_weights=True, which materialises the
    attention matrix and disables every fast kernel. The replacement must not
    inherit that default."""
    import inspect

    sig = inspect.signature(FlashMultiheadAttention.forward)
    assert sig.parameters["need_weights"].default is False


def test_flash_mha_rejects_masks_it_does_not_implement():
    mine = FlashMultiheadAttention(32, 4)
    x = torch.randn(1, 4, 32)
    with pytest.raises(NotImplementedError):
        mine(x, x, x, attn_mask=torch.zeros(4, 4))


# ------------------------------------------------- coordinate range
def test_original_coord_range_wastes_most_of_the_diffusion_range():
    """coord_min_max=(-32,32) leaves the lateral axis using under a tenth of
    [-1,1]. Pinned because this, not the parameterisation, is the dominant
    error source on this architecture."""
    rep = coord_range_report(REAL_WP, (-32.0, 32.0))
    fwd, lat = rep["per_axis_fraction_of_range_used"]
    assert fwd < 0.30, fwd
    assert lat < 0.10, lat
    assert abs(rep["per_axis_centre_offset"][0]) > 0.3, "forward axis is far off-centre"


def test_fitted_range_centres_and_fills_both_axes():
    fitted = fit_coord_range(REAL_WP, margin=0.15)
    rep = coord_range_report(REAL_WP, fitted)
    for used in rep["per_axis_fraction_of_range_used"]:
        assert used > 0.8, used
    for off in rep["per_axis_centre_offset"]:
        assert abs(off) < 1e-3, off


def test_fit_coord_range_is_per_dimension_not_symmetric():
    """A symmetric range cannot centre data whose forward axis is strictly
    positive; that was the bug in the first attempt at this fix."""
    fitted = fit_coord_range(REAL_WP)
    assert isinstance(fitted, list) and len(fitted) == 2
    (lo_x, hi_x), (lo_y, hi_y) = fitted
    assert lo_x > 0, "forward range must not be forced symmetric about zero"
    assert lo_y < 0 < hi_y, "lateral range straddles zero"


# ------------------------------------------------------ diffusion head
def test_normalize_denormalize_round_trips_per_axis():
    head = DiffusionHead(cond_dim=16, coord_min_max=fit_coord_range(REAL_WP))
    assert torch.allclose(head._denormalize(head._normalize(REAL_WP)), REAL_WP, atol=1e-4)


def test_scalar_coord_range_still_accepted_for_backwards_compatibility():
    head = DiffusionHead(cond_dim=16, coord_min_max=(-32.0, 32.0))
    assert head.coord_min_max.shape == (2, 2)
    assert torch.allclose(head._denormalize(head._normalize(REAL_WP)), REAL_WP, atol=1e-3)


def test_coord_range_is_saved_with_the_checkpoint():
    """Normalising at inference with a different range than training silently
    rescales every trajectory, so the range must be a persistent buffer."""
    head = DiffusionHead(cond_dim=16, coord_min_max=fit_coord_range(REAL_WP))
    assert "coord_min_max" in head.state_dict()


def test_v_and_epsilon_both_round_trip_their_own_target():
    for ptype in ("v", "epsilon"):
        head = DiffusionHead(cond_dim=16, prediction_type=ptype)
        head.schedule.to(torch.device("cpu"))
        x0 = torch.randn(3, 10, 2) * 0.4
        noise = torch.randn_like(x0)
        t = torch.randint(0, 90, (3,))
        x_t = head.schedule.q_sample(x0, t, noise)
        target = (head.schedule.velocity(x0, t, noise) if ptype == "v" else noise)
        rec_x0, rec_eps = head._resolve(x_t, target, t)
        assert torch.allclose(rec_x0, x0, atol=1e-3), ptype
        assert torch.allclose(rec_eps, noise, atol=1e-3), ptype


def test_invalid_prediction_type_is_rejected():
    with pytest.raises(ValueError, match="prediction_type"):
        DiffusionHead(cond_dim=16, prediction_type="x0")


def test_compute_loss_keeps_the_original_return_shape():
    """The caller reduces this itself; changing the shape would break it."""
    head = DiffusionHead(cond_dim=16)
    loss = head.compute_loss(torch.randn(4, 10, 16), torch.randn(4, 10, 2))
    assert loss.shape == (4, 10)


def test_sample_output_stays_inside_the_configured_range():
    head = DiffusionHead(cond_dim=16, coord_min_max=fit_coord_range(REAL_WP)).eval()
    out = head.sample(torch.randn(2, 10, 16))
    (lo_x, hi_x), (lo_y, hi_y) = fit_coord_range(REAL_WP)
    assert out[..., 0].min() >= lo_x - 1e-3 and out[..., 0].max() <= hi_x + 1e-3
    assert out[..., 1].min() >= lo_y - 1e-3 and out[..., 1].max() <= hi_y + 1e-3


def test_collision_head_interface_is_unchanged():
    head = CollisionHead(cond_dim=64)
    cond = torch.randn(5, 1, 64)
    assert head(cond).shape == (5,)
    assert head.compute_loss(cond, torch.zeros(5)).shape == (5,)
    assert head.predict_prob(cond).shape == (5,)


# ----------------------------------------------------- cross fusion
def test_cross_attention_fusion_keeps_the_original_state_dict_layout():
    """Existing cross_fusion weights must load into the patched module."""
    import torch.nn as nn

    from carlavla_patches.dual_vision_model import CrossAttentionFusion

    patched = CrossAttentionFusion(1024, 1152, hidden=896)

    class OriginalFusion(nn.Module):
        def __init__(self, dino_dim, siglip_dim, hidden=896, n_heads=8):
            super().__init__()
            self.dino_proj = nn.Linear(dino_dim, hidden)
            self.siglip_proj = nn.Linear(siglip_dim, hidden)
            self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                     nn.Linear(hidden * 2, hidden))
            self.norm2 = nn.LayerNorm(hidden)

    original = OriginalFusion(1024, 1152)
    assert sorted(original.state_dict()) == sorted(patched.state_dict())
    patched.load_state_dict(original.state_dict(), strict=True)

    torch.manual_seed(0)
    dino, siglip = torch.randn(2, 256, 1024), torch.randn(2, 256, 1152)
    with torch.no_grad():
        q = original.dino_proj(dino)
        kv = original.siglip_proj(siglip)
        a, _ = original.cross_attn(q, kv, kv)
        ref = original.norm2(original.norm1(q + a) + original.ffn(original.norm1(q + a)))
        got = patched(dino, siglip)
    assert torch.allclose(ref, got, atol=1e-5), float((ref - got).abs().max())
