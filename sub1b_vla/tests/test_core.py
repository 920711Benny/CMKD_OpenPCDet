"""Unit tests for the pieces where a silent error would be most expensive:
frame conventions, loss semantics, diffusion sampling, and the async runtime's
non-blocking contract.

    python -m pytest sub1b_vla/tests -q
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch

from sub1b_vla.carla_agent.controller import TrajectoryController
from sub1b_vla.carla_agent.sensors import CameraRig
from sub1b_vla.losses.composite import LossWeights, combine
from sub1b_vla.losses.consistency import (
    ConsistencyThresholds, action_cot_alignment_score, causal_consistency_loss,
    compute_dynamics, intent_violations,
)
from sub1b_vla.models.coc_prompt import INTENTS, INTENT_TO_ID, CoCSample, parse_intent
from sub1b_vla.models.diffusion_head import CoCDiffusionHead, NoiseSchedule
from sub1b_vla.utils import load_config

CFG = "sub1b_vla/configs/tiny_cpu.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG)


@pytest.fixture(scope="module")
def model(cfg):
    from sub1b_vla.models.vla_agent import DualHeadDiffusionVLA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DualHeadDiffusionVLA(cfg).eval()


# ---------------------------------------------------------------- dynamics
def _straight(speed_ms, n=11, dt=0.2):
    x = np.cumsum(np.full(n, speed_ms * dt))
    return torch.tensor(np.stack([x, np.zeros(n)], 1), dtype=torch.float32)[None]


def test_dynamics_recovers_constant_speed():
    dyn = compute_dynamics(_straight(5.0), dt=0.2)
    assert dyn.final_speed.item() == pytest.approx(5.0, abs=0.05)
    assert abs(dyn.accel.item()) < 0.05
    assert dyn.displacement.item() == pytest.approx(11.0, abs=0.05)


def test_dynamics_detects_deceleration():
    n, dt = 11, 0.2
    v, x, pts = 8.0, 0.0, []
    for _ in range(n):
        v = max(0.0, v - 2.0 * dt)
        x += v * dt
        pts.append((x, 0.0))
    dyn = compute_dynamics(torch.tensor([pts], dtype=torch.float32), dt=dt)
    assert dyn.accel.item() < -1.0, "constant braking must yield negative acceleration"


def test_stop_trajectory_has_zero_violation_for_stop_intent():
    wp = torch.zeros(1, 11, 2)
    v = intent_violations(compute_dynamics(wp))
    assert v[0, INTENT_TO_ID["stop"]].item() == pytest.approx(0.0, abs=1e-6)
    assert v[0, INTENT_TO_ID["accelerate"]].item() > 0.0


def test_moving_trajectory_violates_stop():
    v = intent_violations(compute_dynamics(_straight(8.0)))
    assert v[0, INTENT_TO_ID["stop"]].item() > 1.0
    assert v[0, INTENT_TO_ID["keep_speed"]].item() == pytest.approx(0.0, abs=1e-6)


def test_left_and_right_turns_are_distinguished():
    n = 11
    x = np.linspace(1, 11, n)
    left = torch.tensor(np.stack([x, np.linspace(0, 5, n)], 1), dtype=torch.float32)[None]
    right = torch.tensor(np.stack([x, -np.linspace(0, 5, n)], 1), dtype=torch.float32)[None]
    vl = intent_violations(compute_dynamics(left))
    vr = intent_violations(compute_dynamics(right))
    assert vl[0, INTENT_TO_ID["turn_left"]] < vl[0, INTENT_TO_ID["turn_right"]]
    assert vr[0, INTENT_TO_ID["turn_right"]] < vr[0, INTENT_TO_ID["turn_left"]]


def test_every_intent_has_a_violation_definition():
    v = intent_violations(compute_dynamics(_straight(5.0)))
    assert v.shape[1] == len(INTENTS)
    assert torch.isfinite(v).all()


# ------------------------------------------------------------- consistency
def test_consistency_loss_is_lower_when_language_agrees_with_trajectory():
    stopped = torch.zeros(1, 11, 2)
    agree = torch.full((1, len(INTENTS)), -10.0)
    agree[0, INTENT_TO_ID["stop"]] = 10.0
    disagree = torch.full((1, len(INTENTS)), -10.0)
    disagree[0, INTENT_TO_ID["accelerate"]] = 10.0
    l_agree, _, _ = causal_consistency_loss(agree, stopped)
    l_dis, _, _ = causal_consistency_loss(disagree, stopped)
    assert l_agree.item() < l_dis.item()


def test_consistency_gradient_reaches_both_streams():
    logits = torch.zeros(2, len(INTENTS), requires_grad=True)
    wp = _straight(6.0).repeat(2, 1, 1).clone().requires_grad_(True)
    loss, _, _ = causal_consistency_loss(logits, wp)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert wp.grad is not None and wp.grad.abs().sum() > 0


def test_alignment_score_rewards_matching_intent():
    stopped = torch.zeros(4, 11, 2)
    good = action_cot_alignment_score(
        torch.full((4,), INTENT_TO_ID["stop"]), stopped)
    bad = action_cot_alignment_score(
        torch.full((4,), INTENT_TO_ID["accelerate"]), stopped)
    assert good.mean().item() == 1.0
    assert bad.mean().item() == 0.0


# ---------------------------------------------------------------- schedule
def test_noise_schedule_is_monotone_and_bounded():
    s = NoiseSchedule(100)
    ab = s.alphas_cumprod
    assert ab.shape[0] == 100
    assert bool((ab[1:] <= ab[:-1] + 1e-6).all()), "alpha_bar must be non-increasing"
    assert float(ab.min()) > 0 and float(ab.max()) <= 1.0


def test_add_noise_then_recover_x0_is_exact_with_true_eps():
    s = NoiseSchedule(100)
    x0 = torch.randn(4, 11, 2)
    eps = torch.randn_like(x0)
    t = torch.randint(0, 60, (4,))
    xt = s.add_noise(x0, eps, t)
    assert torch.allclose(s.to_x0(xt, eps, t), x0, atol=1e-4)


def test_diffusion_sampling_shape_and_determinism():
    head = CoCDiffusionHead(spatial_dim=16, sem_dim=24, dim=32, depth=1, heads=4,
                            pred_len=11, train_steps=100, infer_steps=10).eval()
    geo, sem = torch.randn(2, 9, 16), torch.randn(2, 4, 24)
    speed, tp, cmd = torch.rand(2) * 10, torch.randn(2, 2), torch.randint(0, 7, (2,))
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = head.sample(geo, sem, speed, tp, cmd, generator=g1)
    b = head.sample(geo, sem, speed, tp, cmd, generator=g2)
    assert a.shape == (2, 11, 2)
    assert torch.allclose(a, b), "DDIM with a fixed seed must be reproducible"


def test_diffusion_loss_returns_reliability_in_unit_interval():
    head = CoCDiffusionHead(spatial_dim=16, sem_dim=24, dim=32, depth=1, heads=4,
                            pred_len=11, train_steps=100, infer_steps=10)
    wp = torch.randn(3, 11, 2)
    per, x0, t, rel = head.loss(wp, torch.randn(3, 9, 16), torch.randn(3, 4, 24),
                                torch.rand(3), torch.randn(3, 2), torch.randint(0, 7, (3,)))
    assert per.shape == (3,) and x0.shape == wp.shape
    assert bool(((rel >= 0) & (rel <= 1)).all())
    assert bool((x0.abs() <= 80.0 + 1e-4).all()), "x0 must be clamped to a physical range"


# -------------------------------------------------------------- geometry
def test_camera_projection_conventions():
    rig = CameraRig()
    px, valid = rig.project_ego_to_image(np.array([[10.0, 0.0], [10.0, 3.0], [10.0, -3.0]]))
    assert valid.all()
    assert px[0, 0] == pytest.approx(rig.width / 2, abs=1e-6), "straight ahead is image centre"
    assert px[1, 0] < px[0, 0], "+y (left) must project left of centre"
    assert px[2, 0] > px[0, 0], "-y (right) must project right of centre"


def test_points_behind_camera_are_invalid():
    rig = CameraRig()
    _, valid = rig.project_ego_to_image(np.array([[-5.0, 0.0]]))
    assert not valid[0]


def test_nearer_points_project_lower_in_frame():
    rig = CameraRig()
    px, _ = rig.project_ego_to_image(np.array([[5.0, 0.0], [30.0, 0.0]]))
    assert px[0, 1] > px[1, 1], "nearer ground points sit lower (larger v)"


def test_fov_matches_focal_length():
    rig = CameraRig(fov=100.0, width=1024)
    assert rig.focal == pytest.approx(1024 / (2 * math.tan(math.radians(100) / 2)))


# ------------------------------------------------------------- controller
def test_steer_sign_follows_carla_convention():
    c = TrajectoryController()
    n = 11
    x = np.linspace(1, 11, n)
    left = c.step(np.stack([x, np.linspace(0, 4, n)], 1), 4.0)
    right = c.step(np.stack([x, -np.linspace(0, 4, n)], 1), 4.0)
    assert left.steer < 0, "CARLA steer is negative for left"
    assert right.steer > 0


def test_zero_trajectory_commands_full_brake():
    out = TrajectoryController().step(np.zeros((11, 2)), 0.0)
    assert out.brake == 1.0 and out.throttle == 0.0


def test_target_speed_tracks_waypoint_spacing():
    c = TrajectoryController(dt=0.2)
    fast = c.step(np.stack([np.cumsum(np.full(11, 8.0 * 0.2)), np.zeros(11)], 1), 8.0)
    slow = c.step(np.stack([np.cumsum(np.full(11, 2.0 * 0.2)), np.zeros(11)], 1), 2.0)
    assert fast.target_speed > slow.target_speed


# ----------------------------------------------------------------- prompt
def test_coc_render_and_parse_roundtrip():
    for intent in INTENTS:
        s = CoCSample(perception="p", causation="c", intent=intent)
        name, idx = parse_intent(s.render())
        assert name == intent and idx == INTENT_TO_ID[intent]


def test_parse_intent_falls_back_to_neutral_not_stop():
    name, _ = parse_intent("total gibberish with no action plan")
    assert name == "keep_speed", "unparseable output must not silently become 'stop'"


# ------------------------------------------------------------------ loss
def test_align_weight_warms_up_then_saturates():
    w = LossWeights(consistency=0.4, align_warmup_steps=100)
    assert w.align_weight(0) == 0.0
    assert w.align_weight(50) == pytest.approx(0.2)
    assert w.align_weight(1000) == pytest.approx(0.4)


def test_combine_respects_weights():
    w = LossWeights(lm=2.0, diffusion=3.0, consistency=1.0, align_warmup_steps=0)
    b = combine(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), w, step=0)
    assert b.total.item() == pytest.approx(6.0)


def test_combine_handles_missing_lm_loss():
    b = combine(None, torch.tensor(2.0), torch.tensor(1.0), LossWeights(align_warmup_steps=0), 0)
    assert torch.isfinite(b.total) and b.lm.item() == 0.0


# ----------------------------------------------------------------- model
def test_parameter_budget_enforced(model):
    rep = model.assert_parameter_budget()
    assert rep.total < model.param_limit
    assert rep.trainable > 0


def test_frozen_backbones_have_no_gradients(model):
    assert not any(p.requires_grad for p in model.encoder.spatial_backbone.parameters())
    assert not any(p.requires_grad for p in model.encoder.semantic_backbone.parameters())


def test_forward_shapes_and_finite_loss(model, cfg):
    b, s = 3, cfg["model"]["image_size"]
    batch = {
        "image": torch.randn(b, 3, s, s),
        "waypoints": torch.randn(b, cfg["model"]["pred_len"], 2),
        "speed": torch.rand(b) * 30,
        "target_point": torch.randn(b, 2),
        "command": torch.randint(0, 7, (b,)),
        "text_ids": torch.randint(3, 100, (b, 12)),
        "text_mask": torch.ones(b, 12, dtype=torch.long),
        "text_labels": torch.randint(3, 100, (b, 12)),
        "intent_id": torch.randint(0, len(INTENTS), (b,)),
        "has_waypoints": torch.ones(b),
    }
    out, aux = model(batch, step=0)
    assert torch.isfinite(out.total)
    assert aux["x0"].shape == (b, cfg["model"]["pred_len"], 2)


def test_drivecot_samples_excluded_from_diffusion_loss(model, cfg):
    b, s = 4, cfg["model"]["image_size"]
    base = {
        "image": torch.randn(b, 3, s, s),
        "waypoints": torch.randn(b, cfg["model"]["pred_len"], 2) * 50,
        "speed": torch.rand(b) * 30,
        "target_point": torch.randn(b, 2),
        "command": torch.randint(0, 7, (b,)),
        "intent_id": torch.zeros(b, dtype=torch.long),
    }
    torch.manual_seed(0)
    all_on, _ = model({**base, "has_waypoints": torch.ones(b)}, step=0)
    torch.manual_seed(0)
    half, _ = model({**base, "has_waypoints": torch.tensor([1.0, 1.0, 0.0, 0.0])}, step=0)
    assert not torch.isclose(all_on.diffusion, half.diffusion), \
        "masking DriveCoT samples must change the diffusion loss"


def test_predict_trajectory_shapes(model, cfg):
    b, s = 2, cfg["model"]["image_size"]
    out = model.predict_trajectory(torch.randn(b, 3, s, s), torch.rand(b) * 20,
                                   torch.randn(b, 2), torch.randint(0, 7, (b,)))
    assert out.waypoints.shape == (b, cfg["model"]["pred_len"], 2)
    assert out.intent_logits.shape == (b, len(INTENTS))
    assert torch.isfinite(out.waypoints).all()


# -------------------------------------------------------------- async
def test_async_runtime_control_path_never_blocks_on_rationale(model, cfg):
    """The control path must return even while the rationale worker is busy --
    this is the whole point of the decoupled pipeline."""
    import time

    from sub1b_vla.carla_agent.async_pipeline import AsyncVLARuntime

    s = cfg["model"]["image_size"]
    with AsyncVLARuntime(model, torch.device("cpu"), cfg) as rt:
        time.sleep(0.15)  # let the rationale worker start
        for _ in range(5):
            p = rt.perceive(np.random.randn(3, s, s).astype(np.float32),
                            10.0, np.zeros(2, np.float32), 3)
            assert p.waypoints.shape == (cfg["model"]["pred_len"], 2)
        rep = rt.latency_report()
    assert rep["trajectory"]["n"] == 5
    text, _ = rt.latest_rationale()
    assert isinstance(text, str)


def test_latest_slot_returns_most_recent_value():
    from sub1b_vla.carla_agent.async_pipeline import LatestSlot

    s = LatestSlot()
    for i in range(5):
        s.put(i)
    v, _, seq = s.get()
    assert v == 4 and seq == 5


def test_priority_gate_signals_yield():
    from sub1b_vla.carla_agent.async_pipeline import PriorityGate

    g = PriorityGate()
    assert not g.should_yield()
    g.control_begin()
    assert g.should_yield()
    g.control_end()
    assert not g.should_yield()


# --------------------------------------------------------------- data
def test_dataset_item_contract(cfg):
    from sub1b_vla.data.dataset import DrivingVLADataset, collate

    ds = DrivingVLADataset(cfg, "train")
    item = ds[0]
    s = cfg["model"]["image_size"]
    assert item["image"].shape == (3, s, s)
    assert item["waypoints"].shape == (cfg["model"]["pred_len"], 2)
    batch = collate([ds[i] for i in range(4)])
    assert batch["image"].shape[0] == 4 and len(batch["scenario"]) == 4


def test_cut_bottom_quarter_removes_exactly_a_quarter():
    from sub1b_vla.data.augment import cut_bottom_quarter

    img = np.zeros((100, 40, 3), np.float32)
    assert cut_bottom_quarter(img).shape[0] == 75


def test_augmentation_is_disabled_for_val_split(cfg):
    from sub1b_vla.data.dataset import DrivingVLADataset

    assert not DrivingVLADataset(cfg, "val").aug.enabled


def test_synthetic_red_light_implies_stop_label():
    from sub1b_vla.data.synthetic import generate_frame

    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(400):
        f = generate_frame(rng)
        if f.scenario == "red_light":
            seen += 1
            assert f.coc.intent == "stop"
            # A stop must come to REST by the horizon; it may cover ground doing so.
            seg = np.linalg.norm(np.diff(np.vstack([[0, 0], f.waypoints]), axis=0), axis=1)
            assert seg[-1] / 0.2 < 0.3, "a stop trajectory must end at rest"
    assert seen > 0, "red_light scenario never sampled"
