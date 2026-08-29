"""Unit tests for the pieces where a silent error would be most expensive:
frame conventions, loss semantics, diffusion sampling, and the async runtime's
non-blocking contract.

    python -m pytest sub1b_vla/tests -q
"""
from __future__ import annotations

import json
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
        for _ in range(5):
            p = rt.perceive(np.random.randn(3, s, s).astype(np.float32),
                            10.0, np.zeros(2, np.float32), 3)
            assert p.waypoints.shape == (cfg["model"]["pred_len"], 2)
        # Give the rationale worker a bounded chance to publish.
        deadline = time.time() + 20.0
        while time.time() < deadline:
            _, frame = rt.latest_rationale()
            if frame >= 0:
                break
            time.sleep(0.1)
        rep = rt.latency_report()
        text, frame = rt.latest_rationale()

    assert rep["trajectory"]["n"] == 5
    assert frame >= 0, "the rationale worker never published"
    # A silent exception in the worker would surface here rather than passing
    # as "some string was returned".
    assert not text.startswith("(rationale unavailable"), text
    assert rep["rationale"]["n"] >= 1, "no rationale latency was recorded"


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


def test_surrogate_red_light_implies_stop_label():
    from sub1b_vla.data.carla_surrogate import generate_frame

    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(400):
        f = generate_frame(rng)
        if f.scenario == "OppositeVehicleRunningRedLight":
            seen += 1
            assert f.coc.intent == "stop"
            # A stop must come to REST by the horizon; it may cover ground doing so.
            seg = np.linalg.norm(np.diff(np.vstack([[0, 0], f.waypoints]), axis=0), axis=1)
            assert seg[-1] / 0.2 < 0.3, "a stop trajectory must end at rest"
            assert "leading_object_traffic.traffic_light" in f.buckets
    assert seen > 0, "red-light scenario never sampled"


def test_surrogate_speaks_carla_vocabulary():
    """Every field a surrogate frame carries must be CARLA's own vocabulary, so
    swapping in the real dataset changes pixels and not field names."""
    from sub1b_vla.data.carla_buckets import SIMLINGO_TRAIN_PARTITIONS
    from sub1b_vla.data.carla_surrogate import (
        COMMANDS, SCENARIOS, TOWNS, WEATHER_PRESETS, generate_frame,
    )

    rng = np.random.default_rng(3)
    for _ in range(200):
        f = generate_frame(rng)
        assert f.scenario in SCENARIOS
        assert f.town in TOWNS
        assert f.weather in WEATHER_PRESETS
        assert 0 <= f.command < len(COMMANDS)
        assert f.buckets and all(b in SIMLINGO_TRAIN_PARTITIONS for b in f.buckets)
        assert f.weight > 0


def test_simlingo_partition_weights_sum_to_one():
    from sub1b_vla.data.carla_buckets import SIMLINGO_TRAIN_PARTITIONS

    assert sum(SIMLINGO_TRAIN_PARTITIONS.values()) == pytest.approx(1.0)


def test_bucket_classifier_separates_hard_braking_from_cruising():
    from sub1b_vla.data.carla_buckets import classify, frame_weight, is_long_tail

    n, dt = 11, 0.2
    v, x, pts = 12.0, 0.0, []
    for _ in range(n):
        v = max(0.0, v - 6.0 * dt)
        x += v * dt
        pts.append((x, 0.0))
    hard = classify(np.array(pts), 12.0, dt, leading_object="walker", vehicle_front=True)
    cruise = classify(np.stack([np.cumsum(np.full(n, 6 * dt)), np.zeros(n)], 1), 6.0, dt)

    assert "acceleration_negative_5" in hard and "leading_object_walker" in hard
    assert is_long_tail(hard) and not is_long_tail(cruise)
    assert frame_weight(hard) > frame_weight(cruise)


def test_intent_logits_do_not_depend_on_target_text(model, cfg):
    """Regression guard: the intent head must read only the visual prefix.

    If it pooled the full sequence it could read the intent straight out of the
    supervised rationale during training and then face a prefix-only sequence at
    inference -- learning nothing that transfers.
    """
    b, s = 3, cfg["model"]["image_size"]
    torch.manual_seed(0)
    enc = model.encoder(torch.randn(b, 3, s, s))
    ids = torch.randint(3, 100, (b, 16))
    mask = torch.ones(b, 16, dtype=torch.long)

    with_text = model.language(enc.semantic_tokens, text_ids=ids, text_mask=mask)
    without = model.language(enc.semantic_tokens)
    assert torch.allclose(with_text.intent_logits, without.intent_logits, atol=1e-5), \
        "intent logits leaked information from the target text"


def test_intent_supervision_is_included_in_lm_term(model, cfg):
    """The intent CE must actually reach the loss, otherwise the head is
    grounded only by the consistency term -- which a constant prediction
    satisfies trivially."""
    b, s = 4, cfg["model"]["image_size"]
    base = {
        "image": torch.randn(b, 3, s, s),
        "waypoints": torch.randn(b, cfg["model"]["pred_len"], 2),
        "speed": torch.rand(b) * 30,
        "target_point": torch.randn(b, 2),
        "command": torch.randint(0, 7, (b,)),
        "has_waypoints": torch.ones(b),
    }
    torch.manual_seed(0)
    with_intent, _ = model({**base, "intent_id": torch.zeros(b, dtype=torch.long)}, step=0)
    torch.manual_seed(0)
    without, _ = model(base, step=0)
    assert with_intent.lm.item() != pytest.approx(without.lm.item()), \
        "intent supervision did not reach the LM term"


def test_generate_returns_requested_length(model, cfg):
    b, s = 2, cfg["model"]["image_size"]
    enc = model.encoder(torch.randn(b, 3, s, s))
    ids = model.language.generate(enc.semantic_tokens, max_new_tokens=7)
    assert ids.shape == (b, 7)


def test_explain_returns_one_string_per_sample(model, cfg):
    b, s = 2, cfg["model"]["image_size"]
    enc = model.encoder(torch.randn(b, 3, s, s))
    texts = model.explain(enc, max_new_tokens=5)
    assert len(texts) == b and all(isinstance(t, str) for t in texts)


def test_split_batch_preserves_all_samples():
    from sub1b_vla.train.train import _split_batch

    batch = {"a": torch.arange(10), "b": torch.arange(20).view(10, 2)}
    for parts in (1, 2, 3, 4, 10):
        chunks = _split_batch(batch, parts)
        assert sum(c["a"].shape[0] for c in chunks) == 10
        assert torch.equal(torch.cat([c["a"] for c in chunks]), batch["a"])


# ----------------------------------------------------------------- bench
def test_leaderboard_parser_computes_per_km_rates():
    from sub1b_vla.bench.metrics import parse_leaderboard_results

    m = parse_leaderboard_results("sub1b_vla/tests/fixtures/leaderboard_results_sample.json")
    assert m.num_routes == 2
    assert m.driving_score == pytest.approx((62.5 + 80.0) / 2)
    assert m.route_completion == pytest.approx(95.0)
    assert m.driven_km == pytest.approx(5.0)
    # 2 collisions (1 vehicle + 1 layout) over 5 km
    assert m.collisions_per_km == pytest.approx(0.4)
    assert m.red_light_per_km == pytest.approx(0.0)
    assert m.sidewalk_per_km == pytest.approx(0.2)


def test_missing_metrics_render_as_double_dash_not_zero():
    from sub1b_vla.bench.metrics import BenchmarkMetrics
    from sub1b_vla.bench.report import render_table

    table = render_table(BenchmarkMetrics(), BenchmarkMetrics())
    assert "--" in table
    assert "0.00" not in table, "an unmeasured metric must never render as a number"


def test_delta_direction_respects_lower_is_better():
    from sub1b_vla.bench.metrics import BenchmarkMetrics
    from sub1b_vla.bench.report import render_table

    base = BenchmarkMetrics(driving_score=50.0, collisions_per_km=1.0)
    ours = BenchmarkMetrics(driving_score=60.0, collisions_per_km=0.5)
    table = render_table(base, ours)
    ds_row = next(l for l in table.splitlines() if "Driving Score" in l)
    col_row = next(l for l in table.splitlines() if "Collision Rate" in l)
    assert "better" in ds_row, "a higher driving score is an improvement"
    assert "better" in col_row, "a lower collision rate is an improvement"


def test_hard_constraint_report_distinguishes_zero_from_unmeasured():
    from sub1b_vla.bench.metrics import BenchmarkMetrics
    from sub1b_vla.bench.run_benchmark import hard_constraint_report

    satisfied = hard_constraint_report(BenchmarkMetrics(red_light_per_km=0.0,
                                                       sidewalk_per_km=0.0))
    assert all("SATISFIED" in line for line in satisfied)
    unknown = hard_constraint_report(BenchmarkMetrics())
    assert all("NOT MEASURED" in line for line in unknown)
    violated = hard_constraint_report(BenchmarkMetrics(red_light_per_km=0.3,
                                                      sidewalk_per_km=0.0))
    assert any("VIOLATED" in line for line in violated)


# ------------------------------------------------------ waypoint normalisation
def test_normalize_denormalize_roundtrip():
    head = CoCDiffusionHead(spatial_dim=8, sem_dim=8, dim=16, depth=1, heads=2,
                            pred_len=11, train_steps=100, infer_steps=10,
                            wp_offset=(20.0, 0.0), wp_scale=(20.0, 15.0))
    wp = torch.randn(3, 11, 2) * torch.tensor([10.0, 4.0])
    assert torch.allclose(head.denormalize(head.normalize(wp)), wp, atol=1e-4)


def test_configured_scale_maps_realistic_waypoints_inside_clip_range():
    """The sampler clips its x0 estimate at +-clip_denoised. If the configured
    scale does not map real waypoints inside that range, valid trajectories are
    silently truncated."""
    head = CoCDiffusionHead(spatial_dim=8, sem_dim=8, dim=16, depth=1, heads=2,
                            pred_len=11, train_steps=100, infer_steps=10,
                            wp_offset=(20.0, 0.0), wp_scale=(20.0, 15.0),
                            clip_denoised=1.0)
    # A 2.2 s horizon: up to ~40 m forward (65 km/h) and +-14 m lateral.
    extreme = torch.tensor([[[0.0, 0.0], [40.0, 14.0], [40.0, -14.0]]])
    n = head.normalize(extreme)
    assert bool((n.abs() <= head.clip_denoised).all()), \
        f"configured scale clips real waypoints: max |normalised| = {n.abs().max():.3f}"


def test_sampled_waypoints_stay_within_physical_bounds():
    """Regression guard for the sampler divergence that produced 164 m spreads:
    on an untrained head, clipping must still bound the output."""
    head = CoCDiffusionHead(spatial_dim=8, sem_dim=8, dim=16, depth=1, heads=2,
                            pred_len=11, train_steps=100, infer_steps=10,
                            wp_offset=(20.0, 0.0), wp_scale=(20.0, 15.0),
                            clip_denoised=1.0).eval()
    wp = head.sample(torch.randn(4, 6, 8), torch.randn(4, 3, 8),
                     torch.rand(4) * 20, torch.randn(4, 2), torch.randint(0, 7, (4,)))
    assert bool((wp[..., 0].abs() <= 41.0).all()), "forward waypoints escaped the data range"
    assert bool((wp[..., 1].abs() <= 16.0).all()), "lateral waypoints escaped the data range"


def test_diffusion_recovers_a_trained_trajectory():
    """End-to-end sanity: overfit one conditioning pair and check the sampler
    reproduces its target. Without waypoint normalisation this fails even at a
    near-zero training loss -- which is exactly how the bug hid."""
    torch.manual_seed(0)
    head = CoCDiffusionHead(spatial_dim=8, sem_dim=8, dim=64, depth=2, heads=4,
                            pred_len=11, train_steps=100, infer_steps=10,
                            wp_offset=(20.0, 0.0), wp_scale=(20.0, 15.0),
                            prediction_type="v")
    geo, sem = torch.randn(1, 6, 8), torch.randn(1, 3, 8)
    speed, tp, cmd = torch.tensor([20.0]), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)
    target = torch.tensor([[[i * 1.5, 0.0] for i in range(1, 12)]], dtype=torch.float32)

    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(2000):
        opt.zero_grad()
        per, _, _, _ = head.loss(target, geo, sem, speed, tp, cmd)
        per.mean().backward()
        opt.step()

    head.eval()
    got = head.sample(geo, sem, speed, tp, cmd)   # the configured 10-step budget
    err = (got - target).abs().mean().item()
    assert err < 3.0, f"sampler did not recover the trained trajectory (mean abs err {err:.2f} m)"


def test_v_and_epsilon_parameterisations_are_both_valid():
    """Both heads must round-trip their own target exactly; the choice between
    them is about conditioning at few steps, not correctness."""
    for ptype in ("v", "epsilon"):
        head = CoCDiffusionHead(spatial_dim=8, sem_dim=8, dim=16, depth=1, heads=2,
                                pred_len=11, train_steps=100, infer_steps=10,
                                prediction_type=ptype)
        x0 = torch.randn(4, 11, 2) * 0.5
        noise = torch.randn_like(x0)
        t = torch.randint(0, 90, (4,))
        xt = head.schedule.add_noise(x0, noise, t)
        goal = head.schedule.velocity(x0, noise, t) if ptype == "v" else noise
        rec_x0, rec_eps = head._resolve(xt, goal, t)
        assert torch.allclose(rec_x0, x0, atol=1e-3), f"{ptype}: x0 round-trip failed"
        assert torch.allclose(rec_eps, noise, atol=1e-3), f"{ptype}: eps round-trip failed"


def test_invalid_prediction_type_is_rejected():
    with pytest.raises(ValueError, match="prediction_type"):
        CoCDiffusionHead(spatial_dim=8, sem_dim=8, dim=16, depth=1, heads=2,
                         prediction_type="x0")


# --------------------------------------------------- shift augmentation
def test_shift_augmentation_rotates_waypoints_with_the_image():
    """A pixel shift without the matching waypoint rotation would teach the
    model to steer into the shift. Rotation is isometric, so ranges survive."""
    from sub1b_vla.data.augment import ShiftAugmentation

    aug = ShiftAugmentation(enabled=True, prob=1.0, max_yaw_deg=6.0, seed=3)
    img = np.random.rand(120, 400, 3).astype(np.float32)
    wp = np.stack([np.linspace(2, 22, 11), np.zeros(11)], 1).astype(np.float32)
    out_img, out_wp = aug(img, wp, focal_px=200.0)

    assert not np.array_equal(img, out_img), "image was not shifted"
    assert abs(out_wp[-1, 1]) > 0.1, "waypoints were not rotated to match"
    assert np.allclose(np.linalg.norm(out_wp, axis=1), np.linalg.norm(wp, axis=1), atol=1e-3), \
        "rotation must preserve range"


def test_shift_augmentation_is_a_noop_when_disabled_or_unlucky():
    from sub1b_vla.data.augment import ShiftAugmentation

    img = np.random.rand(40, 80, 3).astype(np.float32)
    wp = np.ones((11, 2), np.float32)
    for aug in (ShiftAugmentation(enabled=False, prob=1.0),
                ShiftAugmentation(enabled=True, prob=0.0)):
        i2, w2 = aug(img, wp, 100.0)
        assert np.array_equal(img, i2) and np.array_equal(wp, w2)


def test_val_split_never_shifts(cfg):
    from sub1b_vla.data.dataset import DrivingVLADataset

    assert not DrivingVLADataset(cfg, "val").shift_aug.enabled


# ------------------------------------------------------------ schedule
def test_schedule_is_epoch_driven_like_simlingo():
    from sub1b_vla.train.train import resolve_schedule

    total, warmup = resolve_schedule(
        {"train": {"max_epochs": 15, "max_steps": 0, "pct_start": 0.05, "warmup_steps": 0}}, 1000)
    assert total == 15000
    assert warmup == 750, "warmup must follow pct_start"


def test_max_steps_caps_an_epoch_schedule():
    from sub1b_vla.train.train import resolve_schedule

    total, _ = resolve_schedule(
        {"train": {"max_epochs": 15, "max_steps": 400, "pct_start": 0.05, "warmup_steps": 0}}, 100)
    assert total == 400


def test_schedule_rejects_an_empty_budget():
    from sub1b_vla.train.train import resolve_schedule

    with pytest.raises(ValueError, match="max_epochs or train.max_steps"):
        resolve_schedule({"train": {"max_epochs": 0, "max_steps": 0}}, 100)


# ------------------------------------------------- instruction following
def test_unsafe_instruction_is_refused_and_trajectory_stays_safe():
    """Language must not override a hazard. An 'accelerate' instruction at a red
    light is refused, and the supervised trajectory remains the stop."""
    from sub1b_vla.data.instructions import sample_instruction

    rng = np.random.default_rng(0)
    refused = 0
    for _ in range(300):
        s = sample_instruction(rng, "OppositeVehicleRunningRedLight", "stop")
        assert s.executed_intent == "stop", "red light must always execute the safe intent"
        if s.refused:
            refused += 1
            assert s.requested_intent in ("accelerate", "keep_speed")
            assert "instruction_refused" in s.rationale()
    assert refused > 0, "refusal must be sampled, not merely possible"


def test_compliant_instruction_is_executed():
    from sub1b_vla.data.instructions import sample_instruction

    rng = np.random.default_rng(4)
    seen = 0
    for _ in range(200):
        s = sample_instruction(rng, "noScenarios", "keep_speed")
        assert not s.refused, "noScenarios has no hazard to refuse for"
        assert s.executed_intent == s.requested_intent
        seen += 1
    assert seen == 200


def test_instruction_rationale_parses_to_the_executed_intent():
    """The rationale must decode to what was DONE, not what was asked -- the
    consistency loss grades the executed trajectory."""
    from sub1b_vla.data.instructions import sample_instruction

    rng = np.random.default_rng(11)
    for _ in range(200):
        s = sample_instruction(rng, "DynamicObjectCrossing", "stop")
        name, _ = parse_intent(s.rationale())
        assert name == s.executed_intent


def test_train_partitions_normalise_and_are_drawn():
    from collections import Counter

    from sub1b_vla.data.dataset import DrivingVLADataset

    cfg = load_config(CFG)
    cfg["data"]["train_partitions"] = {"driving": 2.0, "drivecot": 1.0, "dreamer": 1.0}
    ds = DrivingVLADataset(cfg, "train")
    assert sum(ds.partitions.values()) == pytest.approx(1.0)
    assert ds.partitions["driving"] == pytest.approx(0.5)
    counts = Counter(ds[i]["sample_type"] for i in range(400))
    assert set(counts) == {"driving", "drivecot", "dreamer"}
    assert counts["driving"] > counts["dreamer"]


def test_empty_partitions_are_rejected():
    from sub1b_vla.data.dataset import DrivingVLADataset

    cfg = load_config(CFG)
    cfg["data"]["train_partitions"] = {"driving": 0.0}
    with pytest.raises(ValueError, match="train_partitions"):
        DrivingVLADataset(cfg, "train")


def test_dreamer_samples_supervise_the_diffusion_head_but_qa_does_not():
    """Dreamer trajectories are real targets (the instruction's implied path);
    QA/commentary waypoints are filler and must be masked out."""
    from sub1b_vla.data.dataset import DrivingVLADataset

    cfg = load_config(CFG)
    ds = DrivingVLADataset(cfg, "train")
    seen = {"driving": 0, "drivecot": 0, "dreamer": 0}
    for i in range(300):
        item = ds[i]
        st = item["sample_type"]
        seen[st] += 1
        expected = 1.0 if st in ("driving", "dreamer") else 0.0
        assert float(item["has_waypoints"]) == expected, st
    assert all(v > 0 for v in seen.values()), seen


def test_route_command_never_overrides_a_hazard_intent():
    """A CHANGELANE route command must not turn a red-light stop into a lane
    change. Route intent is subordinate to safety, in the data as in the model."""
    from sub1b_vla.data.carla_surrogate import SCENARIO_SEMANTICS, generate_frame

    hazard_scenarios = {k for k, v in SCENARIO_SEMANTICS.items() if v[3] is not None}
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(600):
        f = generate_frame(rng)
        if f.scenario in hazard_scenarios:
            checked += 1
            assert f.intent == SCENARIO_SEMANTICS[f.scenario][0], (
                f"{f.scenario} intent became {f.intent}")
    assert checked > 0


# ------------------------------------------- real SimLingo/CARLA layout prep
def _write_simlingo_fixture(root):
    """A tiny dataset in SimLingo's exact on-disk layout."""
    import gzip

    route = root / "data" / "Town12_Rep0_route17"
    for sub in ("measurements", "rgb", "rgb_augmented"):
        (route / sub).mkdir(parents=True, exist_ok=True)
    comm = root / "commentary" / "Town12_Rep0_route17" / "commentary"
    comm.mkdir(parents=True, exist_ok=True)

    x, v = 0.0, 9.0
    for i in range(40):
        red = 12 <= i < 26
        v = max(0.0, v - 3.0 * 0.2) if red else min(11.0, v + 1.0 * 0.2)
        x += v * 0.2
        m = {"x": float(x), "y": 0.0, "theta": 0.0, "speed": float(v), "command": 4,
             "target_point": [float(x + 12.0), 0.0], "light_hazard": bool(red),
             "augmentation_translation": 0.0, "augmentation_rotation": 0.0}
        with gzip.open(route / "measurements" / f"{i:04d}.json.gz", "wt") as f:
            json.dump(m, f)
        for d in ("rgb", "rgb_augmented"):
            (route / d / f"{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xdb")
        with gzip.open(comm / f"{i:04d}.json.gz", "wt") as f:
            json.dump({"commentary": "I brake because the light ahead is red." if red
                       else "The road ahead is clear."}, f)


def test_prepare_reads_the_real_simlingo_layout(tmp_path):
    """The primary data path: gzipped measurements, sibling commentary, and
    CARLA behaviour buckets derived from the expert's own trajectory."""
    import subprocess
    import sys

    _write_simlingo_fixture(tmp_path)
    r = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "sub1b_vla.data.prepare_carla_data",
         "--root", str(tmp_path), "--out", str(tmp_path), "--val-frac", "0.1"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr[-2000:]

    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert train, "no training records produced"
    types = {r["sample_type"] for r in train}
    assert "driving" in types and "drivecot" in types, types

    driving = [r for r in train if r["sample_type"] == "driving"]
    # Every driving record carries the CARLA bucket fields the sampler needs.
    for r in driving:
        assert r["buckets"] and r["weight"] > 0
        assert r["town"] == "Town12"
        assert r["image"].startswith("data/")
    # The red-light frames must be bucketed as yielding to a traffic light.
    assert any("leading_object_traffic.traffic_light" in r["buckets"] for r in driving)
    # SimLingo's own commentary text is used as the rationale.
    assert any("red" in r["causation"] for r in driving)


def test_recorded_augmentation_correction_is_exact():
    """rgb_augmented frames are real re-renders, so the recorded offsets give an
    exact correction -- rotation AND translation, unlike the 2-D shift path."""
    from sub1b_vla.data.prepare_carla_data import apply_recorded_augmentation

    wps = np.stack([np.linspace(2, 22, 11), np.zeros(11)], 1).astype(np.float32)
    same = apply_recorded_augmentation(wps, 0.0, 0.0)
    assert np.array_equal(same, wps), "a null augmentation must be a no-op"

    shifted = apply_recorded_augmentation(wps, 1.5, 0.0)
    assert np.allclose(shifted[:, 1], -1.5), "a +1.5 m camera shift moves targets -1.5 m"

    rotated = apply_recorded_augmentation(wps, 0.0, 10.0)
    assert np.allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(wps, axis=1), atol=1e-3)
    assert rotated[-1, 1] < -0.5, "a +10 deg camera yaw moves targets to the right"
