# Sub-1B Dual-Head Diffusion VLA — a CARLA driving agent under 1B parameters

A Vision-Language-Action driving agent built around three ideas that are not in
SimLingo or in naive "VLM + waypoint head" wrappers:

1. **Dual-Head Asymmetric Visual Encoder** — geometry and semantics are routed to
   different consumers instead of through one shared token stream.
2. **Chain-of-Causation (CoC) diffusion action decoder** — a conditional DDPM over
   waypoints, conditioned on geometry, causal-rationale tokens and ego state as
   three *separate* attention sources.
3. **Causal Consistency Loss** — a differentiable mutual-agreement penalty that
   makes the stated reason and the driven trajectory constrain each other.

Plus a **decoupled asynchronous runtime** so verbose reasoning never blocks
actuation.

---

## 1. Architecture

```
                RGB front camera (1024x512, FOV 100, z=2.3m, pitch=-5deg)
                                     |
                     +---------------+---------------+
                     |                               |
        Spatial-Geometric branch          Semantic-Reasoning branch
            DINOv2-small (frozen)          SigLIP-base/16 (frozen)
            22.06 M params                  92.88 M params
                     |                               |
            dense patch grid                 32 learned queries
            (no pooling: the                 (heavy compression:
             decoder needs metric             the LLM needs what/why,
             detail)                          not per-patch geometry)
                     |                               |
              SpatialProjector              TokenCompressor -> 896-d
                  -> 256-d                          |
                     |                     geometry->semantics cross-attn
                     |                               |
                     |                    Qwen2-0.5B decoder (frozen + LoRA r=16)
                     |                     494.03 M params, 2.16 M adapters
                     |                               |
                     |                    CoC rationale + intent logits
                     |                               |
                     +--------> CoC Diffusion Decoder <-------+
                                (100 train steps / 10-step DDIM)
                                          |
                                  11 waypoints @ 0.2 s
                                          |
                            PID (longitudinal) + Pure Pursuit (lateral)
```

**Why the asymmetry matters.** Pooling the vision stream to a handful of tokens
is standard because it is what an LLM can ingest — and it destroys exactly the
dense spatial detail a trajectory decoder needs. Splitting the streams lets each
consumer get the representation it actually wants, at a cost of one extra frozen
backbone (93 M) rather than a larger LLM.

### Parameter budget (verified, hard-gated)

`assert_parameter_budget()` raises at construction if the total reaches 1 B.

| component | params | trainable |
|---|---:|---:|
| DINOv2-small (frozen) | 22,056,192 | 0 |
| SigLIP-base/16 vision tower (frozen) | 92,884,224 | 0 |
| Qwen2-0.5B decoder (frozen) | 494,032,768 | 0 |
| LoRA adapters (r=16, q/k/v/o × 24 layers) | 2,162,688 | 2,162,688 |
| dual-head projectors + fusion | 10,829,697 | 10,829,697 |
| intent head | 407,240 | 407,240 |
| CoC diffusion decoder | 7,444,794 | 7,444,794 |
| **TOTAL** | **629,817,603 (0.630 B)** | **20,844,419 (20.8 M)** |

The headline decision: **InternVL2-1B's native 0.30 B InternViT tower is dropped**
and replaced by the dual-head encoder; only its Qwen2-0.5B decoder is kept. Keeping
both towers would put the model at ~1.1 B and break the constraint. Verify with:

```bash
./run_pipeline.sh budget          # exits non-zero if over budget
```

### Causal Consistency Loss

For each canonical intent `k` a differentiable violation `V_k(tau) >= 0` is zero
exactly when the trajectory shows the dynamics that intent claims. The loss is the
violation *expected under the language model's own intent distribution*:

```
L_consistency = sum_k p_k(language) * V_k(trajectory)
L_total       = λ_lm·L_LM + λ_diff·L_diffusion + λ_align·L_consistency
```

Gradient flows both ways: through `p_k` the LM is pushed off intents the
trajectory contradicts; through `V_k` the trajectory is pushed to satisfy the
believed intent. Neither is ground truth for the other.

Two implementation details that matter and are easy to get wrong:

* The term consumes the diffusion head's `x0` estimate, which is **meaningless at
  high noise levels** (it divides by `sqrt(alpha_bar) -> 0`). Samples are weighted
  by `alpha_bar(t)` and `x0` is clamped to a physical range. Without this the term
  reached ~4000 and reported 198 km/h final speeds.
* `stop` means **coming to rest within the horizon**, not standing still. Braking
  from 45 km/h covers real ground; penalising that displacement would make every
  legitimate stop-line approach a violation.

### Decoupled asynchronous control

| path | rate | work | blocks the vehicle? |
|---|---|---|---|
| fast | ≥10 Hz target | encode → 10-step DDIM → waypoints → PID/pure-pursuit | yes (it *is* control) |
| slow | ~1 Hz | cached encoding → autoregressive CoC rationale → HUD | **never** |

The rationale worker publishes into a latest-wins slot and yields at a cooperative
priority gate whenever the control loop has work pending. A late rationale shows as
a slightly stale HUD line (with its frame age), not as a stalled vehicle.

---

## 2. Output Separation Protocol

* **Simulator window / pygame HUD** — video, projected diffusion trajectory, both
  attention maps (cyan = geometry, amber = semantics), live CoC text, telemetry.
  Keys: `S` toggles the spatial map, `D` the semantic map.
* **Terminal** — the benchmark table only. `bench.report.quiet_terminal()` silences
  logging, warnings and stray stdout during a run; stderr stays open so a crash is
  still visible.

---

## 3. Running it

```bash
pip install -r requirements.txt

# 0. gates that need no data
./run_pipeline.sh budget
./run_pipeline.sh test

# 1. data: CARLA expert logs -> manifests (CoC labels from privileged state)
./run_pipeline.sh prepare /path/to/carla_dataset

# 2. train (single RTX PRO 6000, bf16-mixed)
./run_pipeline.sh train

# 3. atomic verification BEFORE spending simulator time
./run_pipeline.sh verify runs/dualhead_coc_diffusion_vla/final.pt

# 4. closed-loop benchmark
./run_pipeline.sh launch runs/.../final.pt     # prints leaderboard commands
./run_pipeline.sh align  runs/.../final.pt     # Action-CoT alignment
./run_pipeline.sh report                       # the table
```

Set `SUB1B_SCRATCH` to a high-capacity directory; `HF_HOME`, `TORCH_HOME` and the
dataset caches are redirected under it automatically.

### Atomic verification gates

Run before any CARLA time is spent; each isolates one primitive skill.

| gate | passes when |
|---|---|
| steering distribution | mean steer separates left/right turn queries by ≥0.25 **and** ≥75% correct polarity (CARLA sign: negative = left) |
| red-light stop adherence | red-light target speed ≤0.5 m/s, brake rate ≥85%, and green-light control speed strictly exceeds it |
| 10-step diffusion latency & stability | end-to-end p95 ≤ budget (80 ms) and sample spread ≤1.0 m across independent noise draws |

The green-light control in gate 2 is what stops a model that simply always brakes
from passing.

### Baselines

`baselines/simlingo.template.json` ships **empty on purpose**. Fill it from the
SimLingo paper's table *for the same suite/routes/weather*, or from your own
reproduction, and record which in `source`. Any metric left null prints as `--`.
The report never defaults, estimates, or carries a value over from another run.

---

## 4. What has and has not been executed

Built and validated in a CPU-only container with **no GPU, no CARLA, and the
HuggingFace hub blocked by the network proxy**. Being precise about which is which:

**Measured here (real numbers):**
* Parameter budget — 0.630 B, from architecture-exact replicas whose counts were
  cross-checked against independent analytic formulas (the tool raises if they
  disagree) and which match the published sizes of all three backbones.
* 37/37 unit tests, covering frame/sign conventions, loss semantics, DDIM
  determinism, and the async runtime's non-blocking contract.
* Full training loop convergence on the procedural sanity dataset.
* Camera projection, controller sign conventions, HUD compositing.

**Not executed here, and therefore reported as `--`:**
* Driving Score / Route Completion / Infraction Score / per-km infraction rates —
  these require a CARLA closed-loop run. The harness computes them from a real
  `results.json`; it will not synthesise them.
* Training on real driving data, and latency on an RTX PRO 6000. CPU latency
  numbers from this container are reported as CPU numbers and are not a proxy for
  the GPU budget.

Because the hub is unreachable here, the backbones fall back to randomly
initialised stubs that **announce themselves loudly** (`[STUB BACKBONE]`,
`[STUB LM]`). Any metric produced in that mode measures plumbing, not driving.
`load_agent_model` refuses to run without a checkpoint unless
`SUB1B_ALLOW_UNTRAINED=1` is set explicitly.

## 5. Layout

```
sub1b_vla/
  models/     dual_head_encoder, diffusion_head, coc_language, lora, coc_prompt, vla_agent
  losses/     consistency (the novel term), composite
  data/       dataset, augment, synthetic, text, prepare_carla_data
  train/      train.py                       single-GPU bf16 loop
  verify/     atomic_checks.py               pre-simulation terminal gates
  carla_agent/ sensors, controller, async_pipeline, hud, agent
  bench/      metrics, report, alignment_eval, run_benchmark
  tools/      param_budget.py                the <1B gate
  tests/      test_core.py
```
