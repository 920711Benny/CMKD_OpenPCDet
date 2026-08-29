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

## 0. Training cost — read this before any number in this repo

**The runs executed inside the build container are smoke tests, not training.**
They took ~20–80 minutes because they are a 7.1M-parameter *stub* model on
procedurally generated toy data, on 4 CPU cores. They prove the code runs. They
prove nothing about driving.

Real training is expensive, and SimLingo's own recipe says so. Quoted from
`RenzKa/simlingo@main` (`train_simlingo_seed1.sh`, `simlingo_seed1.yaml`), and
mirrored in `baselines/simlingo_training_recipe.json`:

| | SimLingo | this repo, `default.yaml` | this repo, `single_gpu_reduced.yaml` |
|---|---|---|---|
| GPUs | **8** | 1 | 1 |
| wall clock | **3-day SLURM limit** (≤576 GPU-h) | ~11 days projected | ~10 h projected |
| epochs | 15 | 15 | 4 |
| effective batch | 48 (6 × 8 GPUs) | 48 (6 × accum 8) | 48 (12 × accum 4) |
| lr / weight decay | 3e-5 / 0.1 | 3e-5 / 0.1 | 1e-4 / 0.1 |
| strategy | deepspeed_stage_2 | single device | single device |
| dataset | full HF release (driving + VQA + commentary + dreamer) | same | subsampled, `data.max_records` |

The projections are not guesses — `tools/compute_budget.py` measures a real
forward+backward at your configured batch size and extrapolates. It refuses to
invent a throughput for hardware it has not seen; supply one with
`--samples-per-s` if you measured it elsewhere.

```bash
python -m sub1b_vla.tools.compute_budget --config sub1b_vla/configs/default.yaml \
    --dataset-frames <frames in your extracted copy>
```

The 3-day figure is SimLingo's SLURM *limit*, not a measured runtime, so ≤576
GPU-hours is an upper bound. Either way: a single-GPU reproduction of the full
recipe is a multi-day job, and `single_gpu_reduced.yaml` exists so you can get a
first real checkpoint overnight — with every deviation from the matched recipe
listed in its header, because results under it are **not** comparable to a
full-recipe run.

### What is matched to SimLingo, and what deliberately differs

Matched, so the comparison is not confounded by tuning: `lr` 3e-5,
`weight_decay` 0.1, `betas` (0.9, 0.999), `pct_start` 0.05, `max_epochs` 15,
effective batch 48, `num_workers` 8, LoRA r=32 / α=64 / dropout 0.1,
`pred_len` 11, `hist_len` 1, `cut_bottom_quarter`, `route_as:
target_point_command`, `use_safety_flag`, `img_augmentation(_prob 0.5)`,
`img_shift_augmentation(_prob 0.5)`, `skip_first_n_frames` 10,
`num_route_points` 20.

Deliberately different — these are the contribution, or are forced by the
hardware target:

| | SimLingo | here | why |
|---|---|---|---|
| vision | InternVL2-1B's InternViT, **trainable** | DINOv2-small + SigLIP-base, **frozen** | the dual-head asymmetric split; frozen keeps trainables at 23M |
| action head | direct waypoint regression | CoC conditional diffusion (v-pred, 10-step DDIM) | multimodal trajectories; the thesis contribution |
| language↔action link | language-action alignment via shared training | explicit differentiable Causal Consistency Loss | makes agreement an optimised quantity, not a hoped-for side effect |
| precision | 16-mixed | bf16-mixed | Blackwell; no loss scaler needed |
| parallelism | deepspeed_stage_2, 8 GPUs | single device + grad accumulation | single-workstation constraint |
| total params | ~0.94B (InternVL2-1B) | **0.632B** | the <1B budget |

`img_shift_augmentation` is an approximation here and is marked as such in the
code: SimLingo re-renders genuinely shifted camera views during data collection,
which we do not have. We reproduce only the *rotational* part — a horizontal
pixel shift of `du` is a camera yaw of `atan(du/f)`, and the waypoints are
rotated to match. Translation is not faked, because a 2-D shift cannot reproduce
parallax and pretending otherwise would teach a geometry that does not exist.

---

## 0b. Data — everything is CARLA

There are exactly two data sources, and both speak CARLA's vocabulary.

**1. The real SimLingo / PDM-Lite CARLA release** (use this for any reported
result). `prepare_carla_data.py` consumes the layout the release actually ships:

```
<root>/data/<route>/measurements/XXXX.json.gz      ego + simulator state
<root>/data/<route>/rgb/XXXX.jpg                   1024x512 front camera
<root>/data/<route>/rgb_augmented/XXXX.jpg         re-rendered shifted view
<root>/commentary/<route>/commentary/XXXX.json.gz  language: commentary
<root>/drivelm/<route>/vqa/XXXX.json.gz            language: VQA
<root>/dreamer/<route>/dreamer/XXXX.json.gz        language: instructions
```

```bash
./run_pipeline.sh prepare database/simlingo      # or --use-augmented
```

When `rgb_augmented` is present it is preferred, because those frames are
genuinely re-rendered and carry the exact `augmentation_translation` /
`augmentation_rotation` the simulator used — so the waypoint correction is exact
in both rotation *and* translation, unlike the rotation-only 2-D approximation.

**2. `carla_surrogate.py`** — procedural frames that use CARLA's own vocabulary
(scenario types from the release's chunk names, Town12/Town13, CARLA weather
presets, RoadOption commands, the same behaviour buckets). It exists so the code
paths, atomic gates and CI can run without a 1 TB download. It is **not** CARLA
output, is labelled as a surrogate everywhere, and must not produce a reported
driving number.

### Balancing: SimLingo's real behaviour buckets

Quoted from `simlingo_training/config/data_module/carla_bucket_v12_dreamer.yaml`
(they sum to exactly 1.0) and implemented in `carla_buckets.py`:

| bucket | weight | | bucket | weight |
|---|---:|---|---|---:|
| `all` | 0.082 | | `start_from_stop` | 0.07 |
| `acceleration_negative_5` | 0.03 | | `vehicle_front` | 0.04 |
| `acceleration_negative_1` | 0.03 | | `vehicle_side` | 0.08 |
| `acceleration_positive_1` | 0.03 | | `leading_object_vehicle` | 0.09 |
| `acceleration_positive_5` | 0.03 | | `leading_object_traffic.stop` | 0.07 |
| `lateral_control_1_2` | 0.12 | | `leading_object_traffic.traffic_light` | 0.07 |
| `lateral_control_higher_5` | 0.12 | | `leading_object_walker` | 0.05 |
| `changed_route` | 0.08 | | `parkinglane` | 0.008 |

Balancing on **behaviour** rather than scenario name is the point: a route
labelled `SignalizedJunctionLeftTurn` spends most of its frames driving straight,
so scenario-level balancing does not rebalance behaviour. A frame lands in
several buckets at once (a hard brake behind a walker is in
`acceleration_negative_5`, `leading_object_walker` and `all`) and its sampling
weight is the **sum** over its buckets, so a frame that is rare on two axes
outranks one that is rare on one.

### The 60% CoT mix

```yaml
data:
  train_partitions: {driving: 0.25, drivecot: 0.60, dreamer: 0.15}
```

`drivecot` covers SimLingo's commentary + DriveLM VQA; `dreamer` is instruction
following. One consequence worth stating plainly: **only `driving` and `dreamer`
carry trajectory targets**, so a 60% CoT mix means 40% of each batch trains the
diffusion head. To keep trajectory supervision long-tail-weighted despite the
smaller share, the 2:8 easy/hard split is applied *within* each sample type.

Applying it globally was a bug, now fixed: language records have no trajectory
and are never long-tail, so they absorbed the entire "easy" 20% and drove the
driving pool to ~98% long-tail. Measured after the fix, on a prepared manifest:

| target | measured |
|---|---|
| CoT share 60% | **60.0%** |
| long-tail within driving 80% | **80.1%** |

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

### Diffusion parameterisation and waypoint normalisation

Two choices here are load-bearing, and both were forced by measurement rather
than taste.

**Targets are normalised into [-1, 1]** by `(wp - offset) / scale`, with the
constants stored as buffers so they travel with the checkpoint. Feeding raw
metres (σ ≈ 5 m forward, values to 31 m) to a schedule that assumes unit-scale
data makes the signal dominate the noise at nearly every timestep: ε-prediction
collapses toward returning its input, the training loss goes low, and sampling
from pure noise diverges. This is what the atomic gates caught — training loss
0.085 alongside a **164 m** spread across noise draws. Min-max rather than
std normalisation is deliberate: it lets the sampler clip `x0` at ±1 without ever
truncating a valid trajectory, which std scaling cannot promise when a 30 m
waypoint is 6σ. Derive the constants for your data with
`python -m sub1b_vla.tools.waypoint_stats --config ...`.

**v-parameterisation** (Salimans & Ho) instead of ε-prediction. ε is poorly
conditioned at low SNR — exactly where a 10-step schedule spends most of its
steps. Measured on a controlled overfit, mean absolute waypoint error:

| parameterisation | DDIM-10 (configured) | DDIM-25 | DDIM-100 |
|---|---:|---:|---:|
| epsilon | 2.607 m | 1.723 m | 0.400 m |
| **v** | **0.690 m** | 0.549 m | 0.716 m |

v at the 10-step budget beats ε at 100 steps. Since ≥10 Hz control is a hard
requirement, this is what makes the short schedule usable. Switch with
`model.prediction_type: epsilon` to reproduce the comparison.

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

**Training is a GPU workflow.** There is no supported CPU training path: on a
single RTX PRO 6000 the reduced recipe is ~14 h, and the same work on CPU is
orders of magnitude slower. `gpu_preflight` refuses to report READY without a
CUDA device.

```bash
pip install -r requirements.txt

# 0. gates that need no data
./run_pipeline.sh budget
./run_pipeline.sh test

# 0b. GPU check BEFORE committing hours: verifies CUDA + bf16, finds the largest
#     batch that fits, measures real throughput, projects wall clock, and exits
#     non-zero on anything that would break or badly slow a long run.
./run_pipeline.sh preflight 3000000

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

### FlashAttention-2 on RTX 6000 Pro / Ubuntu 24.04

Two separate things are called "flash attention", and they fail differently:

| | what it is | how this repo uses it |
|---|---|---|
| **flash-attn package** | standalone CUDA kernels | `attn_implementation: flash_attention_2` on the HuggingFace backbones |
| **PyTorch SDPA flash backend** | built into torch | every attention module in this repo, via `F.scaled_dot_product_attention` |

**`nn.MultiheadAttention` never reaches either.** Its fused path is a separate
native kernel with its own conditions and it falls back to unfused math whenever
they are not met — `need_weights=True` being the usual culprit. A model built
from `nn.MultiheadAttention` gets no benefit from an installed flash-attn, no
matter how carefully it was compiled. Every attention here therefore goes
through `models/attention.py::SDPAAttention`, which dispatches via SDPA and has
**identical parameter count** to `nn.MultiheadAttention` (a test pins this, so
the <1B budget cannot move).

Attention weights are still needed for the HUD's attention maps, so that path
exists — but explicitly and separately, because materialising the weight matrix
is exactly what disables the flash kernel. Training uses the fused path.

Install (Blackwell is sm_120 and needs a CUDA 12.8+ toolchain):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
MAX_JOBS=4 pip install flash-attn --no-build-isolation   # ~30-60 min from source
./run_pipeline.sh preflight                              # verify it is ACTUALLY used
```

**Verify rather than assume.** flash-attn wheels have historically targeted
sm80–sm90; one built for the wrong architecture imports cleanly and is then
never used. `gpu_preflight` probes each SDPA backend *by running it* and reports
what executed, warns when the flash backend is unusable, and prints which
implementation each backbone actually loaded — the loader walks
`flash_attention_2 → sdpa → eager` and records the winner, so a mismatch
degrades visibly instead of silently costing throughput.

FlashAttention needs bf16/fp16 inputs; fp32 falls back to the math backend
silently. The `bf16-mixed` default satisfies this.

### GPU settings

| setting | default | why |
|---|---|---|
| `precision` | `bf16-mixed` | Blackwell; no loss scaler needed. Preflight blocks if it silently falls back to fp32. |
| `allow_tf32` / `cudnn_benchmark` | on | input shape is fixed for a whole run, so autotuning pays for itself immediately |
| `batch_size` × `accumulate_grad_batches` | 24 × 2 = **48** | SimLingo uses 6 per GPU because it *trains* its vision encoder; both of ours are frozen, so no backbone activations are kept for the backward pass and a much larger micro-batch fits. The effective batch stays at SimLingo's 48. |
| `compile` | `null` | `torch.compile` costs minutes of warmup and a failure part-way into a multi-hour run is expensive. Turn it on once a short run has proven the model trains. |
| `gradient_checkpointing` | `false` | only needed if preflight reports the configured batch does not fit |

Run `./run_pipeline.sh preflight <frames>` to find the real maximum batch on your
card and raise `batch_size` accordingly — it measures rather than guesses.

### Atomic verification gates

Run before any CARLA time is spent; each isolates one primitive skill.

| gate | passes when |
|---|---|
| steering distribution | mean steer separates left/right turn queries by ≥0.25 **and** ≥75% correct polarity (CARLA sign: negative = left) |
| red-light stop adherence | red-light target speed ≤0.5 m/s, brake rate ≥85%, and green-light control speed strictly exceeds it |
| 10-step diffusion latency & stability | end-to-end p95 ≤ budget (80 ms) and sample spread ≤1.0 m across independent noise draws |

The green-light control in gate 2 is what stops a model that simply always brakes
from passing.

### Instruction following (Dreamer-style)

`data/instructions.py` supplies SimLingo's third supervision mode: a frame paired
with an instruction and the trajectory that instruction implies. It pairs
naturally with the consistency loss — an instruction sample supplies the intent
from *language input* rather than from the scene, so the same violation function
grades whether the instruction was obeyed.

**Safety is not overridable.** ~35% of instructions in a hazard scenario are
drawn from the unsafe set for that scenario ("accelerate" at a red light). Those
must be **refused**, and the supervised trajectory stays the safe one. Refusal is
trained as a positive behaviour, not as the absence of compliance.

Evaluation reports three numbers, because any one alone is gameable:

| metric | what a degenerate policy scores |
|---|---|
| safe-instruction compliance | obey-everything: 1.0 |
| unsafe-instruction refusal | obey-nothing / always-crawl: ~1.0 |
| ...of which **correct safe action** | always-crawl: low — this is the discriminating one |

On the CPU smoke checkpoint (which saw **no** instruction data, so this is a
baseline not a capability): compliance 0.659, refusal 0.944, correct-safe-action
0.648. The 0.30 gap between refusal and correct-safe-action is precisely the
timidity artifact the third metric exists to expose.

Mixing follows SimLingo's `train_partitions`:

```yaml
data:
  train_partitions: {driving: 0.45, drivecot: 0.30, dreamer: 0.25}
```

`drivecot` (QA/commentary) samples supervise language only and are masked out of
the diffusion loss; `dreamer` samples carry a real trajectory target and are not.

### Baselines

`baselines/simlingo.template.json` ships **empty on purpose**. Fill it from the
SimLingo paper's table *for the same suite/routes/weather*, or from your own
reproduction, and record which in `source`. Any metric left null prints as `--`.
The report never defaults, estimates, or carries a value over from another run.

---

## 4. What has and has not been executed

Built and validated in a CPU-only container with **no GPU, no CARLA, and the
HuggingFace hub blocked by the network proxy**. Being precise about which is which:

**Measured here (real numbers, on the CPU smoke model — see §0):**
* Parameter budget — **0.632 B** total, 23.0 M trainable, from architecture-exact
  replicas whose counts were cross-checked against independent analytic formulas
  (the tool raises if they disagree) and which match the published sizes of all
  three backbones.
* **58 unit tests**, covering frame/sign conventions, loss semantics, DDIM
  determinism, waypoint normalisation, shift-augmentation label correctness, the
  epoch/warmup schedule, and the async runtime's non-blocking contract.
* The v-vs-ε comparison table above, from a controlled overfit.
* Atomic gates on the smoke checkpoint — **2 of 3 pass**:

  | gate | result |
  |---|---|
  | steering distribution | **PASS** — separation +0.270 (≥0.25), polarity 1.00 |
  | 10-step diffusion latency & stability | **PASS** — p95 27.9 ms / 40.4 Hz on CPU (budget 80 ms), sample spread 0.262 m |
  | red-light stop adherence | **FAIL** — final speed 1.92 m/s (need ≤0.5); red *is* separated from green (1.92 vs 5.77) but does not reach rest |

  The failing gate is doing its job: a stub-backbone model trained for 80 CPU-minutes
  on toy data has not learned to stop. It is reported as FAIL, not tuned into a pass.
* Action-CoT alignment 0.536 over 192 samples (same caveat — it measures the
  smoke model, and is strongest where the synthetic cue is most explicit:
  left_turn 0.91, green_light 0.83).

**Not executed here, and therefore reported as `--`:**
* Driving Score / Route Completion / Infraction Score / per-km infraction rates —
  these require a CARLA closed-loop run. The harness computes them from a real
  `results.json`; it will not synthesise them.
* Training on real driving data (see §0 for what that actually costs), and
  latency on an RTX PRO 6000. CPU latency is reported as CPU latency, with the
  device recorded in the latency JSON and surfaced in the report's provenance.

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
              compute_budget.py              measured training-time projection
              waypoint_stats.py              diffusion normalisation constants
              demo_hud.py                    offline HUD / async-runtime exercise
  tests/      test_core.py
```
