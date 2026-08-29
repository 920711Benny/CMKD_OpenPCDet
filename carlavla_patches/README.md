# CarlaVLA patches

Drop-in replacements for three files in your `simlingo_training` tree. Class
names, attribute names, method signatures and **state-dict layouts are
unchanged**, so existing checkpoints load without renaming a key.

```
carlavla_patches/dual_vision_model.py  ->  simlingo_training/models/encoder/dual_vision_model.py
carlavla_patches/diffusion_decoder.py  ->  simlingo_training/models/diffusion_decoder.py
carlavla_patches/flash_mha.py          ->  new file, imported by dual_vision_model
```

`flash_mha.py` imports nothing from `simlingo_training`, so it can live beside
`dual_vision_model.py`. The relative import at the top of that file is
`from .flash_mha import FlashMultiheadAttention` — adjust if you place it
elsewhere.

16 tests cover these patches: `python -m pytest carlavla_patches/tests -q`.

---

## The measurement that drove this

Controlled overfit on your `ConvDenoiser1D` at your configured
`diffusion_infer_steps: 10`, same seed and step count throughout. Mean absolute
waypoint error:

| setting | DDIM-10 | DDIM-100 |
|---|---:|---:|
| **original** — epsilon + `coord_min_max=(-32,32)` | 25.418 m | 22.992 m |
| epsilon + fitted per-dim range | 4.964 m | 4.142 m |
| v-prediction + `(-32,32)` | 21.152 m | 19.457 m |
| **patched** — v-prediction + fitted range | **3.412 m** | 1.253 m |

**The coordinate range is the dominant factor, not the parameterisation.** It is
worth 5.1× on its own; v-prediction adds a further 1.5×. I previously told you
v-prediction was the bigger lever — that was measured on my own cross-attention
head, and it does not transfer to your Conv1d denoiser. The ranking above is the
one that applies to your code.

Absolute values are from an overfit on random conditioning and are not driving
accuracy; only the relative ordering is meaningful.

---

## Fix 1 — `coord_min_max` must come from the data

Your documented real range is x:[2.5, 19.3], y:[-0.01, 4.8]. Under `(-32, 32)`:

| axis | fraction of [-1,1] used | centre offset |
|---|---:|---:|
| forward | 26.2 % | +0.34 |
| lateral | **7.5 %** | +0.07 |

Diffusion assumes roughly zero-centred, unit-scale data. The lateral axis was
using **under a tenth** of the schedule's dynamic range, and neither axis was
centred. After fitting per-dimension: both axes use 87 %, centred at 0.

The range is now **per-dimension**, which matters — a symmetric `(-a, a)` cannot
centre an axis that is strictly positive. (My first attempt at this fix used a
symmetric range and barely helped; the test
`test_fit_coord_range_is_per_dimension_not_symmetric` pins it.)

```python
from carlavla_patches.diffusion_decoder import fit_coord_range, coord_range_report

coord_range_report(waypoints, (-32.0, 32.0))   # what your current setting wastes
fit_coord_range(waypoints, margin=0.15)        # -> [[lo_x, hi_x], [lo_y, hi_y]]
```

Compute it once over a sample of your training waypoints and pass it to
`DiffusionHead(coord_min_max=...)`. A scalar `(lo, hi)` is still accepted and
expanded to both axes, so nothing breaks if you do not.

`coord_min_max` is now a **persistent** buffer: normalising at inference with a
different range than training silently rescales every trajectory, so it must
travel with the checkpoint.

## Fix 2 — v-prediction

epsilon recovers `x0` by dividing by `sqrt(alpha_bar)`, which goes to zero at
high `t` — so prediction error is amplified without bound exactly where a
10-step schedule spends most of its steps. v-prediction is well conditioned
across the whole schedule.

`prediction_type="epsilon"` is still available. The denoiser weights are
unchanged in shape, only their target differs, **so switching requires
retraining** — an existing checkpoint will not transfer.

## Fix 3 — cross-attention that reaches FlashAttention-2

`CrossAttentionFusion` called `self.cross_attn(q, kv, kv)` on an
`nn.MultiheadAttention`. `need_weights` **defaults to `True`**, so every forward
pass materialised the full attention matrix and discarded it — which disables
even nn.MultiheadAttention's own fused kernel, let alone FlashAttention-2.

`FlashMultiheadAttention` keeps the identical parameter layout
(`in_proj_weight` (3E,E), `in_proj_bias` (3E,), `out_proj.*`) so checkpoints load
unchanged, defaults `need_weights=False`, and dispatches to `flash_attn_func` or
torch's pinned FLASH_ATTENTION backend. Verified bit-identical to
`nn.MultiheadAttention` (max diff 0.0).

## Fix 4 — two latent bugs in `dual_vision_model.py`

**`forward` wrapped both backbones in an unconditional `torch.no_grad()`.**
Correct while `freeze=True`, but it means `freeze=False` silently trains
nothing: the constructor clears `requires_grad`, and the `no_grad` block blocks
the graph regardless. Now conditional on `freeze`.

**`LingoDualVisionModel.__init__` never forwarded `freeze`** to
`DualVisionEncoder`, so the parameter was dead and the default always applied.
Now plumbed through.

Neither changes behaviour under your current config (`freeze: true`) — they
matter the moment you try to unfreeze.
