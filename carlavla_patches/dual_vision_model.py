"""dual_vision_model.py -- patched.

Drop-in replacement. Class names, attribute names and the state_dict layout are
unchanged, so existing checkpoints load without renaming a single key.

Three fixes:

1. CrossAttentionFusion used nn.MultiheadAttention, called as
   `self.cross_attn(q, kv, kv)`. `need_weights` defaults to **True**, so every
   forward pass materialised a full [N, 256, 256] attention matrix per head and
   then discarded it -- and that alone disables even nn.MultiheadAttention's own
   fused kernel, let alone FlashAttention-2. Replaced with
   FlashMultiheadAttention, which keeps the identical parameter layout
   (in_proj_weight / in_proj_bias / out_proj.*) and dispatches to flash-attn.

2. `forward` wrapped both backbones in an unconditional `with torch.no_grad()`.
   That is correct while freeze=True, but it means freeze=False silently trains
   nothing: the constructor would clear requires_grad, and the no_grad block
   would still block the graph. The context is now conditional on freeze.

3. `LingoDualVisionModel` never forwarded `freeze` to `DualVisionEncoder`, so
   the parameter was dead and the default always applied. It is now plumbed
   through.

The original file's documented reasoning is preserved: DINOv2's processor
centre-crops (losing edge field of view) while SigLIP's does not, so the two
grids do not correspond position-for-position, which is why fusion is
cross-attention rather than a positional concat.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from .flash_mha import FlashMultiheadAttention

HIDDEN_SIZE = 896
FIXED_GRID = 16
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


class CrossAttentionFusion(nn.Module):
    """DINOv2 tokens query the whole SigLIP feature map.

    Each DINO token attends over all of SigLIP rather than assuming its own grid
    cell corresponds to the same real-world region, which it does not: the two
    processors crop differently.
    """

    def __init__(self, dino_dim: int, siglip_dim: int, hidden: int = HIDDEN_SIZE,
                 n_heads: int = 8):
        super().__init__()
        self.dino_proj = nn.Linear(dino_dim, hidden)
        self.siglip_proj = nn.Linear(siglip_dim, hidden)
        # FIX 1: state-dict compatible with nn.MultiheadAttention, but reaches
        # FlashAttention-2 and never materialises the attention matrix.
        self.cross_attn = FlashMultiheadAttention(hidden, n_heads)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, dino_feat: torch.Tensor, siglip_feat: torch.Tensor) -> torch.Tensor:
        q = self.dino_proj(dino_feat)
        kv = self.siglip_proj(siglip_feat)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = self.norm1(q + attn_out)
        return self.norm2(x + self.ffn(x))


class DualVisionEncoder(nn.Module):
    def __init__(self, dinov2_path: str, siglip_path: str, freeze: bool = True,
                 attn_implementation: str = "flash_attention_2"):
        super().__init__()
        self.freeze = freeze
        self.dinov2 = _load_vision(dinov2_path, attn_implementation)
        self.siglip = _load_vision(siglip_path, attn_implementation, vision_only=True)

        dino_proc = AutoImageProcessor.from_pretrained(dinov2_path)
        siglip_proc = AutoImageProcessor.from_pretrained(siglip_path)

        self.register_buffer("dino_mean",
                             torch.tensor(dino_proc.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("dino_std",
                             torch.tensor(dino_proc.image_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("siglip_mean",
                             torch.tensor(siglip_proc.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("siglip_std",
                             torch.tensor(siglip_proc.image_std).view(1, 3, 1, 1), persistent=False)

        self.dino_size = (dino_proc.crop_size["height"]
                          if getattr(dino_proc, "crop_size", None) else
                          dino_proc.size.get("height", dino_proc.size.get("shortest_edge", 518)))
        self.siglip_size = siglip_proc.size.get("height",
                                                siglip_proc.size.get("shortest_edge", 384))

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.register_buffer("internvit_mean",
                             torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("internvit_std",
                             torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        dino_dim = self.dinov2.config.hidden_size
        siglip_dim = self.siglip.config.hidden_size

        if freeze:
            for p in self.dinov2.parameters():
                p.requires_grad = False
            for p in self.siglip.parameters():
                p.requires_grad = False
            self.dinov2.eval()
            self.siglip.eval()

        self.cross_fusion = CrossAttentionFusion(dino_dim, siglip_dim, hidden=HIDDEN_SIZE)

    def train(self, mode: bool = True):
        """Keep frozen backbones in eval so their dropout/BN never activate."""
        super().train(mode)
        if self.freeze:
            self.dinov2.eval()
            self.siglip.eval()
        return self

    def _preprocess(self, pixel_values_internvit_normalized, size, mean, std):
        """Undo InternViT's ImageNet normalisation, then apply each encoder's own.

        The incoming pixel_values are NOT [0,255]: preprocess_image_batch has
        already applied ToTensor + Normalize with InternViT's statistics, so they
        sit around [-2, 2]. Re-normalising without inverting first feeds both
        encoders corrupted input.
        """
        x = pixel_values_internvit_normalized.float()
        x = x * self.internvit_std.to(x.dtype) + self.internvit_mean.to(x.dtype)
        x = x.clamp(0.0, 1.0)
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return (x - mean.to(x.dtype)) / std.to(x.dtype)

    def _to_fixed_grid(self, feat_seq, has_cls_token):
        if has_cls_token:
            feat_seq = feat_seq[:, 1:, :]
        n, p, c = feat_seq.shape
        side = int(p ** 0.5)
        assert side * side == p, f"patch token count {p} is not a perfect square"
        grid = feat_seq.transpose(1, 2).reshape(n, c, side, side)
        grid = F.interpolate(grid, size=(FIXED_GRID, FIXED_GRID),
                             mode="bilinear", align_corners=False)
        return grid.reshape(n, c, FIXED_GRID * FIXED_GRID).transpose(1, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = next(self.dinov2.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        dino_in = self._preprocess(pixel_values, self.dino_size, self.dino_mean, self.dino_std)
        siglip_in = self._preprocess(pixel_values, self.siglip_size,
                                     self.siglip_mean, self.siglip_std)
        dino_in = dino_in.to(next(self.dinov2.parameters()).dtype)
        siglip_in = siglip_in.to(next(self.siglip.parameters()).dtype)

        # FIX 2: conditional on freeze. An unconditional no_grad() made
        # freeze=False silently train nothing.
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            dino_feat = self.dinov2(dino_in).last_hidden_state
            siglip_feat = self.siglip(siglip_in).last_hidden_state

        dino_feat = self._to_fixed_grid(dino_feat, has_cls_token=True)
        siglip_feat = self._to_fixed_grid(siglip_feat, has_cls_token=False)
        return self.cross_fusion(dino_feat, siglip_feat)


def _load_vision(path: str, attn_implementation: str, vision_only: bool = False):
    """Load a vision tower, requesting FlashAttention-2.

    Falls back only if the checkpoint's architecture has no flash kernel, and
    says so -- silence here is what makes a slow run inexplicable.
    """
    try:
        model = AutoModel.from_pretrained(path, attn_implementation=attn_implementation)
    except Exception as exc:  # noqa: BLE001
        print(f"\033[93m[dual_vision] {path} could not load with "
              f"attn_implementation={attn_implementation!r} ({type(exc).__name__}); "
              f"loading without it.\033[0m")
        model = AutoModel.from_pretrained(path)
    return model.vision_model if vision_only else model


class LingoDualVisionModel(nn.Module):
    def __init__(self, dinov2_path: str, siglip_path: str, freeze: bool = True,
                 *args, **kwargs):
        super().__init__()
        # FIX 3: freeze is forwarded. It used to be dropped, so the default
        # always applied regardless of the caller.
        self.model = DualVisionEncoder(dinov2_path, siglip_path, freeze=freeze)
        self.processor = None
        self.use_global_img = None
        self.num_image_token = FIXED_GRID * FIXED_GRID

    def replace_placeholder_tokens(
        self,
        adaptor_dict: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        placeholder_values: Optional[List[dict]] = None,
        wp_encoder: Optional[nn.Module] = None,
    ):
        self.tokenizer = (self.processor.tokenizer
                          if 'tokenizer' in self.processor.__dict__ else self.processor)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if inputs_embeds is None:
            inputs_embeds = adaptor_dict['language_inputs']
            input_ids = adaptor_dict['language__ids']

            smallest_added_id = self.tokenizer.additional_special_tokens_ids[0]
            special_ids = torch.tensor(
                list(set(input_ids[(input_ids >= smallest_added_id)].tolist())),
                device=input_ids.device,
            ).view(-1, 1, 1)

            if special_ids.size(0) > 0 and len(placeholder_values) > 0:
                wp_encoder_dtype = wp_encoder.mlp[0].weight.dtype
                mask = input_ids == special_ids
                cumsum_mask = torch.cumsum(mask.float(), dim=2)
                first_occurrence_mask = (cumsum_mask == 1) & mask
                first_occurrences = torch.argmax(first_occurrence_mask.float(),
                                                 dim=2).transpose(0, 1)
                special_token_pos = first_occurrences.nonzero()
                coords = [
                    torch.tensor(placeholder_values[b_id][special_ids[key_id].item()],
                                 device=input_ids.device, dtype=wp_encoder_dtype)
                    for key_id, b_id in zip(special_token_pos[:, 1], special_token_pos[:, 0])
                ]
                coords_length_org = [len(c) for c in coords]
                coords = torch.cat(coords)
                wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)
                wp_embeds = torch.split(wp_embeds, coords_length_org)
                first_occurrences_filtered = [first_occurrences[i]
                                              for i in special_token_pos[:, 0]]
                for i, (pos, first_occurrence) in enumerate(
                        zip(special_token_pos, first_occurrences_filtered)):
                    start = first_occurrence[pos[1]]
                    end = start + coords_length_org[i]
                    inputs_embeds[pos[0], start:end] = wp_embeds[i]

            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                _, N_embed, C_embed = inputs_embeds.shape
                BS, T, NP, C, H, W = pixel_values.shape
                assert T == 1, "Only one frame is supported for now"
                pixel_values_flat = pixel_values.view(BS, NP, C, H, W).reshape(BS * NP, C, H, W)

                image_features = self.model(pixel_values_flat)
                vit_embeds = image_features.reshape(-1, C_embed)

                inputs_embeds = inputs_embeds.reshape(BS * N_embed, C_embed)
                input_ids_flat = input_ids.reshape(BS * N_embed)
                selected = (input_ids_flat == self.img_context_token_id)

                n_token = selected.sum()
                assert n_token == vit_embeds.shape[0], (
                    f"<IMG_CONTEXT> token count ({n_token}) does not match "
                    f"vit_embeds ({vit_embeds.shape[0]})")
                inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds
                inputs_embeds = inputs_embeds.reshape(BS, N_embed, C_embed)
                input_ids = input_ids_flat.reshape(BS, N_embed)

            adaptor_dict['language_inputs'] = inputs_embeds
            start_id = adaptor_dict['perm'][:, 0]
            for b, i in enumerate(start_id):
                adaptor_dict['inputs'][b][:len(adaptor_dict['language_inputs'][b]) - i] = \
                    inputs_embeds[b][i:]

        return adaptor_dict
