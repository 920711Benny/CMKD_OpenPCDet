"""Chain-of-Causation language backbone.

Wraps a small causal LM (InternVL2-1B's Qwen2-0.5B decoder by default -- hidden
size 896, matching the configured `embed_dim`) and prefixes it with the semantic
tokens produced by the dual-head encoder. Only LoRA adapters + the projection
layers train.

Two outputs matter downstream:
  * `lm_loss`      -- next-token loss on the CoC rationale.
  * `intent_logits`-- a light classification head over the canonical intents,
                      read off the pooled rationale representation. This is what
                      the consistency loss can differentiate through; parsing
                      generated text is not differentiable.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coc_prompt import INTENTS
from .lora import DEFAULT_LORA_TARGETS, inject_lora, mark_only_lora_trainable


@dataclass
class LanguageOutput:
    lm_loss: torch.Tensor | None
    intent_logits: torch.Tensor      # (B, num_intents)
    hidden: torch.Tensor             # (B, T, D) final hidden states
    pooled: torch.Tensor             # (B, D) over the full sequence
    pooled_prefix: torch.Tensor      # (B, D) over the visual prefix only


class StubDecoderBlock(nn.Module):
    """Pre-norm causal block. Deliberately uses q_proj/k_proj/v_proj/o_proj so
    the offline stub is adapted by exactly the same LoRA targeting rule as the
    real Qwen2 decoder -- a stub that dodged the adapter path would validate
    nothing."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.n1 = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.n2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        h = self.n1(x)
        b, t, _ = h.shape
        q, k, v = self._split(self.q_proj(h)), self._split(self.k_proj(h)), self._split(self.v_proj(h))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, t, -1)
        x = x + self.o_proj(a)
        return x + self.mlp(self.n2(x))


class StubCausalLM(nn.Module):
    """Small decoder-only LM used when the real checkpoint is unavailable."""

    def __init__(self, vocab_size: int = 4096, hidden: int = 896, layers: int = 2, heads: int = 8):
        super().__init__()
        self.hidden_size = hidden
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([StubDecoderBlock(hidden, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor):
        h = inputs_embeds
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h, self.lm_head(h)

    def get_input_embeddings(self):
        return self.embed


class CoCLanguageModel(nn.Module):
    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2-1B",
        embed_dim: int = 896,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] | None = None,
        allow_stub: bool = True,
        use_language_tower_only: bool = True,
    ):
        super().__init__()
        self.model_id = model_id
        self.is_stub = False
        self.backbone, self.hidden_size, self.tokenizer = self._build(
            model_id, embed_dim, allow_stub, use_language_tower_only
        )
        targets = tuple(lora_targets) if lora_targets else DEFAULT_LORA_TARGETS
        n = inject_lora(
            self.backbone, target_substrings=targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
        )
        if n == 0:
            warnings.warn(
                "LoRA injected into 0 modules -- check target_substrings for this backbone.",
                RuntimeWarning, stacklevel=2,
            )
        mark_only_lora_trainable(self.backbone)
        self.lora_modules = n

        self.in_proj = (
            nn.Identity() if embed_dim == self.hidden_size
            else nn.Linear(embed_dim, self.hidden_size)
        )
        self.intent_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, len(INTENTS)),
        )

    def _build(self, model_id, embed_dim, allow_stub, tower_only):
        """Load the LM. `tower_only` keeps InternVL's language decoder and drops
        its native InternViT tower -- our dual-head encoder replaces it, and
        dropping it is what keeps the total under the 1B parameter budget."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float32
            )
            if tower_only:
                for attr in ("language_model", "llm", "text_model"):
                    if hasattr(model, attr):
                        model = getattr(model, attr)
                        break
                else:
                    warnings.warn(
                        f"No language tower found on {model_id}; using the full model. "
                        "Verify the parameter budget.", RuntimeWarning, stacklevel=2,
                    )
            hidden = model.config.hidden_size
            return model, hidden, tok
        except Exception as exc:  # noqa: BLE001
            if not allow_stub:
                raise
            warnings.warn(
                f"[STUB LM] '{model_id}' unavailable ({type(exc).__name__}: {exc}). "
                "Using randomly-initialised StubCausalLM. Text output is NOT meaningful.",
                RuntimeWarning, stacklevel=2,
            )
            self.is_stub = True
            return StubCausalLM(hidden=embed_dim), embed_dim, None

    # ---- internals -------------------------------------------------------
    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(ids)

    def _run(self, inputs_embeds, attention_mask=None):
        if isinstance(self.backbone, StubCausalLM):
            return self.backbone(inputs_embeds)
        out = self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        hidden = out.hidden_states[-1] if out.hidden_states is not None else out.last_hidden_state
        return hidden, out.logits

    # ---- API -------------------------------------------------------------
    def forward(
        self,
        semantic_tokens: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> LanguageOutput:
        prefix = self.in_proj(semantic_tokens)
        b, n_prefix, _ = prefix.shape

        if text_ids is not None:
            text_embeds = self._embed_tokens(text_ids)
            inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
        else:
            inputs_embeds = prefix

        attn = None
        if text_mask is not None:
            attn = torch.cat(
                [torch.ones(b, n_prefix, device=text_mask.device, dtype=text_mask.dtype), text_mask],
                dim=1,
            )

        hidden, logits = self._run(inputs_embeds, attn)

        lm_loss = None
        if labels is not None:
            # Prefix positions never carry a text label.
            pad = torch.full((b, n_prefix), -100, device=labels.device, dtype=labels.dtype)
            full = torch.cat([pad, labels], dim=1)
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                full[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        if attn is not None:
            m = attn.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            pooled = hidden.mean(dim=1)

        # The intent head reads ONLY the visual prefix positions. Pooling the
        # full sequence would let it read the intent straight out of the target
        # text during training (a label leak) and then face a prefix-only
        # sequence at inference -- it would learn nothing transferable.
        pooled_prefix = hidden[:, :n_prefix].mean(dim=1)

        return LanguageOutput(
            lm_loss=lm_loss,
            intent_logits=self.intent_head(pooled_prefix),
            hidden=hidden,
            pooled=pooled,
            pooled_prefix=pooled_prefix,
        )

    @torch.no_grad()
    def generate(self, semantic_tokens, prompt_ids=None, max_new_tokens: int = 64):
        """Greedy decode of the CoC rationale, for the async HUD stream.

        Uses KV caching on real backbones: without it each new token re-runs the
        whole prefix, making the rationale O(n^2) and starving the control loop
        it is supposed to stay out of the way of.
        """
        prefix = self.in_proj(semantic_tokens)
        embeds = prefix if prompt_ids is None else torch.cat(
            [prefix, self._embed_tokens(prompt_ids)], dim=1)

        if isinstance(self.backbone, StubCausalLM):
            return self._generate_uncached(embeds, semantic_tokens, max_new_tokens)

        out_ids: list[torch.Tensor] = []
        past = None
        step_embeds = embeds
        eos = getattr(self.tokenizer, "eos_token_id", None)
        for _ in range(max_new_tokens):
            out = self.backbone(inputs_embeds=step_embeds, past_key_values=past,
                                use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(dim=-1)
            out_ids.append(nxt)
            if eos is not None and bool((nxt == eos).all()):
                break
            step_embeds = self._embed_tokens(nxt[:, None])
        if not out_ids:
            return torch.zeros(semantic_tokens.shape[0], 0, dtype=torch.long)
        return torch.stack(out_ids, dim=1)

    def _generate_uncached(self, embeds, semantic_tokens, max_new_tokens):
        out_ids: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            _, logits = self._run(embeds, None)
            nxt = logits[:, -1].argmax(dim=-1)
            out_ids.append(nxt)
            embeds = torch.cat([embeds, self._embed_tokens(nxt[:, None])], dim=1)
        if not out_ids:
            return torch.zeros(semantic_tokens.shape[0], 0, dtype=torch.long)
        return torch.stack(out_ids, dim=1)

    def decode(self, ids: torch.Tensor) -> list[str]:
        if self.tokenizer is None:
            return ["<stub-lm: no tokenizer>"] * ids.shape[0]
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
