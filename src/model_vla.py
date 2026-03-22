from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, CLIPVisionModel


@dataclass
class MiniVLAOutput:
    loss: torch.Tensor | None
    loss_action: torch.Tensor
    loss_lang: torch.Tensor | None
    pred_action: torch.Tensor
    logits: torch.Tensor | None


class MiniVLA(nn.Module):
    def __init__(
        self,
        vision_model_name: str,
        llm_model_name: str,
        projector_hidden_dim: int = 1024,
        action_head_hidden_dim: int = 1024,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.config.output_hidden_states = True

        vision_dim = self.vision_encoder.config.hidden_size
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, llm_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(llm_dim, action_head_hidden_dim),
            nn.GELU(),
            nn.Linear(action_head_hidden_dim, 3),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        lang_loss_weight: float = 0.3,
        action_loss_weight: float = 1.0,
    ) -> MiniVLAOutput:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        visual_prefix = self.projector(pooled).unsqueeze(1)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
        visual_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        combined_labels = None
        if labels is not None:
            ignore = torch.full(
                (labels.size(0), 1),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([ignore, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        raw_action = self.action_head(last_hidden)
        pred_action = torch.stack(
            [
                torch.tanh(raw_action[:, 0]),
                torch.sigmoid(raw_action[:, 1]),
                torch.sigmoid(raw_action[:, 2]),
            ],
            dim=1,
        )

        loss_lang = outputs.loss if labels is not None else None
        if actions is None:
            loss_action = torch.tensor(0.0, device=combined_embeds.device)
        else:
            loss_action = F.smooth_l1_loss(pred_action, actions)

        loss = None
        if loss_lang is not None:
            loss = action_loss_weight * loss_action + lang_loss_weight * loss_lang
        elif actions is not None:
            loss = action_loss_weight * loss_action

        return MiniVLAOutput(
            loss=loss,
            loss_action=loss_action,
            loss_lang=loss_lang,
            pred_action=pred_action,
            logits=outputs.logits if hasattr(outputs, "logits") else None,
        )
