from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, PreTrainedTokenizerBase

from .prompts import build_prompt, build_target_text


@dataclass
class Sample:
    pixel_values: torch.Tensor
    prompt_text: str
    target_text: str
    action: torch.Tensor


class DrivingVLADataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_processor: CLIPImageProcessor,
        image_size: int = 224,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_processor = image_processor
        self.image_size = image_size
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        speed_kmh = float(row["speed_kmh"])
        prompt_text = build_prompt(row["command"], speed_kmh)
        target_text = build_target_text(
            commentary=row["commentary"],
            steer=float(row["steer"]),
            throttle=float(row["throttle"]),
            brake=float(row["brake"]),
        )
        action = torch.tensor(
            [float(row["steer"]), float(row["throttle"]), float(row["brake"])],
            dtype=torch.float32,
        )
        return Sample(
            pixel_values=pixel_values,
            prompt_text=prompt_text,
            target_text=target_text,
            action=action,
        )


class VLACollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_text_length: int = 160) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: list[Sample]) -> dict[str, Any]:
        prompts = [sample.prompt_text for sample in batch]
        targets = [sample.target_text for sample in batch]
        model_texts = [f"{p}\n{t}" for p, t in zip(prompts, targets)]
        tokenized = self.tokenizer(
            model_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            _ = None
        prompt_tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        prompt_lengths = prompt_tokenized["attention_mask"].sum(dim=1)
        for i, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[i, :prompt_len] = -100
        return {
            "pixel_values": torch.stack([sample.pixel_values for sample in batch]),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "actions": torch.stack([sample.action for sample in batch]),
        }
