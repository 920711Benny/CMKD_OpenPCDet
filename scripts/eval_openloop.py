#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, CLIPImageProcessor

from src.dataset import DrivingVLADataset, VLACollator
from src.model_vla import MiniVLA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_indices(manifest_path: str, split_name: str) -> list[int]:
    indices = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == split_name:
                indices.append(idx)
    return indices


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["llm_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained(model_cfg["vision_model"])

    dataset = DrivingVLADataset(args.manifest, image_processor=image_processor, image_size=train_cfg["image_size"])
    val_indices = filter_indices(args.manifest, train_cfg["val_split"])
    val_dataset = Subset(dataset, val_indices)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=VLACollator(tokenizer))

    model = MiniVLA(
        vision_model_name=model_cfg["vision_model"],
        llm_model_name=model_cfg["llm_model"],
        projector_hidden_dim=model_cfg["projector_hidden_dim"],
        action_head_hidden_dim=model_cfg["action_head_hidden_dim"],
        lora_r=model_cfg["lora"]["r"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        lora_target_modules=model_cfg["lora"]["target_modules"],
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=None,
                actions=batch["actions"],
            )
            err = torch.abs(out.pred_action - batch["actions"]).mean(dim=1)
            errors.extend(err.cpu().tolist())
    mean_err = sum(errors) / max(len(errors), 1)
    print(f"mean_abs_action_error={mean_err:.6f}")


if __name__ == "__main__":
    main()
