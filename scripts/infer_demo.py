#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import yaml
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from src.model_vla import MiniVLA
from src.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--speed-kmh", type=float, required=True)
    parser.add_argument("--command", default="follow_lane")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["llm_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained(model_cfg["vision_model"])

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
    model.eval()

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
    prompt = build_prompt(args.command, args.speed_kmh)
    tokenized = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            labels=None,
            actions=None,
        )
    steer, throttle, brake = out.pred_action.squeeze(0).tolist()
    print(prompt)
    print(f"Predicted steer={steer:.4f} throttle={throttle:.4f} brake={brake:.4f}")


if __name__ == "__main__":
    main()
