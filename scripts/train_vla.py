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
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-train-samples", type=int, default=0)
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


def evaluate(model: MiniVLA, loader: DataLoader, device: torch.device, cfg: dict) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                actions=batch["actions"],
                lang_loss_weight=cfg["lang_loss_weight"],
                action_loss_weight=cfg["action_loss_weight"],
            )
            total += float(out.loss.item())
            count += 1
    model.train()
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["llm_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained(model_cfg["vision_model"])

    dataset = DrivingVLADataset(args.manifest, image_processor=image_processor, image_size=train_cfg["image_size"])
    train_indices = filter_indices(args.manifest, train_cfg["train_split"])
    val_indices = filter_indices(args.manifest, train_cfg["val_split"])
    if args.limit_train_samples > 0:
        train_indices = train_indices[: args.limit_train_samples]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    collator = VLACollator(tokenizer, max_text_length=model_cfg["max_text_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collator,
    )

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and train_cfg["mixed_precision"] in {"fp16", "bf16"}))

    best_val = float("inf")
    global_step = 0
    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            use_amp = device.type == "cuda" and train_cfg["mixed_precision"] in {"fp16", "bf16"}
            amp_dtype = torch.float16 if train_cfg["mixed_precision"] == "fp16" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    actions=batch["actions"],
                    lang_loss_weight=train_cfg["lang_loss_weight"],
                    action_loss_weight=train_cfg["action_loss_weight"],
                )
                loss = out.loss / train_cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()

            if step % train_cfg["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % train_cfg["log_every"] == 0:
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={out.loss.item():.4f} action={out.loss_action.item():.4f} "
                        f"lang={(out.loss_lang.item() if out.loss_lang is not None else 0.0):.4f}"
                    )
                if global_step % train_cfg["save_every"] == 0:
                    val_loss = evaluate(model, val_loader, device, train_cfg)
                    ckpt = output_dir / f"step_{global_step}.pt"
                    torch.save({"model": model.state_dict(), "val_loss": val_loss}, ckpt)
                    print(f"saved {ckpt} val_loss={val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({"model": model.state_dict(), "val_loss": val_loss}, output_dir / "best.pt")
                        print(f"updated best.pt val_loss={val_loss:.4f}")

    final_val = evaluate(model, val_loader, device, train_cfg)
    torch.save({"model": model.state_dict(), "val_loss": final_val}, output_dir / "last.pt")
    print(f"training complete val_loss={final_val:.4f}")


if __name__ == "__main__":
    main()
