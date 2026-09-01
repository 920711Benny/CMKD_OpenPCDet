"""Action-CoT Alignment Score.

Fraction of decisions whose executed trajectory actually exhibits the dynamics
of the intent the model stated. Reported alongside driving metrics because a
rationale that does not predict the action is not an explanation of it.

Two intent sources are supported:
  * `head`   -- the differentiable intent classifier (always available)
  * `parsed` -- the intent parsed out of the generated CoC text, which is the
                stricter and more honest measure since it scores what the
                operator actually reads on the HUD.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..data.dataset import DrivingVLADataset, collate
from ..losses.consistency import action_cot_alignment_score
from ..models.coc_prompt import parse_intent


@torch.inference_mode()
def evaluate(model, cfg, device, num_batches=16, batch_size=8, intent_source="head"):
    ds = DrivingVLADataset(cfg, "val", tokenizer=model.language.tokenizer)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    scores, per_scenario = [], {}
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.predict_trajectory(
            batch_dev["image"], batch_dev["speed"],
            batch_dev["target_point"], batch_dev["command"],
        )
        if intent_source == "parsed":
            enc = model.encoder(batch_dev["image"])
            texts = model.explain(enc, max_new_tokens=48)
            ids = torch.tensor([parse_intent(t)[1] for t in texts], device=device)
        else:
            ids = out.intent_logits.argmax(-1)
        s = action_cot_alignment_score(ids, out.waypoints, dt=cfg["model"].get("waypoint_dt", 0.2))
        scores.append(s.float().cpu().numpy())
        for sc, val in zip(batch["scenario"], s.float().cpu().numpy()):
            per_scenario.setdefault(sc, []).append(float(val))

    all_s = np.concatenate(scores) if scores else np.zeros(0)
    return {
        "alignment_score": float(all_s.mean()) if all_s.size else None,
        "n": int(all_s.size),
        "intent_source": intent_source,
        "per_scenario": {k: float(np.mean(v)) for k, v in sorted(per_scenario.items())},
    }


def main():
    from ..carla_agent.agent import load_agent_model  # noqa: PLC0415
    from ..utils import load_config  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--intent-source", choices=["head", "parsed"], default="head")
    ap.add_argument("--num-batches", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_agent_model(cfg, args.checkpoint, device)
    res = evaluate(model, cfg, device, num_batches=args.num_batches,
                   intent_source=args.intent_source)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
