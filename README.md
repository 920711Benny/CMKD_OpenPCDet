# SimLingo-style Mini VLA Starter

This repository contains a lightweight starter project for building a SimLingo-style vision-language-action (VLA) driving model on a single 12GB GPU.

## What this project does
- Downloads a CARLA driving dataset from Hugging Face.
- Uses a Hugging Face vision encoder and Qwen language model.
- Trains a lightweight VLA model with LoRA plus an action head.
- Supports open-loop evaluation and a simple inference demo.

## Recommended defaults
- Dataset: `immanuelpeter/carla-autopilot-multimodal-dataset`
- Vision encoder: `openai/clip-vit-base-patch32`
- Language model: `Qwen/Qwen2.5-1.5B-Instruct`

## Storage layout for this machine
Use the external drive **only for dataset files**:
- External dataset root: `/media/systemlab/32092FEF5BF8D0C2/simlingo_vla_data`

Keep everything else on the computer's internal storage:
- Repo checkout: `~/CMKD_OpenPCDet`
- Python virtualenv: `~/CMKD_OpenPCDet/.venv`
- Training outputs/checkpoints: `~/CMKD_OpenPCDet/outputs`
- Hugging Face caches (optional): internal disk unless you decide otherwise

## If GitHub is missing the latest files
If the GitHub remote does not contain the latest local work yet, create a portable export package from this repo and move it to the target machine.

### Create local export packages
```bash
bash scripts/package_release.sh
```

This generates both:
- `dist/CMKD_OpenPCDet-<branch>-<commit>.tar.gz`
- `dist/CMKD_OpenPCDet-<branch>-<commit>.bundle`

### Fastest install path
On the target machine:
```bash
mkdir -p ~/packages
cp /path/to/CMKD_OpenPCDet-*.tar.gz ~/packages/
cd ~
tar -xzf ~/packages/CMKD_OpenPCDet-*.tar.gz
cd CMKD_OpenPCDet
```

After extraction, continue with the setup commands below.

## Quick start
```bash
git clone https://github.com/920711Benny/CMKD_OpenPCDet.git
cd CMKD_OpenPCDet
git fetch --all --prune
git checkout work || git checkout main || git checkout master

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -m compileall src scripts

export DATA_ROOT=/media/systemlab/32092FEF5BF8D0C2/simlingo_vla_data
mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/processed" outputs

python scripts/download_dataset.py \
  --dataset-id immanuelpeter/carla-autopilot-multimodal-dataset \
  --output-dir "$DATA_ROOT/raw" \
  --max-samples 2000

python scripts/build_manifest.py \
  --source-dir "$DATA_ROOT/raw" \
  --output-path "$DATA_ROOT/processed/manifest.jsonl"

python scripts/train_vla.py \
  --manifest "$DATA_ROOT/processed/manifest.jsonl" \
  --output-dir outputs/mini_vla_run \
  --limit-train-samples 128

python scripts/eval_openloop.py \
  --manifest "$DATA_ROOT/processed/manifest.jsonl" \
  --checkpoint outputs/mini_vla_run/best.pt

python scripts/infer_demo.py \
  --image "$DATA_ROOT/raw/images/sample_000000.jpg" \
  --speed-kmh 8.0 \
  --command follow_lane \
  --checkpoint outputs/mini_vla_run/best.pt
```

## Full command sequence
If you want a copy-paste sequence that keeps **only the dataset** on the external drive and leaves the repo, environment, and outputs on the computer disk:

```bash
git clone https://github.com/920711Benny/CMKD_OpenPCDet.git
cd CMKD_OpenPCDet
git fetch --all --prune
git checkout work || git checkout main || git checkout master

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m compileall src scripts

export DATA_ROOT=/media/systemlab/32092FEF5BF8D0C2/simlingo_vla_data
mkdir -p "$DATA_ROOT/raw/images" "$DATA_ROOT/processed" outputs

df -h "$DATA_ROOT"

python scripts/download_dataset.py \
  --dataset-id immanuelpeter/carla-autopilot-multimodal-dataset \
  --output-dir "$DATA_ROOT/raw" \
  --max-samples 2000

python scripts/build_manifest.py \
  --source-dir "$DATA_ROOT/raw" \
  --output-path "$DATA_ROOT/processed/manifest.jsonl"

python - <<'PY'
from pathlib import Path
p = Path('/media/systemlab/32092FEF5BF8D0C2/simlingo_vla_data/processed/manifest.jsonl')
print('manifest exists:', p.exists())
print('manifest path:', p)
PY

python scripts/train_vla.py \
  --manifest "$DATA_ROOT/processed/manifest.jsonl" \
  --output-dir outputs/mini_vla_run \
  --limit-train-samples 128

python scripts/eval_openloop.py \
  --manifest "$DATA_ROOT/processed/manifest.jsonl" \
  --checkpoint outputs/mini_vla_run/best.pt

python scripts/infer_demo.py \
  --image "$DATA_ROOT/raw/images/sample_000000.jpg" \
  --speed-kmh 8.0 \
  --command follow_lane \
  --checkpoint outputs/mini_vla_run/best.pt
```

## Notes
- Start with a small subset (`--max-samples`) because the source dataset is large.
- The generated language supervision is template-based and meant to bootstrap a first VLA prototype.
- This project is designed for single-GPU experimentation, not full SimLingo reproduction.
- The dataset download path can be redirected anywhere by changing `DATA_ROOT`; the training output path remains local unless you change `--output-dir`.
