#!/usr/bin/env bash
# Preflight + launch for CarlaVLA v3 training on the RTX 6000 Pro box.
#
#   ./launch_training.sh /path/to/simlingo [config_name]
#
# Refuses to start rather than wasting hours on a run that was doomed at step 0.
# Every check below corresponds to a failure that is expensive to discover late.
set -euo pipefail

REPO="${1:-}"
CFG_NAME="${2:-config_tuned}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$REPO" ]; then sed -n '2,8p' "$0"; exit 1; fi
cd "$REPO"

fail() { echo "BLOCKED: $*" >&2; exit 1; }
ok()   { echo "  ok    $*"; }

echo "=== preflight ==="

# 1. GPUs
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found; no GPU visible."
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ "$NGPU" -ge 1 ] || fail "no GPUs reported by nvidia-smi."
ok "$NGPU GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ')"

# 2. torch sees them, and bf16 works (the config asks for bf16-mixed)
python3 - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("BLOCKED: torch.cuda.is_available() is False -- wrong wheel for this driver?")
if not torch.cuda.is_bf16_supported():
    sys.exit("BLOCKED: bf16 unsupported; the config requests precision: bf16-mixed.")
print(f"  ok    torch {torch.__version__} (cuda {torch.version.cuda}), bf16 supported")
PY

# 3. FlashAttention-2, by RUNNING it. A wheel built for the wrong compute
#    capability imports cleanly and is then never used.
python3 - <<'PY' || exit 1
import sys, torch
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
try:
    from flash_attn import flash_attn_func
    flash_attn_func(q, q, q)
    print("  ok    flash-attn package kernel runs")
except Exception as e:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import torch.nn.functional as F
        t = q.transpose(1, 2)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(t, t, t)
        print(f"  ok    torch FLASH_ATTENTION kernel runs (flash-attn package unusable: {type(e).__name__})")
    except Exception as e2:
        sys.exit(f"BLOCKED: no FlashAttention available. package={e}; torch={e2}")
PY

# 4. Patches applied?
grep -q "FlashMultiheadAttention" simlingo_training/models/encoder/dual_vision_model.py 2>/dev/null \
  && ok "dual_vision_model.py patched" \
  || echo "  WARN  dual_vision_model.py NOT patched -- cross-attention will not reach flash-attn."
grep -q "prediction_type" simlingo_training/models/diffusion_decoder.py 2>/dev/null \
  && ok "diffusion_decoder.py patched" \
  || echo "  WARN  diffusion_decoder.py NOT patched -- epsilon + coord(-32,32) is the slow path."

# 5. Dataset present
[ -d database ] || fail "no ./database directory; extract the SimLingo dataset first."
NROUTE=$(find database -maxdepth 6 -type d -name measurements 2>/dev/null | wc -l)
[ "$NROUTE" -gt 0 ] || fail "no */measurements directories under ./database."
ok "$NROUTE route(s) with measurements"

# 6. Config
CFG="simlingo_training/config/$CFG_NAME.yaml"
[ -f "$CFG" ] || CFG="$HERE/../configs_carlavla/$CFG_NAME.yaml"
[ -f "$CFG" ] || fail "config $CFG_NAME.yaml not found."
ok "config $CFG"
python3 - "$CFG" <<'PY' || exit 1
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
dm, m = c["data_module"], c["model"]
eff = dm["batch_size"] * c.get("gpus", 1)
print(f"  ok    batch {dm['batch_size']} x {c.get('gpus',1)} GPUs = effective {eff}, "
      f"lr {m['lr']}, {c['max_epochs']} epochs, {c['precision']}")
if dm.get("num_workers", 0) == 0:
    sys.exit("BLOCKED: num_workers=0 will starve the GPUs on image decoding.")
if eff != 48 and abs(m["lr"] - 3e-5) < 1e-9:
    sys.exit(f"BLOCKED: effective batch {eff} with SimLingo's batch-48 lr of 3e-5. "
             "Scale the lr or restore batch 48 via accumulation.")
PY

# 7. Disk for checkpoints
FREE=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$FREE" -ge 50 ] || echo "  WARN  only ${FREE}G free; a 15-epoch run writes many checkpoints."
ok "${FREE}G free"

echo
echo "=== launching ==="
export PYTHONPATH="${CARLA_ROOT:-}/PythonAPI/carla:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
set -x
exec python3 simlingo_training/train.py --config-name "$CFG_NAME"
