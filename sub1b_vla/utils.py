"""Config loading, scratch-dir redirection, small shared helpers."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path


# --------------------------------------------------------------------------
# YAML
# --------------------------------------------------------------------------
def _parse_scalar(s: str):
    s = s.strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):        # inline mapping
        body = s[1:-1].strip()
        out = {}
        if body:
            for part in _split_top(body):
                k, _, v = part.partition(":")
                out[k.strip()] = _parse_scalar(v)
        return out
    if s.startswith("[") and s.endswith("]"):
        body = s[1:-1].strip()
        return [_parse_scalar(p) for p in _split_top(body)] if body else []
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s.strip("'\"")


def _split_top(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _mini_yaml(text: str) -> dict:
    """Parser for the nested-mapping subset used by this project's configs."""
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#")[0].rstrip() if not _in_quotes_hash(raw) else raw.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            node: dict = {}
            parent[key.strip()] = node
            stack.append((indent, node))
        else:
            parent[key.strip()] = _parse_scalar(val)
    return root


def _in_quotes_hash(line: str) -> bool:
    return bool(re.search(r"['\"][^'\"]*#", line))


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    text = Path(path).read_text()
    try:
        import yaml  # noqa: PLC0415

        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = _mini_yaml(text)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        set_nested(cfg, key.strip(), _parse_scalar(val))
    return cfg


def set_nested(cfg: dict, dotted: str, value):
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def setup_scratch_dirs(cfg: dict | None = None) -> dict:
    """Redirect HF_HOME / TORCH_HOME to a high-capacity scratch directory.

    Honours an existing SUB1B_SCRATCH; otherwise uses <out_dir>/../scratch so
    multi-GB checkpoints never land on a small home partition.
    """
    scratch = os.environ.get("SUB1B_SCRATCH")
    if scratch is None:
        base = Path((cfg or {}).get("train", {}).get("out_dir", "./runs")).resolve().parent
        scratch = str(base / "scratch")
    root = Path(scratch)
    for name, sub in (("HF_HOME", "hf"), ("TORCH_HOME", "torch"),
                      ("HF_DATASETS_CACHE", "hf_datasets"), ("XDG_CACHE_HOME", "xdg")):
        p = root / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(p))
    return {"scratch": str(root)}


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"
