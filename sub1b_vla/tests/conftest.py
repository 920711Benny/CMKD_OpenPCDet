"""Test-suite configuration.

The attention layer requires FlashAttention-2 and refuses to fall back. CI and
this repo's CPU test runs have no CUDA device, so the suite opts in explicitly
to the math kernel. This is set HERE, in the tests only -- never in library code
or a training config -- so a real run can never inherit it.
"""
import os

os.environ.setdefault("SUB1B_ATTENTION_ALLOW_FALLBACK", "1")
