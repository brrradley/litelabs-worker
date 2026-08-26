"""Compatibility launcher for the baked trusted Demucs v3 sax checkpoint.

PyTorch 2.6 changed torch.load(weights_only) to True by default. Demucs 3 model
packages contain their model class and therefore require the legacy full-package
loader. This shim is used only for the pinned, checksum-verified sax checkpoint
already baked into the LiteLABS image.
"""
from __future__ import annotations

import runpy

import torch

_original_torch_load = torch.load


def _legacy_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _legacy_torch_load
runpy.run_module("demucs", run_name="__main__")
