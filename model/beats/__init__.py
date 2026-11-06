# Vendored from https://github.com/microsoft/unilm/tree/master/beats at commit
# 732d834db70ee0fc3886b4bcbcfb4ce7fb829be2 (MIT License, Copyright (c) Microsoft
# Corporation; see LICENSE in this directory). Only BEATs.py, backbone.py and modules.py
# are kept (no tokenizer/quantizer). Local edits, all marked "OS-SCL:" or listed here:
#   - package-relative imports (`from .backbone`, `from .modules`)
#   - backbone.TransformerEncoder: optional per-layer activation checkpointing when the
#     attribute `grad_ckpt` is set (off by default -> upstream behaviour)
#   - backbone.MultiheadAttention: optional fused F.scaled_dot_product_attention path when the
#     attribute `sdpa` is set (no padding / attn mask / weights requested); off by default
from .BEATs import BEATs, BEATsConfig  # noqa: F401
