"""Activation quantization via forward hooks on nn.Linear layers.

Inserts quantize-dequantize (fake quantization) on the input to each
linear layer at inference time.  Supports per-token and per-tensor modes.

Usage:
    handles = add_act_quant_hooks(model, bits=4, sym=True, per_token=True)
    # ... run evaluation ...
    remove_act_quant_hooks(handles)
"""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _fake_quantize_activation(x, bits, sym=True, per_token=True):
    """Quantize-dequantize activation tensor x in-place (no gradient).

    Args:
        x: input tensor, typically (batch, seq, hidden)
        bits: bit width for activation quantization
        sym: symmetric quantization
        per_token: if True, compute scale per token (last dim);
                   if False, per-tensor scale
    Returns:
        Fake-quantized tensor (same shape/dtype as x)
    """
    maxq = 2 ** (bits - 1) - 1
    dtype = x.dtype
    x = x.float()

    if per_token:
        # Scale per token: always reduce over last dim (hidden/head_dim)
        # Works for 2D (tokens, hidden), 3D (batch, seq, hidden),
        # and 4D (batch, heads, seq, head_dim)
        amax = x.abs().amax(dim=-1, keepdim=True)
    else:
        amax = x.abs().amax()

    amax = amax.clamp(min=1e-5)
    scale = amax / maxq

    if sym:
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        result = scale * q
    else:
        # Asymmetric: shift to [0, 2*maxq+1]
        xmin = x.min() if not per_token else x.amin(dim=-1, keepdim=True)
        xmax = x.max() if not per_token else x.amax(dim=-1, keepdim=True)
        full_maxq = 2 ** bits - 1
        scale_asym = (xmax - xmin).clamp(min=1e-8) / full_maxq
        zero = torch.round(-xmin / scale_asym)
        q = torch.clamp(torch.round(x / scale_asym + zero), 0, full_maxq)
        result = scale_asym * (q - zero)

    return result.to(dtype)


def add_act_quant_hooks(model, bits=4, sym=True, per_token=True):
    """Register forward pre-hooks that fake-quantize inputs to all Linear layers.

    Returns list of hook handles (call remove_act_quant_hooks to clean up).
    """
    handles = []
    count = 0

    def _make_hook(name):
        def hook(module, args):
            x = args[0]
            x_q = _fake_quantize_activation(x, bits, sym=sym, per_token=per_token)
            return (x_q,) + args[1:]
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_pre_hook(_make_hook(name))
            handles.append(h)
            count += 1

    log.info("Added activation quantization hooks to %d Linear layers (A%d, sym=%s, per_token=%s)",
             count, bits, sym, per_token)
    return handles


def remove_act_quant_hooks(handles):
    """Remove all activation quantization hooks."""
    for h in handles:
        h.remove()
    log.info("Removed %d activation quantization hooks", len(handles))
