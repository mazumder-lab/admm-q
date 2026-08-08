"""SmoothQuant preprocessing for Qwen3 models.

Migrates quantization difficulty from activations to weights via per-channel
scaling, absorbed into the preceding LayerNorm.  Applied *before* any
quantization method (RTN / GPTQ / ADMM).

Reference: Xiao et al., "SmoothQuant: Accurate and Efficient
Post-Training Quantization for Large Language Models", ICML 2023.
"""

import gc
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Smoothing groups for Qwen3: (layernorm_attr, [linear_attrs])
_SMOOTH_GROUPS = [
    ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
]

# Linear names whose activation scales we need to collect
_ABSORBABLE_LINEARS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "mlp.gate_proj", "mlp.up_proj",
]


def _get_submodule(module, dotted_name):
    """Retrieve a sub-module by dotted name (e.g. 'self_attn.q_proj')."""
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


@torch.no_grad()
def collect_act_scales(model, dataloader, dev, nsamples=128):
    """Collect per-channel activation scales for absorbable linears.

    Uses the same Catcher + layer-by-layer forward pattern as the
    quantization pipeline but only records max(|input|) per channel.

    Returns dict mapping "layer_idx.linear_name" -> 1-D Tensor of shape
    (in_features,).
    """
    log.info("Collecting activation scales for SmoothQuant ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move embedding + norm to device for input capture
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    seqlen = 2048
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}
    extra_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            if "position_embeddings" in kwargs:
                extra_kwargs["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    act_scales = {}  # "layer_idx.linear_name" -> 1-D tensor

    for i in range(len(layers)):
        log.info("Collecting scales for layer %d/%d", i, len(layers) - 1)
        layer = layers[i].to(dev)

        # Build forward kwargs
        fwd_kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
        if extra_kwargs.get("position_embeddings") is not None:
            fwd_kwargs["position_embeddings"] = extra_kwargs["position_embeddings"]

        # Register hooks on absorbable linears
        hooks = []
        for lname in _ABSORBABLE_LINEARS:
            linear = _get_submodule(layer, lname)
            key = f"{i}.{lname}"
            act_scales[key] = torch.zeros(linear.in_features, device=dev, dtype=torch.float32)

            def _hook(mod, inp, out, _key=key):
                x = inp[0].data  # (batch, seq, hidden) or (seq, hidden)
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                act_scales[_key] = torch.max(act_scales[_key], x.abs().amax(dim=0).float())

            hooks.append(linear.register_forward_hook(_hook))

        # Forward all samples through this layer
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kwargs)[0]

        for h in hooks:
            h.remove()

        # Move scales to CPU to save GPU memory
        for lname in _ABSORBABLE_LINEARS:
            key = f"{i}.{lname}"
            act_scales[key] = act_scales[key].cpu()

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Activation scale collection done.")
    return act_scales


@torch.no_grad()
def smooth_model(model, act_scales, alpha=0.5):
    """Apply SmoothQuant scaling to the model in-place.

    For each smoothing group (layernorm -> linears):
      s_j = act_scales_j^alpha / weight_scales_j^(1-alpha)
      layernorm.weight /= s
      linear.weight *= s  (along input channel dimension)
    """
    log.info("Applying SmoothQuant (alpha=%.2f) ...", alpha)
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        for ln_name, linear_names in _SMOOTH_GROUPS:
            ln = getattr(layer, ln_name)

            # Gather activation scales (take element-wise max across linears in the group)
            group_act_scale = None
            for lname in linear_names:
                key = f"{i}.{lname}"
                s = act_scales[key].float()
                if group_act_scale is None:
                    group_act_scale = s.clone()
                else:
                    group_act_scale = torch.max(group_act_scale, s)

            # Compute weight scales: max over group linears of per-input-channel |W|
            weight_scale = None
            for lname in linear_names:
                linear = _get_submodule(layer, lname)
                ws = linear.weight.data.float().abs().amax(dim=0)  # (in_features,)
                if weight_scale is None:
                    weight_scale = ws
                else:
                    weight_scale = torch.max(weight_scale, ws)

            # Compute smoothing factor
            # s = act^alpha / weight^(1-alpha), clamped for numerical stability
            s = (group_act_scale.pow(alpha) / weight_scale.pow(1 - alpha)).clamp(min=1e-5)

            # Apply: absorb s^{-1} into LayerNorm, scale linear inputs by s
            device = ln.weight.device
            s = s.to(device)

            ln.weight.data = ln.weight.data.float().div_(s).to(ln.weight.dtype)

            for lname in linear_names:
                linear = _get_submodule(layer, lname)
                linear.weight.data = (
                    linear.weight.data.float().mul_(s.view(1, -1)).to(linear.weight.dtype)
                )

        log.info("Smoothed layer %d/%d", i, len(layers) - 1)

    log.info("SmoothQuant applied.")
