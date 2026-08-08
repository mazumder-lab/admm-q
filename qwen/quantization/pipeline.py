"""Layer-wise weight quantization for Qwen3 models.

Adapted from admmq_colm26/pipeline.py for the Qwen3 architecture.
One wrapper class (add_batch + fasterquant + free),
one layer-wise function per method.

    quantize_weights_admm(model, dataloader, dev, ...)
    quantize_weights_gptq(model, dataloader, dev, ...)
    quantize_weights_rtn(model, dev, ...)
"""

# Limit OpenMP/MKL threads before any torch/numpy to avoid "Aborted (core dumped)"
# when torch.linalg.eigh/cholesky on large matrices spawn too many threads.
import os
_env_set = os.environ.get("OMP_NUM_THREADS")
if _env_set is None:
    os.environ["OMP_NUM_THREADS"] = "4"
if os.environ.get("MKL_NUM_THREADS") is None:
    os.environ["MKL_NUM_THREADS"] = "4"

import gc
import logging
import math

import torch
# Limit PyTorch intra-op threads to avoid thread exhaustion with ADMM/GPTQ linalg.
try:
    torch.set_num_threads(4)
except Exception:
    pass
import torch.nn as nn

from .solvers import ADMM_quantize, GPTQ_quantize, compute_scale_zero, fake_quant

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class QuantWrapper:
    """Collects Hessian and applies weight quantization (ADMM or GPTQ)."""

    def __init__(self, layer, method="admm"):
        assert method in ("admm", "gptq")
        self.layer = layer
        self.method = method
        self.dev = layer.weight.device
        self.columns = layer.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t().float()

        if self.method == "admm":
            self.H += inp.matmul(inp.t())
        else:
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(2 / self.nsamples) * inp
            self.H += inp.matmul(inp.t())
            return

        self.nsamples += tmp

    def fasterquant(self, maxq=7, sym=True, mse=False, groupsize=-1, label=''):
        W = self.layer.weight.data.clone().float()
        layer_dtype = self.layer.weight.data.dtype

        if self.method == "admm":
            Q, _ = ADMM_quantize(
                W, self.H.clone(),
                maxq=maxq, sym=sym,
                rho=0.1, max_iter=300, update_iter=3, switch_iter=30,
                mse=mse, groupsize=groupsize, label=label,
            )
        else:
            Q, _ = GPTQ_quantize(
                W, self.H.clone(), blocksize=128,
                maxq=maxq, sym=sym,
                mse=mse, groupsize=groupsize, label=label,
            )

        self.layer.weight.data = Q.to(layer_dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Qwen3 layer structure (no .module suffix, unlike LLaMA reference)
# ---------------------------------------------------------------------------

_QWEN3_SEQ = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


def _find_linears(module, name=""):
    """Return {dotted_name: nn.Linear} for all Linear sub-modules."""
    if isinstance(module, nn.Linear):
        return {name: module}
    res = {}
    for cname, child in module.named_children():
        full = f"{name}.{cname}" if name else cname
        res.update(_find_linears(child, full))
    return res


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def _quantize_weights_calibrated(model, dataloader, dev, method, *,
                                  w_bits=4, w_sym=True, w_clip=True, w_groupsize=-1,
                                  nsamples=128, sequential=None):
    """Calibrated weight quantization (ADMM or GPTQ), layer-wise."""
    label = method.upper()
    log.info("-----%s Quantization-----", label)
    if sequential is None:
        sequential = _QWEN3_SEQ

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move embedding + norm to device for input capture
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'rotary_emb'):
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
            # Capture position_embeddings if passed (Qwen3 rotary)
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
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    maxq = 2 ** (w_bits - 1) - 1 if w_sym else 2 ** w_bits - 1

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = _find_linears(layer)

        # Build forward kwargs
        fwd_kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
        if extra_kwargs.get("position_embeddings") is not None:
            fwd_kwargs["position_embeddings"] = extra_kwargs["position_embeddings"]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}

            wrappers = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                wrappers[name] = QuantWrapper(subset[name], method=method)

            def add_batch(name):
                def tmp(_, inp, out):
                    wrappers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrappers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **fwd_kwargs)[0]
            for h in handles:
                h.remove()

            for name in wrappers:
                wrappers[name].fasterquant(
                    maxq=maxq, sym=w_sym, mse=w_clip, groupsize=w_groupsize,
                    label=f'L{i}.{name}',
                )
                wrappers[name].free()

        # Forward pass with quantized weights to get inputs for next layer
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kwargs)[0]

        layers[i] = layer.cpu()
        del layer, wrappers
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    gc.collect()
    torch.cuda.empty_cache()
    log.info("-----%s Quantization Done-----", label)


def quantize_weights_admm(model, dataloader, dev="cuda", **kwargs):
    """ADMM weight quantization (layer-wise, calibrated)."""
    _quantize_weights_calibrated(model, dataloader, dev, "admm", **kwargs)


def quantize_weights_gptq(model, dataloader, dev="cuda", **kwargs):
    """GPTQ weight quantization (layer-wise, calibrated)."""
    _quantize_weights_calibrated(model, dataloader, dev, "gptq", **kwargs)


@torch.no_grad()
def quantize_weights_rtn(model, dev="cuda", *,
                         w_bits=4, w_sym=True, w_clip=True, w_groupsize=-1,
                         sequential=None):
    """RTN weight quantization (no calibration needed)."""
    log.info("-----RTN Quantization-----")
    if sequential is None:
        sequential = _QWEN3_SEQ

    layers = model.model.layers
    torch.cuda.empty_cache()
    maxq = 2 ** (w_bits - 1) - 1 if w_sym else 2 ** w_bits - 1

    for i in range(len(layers)):
        print(f"\rRTN Layer {i}/{len(layers)}", end="", flush=True)
        layer = layers[i].to(dev)
        full = _find_linears(layer)

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            for name, linear in subset.items():
                W = linear.weight.data.float()
                scale_W, zero_W = compute_scale_zero(
                    W, maxq, sym=w_sym, mse=w_clip, groupsize=w_groupsize,
                )
                Q = fake_quant(W, scale_W, zero_W, maxq, w_sym)
                linear.weight.data = Q.to(linear.weight.dtype)

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    print()
    gc.collect()
    torch.cuda.empty_cache()
    log.info("-----RTN Quantization Done-----")
