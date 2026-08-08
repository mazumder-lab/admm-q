# coding=utf-8
# ADMM weight quantization integration for SpinQuant.
#
# Drop-in replacement for the GPTQ path: same calibration data, same Hessian
# collection, same layer-by-layer processing — only the core quantizer differs.
#
# Usage in main.py:
#     from eval_utils import admmq_utils
#     quantizers = admmq_utils.admm_fwrd(model, trainloader, "cuda", args)

import logging
import math
import os
import sys

import torch
import torch.nn as nn
import tqdm

from utils import quant_utils, utils

# ---------------------------------------------------------------------------
# Import ADMM solver from admmq_colm26 package
# ---------------------------------------------------------------------------
_admmq_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "admmq--colm26",
)
if _admmq_path not in sys.path:
    sys.path.insert(0, _admmq_path)

from admmq_colm26.solvers import ADMM_quantize  # noqa: E402


class ADMMQuant:
    """Layer wrapper that collects the Hessian and runs ADMM quantization.

    __init__, add_batch, and free are identical to the GPTQ class so that
    both quantizers share the exact same calibration pipeline.
    Only fasterquant differs (ADMM + pairswap instead of greedy GPTQ).
    """

    # ---- identical to GPTQ ------------------------------------------------
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)
    # -----------------------------------------------------------------------

    def fasterquant(self, maxq, sym=True, mse=False, groupsize=-1, label=""):
        """Run ADMM + pairswap quantization on this layer's weights."""
        W = self.layer.weight.data.clone().float()
        layer_dtype = self.layer.weight.data.dtype

        Q, recon = ADMM_quantize(
            W,
            self.H,
            maxq=maxq,
            sym=sym,
            rho=0.1,
            max_iter=300,
            update_iter=3,
            switch_iter=30,
            mse=mse,
            groupsize=groupsize,
            label=label,
        )

        self.layer.weight.data = Q.to(layer_dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in ADMM quantized weights")
            raise ValueError("NaN in ADMM quantized weights")


@torch.no_grad()
def admm_fwrd(model, dataloader, dev, args):
    """ADMM weight quantization — drop-in replacement for gptq_fwrd.

    Same calibration, same layer ordering, same Hessian collection.
    Only the weight quantizer (fasterquant) differs.
    """
    logging.info("-----ADMM Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # --- Capture first-layer inputs (identical to gptq_fwrd) ---------------
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
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
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    # --- Layer-by-layer quantization ---------------------------------------
    quantizers = {}
    sym = not args.w_asym

    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for names in sequential:
            subset = {n: full[n] for n in names}

            # Create ADMM wrappers and collect Hessians
            admm = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                admm[name] = ADMMQuant(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    admm[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            for h in handles:
                h.remove()

            # Quantize each sublayer
            for name in subset:
                layer_weight_bits = args.w_bits
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                # Use quant_utils.get_minq_maxq for maxq (shared with GPTQ/RTN)
                _, layer_maxq = quant_utils.get_minq_maxq(layer_weight_bits, sym)
                layer_maxq = int(layer_maxq.item())

                admm[name].fasterquant(
                    maxq=layer_maxq,
                    sym=sym,
                    mse=args.w_clip,
                    groupsize=args.w_groupsize,
                    label=f"L{i}/{name.split('.')[-2]}",
                )
                admm[name].free()

        # Re-run layer with quantized weights for next layer's inputs
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del admm
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----ADMM Quantization Done-----\n")
    return quantizers
