"""Perplexity evaluation on WikiText-2, PTB, and C4.

Uses layer-by-layer forward pass for memory efficiency (no need to fit
entire model on GPU at once). Based on admmq_colm26/eval_utils.py.
"""

import logging

import datasets
import torch
import torch.nn as nn
from tqdm import tqdm

log = logging.getLogger(__name__)


def _get_test_enc(tokenizer, dataset_name, seqlen=2048):
    """Load and tokenize evaluation dataset.

    Returns:
        tokenizer output with .input_ids (1, text_len)
    """
    if dataset_name == "wikitext2":
        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        text = "\n\n".join(ds["test"]["text"])
        return tokenizer(text, return_tensors="pt")
    elif dataset_name == "ptb":
        ds = datasets.load_dataset(
            "ptb_text_only", "penn_treebank", split="test", trust_remote_code=True,
        )
        text = " ".join(ds["sentence"]).replace("<unk>", "")
        return tokenizer(text, return_tensors="pt")
    elif dataset_name == "c4":
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        enc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        enc.input_ids = enc.input_ids[:, : (256 * seqlen)]
        return enc
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _unwrap_layer_output(out):
    """Handle both tuple returns (LLaMA-style) and tensor returns (Qwen3-style)."""
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset_name, device="cuda",
                        seqlen=2048, batch_size=4):
    """Compute perplexity with layer-by-layer forward pass.

    Memory efficient: only one transformer layer on GPU at a time.

    Returns:
        float: perplexity value
    """
    log.info("Evaluating perplexity on %s ...", dataset_name)

    testenc = _get_test_enc(tokenizer, dataset_name, seqlen=seqlen)
    input_ids = testenc.input_ids

    nsamples = input_ids.numel() // seqlen
    if nsamples == 0:
        log.warning("Not enough tokens for perplexity evaluation on %s", dataset_name)
        return float("inf")

    input_ids = input_ids[:, : nsamples * seqlen].view(nsamples, seqlen).to(device)
    batches = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(batches)

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move embedding to device
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    # Capture first-layer inputs
    dtype = next(iter(model.parameters())).dtype
    inps = [0] * nbatches
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
    for batch in batches:
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = [0] * nbatches
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    # Build fwd_kwargs — position_embeddings has batch dim 1, broadcasts to any batch size
    fwd_kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
    if extra_kwargs.get("position_embeddings") is not None:
        fwd_kwargs["position_embeddings"] = extra_kwargs["position_embeddings"]

    # Process layer by layer
    for i in tqdm(range(len(layers)), desc=f"PPL({dataset_name})"):
        layer = layers[i].to(device)
        for j in range(nbatches):
            # Qwen3 decoder layer returns a tensor (not a tuple like LLaMA),
            # so use _unwrap_layer_output instead of [0]
            outs[j] = _unwrap_layer_output(layer(inps[j], **fwd_kwargs))
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Final norm + lm_head -> logits -> PPL
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    nlls = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = batches[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)

    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    torch.cuda.empty_cache()

    ppl = torch.exp(torch.cat(nlls).mean()).item()
    model.config.use_cache = use_cache
    log.info("%s perplexity: %.4f (nsamples=%d)", dataset_name, ppl, nsamples)
    return ppl
