"""Evaluation utilities: WikiText-2/C4/PTB perplexity and 0-shot accuracy.

Usage:
    from admmq_colm26.eval_utils import ppl_eval, zeroshot_eval

    wiki_ppl = ppl_eval(model, testenc, seqlen=2048, batch_size=4)
    avg_acc, per_task = zeroshot_eval(model, tokenizer, batch_size=4)
"""

import logging
import os

import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def ppl_eval(model, testenc, seqlen=2048, batch_size=4, dev=None):
    """Compute perplexity on a tokenized eval dataset.

    Args:
        model: LlamaForCausalLM (or compatible) with .model.layers, .model.norm, .lm_head
        testenc: tokenizer output with .input_ids (1, text_len)
        seqlen: sequence length for chunking
        batch_size: eval batch size
        dev: device (defaults to cuda)

    Returns:
        float: perplexity
    """
    if dev is None:
        dev = DEV
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Move embedding to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    # Chunk text into sequences
    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // seqlen
    input_ids = input_ids[:, : nsamples * seqlen].view(nsamples, seqlen).to(dev)
    batches = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(batches)

    # Capture first-layer inputs via hook
    dtype = next(iter(model.parameters())).dtype
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
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
    for batch in batches:
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    position_ids = cache["position_ids"]
    torch.cuda.empty_cache()

    outs = [0] * nbatches
    attention_mask = cache["attention_mask"]

    # Process layer by layer (memory efficient)
    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)
        for j in range(nbatches):
            outs[j] = layer(
                inps[j], attention_mask=attention_mask, position_ids=position_ids
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Final norm + lm_head → logits → PPL
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
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

    ppl = torch.exp(torch.cat(nlls).mean()).item()
    model.config.use_cache = use_cache
    log.info("PPL: %.3f", ppl)
    return ppl


def zeroshot_eval(model, tokenizer, batch_size=4, tasks=None):
    """Run 0-shot evaluation with lm-eval-harness.

    Args:
        model: the (quantized) model on GPU
        tokenizer: matching tokenizer
        batch_size: eval batch size
        tasks: list of task names (defaults to the 9 SpinQuant tasks)

    Returns:
        (avg_acc, dict[task_name, acc])
    """
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    try:
        from peft import PeftModel  # noqa: F401
    except ImportError:
        import sys
        import types
        peft_stub = types.ModuleType("peft")
        peft_stub.PeftModel = None
        peft_stub.__version__ = "0.0.0"
        sys.modules["peft"] = peft_stub

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if tasks is None:
        tasks = [
            "piqa", "arc_easy", "arc_challenge", "hellaswag",
            "winogrande", "social_iqa", "openbookqa", "boolq", "rte",
        ]

    model.cuda()
    lm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=lm_model, tasks=tasks, num_fewshot=0, batch_size=batch_size,
    )

    task_accs = {}
    for task_name in tasks:
        task_result = results["results"][task_name]
        acc_key = "acc_norm,none" if "acc_norm,none" in task_result else "acc,none"
        task_accs[task_name] = task_result[acc_key]
        log.info("  %s: %.4f", task_name, task_accs[task_name])

    avg_acc = sum(task_accs.values()) / len(task_accs)
    log.info("0-shot average: %.4f (%.2f%%)", avg_acc, avg_acc * 100)
    return avg_acc, task_accs
