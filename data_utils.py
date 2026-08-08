"""Data loading utilities for evaluation and calibration."""

import random

import datasets
import torch
import transformers


def get_wikitext2(tokenizer, seqlen=2048, nsamples=128, seed=0, eval_mode=True):
    """Load WikiText-2 for evaluation (test) or calibration (train).

    Returns:
        eval_mode=True:  tokenizer output with .input_ids (1, text_len)
        eval_mode=False: list of (inp, tar) tuples, each (1, seqlen)
    """
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    if eval_mode:
        text = "\n\n".join(ds["test"]["text"])
        return tokenizer(text, return_tensors="pt")

    text = "\n\n".join(ds["train"]["text"])
    enc = tokenizer(text, return_tensors="pt")
    random.seed(seed)
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader


def get_c4(tokenizer, seqlen=2048, nsamples=128, seed=0, eval_mode=True):
    """Load C4 for evaluation (validation) or calibration (train).

    Returns: same format as get_wikitext2.
    """
    if eval_mode:
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        enc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        enc.input_ids = enc.input_ids[:, : (256 * seqlen)]
        return enc

    traindata = datasets.load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    random.seed(seed)
    loader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            enc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] > seqlen:
                break
        start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, start : start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader


def get_ptb(tokenizer, seqlen=2048, nsamples=128, seed=0, eval_mode=True):
    """Load Penn Treebank for evaluation (test) or calibration (train).

    Returns: same format as get_wikitext2.
    """
    if eval_mode:
        testdata = datasets.load_dataset(
            "ptb_text_only", "penn_treebank", split="test", trust_remote_code=True
        )
        text = " ".join(testdata["sentence"]).replace("<unk>", "")
        return tokenizer(text, return_tensors="pt")

    traindata = datasets.load_dataset(
        "ptb_text_only", "penn_treebank", split="train", trust_remote_code=True
    )
    text = " ".join(traindata["sentence"]).replace("<unk>", "")
    enc = tokenizer(text, return_tensors="pt")
    random.seed(seed)
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader
