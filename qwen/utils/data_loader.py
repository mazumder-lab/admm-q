"""Calibration data loading utilities."""

import logging
import random

import torch
from datasets import load_dataset

log = logging.getLogger(__name__)


def get_calibration_dataloader(tokenizer, nsamples=128, seqlen=2048, seed=42, dataset="wikitext2"):
    """Load calibration data and return list of tokenized batches.

    Each batch is a tuple of (input_ids,) with shape (1, seqlen).
    """
    random.seed(seed)
    log.info("Loading calibration data: %s (nsamples=%d, seqlen=%d)", dataset, nsamples, seqlen)

    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(data["text"])
    elif dataset == "c4":
        data = load_dataset("allenai/c4", "en", split="train", streaming=True)
        text_chunks = []
        for i, example in enumerate(data):
            text_chunks.append(example["text"])
            if len(text_chunks) >= nsamples * 5:
                break
        text = "\n\n".join(text_chunks)
    else:
        raise ValueError(f"Unknown calibration dataset: {dataset}")

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    dataloader = []
    total_tokens = input_ids.shape[0]
    for _ in range(nsamples):
        start = random.randint(0, total_tokens - seqlen - 1)
        inp = input_ids[start : start + seqlen].unsqueeze(0)
        dataloader.append((inp,))

    log.info("Calibration data ready: %d samples of length %d", nsamples, seqlen)
    return dataloader
