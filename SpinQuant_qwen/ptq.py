# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader_wiki = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    wiki_ppl = eval_utils.evaluator(model, testloader_wiki, utils.DEV, ptq_args)
    log.info("WikiText-2 PPL: {:.3f}".format(wiki_ppl))

    testloader_c4 = data_utils.get_c4(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    c4_ppl = eval_utils.evaluator(model, testloader_c4, utils.DEV, ptq_args)
    log.info("C4 PPL: {:.3f}".format(c4_ppl))

    testloader_ptb = data_utils.get_ptb(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    ptb_ppl = eval_utils.evaluator(model, testloader_ptb, utils.DEV, ptq_args)
    log.info("PTB PPL: {:.3f}".format(ptb_ppl))

    if getattr(ptq_args, 'eval_zeroshot', False) and local_rank == 0:
        log.info("Starting 0-shot evaluation with lm-eval-harness...")
        try:
            from peft import PeftModel  # noqa: F401
        except ImportError:
            # peft may be broken on some nodes; stub it so lm_eval can still load
            import types, sys
            peft_stub = types.ModuleType("peft")
            peft_stub.PeftModel = None
            peft_stub.__version__ = "0.0.0"
            sys.modules["peft"] = peft_stub
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        # PPL evaluator leaves embed_tokens/norm/layers on CPU; move everything back
        model.cuda()

        batch_size = training_args.per_device_eval_batch_size
        lm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

        # ARC-easy, ARC-challenge, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande, RTE
        tasks = ["piqa", "arc_easy", "arc_challenge", "hellaswag",
                 "winogrande", "social_iqa", "openbookqa", "boolq", "rte"]
        results = lm_eval.simple_evaluate(
            model=lm_model,
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
        )
        task_accs = {}
        for task_name in tasks:
            task_result = results["results"][task_name]
            task_accs[task_name] = task_result["acc,none"]
            log.info(f"  {task_name}: {task_accs[task_name]:.4f}")
        avg_acc = sum(task_accs.values()) / len(task_accs)
        log.info(f"0-shot average accuracy: {avg_acc:.4f} ({avg_acc * 100:.2f}%)")

    dist.barrier()


if __name__ == "__main__":
    train()
