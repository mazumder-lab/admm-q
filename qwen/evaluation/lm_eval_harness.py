"""Downstream task evaluation via lm-evaluation-harness.

Uses the 9 zero-shot tasks from SpinQuant evaluation protocol.
Returns per-task accuracy and average accuracy.
"""

import logging
import os

# Must be set before datasets is imported anywhere
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

log = logging.getLogger(__name__)

# 9 zero-shot tasks from SpinQuant evaluation protocol
DEFAULT_TASKS = [
    "piqa", "arc_easy", "arc_challenge", "hellaswag",
    "winogrande", "social_iqa", "openbookqa", "boolq", "rte",
]


def evaluate_downstream(model, tokenizer, tasks=None, batch_size=4, device="cuda"):
    """Run 0-shot evaluation with lm-eval-harness.

    Args:
        model: Pre-loaded transformers model.
        tokenizer: Pre-loaded tokenizer.
        tasks: List of task names, or None for all 9 default tasks.
        batch_size: Batch size for evaluation.
        device: Device string.

    Returns:
        dict: {
            "avg_acc": float,
            "per_task": {task_name: acc, ...}
        }
    """
    # Stub peft if not installed (lm-eval tries to import it)
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
        tasks = DEFAULT_TASKS

    log.info("Evaluating 0-shot tasks: %s", tasks)

    model.to(device)
    lm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=lm_model, tasks=tasks, num_fewshot=0, batch_size=batch_size,
    )

    model.cpu()
    torch.cuda.empty_cache()

    # Extract accuracy per task (prefer acc_norm if available)
    task_accs = {}
    for task_name in tasks:
        task_result = results["results"].get(task_name, {})
        acc_key = "acc_norm,none" if "acc_norm,none" in task_result else "acc,none"
        if acc_key in task_result:
            task_accs[task_name] = task_result[acc_key]
        else:
            task_accs[task_name] = None
            log.warning("No metric found for %s", task_name)

    valid_accs = [v for v in task_accs.values() if v is not None]
    avg_acc = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

    log.info("0-shot per-task: %s", {k: f"{v:.4f}" if v else "N/A" for k, v in task_accs.items()})
    log.info("0-shot average: %.4f (%.2f%%)", avg_acc, avg_acc * 100)

    return {"avg_acc": avg_acc, "per_task": task_accs}


# Need torch for cuda cache cleanup
import torch
