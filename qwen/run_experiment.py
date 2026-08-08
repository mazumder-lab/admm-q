"""Main entry point for Qwen3 quantization experiments."""

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import argparse
import json
import logging
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directories to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from quantization.pipeline import quantize_weights_admm, quantize_weights_gptq, quantize_weights_rtn
from utils.data_loader import get_calibration_dataloader
from evaluation.perplexity import evaluate_perplexity
from evaluation.lm_eval_harness import evaluate_downstream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Quantization Experiment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--method", type=str, required=True, choices=["dense", "rtn", "gptq", "admm"],
                        help="Quantization method")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4], help="Bit width")
    parser.add_argument("--groupsize", type=int, default=-1, help="Group size (-1 for per-column)")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")
    parser.add_argument("--sym", action="store_true", default=True, help="Symmetric quantization")
    parser.add_argument("--no_clip", action="store_true", default=False, help="Disable MSE clipping")
    parser.add_argument("--output_dir", type=str, default=None, help="Save quantized model here")
    parser.add_argument("--results_dir", type=str, default=None, help="Save results JSON here")
    parser.add_argument("--eval_ppl", type=str, default="wikitext2,ptb,c4",
                        help="Comma-separated ppl datasets (wikitext2,ptb,c4)")
    parser.add_argument("--eval_downstream", type=str, default=None,
                        help="Comma-separated downstream tasks (or 'all' for 9 zero-shot tasks)")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--smooth", action="store_true", default=False,
                        help="Enable SmoothQuant preprocessing before quantization")
    parser.add_argument("--smooth_alpha", type=float, default=0.5,
                        help="SmoothQuant migration strength (0=all weight, 1=all activation)")
    parser.add_argument("--act_bits", type=int, default=None,
                        help="Activation bit width (e.g. 4 for W4A4). None = no activation quantization")
    parser.add_argument("--act_per_token", action="store_true", default=True,
                        help="Per-token activation quantization (default True)")
    parser.add_argument("--spin", action="store_true", default=False,
                        help="Enable SpinQuant (random rotation) preprocessing before quantization")
    parser.add_argument("--kv_bits", type=int, default=None,
                        help="KV cache quantization bit width (e.g. 8 for KV8). Requires --spin")
    args = parser.parse_args()

    if args.spin and args.smooth:
        parser.error("--spin and --smooth are mutually exclusive")
    if args.kv_bits is not None and not args.spin:
        parser.error("--kv_bits requires --spin")

    log.info("=" * 60)
    log.info("Experiment: method=%s, bits=%d, model=%s", args.method, args.bits, args.model_path)
    log.info("=" * 60)

    # --- Load model and tokenizer ---
    log.info("Loading model from %s ...", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    model_name = os.path.basename(args.model_path)
    log.info("Model loaded: %s (%d layers, hidden=%d)",
             model_name, len(model.model.layers), model.config.hidden_size)

    # --- Load calibration data (needed for smooth, gptq, admm) ---
    dataloader = None
    needs_calib = args.method in ("gptq", "admm") or (args.smooth and args.method != "dense")
    if needs_calib:
        log.info("Loading calibration data...")
        dataloader = get_calibration_dataloader(
            tokenizer, nsamples=args.nsamples, seqlen=args.seqlen,
        )

    # --- SpinQuant preprocessing ---
    spin_handles = None
    if args.spin and args.method != "dense":
        from quantization.spinquant import spin_model
        log.info("SpinQuant enabled (kv_bits=%s)", args.kv_bits)
        spin_handles = spin_model(model, kv_bits=args.kv_bits)

    # --- SmoothQuant preprocessing ---
    if args.smooth and args.method != "dense":
        from quantization.smooth import collect_act_scales, smooth_model
        log.info("SmoothQuant enabled (alpha=%.2f)", args.smooth_alpha)
        act_scales = collect_act_scales(model, dataloader, args.device, nsamples=args.nsamples)
        smooth_model(model, act_scales, alpha=args.smooth_alpha)
        del act_scales

    # --- Quantize ---
    t0 = time.time()
    if args.method == "dense":
        log.info("Dense baseline, skipping quantization.")
    elif args.method == "rtn":
        quantize_weights_rtn(
            model, args.device,
            w_bits=args.bits, w_sym=args.sym, w_clip=not args.no_clip,
            w_groupsize=args.groupsize,
        )
    elif args.method in ("gptq", "admm"):
        if args.method == "gptq":
            quantize_weights_gptq(
                model, dataloader, args.device,
                w_bits=args.bits, w_sym=args.sym, w_clip=not args.no_clip,
                w_groupsize=args.groupsize, nsamples=args.nsamples,
            )
        else:
            quantize_weights_admm(
                model, dataloader, args.device,
                w_bits=args.bits, w_sym=args.sym, w_clip=not args.no_clip,
                w_groupsize=args.groupsize, nsamples=args.nsamples,
            )
    quant_time = time.time() - t0
    log.info("Quantization time: %.1f seconds", quant_time)

    # --- Save quantized model ---
    if args.output_dir and args.method != "dense":
        save_path = os.path.join(args.output_dir, f"{model_name}-{args.method}-w{args.bits}")
        log.info("Saving quantized model to %s ...", save_path)
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log.info("Model saved.")

    # --- Activation quantization hooks (for WxAx evaluation) ---
    act_quant_handles = None
    if args.act_bits is not None:
        from quantization.act_quant import add_act_quant_hooks
        act_quant_handles = add_act_quant_hooks(
            model, bits=args.act_bits, sym=args.sym, per_token=args.act_per_token,
        )

    # --- Evaluate perplexity ---
    ppl_results = {}
    if args.eval_ppl:
        ppl_datasets = [d.strip() for d in args.eval_ppl.split(",")]
        for ds in ppl_datasets:
            ppl = evaluate_perplexity(model, tokenizer, ds, device=args.device,
                                     seqlen=args.seqlen, batch_size=args.batch_size)
            ppl_results[ds] = ppl

    # --- Evaluate downstream ---
    downstream_results = {}
    if args.eval_downstream:
        if args.eval_downstream == "all":
            tasks = None
        else:
            tasks = [t.strip() for t in args.eval_downstream.split(",")]
        downstream_results = evaluate_downstream(
            model, tokenizer, tasks=tasks, batch_size=args.batch_size, device=args.device,
        )

    # --- Remove activation quantization hooks ---
    if act_quant_handles is not None:
        from quantization.act_quant import remove_act_quant_hooks
        remove_act_quant_hooks(act_quant_handles)

    # --- Remove SpinQuant R4 hooks ---
    if spin_handles is not None:
        for h in spin_handles:
            h.remove()
        log.info("Removed %d SpinQuant R4 hooks.", len(spin_handles))

    # --- Save results ---
    # Parse model name parts
    parts = model_name.split("-")
    variant = "base" if "base" in model_name else "instruct"
    model_short = "-".join(p for p in parts if p not in ("base", "instruct"))

    result = {
        "model": model_short,
        "variant": variant,
        "method": args.method,
        "smooth": args.smooth,
        "smooth_alpha": args.smooth_alpha if args.smooth else None,
        "spin": args.spin,
        "kv_bits": f"KV{args.kv_bits}" if args.kv_bits else None,
        "bits": f"W{args.bits}" if args.method != "dense" else "FP16",
        "act_bits": f"A{args.act_bits}" if args.act_bits else None,
        "group_size": args.groupsize,
        "sym": args.sym,
        "nsamples": args.nsamples,
        "quant_time_seconds": round(quant_time, 1),
        "perplexity": ppl_results,
        "downstream": downstream_results,
    }

    log.info("=" * 60)
    log.info("Results: %s", json.dumps(result, indent=2))
    log.info("=" * 60)

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        w_str = f"w{args.bits}" if args.method != "dense" else "fp16"
        a_str = f"a{args.act_bits}" if args.act_bits else ""
        kv_str = f"kv{args.kv_bits}" if args.kv_bits else ""
        smooth_str = "smooth_" if args.smooth else ""
        spin_str = "spin_" if args.spin else ""
        fname = f"{model_name}_{spin_str}{smooth_str}{args.method}_{w_str}{a_str}{kv_str}.json"
        result_path = os.path.join(args.results_dir, fname)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Results saved to %s", result_path)


if __name__ == "__main__":
    main()
