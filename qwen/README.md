# Qwen3 Quantization Pipeline

Post-training quantization for Qwen3 models (0.6B--8B) with RTN, GPTQ, and ADMM-Q.
Supports SmoothQuant and SpinQuant preprocessing, activation quantization (WxAx),
and KV cache quantization.

## Structure

```
run_experiment.py              # Main entry point
quantization/
    solvers.py                 # ADMM, GPTQ, PairSwap solvers (model-agnostic)
    pipeline.py                # Layer-wise weight quantization for Qwen3
    smooth.py                  # SmoothQuant preprocessing
    spinquant.py               # SpinQuant (random rotation) preprocessing
    act_quant.py               # Activation quantization hooks (fake quant)
evaluation/
    perplexity.py              # WikiText-2, PTB, C4 perplexity (layer-by-layer)
    lm_eval_harness.py         # 9-task zero-shot evaluation via lm-eval
utils/
    data_loader.py             # Calibration data loading (WikiText-2 train)
```

## Usage

```bash
# Dense baseline
python run_experiment.py --model_path <model> --method dense \
    --eval_ppl wikitext2,ptb,c4 --eval_downstream all

# ADMM W4
python run_experiment.py --model_path <model> --method admm --bits 4 \
    --eval_ppl wikitext2,ptb,c4 --eval_downstream all

# SmoothQuant + ADMM W4A8
python run_experiment.py --model_path <model> --method admm --bits 4 \
    --smooth --act_bits 8 --eval_ppl wikitext2,ptb,c4

# SpinQuant + ADMM W4A4KV4
python run_experiment.py --model_path <model> --method admm --bits 4 \
    --spin --act_bits 4 --kv_bits 4 --eval_ppl wikitext2,ptb,c4
```

## Methods

| Flag | Description |
|------|-------------|
| `--method rtn/gptq/admm` | Weight quantization method |
| `--bits 2/3/4` | Weight bit width |
| `--smooth` | SmoothQuant preprocessing (mutually exclusive with --spin) |
| `--spin` | SpinQuant preprocessing (mutually exclusive with --smooth) |
| `--act_bits N` | Activation quantization (e.g. 4 or 8) |
| `--kv_bits N` | KV cache quantization (requires --spin) |

## Evaluation

- **PPL**: WikiText-2 (test), PTB (test), C4 (validation) -- layer-by-layer forward
- **Zero-shot**: 9 tasks (piqa, arc_easy, arc_challenge, hellaswag, winogrande,
  social_iqa, openbookqa, boolq, rte) via lm-eval-harness

## Dependencies

- PyTorch 2.6.0+cu124
- transformers 5.2.0
- datasets < 3 (for social_iqa compatibility)
- lm-eval 0.4.11
- scipy, tqdm
