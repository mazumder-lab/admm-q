# ADMM-Q

Code for our COLM 2026 paper:

> **ADMM-Q: An Improved Hessian-based Weight Quantizer for Post-Training Quantization of Large Language Models**
> Ryan Lucas, Mehdi Makni, Xiang Meng, Adam Deng, Rahul Mazumder
> [arXiv:2605.11222](https://arxiv.org/abs/2605.11222)

ADMM-Q is a weight quantizer for post-training quantization (PTQ) of LLMs. It solves the same layer-wise problem as GPTQ, but with a different algorithm, and it does noticeably better at low bit widths. It is meant to be used as a drop-in replacement for GPTQ (or RTN) inside an existing quantization pipeline, so it works alongside approaches like weight clipping, SmoothQuant and SpinQuant rotations.

There are two implementations in this repo:

- **PyTorch** (top level, `qwen/`, `SpinQuant_qwen/`): the code behind the experiments in the paper.
- **JAX / Qwix** (`qwix_admmq/`): a port of ADMM-Q to [Qwix](https://github.com/google/qwix), Google's JAX quantization library, written by **Adam Deng**. It plugs into Qwix the same way Qwix's own GPTQ does. See [JAX / Qwix implementation](#jax--qwix-implementation) below.

## What problem this solves

Most PTQ methods quantize a model one linear layer at a time. For a layer with weights `W` and calibration inputs `X`, the goal is to find quantized weights `Ŵ` that keep the layer's output close to the original:

```
minimize    || W X - Ŵ X ||²  =  tr( (W - Ŵ) H (W - Ŵ)ᵀ ),   H = X Xᵀ
subject to  every entry of Ŵ lies on the quantization grid
```

`H` is (up to a constant) the Hessian of this loss, which is why these are called Hessian-based quantizers. The constraint makes the problem combinatorial, so nobody solves it exactly.

- **RTN** ignores `H` and just rounds each weight to the nearest grid point.
- **GPTQ** goes through the columns once, rounding one column and pushing the error onto the columns that haven't been quantized yet. It is fast, but every rounding decision is final, and at 3 bits and below the errors pile up.

ADMM-Q instead keeps a continuous copy of the weights and keeps updating it for many iterations, while slowly forcing it onto the grid. Earlier decisions can be undone if they turn out to be bad.

## How ADMM-Q works

We split the weights into two copies: `B`, which is continuous, and `D`, which must be on the grid, and add the constraint `B = D`. Then we run a version of the Alternating Direction Method of Multipliers (ADMM) on the split problem. Each iteration does three cheap steps:

1. **D-step (projection).** Round `B + V/ρ` onto the quantization grid.
2. **B-step (least squares).** Solve `(H + ρI) B = H Wᵀ + ρD - V`, which pulls `B` toward lower reconstruction error while staying near `D`. (`B` and `D` are stored transposed, as `in_features × out_features`.)
3. **Dual step.** `V ← V + ρ(B - D)`.

Here `V` is the dual variable and `ρ` is the penalty. Early on, `ρ` is small and `B` is free to move to a good region. We increase `ρ` on a schedule (by default ×1.1 every 3 iterations), which slowly forces `B` and `D` to agree. When they stop changing, we stop. The paper gives convergence guarantees for this scheme.

A few extra pieces make it practical at LLM scale (all in `solvers.py`):

- **Preconditioning.** `H` is rescaled to have a unit diagonal before ADMM starts. This keeps the problem well behaved across input channels with very different scales.
- **One eigendecomposition per layer.** We compute `H = Q Λ Qᵀ` once (in float64). After that, `(H + ρI)⁻¹ = Q (Λ + ρI)⁻¹ Qᵀ` is cheap to rebuild every time `ρ` changes.
- **Grid refresh on stagnation.** If the error stops improving, we try recomputing the scale/zero point from the current iterate once. We only keep the new grid if it actually lowers the Hessian-weighted error.
- **Local search for hard layers.** If `H` is badly conditioned (condition number above 5e4), we use a smaller starting `ρ` and a slower schedule, and then finish with a greedy pair-swap search (`pairswap_batched`) that tries moving pairs of weights to neighboring grid points and keeps the moves that help.

The rounding, clipping and scale/zero logic is copied exactly from SpinQuant's `WeightQuantizer`, so RTN, GPTQ and ADMM-Q in this repo all quantize onto exactly the same grid. Any difference in the results comes from the solver.

## Results

Headline numbers from the paper, for Qwen3-8B, WikiText-2 perplexity (lower is better), with ADMM-Q swapped in for GPTQ and nothing else changed:

| Setting | Pipeline | GPTQ | ADMM-Q |
|---|---|---|---|
| W3A16 | weight-only | 12.85 | **10.06** |
| W4A8 | SmoothQuant | 9.29 | **8.68** |
| W2A4KV4 | SpinQuant | 66.11 | **19.42** |

See the paper for the full tables (Qwen3 0.6B to 8B, PTB, C4 and 9 zero-shot tasks).

## What's in this repo

```
solvers.py           The algorithms: ADMM_quantize, GPTQ_quantize, pairswap_batched,
                     and the quantization primitives. No model-specific code.
data_utils.py        WikiText-2 / C4 / PTB loading for calibration and evaluation
eval_utils.py        Perplexity and zero-shot evaluation helpers

qwen/                Our own end-to-end pipeline for Qwen3 (0.6B to 8B):
                     RTN / GPTQ / ADMM-Q, optional SmoothQuant or SpinQuant
                     preprocessing, activation and KV cache quantization.
                     See qwen/README.md.

SpinQuant_qwen/      A fork of Meta's SpinQuant with ADMM-Q added as a weight
                     quantizer (--w_admm). Used for the rotation (W4A4KV4,
                     W2A4KV4) experiments.

qwix_admmq/          ADMM-Q for JAX, as a Qwix calibration algorithm
                     (Adam Deng). Has its own README.md.
```

`qwen/quantization/solvers.py` is a copy of the top-level `solvers.py`. If you change one, change the other.

## Installation

We used:

- Python 3.10+, PyTorch 2.6.0 (CUDA 12.4)
- transformers 5.2.0
- datasets < 3 (newer versions break `social_iqa` in lm-eval)
- lm-eval 0.4.11
- scipy, tqdm

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.2.0" "datasets<3" "lm-eval==0.4.11" scipy tqdm
```

The SpinQuant experiments also need `fast-hadamard-transform` and the packages in `SpinQuant_qwen/requirement.txt`. See `SpinQuant_qwen/README.md`.

The JAX / Qwix version doesn't need PyTorch at all. It needs JAX, `flax>=0.12`, `absl-py` and a checkout of [Qwix](https://github.com/google/qwix) on your `PYTHONPATH`.

## Usage

### Using the solver directly

The solver only needs a weight matrix and its `XᵀX` matrix, so it's easy to plug into your own code. Collect `H` the same way GPTQ does (a forward hook that accumulates `xᵀx` over the calibration tokens), then call:

```python
from solvers import ADMM_quantize   # run from the repo root

# W: (out_features, in_features), H: (in_features, in_features)
bits = 4
maxq = 2 ** (bits - 1) - 1          # symmetric grid; use 2**bits - 1 with sym=False

W_q, err = ADMM_quantize(W, H, maxq=maxq, sym=True, groupsize=-1)
layer.weight.data = W_q.to(layer.weight.dtype)
```

`W_q` is the dequantized ("fake quantized") weight, already on the grid, and `err` is the Hessian-weighted reconstruction error `tr((W - W_q) H (W - W_q)ᵀ)`. Use `groupsize=128` for group-wise quantization. `GPTQ_quantize` takes the same arguments if you want a direct comparison.

### Qwen3 pipeline

```bash
cd qwen

# ADMM-Q, 3-bit weights only
python run_experiment.py --model_path Qwen/Qwen3-8B --method admm --bits 3 \
    --eval_ppl wikitext2,ptb,c4 --eval_downstream all

# SmoothQuant + ADMM-Q, W4A8
python run_experiment.py --model_path Qwen/Qwen3-8B --method admm --bits 4 \
    --smooth --act_bits 8 --eval_ppl wikitext2,ptb,c4

# SpinQuant + ADMM-Q, W4A4KV4
python run_experiment.py --model_path Qwen/Qwen3-8B --method admm --bits 4 \
    --spin --act_bits 4 --kv_bits 4 --eval_ppl wikitext2,ptb,c4
```

Swap `--method admm` for `gptq` or `rtn` to get the baselines with the same setup. Calibration uses 128 random 2048-token windows from the WikiText-2 training set by default. All flags are listed in `qwen/README.md`.

### SpinQuant with learned rotations

Inside `SpinQuant_qwen/`, the workflow is the same as upstream SpinQuant: first learn the rotation, then run PTQ with it. To use ADMM-Q, add `--w_admm` to the PTQ command. Ready-made Slurm scripts for Llama-3.2-1B-Instruct are in `SpinQuant_qwen/replication/llama3_1B_instruct/slurm/` (`rot_*.sbatch` learns the rotation, `eval_{rtn,gptq,admm}_*.sbatch` evaluate each quantizer with it).

## JAX / Qwix implementation

*Author: Adam Deng*

Adam Deng also implemented ADMM-Q in JAX, as an add-on for [Qwix](https://github.com/google/qwix), Google's quantization library for JAX/Flax models. It lives in `qwix_admmq/`. It is written to mirror Qwix's own GPTQ (`qwix.contrib.gptq`) file for file, so anyone who already uses GPTQ in Qwix can switch to ADMM-Q by changing one import:

| File | Mirrors in Qwix | Contents |
|---|---|---|
| `admmq_core.py` | `qwix/contrib/gptq_core.py` | `quantize_weight(W, H, how)`: the ADMM-Q solver, on Qwix's grid |
| `admmq.py` | `qwix/contrib/gptq.py` | `AdmmqRule`, `AdmmqCalibrationProvider`, `quantize_params` |
| `admmq_test.py` | `qwix/tests/contrib/gptq_test.py` | unit and end-to-end tests (Linen, NNX, einsum) |

The workflow is the usual Qwix three steps: run calibration batches to collect `X Xᵀ` for each weight, call `admmq.quantize_params` to quantize the weights with ADMM-Q, then serve the model with Qwix's `PtqProvider`.

```python
import jax.numpy as jnp, qwix
from qwix_admmq import admmq

rules = [admmq.AdmmqRule(module_path='.*', weight_qtype=jnp.int4)]
cal_model = qwix.quantize_model(model, admmq.AdmmqCalibrationProvider(rules))
# ... run calibration batches, then admmq.quantize_params(...), then serve
```

The full example is in `qwix_admmq/README.md`.

The algorithm is the same one described above (scale search, ADMM with one float64 eigendecomposition per weight, the guarded grid refresh, and PairSwap on ill-conditioned layers). The differences come from following Qwix's conventions instead of SpinQuant's:

- It uses Qwix's quantization grid and calibration. For int4 the codes are in [-8, 7] and symmetric scales are absmax / 7.5. Any `weight_calibration_method` and `tile_size` that Qwix supports also work.
- It supports integer types (`int2` to `int8`). Other types (fp8, nf4, MX formats) fall back to Qwix's plain PTQ with a warning.
- float64 and full-precision float32 matmuls are turned on only inside the solver, so it doesn't change JAX settings for the rest of your program. (By default JAX may use TF32 on NVIDIA GPUs, which made the results slightly worse.)

All 13 tests pass. In those tests, on random 128×512 layers, ADMM-Q's reconstruction error was 31% lower than Qwix's GPTQ with per-channel scales, 19% lower with group size 128, and 58% lower on an ill-conditioned layer. These are small synthetic checks, not model-level results.

One caveat: Qwix's own GPTQ `gptq_core.quantize_weight` (as of commit 67ff467) crashes on asymmetric rules, so the comparison above uses symmetric rules only. Details are in `qwix_admmq/README.md`.

To run the tests:

```bash
PYTHONPATH=/path/to/qwix:/path/to/this/repo python qwix_admmq/admmq_test.py
```

## Practical notes

- **Cost.** ADMM-Q does more work per layer than GPTQ (one eigendecomposition plus up to 300 cheap iterations), so quantization takes longer. It is still a one-time cost and fits on a single GPU for the models in the paper.
- **Threads.** `solvers.py` sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 4 if they aren't already set. Without this, `eigh` on large matrices (e.g. 8192×8192 on CPU) can run out of threads and crash on shared machines.
- **Logging.** With `verbose=True` the solver prints the relative error every few iterations, the `ρ` value, and whether the pair-swap step was triggered. That's usually the first place to look if a layer looks off.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lucas2026admmq,
  title     = {{ADMM-Q}: An Improved Hessian-based Weight Quantizer for Post-Training Quantization of Large Language Models},
  author    = {Lucas, Ryan and Makni, Mehdi and Meng, Xiang and Deng, Adam and Mazumder, Rahul},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  eprint    = {2605.11222},
  archivePrefix = {arXiv}
}
```

## Acknowledgements

The JAX / Qwix implementation in `qwix_admmq/` was written by Adam Deng.

The quantization primitives and the `SpinQuant_qwen/` directory are built on [SpinQuant](https://github.com/facebookresearch/SpinQuant) (Liu et al., 2024), which is released under its own license (see `SpinQuant_qwen/LICENSE`). Our GPTQ baseline follows [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2023).
