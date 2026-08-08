"""Core quantization solvers: ADMM, GPTQ, PairSwap, and RTN.

All functions operate on raw weight matrices (torch.Tensor) and Hessian / X^T X
matrices.  No dependency on model structure, transformers, or SpinQuant.

Quantization primitives (scale/zero computation, fake-quantize) are copied from
SpinQuant's ``utils.quant_utils`` (WeightQuantizer) so that ADMM, GPTQ, and RTN
all use *exactly* the same rounding/clipping logic — byte-for-byte identical.
"""

# Limit OpenMP/MKL threads before torch to avoid Aborted (core dumped) from
# thread exhaustion when running eigh/cholesky on large matrices (e.g. 8192x8192).
import os
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "4"
if os.environ.get("MKL_NUM_THREADS") is None:
    os.environ["MKL_NUM_THREADS"] = "4"

import math
import time

import torch


# ---------------------------------------------------------------------------
# Quantization primitives — copied from SpinQuant utils/quant_utils.py
# for self-contained distribution.  Keep in sync with that file.
# ---------------------------------------------------------------------------

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def fake_quant(x, scale, zero, maxq, sym):
    """Quantize-dequantize (same as SpinQuant's WeightQuantizer round+clip)."""
    if sym:
        return sym_quant_dequant(x, scale, maxq)
    else:
        return asym_quant_dequant(x, scale, zero, maxq)


# ---------------------------------------------------------------------------
# Scale / zero computation — copied from WeightQuantizer.find_params
# ---------------------------------------------------------------------------


def compute_scale_zero(W, maxq, sym=True, mse=True, groupsize=-1,
                       norm=2.4, grid=100, maxshrink=0.8):
    """Compute quantization scale/zero for W (D_out, D_in).

    Logic is identical to WeightQuantizer.find_params (perchannel=True) from
    SpinQuant's ``utils/quant_utils.py``.

    Returns (scale, zero) with shapes:
        - per-channel (groupsize <= 0): (D_out, 1)
        - groupsize > 0:               (D_out, D_in)
    """
    dev = W.device

    if groupsize > 0:
        return _compute_scale_zero_groupwise(
            W, maxq, sym=sym, mse=mse, groupsize=groupsize,
            norm=norm, grid=grid, maxshrink=maxshrink,
        )

    # --- Per-channel (same as WeightQuantizer.find_params, perchannel=True) ---
    x = W.flatten(1)  # (D_out, D_in) — no-op for 2D

    tmp = torch.zeros(x.shape[0], device=dev)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
        scale = xmax / maxq
        zero = torch.zeros_like(scale)
    else:
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        scale = (xmax - xmin).clamp(min=1e-5) / maxq
        zero = torch.round(-xmin / scale)

    if mse:
        best = torch.full([x.shape[0]], float("inf"), device=dev)
        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid
            xmin1 = p * xmin
            xmax1 = p * xmax

            if sym:
                scale1 = xmax1 / maxq
                zero1 = torch.zeros_like(scale1)
                q = sym_quant_dequant(x, scale1.unsqueeze(1), maxq)
            else:
                scale1 = (xmax1 - xmin1) / maxq
                zero1 = torch.round(-xmin1 / scale1)
                q = asym_quant_dequant(
                    x, scale1.unsqueeze(1), zero1.unsqueeze(1), maxq,
                )

            q -= x
            q.abs_()
            q.pow_(norm)
            err = torch.sum(q, 1)
            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                scale[tmp] = scale1[tmp]
                zero[tmp] = zero1[tmp]

    shape = [-1] + [1] * (len(W.shape) - 1)
    scale = scale.reshape(shape)
    zero = zero.reshape(shape)
    return scale, zero


def _compute_scale_zero_groupwise(W, maxq, sym=True, mse=True, groupsize=128,
                                  norm=2.4, grid=100, maxshrink=0.8):
    """Groupwise variant — same as WeightQuantizer.find_params_weight_groupwise."""
    init_shape = W.shape
    x = W.reshape(W.shape[-2], W.shape[-1] // groupsize, groupsize)

    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.amin(x, dim=-1, keepdim=True)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
        scale = xmax / maxq
        zero = torch.zeros_like(scale)
    else:
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        scale = (xmax - xmin).clamp(min=1e-5) / maxq
        zero = torch.round(-xmin / scale)

    scale = scale.repeat(1, 1, groupsize)
    zero = zero.repeat(1, 1, groupsize)

    if mse:
        best = torch.full(
            [x.shape[0], x.shape[1]], float("inf"), device=x.device,
        ).type_as(x)
        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid
            xmin1 = p * xmin
            xmax1 = p * xmax

            if sym:
                scale1 = xmax1 / maxq
                zero1 = torch.zeros_like(scale1)
                scale1 = scale1.repeat(1, 1, groupsize)
                zero1 = zero1.repeat(1, 1, groupsize)
                q = sym_quant_dequant(x, scale1, maxq)
            else:
                scale1 = (xmax1 - xmin1) / maxq
                zero1 = torch.round(-xmin1 / scale1)
                scale1 = scale1.repeat(1, 1, groupsize)
                zero1 = zero1.repeat(1, 1, groupsize)
                q = asym_quant_dequant(x, scale1, zero1, maxq)

            q -= x
            q.abs_()
            q.pow_(norm)
            err = torch.sum(q, -1)
            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                scale[tmp] = scale1[tmp]
                zero[tmp] = zero1[tmp]

    scale = scale.reshape(init_shape)
    zero = zero.reshape(init_shape)
    return scale, zero


# ---------------------------------------------------------------------------
# ADMM-aware scale/zero computation
# ---------------------------------------------------------------------------


def admm_compute_scale_zero(W_quant, W_target, maxq, sym=True, groupsize=-1,
                            norm=2.0, grid=100, maxshrink=0.8):
    """Find scale/zero that minimise ||fake_quant(W_quant, s, z) - W_target||^norm.

    The clip-ratio grid search sweeps over the min/max of *W_quant* (which
    defines the quantisation grid) while measuring error against *W_target*.
    When W_quant == W_target this reduces to the standard MSE clip search
    (like compute_scale_zero) but with norm=2.0 by default.

    Returns (scale, zero) with shapes:
        - per-channel (groupsize <= 0): (D_out, 1)
        - groupsize > 0:               (D_out, D_in)
    """
    if groupsize > 0:
        return _admm_compute_scale_zero_groupwise(
            W_quant, W_target, maxq, sym=sym, groupsize=groupsize,
            norm=norm, grid=grid, maxshrink=maxshrink,
        )

    # --- Per-channel ---
    x = W_quant.flatten(1)   # (D_out, D_in) — what gets quantised
    t = W_target.flatten(1)  # (D_out, D_in) — error reference
    dev = x.device

    tmp = torch.zeros(x.shape[0], device=dev)
    xmin = torch.minimum(x.min(1)[0], tmp)  # (D_out,)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
        scale = xmax / maxq
        zero = torch.zeros_like(scale)
    else:
        bad = (xmin == 0) & (xmax == 0)
        xmin[bad] = -1
        xmax[bad] = +1
        scale = (xmax - xmin).clamp(min=1e-5) / maxq
        zero = torch.round(-xmin / scale)

    best = torch.full([x.shape[0]], float("inf"), device=dev)
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        xmin1 = p * xmin
        xmax1 = p * xmax

        if sym:
            scale1 = xmax1 / maxq
            zero1 = torch.zeros_like(scale1)
            q = sym_quant_dequant(x, scale1.unsqueeze(1), maxq)
        else:
            scale1 = (xmax1 - xmin1) / maxq
            zero1 = torch.round(-xmin1 / scale1)
            q = asym_quant_dequant(
                x, scale1.unsqueeze(1), zero1.unsqueeze(1), maxq,
            )

        q -= t          # error against TARGET
        q.abs_()
        q.pow_(norm)
        err = torch.sum(q, 1)
        tmp = err < best
        if torch.any(tmp):
            best[tmp] = err[tmp]
            scale[tmp] = scale1[tmp]
            zero[tmp] = zero1[tmp]

    shape = [-1] + [1] * (len(W_quant.shape) - 1)
    scale = scale.reshape(shape)
    zero = zero.reshape(shape)
    return scale, zero


def _admm_compute_scale_zero_groupwise(W_quant, W_target, maxq, sym=True,
                                       groupsize=128, norm=2.0, grid=100,
                                       maxshrink=0.8):
    """Groupwise variant of admm_compute_scale_zero."""
    init_shape = W_quant.shape
    x = W_quant.reshape(W_quant.shape[-2], W_quant.shape[-1] // groupsize, groupsize)
    t = W_target.reshape(W_target.shape[-2], W_target.shape[-1] // groupsize, groupsize)

    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.amin(x, dim=-1, keepdim=True)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
        scale = xmax / maxq
        zero = torch.zeros_like(scale)
    else:
        bad = (xmin == 0) & (xmax == 0)
        xmin[bad] = -1
        xmax[bad] = +1
        scale = (xmax - xmin).clamp(min=1e-5) / maxq
        zero = torch.round(-xmin / scale)

    scale = scale.repeat(1, 1, groupsize)
    zero = zero.repeat(1, 1, groupsize)

    best = torch.full(
        [x.shape[0], x.shape[1]], float("inf"), device=x.device,
    ).type_as(x)
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        xmin1 = p * xmin
        xmax1 = p * xmax

        if sym:
            scale1 = xmax1 / maxq
            zero1 = torch.zeros_like(scale1)
            scale1 = scale1.repeat(1, 1, groupsize)
            zero1 = zero1.repeat(1, 1, groupsize)
            q = sym_quant_dequant(x, scale1, maxq)
        else:
            scale1 = (xmax1 - xmin1) / maxq
            zero1 = torch.round(-xmin1 / scale1)
            scale1 = scale1.repeat(1, 1, groupsize)
            zero1 = zero1.repeat(1, 1, groupsize)
            q = asym_quant_dequant(x, scale1, zero1, maxq)

        q -= t          # error against TARGET
        q.abs_()
        q.pow_(norm)
        err = torch.sum(q, -1)
        tmp = err < best
        if torch.any(tmp):
            best[tmp] = err[tmp]
            scale[tmp] = scale1[tmp]
            zero[tmp] = zero1[tmp]

    scale = scale.reshape(init_shape)
    zero = zero.reshape(init_shape)
    return scale, zero

def _maybe_refresh_admm_quant_grid(
    B,
    W,
    X_norm,
    XtX_raw_f,
    scale_W,
    zero_W,
    maxq,
    sym,
    groupsize,
    refresh_accept_rel,
    verbose=False,
):

    scale_B = scale_W.T
    zero_B = zero_W.T
    B_curr = B

    B_proj_old = fake_quant(
        B_curr / X_norm[:, None], scale_B, zero_B, maxq, sym
    ) * X_norm[:, None]
    W_proj_old = B_proj_old.t() / X_norm
    diff_old = W_proj_old - W
    old_raw_recon = torch.sum((diff_old @ XtX_raw_f) * diff_old).item()

    W_iter = (B_curr / X_norm[:, None]).T.contiguous()
    new_scale_W, new_zero_W = admm_compute_scale_zero(
        W_iter, W, maxq, sym=sym, groupsize=groupsize
    )
    new_scale_W = new_scale_W.float()
    new_zero_W = new_zero_W.float()
    new_scale_B = new_scale_W.T
    new_zero_B = new_zero_W.T

    B_proj_new = fake_quant(
        B_curr / X_norm[:, None], new_scale_B, new_zero_B, maxq, sym
    ) * X_norm[:, None]
    W_proj_new = B_proj_new.t() / X_norm
    diff_new = W_proj_new - W
    new_raw_recon = torch.sum((diff_new @ XtX_raw_f) * diff_new).item()

    rel_improve = (
        (old_raw_recon - new_raw_recon) / max(abs(old_raw_recon), 1e-12)
    )

    accepted = rel_improve > refresh_accept_rel

    if verbose:
        if accepted:
            print(
                'accepted scale/zero refresh on stagnation: '
                f'raw_recon {old_raw_recon:.4e}->{new_raw_recon:.4e}, '
                f'rel_improve={rel_improve:.2e}, '
                f'[{new_scale_W.min():.6e}, {new_scale_W.max():.6e}]'
            )
        else:
            print(
                'rejected scale/zero refresh on stagnation: '
                f'raw_recon {old_raw_recon:.4e}->{new_raw_recon:.4e}, '
                f'rel_improve={rel_improve:.2e} '
                f'< {refresh_accept_rel:.2e}'
            )

    if accepted:
        return new_scale_W, new_zero_W, True
    return scale_W, zero_W, False


def ADMM_quantize(W_raw, XtX_raw, maxq=7, sym=True,
                  rho=0.1, rho_update=1.1,
                  max_iter=300, update_iter=3, switch_iter=30,
                  percdamp=0.01, mse=True, groupsize=-1,
                  verbose=True, label='',
                  stagnation_window=4,
                  stagnation_tol=1e-3,
                  refresh_on_stagnation=True,
                  refresh_accept_rel=1e-3):
    """Full-matrix ADMM weight quantization.

    This stays close to the original implementation and adds one optional,
    guarded modification: if progress stagnates, try recomputing the
    quantization grid once from the current iterate, and only accept the new
    grid if it improves the projected raw Hessian-weighted reconstruction error
    at the current iterate.

    Returns:
        ``(result, recon)`` -- dequantized weight matrix and Hessian-weighted
        reconstruction error.
    """
    dev = W_raw.device
    W = W_raw.float()
    XtX = XtX_raw.float().clone()
    XtX_raw_f = XtX_raw.float()

    diag_idx = torch.arange(XtX.shape[0], device=dev)
    damp = percdamp * torch.mean(torch.diag(XtX)).item()
    XtX[diag_idx, diag_idx] += damp

    X_norm = torch.diag(XtX).sqrt().clamp_min(1e-8)
    XtX = XtX / X_norm
    XtX = (XtX.T / X_norm).T
    YtX = torch.matmul(W * X_norm, XtX)

    B = (W * X_norm).t().clone()
    B_orig = B.clone()
    V = torch.zeros_like(B)

    L, Q = torch.linalg.eigh(XtX.double())

    cond_num = (L[-1] / L[0].clamp(min=1e-12)).item()
    needs_pairswap = cond_num > 5e4
    if needs_pairswap:
        if cond_num > 3e5:
            rho = 0.01
        else:
            rho = 0.05
        rho_update = 1.05
        print(
            f'[{label}] Condition number {cond_num:.2e} > 5.00e+04, '
            f'rho={rho:.2f} + rho_update={rho_update:.2f} + pairswap'
        )

    if verbose:
        print(f'[{label}] eigs: min={L[0]:.4e}, max={L[-1]:.4e}, damp={damp:.4e}')

    XTX_inv = (Q @ ((1 / (L + rho)) * Q).T).float()
    Res0 = torch.sum(B_orig * YtX.T)

    scale_W, zero_W = compute_scale_zero(
        W, maxq, sym=sym, mse=mse, groupsize=groupsize
    )
    scale_W = scale_W.float()
    zero_W = zero_W.float()
    scale_B = scale_W.T
    zero_B = zero_W.T

    if verbose:
        print(f'[{label}] scale: [{scale_W.min():.6e}, {scale_W.max():.6e}]')

    B_normed = B / X_norm[:, None]
    D = fake_quant(B_normed, scale_B, zero_B, maxq, sym) * X_norm[:, None]
    D_prev = D.clone()
    D_base = torch.sum(D**2).clamp_min(1e-12)

    error_history = []
    did_refresh = False

    for i_admm in range(max_iter):
        D_quant = (V + rho * B) / rho
        D = fake_quant(D_quant / X_norm[:, None], scale_B, zero_B, maxq, sym) * X_norm[:, None]

        B = XTX_inv @ (YtX.T - V + rho * D)
        V = V + rho * (B - D)

        if (i_admm + 1) % update_iter == 0:
            D_change = torch.sum((D - D_prev)**2)
            D_prev = D.clone()
            d_rel = (D_change / D_base).item()

            Btest = fake_quant(B / X_norm[:, None], scale_B, zero_B, maxq, sym) * X_norm[:, None]
            Resc = torch.matmul(XtX, Btest) - YtX.T
            Resc = torch.sum((Btest - B_orig) * Resc)
            errorc = (torch.sum(Resc) / Res0).item()
            error_history.append(errorc)

            if verbose:
                print(f'[{label}] iter {i_admm+1}: error={errorc:.6f}, D={d_rel:.6f}, '
                      f'rho={rho:.4f}, |B-D|={torch.norm(B-D).item():.4e}')

            if (
                refresh_on_stagnation
                and not did_refresh
                and len(error_history) >= max(2, stagnation_window)
            ):
                recent = error_history[-stagnation_window:]
                recent_rel_improve = (
                    (recent[0] - min(recent)) / max(abs(recent[0]), 1e-12)
                )

                if recent_rel_improve < stagnation_tol:
                    scale_W, zero_W, _ = _maybe_refresh_admm_quant_grid(
                        B=B,
                        W=W,
                        X_norm=X_norm,
                        XtX_raw_f=XtX_raw_f,
                        scale_W=scale_W,
                        zero_W=zero_W,
                        maxq=maxq,
                        sym=sym,
                        groupsize=groupsize,
                        refresh_accept_rel=refresh_accept_rel,
                        verbose=verbose,
                    )
                    scale_B = scale_W.T
                    zero_B = zero_W.T
                    did_refresh = True

            rho *= rho_update
            rho = min(rho, 1e6)
            XTX_inv = (Q @ ((1 / (L + rho)) * Q).T).float()

            if i_admm >= switch_iter and d_rel < 1e-6:
                if verbose:
                    print(f'[{label}] Converged at iter {i_admm+1}')
                break

    B_final = fake_quant(B / X_norm[:, None], scale_B, zero_B, maxq, sym) * X_norm[:, None]
    W_q = B_final.t() / X_norm
    result = fake_quant(W_q, scale_W, zero_W, maxq, sym)

    if needs_pairswap:
        result, _ = pairswap_batched(
        W_raw, XtX_raw, scale_W, zero_W,
        maxq=maxq, sym=sym,
        groupsize=groupsize,
        D_init=result,
        n_pairs=W_raw.shape[1],
        label=f'{label}-pairswap',
    )

    diff = result - W
    recon = torch.sum((diff @ XtX_raw_f) * diff).item()
    print(f'[{label}] ADMM recon: {recon:.4e}')
    return result, recon

# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------

def GPTQ_quantize(W_raw, XtX_raw, maxq=7, sym=True,
                  blocksize=128, percdamp=0.01, mse=True,
                  groupsize=-1, label=''):
    """Greedy GPTQ weight quantization with optional groupsize.

    Faithful port of SpinQuant's ``GPTQ.fasterquant`` (gptq_utils.py).
    When *groupsize* > 0, scale/zero are recomputed at every group boundary
    (dynamic groups, no act-order), exactly as in the original GPTQ repo.

    Returns:
        ``(Q, recon)`` -- dequantized weight matrix and Hessian-weighted
        reconstruction error.
    """
    W = W_raw.float().clone()
    D_out, D_in = W.shape
    H = XtX_raw.float().clone()

    # Dead weight handling
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # Damping
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(D_in, device=W.device)
    H[diag_idx, diag_idx] += damp

    # Initial quantization parameters (per-channel)
    scale_cur, zero_cur = compute_scale_zero(
        W_raw.float(), maxq, sym=sym, mse=mse, groupsize=-1,
    )
    s = scale_cur.squeeze(1)    # (D_out,)
    z = zero_cur.squeeze(1)     # (D_out,)

    # Cholesky of inverse Hessian
    H_chol = torch.linalg.cholesky(H)
    Hinv_chol = torch.cholesky_inverse(H_chol)
    Hinv = torch.linalg.cholesky(Hinv_chol, upper=True)

    Q = torch.zeros_like(W)

    for i1 in range(0, D_in, blocksize):
        i2 = min(i1 + blocksize, D_in)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]

            # Recompute scale/zero at group boundaries
            if groupsize > 0 and (i1 + i) % groupsize == 0:
                g_start = i1 + i
                g_end = min(g_start + groupsize, D_in)
                gs, gz = compute_scale_zero(
                    W[:, g_start:g_end], maxq, sym=sym, mse=mse, groupsize=-1,
                )
                s = gs.squeeze(1)  # (D_out,)
                z = gz.squeeze(1)  # (D_out,)

            q = fake_quant(w, s, z, maxq, sym)
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    diff = Q - W_raw.float()
    recon = torch.sum((diff @ XtX_raw.float()) * diff).item()
    gs_str = f'gs={groupsize}' if groupsize > 0 else 'per-ch'
    print(f'[{label}] GPTQ recon: {recon:.4e} ({gs_str})')
    return Q, recon


def pairswap_batched(
    W_raw,
    XtX_raw,
    scale=None,
    zero=None,
    maxq=7,
    sym=True,
    groupsize=-1,
    n_rounds=5,
    n_pairs=4096,
    batch_size=64,
    D_init=None,
    label='',
):
    """Greedy pair-swap local search.  Always applies the single best pair per
    batch (no top-k, no hedging).

    Each round samples ``n_pairs`` candidate pairs in batches of ``batch_size``.
    For each batch the single best-improving pair is applied greedily.
    Stops early when a full round produces zero flips.

    Returns:
        ``(D, recon)`` -- dequantized weight matrix and Hessian-weighted
        reconstruction error.
    """
    W = W_raw.float()
    H = XtX_raw.float().clone()
    D_out, D_in = W.shape

    diag_idx = torch.arange(D_in, device=W.device)
    damp = 1e-4 * torch.mean(torch.diag(H)).item()
    H[diag_idx, diag_idx] += damp

    # Scale / zero
    if scale is None:
        scale_W, zero_W = compute_scale_zero(
            W, maxq, sym=sym, mse=True, groupsize=groupsize,
        )
        scale_W = scale_W.float()
        zero_W = zero_W.float()
    else:
        scale_W = scale.float()
        zero_W = zero.float()

    if D_init is not None:
        D = D_init.float().clone()
    else:
        D = fake_quant(W, scale_W, zero_W, maxq, sym)

    per_ch = (scale_W.shape[1] == 1)
    if per_ch:
        s = scale_W.squeeze(1)       # (D_out,)
        s_col = s.unsqueeze(1)       # (D_out, 1)

    diff0 = D - W
    init_recon = torch.sum((diff0 @ XtX_raw.float()) * diff0).item()
    print(f'[{label}] init recon={init_recon:.4e}')

    H_diag = torch.diag(H)
    B = min(int(batch_size), D_in // 2)
    if B <= 0:
        return D, init_recon
    n_batches = max(1, int(n_pairs) // B)

    sign1 = torch.tensor([+1.0, +1.0, -1.0, -1.0], device=W.device, dtype=W.dtype)
    sign2 = torch.tensor([+1.0, -1.0, +1.0, -1.0], device=W.device, dtype=W.dtype)

    if per_ch:
        d1v_all = sign1.view(4, 1, 1) * s.view(1, D_out, 1)
        d2v_all = sign2.view(4, 1, 1) * s.view(1, D_out, 1)

    tick = time.time()
    total_flips = 0

    for rd in range(n_rounds):
        G = (W - D) @ H
        n_flips = 0

        for _ in range(n_batches):
            perm = torch.randperm(D_in, device=W.device)[:2 * B]
            j1 = perm[:B].contiguous()
            j2 = perm[B:].contiguous()

            g1, g2 = G[:, j1], G[:, j2]
            d1_col, d2_col = D[:, j1], D[:, j2]
            h11, h22, h12 = H_diag[j1], H_diag[j2], H[j1, j2]

            # Per-column scales
            if per_ch:
                s1_col, s2_col = s_col, s_col
            else:
                s1_col = scale_W[:, j1]
                s2_col = scale_W[:, j2]

            # Bounds check
            if sym:
                q1 = torch.round(d1_col / s1_col)
                can_up1, can_dn1 = (q1 < maxq), (q1 > -(maxq + 1))
                q2 = torch.round(d2_col / s2_col)
                can_up2, can_dn2 = (q2 < maxq), (q2 > -(maxq + 1))
            else:
                zw1 = zero_W if per_ch else zero_W[:, j1]
                zw2 = zero_W if per_ch else zero_W[:, j2]
                q1 = torch.round(d1_col / s1_col) + zw1
                can_up1, can_dn1 = (q1 < maxq), (q1 > 0)
                q2 = torch.round(d2_col / s2_col) + zw2
                can_up2, can_dn2 = (q2 < maxq), (q2 > 0)

            v_all = torch.stack([
                can_up1 & can_up2, can_up1 & can_dn2,
                can_dn1 & can_up2, can_dn1 & can_dn2,
            ], dim=0)

            # Delta-loss for all 4 directions
            g1b, g2b = g1.unsqueeze(0), g2.unsqueeze(0)
            h11b = h11.view(1, 1, B)
            h22b = h22.view(1, 1, B)
            h12b = h12.view(1, 1, B)

            if per_ch:
                d1v, d2v = d1v_all, d2v_all
            else:
                d1v = sign1.view(4, 1, 1) * s1_col.unsqueeze(0)
                d2v = sign2.view(4, 1, 1) * s2_col.unsqueeze(0)

            dL = (-2.0 * d1v * g1b + d1v * d1v * h11b
                  -2.0 * d2v * g2b + d2v * d2v * h22b
                  +2.0 * d1v * d2v * h12b)
            dL = torch.where(v_all, dL, torch.zeros_like(dL))

            # Best direction per pair, summed across output channels
            best_dL, best_id = dL.min(dim=0)          # (D_out, B)
            accept = best_dL < 0
            improve = (-best_dL).masked_fill(~accept, 0.0).sum(dim=0)  # (B,)

            # Pick the single best pair
            best_pair = improve.argmax()
            if improve[best_pair] <= 0:
                continue

            # Apply
            bid = best_id[:, best_pair]                # (D_out,)
            acc = accept[:, best_pair]                 # (D_out,)
            sg1 = sign1[bid]                           # (D_out,)
            sg2 = sign2[bid]

            if per_ch:
                d1_apply = (s * sg1).masked_fill(~acc, 0.0)
                d2_apply = (s * sg2).masked_fill(~acc, 0.0)
            else:
                d1_apply = (scale_W[:, j1[best_pair]] * sg1).masked_fill(~acc, 0.0)
                d2_apply = (scale_W[:, j2[best_pair]] * sg2).masked_fill(~acc, 0.0)

            n_batch_flips = int(acc.sum().item())
            if n_batch_flips == 0:
                continue
            n_flips += n_batch_flips

            D[:, j1[best_pair]] += d1_apply
            D[:, j2[best_pair]] += d2_apply

            # Rank-2 gradient update
            delta = torch.stack([d1_apply, d2_apply], dim=1)  # (D_out, 2)
            idx = torch.tensor([j1[best_pair], j2[best_pair]], device=W.device)  # (2,)
            G -= delta @ H[idx, :]

        total_flips += n_flips
        diff = D - W
        recon = torch.sum((diff @ XtX_raw.float()) * diff).item()
        elapsed = time.time() - tick
        print(f'[{label}] round {rd+1}: flips={n_flips}, recon={recon:.4e}, time={elapsed:.1f}s')
        if n_flips == 0:
            print(f'[{label}] converged at round {rd+1}')
            break

    D = fake_quant(D, scale_W, zero_W, maxq, sym)
    diff = D - W
    recon = torch.sum((diff @ XtX_raw.float()) * diff).item()
    print(f'[{label}] FINAL: recon={recon:.4e}, total_flips={total_flips}, time={time.time()-tick:.1f}s')
    return D, recon
