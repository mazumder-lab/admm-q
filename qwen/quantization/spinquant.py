"""SpinQuant: Random rotation preprocessing for Qwen3 models.

Applies orthogonal rotations to weights and activations to make distributions
more uniform before quantization. Supports W4A8KV8 configuration.

Four rotation matrices:
  R1: hidden_size x hidden_size  — inter-layer, fused offline into all weights
  R2: head_dim x head_dim        — per-head value rotation, fused into v_proj/o_proj
  R3: head_dim x head_dim        — Hadamard on Q/K after RoPE (online, attention patch)
  R4: intermediate_size x intermediate_size — hybrid (weight offline + input online hook)

Reference: Ashkboos et al., "SpinQuant: LLM Quantization with Learned Rotations", 2024.
"""

import logging
import types
from collections.abc import Callable

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def random_orthogonal_matrix(size, device="cpu", dtype=torch.float64):
    """Generate a random orthogonal matrix via QR decomposition of Gaussian noise."""
    H = torch.randn(size, size, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(H)
    # Ensure uniform Haar measure: fix sign so diagonal of R is positive
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    return Q


def _is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(size, device="cpu", dtype=torch.float64):
    """Walsh-Hadamard matrix (Sylvester construction) with random sign flip.

    Only works for power-of-2 sizes. Falls back to random_orthogonal_matrix otherwise.
    """
    if not _is_power_of_2(size):
        log.info("Size %d is not power-of-2, using random orthogonal matrix instead of Hadamard", size)
        return random_orthogonal_matrix(size, device=device, dtype=dtype)

    # Sylvester construction: H_1 = [1], H_{2n} = [[H_n, H_n], [H_n, -H_n]] / sqrt(2)
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < size:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / (2 ** 0.5)

    # Random sign flip for randomness (still orthogonal)
    D = torch.randint(0, 2, (size,), device=device, dtype=dtype) * 2 - 1
    H = H * D.unsqueeze(0)  # multiply columns by random ±1
    return H


# ---------------------------------------------------------------------------
# Embedding untying
# ---------------------------------------------------------------------------

def untie_embeddings(model):
    """Untie lm_head from embed_tokens if they share the same weight tensor."""
    if not getattr(model.config, "tie_word_embeddings", False):
        log.info("Embeddings already untied.")
        return

    log.info("Untying lm_head from embed_tokens ...")
    model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.clone())
    model.config.tie_word_embeddings = False


# ---------------------------------------------------------------------------
# RMSNorm fusion
# ---------------------------------------------------------------------------

def _get_submodule(module, dotted_name):
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


@torch.no_grad()
def fuse_rmsnorms(model):
    """Absorb RMSNorm gamma into adjacent linear weights, then set gamma=1.

    Groups:
      input_layernorm -> q_proj, k_proj, v_proj
      post_attention_layernorm -> gate_proj, up_proj
      model.norm -> lm_head

    q_norm/k_norm are NOT fused (per-head, different dimension).
    """
    log.info("Fusing RMSNorm weights into adjacent linears ...")
    layers = model.model.layers

    for i, layer in enumerate(layers):
        # input_layernorm -> q/k/v_proj
        ln = layer.input_layernorm
        gamma = ln.weight.data.float()
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            linear = _get_submodule(layer, name)
            # W_new = W * diag(gamma)  i.e. scale each input column
            linear.weight.data = (linear.weight.data.float() * gamma.view(1, -1)).to(linear.weight.dtype)
        ln.weight.data.fill_(1.0)

        # post_attention_layernorm -> gate_proj, up_proj
        ln2 = layer.post_attention_layernorm
        gamma2 = ln2.weight.data.float()
        for name in ["mlp.gate_proj", "mlp.up_proj"]:
            linear = _get_submodule(layer, name)
            linear.weight.data = (linear.weight.data.float() * gamma2.view(1, -1)).to(linear.weight.dtype)
        ln2.weight.data.fill_(1.0)

    # model.norm -> lm_head
    final_norm = model.model.norm
    gamma_final = final_norm.weight.data.float()
    model.lm_head.weight.data = (
        model.lm_head.weight.data.float() * gamma_final.view(1, -1)
    ).to(model.lm_head.weight.dtype)
    final_norm.weight.data.fill_(1.0)

    log.info("RMSNorm fusion complete.")


# ---------------------------------------------------------------------------
# R1: Inter-layer rotation (offline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_r1_rotation(model, R1):
    """Fuse R1 into all weights. Redefines residual stream as h' = h @ R1.

    embed_tokens:  E_new = E @ R1
    q/k/v_proj:    W_new = W @ R1        (input side)
    o_proj:        W_new = R1^T @ W      (output side)
    gate/up_proj:  W_new = W @ R1        (input side)
    down_proj:     W_new = R1^T @ W      (output side)
    lm_head:       W_new = W @ R1        (input side)
    """
    log.info("Applying R1 rotation (hidden_size=%d) ...", R1.shape[0])
    dtype = model.model.embed_tokens.weight.dtype
    R1f = R1.float()
    R1T = R1f.t()

    # embed_tokens: E @ R1
    model.model.embed_tokens.weight.data = (
        model.model.embed_tokens.weight.data.float() @ R1f
    ).to(dtype)

    for layer in model.model.layers:
        # q/k/v_proj: W @ R1 (W has shape [out, in], so W_new = W @ R1 means right-multiply input dim)
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "mlp.gate_proj", "mlp.up_proj"]:
            linear = _get_submodule(layer, name)
            linear.weight.data = (linear.weight.data.float() @ R1f).to(dtype)

        # o_proj: R1^T @ W (left-multiply output dim: W has shape [hidden, hidden])
        # W[out, in] -> W_new[out, in] = R1^T @ W means new_row_i = sum_j R1^T[i,j] * W[j,:]
        # But o_proj output goes to residual. o_proj.weight is [hidden_size, num_heads*head_dim].
        # Output = x @ W^T, so new output = R1^T @ (x @ W^T) = x @ (W @ R1).
        # Wait - let me re-derive. y = hidden @ W_o^T adds to residual.
        # We want y' = y @ R1 (to stay in rotated residual stream).
        # y' = hidden @ W_o^T @ R1 = hidden @ (R1^T @ W_o)^T
        # So W_o_new = R1^T @ W_o
        layer.self_attn.o_proj.weight.data = (
            R1T @ layer.self_attn.o_proj.weight.data.float()
        ).to(dtype)

        # down_proj: same as o_proj
        layer.mlp.down_proj.weight.data = (
            R1T @ layer.mlp.down_proj.weight.data.float()
        ).to(dtype)

    # lm_head: W @ R1
    model.lm_head.weight.data = (
        model.lm_head.weight.data.float() @ R1f
    ).to(dtype)

    log.info("R1 rotation applied.")


# ---------------------------------------------------------------------------
# R2: Per-head value rotation (offline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_r2_rotation(model, R2):
    """Fuse R2 (head_dim x head_dim) into v_proj and o_proj per head.

    v_proj: W_v_k_new = R2^T @ W_v_k  →  v_new = v @ R2 per KV-head
    o_proj: W_o_h_new = W_o_h @ R2     →  cancels R2: (a@R2) @ (W@R2)^T = a @ R2 @ R2^T @ W^T = a @ W^T
    """
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    log.info("Applying R2 rotation (head_dim=%d, num_kv_heads=%d, num_heads=%d) ...",
             head_dim, num_kv_heads, num_heads)
    dtype = model.model.layers[0].self_attn.v_proj.weight.dtype
    R2f = R2.float()
    R2fT = R2f.t()

    for layer in model.model.layers:
        # v_proj: shape [num_kv_heads * head_dim, hidden_size]
        # R2^T @ W_v_k gives v_new = input @ (R2^T @ W_v_k)^T = input @ W_v_k^T @ R2 = v @ R2
        v_weight = layer.self_attn.v_proj.weight.data.float()
        for k in range(num_kv_heads):
            s = k * head_dim
            e = s + head_dim
            v_weight[s:e, :] = R2fT @ v_weight[s:e, :]
        layer.self_attn.v_proj.weight.data = v_weight.to(dtype)

        # o_proj: shape [hidden_size, num_heads * head_dim]
        # W_o_h @ R2 gives: (a_h @ R2) @ (W_o_h @ R2)^T = a_h @ R2 @ R2^T @ W_o_h^T = a_h @ W_o_h^T
        o_weight = layer.self_attn.o_proj.weight.data.float()
        for h in range(num_heads):
            s = h * head_dim
            e = s + head_dim
            o_weight[:, s:e] = o_weight[:, s:e] @ R2f
        layer.self_attn.o_proj.weight.data = o_weight.to(dtype)

    log.info("R2 rotation applied.")


# ---------------------------------------------------------------------------
# R4: down_proj Hadamard (offline part)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_r4_offline(model, H4_per_layer):
    """Absorb H4 into down_proj weights: W_d_new = W_d @ H4.

    H4_per_layer: list of (intermediate_size x intermediate_size) matrices, one per layer.
    """
    log.info("Applying R4 offline (into down_proj weights) ...")
    dtype = model.model.layers[0].mlp.down_proj.weight.dtype

    for i, layer in enumerate(model.model.layers):
        H4f = H4_per_layer[i].float()
        # down_proj.weight shape: [hidden_size, intermediate_size]
        # Forward: y = x @ W^T. With rotation: y = (x @ H4) @ (W @ H4)^T = x @ W^T
        # So W_new = W @ H4
        layer.mlp.down_proj.weight.data = (
            layer.mlp.down_proj.weight.data.float() @ H4f
        ).to(dtype)

    log.info("R4 offline applied to %d layers.", len(model.model.layers))


# ---------------------------------------------------------------------------
# R4: down_proj Hadamard (online hook)
# ---------------------------------------------------------------------------

def add_r4_hooks(model, H4_per_layer):
    """Register forward pre-hooks on each down_proj: x' = x @ H4.

    Returns list of hook handles for later removal.
    """
    handles = []

    for i, layer in enumerate(model.model.layers):
        H4 = H4_per_layer[i]
        down_proj = layer.mlp.down_proj

        # Register H4 as buffer so it moves with model.to(device)
        down_proj.register_buffer("_spin_H4", H4.to(down_proj.weight.dtype))

        def _make_hook(mod_ref):
            def hook(module, args):
                x = args[0]
                H = module._spin_H4.to(dtype=x.dtype)
                x_rot = x @ H
                return (x_rot,) + args[1:]
            return hook

        h = down_proj.register_forward_pre_hook(_make_hook(down_proj))
        handles.append(h)

    log.info("Added R4 forward pre-hooks to %d down_proj layers.", len(handles))
    return handles


# ---------------------------------------------------------------------------
# R3: Attention forward patch (Q/K Hadamard + KV quantization)
# ---------------------------------------------------------------------------

def _make_patched_attention_forward(original_apply_rotary_pos_emb):
    """Create a patched forward method for Qwen3Attention.

    Inserts H3 rotation on Q/K after RoPE and optional KV fake-quantization.
    """
    from quantization.act_quant import _fake_quantize_activation

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = original_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --- SpinQuant R3: apply Hadamard to Q and K after RoPE ---
        H3 = self._spin_H3.to(dtype=query_states.dtype)
        query_states = query_states @ H3
        key_states = key_states @ H3

        # --- KV quantization (fake quant for KV cache) ---
        kv_bits = getattr(self, "_spin_kv_bits", None)
        if kv_bits is not None:
            key_states = _fake_quantize_activation(key_states, kv_bits, sym=True, per_token=True)
            value_states = _fake_quantize_activation(value_states, kv_bits, sym=True, per_token=True)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            None,  # will get eager_attention_forward from the registry
        )
        # Fallback to eager if None
        if attention_interface is None:
            from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return patched_forward


def patch_attention_forward(model, H3, kv_bits=None):
    """Monkey-patch Qwen3Attention.forward to insert R3 and optional KV quantization."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as orig_rope

    patched_fn = _make_patched_attention_forward(orig_rope)

    count = 0
    dtype = model.model.layers[0].self_attn.q_proj.weight.dtype
    for layer in model.model.layers:
        attn = layer.self_attn
        # Register H3 as buffer so it moves with model.to(device)
        attn.register_buffer("_spin_H3", H3.to(dtype))
        attn._spin_kv_bits = kv_bits
        attn.forward = types.MethodType(patched_fn, attn)
        count += 1

    log.info("Patched attention forward on %d layers (R3 Hadamard%s).",
             count, f" + KV{kv_bits}" if kv_bits else "")


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@torch.no_grad()
def spin_model(model, seed=42, kv_bits=None):
    """Apply SpinQuant rotations to a Qwen3 model in-place.

    Steps:
      1. Untie embeddings (if tied)
      2. Fuse RMSNorm weights into adjacent linears
      3. Generate R1, R2, H3, H4
      4. Apply R1 (inter-layer) offline
      5. Apply R2 (per-head value) offline
      6. Apply R4 offline (into down_proj) + online hooks
      7. Patch attention forward (R3 + optional KV quant)

    Args:
        model: Qwen3ForCausalLM model
        seed: Random seed for reproducible rotations
        kv_bits: If set, fake-quantize K/V in attention to this bit width

    Returns:
        List of R4 hook handles (call handle.remove() to clean up)
    """
    torch.manual_seed(seed)

    config = model.config
    hidden_size = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
    intermediate_size = config.intermediate_size
    num_layers = len(model.model.layers)

    log.info("SpinQuant: hidden=%d, head_dim=%d, intermediate=%d, layers=%d, kv_bits=%s",
             hidden_size, head_dim, intermediate_size, num_layers, kv_bits)

    # Step 1: Untie embeddings
    untie_embeddings(model)

    # Step 2: Fuse RMSNorm
    fuse_rmsnorms(model)

    # Step 3: Generate rotation matrices
    device = "cpu"

    # R1: hidden_size x hidden_size
    if _is_power_of_2(hidden_size):
        R1 = hadamard_matrix(hidden_size, device=device)
    else:
        R1 = random_orthogonal_matrix(hidden_size, device=device)
    log.info("R1: %s (%dx%d)", "Hadamard" if _is_power_of_2(hidden_size) else "QR",
             hidden_size, hidden_size)

    # R2: head_dim x head_dim (always power-of-2 for Qwen3)
    R2 = hadamard_matrix(head_dim, device=device)
    log.info("R2: Hadamard (%dx%d)", head_dim, head_dim)

    # H3: head_dim x head_dim Hadamard for Q/K
    H3 = hadamard_matrix(head_dim, device=device)
    log.info("H3: Hadamard (%dx%d)", head_dim, head_dim)

    # H4: one per layer, intermediate_size x intermediate_size
    H4_per_layer = []
    for _ in range(num_layers):
        if _is_power_of_2(intermediate_size):
            H4 = hadamard_matrix(intermediate_size, device=device)
        else:
            H4 = random_orthogonal_matrix(intermediate_size, device=device)
        H4_per_layer.append(H4)
    log.info("H4: %s (%dx%d) x %d layers",
             "Hadamard" if _is_power_of_2(intermediate_size) else "QR",
             intermediate_size, intermediate_size, num_layers)

    # Step 4: Apply R1
    apply_r1_rotation(model, R1)

    # Step 5: Apply R2
    apply_r2_rotation(model, R2)

    # Step 6: Apply R4 offline + online hooks
    apply_r4_offline(model, H4_per_layer)
    r4_handles = add_r4_hooks(model, H4_per_layer)

    # Step 7: Patch attention forward (R3 + KV quant)
    patch_attention_forward(model, H3, kv_bits=kv_bits)

    # Verify orthogonality
    for name, mat in [("R1", R1), ("R2", R2), ("H3", H3), ("H4[0]", H4_per_layer[0])]:
        I = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
        err = (mat @ mat.t() - I).abs().max().item()
        assert err < 1e-4, f"{name} not orthogonal: max error {err}"
    log.info("All rotation matrices verified orthogonal.")

    log.info("SpinQuant applied successfully.")
    return r4_handles
