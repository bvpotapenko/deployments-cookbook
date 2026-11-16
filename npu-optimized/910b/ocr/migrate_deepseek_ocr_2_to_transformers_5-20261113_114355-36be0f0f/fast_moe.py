"""Fast MoE experts forward for DeepSeek-OCR-2 on the Ascend 910B.

The stock ``DeepseekOcr2TextExperts.forward`` dispatches (through
``@use_experts_implementation``) to ``aten::_grouped_mm``, which torch_npu
decomposes into one ``mm`` per expert per projection: 64 experts x 2
projections x 11 sparse layers = 1408 kernel launches per decode step, which
is launch-bound (~45 ms/step measured on this card).

This module replaces the forward of each experts module with a dense
formulation over all experts that produces the *same* result and needs a
handful of kernels per layer:

  1. ``mm`` of the flattened token rows against ``gate_up_proj`` reshaped to
     ``[E*2I, H]`` -> all experts' gate/up activations for every token;
  2. SiLU(gate) * up, scaled by a per-(token, expert) weight map that is
     zero outside each token's routed experts (a scatter of the router's
     top-k weights into a ``[T, E]`` map);
  3. one ``mm`` against ``down_proj`` transposed-flattened to ``[E*I, H]``:
     because non-selected experts' rows were zeroed, the flat matrix product
     computes exactly the routed sum ``sum_k w_k * down_{e_k}(y_{e_k})``.

The math over the selected experts is unchanged; the only difference from
the stock path is bf16 reduction order (kernel-level), the same class of
difference as switching attention implementations. The weights themselves are
never modified; ``_down_flat`` is a contiguous in-memory transpose used as a
GEMM operand, built once per process.

Tokens are processed in chunks so the ``[chunk, E*2I]`` activation stays
bounded (4096 rows -> under 1 GB in bf16).
"""

from __future__ import annotations

import torch

CHUNK = 4096


def install_fast_experts(model):
    """Patch every sparse layer's experts module to the fast forward."""
    layers = model.model.language_model.layers
    patched = 0
    for layer in layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue
        if not hasattr(experts, "_fast_gate_up_flat"):
            experts._fast_gate_up_flat = experts.gate_up_proj.reshape(
                experts.num_experts * 2 * experts.intermediate_dim, experts.hidden_dim
            )
            experts._fast_down_flat = experts.down_proj.transpose(1, 2).contiguous().view(
                experts.num_experts * experts.intermediate_dim, experts.hidden_dim
            )
        experts.forward = _fast_forward.__get__(experts, type(experts))
        patched += 1
    return patched


def _fast_forward(self, hidden_states, top_k_index, top_k_weights):
    num_experts = self.num_experts
    intermediate = self.intermediate_dim
    hidden = self.hidden_dim
    gate_up_flat = self._fast_gate_up_flat  # [E*2I, H]
    down_flat = self._fast_down_flat  # [E*I, H]
    total, _ = hidden_states.shape
    out = torch.empty_like(hidden_states)
    weights = top_k_weights.to(out.dtype)  # [T, K]
    step = CHUNK if total > CHUNK else total
    for start in range(0, total, step):
        stop = min(start + step, total)
        rows = stop - start
        x = hidden_states[start:stop]
        gate_up = torch.mm(x, gate_up_flat.t()).view(rows, num_experts, 2, intermediate)
        y = self.act_fn(gate_up[:, :, 0]) * gate_up[:, :, 1]  # [t, E, I]
        # weight map: zero everywhere except each token's routed experts
        routed = torch.zeros(rows, num_experts, dtype=y.dtype, device=y.device)
        routed.scatter_(1, top_k_index[start:stop], weights[start:stop])
        y = y * routed.unsqueeze(-1)
        out[start:stop] = torch.mm(y.reshape(rows, num_experts * intermediate), down_flat)
    return out
