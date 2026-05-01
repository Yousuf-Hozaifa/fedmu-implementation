"""
models/lora_layer.py

Manual LoRA implementation for Fed-MU.

We implement LoRA from scratch (rather than using PEFT directly) because:
  1. We need independent gradient tracking per adapter (A_v, A_l, A×)
  2. We need to apply gradient ASCENT to A× during unlearning
  3. We need to extract per-adapter residual outputs BA·x for E6 probe
  4. PEFT merges adapters in ways that complicate independent manipulation

Architecture of one LoRA-adapted linear layer:

    input x (dim: in_features)
        |
        ├──────────────────────────────────┐
        │                                  │
    W · x (frozen base)          B · A · x · scaling
    [out_features, in_features]  [out_features] ← LoRA residual
        │                                  │
        └─────────────── + ───────────────┘
                          │
                    output (dim: out_features)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    A linear layer with an attached LoRA adapter.

    Replaces nn.Linear(in_features, out_features) while keeping the
    original weight frozen and adding trainable low-rank matrices A and B.

    Args:
        in_features:   input dimension
        out_features:  output dimension
        rank:          LoRA rank (r) — controls adapter capacity
        lora_alpha:    LoRA scaling factor (alpha)
        dropout:       dropout applied to LoRA path (regularisation)
        base_layer:    optional pre-existing nn.Linear to wrap
                       (if None, creates a new random layer)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        lora_alpha: float = 8.0,
        dropout: float = 0.05,
        base_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank   # The (alpha/r) scaling factor

        # ── Base layer (FROZEN) ─────────────────────────────────────────
        if base_layer is not None:
            # Wrap an existing layer
            self.weight = base_layer.weight
            self.bias = base_layer.bias
            self._use_existing = True
        else:
            # Create a new base layer
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features), requires_grad=False
            )
            self.bias = nn.Parameter(
                torch.zeros(out_features), requires_grad=False
            )
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self._use_existing = False

        # Freeze the base weight — it is never updated
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # ── LoRA matrices A and B (TRAINABLE) ───────────────────────────
        # A: initialised with Kaiming uniform (standard LoRA init)
        # B: initialised to ZERO (so initial output = base output exactly)
        # This zero-init of B means: at the start of training, the LoRA
        # residual contributes nothing, and training starts from the
        # pre-trained base model's behaviour. This is crucial for stability.
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is already zeros from above

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(p=dropout)

        # Track whether this adapter is active
        self.lora_enabled = True

    def get_lora_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute only the LoRA residual: B · A · dropout(x) · scaling

        This is the 'BA·x' term used in the E6 identity leakage probe.
        The probe will extract THIS value to test whether patient
        identity is encoded in this adapter.

        Args:
            x: input tensor [..., in_features]
        Returns:
            residual: [..., out_features]
        """
        dropped = self.lora_dropout(x)
        # A: [rank, in_features], x: [..., in_features]
        # A·x: [..., rank]
        ax = dropped @ self.lora_A.T       # [..., rank]
        # B: [out_features, rank]
        # B·(A·x): [..., out_features]
        bax = ax @ self.lora_B.T           # [..., out_features]
        return bax * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward: base(x) + lora_residual(x)
        """
        # Base computation (frozen weights)
        base_out = x @ self.weight.T
        if self.bias is not None:
            base_out = base_out + self.bias

        if not self.lora_enabled:
            return base_out

        # LoRA residual
        lora_out = self.get_lora_residual(x)

        return base_out + lora_out

    def freeze(self):
        """Freeze this LoRA adapter (no gradient updates)."""
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

    def unfreeze(self):
        """Unfreeze this LoRA adapter (allow gradient updates)."""
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

    def get_gradient_norm(self) -> Optional[float]:
        """
        Compute the Frobenius norm of gradients on this adapter.
        Used for computing ρ(p) in Step 4.

        Returns None if no gradient has been computed yet.
        """
        norms = []
        for param in [self.lora_A, self.lora_B]:
            if param.grad is not None:
                norms.append(param.grad.detach().norm(2).item() ** 2)
        if not norms:
            return None
        return sum(norms) ** 0.5   # Frobenius norm = sqrt(sum of squared norms)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.lora_alpha}, "
                f"scaling={self.scaling:.3f}")