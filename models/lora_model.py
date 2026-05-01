"""
models/lora_model.py

Wraps FedMUModel with three separate LoRA adapters:
  A_v  → vision encoder q_proj, v_proj (CLIP)
  A_l  → text encoder q_proj, v_proj (DistilBERT)
  A×   → CrossModalProjection layer1, layer2

The three adapters are managed independently so that:
  - During pre-training: all three adapters train together
  - During unlearning: only A× is active; A_v and A_l are frozen
  - For ρ(p): gradients are tracked per-adapter
  - For E6 probe: residuals are extracted per-adapter
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.multimodal_model import FedMUModel, CrossModalProjection
from models.lora_layer import LoRALinear


class LoRAAdapter(nn.Module):
    """
    A named group of LoRA-adapted layers that can be
    frozen/unfrozen and queried for gradients as a unit.

    Think of this as one of the three adapters: A_v, A_l, or A×.
    """

    def __init__(self, name: str):
        super().__init__()
        self.adapter_name = name
        self.lora_layers: nn.ModuleDict = nn.ModuleDict()

    def add_layer(self, layer_name: str, lora_linear: LoRALinear):
        """Register a LoRALinear layer under this adapter group."""
        # ModuleDict keys cannot contain dots
        safe_name = layer_name.replace(".", "_")
        self.lora_layers[safe_name] = lora_linear

    def freeze(self):
        """Freeze all LoRA layers in this adapter group."""
        for layer in self.lora_layers.values():
            layer.freeze()

    def unfreeze(self):
        """Unfreeze all LoRA layers in this adapter group."""
        for layer in self.lora_layers.values():
            layer.unfreeze()

    def get_total_gradient_norm(self) -> Optional[float]:
        """
        Compute the combined Frobenius norm of all gradients
        in this adapter group.

        This is the ||∇_{A_name} L||_F term in the ρ(p) formula.
        """
        total_sq = 0.0
        has_grads = False

        for layer in self.lora_layers.values():
            for param in [layer.lora_A, layer.lora_B]:
                if param.grad is not None:
                    total_sq += param.grad.detach().norm(2).item() ** 2
                    has_grads = True

        return total_sq ** 0.5 if has_grads else None

    def get_state_dict(self) -> dict:
        """Export only the LoRA parameters (A and B matrices)."""
        state = {}
        for layer_name, layer in self.lora_layers.items():
            state[f"{layer_name}.lora_A"] = layer.lora_A.data.clone()
            state[f"{layer_name}.lora_B"] = layer.lora_B.data.clone()
        return state

    def load_state_dict_lora(self, state: dict):
        """Load LoRA parameters from a state dict."""
        for layer_name, layer in self.lora_layers.items():
            if f"{layer_name}.lora_A" in state:
                layer.lora_A.data = state[f"{layer_name}.lora_A"].clone()
                layer.lora_B.data = state[f"{layer_name}.lora_B"].clone()

    def get_residuals(self, layer_name: str,
                      x: torch.Tensor) -> torch.Tensor:
        """Extract LoRA residual BA·x for a specific layer (for E6 probe)."""
        safe_name = layer_name.replace(".", "_")
        if safe_name in self.lora_layers:
            return self.lora_layers[safe_name].get_lora_residual(x)
        raise KeyError(f"Layer {layer_name} not in adapter {self.adapter_name}")


class FedMULoRAModel(nn.Module):
    """
    FedMUModel with three independent LoRA adapters.

    This is the complete model used throughout the Fed-MU experiments.
    All three adapters train during pre-training.
    Only A× trains during unlearning (A_v and A_l are frozen).

    Adapter placement:
      A_v: CLIP vision encoder → attention q_proj, v_proj
      A_l: DistilBERT → attention q_proj, v_proj
      A×:  CrossModalProjection → layer1, layer2
    """

    def __init__(
        self,
        base_model: Optional[FedMUModel] = None,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        # Build or accept the base model
        if base_model is None:
            self.base = FedMUModel(freeze_encoders=False)
        else:
            self.base = base_model
            # Freeze base encoder weights — LoRA layers will be unfrozen separately
            for param in self.base.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.base.text_encoder.parameters():
                param.requires_grad = False

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # ── Create three adapter groups ─────────────────────────────────
        self.adapter_v     = LoRAAdapter("A_v")
        self.adapter_l     = LoRAAdapter("A_l")
        self.adapter_cross = LoRAAdapter("A_cross")

        # ── Attach A_v to CLIP vision encoder ───────────────────────────
        self._attach_lora_to_clip(lora_rank, lora_alpha, lora_dropout)

        # ── Attach A_l to DistilBERT text encoder ───────────────────────
        self._attach_lora_to_distilbert(lora_rank, lora_alpha, lora_dropout)

        # ── Attach A× to CrossModalProjection ───────────────────────────
        self._attach_lora_to_projection(lora_rank, lora_alpha, lora_dropout)

        print(self._summary())

    def _attach_lora_to_clip(self, rank, alpha, dropout):
        """
        Replace q_proj and v_proj in CLIP's last 2 transformer layers
        with LoRA-adapted versions and register them in adapter_v.
        """
        vision_layers = self.base.vision_encoder.vision_model.encoder.layers
        # Target the last 2 layers (highest-level features)
        target_layer_indices = list(range(len(vision_layers) - 2,
                                          len(vision_layers)))

        for layer_idx in target_layer_indices:
            layer = vision_layers[layer_idx]
            attn = layer.self_attn

            for proj_name in ["q_proj", "v_proj"]:
                original = getattr(attn, proj_name)
                lora_layer = LoRALinear(
                    in_features=original.in_features,
                    out_features=original.out_features,
                    rank=rank,
                    lora_alpha=alpha,
                    dropout=dropout,
                    base_layer=original,
                )
                setattr(attn, proj_name, lora_layer)
                key = f"clip_layer{layer_idx}_{proj_name}"
                self.adapter_v.add_layer(key, lora_layer)

    def _attach_lora_to_distilbert(self, rank, alpha, dropout):
        """
        Replace q_lin and k_lin (equivalent to q_proj, v_proj)
        in DistilBERT's last 2 transformer layers.
        """
        bert_layers = self.base.text_encoder.transformer.layer
        target_layer_indices = list(range(len(bert_layers) - 2,
                                          len(bert_layers)))

        for layer_idx in target_layer_indices:
            layer = bert_layers[layer_idx]
            attn = layer.attention

            # DistilBERT uses q_lin, k_lin, v_lin (not q_proj)
            for proj_name in ["q_lin", "v_lin"]:
                original = getattr(attn, proj_name)
                lora_layer = LoRALinear(
                    in_features=original.in_features,
                    out_features=original.out_features,
                    rank=rank,
                    lora_alpha=alpha,
                    dropout=dropout,
                    base_layer=original,
                )
                setattr(attn, proj_name, lora_layer)
                key = f"bert_layer{layer_idx}_{proj_name}"
                self.adapter_l.add_layer(key, lora_layer)

    def _attach_lora_to_projection(self, rank, alpha, dropout):
        """
        Replace CrossModalProjection's layer1 and layer2 with
        LoRA-adapted versions. This is the A× adapter — the
        primary unlearning target.
        """
        proj = self.base.projection
        in1  = proj.layer1.in_features
        out1 = proj.layer1.out_features
        in2  = proj.layer2.in_features
        out2 = proj.layer2.out_features

        lora_layer1 = LoRALinear(
            in_features=in1, out_features=out1,
            rank=rank, lora_alpha=alpha, dropout=dropout,
            base_layer=proj.layer1,
        )
        lora_layer2 = LoRALinear(
            in_features=in2, out_features=out2,
            rank=rank, lora_alpha=alpha, dropout=dropout,
            base_layer=proj.layer2,
        )

        proj.layer1 = lora_layer1
        proj.layer2 = lora_layer2

        self.adapter_cross.add_layer("projection_layer1", lora_layer1)
        self.adapter_cross.add_layer("projection_layer2", lora_layer2)

    def _summary(self) -> str:
        def count(adapter):
            return sum(
                p.numel()
                for layer in adapter.lora_layers.values()
                for p in [layer.lora_A, layer.lora_B]
            )
        nv = count(self.adapter_v)
        nl = count(self.adapter_l)
        nc = count(self.adapter_cross)
        total = nv + nl + nc
        return (
            f"\n  LoRA Adapters attached:"
            f"\n    A_v  ({len(self.adapter_v.lora_layers)} layers): "
            f"{nv:,} params"
            f"\n    A_l  ({len(self.adapter_l.lora_layers)} layers): "
            f"{nl:,} params"
            f"\n    A×   ({len(self.adapter_cross.lora_layers)} layers): "
            f"{nc:,} params"
            f"\n    Total LoRA params: {total:,}"
        )

    # ── Adapter parameter helpers ──────────────────────────────────────────

    def named_lora_parameters(self):
        """
        Yield (name, param) for every LoRA parameter, with names that
        include the adapter group (adapter_v, adapter_l, adapter_cross).

        Use this instead of named_parameters() when you need to identify
        which adapter a parameter belongs to.  PyTorch's named_parameters()
        deduplicates by object identity and always yields the self.base.*
        path first, so adapter group names never appear there.
        """
        seen = set()
        for adapter_attr, adapter in [
            ("adapter_v",     self.adapter_v),
            ("adapter_l",     self.adapter_l),
            ("adapter_cross", self.adapter_cross),
        ]:
            for layer_key, layer in adapter.lora_layers.items():
                for param_name, param in [("lora_A", layer.lora_A),
                                          ("lora_B", layer.lora_B)]:
                    pid = id(param)
                    if pid not in seen:
                        seen.add(pid)
                        full_name = (f"{adapter_attr}.lora_layers"
                                     f".{layer_key}.{param_name}")
                        yield full_name, param

    def is_adapter_trainable(self, adapter_name: str) -> bool:
        """
        Return True if ANY parameter in the named adapter has
        requires_grad=True.  adapter_name must be one of:
        'adapter_v', 'adapter_l', 'adapter_cross'.
        """
        adapter = getattr(self, adapter_name)
        return any(
            p.requires_grad
            for layer in adapter.lora_layers.values()
            for p in [layer.lora_A, layer.lora_B]
        )

    # ── Freeze / unfreeze controls ─────────────────────────────────────────

    def set_unlearning_mode(self):
        """
        Prepare for unlearning: freeze A_v and A_l, keep only A× trainable.
        Base model and classifier are also frozen.

        This is Step 1 of the LoRA-Subtraction algorithm.
        """
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only A×
        self.adapter_cross.unfreeze()

        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"  Unlearning mode: only A× trainable ({trainable:,} params)")

    def set_pretraining_mode(self):
        """
        Prepare for pre-training: all three LoRA adapters + classifier train.
        Base encoders stay frozen.
        """
        # Freeze base encoders
        for param in self.base.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.base.text_encoder.parameters():
            param.requires_grad = False

        # Unfreeze all LoRA adapters
        self.adapter_v.unfreeze()
        self.adapter_l.unfreeze()
        self.adapter_cross.unfreeze()

        # Unfreeze classifier
        for param in self.base.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"  Pre-training mode: all adapters + classifier "
              f"trainable ({trainable:,} params)")

    # ── Gradient norm queries (for ρ(p)) ──────────────────────────────────

    def get_adapter_gradient_norms(self) -> Dict[str, Optional[float]]:
        """
        Return the Frobenius norm of gradients for each adapter.

        This is used to compute ρ(p):
          ρ(p) = ||∇_{A×} L||_F / (||∇_{Av} L||_F + ||∇_{Al} L||_F)

        Call this AFTER loss.backward() and BEFORE optimizer.step().
        """
        return {
            "A_v":    self.adapter_v.get_total_gradient_norm(),
            "A_l":    self.adapter_l.get_total_gradient_norm(),
            "A_cross": self.adapter_cross.get_total_gradient_norm(),
        }

    # ── State dict helpers (for federated parameter passing) ──────────────

    def get_cross_adapter_state(self) -> dict:
        """Export A× parameters only (sent to server during aggregation)."""
        return self.adapter_cross.get_state_dict()

    def set_cross_adapter_state(self, state: dict):
        """Load A× parameters (received from server after aggregation)."""
        self.adapter_cross.load_state_dict_lora(state)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> dict:
        """
        Forward pass delegates to the base model.
        LoRA layers are already in-place inside the base model,
        so this is transparent.
        """
        return self.base(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_embeddings=return_embeddings,
        )