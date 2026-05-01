"""
unlearning/verification/kl_verifier.py

KL-divergence ratio verification (Layer 3 of the 3-layer check).

Compares the unlearned model's output distribution to a reference
model trained WITHOUT the target patient.

In the full Fed-MU system, the reference would be a proper SISA
retrained model. In the toy setting, we simulate the reference by
training a fresh model on the retention dataset only.

KL ratio formula:
  KL_ratio = KL(unlearned_model || reference_model)
             ─────────────────────────────────────────
             KL(random_model || reference_model)

  Target: KL_ratio < 0.10
  Interpretation: the unlearned model is close to the reference,
  relative to a random baseline.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import sys
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_kl_divergence(
    probs_p: torch.Tensor,
    probs_q: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """
    Compute KL(P || Q) = Σ P * log(P/Q)

    For multi-label classification, we compute per-class KL and average.

    Args:
        probs_p: predicted probabilities from model P  [n, 14]
        probs_q: predicted probabilities from model Q  [n, 14]

    Returns:
        scalar KL divergence value
    """
    p = probs_p.clamp(eps, 1 - eps)
    q = probs_q.clamp(eps, 1 - eps)

    # KL for Bernoulli: p*log(p/q) + (1-p)*log((1-p)/(1-q))
    kl = (p * (p / q).log() +
          (1 - p) * ((1 - p) / (1 - q)).log())

    return kl.mean().item()


class KLVerifier:
    """
    Computes KL-divergence ratio between:
      - the unlearned model
      - a reference model (trained without the target patient)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def collect_probs(
        self,
        model,
        dataset,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        Collect output probabilities from a model on a dataset.

        Returns:
            probs: [n_patients, 14] tensor of sigmoid probabilities
        """
        model.eval()
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
        )
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["image"].to(self.device)
                input_ids    = batch["text_input_ids"].to(self.device)
                attn_mask    = batch["text_attention_mask"].to(self.device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                )
                probs = torch.sigmoid(outputs["logits"])
                all_probs.append(probs.cpu())

        return torch.cat(all_probs, dim=0)   # [n, 14]

    def train_reference_model(
        self,
        global_model_class,
        retention_dataset,
        config: dict,
        n_steps: int = 20,
    ):
        """
        Train a lightweight reference model on the retention dataset.
        This simulates what SISA retraining would produce.

        In the toy setting: we fine-tune the base model for a few steps.
        """
        import torch.optim as optim
        from models.multimodal_model import FedMUModel
        from models.lora_model import FedMULoRAModel

        print("  Training reference model (retention only, no target patient)...")

        base = FedMUModel(freeze_encoders=False)
        ref_model = FedMULoRAModel(
            base_model=base,
            lora_rank=config["lora"]["rank"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
        )
        ref_model.set_pretraining_mode()

        loader = DataLoader(
            retention_dataset,
            batch_size=min(4, len(retention_dataset)),
            shuffle=True,
        )
        optimizer = optim.AdamW(
            [p for p in ref_model.parameters() if p.requires_grad],
            lr=1e-3,
        )

        ref_model.train()
        for step in range(n_steps):
            batch = next(iter(loader))
            optimizer.zero_grad()

            outputs = ref_model(
                pixel_values=batch["image"],
                input_ids=batch["text_input_ids"],
                attention_mask=batch["text_attention_mask"],
                labels=batch["labels"],
            )
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in ref_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(f"    Reference step {step+1}/{n_steps}: "
                      f"loss={outputs['loss'].item():.4f}")

        return ref_model

    def compute_ratio(
        self,
        unlearned_model,
        reference_model,
        random_model,
        test_dataset,
    ) -> Dict:
        """
        Compute:
          KL_unlearned = KL(unlearned || reference)
          KL_random    = KL(random    || reference)
          ratio        = KL_unlearned / KL_random

        Target: ratio < 0.10
        """
        print("  Collecting probabilities from unlearned model...")
        probs_unlearned  = self.collect_probs(unlearned_model,  test_dataset)

        print("  Collecting probabilities from reference model...")
        probs_reference  = self.collect_probs(reference_model,  test_dataset)

        print("  Collecting probabilities from random model...")
        probs_random     = self.collect_probs(random_model,     test_dataset)

        kl_unlearned = compute_kl_divergence(probs_unlearned, probs_reference)
        kl_random    = compute_kl_divergence(probs_random,    probs_reference)

        # Avoid division by zero
        ratio = kl_unlearned / max(kl_random, 1e-10)

        passes = ratio < 0.10

        return {
            "kl_unlearned_vs_reference": float(kl_unlearned),
            "kl_random_vs_reference":    float(kl_random),
            "kl_ratio":                  float(ratio),
            "passes_threshold":          passes,
            "interpretation": (
                f"KL ratio = {ratio:.4f}. "
                + ("✓ PASSES: unlearned model is close to reference."
                   if passes else
                   f"✗ FAILS: ratio {ratio:.4f} ≥ 0.10. "
                   f"Model output distribution differs from reference.")
            ),
        }