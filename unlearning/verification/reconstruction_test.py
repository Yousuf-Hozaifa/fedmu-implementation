"""
unlearning/verification/reconstruction_test.py

Tests whether the model can reconstruct the forgotten patient's
report from their X-ray image after unlearning.

A model that truly forgot patient_0000 should be unable to produce
their specific report with above-chance similarity.

Method:
  1. Before unlearning: given patient_0000's X-ray, measure how
     similar the model's prediction is to their actual report
     (BLEU score + exact label match)
  2. After unlearning: repeat the same measurement
  3. A significant drop in similarity = evidence of forgetting

We use a simplified "reconstruction" since our toy model outputs
class logits rather than free text. We measure:
  - Label reconstruction: how accurately predicted labels match true labels
  - Feature similarity: cosine similarity of output embeddings to baseline
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ReconstructionTester:
    """
    Tests whether the model can reconstruct patient-specific outputs.

    Two measurements:
      1. Label reconstruction accuracy (classification)
      2. Embedding similarity (how similar the model's internal
         representation is to the pre-unlearning representation)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def _get_outputs(
        self,
        model,
        sample: dict,
    ) -> Dict:
        """Get model outputs for a single patient sample."""
        model.eval()
        with torch.no_grad():
            pixel_values = sample["image"].unsqueeze(0).to(self.device)
            input_ids    = sample["text_input_ids"].unsqueeze(0).to(self.device)
            attn_mask    = sample["text_attention_mask"].unsqueeze(0).to(self.device)
            labels       = sample["labels"].unsqueeze(0).to(self.device)

            outputs = self.model_call(
                model, pixel_values, input_ids, attn_mask, labels
            )

        probs   = torch.sigmoid(outputs["logits"].squeeze(0))
        preds   = (probs > 0.5).float()
        true    = labels.squeeze(0)
        correct = (preds == true).float()

        return {
            "logits":          outputs["logits"].squeeze(0).cpu(),
            "probs":           probs.cpu(),
            "preds":           preds.cpu(),
            "true_labels":     true.cpu(),
            "label_accuracy":  correct.mean().item(),
            "loss":            outputs["loss"].item(),
            "visual_emb":      outputs.get("visual_emb",
                               torch.zeros(1)).squeeze().cpu(),
            "projected_emb":   outputs.get("projected_emb",
                               torch.zeros(1)).squeeze().cpu(),
        }

    def model_call(self, model, pv, ii, am, lb):
        """Wrapper to call model with return_embeddings=True."""
        return model(
            pixel_values=pv,
            input_ids=ii,
            attention_mask=am,
            labels=lb,
            return_embeddings=True,
        )

    def measure(
        self,
        model,
        target_sample:  dict,
        baseline_outputs: Optional[Dict] = None,
    ) -> Dict:
        """
        Measure reconstruction capability for a single patient.

        Args:
            model:             the model to test
            target_sample:     patient's data
            baseline_outputs:  pre-unlearning outputs (for comparison)

        Returns:
            dict with label_accuracy, embedding_similarity, loss
        """
        outputs = self._get_outputs(model, target_sample)

        result = {
            "label_accuracy": outputs["label_accuracy"],
            "loss":           outputs["loss"],
            "probs":          outputs["probs"].tolist(),
        }

        if baseline_outputs is not None:
            # Cosine similarity of projected embeddings (cross-modal binding)
            proj_base  = baseline_outputs["projected_emb"]
            proj_after = outputs["projected_emb"]

            if proj_base.shape == proj_after.shape and proj_base.norm() > 1e-8:
                cos_sim = F.cosine_similarity(
                    proj_base.unsqueeze(0),
                    proj_after.unsqueeze(0),
                ).item()
            else:
                cos_sim = None

            result["embedding_similarity"] = cos_sim

            # Logit correlation
            log_base  = baseline_outputs["logits"]
            log_after = outputs["logits"]
            if log_base.shape == log_after.shape:
                corr = torch.corrcoef(
                    torch.stack([log_base, log_after])
                )[0, 1].item()
                result["logit_correlation"] = corr
            else:
                result["logit_correlation"] = None

        return result

    def compare_before_after(
        self,
        model_before,
        model_after,
        target_sample: dict,
    ) -> Dict:
        """
        Full before/after reconstruction comparison.

        A successful unlearning should show:
          - Decreased label accuracy on target patient's data
          - Decreased embedding similarity
          - Increased loss on target patient's data
        """
        print("  Measuring reconstruction BEFORE unlearning...")
        baseline = self._get_outputs(model_before, target_sample)
        before   = self.measure(model_before, target_sample, None)

        print("  Measuring reconstruction AFTER unlearning...")
        after    = self.measure(model_after, target_sample, baseline)

        # Compute changes
        delta_acc  = after["label_accuracy"] - before["label_accuracy"]
        delta_loss = after["loss"]           - before["loss"]

        forgetting_detected = delta_loss > 0.01  # Loss increased = model forgot

        return {
            "before": before,
            "after":  after,
            "delta_label_accuracy":       delta_acc,
            "delta_loss":                 delta_loss,
            "embedding_similarity_after": after.get("embedding_similarity"),
            "logit_correlation_after":    after.get("logit_correlation"),
            "forgetting_detected":        forgetting_detected,
            "interpretation": (
                f"Loss change: {delta_loss:+.4f}, "
                f"Accuracy change: {delta_acc:+.4f}. "
                + ("✓ Forgetting detected: loss increased on target patient."
                   if forgetting_detected else
                   "✗ No clear forgetting signal — loss unchanged or decreased.")
            ),
        }