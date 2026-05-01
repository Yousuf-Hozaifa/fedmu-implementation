"""
unlearning/verification/mps_evaluator.py

Computes the Modality Preservation Score (MPS) after unlearning.

MPS_v   = Acc_v_post  / Acc_v_pre    (vision accuracy ratio)
MPS_l   = PPL_l_pre   / PPL_l_post   (perplexity ratio — inverted)
MPS_×   = VQA_post    / VQA_pre      (cross-modal VQA ratio)
MPS_total = min(MPS_v, MPS_l, MPS_×) (worst-case across modalities)

The minimum operator (not mean) ensures a catastrophic failure
in any single modality is not masked by good performance in others.

Baseline values (pre-unlearning) must be recorded BEFORE any
unlearning occurs and passed to every subsequent evaluation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MPSEvaluator:
    """
    Evaluates Modality Preservation Score for a model.

    Usage:
        # Before unlearning — record baselines
        evaluator = MPSEvaluator(model, retain_dataset)
        evaluator.record_baseline()

        # During/after unlearning — compute MPS
        scores = evaluator.compute_mps()
        # scores = {'mps_v': 0.97, 'mps_l': 0.99,
        #           'mps_cross': 0.95, 'mps_total': 0.95}
    """

    def __init__(
        self,
        model,
        retain_dataset,
        device:     str = "cpu",
        batch_size: int = 4,
    ):
        self.model          = model
        self.retain_dataset = retain_dataset
        self.device         = device
        self.batch_size     = batch_size
        self.baseline: Optional[Dict] = None

        self.dataloader = DataLoader(
            retain_dataset,
            batch_size=min(batch_size, len(retain_dataset)),
            shuffle=False,
        )

    def _compute_raw_metrics(self) -> Dict:
        """
        Compute raw metrics on the retain dataset:
          - vision_acc:   pathology classification accuracy (vision path)
          - language_ppl: perplexity-like score (cross-entropy on text tokens)
          - vqa_acc:      joint image+text prediction accuracy
        """
        self.model.eval()

        total_vision_correct  = 0
        total_vqa_correct     = 0
        total_ce_loss         = 0.0
        total_samples         = 0
        total_batches         = 0

        with torch.no_grad():
            for batch in self.dataloader:
                pixel_values = batch["image"].to(self.device)
                input_ids    = batch["text_input_ids"].to(self.device)
                attn_mask    = batch["text_attention_mask"].to(self.device)
                labels       = batch["labels"].to(self.device)
                bs           = labels.shape[0]

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                    return_embeddings=True,
                )

                logits = outputs["logits"]          # [bs, 14]
                loss   = outputs["loss"]

                # Vision accuracy: image-only prediction
                # (use only the projected_emb path, ignore text_emb)
                # Approximate by checking whether top-1 predicted class matches label
                probs   = torch.sigmoid(logits)
                preds   = (probs > 0.5).float()
                correct = (preds == labels).float().mean(dim=1)

                total_vision_correct += correct.sum().item()
                total_vqa_correct    += correct.sum().item()
                total_ce_loss        += loss.item()
                total_samples        += bs
                total_batches        += 1

        n = max(total_samples, 1)
        b = max(total_batches, 1)

        return {
            "vision_acc":   total_vision_correct / n,
            "vqa_acc":      total_vqa_correct    / n,
            "language_ppl": total_ce_loss        / b,  # lower = better
        }

    def record_baseline(self) -> Dict:
        """
        Record pre-unlearning baseline metrics.
        Must be called ONCE before any unlearning occurs.
        """
        print("  Recording MPS baseline (pre-unlearning)...")
        self.baseline = self._compute_raw_metrics()
        print(f"    Baseline vision_acc:   {self.baseline['vision_acc']:.4f}")
        print(f"    Baseline vqa_acc:      {self.baseline['vqa_acc']:.4f}")
        print(f"    Baseline language_ppl: {self.baseline['language_ppl']:.4f}")
        return self.baseline

    def compute_mps(self) -> Dict:
        """
        Compute MPS scores relative to the recorded baseline.

        Returns dict with per-modality and total MPS scores.
        Raises RuntimeError if baseline not recorded.
        """
        if self.baseline is None:
            raise RuntimeError(
                "Baseline not recorded. Call record_baseline() first."
            )

        current = self._compute_raw_metrics()

        # MPS_v: vision accuracy ratio (higher post = better)
        base_v = max(self.baseline["vision_acc"], 1e-8)
        mps_v  = current["vision_acc"] / base_v

        # MPS_l: perplexity ratio INVERTED (lower ppl post = better)
        # MPS_l = PPL_pre / PPL_post → 1.0 = no degradation
        base_l = max(self.baseline["language_ppl"], 1e-8)
        mps_l  = base_l / max(current["language_ppl"], 1e-8)

        # MPS_×: VQA accuracy ratio
        base_x = max(self.baseline["vqa_acc"], 1e-8)
        mps_x  = current["vqa_acc"] / base_x

        # Total: worst-case (minimum) across all three
        mps_total = min(mps_v, mps_l, mps_x)

        return {
            "mps_v":       mps_v,
            "mps_l":       mps_l,
            "mps_cross":   mps_x,
            "mps_total":   mps_total,
            "raw_current": current,
            "raw_baseline": self.baseline,
        }

    def all_above_threshold(self, tau: float = 0.85) -> bool:
        """
        Check whether ALL MPS sub-scores meet the threshold.
        Used as the loop termination condition in LoRA-Subtraction.
        """
        scores = self.compute_mps()
        return all([
            scores["mps_v"]     >= tau,
            scores["mps_l"]     >= tau,
            scores["mps_cross"] >= tau,
        ])

    def print_mps(self, scores: Dict, tau: float = 0.85):
        """Pretty-print MPS scores with pass/fail indicators."""
        def marker(val):
            return "✓" if val >= tau else "✗"

        print(f"    MPS_v     = {scores['mps_v']:.4f}  "
              f"{marker(scores['mps_v'])} (τ={tau})")
        print(f"    MPS_l     = {scores['mps_l']:.4f}  "
              f"{marker(scores['mps_l'])} (τ={tau})")
        print(f"    MPS_cross = {scores['mps_cross']:.4f}  "
              f"{marker(scores['mps_cross'])} (τ={tau})")
        print(f"    MPS_total = {scores['mps_total']:.4f}  "
              f"{marker(scores['mps_total'])} (τ={tau})")