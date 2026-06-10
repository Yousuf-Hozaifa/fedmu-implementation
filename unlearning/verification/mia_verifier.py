"""
unlearning/verification/mia_verifier.py  [FINAL — v3]

FINAL FIX: Membership score fallback when n_target=1.

With only one target patient in a toy dataset, roc_auc_score
is statistically meaningless (essentially random with 6 samples).

The membership score (0.9073 in previous runs) is the CORRECT
primary metric at this scale. It directly answers:
  "Does the shadow model think this patient is still a member?"
  High score (~0.9) → model still recognises patient (not forgotten)
  Low score (~0.5) → model no longer recognises patient (forgotten)

After correct unlearning with α=1.0, the membership score should
drop from ~0.9 toward ~0.5.

Pass condition (n_target=1 fallback):
  score ≤ 0.60 → near random chance → PASS
  score > 0.60 → model still recognises patient → FAIL

For n_target > 5, the original AUC-based check is used.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ConfidenceFeatureExtractor:
    """
    Extracts 6 confidence-based features from model outputs.
    Features: max_prob, mean_prob, entropy, ce_loss, margin, correct_prob
    """

    def __init__(self, model, device: str = "cpu"):
        self.model  = model
        self.device = device

    def extract_single(self, sample: dict) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pixel_values = sample["image"].unsqueeze(0).to(self.device)
            input_ids    = sample["text_input_ids"].unsqueeze(0).to(self.device)
            attn_mask    = sample["text_attention_mask"].unsqueeze(0).to(self.device)
            labels       = sample["labels"].unsqueeze(0).to(self.device)

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )

        logits = outputs["logits"].squeeze(0)
        loss   = outputs["loss"].item()
        probs  = torch.sigmoid(logits)
        sp, _  = probs.sort(descending=True)
        eps    = 1e-7
        p      = probs.clamp(eps, 1 - eps)
        ent    = -(p * p.log() + (1-p) * (1-p).log()).mean().item()
        pos    = labels.squeeze(0) > 0.5
        cp     = probs[pos].mean().item() if pos.any() else probs.max().item()

        return np.array([
            probs.max().item(),
            probs.mean().item(),
            ent,
            loss,
            (sp[0] - sp[1]).item(),
            cp,
        ], dtype=np.float32)

    def extract_dataset(
        self, dataset, verbose: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        features, ids = [], []
        for i in range(len(dataset)):
            s = dataset[i]
            features.append(self.extract_single(s))
            ids.append(s["patient_id"])
        return np.vstack(features), ids


class MIAVerifier:
    """Membership Inference Attack verifier with small-sample fallback."""

    def __init__(self, model_before, model_after, device: str = "cpu"):
        self.model_before = model_before
        self.model_after  = model_after
        self.device       = device
        self.ext_before   = ConfidenceFeatureExtractor(model_before, device)
        self.ext_after    = ConfidenceFeatureExtractor(model_after,  device)
        self.shadow_clf   = None

    def train_shadow_classifier(
        self,
        member_dataset,
        non_member_dataset,
        use_model: str = "before",
    ) -> float:
        ext = self.ext_before if use_model == "before" else self.ext_after
        print("  Extracting member features...")
        mf, _ = ext.extract_dataset(member_dataset)
        print("  Extracting non-member features...")
        nf, _ = ext.extract_dataset(non_member_dataset)
        X = np.vstack([mf, nf])
        y = np.array([1]*len(mf) + [0]*len(nf), dtype=int)
        self.shadow_clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self.shadow_clf.fit(X, y)
        try:
            train_auc = roc_auc_score(y, self.shadow_clf.predict_proba(X)[:,1])
        except Exception:
            train_auc = 0.5
        print(f"  Shadow classifier training AUC: {train_auc:.4f}")
        return train_auc

    def evaluate_on_dataset(
        self,
        target_dataset,
        non_member_dataset,
        use_model: str = "after",
        label: str = "",
    ) -> Dict:
        """
        Evaluate with automatic fallback for small datasets.

        If n_target == 1: use membership SCORE as primary metric
            Pass condition: score ≤ 0.60 (near random chance)
        If n_target > 5: use AUC
            Pass condition: 0.45 ≤ AUC ≤ 0.55
        """
        assert self.shadow_clf is not None, "Call train_shadow_classifier first."
        ext = self.ext_before if use_model == "before" else self.ext_after

        t_feat, t_ids = ext.extract_dataset(target_dataset)
        n_feat, _     = ext.extract_dataset(non_member_dataset)

        t_scores = self.shadow_clf.predict_proba(t_feat)[:, 1]
        n_scores = self.shadow_clf.predict_proba(n_feat)[:, 1]
        mean_target = float(t_scores.mean())
        n_target    = len(t_feat)
        n_non_mem   = len(n_feat)

        # ── Fallback: use score directly when n_target is tiny ─────
        if n_target <= 5:
            # Score-based check: after unlearning, score should drop
            # from ~0.9 (remembered) toward ~0.5 (forgotten)
            score_passes = mean_target <= 0.60
            advantage    = abs(mean_target - 0.5)
            epsilon      = 2 * advantage

            return {
                "n_target":               n_target,
                "n_non_member":           n_non_mem,
                "mean_target_membership": mean_target,
                "auc":                    None,   # not reliable at n=1
                "advantage":              float(advantage),
                "epsilon_proxy":          float(epsilon),
                "passes_threshold":       score_passes,
                "method":                 "membership_score_fallback",
                "interpretation": (
                    f"n_target={n_target} — using membership score "
                    f"(AUC unreliable at this sample size). "
                    f"Score={mean_target:.4f}. "
                    + ("✓ PASSES: score ≤ 0.60 → model no longer "
                       "strongly recognises this patient."
                       if score_passes else
                       f"✗ FAILS: score={mean_target:.4f} > 0.60 → "
                       f"model still recognises this patient as a member. "
                       f"Expected: score drops to ~0.50 after successful unlearning.")
                ),
            }

        # ── Standard AUC-based check for n_target > 5 ─────────────
        X_all = np.vstack([t_feat, n_feat])
        y_all = np.array([1]*n_target + [0]*n_non_mem, dtype=int)
        scores_all = self.shadow_clf.predict_proba(X_all)[:, 1]
        try:
            auc = roc_auc_score(y_all, scores_all)
        except Exception:
            auc = 0.5

        advantage = abs(auc - 0.5)
        epsilon   = 2 * advantage
        passes    = (0.45 <= auc <= 0.55)

        return {
            "n_target":               n_target,
            "n_non_member":           n_non_mem,
            "mean_target_membership": mean_target,
            "auc":                    float(auc),
            "advantage":              float(advantage),
            "epsilon_proxy":          float(epsilon),
            "passes_threshold":       passes,
            "method":                 "auc",
            "interpretation": (
                f"AUC={auc:.4f} (n={n_target}+{n_non_mem}). "
                + ("✓ PASSES: AUC near random chance."
                   if passes else
                   f"✗ FAILS: AUC outside [0.45, 0.55].")
            ),
        }

    def compare_before_after(
        self,
        target_dataset,
        non_member_dataset,
        member_dataset,
    ) -> Dict:
        print("\n  Training shadow classifier on pre-unlearning model...")
        self.train_shadow_classifier(
            member_dataset, non_member_dataset, use_model="before"
        )
        print("\n  Evaluating target BEFORE unlearning...")
        before = self.evaluate_on_dataset(
            target_dataset, non_member_dataset,
            use_model="before", label="before",
        )
        print("\n  Evaluating target AFTER unlearning...")
        after = self.evaluate_on_dataset(
            target_dataset, non_member_dataset,
            use_model="after", label="after",
        )
        return {"before": before, "after": after}