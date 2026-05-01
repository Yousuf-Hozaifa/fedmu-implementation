"""
unlearning/verification/mia_verifier.py

Membership Inference Attack verification for machine unlearning.

The MIA asks: "Can an attacker determine whether patient_0000
was in the training data?" After successful unlearning, the
model should behave identically on patient_0000's data as on
data it never saw — making the attack no better than guessing.

Method (shadow model approach):
  1. Collect confidence features from the model for:
       MEMBERS:     patients the model WAS trained on
       NON-MEMBERS: patients the model was NOT trained on
  2. Train a binary classifier to distinguish members/non-members
  3. Apply to the target patient → check their membership score
  4. Compare: before unlearning vs after unlearning

Target: AUC ≤ 0.55 after unlearning
This implies MIA advantage γ = |AUC - 0.5| ≤ 0.05
And via the ε-bridge: ε ≲ 2γ ≤ 0.10
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ConfidenceFeatureExtractor:
    """
    Extracts confidence-based features from model outputs.
    These features are what the MIA shadow classifier uses.

    Features per sample:
      1. max_prob:      highest sigmoid probability across 14 classes
      2. mean_prob:     mean probability across all 14 classes
      3. entropy:       prediction entropy (higher = more uncertain)
      4. loss:          cross-entropy loss value
      5. margin:        gap between top-2 probabilities
      6. correct_prob:  probability on the true positive class

    High-confidence, low-entropy, low-loss = model likely MEMORISED this sample
    Low-confidence, high-entropy, high-loss = model likely NEVER SAW this sample
    """

    def __init__(self, model, device: str = "cpu"):
        self.model  = model
        self.device = device

    def extract_single(self, sample: dict) -> np.ndarray:
        """
        Extract features for a single patient sample.
        Returns a 1D numpy array of shape (6,).
        """
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

            logits = outputs["logits"].squeeze(0)  # [14]
            loss   = outputs["loss"].item()

            probs = torch.sigmoid(logits)          # [14]
            sorted_probs, _ = probs.sort(descending=True)

            # Feature 1: max probability
            max_prob = probs.max().item()

            # Feature 2: mean probability
            mean_prob = probs.mean().item()

            # Feature 3: prediction entropy
            # H = -Σ p*log(p) - (1-p)*log(1-p) for each label
            eps    = 1e-7
            p      = probs.clamp(eps, 1 - eps)
            ent    = -(p * p.log() + (1 - p) * (1 - p).log()).mean().item()
            entropy = ent

            # Feature 4: cross-entropy loss
            ce_loss = loss

            # Feature 5: margin between top-2 predictions
            margin = (sorted_probs[0] - sorted_probs[1]).item()

            # Feature 6: mean probability on positive classes
            pos_mask = (labels.squeeze(0) > 0.5)
            if pos_mask.any():
                correct_prob = probs[pos_mask].mean().item()
            else:
                correct_prob = probs.max().item()

        return np.array([
            max_prob, mean_prob, entropy, ce_loss, margin, correct_prob
        ], dtype=np.float32)

    def extract_dataset(
        self,
        dataset,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for all patients in a dataset.

        Returns:
            features:    (n_patients, 6) numpy array
            patient_ids: list of patient IDs in same order
        """
        features    = []
        patient_ids = []

        for i in range(len(dataset)):
            sample = dataset[i]
            feat   = self.extract_single(sample)
            features.append(feat)
            patient_ids.append(sample["patient_id"])

            if verbose and (i + 1) % 10 == 0:
                print(f"    Extracted features: {i+1}/{len(dataset)}")

        return np.vstack(features), patient_ids


class MIAVerifier:
    """
    Full Membership Inference Attack verifier.

    Trains a shadow classifier on known members/non-members,
    then tests whether the target patient is still a member
    after unlearning.
    """

    def __init__(
        self,
        model_before,     # model BEFORE unlearning
        model_after,      # model AFTER unlearning
        device: str = "cpu",
    ):
        self.model_before = model_before
        self.model_after  = model_after
        self.device       = device
        self.extractor_before = ConfidenceFeatureExtractor(model_before, device)
        self.extractor_after  = ConfidenceFeatureExtractor(model_after,  device)
        self.shadow_clf       = None

    def train_shadow_classifier(
        self,
        member_dataset,       # patients model WAS trained on
        non_member_dataset,   # patients model was NOT trained on
        use_model: str = "before",
    ):
        """
        Train the shadow binary classifier.

        member_label    = 1  (was in training)
        non_member_label = 0 (was NOT in training)

        Args:
            use_model: 'before' or 'after' — which model to extract from
        """
        extractor = (self.extractor_before if use_model == "before"
                     else self.extractor_after)

        print("  Extracting member features...")
        mem_features, _ = extractor.extract_dataset(member_dataset)

        print("  Extracting non-member features...")
        non_features, _ = extractor.extract_dataset(non_member_dataset)

        X = np.vstack([mem_features, non_features])
        y = np.array(
            [1] * len(mem_features) + [0] * len(non_features),
            dtype=int,
        )

        # Logistic Regression: interpretable and fast on CPU
        self.shadow_clf = LogisticRegression(
            max_iter=500, C=1.0, random_state=42
        )
        self.shadow_clf.fit(X, y)

        # Training AUC (upper bound — tells us the attack's max power)
        train_preds = self.shadow_clf.predict_proba(X)[:, 1]
        train_auc   = roc_auc_score(y, train_preds)
        print(f"  Shadow classifier training AUC: {train_auc:.4f}")
        print(f"  (This is the attack's capability — "
              f"higher = attacker is stronger)")

        return train_auc

    def test_target_patient(
        self,
        target_sample:     dict,
        non_member_sample: dict,
        use_model: str = "after",
    ) -> Dict:
        """
        Test whether the target patient is still a "member"
        according to the shadow classifier.

        Args:
            target_sample:     the patient to forget
            non_member_sample: a held-out patient as negative reference
            use_model: which model to extract from ('before' or 'after')

        Returns:
            dict with membership_score, auc, advantage, interpretation
        """
        if self.shadow_clf is None:
            raise RuntimeError("Call train_shadow_classifier() first.")

        extractor = (self.extractor_before if use_model == "before"
                     else self.extractor_after)

        # Extract features for target (true label = 1, was a member)
        target_feat   = extractor.extract_single(target_sample)

        # Extract features for non-member reference (true label = 0)
        non_mem_feat  = extractor.extract_single(non_member_sample)

        # Get membership scores
        both_feats   = np.vstack([target_feat, non_mem_feat])
        scores       = self.shadow_clf.predict_proba(both_feats)[:, 1]

        target_score = scores[0]   # membership score for the target
        non_mem_score = scores[1]  # membership score for non-member

        # AUC with just these 2 samples (simplified — real E4 uses more)
        try:
            auc = roc_auc_score([1, 0], [target_score, non_mem_score])
        except Exception:
            auc = 0.5  # degenerate case

        advantage = abs(auc - 0.5)
        epsilon   = 2 * advantage  # ε-bridge conjecture

        return {
            "target_membership_score": float(target_score),
            "non_member_score":        float(non_mem_score),
            "auc":                     float(auc),
            "advantage":               float(advantage),
            "epsilon_proxy":           float(epsilon),
            "passes_threshold":        auc <= 0.55,
            "interpretation": (
                f"AUC={auc:.4f} → Adv={advantage:.4f} → ε≲{epsilon:.4f}. "
                + ("✓ PASSES: model appears to have forgotten the patient."
                   if auc <= 0.55 else
                   "✗ FAILS: model may still remember the patient.")
            ),
        }

    def compare_before_after(
        self,
        target_sample:     dict,
        non_member_sample: dict,
        member_dataset,
        non_member_dataset,
    ) -> Dict:
        """
        Full before/after comparison:
          1. Train shadow model on pre-unlearning model
          2. Test target before unlearning (should show membership)
          3. Test target after unlearning (should show non-membership)
        """
        print("\n  Training shadow classifier on pre-unlearning model...")
        self.train_shadow_classifier(
            member_dataset, non_member_dataset,
            use_model="before",
        )

        print("\n  Testing target patient BEFORE unlearning...")
        before = self.test_target_patient(
            target_sample, non_member_sample,
            use_model="before",
        )

        print("\n  Testing target patient AFTER unlearning...")
        after = self.test_target_patient(
            target_sample, non_member_sample,
            use_model="after",
        )

        return {"before": before, "after": after}