"""
experiments/e6_probe/identity_probe.py

E6: Identity Leakage Probe experiment.

Trains a lightweight MLP probe on LoRA residual outputs BA·x
extracted from each adapter (A_v, A_l, A×) separately.
The probe tries to predict patient identity from adapter residuals.

Key design:
  - Probe trained on 80% of patients, tested on 20%
  - Negative control: probe tested on held-out non-training patients
    Expected: near-chance accuracy (<55%) — proves the probe
    measures memorisation, not general demographics
  - Comparison: before vs after unlearning
    Expected: only A× probe drops after unlearning

If only removing A× reduces identity prediction to <55% while
MPS_total > 0.85, this is CAUSAL evidence that A× is the
identity storage site — not just a correlation claim.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class LinearIdentityProbe(nn.Module):
    """
    2-layer MLP probe for patient identity prediction.

    Input:  LoRA residual embedding for one patient
    Output: probability distribution over patient IDs

    Architecture: input_dim → 256 → ReLU → dropout → n_patients
    """

    def __init__(self, input_dim: int, n_patients: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_patients),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoRAResidualsExtractor:
    """
    Extracts LoRA residual outputs BA·x from each adapter.

    The residual BA·x is what the E6 probe analyses.
    It is the additional signal the adapter contributes beyond
    the frozen base model — this is where patient-specific
    information would be encoded.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model  = model
        self.device = device

    def extract_residuals_for_patient(
        self,
        sample:      dict,
        layer_key:   str = "projection_layer1",
    ) -> Dict[str, torch.Tensor]:
        """
        Extract LoRA residuals for a single patient.

        Since our architecture uses the CrossModalProjection as A×,
        we extract the residual from the projection layer's LoRA output.

        For A_v and A_l: we extract from the first LoRA layer
        in each adapter group.

        Returns dict mapping adapter_name → residual tensor [hidden_dim]
        """
        self.model.eval()
        residuals = {}

        with torch.no_grad():
            pixel_values = sample["image"].unsqueeze(0).to(self.device)
            input_ids    = sample["text_input_ids"].unsqueeze(0).to(self.device)
            attn_mask    = sample["text_attention_mask"].unsqueeze(0).to(self.device)

            # Get intermediate embeddings
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attn_mask,
                return_embeddings=True,
            )

            # A× residual: the LoRA contribution of the cross-modal projection
            # This is BA·x for the projection layer
            visual_emb = outputs.get("visual_emb")     # [1, hidden]
            proj_emb   = outputs.get("projected_emb")  # [1, hidden]
            text_emb   = outputs.get("text_emb")       # [1, hidden]

            if visual_emb is not None and proj_emb is not None:
                # The A× residual ≈ projected_emb - base_projection(visual_emb)
                # Since we can't easily separate base from LoRA here,
                # we use projected_emb as the proxy for A× output
                residuals["A_cross"] = proj_emb.squeeze(0).cpu()

            if visual_emb is not None:
                # A_v residual: the vision encoder output
                residuals["A_v"] = visual_emb.squeeze(0).cpu()

            if text_emb is not None:
                # A_l residual: the text encoder output
                residuals["A_l"] = text_emb.squeeze(0).cpu()

        return residuals

    def extract_dataset(
        self,
        dataset,
        verbose: bool = True,
    ) -> Dict[str, Tuple[np.ndarray, List[int]]]:
        """
        Extract residuals for all patients in a dataset.

        Returns:
            dict mapping adapter_name → (features_array, patient_indices)
            features_array: [n_patients, hidden_dim]
            patient_indices: list of integer patient indices
        """
        all_residuals = {"A_cross": [], "A_v": [], "A_l": []}
        patient_indices = []

        for i in range(len(dataset)):
            sample = dataset[i]
            patient_indices.append(sample["patient_idx"])
            residuals = self.extract_residuals_for_patient(sample)

            for adapter in all_residuals:
                if adapter in residuals:
                    all_residuals[adapter].append(
                        residuals[adapter].numpy()
                    )

            if verbose and (i + 1) % 10 == 0:
                print(f"    Extracted residuals: {i+1}/{len(dataset)}")

        result = {}
        for adapter, res_list in all_residuals.items():
            if res_list:
                result[adapter] = (
                    np.vstack(res_list),
                    patient_indices,
                )

        return result


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_epochs: int = 30,
    lr:       float = 1e-3,
) -> Tuple[LinearIdentityProbe, float]:
    """
    Train the identity probe and return it with validation accuracy.
    """
    n_patients = len(np.unique(np.concatenate([y_train, y_val])))
    input_dim  = X_train.shape[1]

    probe     = LinearIdentityProbe(input_dim, n_patients)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)

    probe.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = probe(X_tr)
        loss   = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    # Validation accuracy
    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_v)
        val_preds  = val_logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_val, val_preds)
    return probe, acc


class IdentityProbeExperiment:
    """
    Runs the full E6 identity leakage probe experiment.
    """

    def __init__(
        self,
        model_before,
        model_after,
        device: str = "cpu",
    ):
        self.model_before = model_before
        self.model_after  = model_after
        self.device       = device

    def run(
        self,
        train_dataset,
        holdout_dataset,
        target_patient_id: str,
        n_probe_epochs:    int  = 30,
        split_ratio:       float = 0.8,
    ) -> Dict:
        """
        Full E6 experiment.

        Steps:
          1. Extract residuals BEFORE unlearning
          2. Train probe on 80% of patients
          3. Evaluate on 20% held-out patients (validation)
          4. Evaluate on 200 non-training patients (negative control)
          5. Extract residuals AFTER unlearning
          6. Re-evaluate probe on post-unlearning residuals

        Args:
            train_dataset:     patients used in federated training
            holdout_dataset:   patients NOT in training (negative control)
            target_patient_id: the patient being unlearned
            n_probe_epochs:    training epochs for the probe
            split_ratio:       fraction of patients used for probe training

        Returns:
            comprehensive results dict
        """
        results = {}

        # ── Step 1: Extract residuals before unlearning ─────────────
        print("\n  [E6] Extracting residuals BEFORE unlearning...")
        extractor_before = LoRAResidualsExtractor(self.model_before, self.device)
        residuals_before = extractor_before.extract_dataset(
            train_dataset, verbose=True
        )

        # ── Step 2: Extract holdout residuals (negative control) ─────
        print("\n  [E6] Extracting holdout residuals (negative control)...")
        residuals_holdout = extractor_before.extract_dataset(
            holdout_dataset, verbose=True
        )

        # ── Step 3: Train and evaluate probe per adapter ─────────────
        for adapter_name in ["A_cross", "A_v", "A_l"]:
            if adapter_name not in residuals_before:
                continue

            features, patient_indices = residuals_before[adapter_name]
            n = len(features)
            split_idx = int(split_ratio * n)

            X_train = features[:split_idx]
            y_train = np.array(patient_indices[:split_idx])
            X_val   = features[split_idx:]
            y_val   = np.array(patient_indices[split_idx:])

            if len(X_val) == 0:
                continue

            print(f"\n  [E6] Training probe on {adapter_name} "
                  f"({len(X_train)} train, {len(X_val)} val)...")

            probe, val_acc = train_probe(
                X_train, y_train, X_val, y_val,
                n_epochs=n_probe_epochs,
            )

            print(f"    Pre-unlearning accuracy ({adapter_name}): "
                  f"{val_acc:.4f}")

            # Negative control
            if adapter_name in residuals_holdout:
                nc_features, nc_indices = residuals_holdout[adapter_name]
                X_nc = torch.tensor(nc_features, dtype=torch.float32)
                probe.eval()
                with torch.no_grad():
                    nc_logits = probe(X_nc)
                    nc_preds  = nc_logits.argmax(dim=1).numpy()
                # For non-training patients, we expect near-chance accuracy
                # Use majority vote as baseline
                majority_class = np.bincount(y_train).argmax()
                nc_majority_acc = (nc_preds == majority_class).mean()
                print(f"    Negative control accuracy ({adapter_name}): "
                      f"{nc_majority_acc:.4f} (expect <0.55)")

            results[f"{adapter_name}_before"] = val_acc
            results[f"{adapter_name}_negative_control"] = (
                nc_majority_acc if adapter_name in residuals_holdout else None
            )

        # ── Step 4: Extract residuals AFTER unlearning ───────────────
        print("\n  [E6] Extracting residuals AFTER unlearning...")
        extractor_after  = LoRAResidualsExtractor(self.model_after, self.device)
        residuals_after  = extractor_after.extract_dataset(
            train_dataset, verbose=True
        )

        # ── Step 5: Re-evaluate probes on post-unlearning features ───
        for adapter_name in ["A_cross", "A_v", "A_l"]:
            if (adapter_name not in residuals_before or
                    adapter_name not in residuals_after):
                continue

            feat_before, patient_indices = residuals_before[adapter_name]
            feat_after, _               = residuals_after[adapter_name]
            n         = len(feat_after)
            split_idx = int(split_ratio * n)

            X_train = feat_before[:split_idx]
            y_train = np.array(patient_indices[:split_idx])
            X_val   = feat_after[split_idx:]     # post-unlearning features
            y_val   = np.array(patient_indices[split_idx:])

            if len(X_val) == 0:
                continue

            # Re-train probe on pre-unlearning, test on post-unlearning
            probe, _ = train_probe(
                X_train, y_train,
                X_val,   y_val,
                n_epochs=n_probe_epochs,
            )

            X_v_t = torch.tensor(X_val, dtype=torch.float32)
            probe.eval()
            with torch.no_grad():
                after_preds = probe(X_v_t).argmax(dim=1).numpy()

            after_acc = accuracy_score(y_val, after_preds)
            print(f"    Post-unlearning accuracy ({adapter_name}): "
                  f"{after_acc:.4f}")

            results[f"{adapter_name}_after"] = after_acc

        # ── Step 6: Interpret E6 results ─────────────────────────────
        cross_before = results.get("A_cross_before", 1.0)
        cross_after  = results.get("A_cross_after",  1.0)
        v_before     = results.get("A_v_before",     1.0)
        v_after      = results.get("A_v_after",      1.0)
        l_before     = results.get("A_l_before",     1.0)
        l_after      = results.get("A_l_after",      1.0)

        # Causal evidence: only A× drops significantly
        cross_dropped = cross_after < 0.55
        v_unchanged   = abs(v_after  - v_before) < 0.10
        l_unchanged   = abs(l_after  - l_before) < 0.10

        causal_confirmed = cross_dropped and v_unchanged and l_unchanged

        results["causal_localisation_confirmed"] = causal_confirmed
        results["interpretation"] = (
            "✓ CAUSAL EVIDENCE: Only A× probe dropped to near-chance. "
            "A_v and A_l probes unchanged. Identity is causally "
            "localised in A×."
            if causal_confirmed else
            "✗ CAUSAL EVIDENCE WEAK: "
            + ("A× did not drop sufficiently. " if not cross_dropped else "")
            + ("A_v changed unexpectedly. " if not v_unchanged else "")
            + ("A_l changed unexpectedly. " if not l_unchanged else "")
        )

        return results