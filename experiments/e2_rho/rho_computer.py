"""
experiments/e2_rho/rho_computer.py

Computes the gradient concentration ratio ρ(p) for each patient.

ρ(p) = ||∇_{A×} L(f(v_p, t_p))||_F
       ─────────────────────────────────────────────────────────
       ||∇_{Av} L(f(v_p, t_p))||_F + ||∇_{Al} L(f(v_p, t_p))||_F

This is the core measurement of the Fed-MU Experiment E2.

Key design decisions:
  1. We enable ALL adapter gradients temporarily to get all three norms.
  2. We use per-patient single-sample forward passes (not batches),
     because ρ(p) is a per-patient property.
  3. We zero gradients before and after each patient measurement.
  4. We restore the model's original trainability state after measuring.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.lora_model import FedMULoRAModel


class RhoComputer:
    """
    Computes ρ(p) for a set of patients using the LoRA adapter model.

    Usage:
        computer = RhoComputer(model)
        results = computer.compute_all(dataset)
        # results is a dict: patient_id → rho_value
    """

    def __init__(self, model: FedMULoRAModel, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def _enable_all_adapter_grads(self):
        """
        Temporarily enable gradients on ALL three adapters
        so we can measure gradient norms across all of them.
        Returns the original requires_grad state so we can restore it.
        """
        original_state = {}
        for name, param in self.model.named_parameters():
            original_state[name] = param.requires_grad
            # Only enable grad for LoRA params
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
        return original_state

    def _restore_grad_state(self, original_state: dict):
        """Restore requires_grad to its pre-measurement state."""
        for name, param in self.model.named_parameters():
            if name in original_state:
                param.requires_grad = original_state[name]

    def compute_single(
        self,
        patient_batch: dict,
    ) -> Dict[str, Optional[float]]:
        """
        Compute gradient norms for a single patient batch.

        Args:
            patient_batch: dict with keys image, text_input_ids,
                           text_attention_mask, labels, patient_id

        Returns:
            dict with keys:
                'norm_cross': ||∇_{A×} L||_F
                'norm_v':     ||∇_{Av} L||_F
                'norm_l':     ||∇_{Al} L||_F
                'rho':        computed ρ(p)
                'patient_id': patient identifier
                'loss':       scalar loss value
        """
        # Save original state and enable all adapter grads
        original_state = self._enable_all_adapter_grads()

        self.model.train()  # train mode so gradients flow
        self.model.zero_grad()

        try:
            # Move batch to device
            pixel_values = patient_batch["image"].unsqueeze(0).to(self.device) \
                if patient_batch["image"].dim() == 3 \
                else patient_batch["image"].to(self.device)
            input_ids    = patient_batch["text_input_ids"].unsqueeze(0).to(self.device) \
                if patient_batch["text_input_ids"].dim() == 1 \
                else patient_batch["text_input_ids"].to(self.device)
            attn_mask    = patient_batch["text_attention_mask"].unsqueeze(0).to(self.device) \
                if patient_batch["text_attention_mask"].dim() == 1 \
                else patient_batch["text_attention_mask"].to(self.device)
            labels       = patient_batch["labels"].unsqueeze(0).to(self.device) \
                if patient_batch["labels"].dim() == 1 \
                else patient_batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )

            loss = outputs["loss"]
            loss_val = loss.item()

            # Backward pass to get gradients
            loss.backward()

            # Read gradient norms from each adapter
            norm_cross = self.model.adapter_cross.get_total_gradient_norm()
            norm_v     = self.model.adapter_v.get_total_gradient_norm()
            norm_l     = self.model.adapter_l.get_total_gradient_norm()

            # Compute ρ(p)
            denom = (norm_v or 0.0) + (norm_l or 0.0)

            if denom < 1e-10:
                rho = None  # Degenerate case: unimodal adapters have zero gradient
            elif norm_cross is None:
                rho = None  # A× has no gradient (should not happen)
            else:
                rho = norm_cross / denom

        finally:
            # Always clean up: zero grads and restore state
            self.model.zero_grad()
            self._restore_grad_state(original_state)

        return {
            "patient_id": patient_batch["patient_id"]
                          if isinstance(patient_batch["patient_id"], str)
                          else patient_batch["patient_id"][0],
            "norm_cross": norm_cross,
            "norm_v":     norm_v,
            "norm_l":     norm_l,
            "rho":        rho,
            "loss":       loss_val,
        }

    def compute_all(
        self,
        dataset,
        verbose: bool = True,
    ) -> Dict[str, dict]:
        """
        Compute ρ(p) for every patient in a dataset.

        Args:
            dataset: PatientDataset instance
            verbose: print progress every 10 patients

        Returns:
            dict mapping patient_id → {rho, norm_cross, norm_v, norm_l, loss}
        """
        results = {}
        n = len(dataset)

        if verbose:
            print(f"  Computing ρ(p) for {n} patients...")

        for i in range(n):
            sample = dataset[i]
            result = self.compute_single(sample)
            patient_id = result["patient_id"]
            results[patient_id] = result

            if verbose and (i + 1) % 10 == 0:
                valid_rhos = [r["rho"] for r in results.values()
                              if r["rho"] is not None]
                mean_rho = sum(valid_rhos) / len(valid_rhos) if valid_rhos else 0
                print(f"    [{i+1}/{n}] mean ρ so far: {mean_rho:.4f}")

        valid = [r for r in results.values() if r["rho"] is not None]
        if verbose:
            print(f"  Done. {len(valid)}/{n} patients with valid ρ(p).")

        return results

    def compute_for_patient_ids(
        self,
        dataset,
        patient_ids: List[str],
    ) -> Dict[str, dict]:
        """
        Compute ρ(p) for a specific subset of patients.
        Used for unlearning targets only (faster than computing all).
        """
        # Build a lookup from patient_id to dataset index
        id_to_idx = {pid: i for i, pid in enumerate(dataset.patient_ids)}

        results = {}
        for pid in patient_ids:
            if pid not in id_to_idx:
                print(f"  Warning: {pid} not found in dataset")
                continue
            sample = dataset[id_to_idx[pid]]
            result = self.compute_single(sample)
            results[pid] = result

        return results