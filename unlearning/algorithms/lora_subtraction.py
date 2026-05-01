"""
unlearning/algorithms/lora_subtraction.py

Implements the LoRA-Subtraction unlearning algorithm from the proposal.

Algorithm (from Figure 3 of the proposal):

  Input: model with A×, patient p's data, retention dataset, config
  Output: updated A× that has forgotten patient p

  Step 1:  Freeze base model, A_v, A_l. Only A× trainable.
  Step 1b: Measure ρ(p). Run E6 baseline probe.
  Step 2:  Compute forget gradient g_f = ∇_{A×} L(f(v_p, t_p))
  Step 3:  Sample retention batch B. Compute g_r = ∇_{A×} L(f(v_b,t_b))
  Step 4:  A×_new = A× + α·g_f − λ·g_r
           Check stability: α < 2/λ_max(H_f)
  Step 5:  Compute MPS. If any MPS < τ: halve α, goto Step 2.
           If α < α_min or t > N_max: ABORT.
           Else: SUCCESS → return updated A×.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import copy
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlearning.verification.mps_evaluator import MPSEvaluator


class UnlearningResult:
    """Structured result from one unlearning attempt."""

    def __init__(
        self,
        patient_id:   str,
        success:      bool,
        iterations:   int,
        final_alpha:  float,
        mps_scores:   Optional[Dict],
        rho_p:        Optional[float],
        failure_reason: Optional[str] = None,
    ):
        self.patient_id     = patient_id
        self.success        = success
        self.iterations     = iterations
        self.final_alpha    = final_alpha
        self.mps_scores     = mps_scores
        self.rho_p          = rho_p
        self.failure_reason = failure_reason

    def to_dict(self) -> dict:
        return {
            "patient_id":     self.patient_id,
            "success":        self.success,
            "iterations":     self.iterations,
            "final_alpha":    self.final_alpha,
            "mps_total":      self.mps_scores.get("mps_total") if self.mps_scores else None,
            "mps_v":          self.mps_scores.get("mps_v")     if self.mps_scores else None,
            "mps_l":          self.mps_scores.get("mps_l")     if self.mps_scores else None,
            "mps_cross":      self.mps_scores.get("mps_cross") if self.mps_scores else None,
            "rho_p":          self.rho_p,
            "failure_reason": self.failure_reason,
        }

    def __repr__(self):
        status = "SUCCESS" if self.success else f"FAILED ({self.failure_reason})"
        mps    = f"{self.mps_scores['mps_total']:.4f}" \
                 if self.mps_scores else "N/A"
        return (f"UnlearningResult({self.patient_id}: {status}, "
                f"iter={self.iterations}, α={self.final_alpha:.2e}, "
                f"MPS_total={mps})")


class LoRASubtractionUnlearner:
    """
    Implements the LoRA-Subtraction algorithm for a single affected node.

    One instance per node. Each node runs this independently on its
    local copy of the model and its local patient data.
    """

    def __init__(
        self,
        model,
        config:     dict,
        device:     str = "cpu",
    ):
        self.model   = model
        self.config  = config
        self.device  = device

        # Hyperparameters from config
        self.alpha_init    = config["unlearning"]["alpha"]            # 0.01
        self.lambda_pres   = config["unlearning"]["lambda_preserve"]  # 0.10
        self.alpha_min     = config["unlearning"]["alpha_min"]        # 1e-5
        self.n_max         = config["unlearning"]["n_max"]            # 10
        self.tau           = config["unlearning"]["tau_mps"]          # 0.85

    def _compute_forget_gradient(
        self,
        patient_sample: dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute g_f = ∇_{A×} L(f(v_p, t_p)).

        This is the ASCENT direction — it increases the loss on
        patient p's image-text pair, making the model worse at
        predicting their data (which is the goal of forgetting).

        Returns: dict of parameter_name → gradient tensor
        """
        self.model.zero_grad()
        self.model.train()

        # Single-sample forward pass on patient p
        pixel_values = patient_sample["image"].unsqueeze(0).to(self.device)
        input_ids    = patient_sample["text_input_ids"].unsqueeze(0).to(self.device)
        attn_mask    = patient_sample["text_attention_mask"].unsqueeze(0).to(self.device)
        labels       = patient_sample["labels"].unsqueeze(0).to(self.device)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()

        # Extract gradients for A× parameters only
        grads = {}
        for name, param in self.model.named_parameters():
            if ("adapter_cross" in name and
                    ("lora_A" in name or "lora_B" in name) and
                    param.grad is not None):
                grads[name] = param.grad.detach().clone()

        self.model.zero_grad()
        return grads

    def _compute_preserve_gradient(
        self,
        retention_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute g_r = ∇_{A×} L(f(v_b, t_b)) on a retention batch.

        This is the DESCENT direction — it reduces the loss on
        all patients OTHER than p, preserving general knowledge.

        Returns: dict of parameter_name → gradient tensor
        """
        self.model.zero_grad()
        self.model.train()

        # Sample one batch from the retention dataset
        batch = next(iter(retention_loader))

        pixel_values = batch["image"].to(self.device)
        input_ids    = batch["text_input_ids"].to(self.device)
        attn_mask    = batch["text_attention_mask"].to(self.device)
        labels       = batch["labels"].to(self.device)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()

        grads = {}
        for name, param in self.model.named_parameters():
            if ("adapter_cross" in name and
                    ("lora_A" in name or "lora_B" in name) and
                    param.grad is not None):
                grads[name] = param.grad.detach().clone()

        self.model.zero_grad()
        return grads

    def _apply_lora_subtraction(
        self,
        g_forget:  Dict[str, torch.Tensor],
        g_preserve: Dict[str, torch.Tensor],
        alpha:     float,
    ):
        """
        Apply the LoRA-Subtraction update in-place:
        A×_new = A× + α·g_f − λ·g_r

        Note: we ADD g_forget (gradient ascent — increases loss on p)
              we SUBTRACT λ·g_preserve (descent — reduces loss on others)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in g_forget and name in g_preserve:
                    delta = (alpha * g_forget[name]
                             - self.lambda_pres * g_preserve[name])
                    param.data.add_(delta)

    def _measure_rho(self, patient_sample: dict) -> Optional[float]:
        """
        Measure ρ(p) for the target patient before unlearning begins.
        Temporarily enables all adapter gradients.
        """
        # Enable all adapter gradients temporarily
        saved_states = {}
        for name, param in self.model.named_parameters():
            saved_states[name] = param.requires_grad
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True

        self.model.zero_grad()
        self.model.train()

        pixel_values = patient_sample["image"].unsqueeze(0).to(self.device)
        input_ids    = patient_sample["text_input_ids"].unsqueeze(0).to(self.device)
        attn_mask    = patient_sample["text_attention_mask"].unsqueeze(0).to(self.device)
        labels       = patient_sample["labels"].unsqueeze(0).to(self.device)

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
        )
        outputs["loss"].backward()

        norm_cross = self.model.adapter_cross.get_total_gradient_norm()
        norm_v     = self.model.adapter_v.get_total_gradient_norm()
        norm_l     = self.model.adapter_l.get_total_gradient_norm()

        self.model.zero_grad()

        # Restore original requires_grad states
        for name, param in self.model.named_parameters():
            param.requires_grad = saved_states.get(name, False)

        denom = (norm_v or 0.0) + (norm_l or 0.0)
        if denom < 1e-10 or norm_cross is None:
            return None
        return norm_cross / denom

    def unlearn(
        self,
        patient_id:       str,
        patient_sample:   dict,
        retention_dataset,
        mps_evaluator:    MPSEvaluator,
        node_id:          int = -1,
        verbose:          bool = True,
    ) -> UnlearningResult:
        """
        Main unlearning procedure for a single patient on a single node.

        Args:
            patient_id:       identifier of patient to forget
            patient_sample:   single dict from PatientDataset[i]
            retention_dataset: dataset of patients to preserve
            mps_evaluator:    pre-initialised MPS evaluator
            node_id:          for logging purposes
            verbose:          print iteration-level details

        Returns:
            UnlearningResult with success flag, iteration count,
            final MPS scores, and failure reason if applicable
        """
        prefix = f"    [Node {node_id}] " if node_id >= 0 else "    "

        if verbose:
            print(f"\n{prefix}Unlearning patient: {patient_id}")

        # ── Step 1: Freeze everything except A× ────────────────────
        self.model.set_unlearning_mode()

        # ── Step 1b: Measure ρ(p) before unlearning ────────────────
        rho_p = self._measure_rho(patient_sample)
        if verbose:
            rho_str = f"{rho_p:.4f}" if rho_p else "N/A"
            print(f"{prefix}ρ(p) = {rho_str} "
                  f"(threshold ρ₀={self.config['statistics']['rho_0']})")
            if rho_p and rho_p > self.config["statistics"]["rho_0"]:
                print(f"{prefix}→ A× is strongly entangled. "
                      f"Targeting confirmed.")
            else:
                print(f"{prefix}→ ρ(p) below ρ₀. Proceeding anyway "
                      f"(failure case — will document).")

        # Build retention data loader (shuffled, batch size ≤ dataset size)
        ret_loader = DataLoader(
            retention_dataset,
            batch_size=min(4, len(retention_dataset)),
            shuffle=True,
        )

        # ── Main unlearning loop ────────────────────────────────────
        alpha     = self.alpha_init
        iteration = 0

        while True:
            iteration += 1

            # ── Step 2: Forget gradient ─────────────────────────────
            g_forget = self._compute_forget_gradient(patient_sample)

            if not g_forget:
                return UnlearningResult(
                    patient_id=patient_id, success=False,
                    iterations=iteration, final_alpha=alpha,
                    mps_scores=None, rho_p=rho_p,
                    failure_reason="no_gradient_in_A_cross",
                )

            # ── Step 3: Preserve gradient ───────────────────────────
            g_preserve = self._compute_preserve_gradient(ret_loader)

            # ── Step 4: Apply LoRA-Subtraction ──────────────────────
            self._apply_lora_subtraction(g_forget, g_preserve, alpha)

            if verbose:
                # Compute gradient norms for logging
                norms_f = [g.norm().item() for g in g_forget.values()]
                norms_p = [g.norm().item() for g in g_preserve.values()]
                gf_norm = sum(norms_f) / len(norms_f) if norms_f else 0
                gr_norm = sum(norms_p) / len(norms_p) if norms_p else 0
                print(f"{prefix}  iter {iteration:2d}: "
                      f"α={alpha:.2e}  "
                      f"|g_f|={gf_norm:.5f}  "
                      f"|g_r|={gr_norm:.5f}")

            # ── Step 5: MPS check ───────────────────────────────────
            mps_scores = mps_evaluator.compute_mps()
            all_pass   = all(
                mps_scores[k] >= self.tau
                for k in ["mps_v", "mps_l", "mps_cross"]
            )

            if verbose:
                print(f"{prefix}  MPS: v={mps_scores['mps_v']:.3f} "
                      f"l={mps_scores['mps_l']:.3f} "
                      f"×={mps_scores['mps_cross']:.3f} "
                      f"total={mps_scores['mps_total']:.3f} "
                      f"{'✓' if all_pass else '✗'}")

            if all_pass:
                if verbose:
                    print(f"{prefix}  ✓ SUCCESS: all MPS ≥ τ={self.tau}")
                return UnlearningResult(
                    patient_id=patient_id, success=True,
                    iterations=iteration, final_alpha=alpha,
                    mps_scores=mps_scores, rho_p=rho_p,
                )

            # ── Abort checks ────────────────────────────────────────
            if iteration >= self.n_max:
                reason = f"max_iterations_exceeded (t={iteration})"
                if verbose:
                    print(f"{prefix}  ✗ ABORT: {reason}")
                return UnlearningResult(
                    patient_id=patient_id, success=False,
                    iterations=iteration, final_alpha=alpha,
                    mps_scores=mps_scores, rho_p=rho_p,
                    failure_reason=reason,
                )

            # Halve α for next iteration
            alpha = alpha / 2.0

            if alpha < self.alpha_min:
                reason = f"alpha_below_minimum ({alpha:.2e} < {self.alpha_min:.2e})"
                if verbose:
                    print(f"{prefix}  ✗ ABORT: {reason}")
                return UnlearningResult(
                    patient_id=patient_id, success=False,
                    iterations=iteration, final_alpha=alpha,
                    mps_scores=mps_scores, rho_p=rho_p,
                    failure_reason=reason,
                )