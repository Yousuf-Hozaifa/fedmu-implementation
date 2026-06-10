"""
unlearning/algorithms/lora_subtraction.py  [FIXED v2]

BUG FIXED: 'no_gradient_in_A_cross' failure.

ROOT CAUSE (original code):
    _compute_forget_gradient() searched gradients via:
        for name, param in self.model.named_parameters():
            if "adapter_cross" in name ...

    PyTorch's named_parameters() deduplicates shared tensor objects.
    Because FedMULoRAModel._attach_lora_to_projection() stores the SAME
    LoRALinear object in both:
        self.base.projection.layer1   (canonical path)
        self.adapter_cross.lora_layers['projection_layer1']  (alias)

    named_parameters() only yields the FIRST canonical path it encounters,
    which is 'base.projection.layer1.lora_A' — never containing the string
    "adapter_cross". The gradient collector therefore finds nothing and
    returns an empty dict, triggering FAILED(no_gradient_in_A_cross).

FIX:
    Replace all name-based iteration with DIRECT tensor access through
    self.model.adapter_cross.lora_layers — this bypasses deduplication
    entirely and accesses the exact same tensors that PyTorch computed
    gradients for.

ALSO FIXED:
    _apply_lora_subtraction() now writes directly to layer tensors
    instead of searching by name, for the same reason.

    _measure_rho() now uses consistent direct access for all three
    adapters, eliminating the inconsistency between E2 measurement
    (0.23) and unlearning measurement (1.53) that was caused by
    deepcopy breaking the shared-tensor relationship after
    load_state_dict(strict=False).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unlearning.verification.mps_evaluator import MPSEvaluator


# ── Diagnostic helper ──────────────────────────────────────────────────────

def _check_gradients_exist(model) -> Dict[str, bool]:
    """
    Diagnostic: check which LoRA parameters have gradients.
    Call after loss.backward() to verify gradient flow.
    Returns dict of parameter_description → has_gradient.
    """
    report = {}
    for adapter_name, adapter in [
        ("A_v",    model.adapter_v),
        ("A_l",    model.adapter_l),
        ("A_cross", model.adapter_cross),
    ]:
        for layer_name, layer in adapter.lora_layers.items():
            report[f"{adapter_name}.{layer_name}.lora_A"] = (
                layer.lora_A.grad is not None
            )
            report[f"{adapter_name}.{layer_name}.lora_B"] = (
                layer.lora_B.grad is not None
            )
    return report


# ── Result container ───────────────────────────────────────────────────────

class UnlearningResult:
    """Structured result from one unlearning attempt."""

    def __init__(
        self,
        patient_id:     str,
        success:        bool,
        iterations:     int,
        final_alpha:    float,
        mps_scores:     Optional[Dict],
        rho_p:          Optional[float],
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
        mps    = (f"{self.mps_scores['mps_total']:.4f}"
                  if self.mps_scores else "N/A")
        return (f"UnlearningResult({self.patient_id}: {status}, "
                f"iter={self.iterations}, α={self.final_alpha:.2e}, "
                f"MPS_total={mps})")


# ── Core algorithm ─────────────────────────────────────────────────────────

class LoRASubtractionUnlearner:
    """
    Implements the LoRA-Subtraction algorithm for a single affected node.
    """

    def __init__(
        self,
        model,
        config: dict,
        device: str = "cpu",
    ):
        self.model        = model
        self.config       = config
        self.device       = device
        self.alpha_init   = config["unlearning"]["alpha"]
        self.lambda_pres  = config["unlearning"]["lambda_preserve"]
        self.alpha_min    = config["unlearning"]["alpha_min"]
        self.n_max        = config["unlearning"]["n_max"]
        self.tau          = config["unlearning"]["tau_mps"]

    # ── FIX: direct tensor access via adapter objects ──────────────────────

    def _collect_cross_adapter_grads(self) -> Dict[str, torch.Tensor]:
        """
        FIX: Collect gradients from A× by directly iterating
        self.model.adapter_cross.lora_layers, bypassing PyTorch's
        named_parameters() deduplication.

        PyTorch deduplication issue: when the same LoRALinear object
        is registered both in base.projection.layer1 AND in
        adapter_cross.lora_layers, named_parameters() only yields
        the canonical path ('base.projection.layer1.lora_A').
        Searching for 'adapter_cross' in the name fails silently.

        Direct access through the adapter object always finds the
        correct tensors regardless of registration order.
        """
        grads = {}
        for layer_name, layer in self.model.adapter_cross.lora_layers.items():
            if layer.lora_A.requires_grad and layer.lora_A.grad is not None:
                grads[f"{layer_name}.lora_A"] = layer.lora_A.grad.detach().clone()
            if layer.lora_B.requires_grad and layer.lora_B.grad is not None:
                grads[f"{layer_name}.lora_B"] = layer.lora_B.grad.detach().clone()
        return grads

    def _compute_forget_gradient(
        self,
        patient_sample: dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute g_f = ∇_{A×} L(f(v_p, t_p)).

        This is the ASCENT direction: increases loss on patient p,
        causing the model to "unlearn" their data.
        """
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

        loss = outputs["loss"]
        loss.backward()

        # FIXED: direct access instead of named_parameters() search
        grads = self._collect_cross_adapter_grads()

        # Diagnostic on first call
        if not grads:
            diag = _check_gradients_exist(self.model)
            print(f"\n  DIAGNOSTIC — gradient existence after backward():")
            for k, v in diag.items():
                print(f"    {k}: {'✓ grad' if v else '✗ NO GRAD'}")
            print(f"  A× requires_grad states:")
            for ln, layer in self.model.adapter_cross.lora_layers.items():
                print(f"    {ln}.lora_A requires_grad={layer.lora_A.requires_grad}")
                print(f"    {ln}.lora_B requires_grad={layer.lora_B.requires_grad}")

        self.model.zero_grad()
        return grads

    def _compute_preserve_gradient(
        self,
        retention_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute g_r = ∇_{A×} L(f(v_b, t_b)) on a retention batch.

        This is the DESCENT direction: maintains model quality on all
        retained patients while patient p is being forgotten.
        """
        self.model.zero_grad()
        self.model.train()

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

        outputs["loss"].backward()

        # FIXED: direct access instead of named_parameters() search
        grads = self._collect_cross_adapter_grads()

        self.model.zero_grad()
        return grads

    def _apply_lora_subtraction(
        self,
        g_forget:   Dict[str, torch.Tensor],
        g_preserve: Dict[str, torch.Tensor],
        alpha:      float,
    ):
        """
        Apply: A×_new = A× + α·g_f − λ·g_r

        FIX: Write directly to each layer's lora_A and lora_B tensors
        instead of searching model.named_parameters() by name string.

        The key names in g_forget/g_preserve match the format
        '{layer_name}.lora_A' and '{layer_name}.lora_B' produced by
        _collect_cross_adapter_grads(), which uses the same layer_name
        as the keys in adapter_cross.lora_layers.
        """
        with torch.no_grad():
            for layer_name, layer in self.model.adapter_cross.lora_layers.items():
                key_a = f"{layer_name}.lora_A"
                key_b = f"{layer_name}.lora_B"

                if key_a in g_forget and key_a in g_preserve:
                    delta_a = (alpha * g_forget[key_a]
                               - self.lambda_pres * g_preserve[key_a])
                    layer.lora_A.data.add_(delta_a)

                if key_b in g_forget and key_b in g_preserve:
                    delta_b = (alpha * g_forget[key_b]
                               - self.lambda_pres * g_preserve[key_b])
                    layer.lora_B.data.add_(delta_b)

    def _measure_rho(self, patient_sample: dict) -> Optional[float]:
        """
        Measure ρ(p) for the target patient.

        FIX: Uses CONSISTENT direct tensor access for all three adapters,
        matching the method used in RhoComputer.compute_single().

        The original code used adapter object access for A× but
        relied on named_parameters() state for A_v and A_l, which
        produced different results after deepcopy + load_state_dict
        (explaining the discrepancy: E2 showed ρ=0.23, unlearner
        showed ρ=1.53 for the same patient).
        """
        # Save and temporarily enable all adapter grads
        saved = {}
        for adapter in [self.model.adapter_v,
                        self.model.adapter_l,
                        self.model.adapter_cross]:
            for ln, layer in adapter.lora_layers.items():
                saved[id(layer.lora_A)] = layer.lora_A.requires_grad
                saved[id(layer.lora_B)] = layer.lora_B.requires_grad
                layer.lora_A.requires_grad_(True)
                layer.lora_B.requires_grad_(True)

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

        # FIX: direct access for all three adapters (consistent with E2)
        def _frobenius(adapter) -> float:
            total_sq = 0.0
            for _, layer in adapter.lora_layers.items():
                for param in [layer.lora_A, layer.lora_B]:
                    if param.grad is not None:
                        total_sq += param.grad.detach().norm(2).item() ** 2
            return total_sq ** 0.5

        norm_cross = _frobenius(self.model.adapter_cross)
        norm_v     = _frobenius(self.model.adapter_v)
        norm_l     = _frobenius(self.model.adapter_l)

        self.model.zero_grad()

        # Restore original requires_grad states
        for adapter in [self.model.adapter_v,
                        self.model.adapter_l,
                        self.model.adapter_cross]:
            for ln, layer in adapter.lora_layers.items():
                layer.lora_A.requires_grad_(saved[id(layer.lora_A)])
                layer.lora_B.requires_grad_(saved[id(layer.lora_B)])

        denom = norm_v + norm_l
        if denom < 1e-10 or norm_cross < 1e-10:
            return None
        return norm_cross / denom

    # ── Main unlearning loop ───────────────────────────────────────────────

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
        """
        prefix = f"    [Node {node_id}] " if node_id >= 0 else "    "

        if verbose:
            print(f"\n{prefix}Unlearning patient: {patient_id}")

        # ── Step 1: Freeze everything except A× ────────────────────
        self.model.set_unlearning_mode()

        # ── Verify set_unlearning_mode() worked correctly ───────────
        cross_trainable = sum(
            1 for _, layer in self.model.adapter_cross.lora_layers.items()
            if layer.lora_A.requires_grad
        )
        if cross_trainable == 0:
            return UnlearningResult(
                patient_id=patient_id, success=False,
                iterations=0, final_alpha=self.alpha_init,
                mps_scores=None, rho_p=None,
                failure_reason="set_unlearning_mode_failed: A× not trainable",
            )

        if verbose:
            print(f"{prefix}  A× trainable layers: {cross_trainable} ✓")

        # ── Step 1b: Measure ρ(p) ───────────────────────────────────
        rho_p = self._measure_rho(patient_sample)
        if verbose:
            rho_str = f"{rho_p:.4f}" if rho_p else "N/A"
            rho_0   = self.config["statistics"]["rho_0"]
            print(f"{prefix}ρ(p) = {rho_str} (threshold ρ₀={rho_0})")
            if rho_p and rho_p > rho_0:
                print(f"{prefix}→ A× is strongly entangled. Targeting confirmed.")
            else:
                print(f"{prefix}→ ρ(p) below ρ₀ (expected on toy model). "
                      f"Proceeding — will document as failure case.")

        ret_loader = DataLoader(
            retention_dataset,
            batch_size=min(4, len(retention_dataset)),
            shuffle=True,
        )

        alpha     = self.alpha_init
        iteration = 0

        while True:
            iteration += 1

            # ── Step 2: Forget gradient (ascent) ───────────────────
            g_forget = self._compute_forget_gradient(patient_sample)

            if not g_forget:
                return UnlearningResult(
                    patient_id=patient_id, success=False,
                    iterations=iteration, final_alpha=alpha,
                    mps_scores=None, rho_p=rho_p,
                    failure_reason=(
                        "no_gradient_in_A_cross — check that A× layers "
                        "have requires_grad=True and are in the computation graph"
                    ),
                )

            # ── Step 3: Preserve gradient (descent) ────────────────
            g_preserve = self._compute_preserve_gradient(ret_loader)

            # ── Step 4: Apply LoRA-Subtraction ──────────────────────
            self._apply_lora_subtraction(g_forget, g_preserve, alpha)

            if verbose:
                nf = sum(g.norm().item() for g in g_forget.values())
                np_ = sum(g.norm().item() for g in g_preserve.values())
                print(f"{prefix}  iter {iteration:2d}: "
                      f"α={alpha:.2e}  "
                      f"|g_f|={nf:.5f}  |g_r|={np_:.5f}  "
                      f"(keys: {list(g_forget.keys())})")

            # ── Step 5: MPS check ───────────────────────────────────
            mps_scores = mps_evaluator.compute_mps()
            all_pass   = all(
                mps_scores[k] >= self.tau
                for k in ["mps_v", "mps_l", "mps_cross"]
            )

            if verbose:
                print(f"{prefix}  MPS: "
                      f"v={mps_scores['mps_v']:.3f} "
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

            alpha = alpha / 2.0

            if alpha < self.alpha_min:
                reason = f"alpha_below_minimum ({alpha:.2e})"
                if verbose:
                    print(f"{prefix}  ✗ ABORT: {reason}")
                return UnlearningResult(
                    patient_id=patient_id, success=False,
                    iterations=iteration, final_alpha=alpha,
                    mps_scores=mps_scores, rho_p=rho_p,
                    failure_reason=reason,
                )