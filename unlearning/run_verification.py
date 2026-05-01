"""
unlearning/run_verification.py

Runs all three verification layers plus E6 for a completed
unlearning run.

Prerequisites:
  - models/checkpoints/pretrained_final.pt
  - models/checkpoints/unlearned_final.pt
  - models/checkpoints/audit_log.json
  - data/splits/unlearning_targets.json

Usage:
    python unlearning/run_verification.py
"""

import sys
import json
import yaml
import torch
import copy
from pathlib import Path
from transformers import CLIPProcessor, DistilBertTokenizer
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.patient_dataset import PatientDataset
from models.multimodal_model import FedMUModel
from models.lora_model import FedMULoRAModel
from federated.nodes.data_splitter import load_unlearning_targets
from unlearning.verification.mia_verifier import MIAVerifier
from unlearning.verification.reconstruction_test import ReconstructionTester
from unlearning.verification.kl_verifier import KLVerifier
from experiments.e6_probe.identity_probe import IdentityProbeExperiment


def load_model(checkpoint_path: str, config: dict) -> FedMULoRAModel:
    """Load a FedMULoRAModel from a checkpoint file."""
    base  = FedMUModel(freeze_encoders=False)
    model = FedMULoRAModel(
        base_model=base,
        lora_rank=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def main():
    print("\n" + "="*60)
    print("  STEP 7: Verification — 3-Layer + E6 Identity Probe")
    print("="*60)

    # ── Config ────────────────────────────────────────────────────
    with open("configs/fedmu_config.yaml") as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["project"]["seed"])

    # ── Check prerequisites ────────────────────────────────────────
    required = [
        "models/checkpoints/pretrained_final.pt",
        "models/checkpoints/unlearned_final.pt",
        "models/checkpoints/audit_log.json",
        "data/splits/unlearning_targets.json",
    ]
    for path in required:
        if not Path(path).exists():
            print(f"\n  ERROR: {path} not found. "
                  f"Complete Steps 5 and 6 first.")
            return

    # ── Load processors ────────────────────────────────────────────
    print("\n[1/6] Loading processors and models...")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=clip_proc.image_processor.image_mean,
            std=clip_proc.image_processor.image_std,
        ),
    ])

    # ── Load models ────────────────────────────────────────────────
    print("  Loading pre-trained model (BEFORE unlearning)...")
    model_before = load_model(
        "models/checkpoints/pretrained_final.pt", config
    )

    print("  Loading unlearned model (AFTER unlearning)...")
    model_after  = load_model(
        "models/checkpoints/unlearned_final.pt", config
    )

    # ── Load datasets ──────────────────────────────────────────────
    print("\n[2/6] Loading datasets...")
    unlearn_targets  = load_unlearning_targets("data/splits")
    target_patient   = unlearn_targets[0]
    print(f"  Target patient: {target_patient}")

    full_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
    )

    # Member dataset: patients that WERE in training
    member_ids    = [pid for pid in full_dataset.patient_ids
                     if pid != target_patient]
    member_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=member_ids,
    )

    # Retention dataset: all patients except target
    retention_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=member_ids,
    )

    # Non-member dataset: simulate with the last 5 patients as holdout
    # (In real setting: patients the model never trained on)
    all_pids = full_dataset.patient_ids
    holdout_ids = all_pids[-5:]
    holdout_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=holdout_ids,
    )

    # Get target patient's sample
    pid_to_idx     = {pid: i for i, pid in enumerate(full_dataset.patient_ids)}
    target_sample  = full_dataset[pid_to_idx[target_patient]]
    # Get a non-member sample for MIA comparison
    non_member_sample = holdout_dataset[0]

    all_results = {}

    # ══════════════════════════════════════════════════════════════
    # LAYER 1: Membership Inference Attack
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  LAYER 1: Membership Inference Attack (MIA)")
    print("─"*60)

    mia = MIAVerifier(model_before, model_after, device="cpu")
    mia_results = mia.compare_before_after(
        target_sample     = target_sample,
        non_member_sample = non_member_sample,
        member_dataset    = member_dataset,
        non_member_dataset = holdout_dataset,
    )

    print(f"\n  BEFORE unlearning:")
    print(f"    Membership score: "
          f"{mia_results['before']['target_membership_score']:.4f}")
    print(f"    AUC: {mia_results['before']['auc']:.4f}")

    print(f"\n  AFTER unlearning:")
    print(f"    Membership score: "
          f"{mia_results['after']['target_membership_score']:.4f}")
    print(f"    AUC: {mia_results['after']['auc']:.4f}")
    print(f"    Advantage: {mia_results['after']['advantage']:.4f}")
    print(f"    ε-proxy: {mia_results['after']['epsilon_proxy']:.4f}")
    print(f"    {mia_results['after']['interpretation']}")

    all_results["layer1_mia"] = mia_results

    # ══════════════════════════════════════════════════════════════
    # LAYER 2: Reconstruction Test
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  LAYER 2: Reconstruction Test")
    print("─"*60)

    recon = ReconstructionTester(device="cpu")
    recon_results = recon.compare_before_after(
        model_before=model_before,
        model_after=model_after,
        target_sample=target_sample,
    )

    print(f"\n  Label accuracy BEFORE: {recon_results['before']['label_accuracy']:.4f}")
    print(f"  Label accuracy AFTER:  {recon_results['after']['label_accuracy']:.4f}")
    print(f"  Loss BEFORE:           {recon_results['before']['loss']:.4f}")
    print(f"  Loss AFTER:            {recon_results['after']['loss']:.4f}")
    print(f"  Delta loss:            {recon_results['delta_loss']:+.4f}")
    if recon_results.get("embedding_similarity_after") is not None:
        print(f"  Embedding similarity:  "
              f"{recon_results['embedding_similarity_after']:.4f}")
    print(f"  {recon_results['interpretation']}")

    all_results["layer2_reconstruction"] = recon_results

    # ══════════════════════════════════════════════════════════════
    # LAYER 3: KL-Divergence Ratio
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  LAYER 3: KL-Divergence Ratio")
    print("─"*60)

    kl_verifier  = KLVerifier(device="cpu")
    print("\n  Training reference model on retention data...")
    reference_model = kl_verifier.train_reference_model(
        global_model_class=FedMULoRAModel,
        retention_dataset=retention_dataset,
        config=config,
        n_steps=20,
    )

    # Random model for normalisation denominator
    base_rand  = FedMUModel(freeze_encoders=False)
    rand_model = FedMULoRAModel(
        base_model=base_rand,
        lora_rank=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
    )

    kl_results = kl_verifier.compute_ratio(
        unlearned_model=model_after,
        reference_model=reference_model,
        random_model=rand_model,
        test_dataset=retention_dataset,
    )

    print(f"\n  KL(unlearned || reference): "
          f"{kl_results['kl_unlearned_vs_reference']:.6f}")
    print(f"  KL(random    || reference): "
          f"{kl_results['kl_random_vs_reference']:.6f}")
    print(f"  KL ratio:                   {kl_results['kl_ratio']:.6f}")
    print(f"  {kl_results['interpretation']}")

    all_results["layer3_kl"] = kl_results

    # ══════════════════════════════════════════════════════════════
    # E6: Identity Leakage Probe
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  EXPERIMENT E6: Identity Leakage Probe")
    print("─"*60)

    e6_exp = IdentityProbeExperiment(
        model_before=model_before,
        model_after=model_after,
        device="cpu",
    )
    e6_results = e6_exp.run(
        train_dataset=full_dataset,
        holdout_dataset=holdout_dataset,
        target_patient_id=target_patient,
        n_probe_epochs=30,
        split_ratio=0.8,
    )

    print(f"\n  Per-adapter probe accuracy:")
    print(f"  {'Adapter':12s}  {'Before':8s}  {'After':8s}  {'Change':8s}  Status")
    print("  " + "-"*55)
    for adapter in ["A_cross", "A_v", "A_l"]:
        before = e6_results.get(f"{adapter}_before")
        after  = e6_results.get(f"{adapter}_after")
        if before is not None and after is not None:
            change = after - before
            status = ("→ DROPPED ✓"  if after  < 0.55 and adapter == "A_cross"
                      else "→ stable ✓" if abs(change) < 0.10 and adapter != "A_cross"
                      else "→ changed")
            print(f"  {adapter:12s}  {before:8.4f}  {after:8.4f}  "
                  f"{change:+8.4f}  {status}")

    nc_key = "A_cross_negative_control"
    if e6_results.get(nc_key) is not None:
        print(f"\n  Negative control (A×, non-training patients): "
              f"{e6_results[nc_key]:.4f} (expect <0.55)")

    print(f"\n  {e6_results['interpretation']}")
    all_results["e6_identity_probe"] = e6_results

    # ══════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  STEP 7 COMPLETE — Verification Summary")
    print("─"*60)

    layer1_pass = mia_results["after"]["passes_threshold"]
    layer2_pass = recon_results["forgetting_detected"]
    layer3_pass = kl_results["passes_threshold"]
    e6_pass     = e6_results.get("causal_localisation_confirmed", False)

    checks = [
        ("Layer 1 — MIA AUC ≤ 0.55",        layer1_pass),
        ("Layer 2 — Reconstruction dropped", layer2_pass),
        ("Layer 3 — KL ratio < 0.10",        layer3_pass),
        ("E6 — Causal localisation in A×",   e6_pass),
    ]

    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'}  {name}")

    n_pass = sum(p for _, p in checks)
    print(f"\n  Overall: {n_pass}/{len(checks)} verification checks passed")

    if n_pass >= 3:
        verdict = ("STRONG: Unlearning is verified by multiple "
                   "independent methods.")
    elif n_pass == 2:
        verdict = ("PARTIAL: Some verification passed. "
                   "More training rounds recommended.")
    else:
        verdict = ("WEAK: Insufficient evidence of unlearning. "
                   "Review hyperparameters.")

    print(f"  Verdict: {verdict}")

    # ── Save report ────────────────────────────────────────────────
    report = {
        "target_patient": target_patient,
        "layer1_mia": {
            "auc_before":     mia_results["before"]["auc"],
            "auc_after":      mia_results["after"]["auc"],
            "advantage":      mia_results["after"]["advantage"],
            "epsilon_proxy":  mia_results["after"]["epsilon_proxy"],
            "passed":         layer1_pass,
        },
        "layer2_reconstruction": {
            "loss_before":    recon_results["before"]["loss"],
            "loss_after":     recon_results["after"]["loss"],
            "delta_loss":     recon_results["delta_loss"],
            "passed":         layer2_pass,
        },
        "layer3_kl": {
            "kl_ratio":   kl_results["kl_ratio"],
            "passed":     layer3_pass,
        },
        "e6_probe": {
            "A_cross_before": e6_results.get("A_cross_before"),
            "A_cross_after":  e6_results.get("A_cross_after"),
            "A_v_before":     e6_results.get("A_v_before"),
            "A_v_after":      e6_results.get("A_v_after"),
            "A_l_before":     e6_results.get("A_l_before"),
            "A_l_after":      e6_results.get("A_l_after"),
            "causal_confirmed": e6_pass,
            "passed":           e6_pass,
        },
        "overall_passed": n_pass,
        "verdict": verdict,
    }

    report_path = "logs/verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Full report saved: {report_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()