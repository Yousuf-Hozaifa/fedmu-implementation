"""
unlearning/run_verification.py  [FIXED v2]

FIXES applied to match updated API signatures:

1. MIAVerifier.compare_before_after() signature changed:
   OLD: compare_before_after(target_sample, non_member_sample,
                              member_dataset, non_member_dataset)
   NEW: compare_before_after(target_dataset, non_member_dataset,
                              member_dataset)
   Reason: new MIA evaluates on full datasets (not single samples)
   to produce reliable AUC values.

2. MIAVerifier result structure changed:
   OLD: results['after']['passes_threshold']  (single-sample)
   NEW: results['after']['passes_threshold']  (dataset-level AUC)
   The key name is the same but semantics differ — now meaningful.

3. IdentityProbeExperiment.run() API unchanged but probe is now binary.
   The call signature is the same; only internal logic changed.

4. Report printing updated to show new MIA fields:
   n_target, n_non_member (previously absent).
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
            print(f"\n  ERROR: {path} not found.")
            print("  Complete Steps 5 and 6 first.")
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
    model_after = load_model(
        "models/checkpoints/unlearned_final.pt", config
    )

    # ── Load datasets ──────────────────────────────────────────────
    print("\n[2/6] Loading datasets...")
    unlearn_targets = load_unlearning_targets("data/splits")
    target_patient  = unlearn_targets[0]
    print(f"  Target patient: {target_patient}")

    full_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
    )

    all_pids = full_dataset.patient_ids

    # Member dataset: patients that WERE in training (excluding target)
    member_ids = [pid for pid in all_pids if pid != target_patient]
    member_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=member_ids,
    )

    # FIX: target_dataset — a dataset containing ONLY the target patient
    # (used by new MIA API which takes datasets, not single samples)
    target_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=[target_patient],
    )

    # Retention dataset: everyone except the target
    retention_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=member_ids,
    )

    # Holdout (non-member) dataset: last 5 patients as simulated non-members
    holdout_ids = all_pids[-5:]
    holdout_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=holdout_ids,
    )

    # Single sample for reconstruction test (Layer 2)
    pid_to_idx    = {pid: i for i, pid in enumerate(full_dataset.patient_ids)}
    target_sample = full_dataset[pid_to_idx[target_patient]]

    all_results = {}

    # ══════════════════════════════════════════════════════════════
    # LAYER 1: Membership Inference Attack
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  LAYER 1: Membership Inference Attack (MIA)")
    print("─"*60)

    mia = MIAVerifier(model_before, model_after, device="cpu")

    # FIX: new API — passes datasets, not single samples
    mia_results = mia.compare_before_after(
        target_dataset=target_dataset,
        non_member_dataset=holdout_dataset,
        member_dataset=member_dataset,
    )

    def _fmt_auc(val):
        """Format AUC — handles None when score fallback is used."""
        return f"{val:.4f}" if val is not None else "N/A (n_target=1, score used)"

    print(f"\n  BEFORE unlearning:")
    print(f"    Samples tested:        "
          f"{mia_results['before'].get('n_target',1)} target + "
          f"{mia_results['before'].get('n_non_member',1)} non-member")
    print(f"    Mean membership score: "
          f"{mia_results['before'].get('mean_target_membership', 0):.4f}")
    print(f"    AUC:                   "
          f"{_fmt_auc(mia_results['before'].get('auc'))}")
    print(f"    Method:                "
          f"{mia_results['before'].get('method', 'auc')}")

    print(f"\n  AFTER unlearning:")
    print(f"    Samples tested:        "
          f"{mia_results['after'].get('n_target',1)} target + "
          f"{mia_results['after'].get('n_non_member',1)} non-member")
    print(f"    Mean membership score: "
          f"{mia_results['after'].get('mean_target_membership', 0):.4f}")
    print(f"    AUC:                   "
          f"{_fmt_auc(mia_results['after'].get('auc'))}")
    print(f"    Advantage (Adv):       "
          f"{mia_results['after'].get('advantage', 0):.4f}")
    print(f"    ε-proxy (≲ 2·Adv):    "
          f"{mia_results['after'].get('epsilon_proxy', 0):.4f}")
    print(f"    Method:                "
          f"{mia_results['after'].get('method', 'auc')}")
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

    print(f"\n  Label accuracy BEFORE: "
          f"{recon_results['before']['label_accuracy']:.4f}")
    print(f"  Label accuracy AFTER:  "
          f"{recon_results['after']['label_accuracy']:.4f}")
    print(f"  Loss BEFORE:           "
          f"{recon_results['before']['loss']:.4f}")
    print(f"  Loss AFTER:            "
          f"{recon_results['after']['loss']:.4f}")
    print(f"  Delta loss:            "
          f"{recon_results['delta_loss']:+.4f}")
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

    kl_verifier = KLVerifier(device="cpu")
    print("\n  Training reference model on retention data...")
    reference_model = kl_verifier.train_reference_model(
        global_model_class=FedMULoRAModel,
        retention_dataset=retention_dataset,
        config=config,
        n_steps=20,
    )

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
    print(f"  KL ratio:                   "
          f"{kl_results['kl_ratio']:.6f}")
    print(f"  {kl_results['interpretation']}")

    all_results["layer3_kl"] = kl_results

    # ══════════════════════════════════════════════════════════════
    # E6: Identity Leakage Probe (binary, fixed)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("  EXPERIMENT E6: Identity Leakage Probe (Binary)")
    print("─"*60)

    e6_exp = IdentityProbeExperiment(
        model_before=model_before,
        model_after=model_after,
        device="cpu",
    )
    # API unchanged — internal probe is now binary
    e6_results = e6_exp.run(
        train_dataset=full_dataset,
        holdout_dataset=holdout_dataset,
        target_patient_id=target_patient,
        n_probe_epochs=40,
        split_ratio=0.8,
    )

    print(f"\n  Per-adapter binary probe accuracy "
          f"(task: is this the target patient?):")
    print(f"  {'Adapter':16s}  {'Before':8s}  {'After':8s}  "
          f"{'Change':8s}  {'AUC after':10s}  Status")
    print("  " + "-"*70)

    for adapter in ["A_cross", "A_v", "A_l"]:
        before = e6_results.get(f"{adapter}_before")
        after  = e6_results.get(f"{adapter}_after")
        auc_a  = e6_results.get(f"{adapter}_after_auc")
        if before is not None and after is not None:
            change = after - before
            if adapter == "A_cross":
                status = ("→ DROPPED ✓" if (before - after) > 0.10
                          else "→ minimal drop")
            else:
                status = ("→ stable ✓" if abs(change) < 0.10
                          else "→ changed ⚠")
            auc_str = f"{auc_a:.4f}" if auc_a is not None else "  N/A"
            print(f"  {adapter:16s}  {before:8.4f}  {after:8.4f}  "
                  f"{change:+8.4f}  {auc_str:10s}  {status}")

    nc_key = "A_cross_negative_control"
    if e6_results.get(nc_key) is not None:
        print(f"\n  Negative control (A×, non-training patients): "
              f"{e6_results[nc_key]:.4f}  "
              f"(expect ~0.50 for binary random chance)")

    print(f"\n  {e6_results['interpretation']}")
    all_results["e6_identity_probe"] = e6_results

    # ══════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  STEP 7 COMPLETE — Verification Summary")
    print("─"*60)

    layer1_pass = mia_results["after"].get("passes_threshold", False)
    layer2_pass = recon_results.get("forgetting_detected", False)
    layer3_pass = kl_results.get("passes_threshold", False)
    e6_pass     = e6_results.get("causal_localisation_confirmed", False)

    layer1_method = mia_results["after"].get("method", "auc")
    layer1_label  = (
        "Layer 1 — MIA score ≤ 0.60 (n=1 fallback)"
        if layer1_method == "membership_score_fallback"
        else "Layer 1 — MIA AUC in [0.45, 0.55]"
    )

    checks = [
        (layer1_label,                             layer1_pass),
        ("Layer 2 — Reconstruction loss increased", layer2_pass),
        ("Layer 3 — KL ratio < 0.10",              layer3_pass),
        ("E6      — Causal localisation in A×",    e6_pass),
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
                   "More training rounds or data recommended.")
    else:
        verdict = ("WEAK: Insufficient evidence of unlearning. "
                   "MPS_total = 1.0 suggests weight changes "
                   "were too small on toy data.")

    print(f"  Verdict: {verdict}")

    # ── Note on MPS = 1.0 ─────────────────────────────────────────
    print(f"\n  Note on MPS_total = 1.00:")
    print(f"  On the toy dataset (50 patients, 2 rounds), the")
    print(f"  single-iteration update (α=0.01) makes a very small")
    print(f"  weight change. MPS ≈ 1.0 means preservation is good")
    print(f"  but also that the forget signal is tiny. The Layer 2")
    print(f"  Δloss and KL ratio are more sensitive detectors of")
    print(f"  whether forgetting actually occurred.")

    # ── Save report ────────────────────────────────────────────────
    report = {
        "target_patient": target_patient,
        "layer1_mia": {
            "auc_before":         mia_results["before"].get("auc"),
            "auc_after":          mia_results["after"].get("auc"),
            "advantage":          mia_results["after"].get("advantage"),
            "epsilon_proxy":      mia_results["after"].get("epsilon_proxy"),
            "n_target":           mia_results["after"].get("n_target"),
            "n_non_member":       mia_results["after"].get("n_non_member"),
            "passed":             layer1_pass,
            "interpretation":     mia_results["after"].get("interpretation"),
        },
        "layer2_reconstruction": {
            "loss_before":     recon_results["before"]["loss"],
            "loss_after":      recon_results["after"]["loss"],
            "delta_loss":      recon_results.get("delta_loss"),
            "acc_before":      recon_results["before"]["label_accuracy"],
            "acc_after":       recon_results["after"]["label_accuracy"],
            "emb_similarity":  recon_results.get("embedding_similarity_after"),
            "passed":          layer2_pass,
            "interpretation":  recon_results.get("interpretation"),
        },
        "layer3_kl": {
            "kl_unlearned": kl_results.get("kl_unlearned_vs_reference"),
            "kl_random":    kl_results.get("kl_random_vs_reference"),
            "kl_ratio":     kl_results.get("kl_ratio"),
            "passed":       layer3_pass,
            "interpretation": kl_results.get("interpretation"),
        },
        "e6_probe": {
            "A_cross_before": e6_results.get("A_cross_before"),
            "A_cross_after":  e6_results.get("A_cross_after"),
            "A_v_before":     e6_results.get("A_v_before"),
            "A_v_after":      e6_results.get("A_v_after"),
            "A_l_before":     e6_results.get("A_l_before"),
            "A_l_after":      e6_results.get("A_l_after"),
            "A_cross_negative_control": e6_results.get("A_cross_negative_control"),
            "causal_confirmed": e6_pass,
            "passed":           e6_pass,
            "interpretation":   e6_results.get("interpretation"),
        },
        "overall_passed": n_pass,
        "verdict":        verdict,
    }

    Path("logs").mkdir(exist_ok=True)
    report_path = "logs/verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Full report saved: {report_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()