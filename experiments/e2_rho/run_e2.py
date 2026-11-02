"""
experiments/e2_rho/run_e2.py

Full Experiment E2 runner.

Steps:
  1. Load model and dataset
  2. Compute ρ(p) for all patients
  3. Run full statistical analysis
  4. Generate plots
  5. Save results to JSON

Usage:
    python experiments/e2_rho/run_e2.py
    python experiments/e2_rho/run_e2.py --trained  (after Step 5)
"""

import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from transformers import CLIPProcessor, DistilBertTokenizer
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.toy_dataset import ensure_toy_dataset
from data.patient_dataset import PatientDataset
from models.multimodal_model import FedMUModel
from models.lora_model import FedMULoRAModel
from experiments.e2_rho.rho_computer import RhoComputer
from experiments.e2_rho.rho_statistics import (
    run_full_e2_analysis, print_e2_report
)
from experiments.e2_rho.rho_visualiser import (
    plot_rho_distribution, plot_per_adapter_norms
)


def run_e2(checkpoint_path: str = None):
    print("\n" + "="*60)
    print("  EXPERIMENT E2: Gradient Concentration Ratio ρ(p)")
    print("="*60)

    # ── Load config ────────────────────────────────────────────────
    with open("configs/fedmu_config.yaml") as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["project"]["seed"])

    # ── Prepare data ───────────────────────────────────────────────
    ensure_toy_dataset("data/toy_patients", config["data"]["total_patients"])

    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=clip_proc.image_processor.image_mean,
                    std=clip_proc.image_processor.image_std),
    ])

    dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
    )
    print(f"\n  Dataset: {len(dataset)} patients loaded.")

    # ── Build model ────────────────────────────────────────────────
    print("\n  Building model with LoRA adapters...")
    base = FedMUModel(freeze_encoders=False)
    model = FedMULoRAModel(
        base_model=base,
        lora_rank=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
    )

    # Load checkpoint if provided (use after Step 5 training)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\n  Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  Checkpoint loaded.")
    else:
        print("\n  No checkpoint — using randomly initialised model.")
        print("  Note: ρ(p) on random weights has no semantic meaning.")
        print("  Run again after Step 5 (pre-training) for meaningful results.")

    # ── Compute ρ(p) ───────────────────────────────────────────────
    print("\n  Computing ρ(p) for all patients...")
    computer = RhoComputer(model, device="cpu")
    raw_results = computer.compute_all(dataset, verbose=True)

    # Extract ρ values
    rho_values  = [r["rho"] for r in raw_results.values()
                   if r["rho"] is not None]
    patient_ids = [pid for pid, r in raw_results.items()
                   if r["rho"] is not None]

    print(f"\n  Computed {len(rho_values)} valid ρ(p) values.")

    # Show per-patient table
    print("\n  Per-patient ρ(p) values:")
    print(f"  {'Patient ID':20s}  {'ρ(p)':8s}  {'||∇A×||':10s}  {'||∇Av||':10s}  {'||∇Al||':10s}")
    print("  " + "-"*62)
    for pid in patient_ids[:15]:  # show first 15
        r = raw_results[pid]
        print(f"  {pid:20s}  {r['rho']:8.4f}  "
              f"{r['norm_cross']:10.6f}  "
              f"{(r['norm_v'] or 0):10.6f}  "
              f"{(r['norm_l'] or 0):10.6f}")
    if len(patient_ids) > 15:
        print(f"  ... ({len(patient_ids)-15} more)")

    # ── Statistical analysis ───────────────────────────────────────
    print("\n  Running statistical analysis...")
    stats = run_full_e2_analysis(rho_values, config, patient_ids)
    print_e2_report(stats)

    # ── Visualisations ─────────────────────────────────────────────
    Path("logs").mkdir(exist_ok=True)
    suffix = " (trained)" if checkpoint_path else " (random init)"

    print("\n  Generating plots...")
    plot_rho_distribution(
        rho_values=rho_values,
        stats=stats,
        rho_0=config["statistics"]["rho_0"],
        save_path="logs/e2_rho_distribution.png",
        title_suffix=suffix,
    )
    plot_per_adapter_norms(
        results=raw_results,
        save_path="logs/e2_adapter_norms.png",
    )

    # ── Save results ───────────────────────────────────────────────
    output = {
        "per_patient": {
            pid: {k: v for k, v in r.items()
                  if isinstance(v, (int, float, str, type(None)))}
            for pid, r in raw_results.items()
        },
        "statistics": {
            k: v for k, v in stats.items()
            if not isinstance(v, dict)
            or all(isinstance(vv, (int, float, str, bool, type(None)))
                   for vv in v.values())
        },
    }

    out_path = "logs/e2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    print("\n" + "="*60)
    print("  E2 COMPLETE")
    ht = stats.get("hypothesis_test", {})
    if ht.get("reject_h0"):
        print("  ✓ Entanglement hypothesis SUPPORTED on this data")
    else:
        print("  ✗ Entanglement hypothesis NOT supported yet")
        print("    (expected on random init — re-run after Step 5)")
    print("="*60 + "\n")

    return stats, raw_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    args = parser.parse_args()
    run_e2(checkpoint_path=args.checkpoint)