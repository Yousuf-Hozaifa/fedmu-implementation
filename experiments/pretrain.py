"""
experiments/pretrain.py

Federated pre-training pipeline (Phase 2 of the Fed-MU proposal).
Patched for Kaggle / IU dataset use.
"""

import sys
import json
import yaml
import torch
import os
from pathlib import Path
from transformers import CLIPProcessor, DistilBertTokenizer
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.patient_dataset import PatientDataset
from models.multimodal_model import FedMUModel
from models.lora_model import FedMULoRAModel
from federated.nodes.data_splitter import (
    split_patients_for_federated,
    load_node_split,
    load_unlearning_targets,
)
from federated.nodes.federated_node import FederatedNode
from federated.aggregation.federated_server import FederatedServer
from utils.training_monitor import TrainingMonitor


def build_node_dataset(
    node_patient_ids, metadata_file, data_root,
    image_transform, tokenizer,
):
    return PatientDataset(
        metadata_file=metadata_file,
        data_root=data_root,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=node_patient_ids,
    )


def main():
    print("\n" + "="*60)
    print("  PHASE 2: Federated Pre-Training")
    print("="*60)

    # ── Config ────────────────────────────────────────────────────
    with open("configs/fedmu_config.yaml") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["project"]["seed"])

    num_rounds    = config["federated"]["num_rounds"]          
    num_nodes     = config["data"]["num_nodes"]                
    n_unlearn     = config["data"]["unlearning_targets"]       
    lora_rank     = config["lora"]["rank"]                     
    lora_alpha    = config["lora"]["alpha"]                    
    lora_dropout  = config["lora"]["dropout"]                  

    # ── Auto-select dataset based on config ──────────────────────
    dataset_path = config['data'].get('dataset_path', '')
    use_real_data = os.path.isdir(dataset_path)

    meta_file = "data/toy_patients/metadata.json"
    data_root = "data/toy_patients"

    if use_real_data:
        from data.iu_dataset import load_iu_dataset, verify_dataset
        print("[pretrain] Using Indiana University Chest X-ray dataset")
        assert verify_dataset(config), "Dataset verification failed — check paths"
        patients = load_iu_dataset(config, seed=config['project']['seed'])
        
        # Save patients to a temp metadata file so the original splitters don't break
        os.makedirs(data_root, exist_ok=True)
        with open(meta_file, "w") as f:
            json.dump(patients, f)
    else:
        print("[pretrain] WARNING: Real dataset not found, falling back to toy data")
        from data.toy_dataset import ensure_toy_dataset
        n = config['data'].get('total_patients', 50)
        
        print("\n[1/6] Preparing toy patient dataset...")
        ensure_toy_dataset(data_root, n)

    # ── Split data across nodes ────────────────────────────────────
    print("\n[2/6] Splitting patients across federated nodes...")
    split_dir = "data/splits"
    os.makedirs(split_dir, exist_ok=True)

    # Force a fresh split if using real data so we don't accidentally load toy splits
    if use_real_data or not Path(split_dir + "/node_0_patients.json").exists():
        node_splits, unlearn_targets = split_patients_for_federated(
            metadata_file=meta_file,
            num_nodes=num_nodes,
            n_unlearn=n_unlearn,
            seed=config["project"]["seed"],
            output_dir=split_dir,
        )
    else:
        print("  Splits found — loading existing splits.")
        node_splits = {
            i: load_node_split(i, split_dir)
            for i in range(num_nodes)
        }
        unlearn_targets = load_unlearning_targets(split_dir)
        
    print(f"  Unlearning targets: {unlearn_targets}")

    # ── Processors and transforms ──────────────────────────────────
    print("\n[3/6] Loading processors...")
    # These are kept for backward compatibility with the tokenizer args in the dataloader
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

    # ── Build global model ─────────────────────────────────────────
    print("\n[4/6] Building global model with LoRA adapters...")
    
    # FedMULoRAModel requires FedMUModel's interface (projection.layer1/layer2,
    # pixel_values kwarg, dict output). MultimodalModel does not provide it.
    base = FedMUModel(freeze_encoders=False)
    
    global_model = FedMULoRAModel(
        base_model=base,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # ── Build federated nodes ──────────────────────────────────────
    print("\n[5/6] Building federated nodes...")
    nodes = []
    for node_id in range(num_nodes):
        pids = node_splits[node_id]
        dataset = build_node_dataset(
            pids, meta_file, data_root, transform, tokenizer
        )
        node = FederatedNode(
            node_id=node_id,
            model=global_model,
            dataset=dataset,
            config=config,
            device="cpu", # Server assigns GPU automatically if available
        )
        nodes.append(node)

    # ── Federated server ───────────────────────────────────────────
    server  = FederatedServer(global_model, nodes, config)
    monitor = TrainingMonitor(log_dir="logs")

    # ── Federated training ─────────────────────────────────────────
    print("\n[6/6] Running federated training...")
    print(f"  Rounds: {num_rounds}  |  Nodes: {num_nodes}  "
          f"|  Epochs/round: 2")

    for round_num in range(num_rounds):
        record = server.run_round(
            n_epochs=2,
            learning_rate=5e-4,
            batch_size=4,
        )
        monitor.record(record)

    # ── Save final checkpoint and audit log ────────────────────────
    os.makedirs("models/checkpoints", exist_ok=True)
    final_ckpt = "models/checkpoints/pretrained_final.pt"
    server.save_final_checkpoint(final_ckpt)
    server.save_audit_log("models/checkpoints/audit_log.json")

    try:
        monitor.print_summary()
        monitor.plot()
    except Exception as e:
        print(f"Skipping plot generation (headless environment): {e}")

    # ── Re-run E2 with trained model ───────────────────────────────
    print("\n" + "="*60)
    print("  Re-running E2 on TRAINED model to measure ρ(p) shift")
    print("="*60)

    # Use the full dataset for E2 (not per-node subsets)
    full_dataset = PatientDataset(
        metadata_file=meta_file,
        data_root=data_root,
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
    )

    from experiments.e2_rho.rho_computer import RhoComputer
    from experiments.e2_rho.rho_statistics import (
        run_full_e2_analysis, print_e2_report,
    )
    from experiments.e2_rho.rho_visualiser import (
        plot_rho_distribution, plot_per_adapter_norms,
    )

    # Auto-detect GPU for evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    computer   = RhoComputer(global_model, device=device)
    raw_results = computer.compute_all(full_dataset, verbose=True)

    rho_values = [r["rho"] for r in raw_results.values()
                  if r["rho"] is not None]

    stats = run_full_e2_analysis(rho_values, config)
    print_e2_report(stats)

    plot_rho_distribution(
        rho_values=rho_values,
        stats=stats,
        rho_0=config["statistics"]["rho_0"],
        save_path="logs/e2_rho_after_training.png",
        title_suffix=" (after federated pre-training)",
    )
    plot_per_adapter_norms(
        results=raw_results,
        save_path="logs/e2_adapter_norms_after_training.png",
    )

    # ── Highlight unlearning targets ───────────────────────────────
    print("\n  ρ(p) for UNLEARNING TARGETS specifically:")
    print(f"  {'Patient ID':20s}  {'ρ(p)':8s}  {'Note'}")
    print("  " + "-"*50)
    for pid in unlearn_targets:
        if pid in raw_results and raw_results[pid]["rho"] is not None:
            rho = raw_results[pid]["rho"]
            note = ("HIGH — strong A× concentration"
                    if rho > config["statistics"]["rho_0"]
                    else "low — A× not dominant yet")
            print(f"  {pid:20s}  {rho:8.4f}  {note}")

    # ── Final summary ──────────────────────────────────────────────
    ht = stats.get("hypothesis_test", {})
    print("\n" + "="*60)
    print("  PRE-TRAINING COMPLETE")
    print(f"  Checkpoint: {final_ckpt}")
    print(f"  Mean ρ(p) after training: "
          f"{stats['descriptive']['mean']:.4f}")
    if ht.get("reject_h0"):
        print("  ✓ H₀ REJECTED — entanglement hypothesis SUPPORTED")
    else:
        print("  ✗ H₀ not rejected — more training or data may help")
    print("  Next: Step 6 — Selective Federated Aggregation")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()