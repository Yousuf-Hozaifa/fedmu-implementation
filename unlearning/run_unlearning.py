"""
unlearning/run_unlearning.py

Full unlearning pipeline for the Fed-MU system.

Orchestrates:
  1. Load trained model and federated setup
  2. Select target patient from unlearning_targets.json
  3. Build retention datasets (all patients except target)
  4. Record MPS baseline
  5. Run LoRA-Subtraction on affected nodes
  6. Selective aggregation (only affected nodes contribute)
  7. Verify unaffected nodes unchanged (E3)
  8. Full MPS evaluation
  9. Save unlearned checkpoint and detailed report
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
from federated.nodes.federated_node import FederatedNode
from federated.nodes.data_splitter import load_node_split, load_unlearning_targets
from federated.aggregation.selective_aggregation import SelectiveAggregator
from unlearning.algorithms.lora_subtraction import LoRASubtractionUnlearner
from unlearning.verification.mps_evaluator import MPSEvaluator


def build_retention_dataset(
    full_dataset: PatientDataset,
    patient_to_exclude: str,
    tokenizer,
    transform,
) -> PatientDataset:
    """
    Build a retention dataset: all patients EXCEPT the target.
    This is fed to the preserve gradient computation.
    """
    remaining_ids = [
        pid for pid in full_dataset.patient_ids
        if pid != patient_to_exclude
    ]
    return PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
        patient_ids=remaining_ids,
    )


def main():
    print("\n" + "="*60)
    print("  STEP 6: LoRA-Subtraction Unlearning Pipeline")
    print("="*60)

    # ── Config ────────────────────────────────────────────────────
    with open("configs/fedmu_config.yaml") as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["project"]["seed"])

    num_nodes = config["data"]["num_nodes"]  # 5

    # ── Check prerequisites ────────────────────────────────────────
    checkpoint_path = "models/checkpoints/pretrained_final.pt"
    audit_log_path  = "models/checkpoints/audit_log.json"

    if not Path(checkpoint_path).exists():
        print(f"\n  ERROR: {checkpoint_path} not found.")
        print("  Run Step 5 first: python experiments/pretrain.py")
        return

    # ── Processors and transforms ──────────────────────────────────
    print("\n[1/9] Loading processors...")
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

    # ── Full dataset ───────────────────────────────────────────────
    print("\n[2/9] Loading full patient dataset...")
    full_dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
    )
    print(f"  {len(full_dataset)} total patients")

    # ── Load target patient ────────────────────────────────────────
    print("\n[3/9] Selecting unlearning target...")
    unlearn_targets = load_unlearning_targets("data/splits")
    target_patient_id = unlearn_targets[0]  # Unlearn the first target
    print(f"  Target patient: {target_patient_id}")

    # Get the patient's sample
    pid_to_idx = {pid: i for i, pid in enumerate(full_dataset.patient_ids)}
    if target_patient_id not in pid_to_idx:
        print(f"  ERROR: {target_patient_id} not in dataset")
        return
    target_sample = full_dataset[pid_to_idx[target_patient_id]]
    print(f"  Report (first 80 chars): {target_sample['report'][:80]}...")

    # ── Build global model and load checkpoint ─────────────────────
    print("\n[4/9] Loading trained model checkpoint...")
    base         = FedMUModel(freeze_encoders=False)
    global_model = FedMULoRAModel(
        base_model=base,
        lora_rank=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    global_model.load_state_dict(ckpt, strict=False)
    print(f"  Checkpoint loaded from: {checkpoint_path}")

    # Save original A× state — needed for E3 verification
    original_cross_state = global_model.get_cross_adapter_state()

    # ── Build federated nodes ──────────────────────────────────────
    print("\n[5/9] Building federated nodes...")
    nodes = []
    for nid in range(num_nodes):
        pids    = load_node_split(nid, "data/splits")
        dataset = PatientDataset(
            metadata_file="data/toy_patients/metadata.json",
            data_root="data/toy_patients",
            image_transform=transform,
            tokenizer=tokenizer,
            max_text_len=128,
            patient_ids=pids,
        )
        node = FederatedNode(nid, global_model, dataset, config)
        # Load the trained checkpoint into each node's model
        node.model.load_state_dict(ckpt, strict=False)
        nodes.append(node)

    # ── Record MPS baseline ────────────────────────────────────────
    print("\n[6/9] Recording MPS baseline (pre-unlearning)...")
    retain_dataset = build_retention_dataset(
        full_dataset, target_patient_id, tokenizer, transform
    )

    # Use the global model for baseline — it reflects the trained state
    mps_evaluator = MPSEvaluator(
        global_model, retain_dataset, device="cpu"
    )
    baseline = mps_evaluator.record_baseline()

    # ── Load selective aggregator ──────────────────────────────────
    print("\n[7/9] Setting up selective aggregation...")
    aggregator   = SelectiveAggregator.from_file(audit_log_path)
    affected_ids = aggregator.get_affected_nodes(target_patient_id)
    print(f"  Patient {target_patient_id} was trained at "
          f"nodes: {affected_ids}")
    print(f"  Unaffected nodes: "
          f"{[n.node_id for n in nodes if n.node_id not in affected_ids]}")

    # ── Run LoRA-Subtraction on affected nodes ─────────────────────
    print("\n[8/9] Running LoRA-Subtraction on affected nodes...")
    node_results  = []
    unlearn_log   = {}

    for node in nodes:
        if node.node_id not in affected_ids:
            print(f"\n  Node {node.node_id}: UNAFFECTED — zero update")
            node_results.append({
                "node_id":   node.node_id,
                "state":     node.model.get_cross_adapter_state(),
                "n_samples": len(node.dataset),
                "result":    None,
                "affected":  False,
            })
            continue

        # This node trained on the target patient
        print(f"\n  Node {node.node_id}: AFFECTED — running LoRA-Subtraction")

        # Build this node's retention dataset
        # (all patients at this node except the target)
        node_retention_ids = [
            pid for pid in node.patient_ids
            if pid != target_patient_id
        ]
        node_retention = PatientDataset(
            metadata_file="data/toy_patients/metadata.json",
            data_root="data/toy_patients",
            image_transform=transform,
            tokenizer=tokenizer,
            max_text_len=128,
            patient_ids=node_retention_ids if node_retention_ids
                       else full_dataset.patient_ids[:5],
        )

        # Create per-node MPS evaluator
        node_mps = MPSEvaluator(
            node.model, node_retention, device="cpu"
        )
        node_mps.record_baseline()

        # Run the unlearning algorithm
        unlearner = LoRASubtractionUnlearner(
            node.model, config, device="cpu"
        )
        result = unlearner.unlearn(
            patient_id=target_patient_id,
            patient_sample=target_sample,
            retention_dataset=node_retention,
            mps_evaluator=node_mps,
            node_id=node.node_id,
            verbose=True,
        )

        unlearn_log[node.node_id] = result.to_dict()
        print(f"\n  Node {node.node_id} result: {result}")

        node_results.append({
            "node_id":   node.node_id,
            "state":     node.model.get_cross_adapter_state(),
            "n_samples": sum(1 for pid in node.patient_ids
                             if pid == target_patient_id),
            "result":    result,
            "affected":  True,
        })

    # ── Selective aggregation ──────────────────────────────────────
    print("\n  Running selective aggregation...")
    aggregated_state, agg_log = aggregator.aggregate(
        patient_id=target_patient_id,
        node_results=node_results,
        unaffected_state=original_cross_state,
    )

    print(f"  Aggregated from nodes: {agg_log['affected_nodes']}")
    print(f"  Weights: {agg_log.get('weights', {})}")
    print(f"  Unaffected nodes unchanged: "
          f"{agg_log['unaffected_nodes']}")

    # Apply to global model
    global_model.set_cross_adapter_state(aggregated_state)
    # Apply to unaffected nodes (zero delta — their state is unchanged)
    for node in nodes:
        if node.node_id not in affected_ids:
            node.model.set_cross_adapter_state(original_cross_state)
        else:
            node.model.set_cross_adapter_state(aggregated_state)

    # ── E3: Verify unaffected nodes unchanged ──────────────────────
    print("\n  E3 Verification: unaffected nodes unchanged?")
    e3_results = aggregator.verify_unaffected_unchanged(
        nodes, target_patient_id, original_cross_state, aggregated_state
    )
    for nid, res in e3_results.items():
        status = ("AFFECTED" if res["is_affected"]
                  else f"unchanged={res['unchanged']} (Δ={res['delta_norm']:.2e})")
        print(f"    Node {nid}: {status}")

    # ── Final MPS evaluation (global model) ────────────────────────
    print("\n[9/9] Final MPS evaluation on global model...")
    final_mps = mps_evaluator.compute_mps()
    mps_evaluator.print_mps(final_mps, tau=config["unlearning"]["tau_mps"])

    # ── Save unlearned checkpoint ──────────────────────────────────
    unlearned_path = "models/checkpoints/unlearned_final.pt"
    torch.save(global_model.state_dict(), unlearned_path)

    # ── Save full report ───────────────────────────────────────────
    report = {
        "target_patient":   target_patient_id,
        "affected_nodes":   affected_ids,
        "unlearning_log":   unlearn_log,
        "aggregation":      agg_log,
        "e3_verification":  {k: {kk: vv for kk, vv in v.items()
                                 if isinstance(vv, (bool, float, int, str,
                                                    type(None)))}
                             for k, v in e3_results.items()},
        "mps_baseline":     baseline,
        "mps_final": {
            "mps_v":     final_mps["mps_v"],
            "mps_l":     final_mps["mps_l"],
            "mps_cross": final_mps["mps_cross"],
            "mps_total": final_mps["mps_total"],
        },
    }

    report_path = "logs/unlearning_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 6 COMPLETE: Unlearning Pipeline")
    print("─"*60)
    print(f"  Target patient:         {target_patient_id}")
    print(f"  Affected nodes:         {affected_ids}")
    print(f"  Unaffected nodes:       "
          f"{[n.node_id for n in nodes if n.node_id not in affected_ids]}")
    any_success = any(
        v.get("success") for v in unlearn_log.values()
    )
    print(f"  Unlearning outcome:     "
          f"{'SUCCESS' if any_success else 'PARTIAL/FAILED'}")
    print(f"  MPS_total (final):      {final_mps['mps_total']:.4f}")
    print(f"  MPS_v:                  {final_mps['mps_v']:.4f}")
    print(f"  MPS_l:                  {final_mps['mps_l']:.4f}")
    print(f"  MPS_cross:              {final_mps['mps_cross']:.4f}")
    print(f"  Unlearned checkpoint:   {unlearned_path}")
    print(f"  Detailed report:        {report_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()