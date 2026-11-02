"""
federated/nodes/data_splitter.py

Splits the toy patient dataset into per-node subsets.

Design:
  - 5 nodes, each with equal-sized patient sets
  - Unlearning targets are assigned to Nodes 1 and 3
    (so they appear in exactly 2 nodes, matching the proposal)
  - Each node's split is saved so experiments are reproducible
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def split_patients_for_federated(
    metadata_file:   str,
    num_nodes:       int  = 5,
    n_unlearn:       int  = 2,
    seed:            int  = 42,
    output_dir:      str  = "data/splits",
) -> Tuple[Dict[int, List[str]], List[str]]:
    """
    Split patients across federated nodes.

    Rules:
      1. Unlearning targets go to Nodes 1 and 3 ONLY
         (simulates a patient who visited two specific clinics)
      2. Remaining patients are distributed uniformly round-robin
      3. Each node ends up with approximately equal patient counts

    Args:
        metadata_file:  path to metadata.json
        num_nodes:      number of federated nodes
        n_unlearn:      number of unlearning target patients
        seed:           random seed
        output_dir:     where to save split files

    Returns:
        (node_splits, unlearning_targets)
        node_splits: dict mapping node_id → list of patient_ids
        unlearning_targets: list of patient_ids designated for unlearning
    """
    rng = random.Random(seed)

    with open(metadata_file) as f:
        metadata = json.load(f)

    # pretrain.py writes a list of patient dicts for the IU dataset
    if isinstance(metadata, list):
        metadata = {p["patient_id"]: p for p in metadata}

    all_pids = list(metadata.keys())
    rng.shuffle(all_pids)

    # Select unlearning targets from the beginning of the shuffled list
    unlearning_targets = all_pids[:n_unlearn]
    remaining_patients = all_pids[n_unlearn:]

    # Initialise node splits
    node_splits: Dict[int, List[str]] = {i: [] for i in range(num_nodes)}

    # Assign unlearning targets to nodes 1 and 3 only
    for pid in unlearning_targets:
        node_splits[1].append(pid)
        node_splits[3].append(pid)

    # Distribute remaining patients round-robin
    for i, pid in enumerate(remaining_patients):
        node_id = i % num_nodes
        node_splits[node_id].append(pid)

    # Save splits
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for node_id, pids in node_splits.items():
        split_path = out_path / f"node_{node_id}_patients.json"
        with open(split_path, "w") as f:
            json.dump({"node_id": node_id, "patient_ids": pids}, f, indent=2)

    with open(out_path / "unlearning_targets.json", "w") as f:
        json.dump({"targets": unlearning_targets}, f, indent=2)

    # Print summary
    print(f"\n  Federated data split ({num_nodes} nodes):")
    for nid, pids in node_splits.items():
        has_target = any(p in unlearning_targets for p in pids)
        marker = " ← contains unlearning target(s)" if has_target else ""
        print(f"    Node {nid}: {len(pids):3d} patients{marker}")
    print(f"  Unlearning targets: {unlearning_targets}")

    return node_splits, unlearning_targets


def load_node_split(node_id: int, split_dir: str = "data/splits") -> List[str]:
    """Load a previously saved node split."""
    path = Path(split_dir) / f"node_{node_id}_patients.json"
    with open(path) as f:
        return json.load(f)["patient_ids"]


def load_unlearning_targets(split_dir: str = "data/splits") -> List[str]:
    """Load the designated unlearning target patient IDs."""
    path = Path(split_dir) / "unlearning_targets.json"
    with open(path) as f:
        return json.load(f)["targets"]