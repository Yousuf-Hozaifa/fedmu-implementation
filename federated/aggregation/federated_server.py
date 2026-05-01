"""
federated/aggregation/federated_server.py

The central server that coordinates federated training.

Responsibilities:
  1. Maintain the global A× adapter state
  2. Distribute the current global state to all nodes
  3. Collect updated states from nodes
  4. Aggregate using weighted FedAvg
  5. Maintain an audit log: patient_id → [node_ids that trained on it]
     (This audit log enables selective aggregation during unlearning)
  6. Save checkpoints after each round
"""

import torch
import json
import copy
from typing import Dict, List, Optional
from pathlib import Path


class FederatedServer:
    """
    Central coordinator for the federated learning system.

    The server NEVER sees raw patient data.
    It only receives and aggregates adapter weight matrices.
    """

    def __init__(
        self,
        global_model,
        nodes: List,
        config: dict,
        checkpoint_dir: str = "models/checkpoints",
    ):
        self.global_model     = global_model
        self.nodes            = nodes
        self.config           = config
        self.checkpoint_dir   = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Round counter and history
        self.current_round  = 0
        self.round_history: List[Dict] = []

        # Audit log: maps patient_id → list of node_ids
        # Built from node.patient_ids at init time
        self.audit_log: Dict[str, List[int]] = {}
        self._build_audit_log()

        print(f"\n  Server: managing {len(nodes)} nodes")
        print(f"  Audit log: {len(self.audit_log)} unique patients tracked")

    def _build_audit_log(self):
        """
        Build the audit log mapping patient_id → [node_ids].
        This is used during unlearning to find affected nodes.
        """
        for node in self.nodes:
            for pid in node.patient_ids:
                if pid not in self.audit_log:
                    self.audit_log[pid] = []
                self.audit_log[pid].append(node.node_id)

    def get_affected_nodes(self, patient_id: str) -> List[int]:
        """
        Return which nodes trained on a given patient.
        Used to determine K_p during selective aggregation.
        """
        return self.audit_log.get(patient_id, [])

    def _fedavg_aggregate(
        self,
        node_results: List[Dict],
        affected_node_ids: Optional[List[int]] = None,
    ) -> dict:
        """
        Weighted FedAvg aggregation.

        A×_global = Σ_{k ∈ K} [n_k / Σ_j n_j] × A×^(k)

        Args:
            node_results:      list of dicts from local_train()
            affected_node_ids: if set, only aggregate from these nodes
                               (selective aggregation for unlearning)

        Returns:
            aggregated A× state dict
        """
        if affected_node_ids is not None:
            results = [r for r in node_results
                       if r["node_id"] in affected_node_ids]
            if not results:
                print("  WARNING: No results from affected nodes — "
                      "returning unchanged state")
                return self.global_model.get_cross_adapter_state()
        else:
            results = node_results

        # Compute weighted sum
        total_samples = sum(r["n_samples"] for r in results)

        aggregated = None
        for result in results:
            weight = result["n_samples"] / total_samples
            state  = result["state"]

            if aggregated is None:
                aggregated = {k: weight * v.clone()
                              for k, v in state.items()}
            else:
                for k in aggregated:
                    aggregated[k] += weight * state[k]

        return aggregated

    def run_round(
        self,
        n_epochs:      int   = 1,
        learning_rate: float = 1e-3,
        batch_size:    int   = 4,
    ) -> Dict:
        """
        Execute one full federated round:
          1. Distribute current global A× to all nodes
          2. Each node trains locally
          3. Server aggregates updates
          4. Update global model
          5. Save checkpoint

        Returns:
            dict with round metrics
        """
        self.current_round += 1
        print(f"\n  ── Round {self.current_round} ──────────────────────────")

        # Step 1: Distribute global A× to all nodes
        global_state = self.global_model.get_cross_adapter_state()
        for node in self.nodes:
            node.receive_global_state(global_state)

        # Step 2: Local training at each node
        node_results = []
        for node in self.nodes:
            result = node.local_train(
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            node_results.append(result)
            print(f"    Node {node.node_id}: "
                  f"loss={result['train_loss']:.4f}, "
                  f"n={result['n_samples']}")

        # Step 3: Aggregate (standard FedAvg — all nodes)
        aggregated_state = self._fedavg_aggregate(node_results)

        # Step 4: Update global model
        self.global_model.set_cross_adapter_state(aggregated_state)

        # Step 5: Evaluate on each node
        val_losses = []
        for node in self.nodes:
            node.receive_global_state(aggregated_state)
            eval_result = node.evaluate(batch_size=batch_size)
            val_losses.append(eval_result["val_loss"])

        mean_train = sum(r["train_loss"] for r in node_results) / len(node_results)
        mean_val   = sum(val_losses) / len(val_losses)

        print(f"    Round {self.current_round} summary: "
              f"train_loss={mean_train:.4f}, "
              f"val_loss={mean_val:.4f}")

        # Step 6: Checkpoint
        ckpt_path = (self.checkpoint_dir /
                     f"round_{self.current_round:03d}.pt")
        torch.save(self.global_model.state_dict(), ckpt_path)
        print(f"    Checkpoint: {ckpt_path}")

        round_record = {
            "round":      self.current_round,
            "train_loss": mean_train,
            "val_loss":   mean_val,
        }
        self.round_history.append(round_record)
        return round_record

    def run_all_rounds(
        self,
        num_rounds:    int   = 2,
        n_epochs:      int   = 1,
        learning_rate: float = 1e-3,
        batch_size:    int   = 4,
    ) -> List[Dict]:
        """Run all federated rounds and return the full history."""
        print(f"\n  Starting federated pre-training "
              f"({num_rounds} rounds, {n_epochs} epoch/round)")
        for _ in range(num_rounds):
            self.run_round(n_epochs, learning_rate, batch_size)

        return self.round_history

    def save_final_checkpoint(self, path: str = "models/checkpoints/final.pt"):
        """Save the final global model state."""
        torch.save(self.global_model.state_dict(), path)
        print(f"\n  Final checkpoint saved: {path}")

    def save_audit_log(self, path: str = "models/checkpoints/audit_log.json"):
        """Save the audit log for use during unlearning."""
        with open(path, "w") as f:
            json.dump(self.audit_log, f, indent=2)
        print(f"  Audit log saved: {path}")