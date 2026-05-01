"""
federated/aggregation/selective_aggregation.py

Implements selective federated aggregation for unlearning.

Standard FedAvg during pre-training:
    Aggregates from ALL nodes → broadcasts to ALL nodes

Selective protocol during unlearning:
    Aggregates from AFFECTED nodes K_p ONLY
    → broadcasts updated A× only to affected nodes
    → unaffected nodes receive ZERO update (provable correctness)

Correctness property:
    For any node k ∉ K_p:
        A×^(k)_after = A×^(k)_before   (identically unchanged)
    This holds by protocol definition, not just empirically.
"""

import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import sys

# Return type alias for aggregate() — (state_dict, log_dict)
Tuple_like = Tuple[dict, dict]

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SelectiveAggregator:
    """
    Performs selective federated aggregation during unlearning.

    Only aggregates updated A× from nodes in K_p (affected nodes).
    Returns the aggregated state and a detailed log of what happened.
    """

    def __init__(self, audit_log: Dict[str, List[int]]):
        """
        Args:
            audit_log: maps patient_id → [node_ids that trained on it]
                       Loaded from models/checkpoints/audit_log.json
        """
        self.audit_log = audit_log

    @classmethod
    def from_file(cls, path: str = "models/checkpoints/audit_log.json"):
        """Load audit log from disk and create aggregator."""
        with open(path) as f:
            log = json.load(f)
        return cls(log)

    def get_affected_nodes(self, patient_id: str) -> List[int]:
        """Return node IDs in K_p for a given patient."""
        return self.audit_log.get(patient_id, [])

    def aggregate(
        self,
        patient_id:        str,
        node_results:      List[Dict],
        unaffected_state:  dict,
    ) -> Tuple_like:
        """
        Perform selective aggregation.

        Args:
            patient_id:       patient being unlearned
            node_results:     list of dicts, each with:
                              {'node_id', 'state', 'n_samples', 'result'}
            unaffected_state: the original A× state (for unaffected nodes)

        Returns:
            (aggregated_state, aggregation_log)
        """
        affected_ids = self.get_affected_nodes(patient_id)

        # Separate affected and unaffected results
        affected   = [r for r in node_results if r["node_id"] in affected_ids]
        unaffected = [r for r in node_results if r["node_id"] not in affected_ids]

        log = {
            "patient_id":        patient_id,
            "affected_nodes":    affected_ids,
            "unaffected_nodes":  [r["node_id"] for r in unaffected],
            "n_affected":        len(affected),
            "n_unaffected":      len(unaffected),
            "aggregation_type":  "selective",
        }

        if not affected:
            log["warning"] = "No affected nodes returned results"
            return unaffected_state, log

        # Weighted average over affected nodes only
        total_samples = sum(r["n_samples"] for r in affected)

        aggregated = None
        for r in affected:
            weight = r["n_samples"] / total_samples
            state  = r["state"]

            if aggregated is None:
                aggregated = {k: weight * v.clone()
                              for k, v in state.items()}
            else:
                for k in aggregated:
                    aggregated[k] += weight * state[k]

        log["weights"] = {
            r["node_id"]: r["n_samples"] / total_samples
            for r in affected
        }

        return aggregated, log

    def verify_unaffected_unchanged(
        self,
        nodes:            list,
        patient_id:       str,
        state_before:     dict,
        state_after:      dict,
    ) -> Dict:
        """
        Verify that unaffected nodes received zero delta.

        For each node k ∉ K_p, confirm:
            ||A×_after - A×_before||_F ≈ 0

        This is the E3 experiment verification.
        """
        affected_ids = self.get_affected_nodes(patient_id)
        results = {}

        for node in nodes:
            nid = node.node_id
            is_affected = nid in affected_ids

            # Get node's current A× state
            current_state = node.model.get_cross_adapter_state()

            # Compute delta
            delta_norm = 0.0
            for key in state_before:
                if key in current_state:
                    diff = (current_state[key] - state_before[key]).norm().item()
                    delta_norm += diff ** 2
            delta_norm = delta_norm ** 0.5

            results[nid] = {
                "node_id":     nid,
                "is_affected": is_affected,
                "delta_norm":  delta_norm,
                "unchanged":   delta_norm < 1e-6 if not is_affected else None,
            }

        return results