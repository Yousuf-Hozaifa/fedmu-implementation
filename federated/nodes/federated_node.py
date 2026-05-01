"""
federated/nodes/federated_node.py

A single federated clinic node.

Each node:
  1. Holds a local copy of the global model
  2. Has its own patient dataset (never shared)
  3. Trains locally for a fixed number of epochs
  4. Returns only the updated LoRA adapter weights

Critically: raw patient data NEVER leaves the node.
Only adapter weight deltas are communicated to the server.

This mirrors what would happen in production: each clinic
trains on its local PACS system and only uploads small
weight matrices (~6KB for rank-4 adapters).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from pathlib import Path
import copy
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.lora_model import FedMULoRAModel


class FederatedNode:
    """
    Represents one clinical site in the federated system.

    Args:
        node_id:   integer identifier (0–4)
        model:     the shared FedMULoRAModel (each node gets its own copy)
        dataset:   PatientDataset for this node's local patients
        config:    fedmu_config dict
        device:    'cpu' or 'cuda'
    """

    def __init__(
        self,
        node_id: int,
        model: FedMULoRAModel,
        dataset,
        config: dict,
        device: str = "cpu",
    ):
        self.node_id   = node_id
        self.config    = config
        self.device    = device
        self.dataset   = dataset

        # Each node has its own independent copy of the model
        # This is essential: nodes must not share memory references
        self.model = copy.deepcopy(model)
        self.model.to(device)

        # Track which patients this node has (for the audit log)
        self.patient_ids: List[str] = dataset.get_patient_ids()

        print(f"  Node {node_id}: {len(self.patient_ids)} patients "
              f"[{self.patient_ids[0]} ... {self.patient_ids[-1]}]")

    def receive_global_state(self, global_cross_state: dict):
        """
        Receive the server's current global A× parameters.
        Updates this node's local A× to match the global state.

        In production: this is a small network transfer (~6KB).
        """
        self.model.set_cross_adapter_state(global_cross_state)

    def local_train(
        self,
        n_epochs:      int   = 1,
        learning_rate: float = 1e-3,
        batch_size:    int   = 4,
    ) -> Dict:
        """
        Train locally on this node's patient data.

        Only A× (cross-modal adapter) trains during pre-training.
        A_v, A_l, and the base encoders remain frozen.

        Returns:
            dict with:
                'state':      updated A× state dict
                'n_samples':  number of training samples (for aggregation weighting)
                'train_loss': mean training loss
                'node_id':    this node's ID
        """
        self.model.set_pretraining_mode()

        dataloader = DataLoader(
            self.dataset,
            batch_size=min(batch_size, len(self.dataset)),
            shuffle=True,
            drop_last=False,
        )

        # Only optimise trainable parameters (A× + classifier)
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.model.train()
        total_loss  = 0.0
        n_batches   = 0

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()

                pixel_values  = batch["image"].to(self.device)
                input_ids     = batch["text_input_ids"].to(self.device)
                attn_mask     = batch["text_attention_mask"].to(self.device)
                labels        = batch["labels"].to(self.device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )

                loss = outputs["loss"]
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )

                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            total_loss += epoch_loss

        mean_loss = total_loss / max(n_batches, 1)

        return {
            "node_id":    self.node_id,
            "state":      self.model.get_cross_adapter_state(),
            "n_samples":  len(self.dataset),
            "train_loss": mean_loss,
        }

    def evaluate(self, batch_size: int = 4) -> Dict:
        """
        Evaluate model on this node's local validation data.
        Returns loss and per-class accuracy.
        """
        self.model.eval()
        dataloader = DataLoader(
            self.dataset,
            batch_size=min(batch_size, len(self.dataset)),
            shuffle=False,
        )

        total_loss = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
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

                total_loss += outputs["loss"].item()

                # Pathology predictions (threshold at 0.5)
                probs = torch.sigmoid(outputs["logits"])
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds  = torch.cat(all_preds,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Per-class accuracy
        correct = (all_preds == all_labels).float()
        accuracy = correct.mean().item()

        return {
            "node_id":  self.node_id,
            "val_loss": total_loss / max(len(dataloader), 1),
            "accuracy": accuracy,
        }

    def contains_patient(self, patient_id: str) -> bool:
        """Check whether this node trained on a specific patient."""
        return patient_id in self.patient_ids