"""
data/patient_dataset.py

PyTorch Dataset class for patient image-text pairs.
Patched for Kaggle / IU dataset to handle absolute paths and robust JSON loading.
"""

import json
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from typing import Optional


class PatientDataset(Dataset):
    def __init__(
        self,
        metadata_file: str,
        data_root: str,
        image_transform=None,
        tokenizer=None,
        max_text_len: int = 128,
        patient_ids: Optional[list] = None,
    ):
        self.data_root = Path(data_root)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        with open(metadata_file) as f:
            all_metadata = json.load(f)

        # Convert to dict if it's a list (handles our pretrain.py temp dump)
        if isinstance(all_metadata, list):
            all_metadata = {p["patient_id"]: p for p in all_metadata}

        # Filter to specific patients if requested (for federated node splits)
        if patient_ids is not None:
            self.metadata = {
                pid: all_metadata[pid]
                for pid in patient_ids
                if pid in all_metadata
            }
        else:
            self.metadata = all_metadata

        self.patient_ids = list(self.metadata.keys())

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        patient_id = self.patient_ids[idx]
        sample = self.metadata[patient_id]

        # ── 1. Load real image if path exists, else synthetic fallback ────────────
        image_path_str = sample.get("image_path", "")
        
        # If the path is absolute (e.g. Kaggle dataset), use it directly. 
        # Otherwise, append it to data_root.
        if os.path.isabs(image_path_str):
            image_path = Path(image_path_str)
        else:
            image_path = self.data_root / image_path_str

        if image_path.exists() and image_path.is_file():
            image = Image.open(image_path).convert("RGB")
            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                # IU X-rays are grayscale, enforce 3 channels for ViT
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.Grayscale(num_output_channels=3), 
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                ])
                image = transform(image)
        else:
            # Fallback: synthetic noise (toy mode)
            image = torch.randn(3, 224, 224)

        # ── 2. Report / Text ──────────────────────────────────────────────────────
        report = sample.get("report", "No findings.")
        
        if self.tokenizer is not None:
            text_enc = self.tokenizer(
                report,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_enc["input_ids"].squeeze(0)
            text_attention_mask = text_enc["attention_mask"].squeeze(0)
        else:
            text_input_ids = report
            text_attention_mask = None

        # ── 3. Labels ──────────────────────────────────────────────────────────────
        # Handle if 'labels' or 'label' key is used
        labels_list = sample.get("labels", sample.get("label", [0]*14))
        labels = torch.tensor(labels_list, dtype=torch.float32)

        return {
            "patient_id": patient_id,
            "patient_idx": sample.get("patient_idx", idx),
            "image": image,                     # [3, 224, 224]
            "text_input_ids": text_input_ids,   # [max_text_len] or str
            "text_attention_mask": text_attention_mask,
            "labels": labels,                   # [14]
            "report": report,                   # raw text
        }

    def get_patient_ids(self) -> list:
        return self.patient_ids.copy()