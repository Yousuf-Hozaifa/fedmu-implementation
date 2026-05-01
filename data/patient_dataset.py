"""
data/patient_dataset.py

PyTorch Dataset class for patient image-text pairs.

This class is architecture-agnostic: it loads images and text and
returns raw tensors. The model (vision encoder + text encoder) is
applied separately, not inside the dataset.

This mirrors how MIMIC-CXR would be loaded for the full Fed-MU system.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Optional


class PatientDataset(Dataset):
    """
    Loads patient (image, report, labels, patient_id) tuples.

    Args:
        metadata_file:  path to metadata.json
        data_root:      root directory containing the images/ folder
        image_transform: torchvision transform for images (applied to PIL Image)
        tokenizer:      HuggingFace tokenizer for the text encoder
        max_text_len:   maximum token length for reports
        patient_ids:    optional list of patient_ids to include
                        (used to create per-node subsets in federated setup)
    """

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

        # Load image
        image_path = self.data_root / sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        # Apply image transform if provided
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            # Default: convert to tensor [3, 224, 224] in [0, 1]
            import torchvision.transforms as T
            default_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            image = default_transform(image)

        # Tokenize text if tokenizer provided
        report = sample["report"]
        if self.tokenizer is not None:
            text_enc = self.tokenizer(
                report,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Remove batch dimension added by return_tensors="pt"
            text_input_ids = text_enc["input_ids"].squeeze(0)
            text_attention_mask = text_enc["attention_mask"].squeeze(0)
        else:
            # Return raw text if no tokenizer
            text_input_ids = report
            text_attention_mask = None

        # Labels: 14-dim float tensor
        labels = torch.tensor(sample["labels"], dtype=torch.float32)

        return {
            "patient_id": patient_id,
            "patient_idx": sample["patient_idx"],
            "image": image,                     # [3, 224, 224]
            "text_input_ids": text_input_ids,   # [max_text_len] or raw str
            "text_attention_mask": text_attention_mask,
            "labels": labels,                   # [14]
            "report": report,                   # raw text (always included)
        }

    def get_patient_ids(self) -> list:
        return self.patient_ids.copy()