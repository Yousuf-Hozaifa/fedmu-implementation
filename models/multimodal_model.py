"""
models/multimodal_model.py

A lightweight multimodal model that mirrors LLaVA-1.5's architecture:

    CLIP ViT-B/32          →  visual tokens
    2-layer MLP projection →  maps visual to text embedding space  ← A× lives here
    DistilBERT             →  processes text + fused visual tokens
    Classification head    →  14-class pathology prediction

Architecture diagram:
    [Image 224x224] → CLIP ViT-B/32 → [visual_emb: 512-dim]
                                            ↓
                                    MLP Projection (512 → 768)  ← CROSS-MODAL BRIDGE
                                            ↓
    [Report text]  → DistilBERT emb → [text_emb: 768-dim]
                                            ↓
                                    Concatenate + Pool (768-dim)
                                            ↓
                                    Classifier → [14 logits]

LoRA adapters will be added on top of this in Step 3.
For now, this is a clean base model.
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPProcessor,
    DistilBertModel,
    DistilBertTokenizer,
)
from typing import Optional, Tuple


class CrossModalProjection(nn.Module):
    """
    2-layer MLP that maps CLIP visual embeddings (512-dim)
    into the DistilBERT embedding space (768-dim).

    This is the cross-modal binding layer — the component that
    Fed-MU's A× adapter targets. It is what creates the
    visual-linguistic joint representation.

    Why 2 layers? LLaVA-1.5 specifically uses a 2-layer MLP here
    (an improvement over LLaVA-1.0's single linear layer) because
    the additional non-linearity allows richer cross-modal mapping.
    """

    def __init__(self, visual_dim: int = 512, text_dim: int = 768):
        super().__init__()
        self.layer1 = nn.Linear(visual_dim, text_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(text_dim, text_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [batch, visual_dim]
        Returns:
            projected: [batch, text_dim]
        """
        x = self.layer1(visual_features)   # [batch, 768]
        x = self.activation(x)
        x = self.layer2(x)                 # [batch, 768]
        return x


class FedMUModel(nn.Module):
    """
    Complete multimodal model for the Fed-MU toy pipeline.

    Components:
      - vision_encoder:  CLIP ViT-B/32 (FROZEN during unlearning)
      - projection:      CrossModalProjection (A× adapter targets here)
      - text_encoder:    DistilBERT (FROZEN during unlearning)
      - classifier:      Linear → 14 pathology classes
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        bert_model_name: str = "distilbert-base-uncased",
        num_labels: int = 14,
        freeze_encoders: bool = True,
    ):
        super().__init__()

        print("Loading CLIP vision encoder...")
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.visual_dim = self.vision_encoder.config.hidden_size  # 768 for ViT-B/32

        print("Loading DistilBERT text encoder...")
        self.text_encoder = DistilBertModel.from_pretrained(bert_model_name)
        self.text_dim = self.text_encoder.config.hidden_size  # 768

        # Cross-modal projection (the A× target)
        self.projection = CrossModalProjection(
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
        )

        # Pathology classifier
        # Input: concatenated [visual_proj + text_cls] = 768 + 768 = 1536
        self.classifier = nn.Sequential(
            nn.Linear(self.text_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

        # Freeze encoders if requested
        # (In Fed-MU, only the LoRA adapters and projection are trained)
        if freeze_encoders:
            self._freeze_encoders()

        self.num_labels = num_labels

    def _freeze_encoders(self):
        """Freeze CLIP and DistilBERT — only projection and classifier train."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        print("  Encoders frozen. Trainable: projection + classifier only.")

    def get_trainable_param_count(self) -> dict:
        """Returns count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual embedding from image tensor.

        Args:
            pixel_values: [batch, 3, 224, 224]
        Returns:
            visual_emb: [batch, visual_dim]  (CLS token embedding)
        """
        outputs = self.vision_encoder(pixel_values=pixel_values)
        # CLS token (index 0) represents the whole image
        return outputs.last_hidden_state[:, 0, :]  # [batch, 768]

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract text embedding from tokenised report.

        Args:
            input_ids:       [batch, seq_len]
            attention_mask:  [batch, seq_len]
        Returns:
            text_emb: [batch, text_dim]  (CLS token)
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state[:, 0, :]  # [batch, 768]

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> dict:
        """
        Full forward pass: image + text → pathology predictions.

        Args:
            pixel_values:     [batch, 3, 224, 224]
            input_ids:        [batch, seq_len]
            attention_mask:   [batch, seq_len]
            labels:           [batch, 14]  (optional, for loss computation)
            return_embeddings: if True, return intermediate embeddings too

        Returns:
            dict with keys: logits, loss (if labels given),
                            visual_emb, text_emb, projected_emb (if return_embeddings)
        """

        # 1. Encode image → visual embedding
        visual_emb = self.encode_image(pixel_values)          # [batch, 768]

        # 2. Project visual into text space (CROSS-MODAL BINDING)
        projected_emb = self.projection(visual_emb)           # [batch, 768]

        # 3. Encode text → text embedding
        text_emb = self.encode_text(input_ids, attention_mask) # [batch, 768]

        # 4. Fuse: concatenate projected visual and text
        fused = torch.cat([projected_emb, text_emb], dim=-1)  # [batch, 1536]

        # 5. Classify
        logits = self.classifier(fused)                        # [batch, 14]

        # 6. Compute loss if labels provided
        loss = None
        if labels is not None:
            # Multi-label binary cross-entropy (each of 14 labels is independent)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        result = {"logits": logits, "loss": loss}

        if return_embeddings:
            result.update({
                "visual_emb": visual_emb,
                "projected_emb": projected_emb,
                "text_emb": text_emb,
            })

        return result