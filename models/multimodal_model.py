"""
models/multimodal_model.py

Two backbones live here:

  * FedMUModel (+ CrossModalProjection): the CPU-friendly model used by the LoRA,
    federated, unlearning and verification code (CLIP ViT-B/32 + DistilBERT).
    models/lora_model.py, experiments/* and unlearning/* all depend on it.
  * MultimodalModel: the larger LLaVA / ViT-L proxy backbone for GPU runs.
    NOTE: it does not yet expose the interface FedMULoRAModel needs
    (projection.layer1/layer2, dict-returning forward).

Fed-MU backbone (MultimodalModel).

Kaggle / IU mode  (use_llava=False, default for T4):
  vision: CLIP ViT-L/14-336  (matches LLaVA-1.5's vision tower)
  language: LLaVA-1.5-7B language head via HF pipeline
  projection: 2-layer MLP matching LLaVA-1.5 dimensions

Full LLaVA mode  (use_llava=True, requires A100 40GB):
  Loads the full LLaVA-1.5-7B model with 4-bit quantisation.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import (AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, DistilBertModel, DistilBertTokenizer,
                          LlavaForConditionalGeneration)


class MultimodalModel(nn.Module):
    def __init__(self, config: dict, use_llava: bool = None):
        super().__init__()

        model_cfg  = config.get('model', {})
        model_name = model_cfg.get('name', 'llava-hf/llava-1.5-7b-hf')
        use_4bit   = model_cfg.get('use_4bit', True)
        vis_name   = model_cfg.get('vision_encoder',
                                   'openai/clip-vit-large-patch14-336')

        # Auto-detect whether to load full LLaVA or lightweight proxy
        if use_llava is None:
            gpu_mem = 0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Full LLaVA needs ≥24 GB GPU RAM
            use_llava = (gpu_mem >= 24.0)
            print(f"[Model] GPU memory: {gpu_mem:.1f} GB → "
                  f"use_llava={use_llava}")

        self.use_llava = use_llava
        self.config    = config

        if use_llava:
            # ── Full LLaVA-1.5-7B (A100 / Colab) ──────────────────────────
            print(f"[Model] Loading full LLaVA-1.5-7B ({model_name}) ...")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ) if use_4bit else None

            self.llava = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(vis_name)

            # Expose sub-modules for LoRA adapter attachment
            self.vision_encoder = self.llava.vision_tower
            self.text_encoder   = self.llava.language_model
            # The MLP projection IS f_× in LLaVA-1.5:
            self.projection     = self.llava.multi_modal_projector
            print("[Model] ✓ LLaVA-1.5-7B loaded")

        else:
            # ── Lightweight proxy (Kaggle T4, CPU) ─────────────────────────
            print("[Model] Loading lightweight proxy (ViT-L/14-336 + MLP) ...")
            self.processor = CLIPImageProcessor.from_pretrained(vis_name)
            self.vision_encoder = CLIPVisionModel.from_pretrained(vis_name)

            # Text encoder
            self.text_encoder = DistilBertModel.from_pretrained(
                "distilbert-base-uncased")
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased")

            # 2-layer MLP projection
            vis_dim  = 1024   # ViT-L/14-336 hidden dim
            text_dim = 768    # DistilBERT hidden dim
            proj_dim = 256    # internal projection dim (toy scale)
            self.projection = nn.Sequential(
                nn.Linear(vis_dim + text_dim, proj_dim * 4),
                nn.GELU(),
                nn.Linear(proj_dim * 4, proj_dim),
            )
            print("[Model] ✓ Lightweight proxy loaded")

        # ── Classification head (14-class pathology) ────────────────────────
        proj_out = 256 if not use_llava else 4096
        self.classifier = nn.Linear(proj_out, 14)

    def forward(self, images, input_ids, attention_mask=None):
        if self.use_llava:
            return self._forward_llava(images, input_ids, attention_mask)
        else:
            return self._forward_proxy(images, input_ids, attention_mask)

    def _forward_proxy(self, images, input_ids, attention_mask=None):
        vis_out  = self.vision_encoder(pixel_values=images)
        vis_feat = vis_out.last_hidden_state[:, 0, :]

        txt_out  = self.text_encoder(input_ids=input_ids,
                                     attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :]

        combined = torch.cat([vis_feat, txt_feat], dim=-1)
        proj     = self.projection(combined)
        logits   = self.classifier(proj)
        return logits

    def _forward_llava(self, images, input_ids, attention_mask=None):
        outputs = self.llava(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.hidden_states[-1][:, 0, :]
        logits = self.classifier(hidden)
        return logits


# ──────────────────────────────────────────────────────────────────────
# FedMUModel: restored from commit ded3b4a (Step 2). Removed by accident in
# the Step 8 rewrite, which broke every module that imports it.
# ──────────────────────────────────────────────────────────────────────
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
