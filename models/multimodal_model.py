"""
models/multimodal_model.py
Fed-MU backbone.

Kaggle / IU mode  (use_llava=False, default for T4):
  vision: CLIP ViT-L/14-336  (matches LLaVA-1.5's vision tower)
  language: LLaVA-1.5-7B language head via HF pipeline
  projection: 2-layer MLP matching LLaVA-1.5 dimensions

Full LLaVA mode  (use_llava=True, requires A100 40GB):
  Loads the full LLaVA-1.5-7B model with 4-bit quantisation.
"""

import yaml
from transformers import (CLIPVisionModel, CLIPImageProcessor,
                          LlavaForConditionalGeneration, AutoTokenizer,
                          BitsAndBytesConfig)
import torch
import torch.nn as nn


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
            import torch
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
            print(f"[Model] Loading lightweight proxy (ViT-L/14-336 + MLP) ...")
            from transformers import CLIPVisionModel, CLIPImageProcessor, \
                                     DistilBertModel, DistilBertTokenizer

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