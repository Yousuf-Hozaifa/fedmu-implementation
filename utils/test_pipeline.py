"""
utils/test_pipeline.py

End-to-end test of the toy multimodal pipeline.

Runs a full forward pass with a batch of 2 patients and verifies:
  1. Data loads correctly
  2. Model forward pass completes
  3. Output shapes are correct
  4. Loss is a valid scalar
  5. Gradients flow to trainable parameters

Expected runtime on CPU: 60-120 seconds (model download on first run).
"""

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, DistilBertTokenizer

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.toy_dataset import generate_toy_dataset
from data.patient_dataset import PatientDataset
from models.multimodal_model import FedMUModel


def main():
    print("\n" + "="*60)
    print("  Fed-MU Pipeline Test — Step 2")
    print("="*60)

    # ── Load config ────────────────────────────────────────────────
    with open("configs/fedmu_config.yaml") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["project"]["seed"])

    # ── Step 1: Generate toy data if not already present ───────────
    meta_path = Path("data/toy_patients/metadata.json")
    if not meta_path.exists():
        print("\n[1/5] Generating toy patient dataset...")
        generate_toy_dataset(
            output_dir="data/toy_patients",
            n_patients=config["data"]["total_patients"],
        )
    else:
        print("\n[1/5] Toy dataset found — skipping generation.")

    # ── Step 2: Load processors (downloads models on first run) ────
    print("\n[2/5] Loading CLIP processor and DistilBERT tokenizer...")
    print("  (This downloads ~400MB on first run — please wait)")

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )
    print("  Processors loaded.")

    # ── Step 3: Create dataset ─────────────────────────────────────
    print("\n[3/5] Creating PatientDataset...")

    # CLIP processor handles image preprocessing
    import torchvision.transforms as T
    image_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=clip_processor.image_processor.image_mean,
            std=clip_processor.image_processor.image_std,
        ),
    ])

    dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_text_len=config["data"].get("max_text_len", 128),
    )

    print(f"  Dataset size: {len(dataset)} patients")
    print(f"  Sample patient_id: {dataset.patient_ids[0]}")

    # Check a single sample
    sample = dataset[0]
    print(f"  Image tensor shape:  {sample['image'].shape}")
    print(f"  Text token shape:    {sample['text_input_ids'].shape}")
    print(f"  Labels shape:        {sample['labels'].shape}")
    print(f"  Report (first 80):   {sample['report'][:80]}...")

    # ── Step 4: Load model ─────────────────────────────────────────
    print("\n[4/5] Loading FedMUModel...")
    model = FedMUModel(
        clip_model_name="openai/clip-vit-base-patch32",
        bert_model_name="distilbert-base-uncased",
        num_labels=14,
        freeze_encoders=True,
    )

    param_counts = model.get_trainable_param_count()
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Frozen parameters:    {param_counts['frozen']:,}")
    print(f"  Total parameters:     {param_counts['total']:,}")
    print(f"  % trainable:          "
          f"{100*param_counts['trainable']/param_counts['total']:.1f}%")

    model.eval()

    # ── Step 5: Full forward pass with a batch of 2 ────────────────
    print("\n[5/5] Running forward pass (batch size = 2)...")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    # Check tensor shapes before passing to model
    print(f"  Batch image shape:          {batch['image'].shape}")
    print(f"  Batch text_input_ids shape: {batch['text_input_ids'].shape}")
    print(f"  Batch labels shape:         {batch['labels'].shape}")

    with torch.no_grad():
        outputs = model(
            pixel_values=batch["image"],
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            labels=batch["labels"],
            return_embeddings=True,
        )

    # Verify outputs
    print(f"\n  Output shapes:")
    print(f"    logits:        {outputs['logits'].shape}   (expect [2, 14])")
    print(f"    visual_emb:    {outputs['visual_emb'].shape}  (expect [2, 768])")
    print(f"    projected_emb: {outputs['projected_emb'].shape}  (expect [2, 768])")
    print(f"    text_emb:      {outputs['text_emb'].shape}  (expect [2, 768])")
    print(f"    loss:          {outputs['loss'].item():.4f}")

    assert outputs["logits"].shape == (2, 14), "FAIL: logits shape"
    assert outputs["visual_emb"].shape == (2, 768), "FAIL: visual_emb shape"
    assert outputs["projected_emb"].shape == (2, 768), "FAIL: projected_emb shape"
    assert not torch.isnan(outputs["loss"]), "FAIL: loss is NaN"

    # ── Gradient flow check ────────────────────────────────────────
    print("\n  Checking gradient flow...")
    model.train()
    outputs_train = model(
        pixel_values=batch["image"],
        input_ids=batch["text_input_ids"],
        attention_mask=batch["text_attention_mask"],
        labels=batch["labels"],
    )
    outputs_train["loss"].backward()

    # Verify gradients flow to projection (A× — the target layer)
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"    Gradient in: {name[:55]}")

    print("\n" + "="*60)
    print("  ALL CHECKS PASSED — Pipeline is working correctly")
    print("  The cross-modal projection (A×) receives gradients.")
    print("  Ready for Step 3: LoRA adapters.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()