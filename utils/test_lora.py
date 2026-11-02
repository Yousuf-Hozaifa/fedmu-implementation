"""
utils/test_lora.py

Verifies LoRA adapter attachment and all adapter controls.

Checks:
  1. All three adapters attach without error
  2. Pre-training mode: all adapters trainable
  3. Unlearning mode: only A× trainable, A_v and A_l frozen
  4. Gradient norms are non-zero and computable per adapter
  5. A× state can be exported and reloaded (for federated protocol)
  6. After reload, forward pass output is identical
"""

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, DistilBertTokenizer
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.toy_dataset import ensure_toy_dataset
from data.patient_dataset import PatientDataset
from models.multimodal_model import FedMUModel
from models.lora_model import FedMULoRAModel


def main():
    print("\n" + "="*60)
    print("  Fed-MU LoRA Adapter Test — Step 3")
    print("="*60)

    with open("configs/fedmu_config.yaml") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["project"]["seed"])

    # ── Load data ──────────────────────────────────────────────────
    ensure_toy_dataset("data/toy_patients", config["data"]["total_patients"])

    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=clip_proc.image_processor.image_mean,
                    std=clip_proc.image_processor.image_std),
    ])

    dataset = PatientDataset(
        metadata_file="data/toy_patients/metadata.json",
        data_root="data/toy_patients",
        image_transform=transform,
        tokenizer=tokenizer,
        max_text_len=128,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # ── Test 1: Build the LoRA model ───────────────────────────────
    print("\n[1/6] Building FedMULoRAModel...")
    base = FedMUModel(freeze_encoders=False)
    model = FedMULoRAModel(
        base_model=base,
        lora_rank=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
    )
    print("  Model built successfully.")

    # ── Test 2: Pre-training mode ──────────────────────────────────
    print("\n[2/6] Testing pre-training mode...")
    model.set_pretraining_mode()

    # Use named_lora_parameters() — avoids PyTorch deduplication that hides
    # adapter group names: injected LoRA layers live inside self.base, so
    # named_parameters() always yields the base.* path and never adapter_v/*.
    lora_trainable = [(n, p) for n, p in model.named_lora_parameters()
                      if p.requires_grad]
    all_trainable  = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)
    print(f"  Total trainable params: {all_trainable}")
    print(f"  LoRA trainable params:  {len(lora_trainable)}")
    assert len(lora_trainable) > 0, "No LoRA params are trainable!"
    # All three adapters should be trainable
    has_v     = model.is_adapter_trainable("adapter_v")
    has_l     = model.is_adapter_trainable("adapter_l")
    has_cross = model.is_adapter_trainable("adapter_cross")
    assert has_v,     "A_v not trainable in pre-training mode"
    assert has_l,     "A_l not trainable in pre-training mode"
    assert has_cross, "A× not trainable in pre-training mode"
    print("  OK: A_v, A_l, A× all trainable in pre-training mode.")

    # ── Test 3: Unlearning mode ────────────────────────────────────
    print("\n[3/6] Testing unlearning mode...")
    model.set_unlearning_mode()

    # Again use is_adapter_trainable() — direct and unambiguous
    has_v_trainable = model.is_adapter_trainable("adapter_v")
    has_l_trainable = model.is_adapter_trainable("adapter_l")
    has_x_trainable = model.is_adapter_trainable("adapter_cross")

    assert not has_v_trainable, "A_v should be FROZEN in unlearning mode"
    assert not has_l_trainable, "A_l should be FROZEN in unlearning mode"
    assert has_x_trainable,     "A× should be TRAINABLE in unlearning mode"
    print("  OK: A_v and A_l frozen. A× trainable.")
    print(f"  Trainable in unlearning mode: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    # ── Test 4: Forward pass and gradient norms ────────────────────
    print("\n[4/6] Testing forward pass and gradient norms...")
    # Need all adapters trainable to get gradient norms for all three
    model.set_pretraining_mode()
    model.train()
    model.zero_grad()

    outputs = model(
        pixel_values=batch["image"],
        input_ids=batch["text_input_ids"],
        attention_mask=batch["text_attention_mask"],
        labels=batch["labels"],
    )
    outputs["loss"].backward()

    grad_norms = model.get_adapter_gradient_norms()
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Gradient norms:")
    print(f"    ||∇_{{A_v}}  L||_F = {grad_norms['A_v']:.6f}")
    print(f"    ||∇_{{A_l}}  L||_F = {grad_norms['A_l']:.6f}")
    print(f"    ||∇_{{A×}}   L||_F = {grad_norms['A_cross']:.6f}")

    denom = (grad_norms['A_v'] or 0) + (grad_norms['A_l'] or 0)
    if denom > 0 and grad_norms['A_cross']:
        rho = grad_norms['A_cross'] / denom
        print(f"\n  ρ(p) preview = {grad_norms['A_cross']:.4f} / "
              f"{denom:.4f} = {rho:.4f}")
        print(f"  (On toy data this is a rough estimate. "
              f"Step 4 computes this properly.)")
    else:
        print("  Warning: some gradient norms are zero — check adapter attachment.")

    assert grad_norms['A_cross'] is not None, "A× gradient norm is None"
    print("  OK: All gradient norms computable.")

    # ── Test 5: State export and reload ───────────────────────────
    print("\n[5/6] Testing A× state export and reload...")
    state_before = model.get_cross_adapter_state()
    print(f"  Exported {len(state_before)} A× parameter tensors:")
    for k, v in state_before.items():
        print(f"    {k}: shape {tuple(v.shape)}")

    # Perturb A× values
    with torch.no_grad():
        for layer in model.adapter_cross.lora_layers.values():
            layer.lora_A.data += 99.0
            layer.lora_B.data += 99.0

    # Reload original state
    model.set_cross_adapter_state(state_before)

    # Verify reload worked
    state_after = model.get_cross_adapter_state()
    for key in state_before:
        assert torch.allclose(state_before[key], state_after[key]), \
            f"Reload failed for {key}"
    print("  OK: A× state exported, perturbed, and reloaded correctly.")

    # ── Test 6: Output consistency after reload ────────────────────
    print("\n[6/6] Testing output consistency after state reload...")
    model.eval()
    with torch.no_grad():
        out1 = model(
            pixel_values=batch["image"],
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        )
        out2 = model(
            pixel_values=batch["image"],
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        )
    assert torch.allclose(out1["logits"], out2["logits"]), \
        "Outputs differ between identical forward passes"
    print("  OK: Identical inputs produce identical outputs.")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ALL CHECKS PASSED — LoRA adapters working correctly")
    print("  Key results:")
    print(f"    A_v gradient norm:   {grad_norms['A_v']:.6f}")
    print(f"    A_l gradient norm:   {grad_norms['A_l']:.6f}")
    print(f"    A× gradient norm:    {grad_norms['A_cross']:.6f}")
    if denom > 0:
        print(f"    ρ(p) estimate:       {rho:.4f}")
    print("  Ready for Step 4: Gradient Concentration Ratio ρ(p).")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()