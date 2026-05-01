"""
notebooks/lora_visualisation.py

Visualises what LoRA adapters do to the weight matrices.
Run this to develop intuition before Step 4.
"""

import torch
import matplotlib
matplotlib.use('Agg')   # no display needed — saves to file
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lora_layer import LoRALinear

print("Generating LoRA visualisation...")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("LoRA: What the Adapter Adds to the Base Weight",
             fontsize=13, fontweight='bold')

# Create a LoRALinear layer
layer = LoRALinear(in_features=64, out_features=64, rank=4, lora_alpha=8)

# Base weight (frozen)
W = layer.weight.data.numpy()

# Simulate a "trained" LoRA by filling A and B with small values
with torch.no_grad():
    layer.lora_A.data = torch.randn(4, 64) * 0.1
    layer.lora_B.data = torch.randn(64, 4) * 0.1

# LoRA residual matrix = B @ A * scaling
BA = (layer.lora_B.data @ layer.lora_A.data * layer.scaling).numpy()

axes[0].imshow(W, cmap='RdBu', vmin=-0.5, vmax=0.5)
axes[0].set_title(f"Base W\n({W.shape[0]}×{W.shape[1]})\nFROZEN")
axes[0].axis('off')

axes[1].imshow(layer.lora_A.data.numpy(), cmap='RdBu', vmin=-0.3, vmax=0.3)
axes[1].set_title(f"LoRA A\n({layer.lora_A.shape[0]}×{layer.lora_A.shape[1]})\ntrainable")
axes[1].axis('off')

axes[2].imshow(layer.lora_B.data.numpy(), cmap='RdBu', vmin=-0.3, vmax=0.3)
axes[2].set_title(f"LoRA B\n({layer.lora_B.shape[0]}×{layer.lora_B.shape[1]})\ntrainable")
axes[2].axis('off')

axes[3].imshow(BA, cmap='RdBu', vmin=-0.05, vmax=0.05)
axes[3].set_title(f"B·A (residual)\n({BA.shape[0]}×{BA.shape[1]})\nwhat gets subtracted\nduring unlearning")
axes[3].axis('off')

plt.tight_layout()
plt.savefig("logs/lora_visualisation.png", dpi=120, bbox_inches='tight')
print("Saved to logs/lora_visualisation.png")

# Print parameter counts
total_W = W.size
total_BA = layer.lora_A.numel() + layer.lora_B.numel()
print(f"\nParameter comparison:")
print(f"  Base W:       {total_W:,} params (frozen)")
print(f"  LoRA A+B:     {total_BA:,} params (trainable)")
print(f"  Reduction:    {100*(1 - total_BA/total_W):.1f}%")
print(f"  Rank used:    {layer.rank}")
print(f"  Scaling:      {layer.scaling:.3f} (= alpha/rank = {layer.lora_alpha}/{layer.rank})")