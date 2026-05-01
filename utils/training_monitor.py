"""
utils/training_monitor.py

Tracks and visualises training progress across federated rounds.
Saves loss curves to logs/.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


class TrainingMonitor:
    """Records and plots training metrics across federated rounds."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.history: List[Dict] = []

    def record(self, round_record: Dict):
        self.history.append(round_record)
        self._save()

    def _save(self):
        with open(self.log_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def plot(self, save_path: str = None):
        if not self.history:
            return

        save_path = save_path or str(self.log_dir / "training_curves.png")
        rounds      = [h["round"]      for h in self.history]
        train_loss  = [h["train_loss"] for h in self.history]
        val_loss    = [h["val_loss"]   for h in self.history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, train_loss, "o-", color="steelblue",
                label="Train loss", linewidth=2, markersize=7)
        ax.plot(rounds, val_loss, "s--", color="coral",
                label="Val loss",   linewidth=2, markersize=7)

        ax.set_xlabel("Federated Round", fontsize=11)
        ax.set_ylabel("Loss",            fontsize=11)
        ax.set_title("Federated Pre-Training: Loss per Round",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks(rounds)

        # Annotate final values
        ax.annotate(f"{train_loss[-1]:.4f}",
                    xy=(rounds[-1], train_loss[-1]),
                    xytext=(5, -15), textcoords="offset points",
                    fontsize=9, color="steelblue")
        ax.annotate(f"{val_loss[-1]:.4f}",
                    xy=(rounds[-1], val_loss[-1]),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=9, color="coral")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Training curve saved: {save_path}")

    def print_summary(self):
        if not self.history:
            print("  No training history.")
            return
        first = self.history[0]
        last  = self.history[-1]
        print(f"\n  Training summary ({len(self.history)} rounds):")
        print(f"    Initial loss:  {first['train_loss']:.4f}")
        print(f"    Final loss:    {last['train_loss']:.4f}")
        reduction = (first['train_loss'] - last['train_loss'])
        print(f"    Reduction:     {reduction:.4f} "
              f"({'improved' if reduction > 0 else 'diverged'})")