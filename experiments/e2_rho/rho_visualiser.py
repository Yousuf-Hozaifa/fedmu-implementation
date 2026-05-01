"""
experiments/e2_rho/rho_visualiser.py

Generates publication-quality plots for the ρ(p) distribution.
Saves to logs/ directory. Requires matplotlib (already installed).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from typing import Dict, List, Optional


def plot_rho_distribution(
    rho_values: List[float],
    stats: Dict,
    rho_0: float = 1.5,
    save_path: str = "logs/e2_rho_distribution.png",
    title_suffix: str = "",
):
    """
    Generate a 2-panel figure:
      Left:  histogram of ρ(p) with H₀ threshold, ρ₀, and mean lines
      Right: box plot with individual patient values overlaid
    """
    valid = [r for r in rho_values if r is not None
             and not math.isnan(r) and not math.isinf(r)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Experiment E2: Gradient Concentration Ratio ρ(p){title_suffix}",
        fontsize=13, fontweight='bold', y=1.01,
    )

    # ── Left: Histogram ────────────────────────────────────────────
    ax = axes[0]
    n_bins = max(10, len(valid) // 3)
    ax.hist(valid, bins=n_bins, color='steelblue', edgecolor='white',
            alpha=0.8, label='ρ(p) values')

    # Threshold lines
    mean_val = sum(valid) / len(valid)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
               label='H₀ threshold (1.0)')
    ax.axvline(x=rho_0, color='orange', linestyle='--', linewidth=2,
               label=f'Conjectured ρ₀={rho_0}')
    ax.axvline(x=mean_val, color='navy', linestyle='-', linewidth=2,
               label=f'Mean ρ={mean_val:.3f}')

    # CI shading (if stats available)
    if "bootstrap_ci" in stats:
        ci = stats["bootstrap_ci"]
        ax.axvspan(ci["ci_lower"], ci["ci_upper"], alpha=0.15,
                   color='navy', label='95% Bootstrap CI')

    ax.set_xlabel('ρ(p)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of ρ(p)', fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(axis='y', alpha=0.3)

    # Annotation: % above H₀
    pct_above = 100 * sum(r > 1.0 for r in valid) / len(valid)
    ax.text(0.97, 0.97, f'{pct_above:.0f}% > 1.0',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color='red',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.8))

    # ── Right: Box plot + jitter ───────────────────────────────────
    ax2 = axes[1]
    bp = ax2.boxplot(valid, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightsteelblue', color='navy'),
                     medianprops=dict(color='navy', linewidth=2),
                     whiskerprops=dict(color='navy'),
                     capprops=dict(color='navy'),
                     flierprops=dict(marker='o', color='steelblue', alpha=0.4))

    # Jitter individual points
    import random
    rng = random.Random(42)
    jitter = [rng.uniform(-0.15, 0.15) for _ in valid]
    ax2.scatter([1 + j for j in jitter], valid,
                alpha=0.4, s=20, color='steelblue', zorder=3)

    # Threshold lines
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                label='H₀ threshold (1.0)')
    ax2.axhline(y=rho_0, color='orange', linestyle='--', linewidth=2,
                label=f'ρ₀={rho_0}')
    ax2.axhline(y=mean_val, color='navy', linestyle='-', linewidth=1.5,
                label=f'Mean={mean_val:.3f}')

    ax2.set_ylabel('ρ(p)', fontsize=11)
    ax2.set_xticks([])
    ax2.set_title('Box Plot with Individual Values', fontsize=11)
    ax2.legend(fontsize=8.5, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    # Annotate p-value if available
    if "hypothesis_test" in stats:
        ht = stats["hypothesis_test"]
        colour = 'green' if ht['reject_h0'] else 'red'
        verdict = "Reject H₀ ✓" if ht['reject_h0'] else "Fail to Reject H₀"
        ax2.text(0.97, 0.03,
                 f"p={ht['p_value']:.4f}\n{verdict}",
                 transform=ax2.transAxes, ha='right', va='bottom',
                 fontsize=9, color=colour,
                 bbox=dict(boxstyle='round,pad=0.3',
                           fc='white', ec=colour, alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_adapter_norms(
    results: Dict[str, dict],
    save_path: str = "logs/e2_adapter_norms.png",
):
    """
    Bar chart comparing mean gradient norms across all three adapters.
    Shows visually whether A× has higher norms than A_v and A_l.
    """
    valid = [r for r in results.values()
             if r.get("norm_cross") and r.get("norm_v") and r.get("norm_l")]

    if not valid:
        print("  No valid per-adapter norm data for plotting.")
        return

    mean_cross = sum(r["norm_cross"] for r in valid) / len(valid)
    mean_v     = sum(r["norm_v"]     for r in valid) / len(valid)
    mean_l     = sum(r["norm_l"]     for r in valid) / len(valid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment E2: Per-Adapter Gradient Norm Comparison",
                 fontsize=13, fontweight='bold')

    # ── Left: Mean norms per adapter ──────────────────────────────
    adapters = ['A_v\n(Vision)', 'A_l\n(Language)', 'A×\n(Cross-Modal)']
    means = [mean_v, mean_l, mean_cross]
    colours = ['#4CAF50', '#2196F3', '#F44336']

    bars = axes[0].bar(adapters, means, color=colours, edgecolor='white',
                       alpha=0.85, width=0.5)
    axes[0].set_ylabel('Mean ||∇L||_F', fontsize=11)
    axes[0].set_title('Mean Gradient Norm per Adapter', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.0002,
                     f'{val:.5f}', ha='center', va='bottom', fontsize=9)

    # ── Right: Scatter of A× vs (A_v + A_l) per patient ──────────
    denom_norms = [r["norm_v"] + r["norm_l"] for r in valid]
    cross_norms = [r["norm_cross"]            for r in valid]
    rhos = [r["rho"] for r in valid if r.get("rho")]

    sc = axes[1].scatter(denom_norms, cross_norms,
                         c=rhos if rhos else 'steelblue',
                         cmap='RdYlGn', alpha=0.7, s=50,
                         vmin=0.5, vmax=2.5)

    # Identity line: A× norm = A_v + A_l norm  →  ρ(p) = 1
    max_val = max(max(denom_norms), max(cross_norms)) * 1.1
    axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5,
                 label='ρ(p) = 1.0 (H₀ boundary)', alpha=0.7)

    # ρ(p) = 1.5 line
    axes[1].plot([0, max_val], [0, max_val * 1.5], '--',
                 color='orange', linewidth=1.5,
                 label='ρ(p) = 1.5 (ρ₀ threshold)', alpha=0.7)

    axes[1].set_xlabel('||∇_{Av}L||_F + ||∇_{Al}L||_F', fontsize=10)
    axes[1].set_ylabel('||∇_{A×}L||_F', fontsize=10)
    axes[1].set_title('A× vs Unimodal Adapter Norms\n(colour = ρ(p))',
                      fontsize=11)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    if rhos:
        cbar = plt.colorbar(sc, ax=axes[1])
        cbar.set_label('ρ(p)', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")