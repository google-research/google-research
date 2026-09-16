# ==============================================================================
# fig_fisher.py
# ------------------------------------------------------------------------------
# Generates Figure 4: Distribution of Treatment Effects under Fisher Randomization
# Inference (90-Day Redlining Task).
#
# Figure Architecture:
#   Plots empirical permutation null distributions (2,000 Monte Carlo draws) of
#   the estimated treatment effect coefficient vs. the observed point estimate
#   (red dashed line) with exact two-sided p-values across Full Sample, Juniors,
#   and Seniors.
#
# Inputs:
#   - Jsons/fig_fisher.json
#
# Outputs:
#   - New/fisher.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import GITHUB_CONFIG
from utils import upload_plot

def render(github_pat=None):
    """
    Renders Figure 4 (Fisher permutation null distributions) to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_fisher.json', 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    plot_configs = [
        ('full', data.get('full')),
        ('junior', data.get('junior')),
        ('senior', data.get('senior'))
    ]
    
    for ax, (subgroup, dat) in zip(axes, plot_configs):
        if not dat: continue
        
        null_betas = [float(x) for x in dat.get('null_betas', []) if pd.notna(x)]
        true_beta = float(dat.get('true_beta', 0.0))
        title = dat.get('title', subgroup)
        p_val = float(dat.get('p_val', 1.0))
        
        if len(null_betas) > 0:
            n_counts, _, _ = ax.hist(null_betas, bins=35, color="#1f77b4", edgecolor="black", alpha=0.75, density=True, label="Null Distribution")
            line = sns.kdeplot(null_betas, color="#003366", linewidth=2, ax=ax)
            
            x_min = min(min(null_betas), true_beta) - 0.15
            x_max = max(max(null_betas), true_beta) + 0.15
            ax.set_xlim(x_min, x_max)

            kde_y = line.get_lines()[0].get_ydata() if line.get_lines() else []
            max_y = max(max(n_counts) if len(n_counts) > 0 else 0, max(kde_y) if len(kde_y) > 0 else 0)
            if max_y > 0:
                ax.set_ylim(bottom=0, top=max_y * 1.30)
            
        ax.axvline(true_beta, color='#d62728', linestyle='--', linewidth=2.5, label=f'Observed = {true_beta:.2f} ($p$={p_val:.3f})')
        ax.set_title(f"Fisher Permutation: {title}", fontsize=12, fontweight='bold')
        ax.set_xlabel(r"Null Distribution of Treatment Effect ($\beta$)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    upload_plot(fig, 'fisher.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
