# ==============================================================================
# fig_rater_human_vs_llm.py
# ------------------------------------------------------------------------------
# Generates Figure A2: Concordance Between Human Expert and LLM Evaluators.
#
# Figure Architecture:
#   Scatter plots comparing average human expert scores vs LLM evaluator scores
#   across 10-day drafting, 90-day drafting, and 90-day redlining, with 45-degree
#   reference lines and Pearson correlation coefficients.
#
# Inputs:
#   - Jsons/fig_raters.json
#
# Outputs:
#   - New/rater_human_vs_llm.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
import numpy as np
from config import GITHUB_CONFIG
from utils import upload_plot

def render(github_pat=None):
    """
    Renders Figure A2 (Human vs LLM rater concordance) to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_raters.json', 'r') as f:
        data = json.load(f)
    
    hl_data = data.get('human_vs_llm', {})
    if not hl_data:
        return
        
    tasks = ['tt1_sum_rat_drades', 'tt2_sum_rat_drades', 'tt2_sum_rat_cri']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    np.random.seed(42)
    for ax, var_key in zip(axes, tasks):
        item = hl_data.get(var_key, {})
        x = np.array(item.get('x', []))
        y = np.array(item.get('y', []))
        label = item.get('label', var_key)
        r = item.get('r', 0.0)
        p = item.get('p', 1.0)
        n = item.get('n', len(x))
        
        if len(x) > 0 and len(y) > 0:
            x_jitter = x + np.random.normal(0, 0.08, size=len(x))
            y_jitter = y + np.random.normal(0, 0.08, size=len(y))
            
            ax.scatter(x_jitter, y_jitter, alpha=0.6, s=40, color='#ff7f0e', edgecolors='none')
            ax.plot([1, 5], [1, 5], 'k--', alpha=0.5, label='45° line')
            
            if p < 0.001:
                p_str = "$p$ < 0.001"
                r_str = f"Pearson r = {r:.2f}***"
            elif p < 0.01:
                p_str = f"$p$ = {p:.3f}"
                r_str = f"Pearson r = {r:.2f}***"
            elif p < 0.05:
                p_str = f"$p$ = {p:.3f}"
                r_str = f"Pearson r = {r:.2f}**"
            elif p < 0.10:
                p_str = f"$p$ = {p:.3f}"
                r_str = f"Pearson r = {r:.2f}*"
            else:
                p_str = f"$p$ = {p:.3f}"
                r_str = f"Pearson r = {r:.2f}"
                
            text_box = f"{r_str}\n{p_str}\n$N$ = {n}"
            ax.text(0.05, 0.95, text_box, transform=ax.transAxes, fontsize=10,
                    va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                    
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel("Human Raters (Average Score)", fontsize=11)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if ax == axes[0]:
            ax.set_ylabel("LLM Rater Score", fontsize=11)
            
    plt.tight_layout()
    upload_plot(fig, 'rater_human_vs_llm.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
