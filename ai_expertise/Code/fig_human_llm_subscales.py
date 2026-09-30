# ==============================================================================
# fig_human_llm_subscales.py
# ------------------------------------------------------------------------------
# Generates Figure: Subscale Concordance Across Human and LLM Evaluators.
#
# Figure Architecture:
#   A 5x3 multi-panel scatter plot grid examining Human vs LLM concordance across
#   each of the 5 quality subdimensions (rows: Enforceability, Technical Accuracy,
#   Strategic Ambiguity, Completeness & Alignment, Clarity) and 3 experimental
#   tasks (columns: 10-day drafting, 90-day drafting, 90-day redlining).
#
# Inputs:
#   - Jsons/fig_human_llm_subscales.json
#
# Outputs:
#   - New/human_llm_subscales.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
import numpy as np
from config import GITHUB_CONFIG
from utils import upload_plot

def render(github_pat=None):
    """
    Renders 5x3 Human vs LLM subscale concordance grid to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    
    try:
        with open('Jsons/fig_human_llm_subscales.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        with open('Jsons/fig_raters.json', 'r') as f:
            data = json.load(f).get('human_llm_subscales', {})
            
    if not data:
        print("No human_llm_subscales data found in Jsons/.")
        return

    dims = ['enf', 'tec', 'str', 'com', 'cla']
    tasks = ['tt1_rat_drades', 'tt2_rat_drades', 'tt2_rat_cri']
    
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 18), sharex=True, sharey=True)

    np.random.seed(42)

    for row_idx, dim in enumerate(dims):
        sub_data = data.get(dim, {})
        dim_label = sub_data.get('label', dim)
        tasks_data = sub_data.get('tasks', {})
        
        for col_idx, task_prefix in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            t_data = tasks_data.get(task_prefix, {})
            task_label = t_data.get('label', task_prefix)
            
            x = np.array(t_data.get('x', []))
            y = np.array(t_data.get('y', []))
            r = t_data.get('r', 0.0)
            p = t_data.get('p', 1.0)
            n = t_data.get('n', len(x))
            
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
                ax.text(0.05, 0.95, text_box, transform=ax.transAxes, fontsize=9,
                        va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

            ax.set_xlim(0.8, 5.2)
            ax.set_ylim(0.8, 5.2)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.grid(True, linestyle=':', alpha=0.6)

            if row_idx == 0:
                ax.set_title(task_label, fontsize=13, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel(f"{dim_label}\nLLM Score", fontsize=11, fontweight='bold')

            if row_idx == 4:
                ax.set_xlabel("Human Raters (Average Score)", fontsize=11)

    plt.tight_layout()
    upload_plot(fig, 'human_llm_subscales.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
