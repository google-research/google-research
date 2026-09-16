# ==============================================================================
# fig_sensitivity.py
# ------------------------------------------------------------------------------
# Generates Figure A3: Sensitivity of Main Results to Seniority Experience Cutoffs.
#
# Figure Architecture:
#   Plots OLS point estimates and 95% confidence intervals across alternative
#   experience threshold triplets (<3/3-6/>=7, <4/4-7/>=8, <5/5-8/>=9) for
#   Juniors (blue), Midlevels (green), and Seniors (orange) across 10-day drafting,
#   90-day drafting, and 90-day redlining exercises.
#
# Inputs:
#   - Jsons/fig_sensitivity.json
#
# Outputs:
#   - New/sensitivity.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
import numpy as np
from config import GITHUB_CONFIG
from utils import upload_plot

def get_stars_sens(p):
    """Formats significance stars for sensitivity plot annotations."""
    if p is None or np.isnan(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def render(github_pat=None):
    """
    Renders Figure A3 (Experience threshold sensitivity) to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_sensitivity.json', 'r') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    group_labels_3 = [('juniors', 'Juniors'), ('midlevels', 'Midlevels'), ('seniors', 'Seniors')]
    
    panel_titles = [
        "Drafting: 10-day task",
        "Drafting: 90-day task",
        "Redlining: 90-day task"
    ]

    for i, (group_key, group_title) in enumerate(group_labels_3):
        ax = axes[i]
        current_y = 0
        y_ticks = []
        y_labels = []

        for out_title in reversed(panel_titles):
            if out_title not in data:
                continue
            pts = data[out_title][group_key]

            for d in reversed(pts):
                y_ticks.append(current_y)
                y_labels.append(f"<{d['x_jun']} / {d['x_jun']}-{d['x_sen']-1} / >={d['x_sen']}")

                y_val = d.get('y')
                if y_val is not None and not np.isnan(y_val):
                    err_low = y_val - d['ci_lower']
                    err_high = d['ci_upper'] - y_val
                    if group_key == 'juniors':
                        color = '#1f77b4'
                        marker = 's'
                    elif group_key == 'seniors':
                        color = '#ff7f0e'
                        marker = 'o'
                    else:
                        color = '#2ca02c'
                        marker = '^'

                    ax.errorbar(y_val, current_y, xerr=[[err_low], [err_high]], fmt=marker, color=color, ecolor=color, capsize=4, elinewidth=2)

                    txt = get_stars_sens(d.get('p'))
                    if txt:
                        ax.text(y_val, current_y + 0.25, txt, ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
                current_y += 1

            y_ticks.append(current_y)
            y_labels.append(out_title)
            current_y += 1.5

        ax.set_title(group_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Treatment Effect (Weighted OLS Coeff)', fontsize=12)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(y_ticks)

        if i == 0:
            ax.set_yticklabels(y_labels, fontsize=10)
            for tick in ax.get_yticklabels():
                if '<' not in tick.get_text() and '>=' not in tick.get_text() and '/' not in tick.get_text():
                    tick.set_fontweight('bold')
                    tick.set_fontsize(12)
        else:
            ax.tick_params(labelleft=False)

        ax.grid(axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    upload_plot(fig, 'sensitivity.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
