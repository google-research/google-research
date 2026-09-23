# ==============================================================================
# fig_sensitivity.py
# ------------------------------------------------------------------------------
# Generates Figure A3: Sensitivity of Main Results to Seniority Experience Cutoffs.
#
# Figure Architecture:
#   Plots OLS point estimates and 95% confidence intervals across alternative
#   experience thresholds (Exp < 3, 5, 7, 9, 11) for Juniors (blue squares) and
#   Seniors (orange circles) across 10-day drafting, 90-day drafting, and
#   90-day redlining exercises.
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

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5), sharey=True)
    group_labels = [('juniors', 'Juniors'), ('seniors', 'Seniors')]
    
    panel_titles = [
        "Drafting: 10-day task",
        "Drafting: 90-day task",
        "Redlining: 90-day task"
    ]

    for i, (group_key, group_title) in enumerate(group_labels):
        ax = axes[i]
        current_y = 0
        y_ticks = []
        y_labels = []
        exp_rows_y = []

        for out_title in reversed(panel_titles):
            if out_title not in data:
                continue
            pts = data[out_title][group_key]

            for d in reversed(pts):
                y_ticks.append(current_y)
                x_cut = d.get('x', d.get('x_jun'))
                y_labels.append(f"Exp < {x_cut}")
                exp_rows_y.append(current_y)

                y_val = d.get('y')
                if y_val is not None and not np.isnan(y_val):
                    err_low = y_val - d['ci_lower']
                    err_high = d['ci_upper'] - y_val
                    if group_key == 'juniors':
                        color = '#1f77b4'
                        marker = 's'
                    else:
                        color = '#ff7f0e'
                        marker = 'o'

                    ax.errorbar(y_val, current_y, xerr=[[err_low], [err_high]], fmt=marker, color=color, ecolor=color, capsize=4, elinewidth=2, zorder=3)

                    txt = get_stars_sens(d.get('p'))
                    if txt:
                        ax.text(y_val, current_y + 0.25, txt, ha='center', va='bottom', fontsize=10, fontweight='bold', color=color, zorder=4)
                current_y += 1

            y_ticks.append(current_y)
            y_labels.append(out_title)
            current_y += 1.5

        for ry in exp_rows_y:
            ax.axhline(ry, color='#dddddd', linestyle='-', linewidth=0.8, zorder=1)

        ax.set_title(group_title, fontsize=14, fontweight='bold', color='#333333')
        ax.set_xlabel('Treatment Effect (Weighted OLS Coeff)', fontsize=11, color='#333333')
        ax.axvline(0, color='black', linestyle='--', linewidth=1, zorder=2)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', length=0)

        if i == 0:
            ax.set_yticklabels(y_labels, fontsize=10, color='#333333')
            for tick in ax.get_yticklabels():
                if 'Exp <' not in tick.get_text():
                    tick.set_fontweight('bold')
                    tick.set_fontsize(11.5)
        else:
            ax.tick_params(labelleft=False)

        ax.grid(axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    upload_plot(fig, 'sensitivity.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
