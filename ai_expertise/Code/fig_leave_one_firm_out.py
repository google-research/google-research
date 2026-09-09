# ==============================================================================
# fig_leave_one_firm_out.py
# ------------------------------------------------------------------------------
# Generates Leave-One-Out Robustness Forest Plot for 90-Day Redlining Task.
#
# Figure Architecture:
#   Forest plot displaying point estimates and 95% confidence intervals for
#   treatment effects on the 90-day redlining task when sequentially removing
#   each individual firm from the estimation sample.
#   Includes:
#     - Full Sample benchmark at the top (no firm removals)
#     - Individual firm exclusions (Firms 1, 2, 5, 6, 8, 9, 10, 11, 12)
#     - Clustered estimates for All Participants, Juniors (<7 yrs), and Seniors (>=7 yrs)
#
# Inputs:
#   - Jsons/fig_leave_one_firm_out.json
#
# Outputs:
#   - New/forest_plot_leave_one_out.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
from config import GITHUB_CONFIG
from utils import upload_plot

def render(github_pat=None):
    """
    Renders the leave-one-firm-out forest plot for the 90-day redlining task to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_leave_one_firm_out.json', 'r') as f:
        data = json.load(f)

    # Filter out firms 3, 4, and 7
    data = [item for item in data if item.get('label') not in ['Firm 3', 'Firm 4', 'Firm 7']]

    fig, ax = plt.subplots(figsize=(10, 8.5))

    styles = {
        'all': {'color': '#333333', 'fmt': 's', 'offset': 0.24, 'label': 'All (Included)', 'lw': 2.2, 'ms': 6.5},
        'jun': {'color': '#1f77b4', 'fmt': '^', 'offset': 0.0, 'label': 'Juniors (<7 yrs)', 'lw': 1.6, 'ms': 6.5},
        'sen': {'color': '#ff7f0e', 'fmt': 'D', 'offset': -0.24, 'label': 'Seniors (≥7 yrs)', 'lw': 1.6, 'ms': 6.0}
    }

    y_pos = []
    labels = []
    N = len(data)

    for idx, item in enumerate(data):
        y_base = N - idx
        y_pos.append(y_base)
        labels.append(item['label'])
        
        for key in ['all', 'jun', 'sen']:
            eff = item['estimates'][key]
            st = styles[key]
            c = eff.get('c')
            if c is None or isinstance(c, str):
                continue
            
            ci_l = eff.get('ci_l', c)
            ci_u = eff.get('ci_u', c)
            stars = eff.get('s', '')
            
            y_coord = y_base + st['offset']
            ax.errorbar(c, y_coord, 
                        xerr=[[c - ci_l], [ci_u - c]],
                        fmt=st['fmt'], color=st['color'], 
                        label=st['label'] if idx == 0 else "",
                        elinewidth=st['lw'], capsize=3.5, markersize=st['ms'])
            
            ann_text = f"{c:.2f}{stars}"
            ax.text(c, y_coord + 0.06, ann_text, fontsize=8, ha='center', va='bottom', color=st['color'])

    # Horizontal separator line between Full sample and Firm 1
    ax.axhline(N - 0.5, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)

    # Vertical line at zero treatment effect
    ax.axvline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    
    # Bold the 'Full sample' tick label
    if len(ax.get_yticklabels()) > 0:
        ax.get_yticklabels()[0].set_fontweight('bold')

    ax.set_xlabel("Treatment Effect on 90-Day Redlining Quality (SD)", fontsize=11, labelpad=8)
    ax.set_ylabel("Excluded Firm", fontsize=11, labelpad=8)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10.5, frameon=True)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    ax.set_ylim(0.3, N + 0.7)
    ax.set_xlim(-0.7, 1.0)

    plt.tight_layout()
    upload_plot(fig, 'forest_plot_leave_one_out.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
