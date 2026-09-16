# ==============================================================================
# fig_forest_plot_subelements.py
# ------------------------------------------------------------------------------
# Generates Figure 1: Impact of AI Access Across Detailed Quality Subelements.
#
# Figure Architecture:
#   Forest plot displaying point estimates and 95% confidence intervals for
#   treatment effects across Pooled Average Score, Enforceability, Technical
#   Accuracy, Strategic Ambiguity, Completeness & Alignment, and Clarity for:
#     - 10-Day Drafting (Panel A)
#     - 90-Day Drafting (Panel B)
#     - 90-Day Redlining (Panel C)
#   Disaggregated by All Participants, Juniors (<7 yrs), and Seniors (>=7 yrs).
#
# Inputs:
#   - Jsons/fig_forest.json
#
# Outputs:
#   - New/forest_plot_subelements.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
from config import GITHUB_CONFIG
from utils import upload_plot

def render(github_pat=None):
    """
    Renders Figure 1 (Forest plot of quality subelements) to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_forest.json', 'r') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    styles = {
        'all': {'color': '#333333', 'fmt': 's', 'offset': 0.25, 'label': 'All (Included)', 'lw': 2.5, 'ms': 7},
        'jun': {'color': '#1f77b4', 'fmt': '^', 'offset': 0.0, 'label': 'Juniors (<7 yrs)', 'lw': 1.5, 'ms': 7},
        'sen': {'color': '#ff7f0e', 'fmt': 'D', 'offset': -0.25, 'label': 'Seniors (≥7 yrs)', 'lw': 1.5, 'ms': 6}
    }
    
    for panel_idx, (ax, (title, items)) in enumerate(zip(axes, data.items())):
        y_pos = []
        labels = []
        
        for idx, item in enumerate(items):
            var_name = item['label']
            if idx == 0 or 'sum_rat' in str(item.get('var', '')) or 'rater summary' in str(var_name).lower():
                var_name = 'Pooled Average Score'
            else:
                subelement_map = {
                    'enf': 'Enforceability',
                    'tec': 'Technical Accuracy',
                    'str': 'Strategic Ambiguity',
                    'com': 'Completeness and Alignment',
                    'cla': 'Clarity'
                }
                for suffix, actual_name in subelement_map.items():
                    if str(item.get('var', '')).endswith(f'_{suffix}') or str(var_name).endswith(f'_{suffix}'):
                        var_name = actual_name
                        break
            y_base = len(items) - idx
            
            for key in ['all', 'jun', 'sen']:
                eff = item[key]
                st = styles[key]
                c = eff.get('c')
                if not isinstance(c, (int, float)) or isinstance(c, str):
                    continue
                
                ci_l = eff.get('ci_l', c)
                ci_u = eff.get('ci_u', c)
                stars = eff.get('s', '')
                
                y_coord = y_base + st['offset']
                ax.errorbar(c, y_coord, 
                            xerr=[[c - ci_l], [ci_u - c]],
                            fmt=st['fmt'], color=st['color'], label=st['label'] if (idx == 0 and panel_idx == 0) else "",
                            elinewidth=st['lw'], capsize=4, markersize=st['ms'])
                
                ann_text = f"{c:.2f}{stars}"
                ax.text(c, y_coord + 0.08, ann_text, fontsize=8, ha='center', va='bottom', color=st['color'])
            
            y_pos.append(y_base)
            labels.append(var_name)
            
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_yticks(y_pos)
        if panel_idx == 0:
            ax.set_yticklabels(labels, fontsize=10)
            ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.22), ncol=3, fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Treatment Effect (Std. Dev.)")
        ax.grid(axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    upload_plot(fig, 'forest_plot_subelements.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
