# ==============================================================================
# fig_genai_usage_bar.py
# ------------------------------------------------------------------------------
# Generates the bar chart of self-reported generative AI usage outside of InFlow
# in the Month 1, Month 2, and Month 3 monthly surveys (ms1_gai, ms2_gai,
# ms3_gai) for included participants, by treatment arm.
#
# Usage Categories:
#   - "No reported usage": 1 or 2 (< 2.5 for averaged multi-responses)
#   - "Occasional usage":  3      ([2.5, 3.5) for averaged multi-responses)
#   - "Regular usage":     4      (>= 3.5 for averaged multi-responses)
#
# Figure Architecture:
#   - Single panel with bars grouped by time period (Month 1, Month 2, Month 3)
#   - Within each month group, plots the percentage of respondents in each
#     usage category with overlaid semi-transparent bars for Control (blue)
#     and Treatment (orange), offset by 75% of a bar width (25% overlap).
#   - Error bars show 95% normal-approximation (Wald) confidence intervals,
#     p +/- 1.96*sqrt(p(1-p)/n), truncated at 0% for display.
#
# Inputs:
#   - Jsons/fig_genai_usage_bar.json
#
# Outputs:
#   - New/genai_usage_bar.png
# ==============================================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from config import GITHUB_CONFIG
from utils import upload_plot

def render(github_pat=None):
    """
    Renders the GenAI usage bar chart (% of respondents in each usage category by
    month and treatment arm, with 95% confidence intervals) to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_genai_usage_bar.json', 'r') as f:
        data = json.load(f)

    categories = data['categories']
    months = data['months']
    # Two-line tick labels, e.g. "No reported usage" -> "No reported\nusage"
    tick_labels = ['\n'.join(c.rsplit(' ', 1)) for c in categories]

    # Set the plot theme explicitly so the figure does not depend on global style
    # state left behind by previously executed figure scripts.
    sns.set_theme(style='whitegrid')
    plt.rcParams['axes.edgecolor'] = 'black'

    fig, ax = plt.subplots(figsize=(11.5, 6), dpi=300)

    bar_width = 0.54
    # Control and Treatment bars are offset by 75% of a bar width (bar centres
    # 0.405 apart), so 25% of each bar overlaps its partner.
    jitter = 0.75 * bar_width / 2.0
    intra_step = 1.15
    group_gap = 1.05
    n_cats = len(categories)

    x_positions = []
    x_labels = []
    separator_positions = []
    month_annotations = []
    arms = {
        'ctrl': {'pct': [], 'ci_l': [], 'ci_u': []},
        'treat': {'pct': [], 'ci_l': [], 'ci_u': []},
    }

    for m_idx, month in enumerate(months):
        base_x = m_idx * (n_cats * intra_step + group_gap)
        month_xs = [base_x + c_idx * intra_step for c_idx in range(n_cats)]
        x_positions.extend(month_xs)
        x_labels.extend(tick_labels)

        for arm_key, arm in arms.items():
            for stat in ('pct', 'ci_l', 'ci_u'):
                arm[stat].extend(month[arm_key][stat])

        n_c = month['ctrl']['n']
        n_t = month['treat']['n']
        month_annotations.append((float(np.mean(month_xs)), month['label'], n_c + n_t, n_c, n_t))

        if m_idx < len(months) - 1:
            separator_positions.append(month_xs[-1] + (intra_step + group_gap) / 2.0)

    x_arr = np.array(x_positions, dtype=float)
    arm_styles = {
        'ctrl': {'x': x_arr - jitter, 'color': '#1f77b4', 'label': 'Control', 'edge_z': 4},
        'treat': {'x': x_arr + jitter, 'color': '#ff7f0e', 'label': 'Treatment', 'edge_z': 5},
    }

    for arm_key, st in arm_styles.items():
        pcts = np.array(arms[arm_key]['pct'], dtype=float)
        # Plot slightly jiggered overlaid bars with translucent fills and crisp outlines
        ax.bar(st['x'], pcts, width=bar_width, color=st['color'], alpha=0.55, label=st['label'], zorder=3)
        ax.bar(st['x'], pcts, width=bar_width, facecolor='none', edgecolor=st['color'], linewidth=2.2, zorder=st['edge_z'])

    # 95% confidence intervals (Wald), truncated to the 0-100% range for display
    ci_tops = []
    for arm_key, st in arm_styles.items():
        pcts = np.array(arms[arm_key]['pct'], dtype=float)
        ci_l = np.clip(np.array(arms[arm_key]['ci_l'], dtype=float), 0.0, 100.0)
        ci_u = np.clip(np.array(arms[arm_key]['ci_u'], dtype=float), 0.0, 100.0)
        ci_tops.extend(ci_u.tolist())
        ax.errorbar(
            st['x'], pcts,
            yerr=[pcts - ci_l, ci_u - pcts],
            fmt='none', ecolor='#222222', elinewidth=1.4,
            capsize=4, capthick=1.4, zorder=6
        )

    # Vertical dashed separators between month groups
    for sep_x in separator_positions:
        ax.axvline(sep_x, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

    # Headroom above the tallest confidence interval for the month header boxes
    max_top = max(ci_tops, default=0.0)
    y_max = max(75.0, np.ceil(max_top * 1.22 / 5.0) * 5.0)
    ax.set_ylim(0, y_max)
    ax.set_xlim(x_positions[0] - 0.95, x_positions[-1] + 0.95)

    # Annotate Month headers at the top of each time-period group
    for center_x, month_label, n_tot, n_c, n_t in month_annotations:
        header_text = f"{month_label}\n(N = {n_tot}: C={n_c}, T={n_t})"
        ax.text(
            center_x, y_max * 0.95, header_text,
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#cccccc', alpha=0.95),
            zorder=7
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_ylabel('Percentage of Respondents (%)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=1)
    ax.grid(axis='x', visible=False)

    legend_handles = [
        Patch(facecolor=(31 / 255, 119 / 255, 180 / 255, 0.5), edgecolor='#1f77b4', linewidth=2.0, label='Control'),
        Patch(facecolor=(255 / 255, 127 / 255, 14 / 255, 0.5), edgecolor='#ff7f0e', linewidth=2.0, label='Treatment'),
    ]
    ax.legend(
        handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.11),
        ncol=2, fontsize=11, frameon=True, facecolor='white', framealpha=0.95, edgecolor='black'
    )

    plt.tight_layout()
    upload_plot(fig, 'genai_usage_bar.png', github_pat, GITHUB_CONFIG)
    plt.close(fig)

if __name__ == '__main__':
    render()
