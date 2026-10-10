# ==============================================================================
# fig_usage_alt.py
# ------------------------------------------------------------------------------
# Generates Figure A4: Longitudinal AI Copilot Adoption by Days Since Onboarding
# (14-Day Moving Average).
#
# Figure Architecture:
#   Visualizes daily active user intensity aligned relative to each firm's
#   onboarding date (Days Since Onboarding = 0), overlaid with shaded execution
#   windows for the 10-day drafting task and 90-day drafting/redlining tasks.
#
# Inputs:
#   - Jsons/fig_usage_alt.json (precomputed 14-day daily moving average active user series)
#
# Outputs:
#   - New/usage_alt.png
# ==============================================================================

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from config import GITHUB_CONFIG
from utils import upload_plot

def render(github_pat=None):
    """
    Renders Figure A4 (Copilot Adoption Over Time) using precomputed JSON series.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT

    if not os.path.exists('Jsons/fig_usage_alt.json'):
        print("Jsons/fig_usage_alt.json not found. Please run analysis.py first.")
        return

    with open('Jsons/fig_usage_alt.json', 'r') as f:
        data = json.load(f)

    series_data = data.get('series', [])
    t1_window = data.get('task1_window', [12.6, 22.5])
    t2_window = data.get('task2_window', [102.4, 125.2])

    if not series_data:
        print("Insufficient series data in Jsons/fig_usage_alt.json.")
        return

    df_series = pd.DataFrame(series_data)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    # Plot 14-day moving average weekly active user series
    ax.plot(
        df_series['days_since_onboarding'],
        df_series['active_users'],
        color='#1f77b4',
        linewidth=2,
        label='14-day moving average active users'
    )

    # Highlight 10-day task window
    ax.axvspan(
        t1_window[0], t1_window[1],
        color='grey', alpha=0.25,
        label='10-day drafting task window'
    )

    # Highlight 90-day task window
    ax.axvspan(
        t2_window[0], t2_window[1],
        color='grey', alpha=0.45,
        label='90-day drafting & redlining task window'
    )

    ax.set_xlim(0, 200)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Days Since Onboarding', fontsize=11, fontweight='bold')
    ax.set_ylabel('Active Users', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=9)

    plt.tight_layout()
    upload_plot(plt, "usage_alt.png", github_pat, GITHUB_CONFIG)
    plt.close()

if __name__ == '__main__':
    render()
