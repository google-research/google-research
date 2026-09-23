# ==============================================================================
# fig_histograms.py
# ------------------------------------------------------------------------------
# Generates Figures 2 & 3: Overlapping Distributions of Expert Quality Ratings
# for Drafting and Redlining Tasks by Experience Strata.
#
# Figure Architecture:
#   - Figures 2A & 2B: Histograms of 10-day and 90-day drafting ratings (New/histograms_drafting.png)
#   - Figure 3: Histograms of 90-day redlining ratings (New/histograms_redlining.png)
#   Overlays Control (blue) and Treatment (orange) distributions across All Lawyers,
#   Junior Lawyers (<7 yrs), and Senior Lawyers (>=7 yrs).
#
# Inputs:
#   - Jsons/fig_histograms.json
#
# Outputs:
#   - New/histograms_drafting.png
#   - New/histograms_redlining.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import GITHUB_CONFIG
from utils import upload_plot

def plot_histogram_axis(ax, data_treat, data_ctrl, is_bottom, ylabel, p_dat=None, title=None, fixed_mapping=False):
    """
    Renders overlapping weighted histogram distributions on a target subplot axis.
    """
    if not data_treat or not data_ctrl:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
        return

    n_total = p_dat.get('n_total', 0) if p_dat else 0

    t_scores = data_treat.get('scores', [])
    t_weights = data_treat.get('weights', [])
    c_scores = data_ctrl.get('scores', [])
    c_weights = data_ctrl.get('weights', [])

    bins = np.linspace(1.0, 5.0, 21)

    if len(c_scores) > 0:
        ax.hist(c_scores, bins=bins, weights=c_weights, density=True, color='#1f77b4', alpha=0.5, label='Control')
        ax.hist(c_scores, bins=bins, weights=c_weights, density=True, histtype='step', color='#1f77b4', linewidth=2.0)
    if len(t_scores) > 0:
        ax.hist(t_scores, bins=bins, weights=t_weights, density=True, color='#ff7f0e', alpha=0.5, label='Treatment')
        ax.hist(t_scores, bins=bins, weights=t_weights, density=True, histtype='step', color='#ff7f0e', linewidth=2.0)

    if n_total > 0:
        n_text = f"N = {n_total}"
        ax.text(0.05, 0.95, n_text, transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim([0.8, 5.2])
    ax.set_ylim(bottom=0)
    if title: ax.set_title(title, fontsize=16, fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    if is_bottom: ax.set_xlabel("Score", fontsize=12)

def render(github_pat=None):
    """
    Renders Drafting and Redlining Histograms to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_histograms.json', 'r') as f:
        data = json.load(f)

    sns.set_theme(style="whitegrid")
    plt.rcParams['axes.edgecolor'] = 'black'

    plot_vars = data['plot_vars']
    frames = data['frames']
    
    var_labels = {
        'tt1_drades_avg': '10d Drafting',
        'tt2_drades_avg': '90d Drafting',
        'tt2_cri_avg': '90d Redlining'
    }

    fig1_drafting, axes1_drafting = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), sharey=True, sharex=True)
    fig1_redlining, axes1_redlining = plt.subplots(nrows=1, ncols=3, figsize=(18, 5.5), sharey=True, sharex=True)
    axes1_redlining = np.array([axes1_redlining])

    for row_idx, var in enumerate(plot_vars):
        if row_idx < 2:
            ax_row = row_idx
            axes_grid = axes1_drafting
            is_bottom = (row_idx == 1)
        else:
            ax_row = 0
            axes_grid = axes1_redlining
            is_bottom = True
            
        for col_idx, frame in enumerate(frames):
            ax = axes_grid[ax_row, col_idx]
            bg_col = '#fdfdfd' if col_idx == 0 else ('#f0f8ff' if col_idx == 1 else '#fffaf0')
            ax.set_facecolor(bg_col)
            
            ylabel = "Density" if col_idx == 0 else None
            p_dat = data['plot1'].get(var, {}).get(frame['key'], {})
            
            plot_histogram_axis(ax, p_dat.get('treat'), p_dat.get('ctrl'), is_bottom, ylabel, p_dat=p_dat, fixed_mapping=True)
            
            if ax_row == 0:
                ax.annotate(frame['title'], xy=(0.5, 1), xytext=(0, 42),
                            xycoords='axes fraction', textcoords='offset points',
                            ha='center', va='bottom', fontsize=16, fontweight='bold',
                            bbox=dict(facecolor=bg_col, edgecolor='none', boxstyle='square,pad=0.4'),
                            annotation_clip=False)

            if col_idx == 1:
                ax.annotate(var_labels.get(var, var), xy=(0.5, 1), xytext=(0, 10),
                            xycoords='axes fraction', textcoords='offset points',
                            ha='center', va='bottom', fontsize=20, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.4', alpha=0.9),
                            annotation_clip=False)

    max_h_drafting = max((patch.get_height() for ax in axes1_drafting.flat for patch in ax.patches if isinstance(patch, plt.Rectangle)), default=1.0)
    axes1_drafting[0, 0].set_ylim(bottom=0, top=max_h_drafting * 1.30)

    max_h_redlining = max((patch.get_height() for ax in axes1_redlining.flat for patch in ax.patches if isinstance(patch, plt.Rectangle)), default=1.0)
    axes1_redlining[0, 0].set_ylim(bottom=0, top=max_h_redlining * 1.30)

    fig1_drafting.tight_layout(rect=[0, 0.05, 1, 0.90], h_pad=4.0)
    upload_plot(fig1_drafting, 'histograms_drafting.png', github_pat, GITHUB_CONFIG)
    plt.close(fig1_drafting)

    fig1_redlining.tight_layout(rect=[0, 0.05, 1, 0.85], h_pad=4.0)
    upload_plot(fig1_redlining, 'histograms_redlining.png', github_pat, GITHUB_CONFIG)
    plt.close(fig1_redlining)

if __name__ == '__main__':
    render()
