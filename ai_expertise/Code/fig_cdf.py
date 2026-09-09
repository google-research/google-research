# ==============================================================================
# fig_cdf.py
# ------------------------------------------------------------------------------
# Generates Empirical Cumulative Distribution Functions (CDFs) of Expert Ratings
# for Drafting and Redlining Tasks by Experience Strata.
#
# Figure Architecture:
#   - Drafting CDFs (New/cdf_drafting.png)
#   - Redlining CDFs (New/cdf_redlining.png)
#   Plots weighted empirical CDFs with 95% robust confidence bands for Control
#   (blue) and Treatment (orange), accompanied by two-sample Kolmogorov-Smirnov
#   stochastic dominance test p-values.
#
# Inputs:
#   - Jsons/fig_cdf.json
#
# Outputs:
#   - New/cdf_drafting.png
#   - New/cdf_redlining.png
# ==============================================================================

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import GITHUB_CONFIG
from utils import upload_plot

def plot_cdf_axis(ax, data_A, data_B, is_bottom, ylabel, p_dat=None, title=None, fixed_mapping=False):
    """
    Renders empirical CDF step curves and 95% confidence bands on a target axis.
    """
    if not data_A or not data_B:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
        return

    n_total = p_dat.get('n_total', 0) if (p_dat and 'n_total' in p_dat) else data_A.get('n_total', 0)
    if 'A' in data_A:
        A = data_A['A']
        B = data_A['B']
        label_A = data_A['label_A']
        label_B = data_A['label_B']
        stat = data_A['stat']
        p_less = data_A['p_less']
        p_greater = data_A['p_greater']
        
        ax.plot(A['score'], A['cdf'], color='#1f77b4', linestyle='-', linewidth=2.5, label=f"Group A: {label_A}\n(Lower Mean)")
        ax.fill_between(A['score'], A['ci_lower'], A['ci_upper'], color='#1f77b4', alpha=0.6)

        ax.plot(B['score'], B['cdf'], color='#ff7f0e', linestyle=':', linewidth=2.5, label=f"Group B: {label_B}\n(Higher Mean)")
        ax.fill_between(B['score'], B['ci_lower'], B['ci_upper'], color='#ff7f0e', alpha=0.4)

        if stat is not None:
            ks_text = f"$H_0$: A \u2265 B\np = {p_less:.3f}\n\n$H_0$: B \u2265 A\np = {p_greater:.3f}\n\nN = {n_total}"
            ax.text(0.05, 0.95, ks_text, transform=ax.transAxes, ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))

        ax.legend(loc='lower right', fontsize=9, framealpha=0.9, edgecolor='black')

    else:
        C = data_B
        T = data_A
        stat = p_dat.get('stat') if p_dat else None
        p_less = p_dat.get('p_less') if p_dat else None
        p_greater = p_dat.get('p_greater') if p_dat else None
        
        ax.plot(C['score'], C['cdf'], color='#1f77b4', linestyle='-', linewidth=2.5, label='Control')
        ax.fill_between(C['score'], C['ci_lower'], C['ci_upper'], color='#1f77b4', alpha=0.6)

        ax.plot(T['score'], T['cdf'], color='#ff7f0e', linestyle='--', linewidth=2.5, label='Treatment')
        ax.fill_between(T['score'], T['ci_lower'], T['ci_upper'], color='#ff7f0e', alpha=0.4)

        if stat is not None:
            ks_text = f"$H_0$: C \u2265 T\np = {p_less:.3f}\n\n$H_0$: T \u2265 C\np = {p_greater:.3f}\n\nN = {n_total}"
            ax.text(0.05, 0.95, ks_text, transform=ax.transAxes, ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))

        ax.legend(loc='lower right', fontsize=9, framealpha=0.9, edgecolor='black')

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim([-0.05, 1.05])
    if title: ax.set_title(title, fontsize=16, fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    if is_bottom: ax.set_xlabel("Score Threshold", fontsize=12)

def render(github_pat=None):
    """
    Renders Drafting and Redlining empirical CDFs to PNG.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/fig_cdf.json', 'r') as f:
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
            
            ylabel = "Cumulative Probability" if col_idx == 0 else None
            p_dat = data['plot1'].get(var, {}).get(frame['key'], {})
            
            plot_cdf_axis(ax, p_dat.get('treat'), p_dat.get('ctrl'), is_bottom, ylabel, p_dat=p_dat, fixed_mapping=True)
            
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

    fig1_drafting.tight_layout(rect=[0, 0.05, 1, 0.90], h_pad=4.0)
    upload_plot(fig1_drafting, 'cdf_drafting.png', github_pat, GITHUB_CONFIG)
    plt.close(fig1_drafting)

    fig1_redlining.tight_layout(rect=[0, 0.05, 1, 0.85], h_pad=4.0)
    upload_plot(fig1_redlining, 'cdf_redlining.png', github_pat, GITHUB_CONFIG)
    plt.close(fig1_redlining)

if __name__ == '__main__':
    render()
