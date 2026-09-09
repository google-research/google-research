# ==============================================================================
# tab_firm_summary.py
# ------------------------------------------------------------------------------
# Generates Table 1: Summary Statistics: Firm Attributes, Within-Firm Randomization,
# and Task Completion.
#
# Econometric Specification:
#   Descriptive summary table reporting cohort recruitment totals, randomized sample
#   sizes, treatment assignment counts, average years of experience, senior practitioner
#   counts, and task completion percentages across firms and control/treatment arms.
#
# Inputs:
#   - Jsons/firm_summary.json
#
# Outputs:
#   - New/firm_summary.tex
# ==============================================================================

import json
from config import GITHUB_CONFIG
from utils import push_to_github

def render(github_pat=None):
    """
    Renders Table 1 (Firm summary attributes & task completion) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/firm_summary.json', 'r') as f:
        content = json.load(f)
    
    data = content['firms']
    totals = content['totals']
    macros = content['macros']

    latex = r"""\begin{sidewaystable}[htbp]
\centering
\begin{threeparttable}
\caption{Summary Statistics: Firm Attributes, Within-Firm Randomization, Task Completion}
\label{tab:firm_summary}
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{l c r@{\hspace{3.5pt}}l r@{\hspace{3.5pt}}l c r@{\hspace{3.5pt}}l r@{\hspace{3.5pt}}l r@{\hspace{3.5pt}}l r@{\hspace{3.5pt}}l r@{\hspace{3.5pt}}l r@{\hspace{3.5pt}}l r@{\hspace{3.5pt}}l}
\toprule
 & & \multicolumn{4}{c}{Sample Size} & \multicolumn{1}{c}{Experience} & \multicolumn{2}{c}{} & \multicolumn{6}{c}{\shortstack{Completion rates among \\ control subjects}} & \multicolumn{6}{c}{\shortstack{Completion rates among \\ treatment subjects}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-7} \cmidrule(lr){10-15} \cmidrule(lr){16-21}
Firm & \shortstack{N \\ (Recruited)} & \multicolumn{2}{c}{\shortstack{N \\ (Randomized)}} & \multicolumn{2}{c}{\shortstack{N \\ (Treated)}} & Average & \multicolumn{2}{c}{\shortstack{N 'Seniors' \\ ($\ge$7 YoE)}} & \multicolumn{2}{c}{\shortstack{10-day \\ drafting}} & \multicolumn{2}{c}{\shortstack{90-day \\ drafting}} & \multicolumn{2}{c}{\shortstack{90-day \\ redlining}} & \multicolumn{2}{c}{\shortstack{10-day \\ drafting}} & \multicolumn{2}{c}{\shortstack{90-day \\ drafting}} & \multicolumn{2}{c}{\shortstack{90-day \\ redlining}} \\
\midrule""" + "\n"

    for i, r in enumerate(data):
        f_rand = r['n_randomized']
        f_treat = r['n_treated']
        f_ctrl = f_rand - f_treat
        
        pct_rand = (f_rand / r['n_recruited']) * 100 if r['n_recruited'] else 0
        pct_treat = (f_treat / f_rand) * 100 if f_rand else 0
        pct_sen = (r['n_senior'] / f_rand) * 100 if f_rand else 0
        
        pct_tt1_treat = (r['c_tt1_treat'] / f_treat) * 100 if f_treat else 0
        pct_tt2_treat = (r['c_tt2_treat'] / f_treat) * 100 if f_treat else 0
        pct_tt2_cri_treat = (r['c_tt2_cri_treat'] / f_treat) * 100 if f_treat else 0
        
        pct_tt1_ctrl = (r['c_tt1_ctrl'] / f_ctrl) * 100 if f_ctrl else 0
        pct_tt2_ctrl = (r['c_tt2_ctrl'] / f_ctrl) * 100 if f_ctrl else 0
        pct_tt2_cri_ctrl = (r['c_tt2_cri_ctrl'] / f_ctrl) * 100 if f_ctrl else 0

        if r['firm_name'] == 'Firm 3':
            c_tt2_cri_ctrl_cols = r"\multicolumn{2}{c}{NA}"
            c_tt2_cri_treat_cols = r"\multicolumn{2}{c}{NA}"
        else:
            c_tt2_cri_ctrl_cols = f"{r['c_tt2_cri_ctrl']:.0f} & ({pct_tt2_cri_ctrl:.0f}\\%)"
            c_tt2_cri_treat_cols = f"{r['c_tt2_cri_treat']:.0f} & ({pct_tt2_cri_treat:.0f}\\%)"

        latex += f"    Firm {i+1} & {r['n_recruited']:.0f} & {r['n_randomized']:.0f} & ({pct_rand:.0f}\\%) & {r['n_treated']:.0f} & ({pct_treat:.0f}\\%) & {r['exp_avg']:.1f} & {r['n_senior']:.0f} & ({pct_sen:.0f}\\%) & {r['c_tt1_ctrl']:.0f} & ({pct_tt1_ctrl:.0f}\\%) & {r['c_tt2_ctrl']:.0f} & ({pct_tt2_ctrl:.0f}\\%) & {c_tt2_cri_ctrl_cols} & {r['c_tt1_treat']:.0f} & ({pct_tt1_treat:.0f}\\%) & {r['c_tt2_treat']:.0f} & ({pct_tt2_treat:.0f}\\%) & {c_tt2_cri_treat_cols} \\\\\n"

    pct_tot_rand = (totals['tot_rand']/totals['tot_rec'])*100 if totals['tot_rec'] else 0
    pct_tot_treat = (totals['tot_treated']/totals['tot_rand'])*100 if totals['tot_rand'] else 0
    pct_tot_sen = (totals['tot_senior']/totals['tot_rand'])*100 if totals['tot_rand'] else 0
    
    pct_tot_tt1_treat = (totals['tot_tt1_treat']/totals['tot_treated'])*100 if totals['tot_treated'] else 0
    pct_tot_tt2_treat = (totals['tot_tt2_treat']/totals['tot_treated'])*100 if totals['tot_treated'] else 0
    pct_tot_tt2_cri_treat = (totals['tot_tt2_cri_treat']/totals['tot_treated'])*100 if totals['tot_treated'] else 0

    pct_tot_tt1_ctrl = (totals['tot_tt1_ctrl']/totals['tot_ctrl'])*100 if totals['tot_ctrl'] else 0
    pct_tot_tt2_ctrl = (totals['tot_tt2_ctrl']/totals['tot_ctrl'])*100 if totals['tot_ctrl'] else 0
    pct_tot_tt2_cri_ctrl = (totals['tot_tt2_cri_ctrl']/totals['tot_ctrl'])*100 if totals['tot_ctrl'] else 0

    latex += rf"""\midrule
    All & {totals['tot_rec']:.0f} & {totals['tot_rand']:.0f} & ({pct_tot_rand:.0f}\%) & {totals['tot_treated']:.0f} & ({pct_tot_treat:.0f}\%) & {totals['exp_mean']:.1f} & {totals['tot_senior']:.0f} & ({pct_tot_sen:.0f}\%) & {totals['tot_tt1_ctrl']:.0f} & ({pct_tot_tt1_ctrl:.0f}\%) & {totals['tot_tt2_ctrl']:.0f} & ({pct_tot_tt2_ctrl:.0f}\%) & {totals['tot_tt2_cri_ctrl']:.0f} & ({pct_tot_tt2_cri_ctrl:.0f}\%) & {totals['tot_tt1_treat']:.0f} & ({pct_tot_tt1_treat:.0f}\%) & {totals['tot_tt2_treat']:.0f} & ({pct_tot_tt2_treat:.0f}\%) & {totals['tot_tt2_cri_treat']:.0f} & ({pct_tot_tt2_cri_treat:.0f}\%) \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\scriptsize
\item[]\hspace{{-\labelsep}}\textit{{Notes:}} Firms arranged in descending order of original cohort size. 'Seniors' refers to lawyers with seven or more years of legal experience (see Section \ref{{sec:sensitivity}} for details pertaining to this choice). Treatment status, experience strata, seniors, and completion rates are conditional on being effectively randomized. Completion rates are split into control and treatment panels. Firm 3 completed the experimental protocol before the redlining exercise was finalized, thus is excluded from the final column of this table.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{sidewaystable}}"""

    push_to_github("firm_summary.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
