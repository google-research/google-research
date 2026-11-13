# ==============================================================================
# tab_varregs.py
# ------------------------------------------------------------------------------
# Generates Tables 5 & 7: Impact of AI Access on Distributional Probability of
# Expert Ratings of Drafting and Redlining Tasks by Subgroup.
#
# Econometric Specification:
#   Linear probability regressions estimating changes in the probability of
#   falling into performance quintiles (Top to Bottom), along with Levene
#   equality-of-variance statistics, across the full sample, juniors, and seniors.
#   All models include firm fixed effects and report HC1 robust standard errors.
#
# Inputs:
#   - Jsons/additional_analysis.json
#
# Outputs:
#   - New/varregs_drafting.tex (Table 5: Drafting distributional regressions)
#   - New/varregs_redlining.tex (Table 7: Redlining distributional regressions)
# ==============================================================================

import json
import pandas as pd
import numpy as np
from config import GITHUB_CONFIG
from utils import push_to_github

def render(github_pat=None):
    """
    Renders Tables 5 & 7 (Drafting & Redlining Variance Regressions) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/additional_analysis.json', 'r') as f:
        data = json.load(f)
        
    results = data.get('varregs', {})
    if not results:
        print("No results found in Jsons/additional_analysis.json")
        return

    # Drafting variables
    drafting_vars = [('tt1_sum_rat_drades', '10 Day Drafting'), ('tt2_sum_rat_drades', '90 Day Drafting')]
    # Redlining variables
    redlining_vars = [('tt2_sum_rat_cri', '90 Day Redlining')]
    
    subgroups = [('Full Sample', 'All'), ('Juniors', 'Juniors'), ('Seniors', 'Seniors')]
    
    def format_stars(pval):
        if pval is None or pd.isna(pval): return ""
        if pval < 0.01: return "$^{***}$"
        if pval < 0.05: return "$^{**}$"
        if pval < 0.10: return "$^{*}$"
        return ""
        
    def build_table(title, label, outcomes):
        latex = "\\begin{table}[H]\n"
        latex += "\\singlespacing\n"
        latex += "\\centering\n"
        latex += "\\begin{threeparttable}\n"
        latex += f"\\caption{{{title}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\setlength{\\tabcolsep}{0pt}\n"
        latex += "\\small\n\n"
        
        s_col = "S[table-format=-1.2, input-symbols={()}, table-space-text-post={$^{***}$}]"
        group_spec = f"*{{{len(subgroups)}}}{{{s_col}}}"
        col_str = "l @{\\extracolsep{\\fill}} " + " @{\\extracolsep{0pt}} c @{\\extracolsep{\\fill}} ".join([group_spec] * len(outcomes)) + " @{}"
        latex += f"\\begin{{tabular*}}{{\\textwidth}}{{{col_str}}}\n"
        latex += "\\toprule\n"
        
        # Main headers
        latex += " & " + " & & ".join([f"\\multicolumn{{3}}{{c}}{{{name}}}" for _, name in outcomes]) + " \\\\\n"
        
        # Cmidrules
        cmd_rules = []
        for i in range(len(outcomes)):
            start = 2 + i * (len(subgroups) + 1)
            end = start + len(subgroups) - 1
            cmd_rules.append(f"\\cmidrule{{{start}-{end}}}")
        latex += " ".join(cmd_rules) + "\n"
        
        # Subheaders
        latex += " & " + " & & ".join([" & ".join([f"\\multicolumn{{1}}{{c}}{{{sub_name}}}" for _, sub_name in subgroups]) for _ in outcomes]) + " \\\\\n"
        
        # N=...
        def get_n(out_key, sg_key):
            obs = results[sg_key][out_key]['q1']['obs']
            if obs is None or pd.isna(obs):
                raise ValueError(f"Missing observation count for outcome={out_key}, subgroup={sg_key}")
            return str(int(obs))
                
        latex += " & " + " & & ".join([" & ".join([f"\\multicolumn{{1}}{{c}}{{(N={get_n(out_key, sg_key)})}}" for sg_key, _ in subgroups]) for out_key, _ in outcomes]) + " \\\\\n"
        latex += "\\midrule\n"
        
        rows = [
            ("Levene Statistic", "Levene"),
            ("Pr(Top (First) Quintile)", "q1"),
            ("Pr(Second Quintile)", "q2"),
            ("Pr(Third Quintile)", "q3"),
            ("Pr(Fourth Quintile)", "q4"),
            ("Pr(Fifth (Bottom) Quintile)", "q5")
        ]
        
        for row_label, stat_key in rows:
            # Coef / stat row
            latex += f"{row_label} & "
            group_coefs = []
            for out_key, _ in outcomes:
                row_coefs = []
                for sg_key, _ in subgroups:
                    res = results.get(sg_key, {}).get(out_key, {}).get(stat_key, {})
                    if stat_key == 'Levene':
                        val = res.get('stat')
                        pval = res.get('pval')
                    else:
                        val = res.get('coef')
                        pval = res.get('pval')
                        
                    if val is None or pd.isna(val):
                        row_coefs.append("")
                    else:
                        stars = format_stars(pval)
                        row_coefs.append(f"{val:.2f}{stars}")
                group_coefs.append(" & ".join(row_coefs))
            latex += " & & ".join(group_coefs) + " \\\\\n"
            
            # SE / pval row
            latex += " & "
            group_ses = []
            for out_key, _ in outcomes:
                row_ses = []
                for sg_key, _ in subgroups:
                    res = results.get(sg_key, {}).get(out_key, {}).get(stat_key, {})
                    if stat_key == 'Levene':
                        val = res.get('pval')
                    else:
                        val = res.get('se')
                        
                    if val is None or pd.isna(val):
                        row_ses.append("")
                    else:
                        row_ses.append(f"({val:.2f})")
                group_ses.append(" & ".join(row_ses))
            latex += " & & ".join(group_ses) + " \\\\\n"
            
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item[]\\hspace{-\\labelsep}\\textit{Notes:} This table presents intent-to-treat estimates on distributional changes and variance equality using rating-level observations, weighted by the inverse of the number of ratings per subject. Summary outcome variables are standardized to mean zero and unit variance using the control group. Cutoffs for quintiles are determined using the full valid sample for each respective outcome. Coefficients indicate the change in probability of falling into a specific quintile. All models include firm fixed effects and report HC1 robust standard errors in parentheses. The Levene statistic tests the null hypothesis of equal variances between treatment and control groups (centered at the median), with corresponding p-values in parentheses. The full sample includes all participants properly randomized. Juniors are defined as having $<7$ years of experience. $^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table}\n"
        
        return latex

    drafting_tex = build_table(
        "Impact of AI Access on Distributional Probability of Expert Ratings of the Drafting Tasks by Subgroup",
        "tab:varregs_drafting",
        drafting_vars
    )
    push_to_github("varregs_drafting.tex", drafting_tex, github_pat, GITHUB_CONFIG)
    
    redlining_tex = build_table(
        "Impact of AI Access on Distributional Probability of Expert Ratings of the Redlining Task by Subgroup",
        "tab:varregs_redlining",
        redlining_vars
    )
    push_to_github("varregs_redlining.tex", redlining_tex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
