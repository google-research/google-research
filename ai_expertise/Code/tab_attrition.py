# ==============================================================================
# tab_attrition.py
# ------------------------------------------------------------------------------
# Generates Table A2: Linear Probability Models for Sample Attrition from Primary
# Experimental Tasks.
#
# Econometric Specification:
#   Linear probability models (OLS with HC3 robust standard errors) estimating
#   the probability of task attrition on treatment assignment, seniority,
#   onboarding comprehension, confidence, and onboarding response indicators.
#
# Inputs:
#   - Jsons/attrition.json
#
# Outputs:
#   - New/attrition.tex
# ==============================================================================

import json
from config import GITHUB_CONFIG
from utils import push_to_github

def render(github_pat=None):
    """
    Renders Table A2 (Sample attrition models) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/attrition.json', 'r') as f:
        content = json.load(f)
    
    obs_counts = content['obs_counts']
    mods = content['models']

    def fmt(val, pval):
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        return f"{val:.2f}$^{{{stars}}}$" if stars else f"{val:.2f}"

    latex = r"""\begin{table}[ht]
\centering
\begin{threeparttable}
\caption{Linear Probability Models for Sample Attrition from Primary Experimental Tasks}
\label{apxtab:attrition}\vspace{0.5cm}
\setlength{\tabcolsep}{4pt}
\small
\begin{tabular}{l *{3}{S[table-format=-1.2, input-symbols={()}, table-space-text-post={$^{***}$}]}}
\toprule
 & {(1)} & {(2)} & {(3)} \\
 & {10-Day Drafting} & {90-Day Drafting} & {90-Day Redlining} \\
\midrule""" + "\n"

    rows = [
        ("Treatment", "group_binary"),
        ("Senior", "senior"),
        ("Understanding of study (Likert)", "comprehension"),
        ("Confidence in ability to complete tasks (Likert)", "confidence"),
        ("Onboarding response", "onboarding_response"),
        ("Constant", "Intercept")
    ]

    m1, m2, m3 = mods['attrition_10d'], mods['attrition_90d'], mods['attrition_90d_cri']
    for label, var in rows:
        latex += f"    {label} & {fmt(m1['params'][var], m1['pvalues'][var])} & {fmt(m2['params'][var], m2['pvalues'][var])} & {fmt(m3['params'][var], m3['pvalues'][var])} \\\\\n"
        latex += f"     & ({m1['bse'][var]:.2f}) & ({m2['bse'][var]:.2f}) & ({m3['bse'][var]:.2f}) \\\\\n"

    latex += rf"""\midrule
    Observations & \multicolumn{{1}}{{c}}{{{obs_counts[0]}}} & \multicolumn{{1}}{{c}}{{{obs_counts[1]}}} & \multicolumn{{1}}{{c}}{{{obs_counts[2]}}} \\
    F-Statistic & {m1['fvalue']:.2f} & {m2['fvalue']:.2f} & {m3['fvalue']:.2f} \\
    Prob $>$ F & {m1['f_pvalue']:.2f} & {m2['f_pvalue']:.2f} & {m3['f_pvalue']:.2f} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\scriptsize
\item[]\hspace{{-\labelsep}}\textit{{Notes:}} Robust standard errors in parentheses. "Understanding of study" and "Confidence in ability" were both collected during the onboarding process, prior to granting access to InFlow. To include the full sample of randomized participants (\stat{{firm_summary_tot_rand_n}}), missing responses for these two covariates were imputed using their sample means. An "Onboarding response" indicator variable is included which equals 1 if the participant provided responses to these questions, and 0 otherwise. Firm 3 completed the experimental protocol before the redlining exercise was finalized, thus is excluded from the final column of this table. $^* p < 0.10$, $^{{**}} p < 0.05$, $^{{***}} p < 0.01$.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}"""

    push_to_github("attrition.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
