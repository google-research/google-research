# ==============================================================================
# tab_balance.py
# ------------------------------------------------------------------------------
# Generates Table A1: Tests of Treatment-Control Covariate Balance.
#
# Econometric Specification:
#   Two-sided Welch's t-tests comparing baseline observable characteristics
#   (response time, comprehension, confidence, experience, seniority) across
#   treatment and control groups.
#
# Inputs:
#   - Jsons/balance.json
#
# Outputs:
#   - New/balance.tex
# ==============================================================================

import json
from config import GITHUB_CONFIG
from utils import push_to_github

def render(github_pat=None):
    """
    Renders Table A1 (Covariate balance tests) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/balance.json', 'r') as f:
        content = json.load(f)
    
    data = content['data']
    macros = content['macros']

    latex = r"""\begin{table}[htbp]\centering
\begin{threeparttable}
\caption{Tests of Treatment-Control Covariate Balance}
\label{apxtab:balance}
\begin{tabular}{l *{5}{r@{.}l}}
\toprule
& \multicolumn{4}{c}{Control} & \multicolumn{4}{c}{Treatment} & \multicolumn{2}{c}{} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
Variable & \multicolumn{2}{c}{N} & \multicolumn{2}{c}{Mean} & \multicolumn{2}{c}{N} & \multicolumn{2}{c}{Mean} & \multicolumn{2}{c}{T-C} \\
\midrule""" + "\n"

    for row in data['rows']:
        m1p, m2p, diffp = f"{row['mean1']:.2f}".split('.'), f"{row['mean2']:.2f}".split('.'), f"{row['diff']:.2f}".split('.')
        se1p, se2p, sedp = f"{row['se1']:.2f}".split('.'), f"{row['se2']:.2f}".split('.'), f"{row['se_diff']:.2f}".split('.')

        latex += rf"    {row['var_name']} & \multicolumn{{2}}{{c}}{{{row['n1']}}} & {m1p[0]} & {m1p[1]} & \multicolumn{{2}}{{c}}{{{row['n2']}}} & {m2p[0]} & {m2p[1]} & {diffp[0]} & {diffp[1]} \\" + "\n"
        latex += rf"    & \multicolumn{{2}}{{c}}{{}} & ({se1p[0]} & {se1p[1]}) & \multicolumn{{2}}{{c}}{{}} & ({se2p[0]} & {se2p[1]}) & ({sedp[0]} & {sedp[1]}) \\[1ex]" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\scriptsize
\item[]\hspace{-\labelsep}\textit{Notes:} Comparison of treatment and control means using two-sided Welch's t-tests. Standard errors are reported in parentheses below the means and differences. "Understanding of study" and "Confidence in ability" were both collected during the onboarding process, prior to granting access to InFlow. * p$<$0.10, ** p$<$0.05, *** p$<$0.01.
\end{tablenotes}
\end{threeparttable}
\end{table}"""
    
    push_to_github("balance.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
