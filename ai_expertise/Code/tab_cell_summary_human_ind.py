# ==============================================================================
# tab_cell_summary_human_ind.py
# ------------------------------------------------------------------------------
# Generates Table 2: Experimental Performance Scores, Overall and by Seniority,
# Using Expert Ratings.
#
# Econometric Specification:
#   Summary table reporting standard normal cell means and standard deviations
#   across Control, Treatment, Junior Control, Junior Treatment, Senior Control,
#   and Senior Treatment cells for expert human ratings.
#
# Inputs:
#   - Jsons/cell_summary.json
#
# Outputs:
#   - New/cell_summary_human_ind.tex
# ==============================================================================

import json
import pandas as pd
from config import GITHUB_CONFIG, STD_USE_POOLED
from utils import push_to_github

def fmt_val(val):
    """Formats numeric floats to 2 decimal places, handling signed zeros."""
    if pd.isna(val) or val is None: return ""
    res = f"{val + 0.0:.2f}"
    return "0.00" if res == "-0.00" else res

STD_NOTE = "the pooled sample mean and variance" if STD_USE_POOLED else "the control group mean and variance"

def render(github_pat=None):
    """
    Renders Table 2 (Cell summary statistics, human ratings) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/cell_summary.json', 'r') as f:
        content = json.load(f)
    
    sections_data = content['data'].get("human_ind")
    if not sections_data: return

    title = "Experimental Performance Scores, Overall and by Seniority, Using Expert Ratings"
    label = "tab:cell_summary_human_ind"
    level_notes = "Main outcome observations are at the individual expert rating level."

    latex = r"""\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{""" + title + r"""}\label{""" + label + r"""}\vspace{0.5cm}
\setlength{\tabcolsep}{8pt}
\small
\begin{tabular}{@{}l *{6}{S[table-format=-1.2, input-symbols={()}]}@{}}
\toprule
 & \multicolumn{2}{c}{All} & \multicolumn{2}{c}{Junior} & \multicolumn{2}{c}{Senior} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
 & \multicolumn{1}{c}{Control} & \multicolumn{1}{c}{Treatment} & \multicolumn{1}{c}{Control} & \multicolumn{1}{c}{Treatment} & \multicolumn{1}{c}{Control} & \multicolumn{1}{c}{Treatment} \\
\midrule""" + "\n"

    for s_idx, sec in enumerate(sections_data):
        title_sec = sec['title']
        latex += f"    \\multicolumn{{7}}{{@{{}}l}}{{{title_sec}}} \\\\\n"

        for row in sec['rows']:
            row_title = "    Pooled Score" if row['idx'] == 0 else r"    \quad " + row['name']
            latex += f"{row_title} & {fmt_val(row['m_c'])} & {fmt_val(row['m_t'])} & {fmt_val(row['m_jc'])} & {fmt_val(row['m_jt'])} & {fmt_val(row['m_sc'])} & {fmt_val(row['m_st'])} \\\\\n"
            latex += f"    & ({fmt_val(row['s_c'])}) & ({fmt_val(row['s_t'])}) & ({fmt_val(row['s_jc'])}) & ({fmt_val(row['s_jt'])}) & ({fmt_val(row['s_sc'])}) & ({fmt_val(row['s_st'])}) \\\\\n"

        obs = sec['obs']
        latex += f"    Observations & \\multicolumn{{1}}{{c}}{{{obs['n_c']}}} & \\multicolumn{{1}}{{c}}{{{obs['n_t']}}} & \\multicolumn{{1}}{{c}}{{{obs['n_jc']}}} & \\multicolumn{{1}}{{c}}{{{obs['n_jt']}}} & \\multicolumn{{1}}{{c}}{{{obs['n_sc']}}} & \\multicolumn{{1}}{{c}}{{{obs['n_st']}}} \\\\\n"

        if s_idx < len(sections_data) - 1:
            latex += r"\midrule" + "\n"

    human_note = r" Observation counts reflect rater-level evaluations. Due to rater miscommunications detailed in Section \ref{sec:modelspecs}, a small number of subjects received more or fewer than exactly two evaluations, resulting in observation counts that slightly diverge from $N\times2$."

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\scriptsize
\item[]\hspace{-\labelsep}\textit{Notes:} Table presents standard normal cell means with standard deviations reported in parentheses below. Sample restricts to successfully randomized subjects completing the study. """ + level_notes + r""" Experience ('Senior') requires $\ge$ 7 years of practice. All dependent variables are standardized using """ + STD_NOTE + r"""\text{. Pooled task scores represent the simple arithmetic average of standardized subcomponents.}""" + human_note + r"""
\end{tablenotes}
\end{threeparttable}
\end{table}"""

    push_to_github(f"cell_summary_human_ind.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
