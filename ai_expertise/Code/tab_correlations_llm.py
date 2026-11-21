# ==============================================================================
# tab_correlations_llm.py
# ------------------------------------------------------------------------------
# Generates Table A4: Relationships Between Quality Dimensions in LLM Ratings.
#
# Econometric Specification:
#   Computes Pearson correlation matrices and Cronbach's alpha internal consistency
#   coefficients across quality subcomponents (Enforceability, Technical Accuracy,
#   Strategic Ambiguity, Completeness/Alignment, Clarity) evaluated by LLM evaluators.
#
# Inputs:
#   - Jsons/additional_analysis.json
#
# Outputs:
#   - New/correlations_llm.tex
# ==============================================================================

import json
from config import GITHUB_CONFIG
from utils import push_to_github

def render(github_pat=None):
    """
    Renders Table A4 (Inter-item correlations, LLM ratings) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/additional_analysis.json', 'r') as f:
        data = json.load(f)
    
    tasks_data = data.get('llm')
    if not tasks_data: return
    
    latex = "\\begin{table}[htbp]\n\\centering\n\\begin{threeparttable}\n\\caption{Relationships Between Quality Dimensions in LLM Ratings}\n\\label{apxtab:correlations_llm}\\vspace{0.5cm}\n\\begin{tabular}{lccccc}\n\\toprule\n"
    
    for task in tasks_data:
        t_title = task['title']
        alpha = task['alpha']
        n_obs = task['n_obs']
        c_mat = task['c_mat']
        v_cols = task['v_cols']
        var_labels = task['var_labels']
        
        latex += f"\\multicolumn{{6}}{{l}}{{{t_title} ($N={n_obs}$, Cronbach's $\\alpha = {alpha:.3f}$)}} \\\\\n"
        latex += "\\midrule\n"
        latex += " & (1) & (2) & (3) & (4) & (5) \\\\\n"
        
        for i in range(v_cols):
            row_str = f"({i+1}) {var_labels[i]}"
            for j in range(5):
                if j < i:
                    val = c_mat[i][j]
                    row_str += f" & {val:.2f}"
                elif j == i:
                    row_str += " & 1.00"
                else:
                    row_str += " &"
            row_str += " \\\\\n"
            latex += row_str
            
        latex += "\\midrule\n"
        
    latex += "\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\n\\item \\textit{Notes:} Pearson correlation coefficients between sub-dimensions of task quality evaluated by LLM raters. Cronbach's $\\alpha$ is reported for each set of sub-dimensions. \n\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n"
    
    push_to_github("correlations_llm.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
