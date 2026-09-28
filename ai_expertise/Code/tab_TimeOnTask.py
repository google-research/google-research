# ==============================================================================
# tab_TimeOnTask.py
# ------------------------------------------------------------------------------
# Generates Table 8: Impact of AI Access on Time Spent on Experimental Tasks.
#
# Econometric Specification:
#   OLS regressions estimating treatment effects on time-on-task (minutes) across
#   10-day drafting, 90-day drafting, and 90-day redlining tasks. All models include
#   firm fixed effects and report HC1 robust standard errors.
#
# Inputs:
#   - Jsons/models_data.json
#
# Outputs:
#   - New/TimeOnTask.tex
# ==============================================================================

import json
from config import GITHUB_CONFIG
from utils import push_to_github
from utils_mock_model import MockModel
from utils import get_f_test_results
from models import build_f_test_rows

def fv(v):
    """Formats numeric values to 2 decimal places."""
    if v is None or v == "": return ""
    return f"{float(v):.2f}"

def fv1(v):
    """Formats numeric values to 1 decimal place."""
    if v is None or v == "": return ""
    return f"{float(v):.1f}"

def stars(p):
    """Calculates conventional significance stars based on p-value."""
    if p is None or p == "": return ""
    p = float(p)
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def render(github_pat=None):
    """
    Renders Table 8 (Time on task) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/models_data.json', 'r') as f:
        all_models = json.load(f)
    if 'TimeOnTask' not in all_models: return
    
    data = all_models['TimeOnTask']
    models = {}
    valid_models = [m for m in data['models'] if m is not None]
    for i, m in enumerate(valid_models):
        models[i+1] = MockModel(m)
    
    noffe_data = all_models.get('TimeOnTask_noFFE', {})
    noffe_valid = [m for m in noffe_data.get('models', []) if m is not None]
    noffe_models = {i+1: MockModel(m) for i, m in enumerate(noffe_valid)}
    
    num_cols = len(models)
    caption = "Impact of AI Access on Time Spent on Experimental Tasks"
    label = "tab:TimeOnTask"
    notes = "Table reports intent-to-treat estimates using OLS regressions for time-on-task measures across both drafting and redlining tasks. All estimations conducted at the subject level with robust (HC1) standard errors. Columns (2), (4) and (6) report split specifications with no global intercept. The hypothesis tests in the bottom panel report two-sided $p$-values."
    
    def join_pairs(cells):
        pairs = [f"{cells[0]} & {cells[1]}", f"{cells[2]} & {cells[3]}", f"{cells[4]} & {cells[5]}"]
        return " & & ".join(pairs)

    latex = r"\begin{table}[H]" + "\n" + r"\singlespacing" + "\n" + r"\centering" + "\n" + r"\begin{threeparttable}" + "\n"
    latex += rf"\caption{{{caption}}}\label{{{label}}}" + "\n"
    latex += r"\setlength{\tabcolsep}{0pt}" + "\n" + r"\small" + "\n"
    latex += r"" + "\n"
    s_col = r"S[table-format=-3.2, input-symbols={()[]}, table-space-text-post={$^{***}$}]"
    group_spec = f"*{{2}}{{{s_col}}}"
    col_str = r"l @{\extracolsep{\fill}} " + r" @{\extracolsep{0pt}} c @{\extracolsep{\fill}} ".join([group_spec] * 3) + r" @{}"
    latex += rf"\begin{{tabular*}}{{\textwidth}}{{{col_str}}}" + "\n" + r"\toprule" + "\n"
    latex += r"  & \multicolumn{2}{c}{\parbox[b]{3.0cm}{\centering 10-day drafting;\\time on task (Minutes)}} & & \multicolumn{2}{c}{\parbox[b]{3.0cm}{\centering 90-day drafting;\\time on task (Minutes)}} & & \multicolumn{2}{c}{\parbox[b]{3.0cm}{\centering 90-day redlining;\\time on task (Minutes)}} \\" + "\n"
    latex += r" \cmidrule{2-3} \cmidrule{5-6} \cmidrule{8-9}" + "\n"
    latex += "  & " + join_pairs([f"{{({i})}}" for i in range(1, num_cols + 1)]) + r" \\" + "\n" + r"\midrule" + "\n"

    row_defs = [
        ('group_binary', 'Treatment'),
        ('treat_x_junior', r'Treat $\times$ Junior'),
        ('treat_x_senior', r'Treat $\times$ Senior'),
        ('junior', 'Junior'),
        ('senior', 'Senior'),
        ('Intercept', 'Constant')
    ]

    for v_key, v_lbl in row_defs:
        r_cells, s_cells, has_val = [], [], False
        for i in range(1, num_cols + 1):
            res = models.get(i)
            if res and v_key in res.params:
                v, p, se = res.params[v_key], res.pvalues[v_key], res.bse[v_key]
                s = "" if v_key in ['const', 'Intercept', 'junior', 'senior'] else stars(p)
                s_fmt = f"$^{{{s}}}$" if s else ""
                r_cells.append(f"{fv(v)}{s_fmt}")
                s_cells.append(f"({fv(se)})")
                has_val = True
            else:
                r_cells.append("{}")
                s_cells.append("{}")
        if has_val:
            latex += f"    {v_lbl} & " + join_pairs(r_cells) + r" \\" + "\n"
            latex += "     & " + join_pairs(s_cells) + r" \\" + "\n"

    latex += "     & " + join_pairs(["{}"] * num_cols) + r" \\[-1ex]" + "\n"
    
    f_tests_info = [
        ('h1', "junior = senior", r"$H_0: \alpha_{\text{junior}} = \alpha_{\text{senior}}$"),
        ('h2', "treat_x_junior = treat_x_senior", r"$H_0: \beta_{\text{junior}} = \beta_{\text{senior}}$"),
        ('h3', "junior + treat_x_junior = senior", r"$H_0: \alpha_{\text{junior}} + \beta_{\text{junior}} = \alpha_{\text{senior}}$"),
        ('h4', "junior + treat_x_junior = senior + treat_x_senior", r"$H_0: \alpha_{\text{junior}} + \beta_{\text{junior}} = \alpha_{\text{senior}} + \beta_{\text{senior}}$")
    ]
    for h_key, test_str, label_str_h in f_tests_info:
        c = []
        for m_idx in range(1, num_cols + 1):
            res = models.get(m_idx)
            if res and m_idx in [2, 4, 6]:
                f_val, p_val = get_f_test_results(res, test_str, label_str_h)
                if p_val is not None:
                    s = stars(p_val)
                    s_fmt = f"$^{{{s}}}$" if s else ""
                    c.append(f"{fv(p_val)}{s_fmt}")
                else:
                    c.append("")
            else:
                c.append("")
        latex += f"    {label_str_h} & " + join_pairs(c) + r" \\" + "\n"
    latex += "     & " + join_pairs([""] * num_cols) + r" \\[-1ex]" + "\n"

    latex += r"\midrule" + "\n"
    fe_rows = [
        ("Firm FE", ["Yes"] * num_cols),
    ]
    for f_lbl, f_vals in fe_rows:
        latex += f"    {f_lbl} & " + join_pairs([rf"\multicolumn{{1}}{{c}}{{{x}}}" for x in f_vals]) + r" \\" + "\n"

    mean_rows = [
        ("Unconditional control mean (all)", [fv1(noffe_models[i].params.get('Intercept')) if i in noffe_models else "" for i in [1, 3, 5]]),
        ("Unconditional control mean (juniors)", [fv1(noffe_models[i].params.get('junior')) if i in noffe_models else "" for i in [2, 4, 6]]),
        ("Unconditional control mean (seniors)", [fv1(noffe_models[i].params.get('senior')) if i in noffe_models else "" for i in [2, 4, 6]]),
    ]
    for m_lbl, m_vals in mean_rows:
        latex += f"    {m_lbl} & " + " & & ".join([rf"\multicolumn{{2}}{{c}}{{{x}}}" for x in m_vals]) + r" \\" + "\n"

    obs_cells, r2_cells = [], []
    for i in range(1, num_cols + 1):
        res = models.get(i)
        obs_cells.append(rf"\multicolumn{{1}}{{c}}{{{int(res.nobs):,}}}" if res else "{}")
        r2_cells.append(f"{fv(res.rsquared)}" if res else "{}")

    latex += "    Observations & " + join_pairs(obs_cells) + r" \\" + "\n"
    latex += "    $R^2$ & " + join_pairs(r2_cells) + r" \\" + "\n"
    latex += r"\bottomrule" + "\n" + r"\end{tabular*}" + "\n"
    latex += r"\begin{tablenotes}[flushleft]" + "\n" + r"\scriptsize" + "\n"
    latex += rf"\item[]\hspace{{-\labelsep}}\textit{{Notes:}} {notes} $^* p<0.10$, $^{{**}} p<0.05$, $^{{***}} p<0.01$." + "\n"
    latex += r"\end{tablenotes}" + "\n" + r"\end{threeparttable}" + "\n" + r"\end{table}"
    
    push_to_github("TimeOnTask.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
