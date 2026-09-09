# ==============================================================================
# tab_DraftingSurvey.py
# ------------------------------------------------------------------------------
# Generates Table A8: Impact of AI Access on Perceptions of Speed, Quality, and
# Satisfaction in Drafting Tasks at 10 and 90 Days.
#
# Econometric Specification:
#   OLS regressions estimating treatment effects on practitioner self-reported
#   measures of drafting speed, quality, and satisfaction at both 10-day and 90-day
#   horizons. All models include firm fixed effects and report HC1 robust standard errors.
#
# Inputs:
#   - Jsons/models_data.json
#
# Outputs:
#   - New/DraftingSurvey.tex
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
    Renders Table A8 (Drafting survey perceptions) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/models_data.json', 'r') as f:
        all_models = json.load(f)
    if 'DraftingSurvey' not in all_models: return
    
    data = all_models['DraftingSurvey']
    models = {}
    valid_models = [m for m in data['models'] if m is not None]
    for i, m in enumerate(valid_models):
        models[i+1] = MockModel(m)
    
    num_cols = len(models)
    caption = "Impact of AI Access on Perceptions of Speed, Quality, and Satisfaction in Drafting Tasks at 10 and 90 Days"
    label = "apxtab:DraftingSurvey"
    notes = "Table reports intent-to-treat estimates using OLS regressions for self-reported speed, quality, and satisfaction outcomes from the 10-day and 90-day post-task surveys. All estimations conducted at the subject level with robust (HC1) standard errors. Columns (2), (4), (6), (8), (10), and (12) report split specifications with no global intercept. Unless otherwise indicated in column titles, all outcomes are standardized with mean 0 and standard deviation of 1 using the control group distribution. $^* p<0.10$, $^{{**}} p<0.05$, $^{{***}} p<0.01$." + "\n" 
    
    latex = r"\begin{sidewaystable}[htbp]" + "\n" + r"\centering" + "\n" + r"\begin{threeparttable}" + "\n"
    latex += rf"\caption{{{caption}}}\label{{{label}}}\vspace{{0.5cm}}" + "\n"
    latex += r"\setlength{\tabcolsep}{2pt}" + "\n" + r"\small" + "\n"
    latex += r"" + "\n"
    n_fmt = f" *{{{num_cols}}}{{S[table-format=-1.2, input-symbols={{()[]}}, table-space-text-post={{$^{{***}}$}}]}}"
    latex += rf"\begin{{tabular}}{{@{{}}l{n_fmt}}}" + "\n" + r"\toprule" + "\n"
    latex += r"  & \multicolumn{6}{c}{10-Day Drafting} & \multicolumn{6}{c}{90-Day Drafting} \\" + "\n"
    latex += r" \cmidrule(lr){2-7} \cmidrule(lr){8-13}" + "\n"
    latex += (
        r"  & \multicolumn{2}{c}{\parbox[b]{2.4cm}{\centering Speed}} "
        r"& \multicolumn{2}{c}{\parbox[b]{2.4cm}{\centering Quality}} "
        r"& \multicolumn{2}{c}{\parbox[b]{2.4cm}{\centering Satisfaction}} "
        r"& \multicolumn{2}{c}{\parbox[b]{2.4cm}{\centering Speed}} "
        r"& \multicolumn{2}{c}{\parbox[b]{2.4cm}{\centering Quality}} "
        r"& \multicolumn{2}{c}{\parbox[b]{2.4cm}{\centering Satisfaction}} \\" + "\n"
    )
    latex += r" \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}" + "\n"
    latex += "  & " + " & ".join([f"{{({i})}}" for i in range(1, num_cols + 1)]) + r" \\" + "\n" + r"\midrule" + "\n"

    row_defs = [
        ('group_binary', 'Treatment'),
        ('treat_x_junior', r'Treat $\times$ Junior'),
        ('treat_x_senior', r'Treat $\times$ Senior'),
        ('junior', 'Junior'),
        ('senior', 'Senior'),
        ('Intercept', 'Constant')
    ]

    for v_key, v_lbl in row_defs:
        r_str, s_str, has_val = f"    {v_lbl}", "    ", False
        for i in range(1, num_cols + 1):
            res = models.get(i)
            if res and v_key in res.params:
                v, p, se = res.params[v_key], res.pvalues[v_key], res.bse[v_key]
                s = "" if v_key in ['const', 'Intercept', 'junior', 'senior'] else stars(p)
                s_fmt = f"$^{{{s}}}$" if s else ""
                r_str += f" & {fv(v)}{s_fmt}"; s_str += f" & ({fv(se)})"; has_val = True
            else:
                r_str += " & {}"; s_str += " & {}"
        if has_val: latex += r_str + r" \\" + "\n" + s_str + r" \\" + "\n"

    latex += r"     & " + " & ".join(["{}"]*num_cols) + r" \\[-1ex]" + "\n"
    latex += build_f_test_rows(models, [2, 4, 6, 8, 10, 12], num_cols, 'DraftingSurvey', {})
    latex += r"\midrule" + "\n"
    fe_rows = [
        ("Firm FE", ["Yes"] * num_cols),
    ]
    for f_lbl, f_vals in fe_rows:
        latex += f"    {f_lbl} & " + " & ".join([rf"\multicolumn{{1}}{{c}}{{{x}}}" for x in f_vals]) + r" \\" + "\n"

    obs_r, r2_r = "    Observations", "    $R^2$"
    for i in range(1, num_cols + 1):
        res = models.get(i)
        obs_r += rf" & \multicolumn{{1}}{{c}}{{{int(res.nobs):,}}}" if res else " & {}"
        r2_r += f" & {fv(res.rsquared)}" if res else " & {}"

    latex += obs_r + r" \\" + "\n" + r2_r + r" \\" + "\n"
    latex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    latex += r"\begin{tablenotes}[flushleft]" + "\n" + r"\scriptsize" + "\n"
    latex += rf"\item[]\hspace{{-\labelsep}}\textit{{Notes:}} {notes}" + "\n"
    latex += r"\end{tablenotes}" + "\n" + r"\end{threeparttable}" + "\n" + r"\end{sidewaystable}"
    
    push_to_github("DraftingSurvey.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
