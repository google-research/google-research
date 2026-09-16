# ==============================================================================
# tab_90dayR_cheating_ind_human.py
# ------------------------------------------------------------------------------
# Generates Table A6: Impact of 90-Day AI Access on Non-AI-Assisted Redlining
# Performance: Controlling for Suspected Non-Adherence Using Expert Ratings.
#
# Econometric Specification:
#   OLS regressions evaluating whether treatment effect patterns on 90-day
#   redlining evaluated by human experts are driven by prompt leakage / unauthorized
#   AI tool use during the offline control task.
#   - Models (1)-(2): Baseline full sample.
#   - Models (3)-(4): Controlling for suspected non-adherence indicator.
#   - Models (5)-(6): Excluding suspected non-adherent subjects.
#
# Inputs:
#   - Jsons/models_data.json
#
# Outputs:
#   - New/90dayR_cheating_ind_human.tex
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
    Renders Table A6 (Suspected non-adherence robustness, human raters) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/models_data.json', 'r') as f:
        all_models = json.load(f)
    if '90dayR_cheating_ind_human' not in all_models: return
    
    data = all_models['90dayR_cheating_ind_human']
    models = {}
    valid_models = [m for m in data['models'] if m is not None]
    for i, m in enumerate(valid_models):
        models[i+1] = MockModel(m)
    
    num_cols = len(models)
    caption = "Impact of 90-Day AI Access on Non-AI-Assisted Redlining Performance: Controlling for Suspected Non-Adherence Using Expert Ratings"
    label = "apxtab:cheating_human"
    notes = r"Table reports intent-to-treat estimates using OLS regressions for 90-day redlining quality evaluated by expert raters. All models report rating-level disaggregated regressions using standard errors clustered at the individual level. Models (1) and (2) report estimates on the full active sample. Models (3) and (4) control for suspected non-adherence, while Models (5) and (6) exclude participants with suspected non-adherence. Non-adherence is flagged by Gemini on the basis of the prompt we provide in Appendix \ref{apx:nonadhereprompt}. All outcome subcomponents are standardized to mean zero and unit variance using the control group calculated locally within the full active sample. All models incorporate firm fixed effects and sub-indicator fixed effects, and are weighted by $w_i=1/n_i$, where $n_i$ is the number of ratings individual $i$ receives so that each individual carries equal total weight. $^* p<0.10$, $^{**} p<0.05$, $^{***} p<0.01$."
    
    latex = r"\begin{sidewaystable}[htbp]" + "\n" + r"\centering" + "\n" + r"\begin{threeparttable}" + "\n"
    latex += rf"\caption{{{caption}}}\label{{{label}}}\vspace{{0.5cm}}" + "\n"
    latex += r"\setlength{\tabcolsep}{4pt}" + "\n" + r"\small" + "\n"
    latex += r"" + "\n"
    n_fmt = f" *{{{num_cols}}}{{S[table-format=-1.2, input-symbols={{()[]}}, table-space-text-post={{$^{{***}}$}}]}}"
    latex += rf"\begin{{tabular}}{{@{{}}l{n_fmt}}}" + "\n" + r"\toprule" + "\n"

    latex += r"  & \multicolumn{2}{c}{Full Sample} & \multicolumn{2}{c}{Controlling for Non-Adherence} & \multicolumn{2}{c}{Excluding Non-Adherence} \\" + "\n"
    latex += r" \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}" + "\n"
    latex += "  & " + " & ".join([f"{{({i})}}" for i in range(1, num_cols + 1)]) + r" \\" + "\n" + r"\midrule" + "\n"

    row_defs = [
        ('group_binary', 'Treatment'),
        ('treat_x_junior', r'Treat $\times$ Junior'),
        ('treat_x_senior', r'Treat $\times$ Senior'),
        ('cheating', 'Reported AI Use'),
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
    latex += build_f_test_rows(models, [2, 4, 6], num_cols, '90dayR_cheating_ind_human', {})
    latex += r"\midrule" + "\n"
    fe_rows = [
        ("Rating-level Disaggregated", ["Yes"] * num_cols),
        ("Firm FE", ["Yes"] * num_cols),
        ("Sub-indicator FE", ["Yes"] * num_cols)
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
    
    push_to_github("90dayR_cheating_ind_human.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
