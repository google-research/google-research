# ==============================================================================
# tab_10dayD_combined_main.py
# ------------------------------------------------------------------------------
# Generates Table 3: Impact of AI Access on Performance in AI-Assisted Drafting Task
# at 10 Days.
#
# Econometric Specification:
#   OLS regressions estimating treatment effects on 10-day drafting quality.
#   - Model (1): Subject-level average of subcomponents with HC1 robust SEs.
#   - Models (2)-(8): Rating-level stacked panels with clustered SEs (practitioner level).
#   - Models (1)-(4): Expert human ratings.
#   - Models (5)-(6): LLM ratings.
#   - Models (7)-(8): Pooled ratings controlling for evaluator modality (is_llm).
#   - Incorporates firm fixed effects and sub-indicator fixed effects.
#
# Inputs:
#   - Jsons/models_data.json
#
# Outputs:
#   - New/10dayD_combined_main.tex
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
    Renders Table 3 (10-day drafting performance) to LaTeX.
    """
    from config import GITHUB_PAT
    github_pat = github_pat or GITHUB_PAT
    with open('Jsons/models_data.json', 'r') as f:
        all_models = json.load(f)
    if '10dayD_combined_main' not in all_models: return
    
    data = all_models['10dayD_combined_main']
    models = {}
    valid_models = [m for m in data['models'] if m is not None]
    for i, m in enumerate(valid_models):
        models[i+1] = MockModel(m)
    
    num_cols = 8
    caption = "Impact of AI Access on Performance in AI-Assisted Drafting Task at 10 Days"
    
    latex = r"\begin{sidewaystable}[htbp]" + "\n" + r"\centering" + "\n" + r"\begin{threeparttable}" + "\n"
    latex += rf"\caption{{{caption}}}\label{{tab:10dayD_combined_main}}\vspace{{0.5cm}}" + "\n"
    latex += r"\setlength{\tabcolsep}{4pt}" + "\n" + r"\small" + "\n"
    latex += r"" + "\n"
    n_fmt = f" *{{{num_cols}}}{{S[table-format=-1.2, input-symbols={{()[]}}, table-space-text-post={{$^{{***}}$}}]}}"
    latex += rf"\begin{{tabular}}{{@{{}}l{n_fmt}}}" + "\n" + r"\toprule" + "\n"

    latex += r"  & \multicolumn{4}{c}{Expert Ratings} & \multicolumn{2}{c}{LLM Ratings} & \multicolumn{2}{c}{Pooled Ratings} \\" + "\n"
    latex += r" \cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}" + "\n"
    latex += "  & " + " & ".join([f"{{({i})}}" for i in range(1, num_cols + 1)]) + r" \\" + "\n" + r"\midrule" + "\n"

    row_defs = [
        ('group_binary', 'Treatment'),
        ('treat_x_junior', r'Treat $\times$ Junior'),
        ('treat_x_senior', r'Treat $\times$ Senior'),
        ('junior', 'Junior'),
        ('senior', 'Senior'),
        ('is_llm', 'LLM Rater'),
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
    latex += build_f_test_rows(models, [4, 6, 8], num_cols, '10dayD_combined_main', {})

    latex += r"\midrule" + "\n"
    fe_rows = [
        ("Subject-level Average", ["Yes", "No", "No", "No", "No", "No", "No", "No"]),
        ("Rating-level Disaggregated", ["No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]),
        ("Firm FE", ["Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes"]),
        ("Sub-indicator FE", ["No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes"]),
        (r"Firm $\times$ Sub-ind. FE", ["No", "No", "Yes", "No", "No", "No", "No", "No"])
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
    latex += rf"\item[]\hspace{{-\labelsep}}\textit{{Notes:}} Table reports intent-to-treat estimates using OLS regressions. Model (1) reports effects on the subject-level average of subcomponents using robust (HC1) standard errors. Models (2)-(8) report rating-level disaggregated regressions using standard errors clustered at the individual level. Pooled models (7)–(8) control for rater type. All outcome subcomponents are standardized to mean zero and unit variance using the control group calculated locally within the full active sample. Models (1), (2), and (4)-(8) incorporate firm fixed effects, with Models (2) and (4)-(8) also including sub-indicator fixed effects. Model (3) incorporates firm $\times$ sub-indicator fixed effects. Models are weighted by $w_i=1/n_i$, where $n_i$ is the number of ratings individual $i$ receives so that each individual carries equal total weight. $^* p<0.10$, $^{{**}} p<0.05$, $^{{***}} p<0.01$." + "\n"
    latex += r"\end{tablenotes}" + "\n" + r"\end{threeparttable}" + "\n" + r"\end{sidewaystable}"
    
    push_to_github("10dayD_combined_main.tex", latex, github_pat, GITHUB_CONFIG)

if __name__ == '__main__':
    render()
