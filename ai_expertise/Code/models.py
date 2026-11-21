# ==============================================================================
# models.py
# ------------------------------------------------------------------------------
# Core Econometric Modeling & Regression Estimation Engine
# "Artificial Intelligence in High-Skill Knowledge Work: Evidence from Patent
#  Drafting and Prosecution"
# ------------------------------------------------------------------------------
# Role & Architecture:
#   1. Estimates Ordinary Least Squares (OLS) and Weighted Least Squares (WLS)
#      regression specifications.
#   2. Implements firm fixed effects and subcomponent fixed effects with explicit
#      reference level handling to prevent multicollinearity.
#   3. Computes heteroskedasticity-robust (HC1) standard errors for subject-level
#      models and clusters standard errors at the practitioner level (email ID)
#      for multi-rating stacked panel regressions.
#   4. Evaluates linear hypothesis tests (F-tests / t-tests) across seniority strata
#      (e.g., testing H0: beta_junior = beta_senior, alpha_junior < alpha_senior).
#   5. Serializes model estimates into structured records consumed by analysis.py.
# ==============================================================================

import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from config import (
    TREATMENT_VAR, FIRM_VAR, UNIQUE_ID_VAR,
    MAIN_OUTCOME_DIMS, SE_TYPE_SUBJECT, SE_NOTE_SUBJECT,
    SE_NOTE_RATING, WLS_NOTE_RATING, SECONDARY_OUTCOMES
)
from utils import fv, stars, get_var_label, push_to_github, get_f_test_results

class DroppedTreatment(patsy.contrasts.Treatment):
    """Treatment contrast that forces dropping reference level even when -1 (no constant) is in formula."""
    def code_with_intercept(self, levels):
        return self.code_without_intercept(levels)

# Placeholder hooks for serialization callback injection from analysis.py
build_main_latex = lambda *args, **kwargs: None
build_combined_main_latex = lambda *args, **kwargs: None
build_speed_latex = lambda *args, **kwargs: None
build_secondary_latex = lambda *args, **kwargs: None

def run_reg(df, formula, hc_type=None, cluster_col=None, wls_weight=None):
    """
    Fits an OLS or WLS regression model using statsmodels formula API.

    Parameters:
        df (pd.DataFrame): Dataframe containing model covariates and outcome.
        formula (str): Patsy formula string specification.
        hc_type (str, optional): Robust standard error type (e.g. 'HC1', 'HC3').
        cluster_col (str, optional): Column name for clustered standard error grouping.
        wls_weight (str, optional): Column name for inverse-variance WLS weights.

    Returns:
        RegressionResultsWrapper: Fitted statsmodels regression results object.
    """
    if wls_weight:
        if wls_weight not in df.columns:
            raise ValueError(f"WLS weight column '{wls_weight}' specified but not found in dataframe.")
        mod = smf.wls(formula, data=df, weights=df[wls_weight])
    else:
        mod = smf.ols(formula, data=df)

    if cluster_col and cluster_col in df.columns:
        groups = df.loc[mod.data.row_labels, cluster_col]
        res = mod.fit(cov_type='cluster', cov_kwds={'groups': groups})
    else:
        res = mod.fit(cov_type=hc_type) if hc_type else mod.fit()

    _ = res.rsquared

    if UNIQUE_ID_VAR in df.columns:
        res.included_ids = df.loc[mod.data.row_labels, UNIQUE_ID_VAR].unique().tolist()
    else:
        res.included_ids = []

    return res

def build_f_test_rows(models, split_indices, num_cols, table_name, master_macros_dict, reverse_all=False, reverse_cols=None):
    """
    Constructs formatted LaTeX rows for linear hypothesis tests across model columns.

    Hypotheses Evaluated (Default / Unreversed):
        - H1: H0: alpha_junior < alpha_senior (Testing baseline quality gap between junior and senior control groups)
        - H2: H0: beta_junior < beta_senior   (Testing whether treatment benefits juniors more than seniors)
        - H3: H0: alpha_junior + beta_junior < alpha_senior (Testing if treated juniors surpass untreated seniors)
        - H4: H0: alpha_junior + beta_junior < alpha_senior + beta_senior (Testing if treated seniors surpass treated juniors)

    Parameters:
        models (dict or list): Dictionary or list of fitted model result objects.
        split_indices (list of int): 1-indexed column indices corresponding to split specifications.
        num_cols (int): Total number of columns in the LaTeX table.
        table_name (str): Identifier for macro dictionary storage.
        master_macros_dict (dict): Target dictionary to populate with test p-values.
        reverse_all (bool): If True, inverts directional inequality hypothesis signs.
        reverse_cols (list of int): Specific column indices where inequalities should be reversed.

    Returns:
        str: Formatted LaTeX snippet containing the hypothesis test rows.
    """
    if reverse_cols is None: reverse_cols = []
    
    if reverse_all:
        f_tests_info = [
            ('h1', "junior = senior", r"$H_0: \alpha_{\text{junior}} > \alpha_{\text{senior}}$"),
            ('h2', "treat_x_junior = treat_x_senior", r"$H_0: \beta_{\text{junior}} > \beta_{\text{senior}}$"),
            ('h3', "junior + treat_x_junior = senior", r"$H_0: \alpha_{\text{junior}} + \beta_{\text{junior}} > \alpha_{\text{senior}}$"),
            ('h4', "junior + treat_x_junior = senior + treat_x_senior", r"$H_0: \alpha_{\text{junior}} + \beta_{\text{junior}} > \alpha_{\text{senior}} + \beta_{\text{senior}}$")
        ]
    else:
        f_tests_info = [
            ('h1', "junior = senior", r"$H_0: \alpha_{\text{junior}} < \alpha_{\text{senior}}$"),
            ('h2', "treat_x_junior = treat_x_senior", r"$H_0: \beta_{\text{junior}} < \beta_{\text{senior}}$"),
            ('h3', "junior + treat_x_junior = senior", r"$H_0: \alpha_{\text{junior}} + \beta_{\text{junior}} < \alpha_{\text{senior}}$"),
            ('h4', "junior + treat_x_junior = senior + treat_x_senior", r"$H_0: \alpha_{\text{junior}} + \beta_{\text{junior}} < \alpha_{\text{senior}} + \beta_{\text{senior}}$")
        ]

    latex = ""
    for f_idx, (h_key, test_str, label_str_h) in enumerate(f_tests_info, start=1):
        row_str = f"    {label_str_h}"
        for m_idx in range(1, num_cols + 1):
            if isinstance(models, dict):
                res = models.get(m_idx)
            elif isinstance(models, (list, tuple)):
                res = models[m_idx] if m_idx < len(models) else None
            else:
                res = None
            if res and m_idx in split_indices:
                calc_label_str_h = label_str_h
                is_reversed_col = (m_idx in reverse_cols and not reverse_all)
                if is_reversed_col:
                    if r'\geq' in calc_label_str_h: calc_label_str_h = calc_label_str_h.replace(r'\geq', r'\leq')
                    elif r'\leq' in calc_label_str_h: calc_label_str_h = calc_label_str_h.replace(r'\leq', r'\geq')
                    elif '<' in calc_label_str_h: calc_label_str_h = calc_label_str_h.replace('<', '>')
                    elif '>' in calc_label_str_h: calc_label_str_h = calc_label_str_h.replace('>', '<')
                
                f_val, p_val = get_f_test_results(res, test_str, calc_label_str_h)
                if p_val is not None:
                    s = stars(p_val)
                    s_fmt = f"$^{{{s}}}$" if s else ""
                    val_str = f"{fv(p_val)}{s_fmt}"
                    if is_reversed_col:
                        val_str = f"[{val_str}]"
                    row_str += f" & {val_str}"
                    master_macros_dict[f"{table_name}_F{f_idx}_m{m_idx}"] = fv(p_val)
                else: row_str += " & "
            else: row_str += " & "
        latex += row_str + r" \\" + "\n"
    latex += r"    " + " &".join([""] * (num_cols + 1)) + r" \\[-1ex]" + "\n"
    return latex

def get_target_cols(df, task, rater, level):
    """
    Extracts standardized outcome column names matching the requested task,
    rater modality, and aggregation level.

    Parameters:
        df (pd.DataFrame): Dataframe containing candidate columns.
        task (str): Task identifier ('10dayD', '90dayD', '90dayR').
        rater (str): Evaluator modality ('human', 'llm', 'pooled').
        level (str): Aggregation level ('ind' for rater-level, 'sub' for subject-level).

    Returns:
        list of str: Matching column names.
    """
    cols = []
    task_map = {'10dayD': 'tt1', '90dayD': 'tt2_rat_drades', '90dayR': 'tt2_rat_cri'}
    task_search = task_map.get(task, task)
    for d in MAIN_OUTCOME_DIMS:
        matched = [c for c in df.columns if task_search in c and f"_{d}_" in c and rater in c and level in c]
        std_matched = [c for c in matched if 'std' in c]
        if std_matched: matched = std_matched
        if matched: cols.append(matched[0])
    return cols

# ==============================================================================
# 1. MAIN EFFECTS REGRESSIONS
# ==============================================================================

def run_main_effects(df_subject_clean, df_raw_clean, global_ref_firm, global_largest_firm, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled):
    """
    Estimates primary treatment effect regressions across tasks, raters, and
    aggregation levels (subject-level averages and disaggregated rating-level panels).
    """
    print("\n--- Generating Original Main Effects Tables ---")
    TASKS = ['10dayD', '90dayD', '90dayR']
    RATERS = ['human', 'llm', 'pooled']
    LEVELS = [('ind', df_raw_clean), ('sub', df_subject_clean)]

    for style in TASKS:
        for rater in RATERS:
            for level_name, df_base in LEVELS:
                df_use = df_base.copy()
                if level_name == 'ind' and rater != 'pooled' and 'Rater_Type' in df_use.columns:
                    df_use = df_use[df_use['Rater_Type'].str.lower() == rater]

                cols = get_target_cols(df_use, style, rater, level_name)
                if not cols: continue

                w_prefix = 'tt1' if '10' in style else 'tt2'
                w_suffix = 'drades' if 'D' in style else 'cri'
                w_col_candidates = [c for c in df_use.columns if 'wls_weight' in c and w_prefix in c and w_suffix in c and rater in c]
                w_use = w_col_candidates[0] if (w_col_candidates and level_name == 'ind') else None

                df_valid = df_use.dropna(subset=cols, how='all').copy()
                if df_valid.empty: continue

                df_valid['avg_score'] = df_valid[cols].mean(axis=1)
                c_col = UNIQUE_ID_VAR if level_name == 'ind' else None
                hc_m = None if level_name == 'ind' else SE_TYPE_SUBJECT

                # Subject-level average specification (Model 1)
                m1 = run_reg(df_valid, f"avg_score ~ {TREATMENT_VAR} + C({FIRM_VAR}, Treatment(reference={global_ref_firm}))", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)

                # Disaggregated rating-level stacked panel specifications
                id_vars = [c for c in df_valid.columns if c not in cols]
                df_melt = df_valid.melt(id_vars=id_vars, value_vars=cols, var_name='subcomponent', value_name='score').dropna(subset=['score'])
                df_melt['firm_sub'] = df_melt[FIRM_VAR].astype(str) + "_" + df_melt['subcomponent'].astype(str)

                log_cov = 'log_exp_uncen' if style in ['10dayD', '90dayD'] else 'log_exp_cen'
                df_melt['treatment_log_exp'] = df_melt[TREATMENT_VAR] * df_melt[log_cov]

                sub_vars = df_melt['subcomponent'].unique().tolist()
                cla_matches = [s for s in sub_vars if 'cla' in s.lower()]
                ref_sub = cla_matches[0] if cla_matches else sub_vars[-1]
                ref_firm_sub = f"'{global_largest_firm}_{ref_sub}'"

                f_fe = f"C({FIRM_VAR}, Treatment(reference={global_ref_firm})) + C(subcomponent, Treatment(reference='{ref_sub}'))"
                f_fe_int = f"C(firm_sub, Treatment(reference={ref_firm_sub}))"
                f_split = f"junior + senior + treat_x_junior + treat_x_senior - 1"

                m2 = run_reg(df_melt, f"score ~ {TREATMENT_VAR} + {f_fe}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                m3 = run_reg(df_melt, f"score ~ {TREATMENT_VAR} + {f_fe_int}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)

                if style in ['10dayD', '90dayD']:
                    m5 = run_reg(df_melt, f"score ~ {f_split} + {f_fe}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                    m6 = run_reg(df_melt, f"score ~ {f_split} + {f_fe_int}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                    build_main_latex(f"{style}_{level_name}_{rater}", style, [None, m1, m2, m3, m5, m6], rater, level_name, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled)
                else:
                    m5 = run_reg(df_melt, f"score ~ {f_split} + {f_fe}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                    build_main_latex(f"{style}_{level_name}_{rater}", style, [None, m1, m2, m3, m5], rater, level_name, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled)

                    if 'cheating' in df_valid.columns:
                        df_mc = df_melt[df_melt['cheating'] == 0].copy()
                        m2_c = run_reg(df_mc, f"score ~ {TREATMENT_VAR} + {f_fe}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                        m5_c = run_reg(df_mc, f"score ~ {f_split} + {f_fe}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                        
                        if level_name == 'ind' and rater in ['human', 'llm']:
                            m2_ctrl = run_reg(df_melt, f"score ~ {TREATMENT_VAR} + cheating + {f_fe}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                            m5_ctrl = run_reg(df_melt, f"score ~ {f_split} + cheating + {f_fe}", hc_type=hc_m, cluster_col=c_col, wls_weight=w_use)
                            build_main_latex(f"{style}_cheating_{level_name}_{rater}", "90dayR_cheating", [None, m2, m5, m2_ctrl, m5_ctrl, m2_c, m5_c], rater, level_name, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled)

# ==============================================================================
# 2. COMBINED MAIN EFFECTS REGRESSIONS (HUMAN, LLM, POOLED)
# ==============================================================================

def run_combined_main_effects(df_subject_clean, df_raw_clean, global_ref_firm, global_largest_firm, master_macros_dict, master_inclusion_dict, github_pat, config):
    """
    Estimates combined multi-column regression tables pooling Human, LLM, and
    Pooled ratings across all specifications.
    Relevant Tables:
      - Table 3: tab_10dayD_combined_main.py (10-Day Drafting)
      - Table 4: tab_90dayD_combined_main.py (90-Day Drafting)
      - Table 6: tab_90dayR_combined_main.py (90-Day Redlining)
    """
    print("\n--- Generating Combined Main Effects Tables ---")
    TASKS = ['10dayD', '90dayD', '90dayR']
    for style in TASKS:
        cols_sub_human = get_target_cols(df_subject_clean, style, 'human', 'sub')
        if not cols_sub_human: continue
        df_s = df_subject_clean.dropna(subset=cols_sub_human, how='all').copy()
        df_s['avg_score'] = df_s[cols_sub_human].mean(axis=1)

        f_firm_base = f"C({FIRM_VAR}, Treatment(reference={global_ref_firm}))"
        m1_comb = run_reg(df_s, f"avg_score ~ {TREATMENT_VAR} + {f_firm_base}", hc_type=SE_TYPE_SUBJECT)

        cols_ind_pooled = get_target_cols(df_raw_clean, style, 'pooled', 'ind')
        cols_ind_human = get_target_cols(df_raw_clean, style, 'human', 'ind')
        cols_ind_llm = get_target_cols(df_raw_clean, style, 'llm', 'ind')
        
        if not cols_ind_pooled or not cols_ind_human or not cols_ind_llm: continue
        
        def prepare_melted_df(cols, is_llm_val):
            d = df_raw_clean.dropna(subset=cols, how='all').copy()
            id_vars = [c for c in d.columns if c not in cols]
            melted = d.melt(id_vars=id_vars, value_vars=cols, var_name='subcomponent', value_name='score').dropna(subset=['score'])
            melted['firm_sub'] = melted[FIRM_VAR].astype(str) + "_" + melted['subcomponent'].astype(str)
            melted['is_llm'] = is_llm_val
            return melted

        df_human_all = prepare_melted_df(cols_ind_human, 0)
        df_llm_all = prepare_melted_df(cols_ind_llm, 1)
        df_pooled_all = prepare_melted_df(cols_ind_pooled, 0)
        if 'Rater_Type' in df_pooled_all.columns:
            df_pooled_all['is_llm'] = (df_pooled_all['Rater_Type'].str.lower() == 'llm').astype(int)

        df_human = df_human_all[(df_human_all['Rater_Type'].str.lower() == 'human') if 'Rater_Type' in df_human_all.columns else True].copy()
        df_llm = df_llm_all[(df_llm_all['Rater_Type'].str.lower() == 'llm') if 'Rater_Type' in df_llm_all.columns else True].copy()
        df_pooled = df_pooled_all.copy()
        
        sub_vars = df_pooled['subcomponent'].unique().tolist()
        cla_matches = [s for s in sub_vars if 'cla' in s.lower()]
        ref_sub = cla_matches[0] if cla_matches else sub_vars[-1]
        ref_firm_sub = f"'{global_largest_firm}_{ref_sub}'"

        sub_vars_human = df_human['subcomponent'].unique().tolist()
        cla_matches_h = [s for s in sub_vars_human if 'cla' in s.lower()]
        ref_sub_h = cla_matches_h[0] if cla_matches_h else sub_vars_human[-1]
        f_sub_base_h = f"C(subcomponent, Treatment(reference='{ref_sub_h}'))"
        f_firm_sub_base_h = f"C(firm_sub, Treatment(reference='{global_largest_firm}_{ref_sub_h}'))"
        
        sub_vars_llm = df_llm['subcomponent'].unique().tolist()
        cla_matches_l = [s for s in sub_vars_llm if 'cla' in s.lower()]
        ref_sub_l = cla_matches_l[0] if cla_matches_l else sub_vars_llm[-1]
        f_sub_base_l = f"C(subcomponent, Treatment(reference='{ref_sub_l}'))"

        f_sub_base_p = f"C(subcomponent, Treatment(reference='{ref_sub}'))"
        f_split = "junior + senior + treat_x_junior + treat_x_senior - 1"

        task_key = f"{'tt1' if '10' in style else 'tt2'}_{'drades' if 'D' in style else 'cri'}"

        w_h_str = f"wls_weight_{task_key}_human"
        w_l_str = f"wls_weight_{task_key}_llm"
        w_p_str = f"wls_weight_{task_key}_pooled"

        w_h_use = w_h_str if w_h_str in df_human.columns else None
        w_l_use = w_l_str if w_l_str in df_llm.columns else None
        w_p_use = w_p_str if w_p_str in df_pooled.columns else None

        m2_comb = run_reg(df_human, f"score ~ {TREATMENT_VAR} + {f_firm_base} + {f_sub_base_h}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
        m3_comb = run_reg(df_human, f"score ~ {TREATMENT_VAR} + {f_firm_sub_base_h}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
        m4_comb = run_reg(df_human, f"score ~ {f_split} + {f_firm_base} + {f_sub_base_h}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)

        m5_comb = run_reg(df_llm, f"score ~ {TREATMENT_VAR} + {f_firm_base} + {f_sub_base_l}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_l_use)
        m6_comb = run_reg(df_llm, f"score ~ {f_split} + {f_firm_base} + {f_sub_base_l}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_l_use)

        m7_comb = run_reg(df_pooled, f"score ~ {TREATMENT_VAR} + is_llm + {f_firm_base} + {f_sub_base_p}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_p_use)
        m8_comb = run_reg(df_pooled, f"score ~ {f_split} + is_llm + {f_firm_base} + {f_sub_base_p}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_p_use)

        combined_models = [None, m1_comb, m2_comb, m3_comb, m4_comb, m5_comb, m6_comb, m7_comb, m8_comb]
        build_combined_main_latex(f"{style}_combined_main", style, combined_models, master_macros_dict, master_inclusion_dict, github_pat, config)

def run_combined_main_effects_noFFE(df_subject_clean, df_raw_clean, global_ref_firm, global_largest_firm, master_macros_dict, master_inclusion_dict, github_pat, config):
    """
    Estimates combined main effects pooling LLM and Human raters without firm fixed effects.
    Relevant Tables:
      - Table 3 Robustness: tab_10dayD_combined_main_noFFE.py
      - Table 4 Robustness: tab_90dayD_combined_main_noFFE.py
      - Table 6 Robustness: tab_90dayR_combined_main_noFFE.py
    """
    print("\n--- Generating Combined Main Effects Tables without FFE ---")
    TASKS = ['10dayD', '90dayD', '90dayR']
    for style in TASKS:
        cols_sub_human = get_target_cols(df_subject_clean, style, 'human', 'sub')
        if not cols_sub_human: continue
        df_s = df_subject_clean.dropna(subset=cols_sub_human, how='all').copy()
        df_s['avg_score'] = df_s[cols_sub_human].mean(axis=1)

        m1_comb = run_reg(df_s, f"avg_score ~ {TREATMENT_VAR}", hc_type=SE_TYPE_SUBJECT)

        cols_ind_pooled = get_target_cols(df_raw_clean, style, 'pooled', 'ind')
        cols_ind_human = get_target_cols(df_raw_clean, style, 'human', 'ind')
        cols_ind_llm = get_target_cols(df_raw_clean, style, 'llm', 'ind')
        
        if not cols_ind_pooled or not cols_ind_human or not cols_ind_llm: continue
        
        def prepare_melted_df(cols, is_llm_val):
            d = df_raw_clean.dropna(subset=cols, how='all').copy()
            id_vars = [c for c in d.columns if c not in cols]
            melted = d.melt(id_vars=id_vars, value_vars=cols, var_name='subcomponent', value_name='score').dropna(subset=['score'])
            melted['firm_sub'] = melted[FIRM_VAR].astype(str) + "_" + melted['subcomponent'].astype(str)
            melted['is_llm'] = is_llm_val
            return melted

        df_human_all = prepare_melted_df(cols_ind_human, 0)
        df_llm_all = prepare_melted_df(cols_ind_llm, 1)
        df_pooled_all = prepare_melted_df(cols_ind_pooled, 0)
        if 'Rater_Type' in df_pooled_all.columns:
            df_pooled_all['is_llm'] = (df_pooled_all['Rater_Type'].str.lower() == 'llm').astype(int)

        df_human = df_human_all[(df_human_all['Rater_Type'].str.lower() == 'human') if 'Rater_Type' in df_human_all.columns else True].copy()
        df_llm = df_llm_all[(df_llm_all['Rater_Type'].str.lower() == 'llm') if 'Rater_Type' in df_llm_all.columns else True].copy()
        df_pooled = df_pooled_all.copy()
        
        sub_vars_human = df_human['subcomponent'].unique().tolist()
        cla_matches_h = [s for s in sub_vars_human if 'cla' in s.lower()]
        ref_sub_h = cla_matches_h[0] if cla_matches_h else sub_vars_human[-1]
        f_sub_base_h = f"C(subcomponent, Treatment(reference='{ref_sub_h}'))"
        f_sub_base_h_split = f"C(subcomponent, DroppedTreatment(reference='{ref_sub_h}'))"
        
        sub_vars_llm = df_llm['subcomponent'].unique().tolist()
        cla_matches_l = [s for s in sub_vars_llm if 'cla' in s.lower()]
        ref_sub_l = cla_matches_l[0] if cla_matches_l else sub_vars_llm[-1]
        f_sub_base_l = f"C(subcomponent, Treatment(reference='{ref_sub_l}'))"
        f_sub_base_l_split = f"C(subcomponent, DroppedTreatment(reference='{ref_sub_l}'))"

        sub_vars = df_pooled['subcomponent'].unique().tolist()
        cla_matches = [s for s in sub_vars if 'cla' in s.lower()]
        ref_sub = cla_matches[0] if cla_matches else sub_vars[-1]
        f_sub_base_p = f"C(subcomponent, Treatment(reference='{ref_sub}'))"
        f_sub_base_p_split = f"C(subcomponent, DroppedTreatment(reference='{ref_sub}'))"
        
        f_split = "junior + senior + treat_x_junior + treat_x_senior - 1"

        task_key = f"{'tt1' if '10' in style else 'tt2'}_{'drades' if 'D' in style else 'cri'}"
        w_h_str = f"wls_weight_{task_key}_human"
        w_l_str = f"wls_weight_{task_key}_llm"
        w_p_str = f"wls_weight_{task_key}_pooled"

        w_h_use = w_h_str if w_h_str in df_human.columns else None
        w_l_use = w_l_str if w_l_str in df_llm.columns else None
        w_p_use = w_p_str if w_p_str in df_pooled.columns else None

        m2_comb = run_reg(df_human, f"score ~ {TREATMENT_VAR} + {f_sub_base_h}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
        m3_comb = run_reg(df_human, f"score ~ {TREATMENT_VAR}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
        m4_comb = run_reg(df_human, f"score ~ {f_split} + {f_sub_base_h_split}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)

        m5_comb = run_reg(df_llm, f"score ~ {TREATMENT_VAR} + {f_sub_base_l}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_l_use)
        m6_comb = run_reg(df_llm, f"score ~ {f_split} + {f_sub_base_l_split}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_l_use)

        m7_comb = run_reg(df_pooled, f"score ~ {TREATMENT_VAR} + is_llm + {f_sub_base_p}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_p_use)
        m8_comb = run_reg(df_pooled, f"score ~ {f_split} + is_llm + {f_sub_base_p_split}", cluster_col=UNIQUE_ID_VAR, wls_weight=w_p_use)

        combined_models = [None, m1_comb, m2_comb, m3_comb, m4_comb, m5_comb, m6_comb, m7_comb, m8_comb]
        build_combined_main_latex(f"{style}_combined_main_noFFE", style, combined_models, master_macros_dict, master_inclusion_dict, github_pat, config)

def get_sec_target(base, df):
    """
    Locates standardized secondary variable names in the subject dataframe.
    """
    if base in ["pat_num", "pat_avg_len"]:
        return base if base in df.columns else None

    if f"{base}_std_sub" in df.columns: return f"{base}_std_sub"
    matched = [c for c in df.columns if base in c and 'std' in c]
    if matched: return matched[0]
    return base if base in df.columns else None

# ==============================================================================
# 3. SECONDARY OUTCOMES REGRESSIONS (SURVEYS, TIME ON TASK, PATENTS)
# ==============================================================================

def run_secondary_effects(df_subject_clean, df_cb, global_ref_firm, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled):
    """
    Estimates treatment effects on secondary administrative and survey outcomes.
    Relevant Tables:
      - Table 8: tab_TimeOnTask.py (Time spent on tasks)
      - Table 9: tab_Patent.py (On-the-job patent metrics)
      - Table A8: tab_DraftingSurvey.py (Speed, quality, satisfaction perceptions)
    """
    print("\n--- Generating Secondary Tables ---")
    df_sec = df_subject_clean.copy()
    f_firm_s = f"C({FIRM_VAR}, Treatment(reference={global_ref_firm}))"
    df_sec['treatment_log_exp'] = df_sec[TREATMENT_VAR] * df_sec['log_exp_cen']

    def get_canonical_data(df, col_name):
        if 'tt1' in col_name:
            ref_cols = get_target_cols(df, '10dayD', 'human', 'sub')
        elif 'tt2' in col_name and 'cri' in col_name:
            ref_cols = get_target_cols(df, '90dayR', 'human', 'sub')
        elif 'tt2' in col_name and 'dra' in col_name:
            ref_cols = get_target_cols(df, '90dayD', 'human', 'sub')
        else:
            ref_cols = None
        
        if ref_cols:
            d = df.dropna(subset=ref_cols, how='all').copy()
        else:
            d = df.copy()
            
        d = d.dropna(subset=[col_name]).copy()
        return d

    speed_vars = ['tt1_sum_dra_tot', 'tt1_sur_dra_spe', 'tt2_sum_dra_tot', 'tt2_sur_dra_spe', 'tt2_sum_cri_tot']
    speed_mods = [None]
    for v in speed_vars:
        col = get_sec_target(v, df_sec)
        if col and col in df_sec.columns:
            d = get_canonical_data(df_sec, col)
            if not d.empty:
                speed_mods.extend([
                    run_reg(d, f"{col} ~ {TREATMENT_VAR} + {f_firm_s}", hc_type=SE_TYPE_SUBJECT),
                    run_reg(d, f"{col} ~ junior + senior + treat_x_junior + treat_x_senior - 1 + {f_firm_s}", hc_type=SE_TYPE_SUBJECT)
                ])
                continue
        speed_mods.extend([None, None, None])

    build_speed_latex(speed_mods, df_cb, rater=None, level="sub", master_macros_dict=master_macros_dict, master_inclusion_dict=master_inclusion_dict, github_pat=github_pat, config=config, std_use_pooled=std_use_pooled)

    for key, config_sec in SECONDARY_OUTCOMES.items():
        s_mods = []
        s_mods_noFFE = []
        for v in config_sec['vars']:
            col = get_sec_target(v, df_sec)
            if key in ["TimeOnTask"]:
                col = v
            if col and col in df_sec.columns:
                d = get_canonical_data(df_sec, col)
                if not d.empty:
                    s_mods.extend([
                        run_reg(d, f"{col} ~ {TREATMENT_VAR} + {f_firm_s}", hc_type=SE_TYPE_SUBJECT),
                        run_reg(d, f"{col} ~ junior + senior + treat_x_junior + treat_x_senior - 1 + {f_firm_s}", hc_type=SE_TYPE_SUBJECT)
                    ])
                    s_mods_noFFE.extend([
                        run_reg(d, f"{col} ~ {TREATMENT_VAR}", hc_type=SE_TYPE_SUBJECT),
                        run_reg(d, f"{col} ~ junior + senior + treat_x_junior + treat_x_senior - 1", hc_type=SE_TYPE_SUBJECT)
                    ])
                    continue
            s_mods.extend([None, None, None])
            s_mods_noFFE.extend([None, None, None])

        if any(m is not None for m in s_mods):
            build_secondary_latex(key, config_sec, [None] + s_mods, df_cb, rater=None, level="sub", master_macros_dict=master_macros_dict, master_inclusion_dict=master_inclusion_dict, github_pat=github_pat, config=config, std_use_pooled=std_use_pooled)
        if any(m is not None for m in s_mods_noFFE):
            build_secondary_latex(f"{key}_noFFE", config_sec, [None] + s_mods_noFFE, df_cb, rater=None, level="sub", master_macros_dict=master_macros_dict, master_inclusion_dict=master_inclusion_dict, github_pat=github_pat, config=config, std_use_pooled=std_use_pooled)