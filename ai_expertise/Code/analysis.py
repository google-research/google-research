# ==============================================================================
# analysis.py
# ------------------------------------------------------------------------------
# Core Statistical & Econometric Engine for InFlow Replication Pipeline
# "Artificial Intelligence in High-Skill Knowledge Work: Evidence from Patent
#  Drafting and Prosecution"
# ------------------------------------------------------------------------------
# Role & Architecture:
#   This module serves as the single centralized location for:
#     1. Data ingestion & cleaning (loading raw CSVs, standardizing column schemas)
#     2. Feature engineering (experience strata, interaction terms, inverse-variance WLS weights)
#     3. Econometric estimations (OLS/WLS regressions with firm fixed effects & clustered SEs)
#     4. Nonparametric inference (Fisher exact permutation tests, Kolmogorov-Smirnov CDF tests)
#     5. Summary statistics, balance tests, attrition models, and distributional variance tests
#     6. Serialization of all statistical structures into JSON files in Jsons/
#
# Inputs:
#   - data/maindata_2026-08-10.csv   (Practitioner-level experimental dataset, N=999)
#   - data/codebook_2026-08-10.csv   (Metadata variable dictionary and label mapping)
#   - data/TTs_raters_2026-08-10.csv (Multi-rater scoring records for human & LLM evaluators)
#
# Outputs:
#   - Jsons/balance.json             (Table A1: Covariate balance tests)
#   - Jsons/firm_summary.json        (Table 1: Firm attributes & within-firm completion)
#   - Jsons/attrition.json           (Table A2: Sample attrition linear probability models)
#   - Jsons/takeup_completion.json   (Experimental takeup & completion summary statistics)
#   - Jsons/cell_summary.json        (Table 2 & Table A5: Standard normal cell means/SDs)
#   - Jsons/models_data.json         (Tables 3, 4, 6, 8, 9, A6, A7, A8: Regression models)
#   - Jsons/fig_fisher.json          (Figure 4: Fisher exact permutation test distributions)
#   - Jsons/fig_cdf.json             (Empirical CDF data with 95% confidence bands)
#   - Jsons/fig_histograms.json      (Figures 2 & 3: Overlapping score histograms)
#   - Jsons/fig_forest.json          (Figure 1: Quality subelement forest plot regressions)
#   - Jsons/fig_sensitivity.json     (Figure A3: Experience cutoff sensitivity analysis)
#   - Jsons/fig_raters.json          (Figures A1 & A2: Inter-rater agreement scatter plots)
#   - Jsons/fig_human_llm_subscales.json (Subscale concordance across evaluator modalities)
#   - Jsons/fig_usage_alt.json       (Figure A4: 14-day daily moving average active user series)
#   - Jsons/additional_analysis.json (Tables 5, 7, A3, A4: Variance regressions & correlations)
# ==============================================================================

import os
import re
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ks_2samp
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW

from config import (
    GITHUB_CONFIG, UNIQUE_ID_VAR, FIRM_VAR, TREATMENT_VAR,
    TREATMENT_GROUP, EXPERIENCE_THRESHOLD, STD_USE_POOLED,
    MAIN_OUTCOME_DIMS, RATER_OUTCOMES, SECONDARY_OUTCOMES,
    VARREGS_INCLUDE_LOW_N_FIRMS, EXCLUDE_INDIVIDUALS,
    EXCLUSION_INDIVIDUALS_FILE, EXCLUDE_FIRMS, EXCLUSION_FIRMS_FILE
)
from utils import fetch_csv_with_fallback, load_exclusion_list, get_firm_exclusion_mappings

import models
from models import (
    run_main_effects, run_combined_main_effects,
    run_combined_main_effects_noFFE, run_secondary_effects, get_target_cols
)

# Ensure output directory exists
os.makedirs('Jsons', exist_ok=True)

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy scalar types and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

STD_NOTE = "the pooled sample mean and variance" if STD_USE_POOLED else "the control group mean and variance"
USE_WEIGHTS_IN_SUMMARY = False


# ==============================================================================
# 1. DATA INGESTION, CLEANING, AND PREPARATION
# ==============================================================================

def load_and_prep_data(github_pat=None):
    """
    Loads raw experimental datasets from data/ (or GitHub fallback), standardizes
    variable schemas, constructs experience strata and interaction terms,
    normalizes performance outcomes against control group baselines, and calculates
    inverse-variance WLS weights.

    Parameters:
        github_pat (str, optional): GitHub Personal Access Token for remote fallback.

    Returns:
        tuple: (df_subject, df_raw, df_cb, global_ref_firm, df_subject_unfiltered)
            - df_subject: Cleaned subject-level dataframe (N=999 randomized practitioners)
            - df_raw: Disaggregated rater-level evaluation records
            - df_cb: Variable dictionary and codebook schema
            - global_ref_firm: Mode firm string formatted as a LaTeX/formula reference
            - df_subject_unfiltered: Complete recruited subject dataset prior to inclusion filters
    """
    print("Loading datasets safely...")
    df_main = fetch_csv_with_fallback('maindata', github_pat, GITHUB_CONFIG)
    df_cb = fetch_csv_with_fallback('codebook', github_pat, GITHUB_CONFIG)
    df_raters = fetch_csv_with_fallback('TTs_raters', github_pat, GITHUB_CONFIG)
    
    if df_main.empty or df_cb.empty or df_raters.empty:
        raise RuntimeError("Failed to load required datasets ('maindata', 'codebook', or 'TTs_raters').")

    # Clean codebook and standardize column names
    df_cb['varname'] = df_cb['varname'].astype(str).str.strip()
    rn_map = dict(zip(df_cb['original'], df_cb['varname']))
    df_main.rename(columns=rn_map, inplace=True)
    df_raters.rename(columns=rn_map, inplace=True)

    if 'Rater ID' in df_raters.columns:
        df_raters.rename(columns={'Rater ID': 'Rater_ID'}, inplace=True)

    # Standardize practitioner identifiers and firm variables
    df_main[UNIQUE_ID_VAR] = df_main[UNIQUE_ID_VAR].astype(str).str.lower().str.strip()
    df_raters[UNIQUE_ID_VAR] = df_raters[UNIQUE_ID_VAR].astype(str).str.lower().str.strip()
    if FIRM_VAR in df_main.columns:
        df_main[FIRM_VAR] = df_main[FIRM_VAR].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    if FIRM_VAR in df_raters.columns:
        df_raters[FIRM_VAR] = df_raters[FIRM_VAR].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    # Apply sample exclusions (individuals and/or firms) right at the loading stage:
    # removing individuals who declined consent to publish based on their hashed UUIDs / firm IDs
    if EXCLUDE_INDIVIDUALS or EXCLUDE_FIRMS:
        excluded_ind_emails = load_exclusion_list(EXCLUSION_INDIVIDUALS_FILE) if EXCLUDE_INDIVIDUALS else set()
        excluded_firm_terms = load_exclusion_list(EXCLUSION_FIRMS_FILE) if EXCLUDE_FIRMS else set()
        firm_names, firm_nums, firm_emails, firm_doms = get_firm_exclusion_mappings(
            df_main, excluded_firm_terms, unique_id_var=UNIQUE_ID_VAR, firm_var=FIRM_VAR
        )
        all_excluded_emails = excluded_ind_emails | firm_emails

        if all_excluded_emails or firm_names:
            n_before_main = len(df_main)
            main_mask = pd.Series(True, index=df_main.index)
            if all_excluded_emails:
                main_mask &= ~df_main[UNIQUE_ID_VAR].isin(all_excluded_emails)
            if firm_names and 'firm_name' in df_main.columns:
                main_mask &= ~df_main['firm_name'].astype(str).str.strip().str.lower().isin(firm_names)
            df_main = df_main[main_mask].copy()
            print(f"  [EXCLUSIONS] Dropped {n_before_main - len(df_main)} practitioner rows from main dataset.")

            n_before_raters = len(df_raters)
            raters_mask = pd.Series(True, index=df_raters.index)
            if all_excluded_emails:
                raters_mask &= ~df_raters[UNIQUE_ID_VAR].isin(all_excluded_emails)
            fn_rater_col = next((c for c in df_raters.columns if 'firm' in c.lower() and 'name' in c.lower()), None)
            if firm_names and fn_rater_col:
                raters_mask &= ~df_raters[fn_rater_col].astype(str).str.strip().str.lower().isin(firm_names)
            df_raters = df_raters[raters_mask].copy()
            print(f"  [EXCLUSIONS] Dropped {n_before_raters - len(df_raters)} evaluation rows from raters dataset.")

    if 'Rater_ID' in df_raters.columns:
        df_raters['Rater_Name'] = df_raters['Rater_ID'].astype(str).str.lower().str.split('@').str[0]
    else:
        df_raters['Rater_Name'] = ''
    df_raters['Rater_Type'] = np.where(df_raters['Rater_Name'] == 'llm', 'llm', 'human')

    df_merged = pd.merge(df_main, df_raters, on=UNIQUE_ID_VAR, how='inner')
    df_subject = df_main.copy()
    df_raw = df_merged.copy()
    
    # Filter to included randomized subjects
    df_subject_unfiltered = df_subject.copy()
    df_subject = df_subject[df_subject['included'] == 1].copy()
    df_subject[TREATMENT_VAR] = (df_subject['group'] == TREATMENT_GROUP).astype(int)
    df_subject['exp'] = pd.to_numeric(df_subject['exp'], errors='coerce')
    df_subject.loc[df_subject['exp'] == 0, 'exp'] = 0.5
    df_subject['log_exp'] = np.log(df_subject['exp']) - np.log(df_subject['exp']).mean()
    df_subject['senior'] = (df_subject['exp'] >= EXPERIENCE_THRESHOLD).astype(int)
    df_subject['junior'] = (df_subject['senior'] == 0).astype(int)

    # Merge subject-level demographics into raw rater records
    subject_cols_to_merge = [UNIQUE_ID_VAR, FIRM_VAR, TREATMENT_VAR, 'exp', 'log_exp', 'senior', 'junior']
    overlap_cols = [c for c in subject_cols_to_merge if c in df_raw.columns and c != UNIQUE_ID_VAR]
    df_raw = df_raw.drop(columns=overlap_cols)

    df_raw = pd.merge(
        df_subject[subject_cols_to_merge],
        df_raw,
        on=UNIQUE_ID_VAR,
        how='inner'
    )

    # Compute composite drafting metrics (average of drafting & design components)
    for tp in ['tt1', 'tt2']:
        for dim in MAIN_OUTCOME_DIMS:
            dra_col = f'{tp}_rat_dra_{dim}'
            des_col = f'{tp}_rat_des_{dim}'
            for c in [dra_col, des_col]:
                if c in df_raw.columns:
                    df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
            drades_col = f'{tp}_rat_drades_{dim}'
            if dra_col in df_raw.columns and des_col in df_raw.columns:
                df_raw[drades_col] = df_raw[[dra_col, des_col]].mean(axis=1)

    for dim in MAIN_OUTCOME_DIMS:
        cri_col = f'tt2_rat_cri_{dim}'
        if cri_col in df_raw.columns:
            df_raw[cri_col] = pd.to_numeric(df_raw[cri_col], errors='coerce')

    # Outcome standardization against Control Group distribution (Mean 0, SD 1)
    main_vars = [f'tt1_rat_drades_{d}' for d in MAIN_OUTCOME_DIMS] + \
                [f'tt2_rat_drades_{d}' for d in MAIN_OUTCOME_DIMS] + \
                [f'tt2_rat_cri_{d}' for d in MAIN_OUTCOME_DIMS]

    rater_sources = {
        'pooled': df_raw.index,
        'human': df_raw[df_raw['Rater_Type'] == 'human'].index,
        'llm': df_raw[df_raw['Rater_Type'] == 'llm'].index
    }

    for var in main_vars:
        if var not in df_raw.columns: continue
        for source_name, source_idx in rater_sources.items():
            src_mask = df_raw.index.isin(source_idx)
            col_name_raw_ind = f"{var}_{source_name}_ind"
            df_raw.loc[src_mask, col_name_raw_ind] = df_raw.loc[src_mask, var]

            control_mask = src_mask & (df_raw[TREATMENT_VAR] == 0)
            mu_ctrl = df_raw.loc[control_mask, var].mean()
            sigma_ctrl = df_raw.loc[control_mask, var].std()

            col_name_std_ind = f"{var}_std_{source_name}_ind"
            if pd.notna(sigma_ctrl) and sigma_ctrl > 0:
                df_raw.loc[src_mask, col_name_std_ind] = (df_raw.loc[src_mask, var] - mu_ctrl) / sigma_ctrl
            else:
                df_raw.loc[src_mask, col_name_std_ind] = df_raw.loc[src_mask, var] - mu_ctrl

            subj_means_raw = df_raw[src_mask].groupby(UNIQUE_ID_VAR)[var].mean()
            col_name_raw_sub = f"{var}_{source_name}_sub"
            df_subject[col_name_raw_sub] = df_subject[UNIQUE_ID_VAR].map(subj_means_raw)

            subj_means_std = df_raw[src_mask].groupby(UNIQUE_ID_VAR)[col_name_std_ind].mean()
            col_name_std_sub = f"{var}_std_{source_name}_sub"
            df_subject[col_name_std_sub] = df_subject[UNIQUE_ID_VAR].map(subj_means_std)

    # Standardize secondary outcome variables (Survey, Time on Task, Patent metrics)
    sec_vars_to_process = []
    for group_info in SECONDARY_OUTCOMES.values():
        sec_vars_to_process.extend(group_info['vars'])
    sec_vars_to_process = list(set([v for v in sec_vars_to_process if v in df_subject.columns]))

    for v in sec_vars_to_process:
        df_subject[v] = pd.to_numeric(df_subject[v], errors='coerce')

    for var in sec_vars_to_process:
        control_series_sec = df_subject.loc[df_subject[TREATMENT_VAR] == 0, var]
        mu_sec = control_series_sec.mean()
        sigma_sec = control_series_sec.std()

        col_name = f"{var}_std_sub"
        if pd.notna(sigma_sec) and sigma_sec > 0:
            df_subject[col_name] = (df_subject[var] - mu_sec) / sigma_sec
        else:
            df_subject[col_name] = df_subject[var] - mu_sec

    # 5. WLS Weights
    WEIGHT_GROUPS = {
        'tt1_drades': [f'tt1_rat_drades_{d}' for d in MAIN_OUTCOME_DIMS],
        'tt2_drades': [f'tt2_rat_drades_{d}' for d in MAIN_OUTCOME_DIMS],
        'tt2_cri': [f'tt2_rat_cri_{d}' for d in MAIN_OUTCOME_DIMS]
    }
    df_raw = df_raw.copy()
    df_subject = df_subject.copy()

    for key, sub_vars in WEIGHT_GROUPS.items():
        valid_vars = [v for v in sub_vars if v in df_raw.columns]
        if not valid_vars: continue

        df_raw[f'n_ratings_{key}_pooled'] = df_raw.dropna(subset=[valid_vars[0]]).groupby(UNIQUE_ID_VAR)[valid_vars[0]].transform('count')
        df_raw[f'wls_weight_{key}_pooled'] = np.where(
            df_raw[f'n_ratings_{key}_pooled'] > 0, 1.0 / df_raw[f'n_ratings_{key}_pooled'], 0
        )

        for rater_type in ['human', 'llm']:
            mask = df_raw['Rater_Type'].str.lower() == rater_type
            df_raw.loc[mask, f'n_ratings_{key}_{rater_type}'] = df_raw[mask].dropna(subset=[valid_vars[0]]).groupby(UNIQUE_ID_VAR)[valid_vars[0]].transform('count')
            df_raw.loc[mask, f'wls_weight_{key}_{rater_type}'] = np.where(
                df_raw.loc[mask, f'n_ratings_{key}_{rater_type}'] > 0,
                1.0 / df_raw.loc[mask, f'n_ratings_{key}_{rater_type}'],
                0
            )

    df_raw = df_raw.copy()
    df_subject = df_subject.copy()

    df_raw['rater_type'] = df_raw['Rater_Type'].str.lower()
    email_cat = df_subject[UNIQUE_ID_VAR].astype('category')
    df_subject['person_id'] = email_cat.cat.codes
    df_subject[FIRM_VAR] = df_subject[FIRM_VAR].astype('category')
    df_raw['person_id'] = pd.Categorical(df_raw[UNIQUE_ID_VAR], categories=email_cat.cat.categories).codes
    df_raw[FIRM_VAR] = pd.Categorical(df_raw[FIRM_VAR], categories=df_subject[FIRM_VAR].cat.categories)
    
    # Construct interaction terms and log experience covariates
    if 'exp' in df_subject.columns:
        df_subject['log_exp_uncen'] = np.log(df_subject['exp'])
        df_subject['treat_x_senior'] = df_subject[TREATMENT_VAR] * df_subject['senior']
        df_subject['treat_x_junior'] = df_subject[TREATMENT_VAR] * df_subject['junior']
        
    if 'exp' in df_raw.columns:
        df_raw['log_exp_uncen'] = np.log(df_raw['exp'])
        df_raw['treat_x_senior'] = df_raw[TREATMENT_VAR] * df_raw['senior']
        df_raw['treat_x_junior'] = df_raw[TREATMENT_VAR] * df_raw['junior']

    # Cheating / prompt-leakage indicator
    if 'redline_suspected' in df_subject.columns:
        df_subject['cheating'] = pd.to_numeric(df_subject['redline_suspected'], errors='coerce').fillna(0)
    elif 'ps_aiused' in df_subject.columns and 'cheating' not in df_subject.columns:
        df_subject['cheating'] = pd.to_numeric(df_subject['ps_aiused'], errors='coerce').fillna(0)
    if 'cheating' in df_subject.columns:
        mapping = df_subject.set_index(UNIQUE_ID_VAR)['cheating'].to_dict()
        df_raw['cheating'] = pd.to_numeric(df_raw[UNIQUE_ID_VAR].map(mapping), errors='coerce').fillna(0)

    # De-fragment dataframes to guarantee high performance and avoid warnings
    df_subject = df_subject.copy()
    df_raw = df_raw.copy()

    if 'log_exp_uncen' in df_subject.columns:
        mean_exp = df_subject['log_exp_uncen'].mean()
        df_subject['log_exp_cen'] = df_subject['log_exp_uncen'] - mean_exp
        if 'log_exp_uncen' in df_raw.columns:
            df_raw['log_exp_cen'] = df_raw['log_exp_uncen'] - mean_exp

    global_largest_firm = df_subject[FIRM_VAR].value_counts().idxmax()
    global_ref_firm = f"'{global_largest_firm}'" if isinstance(global_largest_firm, str) else f"{global_largest_firm}"

    return df_subject, df_raw, df_cb, global_ref_firm, df_subject_unfiltered


def get_full_randomized(df_unf, df_clean):
    """
    Constructs the complete randomized cohort from unfiltered recruitment records,
    restricting to included participants (included==1) and merging key completion metrics.
    """
    df = df_unf.copy()
    df = df[df['included'] == 1].copy()
    
    df[TREATMENT_VAR] = (df['group'] == TREATMENT_GROUP).astype(int)
    df['exp'] = pd.to_numeric(df['exp'], errors='coerce')
    df.loc[df['exp'] == 0, 'exp'] = 0.5
    df['senior'] = (df['exp'] >= EXPERIENCE_THRESHOLD).astype(int)
    df['junior'] = (df['senior'] == 0).astype(int)
    
    cols_to_merge = [UNIQUE_ID_VAR, 'tt1_rat_drades_enf_pooled_sub', 'tt2_rat_drades_enf_pooled_sub', 'tt2_rat_cri_enf_pooled_sub', 'cheating']
    cols_to_merge = [c for c in cols_to_merge if c in df_clean.columns or c == UNIQUE_ID_VAR]
    
    df = df.merge(df_clean[cols_to_merge], on=UNIQUE_ID_VAR, how='left')
    return df


# ==============================================================================
# 2. BALANCE & ATTRITION ANALYSES
# ==============================================================================

def analyze_balance(df_subject):
    """
    Performs treatment-control covariate balance tests across baseline variables
    (response time, comprehension, confidence, experience, seniority).
    Outputs: Jsons/balance.json (Table A1: Balance Table).
    """
    vars_bal = ['responsetime', 'comprehension', 'confidence', 'exp', 'senior']
    var_names = {
        'responsetime': 'Time (days) for response after onboarding email',
        'comprehension': 'Understanding of study (Likert)',
        'confidence': 'Confidence in ability to complete tasks (Likert)',
        'exp': 'Years of experience',
        'senior': 'Seniority (Exp >= 7)'
    }

    df_bal = df_subject.copy()
    for v in vars_bal:
        df_bal[v] = pd.to_numeric(df_bal[v], errors='coerce')

    g1 = df_bal[df_bal[TREATMENT_VAR] == 0]
    g2 = df_bal[df_bal[TREATMENT_VAR] == 1]

    data = {'rows': []}
    macros = {}

    for v in vars_bal:
        d1, d2 = g1[v].dropna(), g2[v].dropna()
        if d1.empty or d2.empty: continue

        mean1, se1 = d1.mean(), d1.sem()
        mean2, se2 = d2.mean(), d2.sem()
        diff = mean2 - mean1
        se_diff = np.sqrt(se1**2 + se2**2)
        stat, p = stats.ttest_ind(d2, d1, equal_var=False)

        macros[f"balance_n_g1_{v}"] = f"{len(d1)}"
        macros[f"balance_mean_g1_{v}"] = f"{mean1:.2f}"
        macros[f"balance_se_g1_{v}"] = f"{se1:.2f}"
        macros[f"balance_n_g2_{v}"] = f"{len(d2)}"
        macros[f"balance_mean_g2_{v}"] = f"{mean2:.2f}"
        macros[f"balance_se_g2_{v}"] = f"{se2:.2f}"
        macros[f"balance_diff_{v}"] = f"{diff:.2f}"
        macros[f"balance_se_diff_{v}"] = f"{se_diff:.2f}"
        macros[f"balance_pvalue_{v}"] = f"{p:.2f}"

        data['rows'].append({
            'var_id': v,
            'var_name': var_names[v],
            'n1': len(d1),
            'mean1': mean1,
            'se1': se1,
            'n2': len(d2),
            'mean2': mean2,
            'se2': se2,
            'diff': diff,
            'se_diff': se_diff,
            'p': p
        })

    with open('Jsons/balance.json', 'w') as f:
        json.dump({'data': data, 'macros': macros}, f, cls=NpEncoder)
    return macros


def analyze_firm_summary(df_subject, df_subject_unfiltered):
    """
    Computes firm-level summary statistics, practitioner counts, experience means,
    and completion rates across control and treatment cohorts.
    Outputs: Jsons/firm_summary.json (Table 1: Firm Summary Table).
    """
    df_unf = df_subject_unfiltered.copy()
    df_unf = df_unf[df_unf[UNIQUE_ID_VAR].notna() & (df_unf[UNIQUE_ID_VAR].astype(str).str.strip() != '')]
    
    df_rand = get_full_randomized(df_unf, df_subject)
    
    col_tt1 = 'tt1_rat_drades_enf_pooled_sub'
    col_tt2 = 'tt2_rat_drades_enf_pooled_sub'
    col_tt2_cri = 'tt2_rat_cri_enf_pooled_sub'
    
    df_rand['retained_tt1'] = df_rand[col_tt1].notna().astype(int)
    df_rand['retained_tt2'] = df_rand[col_tt2].notna().astype(int)
    df_rand['retained_tt2_cri'] = df_rand[col_tt2_cri].notna().astype(int)
    
    df_rand['c_tt1_treat'] = df_rand['retained_tt1'] * df_rand[TREATMENT_VAR]
    df_rand['c_tt1_ctrl']  = df_rand['retained_tt1'] * (1 - df_rand[TREATMENT_VAR])
    df_rand['c_tt2_treat'] = df_rand['retained_tt2'] * df_rand[TREATMENT_VAR]
    df_rand['c_tt2_ctrl']  = df_rand['retained_tt2'] * (1 - df_rand[TREATMENT_VAR])
    df_rand['c_tt2_cri_treat'] = df_rand['retained_tt2_cri'] * df_rand[TREATMENT_VAR]
    df_rand['c_tt2_cri_ctrl']  = df_rand['retained_tt2_cri'] * (1 - df_rand[TREATMENT_VAR])

    firms_rec = df_unf.groupby(FIRM_VAR).size().rename('n_recruited')
    firms_rand = df_rand.groupby(FIRM_VAR).agg(
        n_randomized=('included', 'count'),
        firm_name=('firm_name', 'first'),
        n_treated=(TREATMENT_VAR, 'sum'),
        exp_avg=('exp', 'mean'),
        n_senior=('senior', 'sum'),
        c_tt1_treat=('c_tt1_treat', 'sum'),
        c_tt1_ctrl=('c_tt1_ctrl', 'sum'),
        c_tt2_treat=('c_tt2_treat', 'sum'),
        c_tt2_ctrl=('c_tt2_ctrl', 'sum'),
        c_tt2_cri_treat=('c_tt2_cri_treat', 'sum'),
        c_tt2_cri_ctrl=('c_tt2_cri_ctrl', 'sum'),
        n_cheating=('cheating', 'sum') if 'cheating' in df_rand.columns else ('included', lambda x: 0)
    )

    firms = pd.concat([firms_rec, firms_rand], axis=1).fillna(0).reset_index()
    firms = firms[firms['n_randomized'] > 0].copy()
    firms.sort_values('n_recruited', ascending=False, kind='stable', inplace=True)
    firms.reset_index(drop=True, inplace=True)

    macros = {}
    tot_rec = firms['n_recruited'].sum()
    tot_rand = firms['n_randomized'].sum()
    tot_treated = firms['n_treated'].sum()
    tot_ctrl = tot_rand - tot_treated
    tot_senior = firms['n_senior'].sum()
    
    tot_tt1_treat = firms['c_tt1_treat'].sum()
    tot_tt1_ctrl = firms['c_tt1_ctrl'].sum()
    tot_tt2_treat = firms['c_tt2_treat'].sum()
    tot_tt2_ctrl = firms['c_tt2_ctrl'].sum()
    tot_tt2_cri_treat = firms['c_tt2_cri_treat'].sum()
    tot_tt2_cri_ctrl = firms['c_tt2_cri_ctrl'].sum()
    tot_cheaters = firms['n_cheating'].sum() if 'n_cheating' in firms.columns else 0

    macros["firm_summary_tot_rec"] = f"{tot_rec:.0f}"
    macros["firm_summary_tot_rand_n"] = f"{tot_rand:.0f}"
    macros["firm_summary_tot_treated_n"] = f"{tot_treated:.0f}"
    macros["firm_summary_tot_senior"] = f"{tot_senior:.0f}"
    macros["firm_summary_tot_tt1_t_n"] = f"{tot_tt1_treat:.0f}"
    macros["firm_summary_tot_tt2_t_n"] = f"{tot_tt2_treat:.0f}"
    macros["firm_summary_tot_tt2_cri_t_n"] = f"{tot_tt2_cri_treat:.0f}"
    macros["firm_summary_tot_tt1_c_n"] = f"{tot_tt1_ctrl:.0f}"
    macros["firm_summary_tot_tt2_c_n"] = f"{tot_tt2_ctrl:.0f}"
    macros["firm_summary_tot_tt2_cri_c_n"] = f"{tot_tt2_cri_ctrl:.0f}"
    macros["firm_summary_tot_cheaters"] = f"{tot_cheaters:.0f}"

    firm_rows = firms.to_dict('records')
    exp_mean = df_rand['exp'].mean()

    out_data = {
        'firms': firm_rows,
        'totals': {
            'tot_rec': tot_rec,
            'tot_rand': tot_rand,
            'tot_treated': tot_treated,
            'tot_ctrl': tot_ctrl,
            'tot_senior': tot_senior,
            'exp_mean': exp_mean,
            'tot_tt1_ctrl': tot_tt1_ctrl,
            'tot_tt2_ctrl': tot_tt2_ctrl,
            'tot_tt2_cri_ctrl': tot_tt2_cri_ctrl,
            'tot_tt1_treat': tot_tt1_treat,
            'tot_tt2_treat': tot_tt2_treat,
            'tot_tt2_cri_treat': tot_tt2_cri_treat,
        },
        'macros': macros
    }

    with open('Jsons/firm_summary.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)
    return macros


def analyze_attrition(df_subject, df_subject_unfiltered):
    """
    Estimates linear probability models of task attrition on treatment assignment,
    seniority, comprehension, and confidence with HC3 robust standard errors.
    Outputs: Jsons/attrition.json (Table A2: Attrition Summary).
    """
    df_rand = get_full_randomized(df_subject_unfiltered, df_subject)

    df_rand['retained_tt1'] = df_rand['tt1_rat_drades_enf_pooled_sub'].notna().astype(int)
    df_rand['retained_tt2'] = df_rand['tt2_rat_drades_enf_pooled_sub'].notna().astype(int)
    df_rand['retained_tt2_cri'] = df_rand['tt2_rat_cri_enf_pooled_sub'].notna().astype(int)

    base_valid = df_rand.dropna(subset=['exp'])
    
    df_rand['onboarding_response'] = (df_rand['comprehension'].notna() & df_rand['confidence'].notna()).astype(int)
    df_rand['comprehension'] = df_rand['comprehension'].fillna(df_rand['comprehension'].mean())
    df_rand['confidence'] = df_rand['confidence'].fillna(df_rand['confidence'].mean())
    
    valid = df_rand.dropna(subset=[TREATMENT_VAR, 'senior', 'comprehension', 'confidence', 'onboarding_response']).copy()

    valid['attrition_10d'] = 1 - valid['retained_tt1']
    valid['attrition_90d'] = 1 - valid['retained_tt2']
    valid['attrition_90d_cri'] = 1 - valid['retained_tt2_cri']

    macros = {}
    g1_base, g2_base = len(base_valid[base_valid[TREATMENT_VAR]==0]), len(base_valid[base_valid[TREATMENT_VAR]==1])
    macros['attrition_count_Group1_baseline'] = f"{g1_base}"
    macros['attrition_count_Group2_baseline'] = f"{g2_base}"
    macros['attrition_count_Dropped_baseline'] = f"{len(df_rand) - len(base_valid)}"

    obs_counts = []
    out_models = {}

    for dep in ['attrition_10d', 'attrition_90d', 'attrition_90d_cri']:
        if dep == 'attrition_90d_cri':
            valid_model = valid[valid['firm_name'] != 'Firm 3']
        else:
            valid_model = valid
        
        obs_counts.append(len(valid_model))
        mod = smf.ols(f"{dep} ~ group_binary + senior + comprehension + confidence + onboarding_response", data=valid_model).fit(cov_type='HC3')
        macros[f"attrition_rsquared_{dep}"] = f"{mod.rsquared:.2f}"
        
        mod_data = {
            'params': mod.params.to_dict(),
            'pvalues': mod.pvalues.to_dict(),
            'bse': mod.bse.to_dict(),
            'fvalue': mod.fvalue,
            'f_pvalue': mod.f_pvalue
        }
        out_models[dep] = mod_data

        for v in ['Intercept', 'group_binary', 'senior', 'comprehension', 'confidence', 'onboarding_response']:
            m_name = 'const' if v == 'Intercept' else v
            macros[f"attrition_coefficient_{dep}_{m_name}"] = f"{mod.params[v]:.2f}"
            macros[f"attrition_pvalue_{dep}_{m_name}"] = f"{mod.pvalues[v]:.2f}"

    out_data = {
        'obs_counts': obs_counts,
        'models': out_models,
        'macros': macros
    }

    with open('Jsons/attrition.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)
    return macros


def analyze_takeup_completion(df_subject):
    """
    Computes overall takeup and task completion counts and percentages.
    Outputs: Jsons/takeup_completion.json.
    """
    df_tc = df_subject.copy()
    
    col_tt1 = 'tt1_rat_drades_enf_pooled_sub'
    col_tt2 = 'tt2_rat_cri_enf_pooled_sub'
    
    g1 = df_tc[df_tc[TREATMENT_VAR] == 0]
    g2 = df_tc[df_tc[TREATMENT_VAR] == 1]
    
    n_ctrl = len(g1)
    n_trt = len(g2)
    
    takeup_ctrl_mask = g1[col_tt1].notna()
    takeup_trt_mask = g2[col_tt1].notna()
    
    takeup_ctrl_n = int(takeup_ctrl_mask.sum())
    takeup_trt_n = int(takeup_trt_mask.sum())
    
    takeup_ctrl_pct = (takeup_ctrl_n / n_ctrl * 100) if n_ctrl else 0
    takeup_trt_pct = (takeup_trt_n / n_trt * 100) if n_trt else 0
    
    comp_ctrl_n = int(g1[col_tt2].notna().sum())
    comp_trt_n = int(g2[col_tt2].notna().sum())
    
    comp_ctrl_pct = (comp_ctrl_n / n_ctrl * 100) if n_ctrl else 0
    comp_trt_pct = (comp_trt_n / n_trt * 100) if n_trt else 0
    
    tot_n = n_ctrl + n_trt
    tot_takeup_n = takeup_ctrl_n + takeup_trt_n
    tot_takeup_pct = (tot_takeup_n / tot_n * 100) if tot_n else 0
    
    tot_comp_n = comp_ctrl_n + comp_trt_n
    tot_comp_pct = (tot_comp_n / tot_n * 100) if tot_n else 0
    
    macros = {
        "tc_n_ctrl": f"{n_ctrl}",
        "tc_n_trt": f"{n_trt}",
        "tc_takeup_ctrl_n": f"{takeup_ctrl_n}",
        "tc_takeup_trt_n": f"{takeup_trt_n}",
        "tc_comp_ctrl_n": f"{comp_ctrl_n}",
        "tc_comp_trt_n": f"{comp_trt_n}",
        "tc_n_tot": f"{tot_n}",
        "tc_takeup_tot_n": f"{tot_takeup_n}",
        "tc_comp_tot_n": f"{tot_comp_n}",
        "tc_takeup_ctrl_pct": f"{takeup_ctrl_pct:.1f}",
        "tc_takeup_trt_pct": f"{takeup_trt_pct:.1f}",
        "tc_takeup_tot_pct": f"{tot_takeup_pct:.1f}",
        "tc_comp_ctrl_pct": f"{comp_ctrl_pct:.1f}",
        "tc_comp_trt_pct": f"{comp_trt_pct:.1f}",
        "tc_comp_tot_pct": f"{tot_comp_pct:.1f}"
    }

    out_data = {
        'n_ctrl': n_ctrl, 'n_trt': n_trt, 'tot_n': tot_n,
        'takeup_ctrl_n': takeup_ctrl_n, 'takeup_ctrl_pct': takeup_ctrl_pct,
        'takeup_trt_n': takeup_trt_n, 'takeup_trt_pct': takeup_trt_pct,
        'tot_takeup_n': tot_takeup_n, 'tot_takeup_pct': tot_takeup_pct,
        'comp_ctrl_n': comp_ctrl_n, 'comp_ctrl_pct': comp_ctrl_pct,
        'comp_trt_n': comp_trt_n, 'comp_trt_pct': comp_trt_pct,
        'tot_comp_n': tot_comp_n, 'tot_comp_pct': tot_comp_pct,
        'macros': macros
    }

    with open('Jsons/takeup_completion.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)
    return macros


# ==============================================================================
# 3. CELL SUMMARY STATISTICS
# ==============================================================================

def analyze_summary_table(df_subject, df_raw):
    """
    Computes cell means and standard deviations across treatment, control,
    junior, and senior cells for Human, LLM, and Pooled ratings.
    Outputs: Jsons/cell_summary.json (Table 2 & Table A5: Cell Summaries).
    """
    df_raw = df_raw.copy()
    if 'included' not in df_raw.columns and 'included' in df_subject.columns:
        if UNIQUE_ID_VAR in df_raw.columns and UNIQUE_ID_VAR in df_subject.columns:
            df_raw['included'] = df_raw[UNIQUE_ID_VAR].map(df_subject.set_index(UNIQUE_ID_VAR)['included']).fillna(0)

    def get_mean_std(df_subset, val_col, weight_col=None):
        if df_subset.empty: return np.nan, np.nan
        df_valid = df_subset.dropna(subset=[val_col])
        if df_valid.empty: return np.nan, np.nan

        vals = df_valid[val_col].values
        if weight_col and weight_col in df_valid.columns:
            ws = df_valid[weight_col].values
            mask = (ws > 0) & (~np.isnan(ws))
            vals = vals[mask]
            ws = ws[mask]

            if len(vals) == 0: return np.nan, np.nan
            if len(vals) == 1: return vals[0], np.nan

            dsw = DescrStatsW(vals, weights=ws, ddof=1)
            return dsw.mean, dsw.std
        else:
            return vals.mean(), vals.std(ddof=1)

    subscale_names = [None, 'Enforceability', 'Technical Accuracy', 'Strategic Ambiguity', r'Completeness \& Alignment', 'Clarity']
    macros = {}
    out_data = {}

    for rater in ['pooled', 'human', 'llm']:
        for level in ['ind', 'sub']:
            suffix = f"_std_{rater}_{level}"
            if level == 'ind':
                df_main_out = df_raw[df_raw['included'] == 1].copy() if 'included' in df_raw.columns else df_raw.copy()
            else:
                df_main_out = df_subject[df_subject['included'] == 1].copy()

            tt1_cols = [f"{v}{suffix}" for v in RATER_OUTCOMES['tt1_drades'] if f"{v}{suffix}" in df_main_out.columns]
            tt2_cols = [f"{v}{suffix}" for v in RATER_OUTCOMES['tt2_drades'] if f"{v}{suffix}" in df_main_out.columns]
            cri_cols = [f"{v}{suffix}" for v in RATER_OUTCOMES['tt2_cri'] if f"{v}{suffix}" in df_main_out.columns]

            if tt1_cols: df_main_out[f'tt1_pooled{suffix}'] = df_main_out[tt1_cols].mean(axis=1)
            if tt2_cols: df_main_out[f'tt2_pooled{suffix}'] = df_main_out[tt2_cols].mean(axis=1)
            if cri_cols: df_main_out[f'tt2_cri_pooled{suffix}'] = df_main_out[cri_cols].mean(axis=1)

            summary_sections = [
                ('A. Main Outcomes: 10-Day Drafting', [f'tt1_pooled{suffix}'] + tt1_cols, subscale_names, df_main_out),
                ('B. Main Outcomes: 90-Day Drafting', [f'tt2_pooled{suffix}'] + tt2_cols, subscale_names, df_main_out),
                ('C. Main Outcomes: 90-Day Redlining', [f'tt2_cri_pooled{suffix}'] + cri_cols, subscale_names, df_main_out)
            ]

            sections_data = []

            for s_idx, (title, vars_list, names_list, df_eval) in enumerate(summary_sections):
                g_c = df_eval[df_eval[TREATMENT_VAR] == 0]
                g_t = df_eval[df_eval[TREATMENT_VAR] == 1]
                g_jc = df_eval[(df_eval[TREATMENT_VAR] == 0) & (df_eval['senior'] == 0)]
                g_jt = df_eval[(df_eval[TREATMENT_VAR] == 1) & (df_eval['senior'] == 0)]
                g_sc = df_eval[(df_eval[TREATMENT_VAR] == 0) & (df_eval['senior'] == 1)]
                g_st = df_eval[(df_eval[TREATMENT_VAR] == 1) & (df_eval['senior'] == 1)]

                sec_rows = []

                for idx, v_std in enumerate(vars_list):
                    if not v_std or v_std not in df_eval.columns: continue

                    weight_col = None
                    if USE_WEIGHTS_IN_SUMMARY and level == 'ind':
                        task_key = None
                        if 'tt1' in v_std: task_key = 'tt1_drades'
                        elif 'cri' in v_std: task_key = 'tt2_cri'
                        elif 'tt2' in v_std: task_key = 'tt2_drades'
                        if task_key: weight_col = f"wls_weight_{task_key}_{rater}"

                    m_c, s_c = get_mean_std(g_c, v_std, weight_col)
                    m_t, s_t = get_mean_std(g_t, v_std, weight_col)
                    m_jc, s_jc = get_mean_std(g_jc, v_std, weight_col)
                    m_jt, s_jt = get_mean_std(g_jt, v_std, weight_col)
                    m_sc, s_sc = get_mean_std(g_sc, v_std, weight_col)
                    m_st, s_st = get_mean_std(g_st, v_std, weight_col)

                    m_glob, s_glob = get_mean_std(df_eval, v_std, weight_col)
                    valid_v = df_eval[v_std].dropna()
                    if not valid_v.empty:
                        def fmt_val(val):
                            if pd.isna(val): return ""
                            res = f"{val + 0.0:.2f}"
                            return "0.00" if res == "-0.00" else res
                        
                        macros[f"summarytables_glob_mean_{v_std}"] = fmt_val(m_glob)
                        macros[f"summarytables_glob_sd_{v_std}"] = fmt_val(s_glob)
                        macros[f"summarytables_glob_N_{v_std}"] = f"{len(valid_v)}"
                        macros[f"summarytables_mean_{v_std}_ctrl"] = fmt_val(m_c)
                        macros[f"summarytables_mean_{v_std}_treat"] = fmt_val(m_t)
                        macros[f"summarytables_mean_{v_std}_ctrl_jun"] = fmt_val(m_jc)
                        macros[f"summarytables_mean_{v_std}_treat_jun"] = fmt_val(m_jt)
                        macros[f"summarytables_mean_{v_std}_ctrl_sen"] = fmt_val(m_sc)
                        macros[f"summarytables_mean_{v_std}_treat_sen"] = fmt_val(m_st)

                    sec_rows.append({
                        'idx': idx,
                        'name': names_list[idx],
                        'm_c': m_c, 's_c': s_c,
                        'm_t': m_t, 's_t': s_t,
                        'm_jc': m_jc, 's_jc': s_jc,
                        'm_jt': m_jt, 's_jt': s_jt,
                        'm_sc': m_sc, 's_sc': s_sc,
                        'm_st': m_st, 's_st': s_st
                    })

                v_pool = vars_list[0] if vars_list else None
                n_c = n_t = n_jc = n_jt = n_sc = n_st = 0
                if v_pool and v_pool in df_eval.columns:
                    n_c = int(g_c[v_pool].notna().sum())
                    n_t = int(g_t[v_pool].notna().sum())
                    n_jc = int(g_jc[v_pool].notna().sum())
                    n_jt = int(g_jt[v_pool].notna().sum())
                    n_sc = int(g_sc[v_pool].notna().sum())
                    n_st = int(g_st[v_pool].notna().sum())

                sections_data.append({
                    'title': title,
                    'rows': sec_rows,
                    'obs': {
                        'n_c': n_c, 'n_t': n_t,
                        'n_jc': n_jc, 'n_jt': n_jt,
                        'n_sc': n_sc, 'n_st': n_st
                    }
                })

            out_data[f"{rater}_{level}"] = sections_data

    with open('Jsons/cell_summary.json', 'w') as f:
        json.dump({'data': out_data, 'macros': macros}, f, cls=NpEncoder)
    
    return macros


# ==============================================================================
# 4. ECONOMETRIC REGRESSION MODELS
# ==============================================================================

all_models_data = {}

def capture_models(table_name, models_list, style, **kwargs):
    """
    Serializes regression model estimation results (coefficients, robust SEs,
    p-values, N, R-squared, hypothesis F-tests) into a structured JSON record.
    """
    model_dicts = []
    
    f_test_strings = [
        "junior = senior",
        "treat_x_junior = treat_x_senior",
        "junior + treat_x_junior = senior",
        "junior + treat_x_junior = senior + treat_x_senior"
    ]
    
    for i, res in enumerate(models_list):
        if res is None:
            model_dicts.append(None)
            continue
            
        m_data = {
            'params': res.params.to_dict(),
            'bse': res.bse.to_dict(),
            'pvalues': res.pvalues.to_dict(),
            'nobs': int(res.nobs),
            'rsquared': float(res.rsquared) if pd.notna(res.rsquared) else None,
            'included_ids': res.included_ids if hasattr(res, 'included_ids') else [],
            'df_resid': int(res.df_resid),
            'f_tests': {}
        }
        
        for t_str in f_test_strings:
            req_vars = [v for v in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', t_str)]
            if all(v in res.params for v in req_vars):
                t_res = res.t_test(t_str)
                tval = t_res.tvalue.item() if hasattr(t_res.tvalue, 'item') else float(t_res.tvalue)
                pval = t_res.pvalue.item() if hasattr(t_res.pvalue, 'item') else float(t_res.pvalue)
                m_data['f_tests'][t_str] = {'tvalue': tval, 'pvalue': pval}
                
        model_dicts.append(m_data)
        
    all_models_data[table_name] = {
        'style': style,
        'models': model_dicts,
        'kwargs': {k: v for k, v in kwargs.items() if k not in ['master_macros_dict', 'master_inclusion_dict', 'github_pat', 'config', 'df_cb']}
    }

models.build_main_latex = lambda table_name, style, models_list, rater, level, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled: capture_models(table_name, models_list, style, rater=rater, level=level, std_use_pooled=std_use_pooled)
models.build_combined_main_latex = lambda table_name, style, models_list, master_macros_dict, master_inclusion_dict, github_pat, config: capture_models(table_name, models_list, style)
models.build_speed_latex = lambda models_list, df_cb, rater, level, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled: capture_models("Speed", models_list, None, rater=rater, level=level, std_use_pooled=std_use_pooled)
models.build_secondary_latex = lambda table_name, config_sec, models_list, df_cb, rater, level, master_macros_dict, master_inclusion_dict, github_pat, config, std_use_pooled: capture_models(table_name, models_list, config_sec.get('style', None) if config_sec else None, rater=rater, level=level, std_use_pooled=std_use_pooled, config_sec_vars=config_sec.get('vars') if config_sec else None)

def analyze_models(df_subject_clean, df_raw_clean, df_cb, global_ref_firm, global_largest_firm, std_use_pooled):
    """
    Executes primary OLS regressions with firm fixed effects & clustered standard errors,
    pooling ratings and estimating treatment effects across tasks and experience strata.
    Outputs: Jsons/models_data.json.
    """
    run_main_effects(df_subject_clean, df_raw_clean, global_ref_firm, global_largest_firm, {}, {}, None, {}, std_use_pooled)
    run_combined_main_effects(df_subject_clean, df_raw_clean, global_ref_firm, global_largest_firm, {}, {}, None, {})
    run_combined_main_effects_noFFE(df_subject_clean, df_raw_clean, global_ref_firm, global_largest_firm, {}, {}, None, {})
    run_secondary_effects(df_subject_clean, df_cb, global_ref_firm, {}, {}, None, {}, std_use_pooled)
    
    with open('Jsons/models_data.json', 'w') as f:
        json.dump(all_models_data, f, cls=NpEncoder)
    
    return {}


# ==============================================================================
# 5. RANDOMIZATION INFERENCE & FISHER PERMUTATION TESTS
# ==============================================================================

def run_reg(df, formula, cluster_col=None, wls_weight=None):
    """Helper OLS/WLS regression runner for permutation loops."""
    if wls_weight and wls_weight in df.columns:
        return smf.wls(formula, data=df, weights=df[wls_weight]).fit(cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
    else:
        return smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[cluster_col]})


def analyze_fisher(df_raw_clean):
    """
    Executes 2,000 Monte Carlo randomization permutations of treatment assignment
    to compute exact two-sided Fisher p-values for primary redlining outcomes.
    Outputs: Jsons/fig_fisher.json (Figure 4: Fisher Permutation Tests).
    """
    df = df_raw_clean.copy()
    if TREATMENT_VAR not in df.columns and 'group' in df.columns:
        df[TREATMENT_VAR] = (df['group'] == 'Group 2').astype(int)

    global_largest_firm = df.drop_duplicates(subset=[UNIQUE_ID_VAR])[FIRM_VAR].mode()[0]
    global_ref_firm = f"'{global_largest_firm}'" if isinstance(global_largest_firm, str) else f"{global_largest_firm}"
    f_firm_base = f"C({FIRM_VAR}, Treatment(reference={global_ref_firm}))"

    cols_ind_human = get_target_cols(df, '90dayR', 'human', 'ind')
    if not cols_ind_human:
        with open('Jsons/fig_fisher.json', 'w') as f:
            json.dump({}, f)
        return {}
    
    d = df.dropna(subset=cols_ind_human, how='all').copy()
    id_vars = [c for c in d.columns if c not in cols_ind_human]
    melted = d.melt(id_vars=id_vars, value_vars=cols_ind_human, var_name='subcomponent', value_name='score').dropna(subset=['score'])
    melted['firm_sub'] = melted[FIRM_VAR].astype(str) + "_" + melted['subcomponent'].astype(str)
    df_human = melted[(melted['Rater_Type'].str.lower() == 'human') if 'Rater_Type' in melted.columns else True].copy()

    sub_vars_human = df_human['subcomponent'].unique().tolist()
    cla_matches_h = [s for s in sub_vars_human if 'cla' in s.lower()]
    ref_sub_h = cla_matches_h[0] if cla_matches_h else sub_vars_human[-1]
    f_sub_base_h = f"C(subcomponent, Treatment(reference='{ref_sub_h}'))"

    w_h_use = "wls_weight_tt2_cri_human" if "wls_weight_tt2_cri_human" in df_human.columns else None

    formula_full = f"score ~ {TREATMENT_VAR} + {f_firm_base} + {f_sub_base_h}"
    f_split = "junior + senior + treat_x_junior + treat_x_senior - 1"
    formula_split = f"score ~ {f_split} + {f_firm_base} + {f_sub_base_h}"
    
    df_human['treat_x_junior'] = df_human[TREATMENT_VAR] * df_human['junior']
    df_human['treat_x_senior'] = df_human[TREATMENT_VAR] * df_human['senior']
    
    mod_true_full = run_reg(df_human, formula_full, cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
    true_beta_full = mod_true_full.params.get(TREATMENT_VAR, np.nan)
    
    mod_true_split = run_reg(df_human, formula_split, cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
    true_beta_junior = mod_true_split.params.get('treat_x_junior', np.nan)
    true_beta_senior = mod_true_split.params.get('treat_x_senior', np.nan)

    np.random.seed(42)
    id_treatments = df_human.drop_duplicates(subset=[UNIQUE_ID_VAR]).set_index(UNIQUE_ID_VAR)[TREATMENT_VAR]
    original_treatments = id_treatments.values
    
    null_betas_full = []
    null_betas_junior = []
    null_betas_senior = []
    
    for _ in range(2000):
        permuted_treatments = np.random.permutation(original_treatments)
        perm_map = dict(zip(id_treatments.index, permuted_treatments))
        df_perm = df_human.copy()
        df_perm[TREATMENT_VAR] = df_perm[UNIQUE_ID_VAR].map(perm_map)
        
        df_perm['treat_x_junior'] = df_perm[TREATMENT_VAR] * df_perm['junior']
        df_perm['treat_x_senior'] = df_perm[TREATMENT_VAR] * df_perm['senior']
        
        mod_perm_full = run_reg(df_perm, formula_full, cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
        null_betas_full.append(mod_perm_full.params[TREATMENT_VAR])
        
        mod_perm_split = run_reg(df_perm, formula_split, cluster_col=UNIQUE_ID_VAR, wls_weight=w_h_use)
        null_betas_junior.append(mod_perm_split.params['treat_x_junior'])
        null_betas_senior.append(mod_perm_split.params['treat_x_senior'])

    out_data = {
        'full': {'true_beta': true_beta_full, 'null_betas': null_betas_full, 'title': 'Full Sample'},
        'junior': {'true_beta': true_beta_junior, 'null_betas': null_betas_junior, 'title': 'Juniors'},
        'senior': {'true_beta': true_beta_senior, 'null_betas': null_betas_senior, 'title': 'Seniors'}
    }
    
    macros = {}
    for subgroup, dat in out_data.items():
        if np.isnan(dat['true_beta']) or len(dat['null_betas']) == 0: continue
        nb = np.array(dat['null_betas'])
        tb = dat['true_beta']
        p_val = np.mean(np.abs(nb) >= np.abs(tb))
        pct = np.mean(nb < tb) * 100
        dat['p_val'] = p_val
        sub_map = {'Full Sample': 'all', 'Juniors': 'junior', 'Seniors': 'senior'}
        sg_short = sub_map.get(subgroup, subgroup)
        macros[f'fisher_beta_{sg_short}'] = f"{tb:.2f}"
        macros[f'fisher_p_{sg_short}'] = f"{p_val:.3f}"
        macros[f'fisher_pct_{sg_short}'] = f"{pct:.1f}"

    with open('Jsons/fig_fisher.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)
        
    return macros


# ==============================================================================
# 6. EMPIRICAL CDFS & OVERLAPPING HISTOGRAMS
# ==============================================================================

def get_cdf_data(series, weights):
    """Computes weighted empirical CDF and 95% robust confidence bands across grid."""
    thresholds = np.linspace(1, 5, 100)
    df_clean = pd.DataFrame({'score': series, 'weight': weights}).dropna()
    n = len(df_clean)
    
    if n == 0 or df_clean['weight'].sum() == 0:
        return {'score': [], 'cdf': [], 'ci_lower': [], 'ci_upper': []}
        
    cdfs, lower_cis, upper_cis = [], [], []
    total_weight = df_clean['weight'].sum()
    eff_n = (total_weight**2) / (df_clean['weight']**2).sum() if total_weight > 0 else 0
    
    for t in thresholds:
        p = df_clean.loc[df_clean['score'] <= t, 'weight'].sum() / total_weight
        cdfs.append(float(p))
        se = np.sqrt((p * (1 - p)) / eff_n) if eff_n > 0 else 0
        margin = 1.96 * se
        lower_cis.append(float(max(0, p - margin)))
        upper_cis.append(float(min(1, p + margin)))

    return {'score': thresholds.tolist(), 'cdf': cdfs, 'ci_lower': lower_cis, 'ci_upper': upper_cis}


def analyze_cdf(df_raw_clean):
    """
    Computes empirical cumulative distribution functions and two-sample Kolmogorov-Smirnov
    tests for stochastic dominance.
    Outputs: Jsons/fig_cdf.json.
    """
    df_plot = df_raw_clean.copy()
    rater_col = next((c for c in df_plot.columns if c.lower() == 'rater_type'), None)
    if rater_col:
        df_plot = df_plot[df_plot[rater_col].str.lower() == 'human'].copy()
        
    plot_vars = []
    for key in ['tt1_drades', 'tt2_drades', 'tt2_cri']:
        cols = RATER_OUTCOMES.get(key, [])
        valid_cols = [c for c in cols if c in df_plot.columns]
        if valid_cols:
            var_name = f'{key}_avg'
            df_plot[var_name] = df_plot[valid_cols].mean(axis=1)
            if FIRM_VAR in df_plot.columns:
                firm_means = df_plot.groupby(FIRM_VAR)[var_name].transform('mean')
                global_mean = df_plot[var_name].mean()
                df_plot[var_name] = np.clip(df_plot[var_name] - firm_means + global_mean, 1.0, 5.0)
            plot_vars.append(var_name)

    if not plot_vars or TREATMENT_VAR not in df_plot.columns:
        with open('Jsons/fig_cdf.json', 'w') as f:
            json.dump({}, f)
        return {}
    
    df_plot['Study Group'] = df_plot[TREATMENT_VAR].map({0: 'Control', 1: 'Treatment'})
    
    weight_map = {
        'tt1_drades_avg': 'wls_weight_tt1_drades_human',
        'tt2_drades_avg': 'wls_weight_tt2_drades_human',
        'tt2_cri_avg': 'wls_weight_tt2_cri_human'
    }
    
    frames_plot1 = [
        {'title': 'All lawyers', 'key': 'all'},
        {'title': 'Junior lawyers', 'key': 'juniors'},
        {'title': 'Senior lawyers', 'key': 'seniors'}
    ]
    
    out_data = {'plot_vars': plot_vars, 'frames': frames_plot1, 'plot1': {}, 'plot2': {}}
    macros = {}
    
    for row_idx, var in enumerate(plot_vars):
        w_col = weight_map.get(var)
        if w_col not in df_plot.columns: continue
        
        out_data['plot1'][var] = {}
        for col_idx, frame in enumerate(frames_plot1):
            if frame['key'] == 'all':
                df_sample = df_plot.dropna(subset=[var, w_col])
            elif frame['key'] == 'juniors':
                df_sample = df_plot[df_plot['junior'] == 1].dropna(subset=[var, w_col])
            else:
                df_sample = df_plot[df_plot['senior'] == 1].dropna(subset=[var, w_col])

            treat_df = df_sample[df_sample['Study Group'] == 'Treatment']
            ctrl_df = df_sample[df_sample['Study Group'] == 'Control']
            
            t_data = treat_df[var].tolist()
            t_weight = treat_df[w_col].tolist()
            c_data = ctrl_df[var].tolist()
            c_weight = ctrl_df[w_col].tolist()
            
            stat, p_less, p_greater = None, None, None
            if len(t_data) > 0 and len(c_data) > 0:
                stat, p_less = ks_2samp(t_data, c_data, alternative='less')
                _, p_greater = ks_2samp(t_data, c_data, alternative='greater')
                macros[f"cdf_fosd_stat_plot1_{var}_{frame['key']}"] = f"{stat:.3f}"
                macros[f"cdf_fosd1_p_plot1_{var}_{frame['key']}"] = f"{p_less:.3f}"
                macros[f"cdf_fosd2_p_plot1_{var}_{frame['key']}"] = f"{p_greater:.3f}"
                
            out_data['plot1'][var][frame['key']] = {
                'treat': get_cdf_data(pd.Series(t_data), pd.Series(t_weight)),
                'ctrl': get_cdf_data(pd.Series(c_data), pd.Series(c_weight)),
                'stat': stat, 'p_less': p_less, 'p_greater': p_greater,
                'n_total': len(t_data) + len(c_data)
            }

        df_sample = df_plot.dropna(subset=[var, w_col])
        ctrl_jun = df_sample[(df_sample['Study Group'] == 'Control') & (df_sample['junior'] == 1)]
        ctrl_sen = df_sample[(df_sample['Study Group'] == 'Control') & (df_sample['senior'] == 1)]
        trt_jun = df_sample[(df_sample['Study Group'] == 'Treatment') & (df_sample['junior'] == 1)]
        trt_sen = df_sample[(df_sample['Study Group'] == 'Treatment') & (df_sample['senior'] == 1)]
        
        out_data['plot2'][var] = {}
        comps = [
            ('ctrlJun_ctrlSen', ctrl_jun, ctrl_sen, 'Untreated Juniors', 'Untreated Seniors'),
            ('trtJun_trtSen', trt_jun, trt_sen, 'Treated Juniors', 'Treated Seniors'),
            ('trtJun_ctrlSen', trt_jun, ctrl_sen, 'Treated Juniors', 'Untreated Seniors')
        ]
        for key, df1, df2, lbl1, lbl2 in comps:
            mean1 = np.average(df1[var], weights=df1[w_col]) if not df1.empty and df1[w_col].sum() > 0 else 0
            mean2 = np.average(df2[var], weights=df2[w_col]) if not df2.empty and df2[w_col].sum() > 0 else 0
            
            if mean1 < mean2:
                data_A, weight_A, label_A = df1[var], df1[w_col], lbl1
                data_B, weight_B, label_B = df2[var], df2[w_col], lbl2
            else:
                data_A, weight_A, label_A = df2[var], df2[w_col], lbl2
                data_B, weight_B, label_B = df1[var], df1[w_col], lbl1
                
            stat, p_less, p_greater = None, None, None
            if len(data_A) > 0 and len(data_B) > 0:
                stat, p_less = ks_2samp(data_B, data_A, alternative='less')
                _, p_greater = ks_2samp(data_B, data_A, alternative='greater')
                macros[f"cdf_fosd_stat_plot2_{var}_{key}"] = f"{stat:.3f}"
                macros[f"cdf_fosd1_p_plot2_{var}_{key}"] = f"{p_less:.3f}"
                macros[f"cdf_fosd2_p_plot2_{var}_{key}"] = f"{p_greater:.3f}"
                
            out_data['plot2'][var][key] = {
                'A': get_cdf_data(data_A, weight_A),
                'B': get_cdf_data(data_B, weight_B),
                'label_A': label_A, 'label_B': label_B,
                'stat': stat, 'p_less': p_less, 'p_greater': p_greater,
                'n_total': len(data_A) + len(data_B)
            }
            
    out_data['frames'] = [{'title': f['title'], 'key': f['key']} for f in frames_plot1]

    with open('Jsons/fig_cdf.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)
        
    return macros


def analyze_histograms(df_raw_clean):
    """
    Computes distribution densities and histogram bins for drafting and redlining tasks.
    Outputs: Jsons/fig_histograms.json (Figures 2 & 3: Outcome Histograms).
    """
    df_plot = df_raw_clean.copy()
    rater_col = next((c for c in df_plot.columns if c.lower() == 'rater_type'), None)
    if rater_col:
        df_plot = df_plot[df_plot[rater_col].str.lower() == 'human'].copy()
        
    plot_vars = []
    for key in ['tt1_drades', 'tt2_drades', 'tt2_cri']:
        cols = RATER_OUTCOMES.get(key, [])
        valid_cols = [c for c in cols if c in df_plot.columns]
        if valid_cols:
            var_name = f'{key}_avg'
            df_plot[var_name] = df_plot[valid_cols].mean(axis=1)
            if FIRM_VAR in df_plot.columns:
                firm_means = df_plot.groupby(FIRM_VAR)[var_name].transform('mean')
                global_mean = df_plot[var_name].mean()
                df_plot[var_name] = np.clip(df_plot[var_name] - firm_means + global_mean, 1.0, 5.0)
            plot_vars.append(var_name)

    if not plot_vars or TREATMENT_VAR not in df_plot.columns:
        with open('Jsons/fig_histograms.json', 'w') as f:
            json.dump({}, f)
        return {}
    
    df_plot['Study Group'] = df_plot[TREATMENT_VAR].map({0: 'Control', 1: 'Treatment'})
    
    weight_map = {
        'tt1_drades_avg': 'wls_weight_tt1_drades_human',
        'tt2_drades_avg': 'wls_weight_tt2_drades_human',
        'tt2_cri_avg': 'wls_weight_tt2_cri_human'
    }
    
    frames_plot1 = [
        {'title': 'All lawyers', 'key': 'all'},
        {'title': 'Junior lawyers', 'key': 'juniors'},
        {'title': 'Senior lawyers', 'key': 'seniors'}
    ]
    
    out_data = {'plot_vars': plot_vars, 'frames': frames_plot1, 'plot1': {}, 'plot2': {}}
    
    for row_idx, var in enumerate(plot_vars):
        w_col = weight_map.get(var)
        if w_col not in df_plot.columns: continue
        
        out_data['plot1'][var] = {}
        for col_idx, frame in enumerate(frames_plot1):
            if frame['key'] == 'all':
                df_sample = df_plot.dropna(subset=[var, w_col])
            elif frame['key'] == 'juniors':
                df_sample = df_plot[df_plot['junior'] == 1].dropna(subset=[var, w_col])
            else:
                df_sample = df_plot[df_plot['senior'] == 1].dropna(subset=[var, w_col])

            treat_df = df_sample[df_sample['Study Group'] == 'Treatment']
            ctrl_df = df_sample[df_sample['Study Group'] == 'Control']
            
            t_data = treat_df[var].tolist()
            t_weight = treat_df[w_col].tolist()
            c_data = ctrl_df[var].tolist()
            c_weight = ctrl_df[w_col].tolist()
            
            stat, p_less, p_greater = None, None, None
            if len(t_data) > 0 and len(c_data) > 0:
                stat, p_less = ks_2samp(t_data, c_data, alternative='less')
                _, p_greater = ks_2samp(t_data, c_data, alternative='greater')
                
            out_data['plot1'][var][frame['key']] = {
                'treat': {'scores': t_data, 'weights': t_weight},
                'ctrl': {'scores': c_data, 'weights': c_weight},
                'stat': stat, 'p_less': p_less, 'p_greater': p_greater,
                'n_total': len(t_data) + len(c_data)
            }

        df_sample = df_plot.dropna(subset=[var, w_col])
        ctrl_jun = df_sample[(df_sample['Study Group'] == 'Control') & (df_sample['junior'] == 1)]
        ctrl_sen = df_sample[(df_sample['Study Group'] == 'Control') & (df_sample['senior'] == 1)]
        trt_jun = df_sample[(df_sample['Study Group'] == 'Treatment') & (df_sample['junior'] == 1)]
        trt_sen = df_sample[(df_sample['Study Group'] == 'Treatment') & (df_sample['senior'] == 1)]
        
        out_data['plot2'][var] = {}
        comps = [
            ('ctrlJun_ctrlSen', ctrl_jun, ctrl_sen, 'Untreated Juniors', 'Untreated Seniors'),
            ('trtJun_trtSen', trt_jun, trt_sen, 'Treated Juniors', 'Treated Seniors'),
            ('trtJun_ctrlSen', trt_jun, ctrl_sen, 'Treated Juniors', 'Untreated Seniors')
        ]
        for key, df1, df2, lbl1, lbl2 in comps:
            mean1 = np.average(df1[var], weights=df1[w_col]) if not df1.empty and df1[w_col].sum() > 0 else 0
            mean2 = np.average(df2[var], weights=df2[w_col]) if not df2.empty and df2[w_col].sum() > 0 else 0
            
            if mean1 < mean2:
                data_A, weight_A, label_A = df1[var], df1[w_col], lbl1
                data_B, weight_B, label_B = df2[var], df2[w_col], lbl2
            else:
                data_A, weight_A, label_A = df2[var], df2[w_col], lbl2
                data_B, weight_B, label_B = df1[var], df1[w_col], lbl1
                
            stat, p_less, p_greater = None, None, None
            if len(data_A) > 0 and len(data_B) > 0:
                stat, p_less = ks_2samp(data_B, data_A, alternative='less')
                _, p_greater = ks_2samp(data_B, data_A, alternative='greater')
                
            out_data['plot2'][var][key] = {
                'A': {'scores': data_A.tolist(), 'weights': weight_A.tolist()},
                'B': {'scores': data_B.tolist(), 'weights': weight_B.tolist()},
                'label_A': label_A, 'label_B': label_B,
                'stat': stat, 'p_less': p_less, 'p_greater': p_greater,
                'n_total': len(data_A) + len(data_B)
            }
            
    out_data['frames'] = [{'title': f['title'], 'key': f['key']} for f in frames_plot1]

    with open('Jsons/fig_histograms.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)
        
    return {}


# ==============================================================================
# 7. SUBCOMPONENT FOREST PLOTS & SENSITIVITY ANALYSES
# ==============================================================================

def get_clean_label(var_name, raw_lbl):
    """Formats subelement variable names cleanly for presentation."""
    if pd.isna(raw_lbl): return var_name
    match = re.search(r'\] (.*?)( -|$)', str(raw_lbl))
    if match: return match.group(1).strip()
    return raw_lbl

def analyze_forest(df_raw_clean, df_cb):
    """
    Estimates treatment effects across all quality subcomponents (Enforceability,
    Technical Accuracy, Strategic Ambiguity, Completeness, Clarity) for forest plots.
    Outputs: Jsons/fig_forest.json (Figure 1: Forest Plot of Subelements).
    """
    df_plot = df_raw_clean.copy()
    if 'rater_type' in df_plot.columns:
        df_plot = df_plot[df_plot['rater_type'] == 'human'].copy()
    elif 'Rater_Type' in df_plot.columns:
        df_plot = df_plot[df_plot['Rater_Type'].str.lower() == 'human'].copy()

    if 'group' in df_plot.columns and TREATMENT_VAR not in df_plot.columns:
        df_plot[TREATMENT_VAR] = (df_plot['group'] == 'Group 2').astype(int)

    df_plot = df_plot.dropna(subset=[UNIQUE_ID_VAR])
    if 'exp' in df_plot.columns:
        df_plot['exp'] = pd.to_numeric(df_plot['exp'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    if 'exp' not in df_plot.columns or df_plot['exp'].isna().all():
        if 'senior' in df_plot.columns:
            df_plot['exp'] = np.where(df_plot['senior'] == 1, 7.0, 3.0)

    EXP_COL = 'exp'
    EXP_THRESHOLD = 7
    df_plot['junior'] = (df_plot[EXP_COL] < EXP_THRESHOLD).astype(int)
    df_plot['senior'] = (df_plot[EXP_COL] >= EXP_THRESHOLD).astype(int)
    df_plot['treat_x_junior'] = df_plot[TREATMENT_VAR] * df_plot['junior']
    df_plot['treat_x_senior'] = df_plot[TREATMENT_VAR] * df_plot['senior']

    label_map = {}
    if df_cb is not None:
        lbl_col = 'label_shorthand' if 'label_shorthand' in df_cb.columns else 'label'
        if lbl_col in df_cb.columns and 'varname' in df_cb.columns:
            label_map = dict(zip(df_cb['varname'].str.strip(), df_cb[lbl_col].str.strip()))

    panels = [
        {"title": "Drafting: 10-day task", "vars": ['tt1_sum_rat_drades'] + RATER_OUTCOMES.get('tt1_drades', []), "weight": "wls_weight_tt1_drades_human"},
        {"title": "Drafting: 90-day task", "vars": ['tt2_sum_rat_drades'] + RATER_OUTCOMES.get('tt2_drades', []), "weight": "wls_weight_tt2_drades_human"},
        {"title": "Redlining: 90-day task", "vars": ['tt2_sum_rat_cri'] + RATER_OUTCOMES.get('tt2_cri', []), "weight": "wls_weight_tt2_cri_human"}
    ]

    def run_model(df_sub, var_name, weight_col, panel_vars=None, target_effect=TREATMENT_VAR):
        if var_name in ['tt1_sum_rat_drades', 'tt2_sum_rat_drades', 'tt2_sum_rat_cri']:
            if panel_vars is None: return np.nan, np.nan, np.nan, ""
            sub_cols = [c for c in panel_vars if c != var_name]
            std_cols = [f"{sc}_std_human_ind" for sc in sub_cols if f"{sc}_std_human_ind" in df_sub.columns]
            if not std_cols: return np.nan, np.nan, np.nan, ""
            df_r = df_sub.dropna(subset=std_cols, how='all').copy()
            id_vars = [c for c in df_r.columns if c not in std_cols]
            df_melt = df_r.melt(id_vars=id_vars, value_vars=std_cols, var_name='subcomponent', value_name='score').dropna(subset=['score'])
            if weight_col in df_melt.columns: df_melt = df_melt[df_melt[weight_col] > 0]
            if len(df_melt) <= 5 or df_melt[TREATMENT_VAR].nunique() <= 1: return np.nan, np.nan, np.nan, ""
            
            largest_firm = df_melt[FIRM_VAR].value_counts().idxmax()
            ref_sub = std_cols[-1]
            for s in std_cols:
                if 'cla' in s.lower():
                    ref_sub = s
                    break
            
            f_split = "junior + senior + treat_x_junior + treat_x_senior - 1"
            if target_effect == TREATMENT_VAR:
                formula = f"score ~ {TREATMENT_VAR} + C({FIRM_VAR}, Treatment(reference='{largest_firm}')) + C(subcomponent, Treatment(reference='{ref_sub}'))"
            else:
                formula = f"score ~ {f_split} + C({FIRM_VAR}, Treatment(reference='{largest_firm}')) + C(subcomponent, Treatment(reference='{ref_sub}'))"
            
            if weight_col:
                model = smf.wls(formula, data=df_melt, weights=df_melt[weight_col]).fit(cov_type='cluster', cov_kwds={'groups': df_melt[UNIQUE_ID_VAR]})
            else:
                model = smf.ols(formula, data=df_melt).fit(cov_type='cluster', cov_kwds={'groups': df_melt[UNIQUE_ID_VAR]})
        else:
            var_name_std = f"{var_name}_std_human_ind"
            if var_name_std not in df_sub.columns or weight_col not in df_sub.columns: return np.nan, np.nan, np.nan, ""
            
            df_clean = df_sub.dropna(subset=[var_name_std, TREATMENT_VAR, weight_col, UNIQUE_ID_VAR, FIRM_VAR]).copy()
            df_clean = df_clean[df_clean[weight_col] > 0]
            if len(df_clean) <= 5 or df_clean[TREATMENT_VAR].nunique() <= 1: return np.nan, np.nan, np.nan, ""
                
            largest_firm = df_clean[FIRM_VAR].value_counts().idxmax()
            f_split = "junior + senior + treat_x_junior + treat_x_senior - 1"
            if target_effect == TREATMENT_VAR:
                formula = f"{var_name_std} ~ {TREATMENT_VAR} + C({FIRM_VAR}, Treatment(reference='{largest_firm}'))"
            else:
                formula = f"{var_name_std} ~ {f_split} + C({FIRM_VAR}, Treatment(reference='{largest_firm}'))"
            
            if weight_col:
                model = smf.wls(formula, data=df_clean, weights=df_clean[weight_col]).fit(cov_type='cluster', cov_kwds={'groups': df_clean[UNIQUE_ID_VAR]})
            else:
                model = smf.ols(formula, data=df_clean).fit(cov_type='cluster', cov_kwds={'groups': df_clean[UNIQUE_ID_VAR]})

        coef = model.params.get(target_effect, np.nan)
        if pd.isna(coef): return np.nan, np.nan, np.nan, np.nan, ""
        ci = model.conf_int(alpha=0.05).loc[target_effect]
        p_val = model.pvalues[target_effect]
        stars_str = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        return coef, ci[0], ci[1], p_val, stars_str

    out_data = {}
    macros = {}
    for panel_idx, panel in enumerate(panels):
        out_data[panel['title']] = []
        p_prefix = "Drafti" if "Drafting" in panel['title'] else "Redlin"
        for var in panel['vars']:
            raw_lbl = label_map.get(var, var)
            clean_lbl = get_clean_label(var, raw_lbl)
            
            c_all, ci_l_all, ci_u_all, p_all, s_all = run_model(df_plot, var, panel['weight'], panel['vars'], TREATMENT_VAR)
            c_jun, ci_l_jun, ci_u_jun, p_jun, s_jun = run_model(df_plot, var, panel['weight'], panel['vars'], 'treat_x_junior')
            c_sen, ci_l_sen, ci_u_sen, p_sen, s_sen = run_model(df_plot, var, panel['weight'], panel['vars'], 'treat_x_senior')
            
            if not np.isnan(c_all):
                macros[f"forest_{p_prefix}_{var}_all"] = f"{c_all:.2f}"
                macros[f"forest_{p_prefix}_{var}_all_pval"] = f"{p_all:.3f}"
                macros[f"forest_{p_prefix}_{var}_all_stars"] = s_all
            if not np.isnan(c_jun): 
                macros[f"forest_{p_prefix}_{var}_junior"] = f"{c_jun:.2f}"
                macros[f"forest_{p_prefix}_{var}_junior_pval"] = f"{p_jun:.3f}"
                macros[f"forest_{p_prefix}_{var}_junior_stars"] = s_jun
            if not np.isnan(c_sen): 
                macros[f"forest_{p_prefix}_{var}_senior"] = f"{c_sen:.2f}"
                macros[f"forest_{p_prefix}_{var}_senior_pval"] = f"{p_sen:.3f}"
                macros[f"forest_{p_prefix}_{var}_senior_stars"] = s_sen
                
            out_data[panel['title']].append({
                'var': var, 'label': clean_lbl,
                'all': {'c': c_all, 'ci_l': ci_l_all, 'ci_u': ci_u_all, 's': s_all},
                'jun': {'c': c_jun, 'ci_l': ci_l_jun, 'ci_u': ci_u_jun, 's': s_jun},
                'sen': {'c': c_sen, 'ci_l': ci_l_sen, 'ci_u': ci_u_sen, 's': s_sen}
            })

    with open('Jsons/fig_forest.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)
    
    return macros


def analyze_leave_one_firm_out(df_raw_clean):
    """
    Estimates leave-one-firm-out robustness for the 90-day redlining task.
    Sequentially removes each firm and estimates treatment effects for
    the full sample, juniors, and seniors using weighted OLS with fixed effects.
    Outputs: Jsons/fig_leave_one_firm_out.json.
    """
    df_plot = df_raw_clean.copy()
    if 'rater_type' in df_plot.columns:
        df_plot = df_plot[df_plot['rater_type'] == 'human'].copy()
    elif 'Rater_Type' in df_plot.columns:
        df_plot = df_plot[df_plot['Rater_Type'].str.lower() == 'human'].copy()

    if 'group' in df_plot.columns and TREATMENT_VAR not in df_plot.columns:
        df_plot[TREATMENT_VAR] = (df_plot['group'] == 'Group 2').astype(int)

    df_plot = df_plot.dropna(subset=[UNIQUE_ID_VAR])
    if 'exp' in df_plot.columns:
        df_plot['exp'] = pd.to_numeric(df_plot['exp'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    if 'exp' not in df_plot.columns or df_plot['exp'].isna().all():
        if 'senior' in df_plot.columns:
            df_plot['exp'] = np.where(df_plot['senior'] == 1, 7.0, 3.0)

    EXP_COL = 'exp'
    EXP_THRESHOLD = 7
    df_plot['junior'] = (df_plot[EXP_COL] < EXP_THRESHOLD).astype(int)
    df_plot['senior'] = (df_plot[EXP_COL] >= EXP_THRESHOLD).astype(int)
    df_plot['treat_x_junior'] = df_plot[TREATMENT_VAR] * df_plot['junior']
    df_plot['treat_x_senior'] = df_plot[TREATMENT_VAR] * df_plot['senior']

    cols_ind_human = get_target_cols(df_plot, '90dayR', 'human', 'ind')
    if not cols_ind_human:
        with open('Jsons/fig_leave_one_firm_out.json', 'w') as f:
            json.dump([], f)
        return {}

    d = df_plot.dropna(subset=cols_ind_human, how='all').copy()
    id_vars = [c for c in d.columns if c not in cols_ind_human]
    melted = d.melt(id_vars=id_vars, value_vars=cols_ind_human, var_name='subcomponent', value_name='score').dropna(subset=['score'])
    df_human = melted.copy()
    df_human[FIRM_VAR] = df_human[FIRM_VAR].astype(str)

    w_col = "wls_weight_tt2_cri_human"
    sub_vars = df_human['subcomponent'].unique().tolist()
    cla_matches = [s for s in sub_vars if 'cla' in s.lower()]
    ref_sub = cla_matches[0] if cla_matches else sub_vars[-1]

    # Load firm ordering from firm_summary.json to match Table 1 exactly
    firm_order = []
    if os.path.exists('Jsons/firm_summary.json'):
        with open('Jsons/firm_summary.json', 'r') as f:
            fs_data = json.load(f)
        for i, r in enumerate(fs_data.get('firms', [])):
            firm_order.append({
                'idx': i + 1,
                'firm_num': str(r['firm_num']),
                'firm_name': r.get('firm_name', '')
            })

    def run_estimates(sub_df):
        if len(sub_df) == 0 or sub_df[TREATMENT_VAR].nunique() <= 1:
            return {
                'all': {'c': np.nan, 'ci_l': np.nan, 'ci_u': np.nan, 'p': np.nan, 's': ''},
                'jun': {'c': np.nan, 'ci_l': np.nan, 'ci_u': np.nan, 'p': np.nan, 's': ''},
                'sen': {'c': np.nan, 'ci_l': np.nan, 'ci_u': np.nan, 'p': np.nan, 's': ''}
            }
        largest_firm = sub_df[FIRM_VAR].value_counts().idxmax()
        f_firm = f"C({FIRM_VAR}, Treatment(reference='{largest_firm}'))"
        f_sub = f"C(subcomponent, Treatment(reference='{ref_sub}'))"
        
        # All participants
        form_all = f"score ~ {TREATMENT_VAR} + {f_firm} + {f_sub}"
        if w_col in sub_df.columns:
            mod_all = smf.wls(form_all, data=sub_df, weights=sub_df[w_col]).fit(cov_type='cluster', cov_kwds={'groups': sub_df[UNIQUE_ID_VAR]})
        else:
            mod_all = smf.ols(form_all, data=sub_df).fit(cov_type='cluster', cov_kwds={'groups': sub_df[UNIQUE_ID_VAR]})
            
        # Split (juniors & seniors)
        f_split = "junior + senior + treat_x_junior + treat_x_senior - 1"
        form_split = f"score ~ {f_split} + {f_firm} + {f_sub}"
        if w_col in sub_df.columns:
            mod_split = smf.wls(form_split, data=sub_df, weights=sub_df[w_col]).fit(cov_type='cluster', cov_kwds={'groups': sub_df[UNIQUE_ID_VAR]})
        else:
            mod_split = smf.ols(form_split, data=sub_df).fit(cov_type='cluster', cov_kwds={'groups': sub_df[UNIQUE_ID_VAR]})

        res = {}
        for k, mod, t_var in [('all', mod_all, TREATMENT_VAR), ('jun', mod_split, 'treat_x_junior'), ('sen', mod_split, 'treat_x_senior')]:
            c = mod.params.get(t_var, np.nan)
            if pd.isna(c):
                res[k] = {'c': np.nan, 'ci_l': np.nan, 'ci_u': np.nan, 'p': np.nan, 's': ''}
            else:
                ci = mod.conf_int(alpha=0.05).loc[t_var]
                p = mod.pvalues[t_var]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                res[k] = {'c': float(c), 'ci_l': float(ci[0]), 'ci_u': float(ci[1]), 'p': float(p), 's': stars}
        return res

    out_records = []
    # 1. Full sample (no removals)
    out_records.append({
        'label': 'Full sample',
        'firm_num': None,
        'firm_name': 'Full active sample',
        'estimates': run_estimates(df_human)
    })

    # 2. Sequential firm removals (excluding firms with 0 redlining participants: Firms 3, 4, 7)
    for f_info in firm_order:
        if f_info['idx'] in [3, 4, 7]:
            continue
        f_num = f_info['firm_num']
        sub_df = df_human[df_human[FIRM_VAR] != f_num].copy()
        out_records.append({
            'label': f"Firm {f_info['idx']}",
            'firm_num': f_num,
            'firm_name': f_info['firm_name'],
            'estimates': run_estimates(sub_df)
        })

    with open('Jsons/fig_leave_one_firm_out.json', 'w') as f:
        json.dump(out_records, f, cls=NpEncoder)

    return {}


def analyze_sensitivity(df_raw_clean):
    """
    Evaluates robustness of treatment effects across alternative junior/midlevel/senior
    experience thresholds ((3, 7), (4, 8), (5, 9)).
    Outputs: Jsons/fig_sensitivity.json (Figure A3: Sensitivity Analysis).
    """
    df_plot = df_raw_clean.copy()
    if 'rater_type' in df_plot.columns:
        df_plot = df_plot[df_plot['rater_type'] == 'human'].copy()
    elif 'Rater_Type' in df_plot.columns:
        df_plot = df_plot[df_plot['Rater_Type'].str.lower() == 'human'].copy()

    if 'group' in df_plot.columns and TREATMENT_VAR not in df_plot.columns:
        df_plot[TREATMENT_VAR] = (df_plot['group'] == 'Group 2').astype(int)

    df_plot = df_plot.dropna(subset=[UNIQUE_ID_VAR])
    if 'exp' in df_plot.columns:
        df_plot['exp'] = pd.to_numeric(df_plot['exp'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    if 'exp' not in df_plot.columns or df_plot['exp'].isna().all():
        if 'senior' in df_plot.columns:
            df_plot['exp'] = np.where(df_plot['senior'] == 1, 7.0, 3.0)

    EXP_COL = 'exp'
    THRESHOLDS_3 = [(3, 7), (4, 8), (5, 9)]

    panels = [
        {"title": "Drafting: 10-day task", "vars": ['tt1_sum_rat_drades'] + RATER_OUTCOMES.get('tt1_drades', []), "weight": "wls_weight_tt1_drades_human"},
        {"title": "Drafting: 90-day task", "vars": ['tt2_sum_rat_drades'] + RATER_OUTCOMES.get('tt2_drades', []), "weight": "wls_weight_tt2_drades_human"},
        {"title": "Redlining: 90-day task", "vars": ['tt2_sum_rat_cri'] + RATER_OUTCOMES.get('tt2_cri', []), "weight": "wls_weight_tt2_cri_human"}
    ]

    sens3_results = {p['title']: {'seniors': [], 'midlevels': [], 'juniors': []} for p in panels}

    for panel in panels:
        out_title = panel['title']
        w_col = panel['weight']

        sub_cols_raw = panel['vars'][1:]
        sub_cols_std = [f"{c}_std_human_ind" for c in sub_cols_raw if f"{c}_std_human_ind" in df_plot.columns]

        if not sub_cols_std or w_col not in df_plot.columns:
            continue

        req_cols = [TREATMENT_VAR, w_col, UNIQUE_ID_VAR, FIRM_VAR, EXP_COL]
        df_sub = df_plot.dropna(subset=sub_cols_std, how='all').copy()
        df_sub = df_sub.dropna(subset=[c for c in req_cols if c in df_sub.columns])
        df_sub = df_sub[df_sub[w_col] > 0]

        if len(df_sub) <= 1: continue

        id_vars = [c for c in df_sub.columns if c not in sub_cols_std]
        df_melt = df_sub.melt(id_vars=id_vars, value_vars=sub_cols_std, var_name='subcomponent', value_name='score').dropna(subset=['score'])

        if df_melt.empty: continue

        largest_firm = df_melt[FIRM_VAR].value_counts().idxmax()
        sub_vars = df_melt['subcomponent'].unique().tolist()
        cla_matches = [s for s in sub_vars if 'cla' in s.lower()]
        ref_sub = cla_matches[0] if cla_matches else sub_vars[-1]

        for (jun_t, sen_t) in THRESHOLDS_3:
            df_thresh = df_melt.copy()
            df_thresh['senior_dummy'] = (df_thresh[EXP_COL] >= sen_t).astype(int)
            df_thresh['junior_dummy'] = (df_thresh[EXP_COL] < jun_t).astype(int)
            df_thresh['midlevel_dummy'] = ((df_thresh[EXP_COL] >= jun_t) & (df_thresh[EXP_COL] < sen_t)).astype(int)
            
            df_thresh['treat_x_senior'] = df_thresh[TREATMENT_VAR] * df_thresh['senior_dummy']
            df_thresh['treat_x_junior'] = df_thresh[TREATMENT_VAR] * df_thresh['junior_dummy']
            df_thresh['treat_x_midlevel'] = df_thresh[TREATMENT_VAR] * df_thresh['midlevel_dummy']

            formula = f"score ~ junior_dummy + midlevel_dummy + senior_dummy + treat_x_junior + treat_x_midlevel + treat_x_senior - 1 + C({FIRM_VAR}, Treatment(reference='{largest_firm}')) + C(subcomponent, Treatment(reference='{ref_sub}'))"

            if w_col:
                model = smf.wls(formula, data=df_thresh, weights=df_thresh[w_col]).fit(cov_type='cluster', cov_kwds={'groups': df_thresh[UNIQUE_ID_VAR]})
            else:
                model = smf.ols(formula, data=df_thresh).fit(cov_type='cluster', cov_kwds={'groups': df_thresh[UNIQUE_ID_VAR]})

            coef_jun = model.params.get('treat_x_junior', np.nan)
            if not np.isnan(coef_jun):
                ci_jun = model.conf_int(alpha=0.05).loc['treat_x_junior']
                p_jun = model.pvalues['treat_x_junior']
                sens3_results[out_title]['juniors'].append({'x_jun': jun_t, 'x_sen': sen_t, 'y': float(coef_jun), 'ci_lower': float(ci_jun[0]), 'ci_upper': float(ci_jun[1]), 'p': float(p_jun)})
            else:
                sens3_results[out_title]['juniors'].append({'x_jun': jun_t, 'x_sen': sen_t, 'y': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p': 1.0})

            coef_mid = model.params.get('treat_x_midlevel', np.nan)
            if not np.isnan(coef_mid):
                ci_mid = model.conf_int(alpha=0.05).loc['treat_x_midlevel']
                p_mid = model.pvalues['treat_x_midlevel']
                sens3_results[out_title]['midlevels'].append({'x_jun': jun_t, 'x_sen': sen_t, 'y': float(coef_mid), 'ci_lower': float(ci_mid[0]), 'ci_upper': float(ci_mid[1]), 'p': float(p_mid)})
            else:
                sens3_results[out_title]['midlevels'].append({'x_jun': jun_t, 'x_sen': sen_t, 'y': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p': 1.0})

            coef_sen = model.params.get('treat_x_senior', np.nan)
            if not np.isnan(coef_sen):
                ci_sen = model.conf_int(alpha=0.05).loc['treat_x_senior']
                p_sen = model.pvalues['treat_x_senior']
                sens3_results[out_title]['seniors'].append({'x_jun': jun_t, 'x_sen': sen_t, 'y': float(coef_sen), 'ci_lower': float(ci_sen[0]), 'ci_upper': float(ci_sen[1]), 'p': float(p_sen)})
            else:
                sens3_results[out_title]['seniors'].append({'x_jun': jun_t, 'x_sen': sen_t, 'y': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p': 1.0})

    with open('Jsons/fig_sensitivity.json', 'w') as f:
        json.dump(sens3_results, f, cls=NpEncoder)

    return {}


# ==============================================================================
# 8. INTER-RATER AGREEMENT & HUMAN VS. LLM CONCORDANCE
# ==============================================================================

def analyze_raters(df_raw):
    """
    Computes inter-rater reliability datasets for:
      1) Pairwise human expert raters (Figure A1: rater1.png)
      2) Human vs LLM evaluators (Figure A2: rater_human_vs_llm.png)
      3) Disaggregated subscales across modalities
    Outputs: Jsons/fig_raters.json and Jsons/fig_human_llm_subscales.json.
    """
    out_data = {
        "rater1": {},
        "human_vs_llm": {}
    }
    tasks = [
        ('tt1_sum_rat_drades', '10-Day Drafting'),
        ('tt2_sum_rat_drades', '90-Day Drafting'),
        ('tt2_sum_rat_cri', '90-Day Redlining')
    ]
    
    df_h = df_raw[df_raw['Rater_Type'].str.lower() == 'human']
    df_h_mean = df_h.groupby(UNIQUE_ID_VAR).mean(numeric_only=True)
    df_l_mean = df_raw[df_raw['Rater_Type'].str.lower() == 'llm'].groupby(UNIQUE_ID_VAR).mean(numeric_only=True)
    common_ids = df_h_mean.index.intersection(df_l_mean.index)

    for var, label in tasks:
        # Pairwise human raters
        h_data = df_h.dropna(subset=[UNIQUE_ID_VAR, var])
        counts = h_data[UNIQUE_ID_VAR].value_counts()
        valid_ids = counts[counts >= 2].index
        r1_vals, r2_vals = [], []
        for uid in valid_ids:
            vals = h_data[h_data[UNIQUE_ID_VAR] == uid][var].values
            r1_vals.append(float(vals[0]))
            r2_vals.append(float(vals[1]))
        r_human, p_human = stats.pearsonr(r1_vals, r2_vals) if len(r1_vals) > 2 else (0.0, 1.0)
        out_data["rater1"][var] = {
            "label": label,
            "x": r1_vals,
            "y": r2_vals,
            "r": float(r_human),
            "p": float(p_human),
            "n": len(r1_vals)
        }

        # Human vs LLM raters
        h_vals, l_vals = [], []
        if var in df_h_mean.columns and var in df_l_mean.columns:
            h_s = df_h_mean.loc[common_ids, var]
            l_s = df_l_mean.loc[common_ids, var]
            valid = h_s.notna() & l_s.notna()
            h_vals = h_s[valid].astype(float).tolist()
            l_vals = l_s[valid].astype(float).tolist()
        r_hl, p_hl = stats.pearsonr(h_vals, l_vals) if len(h_vals) > 2 else (0.0, 1.0)
        out_data["human_vs_llm"][var] = {
            "label": label,
            "x": h_vals,
            "y": l_vals,
            "r": float(r_hl),
            "p": float(p_hl),
            "n": len(h_vals)
        }

    # Human vs LLM subscales across the 3 tasks
    subelement_map = {
        'enf': 'Enforceability',
        'tec': 'Technical Accuracy',
        'str': 'Strategic Ambiguity',
        'com': 'Completeness and Alignment',
        'cla': 'Clarity'
    }
    
    tasks_info = [
        ('tt1_rat_drades', '10-Day Drafting'),
        ('tt2_rat_drades', '90-Day Drafting'),
        ('tt2_rat_cri', '90-Day Redlining')
    ]
    
    out_data["human_llm_subscales"] = {}
    
    for dim in MAIN_OUTCOME_DIMS:
        dim_label = subelement_map.get(dim, dim)
        out_data["human_llm_subscales"][dim] = {
            "label": dim_label,
            "tasks": {}
        }
        
        for task_prefix, task_label in tasks_info:
            var = f"{task_prefix}_{dim}"
            h_vals, l_vals = [], []
            if var in df_h_mean.columns and var in df_l_mean.columns:
                h_s = df_h_mean.loc[common_ids, var]
                l_s = df_l_mean.loc[common_ids, var]
                valid = h_s.notna() & l_s.notna()
                h_vals = h_s[valid].astype(float).tolist()
                l_vals = l_s[valid].astype(float).tolist()
                
            r_val, p_val = stats.pearsonr(h_vals, l_vals) if len(h_vals) > 2 and np.std(h_vals) > 0 and np.std(l_vals) > 0 else (0.0, 1.0)
            
            out_data["human_llm_subscales"][dim]["tasks"][task_prefix] = {
                "label": task_label,
                "var": var,
                "x": h_vals,
                "y": l_vals,
                "r": float(r_val),
                "p": float(p_val),
                "n": len(h_vals)
            }

    with open('Jsons/fig_human_llm_subscales.json', 'w') as f:
        json.dump(out_data["human_llm_subscales"], f, cls=NpEncoder)

    with open('Jsons/fig_raters.json', 'w') as f:
        json.dump(out_data, f, cls=NpEncoder)

    return {}


# ==============================================================================
# 9. LONGITUDINAL COPILOT USAGE DYNAMICS
# ==============================================================================

def analyze_usage(df_main):
    """
    Constructs 14-day daily moving average active user time series aligned to
    each firm's onboarding date, along with 10-day and 90-day task window boundaries.
    Outputs: Jsons/fig_usage_alt.json (Figure A4: Copilot Adoption Dynamics).
    """
    from auth import get_github_pat

    github_pat = get_github_pat()
    df_ts = fetch_csv_with_fallback('All usage measures - Merged', github_pat, GITHUB_CONFIG)
    df_key = fetch_csv_with_fallback('All usage measures - Domain Key', github_pat, GITHUB_CONFIG)
    df_dates_raw = fetch_csv_with_fallback('Participant-level tasking hub', github_pat, config=GITHUB_CONFIG)

    # Apply sample exclusions to usage data right at the loading stage
    if EXCLUDE_INDIVIDUALS or EXCLUDE_FIRMS:
        excluded_ind_emails = load_exclusion_list(EXCLUSION_INDIVIDUALS_FILE) if EXCLUDE_INDIVIDUALS else set()
        excluded_firm_terms = load_exclusion_list(EXCLUSION_FIRMS_FILE) if EXCLUDE_FIRMS else set()
        firm_names, firm_nums, firm_emails, firm_doms = get_firm_exclusion_mappings(
            df_dates_raw, excluded_firm_terms, unique_id_var=UNIQUE_ID_VAR, firm_var=FIRM_VAR
        )
        all_excluded_emails = excluded_ind_emails | firm_emails

        if df_dates_raw is not None and not df_dates_raw.empty:
            email_col = next((c for c in df_dates_raw.columns if 'email' in c.lower()), None)
            firm_col = next((c for c in df_dates_raw.columns if 'firm' in c.lower() and 'name' in c.lower()), None)
            firm_num_col = next((c for c in df_dates_raw.columns if 'firm' in c.lower() and ('num' in c.lower() or 'number' in c.lower())), None)

            d_mask = pd.Series(True, index=df_dates_raw.index)
            if email_col and all_excluded_emails:
                d_mask &= ~df_dates_raw[email_col].astype(str).str.strip().str.lower().isin(all_excluded_emails)
            if firm_col and firm_names:
                d_mask &= ~df_dates_raw[firm_col].astype(str).str.strip().str.lower().isin(firm_names)
            if firm_num_col and firm_nums:
                d_mask &= ~df_dates_raw[firm_num_col].astype(str).str.strip().str.lower().str.replace(r'\.0$', '', regex=True).isin(firm_nums)
            n_before_dates = len(df_dates_raw)
            df_dates_raw = df_dates_raw[d_mask].copy()
            print(f"  [EXCLUSIONS] Dropped {n_before_dates - len(df_dates_raw)} rows from tasking hub dates dataset.")

        if df_key is not None and not df_key.empty and EXCLUDE_FIRMS and (firm_names or firm_doms):
            dom_col = next((c for c in df_key.columns if 'domain' in c.lower()), 'userDomain')
            fn_col = next((c for c in df_key.columns if 'firm' in c.lower()), 'firm_name')
            k_mask = pd.Series(True, index=df_key.index)
            if dom_col in df_key.columns and firm_doms:
                k_mask &= ~df_key[dom_col].astype(str).str.strip().str.lower().isin(firm_doms)
            if fn_col in df_key.columns and firm_names:
                k_mask &= ~df_key[fn_col].astype(str).str.strip().str.lower().isin(firm_names)
            n_before_key = len(df_key)
            df_key = df_key[k_mask].copy()
            print(f"  [EXCLUSIONS] Dropped {n_before_key - len(df_key)} rows from domain key dataset.")

        if df_ts is not None and not df_ts.empty and EXCLUDE_FIRMS and firm_doms:
            dom_col = next((c for c in df_ts.columns if 'domain' in c.lower()), 'userDomain')
            if dom_col in df_ts.columns:
                n_before_ts = len(df_ts)
                df_ts = df_ts[~df_ts[dom_col].astype(str).str.strip().str.lower().isin(firm_doms)].copy()
                print(f"  [EXCLUSIONS] Dropped {n_before_ts - len(df_ts)} rows from usage timeseries dataset.")

    if df_ts is not None and not df_ts.empty and df_dates_raw is not None and not df_dates_raw.empty:
        df_dates_copy = df_dates_raw.copy()
        df_dates_copy.columns = df_dates_copy.columns.astype(str).str.strip()
        onboard_col = next((c for c in df_dates_copy.columns if 'onboard' in c.lower()), '0. Date of onboarding email')
        tt1_s_col = next((c for c in df_dates_copy.columns if 'tt1' in c.lower() and 'sent' in c.lower()), '3. Date TT1 sent')
        tt1_r_col = next((c for c in df_dates_copy.columns if c.strip() == '4. TT1' or ('tt1' in c.lower() and 'received' in c.lower())), '4. TT1')
        tt2_s_col = next((c for c in df_dates_copy.columns if 'tt2' in c.lower() and 'sent' in c.lower()), '6. Date TT2 sent')
        tt2_r_col = next((c for c in df_dates_copy.columns if c.strip() == '7. TT2' or ('tt2' in c.lower() and 'received' in c.lower())), '7. TT2')
        firm_col = next((c for c in df_dates_copy.columns if 'firm' in c.lower() and 'name' in c.lower()), 'Firm name')

        df_dates_copy['onboard'] = pd.to_datetime(df_dates_copy[onboard_col], errors='coerce')
        df_dates_copy['tt1_s'] = pd.to_datetime(df_dates_copy[tt1_s_col], errors='coerce')
        df_dates_copy['tt1_r'] = pd.to_datetime(df_dates_copy[tt1_r_col], errors='coerce')
        df_dates_copy['tt2_s'] = pd.to_datetime(df_dates_copy[tt2_s_col], errors='coerce')
        df_dates_copy['tt2_r'] = pd.to_datetime(df_dates_copy[tt2_r_col], errors='coerce')

        firm_dates_agg = df_dates_copy.groupby(firm_col).agg({
            'onboard': 'min',
            'tt1_s': 'mean',
            'tt1_r': 'mean',
            'tt2_s': 'mean',
            'tt2_r': 'mean'
        }).dropna(subset=['onboard']).reset_index()

        t1_start = float((firm_dates_agg['tt1_s'] - firm_dates_agg['onboard']).dt.days.mean())
        t1_end = float((firm_dates_agg['tt1_r'] - firm_dates_agg['onboard']).dt.days.mean())
        t2_start = float((firm_dates_agg['tt2_s'] - firm_dates_agg['onboard']).dt.days.dropna().mean())
        t2_end = float((firm_dates_agg['tt2_r'] - firm_dates_agg['onboard']).dt.days.dropna().mean())

        firm_dates_agg['firm_clean'] = firm_dates_agg[firm_col].str.strip().str.lower()
        df_key_copy = df_key.copy()
        domain_col_key = next((c for c in df_key_copy.columns if 'domain' in c.lower()), 'userDomain')
        firm_col_key = next((c for c in df_key_copy.columns if 'firm' in c.lower()), 'Firm Name')
        df_key_copy['firm_clean'] = df_key_copy[firm_col_key].str.strip().str.lower()
        df_key_copy['join_key'] = df_key_copy[domain_col_key].str.strip().str.lower()

        df_firm_map = pd.merge(df_key_copy, firm_dates_agg, on='firm_clean', how='left')

        f18_match = firm_dates_agg[firm_dates_agg['firm_clean'].str.contains(r'firm\s*18')]
        if not f18_match.empty and 'firm_18_domain' not in df_firm_map['join_key'].values:
            row = f18_match.iloc[0].to_dict()
            row['join_key'] = 'firm_18_domain'
            row[firm_col_key] = row[firm_col]
            df_firm_map = pd.concat([df_firm_map, pd.DataFrame([row])], ignore_index=True)

        df_ts_clean = df_ts.copy()
        df_ts_clean['Week Beginning'] = pd.to_datetime(df_ts_clean['Week Beginning'], errors='coerce')
        domain_col_ts = next((c for c in df_ts_clean.columns if 'domain' in c.lower()), 'userDomain')
        df_ts_clean['join_key'] = df_ts_clean[domain_col_ts].str.strip().str.lower()
        if 'Active Users' in df_ts_clean.columns:
            df_ts_clean['Active Users'] = df_ts_clean['Active Users'].astype(str).str.replace(',', '', regex=False)
            df_ts_clean['Active Users'] = pd.to_numeric(df_ts_clean['Active Users'], errors='coerce').fillna(0)

        df_usage_onboard = pd.merge(df_ts_clean, df_firm_map[['join_key', 'onboard', firm_col_key]], on='join_key', how='inner')

        daily_records = []
        for _, row in df_usage_onboard.iterrows():
            w_start = row['Week Beginning']
            onboard = row['onboard']
            if pd.isna(onboard): continue
            u = row['Active Users']
            for day_offset in range(7):
                curr_date = w_start + pd.Timedelta(days=day_offset)
                rel_day = (curr_date - onboard).days
                daily_records.append({
                    'rel_day': rel_day,
                    'active_users_daily': u / 7.0,
                    'firm': row[firm_col_key]
                })

        df_daily = pd.DataFrame(daily_records)
        df_daily_agg = df_daily[(df_daily['rel_day'] >= 0) & (df_daily['rel_day'] <= 210)].groupby('rel_day')['active_users_daily'].sum().reset_index()
        all_days = pd.DataFrame({'rel_day': np.arange(0, 211)})
        df_daily_full = pd.merge(all_days, df_daily_agg, on='rel_day', how='left').fillna(0)
        df_daily_full['ma_14d_weekly'] = df_daily_full['active_users_daily'].rolling(window=14, min_periods=1, center=True).mean() * 7.0

        series_data = df_daily_full.rename(columns={'rel_day': 'days_since_onboarding', 'ma_14d_weekly': 'active_users'}).to_dict(orient='records')

        out_alt = {
            'task1_window': [t1_start, t1_end],
            'task2_window': [t2_start, t2_end],
            'series': series_data
        }
        with open('Jsons/fig_usage_alt.json', 'w') as f:
            json.dump(out_alt, f, cls=NpEncoder)

    return {}


# ==============================================================================
# 10. CORRELATIONS & DISTRIBUTIONAL VARIANCE REGRESSIONS
# ==============================================================================

def cronbach_alpha(df):
    """Calculates Cronbach's alpha internal consistency coefficient across quality items."""
    df_corr = df.dropna()
    k = df_corr.shape[1]
    if k < 2: return np.nan
    v_i = df_corr.var(ddof=1).sum()
    v_t = df_corr.sum(axis=1).var(ddof=1)
    if v_t == 0: return np.nan
    return (k / (k - 1)) * (1 - v_i / v_t)


def analyze_correlations_logic(df_raw, df_cb):
    """
    Computes inter-item correlation matrices and Cronbach's alpha internal consistency
    for Human, LLM, and Pooled evaluators across all three experimental exercises.
    Outputs: Jsons/additional_analysis.json (Tables A3 & A4: Correlation Tables).
    """
    tasks = [
        ('tt1_drades', 'A. 10-day Drafting Task', '10_day_draft'),
        ('tt2_drades', 'B. 90-day Drafting Task', '90_day_draft'),
        ('tt2_cri', 'C. 90-day Redlining Task', '90_day_redline')
    ]
    var_labels = ['Enforceability', 'Accuracy', 'Ambiguity', 'Completeness/Alignment', 'Clarity']
    out_data = {}
    
    for r_type in ['human', 'llm', 'pooled']:
        if r_type == 'pooled':
            df_slice = df_raw.copy()
        else:
            df_slice = df_raw[df_raw['Rater_Type'] == r_type].copy()
            
        df_corr = df_slice.groupby(UNIQUE_ID_VAR).mean(numeric_only=True)
        task_res = []
        
        for t_key, t_title, m_key in tasks:
            sub_vars = RATER_OUTCOMES.get(t_key, [])
            v_cols = [v for v in sub_vars if v in df_corr.columns]
            if not v_cols: continue
            
            df_sub = df_corr[v_cols].dropna()
            alpha = cronbach_alpha(df_sub)
            n_obs = len(df_sub)
            c_mat = df_sub.corr().values.tolist()
            
            task_res.append({
                'title': t_title,
                'm_key': m_key,
                'alpha': alpha,
                'n_obs': n_obs,
                'c_mat': c_mat,
                'v_cols': len(v_cols),
                'var_labels': var_labels
            })
            
        out_data[r_type] = task_res
        
    return out_data


def analyze_varregs_logic(df_raw, df_cb, include_small_firms=None):
    """
    Estimates distributional impact regressions across quality quintiles and
    Levene equality-of-variance tests with firm fixed effects.
    Outputs: Jsons/additional_analysis.json (Tables 5 & 7: Variance Regressions).
    """
    if include_small_firms is None:
        include_small_firms = VARREGS_INCLUDE_LOW_N_FIRMS

    df_filtered = df_raw.copy()
    OUTCOME_VARS = ['tt1_sum_rat_drades', 'tt2_sum_rat_drades', 'tt2_sum_rat_cri']
    OUTCOME_KEYS = ['tt1_drades', 'tt2_drades', 'tt2_cri']
    SUBGROUPS = {None: 'Full Sample', 'junior == 1': 'Juniors', 'senior == 1': 'Seniors'}
    df_panel = df_filtered.dropna(subset=[UNIQUE_ID_VAR]).copy()
    df_panel[FIRM_VAR] = df_panel[FIRM_VAR].astype(str)
    
    if 'rater_type' in df_panel.columns:
        df_panel = df_panel[df_panel['rater_type'] == 'human'].copy()
        
    if not include_small_firms:
        firm_counts = df_panel.groupby(FIRM_VAR)[UNIQUE_ID_VAR].nunique()
        valid_firms = firm_counts[firm_counts > 2].index
        df_panel = df_panel[df_panel[FIRM_VAR].isin(valid_firms)].copy()
        
    for col in OUTCOME_VARS:
        if col in df_panel.columns:
            df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce') - 1
            
    for col in OUTCOME_VARS:
        valid_mask = df_panel[col].notna()
        control_mask = (df_panel[TREATMENT_VAR] == 0) & valid_mask
        control_mean = df_panel.loc[control_mask, col].mean()
        control_sd = df_panel.loc[control_mask, col].std()
        
        if pd.notna(control_sd) and control_sd > 0:
            df_panel[col + '_std'] = (df_panel[col] - control_mean) / control_sd
        else:
            df_panel[col + '_std'] = df_panel[col]
            
        pct = df_panel.loc[valid_mask, col + '_std'].rank(pct=True, method='average')
        df_panel.loc[valid_mask, col + '_q1'] = (pct > 0.80).astype(int)
        df_panel.loc[valid_mask, col + '_q2'] = ((pct > 0.60) & (pct <= 0.80)).astype(int)
        df_panel.loc[valid_mask, col + '_q3'] = ((pct > 0.40) & (pct <= 0.60)).astype(int)
        df_panel.loc[valid_mask, col + '_q4'] = ((pct > 0.20) & (pct <= 0.40)).astype(int)
        df_panel.loc[valid_mask, col + '_q5'] = (pct <= 0.20).astype(int)
        
    results = {}
    for sg_name in SUBGROUPS.values():
        results[sg_name] = {out: {} for out in OUTCOME_VARS}
        
    for sg_filter, sg_name in SUBGROUPS.items():
        df_sg = df_panel.query(sg_filter).copy() if sg_filter else df_panel.copy()
        
        for o_idx, out in enumerate(OUTCOME_VARS):
            weight_col = f"wls_weight_{OUTCOME_KEYS[o_idx]}_human"
            if weight_col not in df_sg.columns: df_sg[weight_col] = 1.0
            
            df_out = df_sg.dropna(subset=[out + '_std', weight_col]).copy()
            df_out = df_out[df_out[weight_col] > 0]
            
            treat_vals = df_out.loc[df_out[TREATMENT_VAR] == 1, out + '_std']
            ctrl_vals = df_out.loc[df_out[TREATMENT_VAR] == 0, out + '_std']
            
            if len(treat_vals) > 1 and len(ctrl_vals) > 1:
                lev_stat, lev_pval = stats.levene(treat_vals, ctrl_vals, center='median')
            else:
                lev_stat, lev_pval = np.nan, np.nan
                
            results[sg_name][out]['Levene'] = {'stat': lev_stat, 'pval': lev_pval}
            
            if len(df_out) == 0: continue
            ref_firm = df_out[FIRM_VAR].value_counts().idxmax()
            
            for dummy in ['q1', 'q2', 'q3', 'q4', 'q5']:
                formula = f"{out}_{dummy} ~ {TREATMENT_VAR} + C({FIRM_VAR}, Treatment(reference='{ref_firm}'))"
                model = smf.wls(formula, data=df_out, weights=df_out[weight_col]).fit(cov_type='HC1')
                results[sg_name][out][dummy] = {'coef': model.params.get(TREATMENT_VAR, np.nan), 'se': model.bse.get(TREATMENT_VAR, np.nan), 'pval': model.pvalues.get(TREATMENT_VAR, np.nan), 'obs': int(model.nobs)}
                    
    return results


def analyze_varregs_noFFE_logic(df_raw, df_cb, include_small_firms=None):
    """
    Estimates distributional impact regressions across quality quintiles and
    Levene equality-of-variance tests without firm fixed effects.
    Outputs: Jsons/additional_analysis.json.
    """
    if include_small_firms is None:
        include_small_firms = VARREGS_INCLUDE_LOW_N_FIRMS

    df_filtered = df_raw.copy()
    OUTCOME_VARS = ['tt1_sum_rat_drades', 'tt2_sum_rat_drades', 'tt2_sum_rat_cri']
    OUTCOME_KEYS = ['tt1_drades', 'tt2_drades', 'tt2_cri']
    SUBGROUPS = {None: 'Full Sample', 'junior == 1': 'Juniors', 'senior == 1': 'Seniors'}
    df_panel = df_filtered.dropna(subset=[UNIQUE_ID_VAR]).copy()
    df_panel[FIRM_VAR] = df_panel[FIRM_VAR].astype(str)
    
    if 'rater_type' in df_panel.columns:
        df_panel = df_panel[df_panel['rater_type'] == 'human'].copy()
        
    if not include_small_firms:
        firm_counts = df_panel.groupby(FIRM_VAR)[UNIQUE_ID_VAR].nunique()
        valid_firms = firm_counts[firm_counts > 2].index
        df_panel = df_panel[df_panel[FIRM_VAR].isin(valid_firms)].copy()
        
    for col in OUTCOME_VARS:
        if col in df_panel.columns:
            df_panel[col] = pd.to_numeric(df_panel[col], errors='coerce') - 1
            
    for col in OUTCOME_VARS:
        valid_mask = df_panel[col].notna()
        control_mask = (df_panel[TREATMENT_VAR] == 0) & valid_mask
        control_mean = df_panel.loc[control_mask, col].mean()
        control_sd = df_panel.loc[control_mask, col].std()
        
        if pd.notna(control_sd) and control_sd > 0:
            df_panel[col + '_std'] = (df_panel[col] - control_mean) / control_sd
        else:
            df_panel[col + '_std'] = df_panel[col]
            
        pct = df_panel.loc[valid_mask, col + '_std'].rank(pct=True, method='average')
        df_panel.loc[valid_mask, col + '_q1'] = (pct > 0.80).astype(int)
        df_panel.loc[valid_mask, col + '_q2'] = ((pct > 0.60) & (pct <= 0.80)).astype(int)
        df_panel.loc[valid_mask, col + '_q3'] = ((pct > 0.40) & (pct <= 0.60)).astype(int)
        df_panel.loc[valid_mask, col + '_q4'] = ((pct > 0.20) & (pct <= 0.40)).astype(int)
        df_panel.loc[valid_mask, col + '_q5'] = (pct <= 0.20).astype(int)
        
    results = {}
    for sg_name in SUBGROUPS.values():
        results[sg_name] = {out: {} for out in OUTCOME_VARS}
        
    for sg_filter, sg_name in SUBGROUPS.items():
        df_sg = df_panel.query(sg_filter).copy() if sg_filter else df_panel.copy()
        
        for o_idx, out in enumerate(OUTCOME_VARS):
            weight_col = f"wls_weight_{OUTCOME_KEYS[o_idx]}_human"
            if weight_col not in df_sg.columns: df_sg[weight_col] = 1.0
            
            df_out = df_sg.dropna(subset=[out + '_std', weight_col]).copy()
            df_out = df_out[df_out[weight_col] > 0]
            
            treat_vals = df_out.loc[df_out[TREATMENT_VAR] == 1, out + '_std']
            ctrl_vals = df_out.loc[df_out[TREATMENT_VAR] == 0, out + '_std']
            
            if len(treat_vals) > 1 and len(ctrl_vals) > 1:
                lev_stat, lev_pval = stats.levene(treat_vals, ctrl_vals, center='median')
            else:
                lev_stat, lev_pval = np.nan, np.nan
                
            results[sg_name][out]['Levene'] = {'stat': lev_stat, 'pval': lev_pval}
            
            if len(df_out) == 0: continue
            
            for dummy in ['q1', 'q2', 'q3', 'q4', 'q5']:
                formula = f"{out}_{dummy} ~ {TREATMENT_VAR}"
                model = smf.wls(formula, data=df_out, weights=df_out[weight_col]).fit(cov_type='HC1')
                results[sg_name][out][dummy] = {'coef': model.params.get(TREATMENT_VAR, np.nan), 'se': model.bse.get(TREATMENT_VAR, np.nan), 'pval': model.pvalues.get(TREATMENT_VAR, np.nan), 'obs': int(model.nobs)}
                    
    return results


def analyze_additional(df_raw, df_cb):
    """
    Coordinates additional empirical analyses (inter-rater correlations, distributional
    variance regressions with and without firm fixed effects).
    Outputs: Jsons/additional_analysis.json.
    """
    all_additional_tex = {}
    
    corr_data = analyze_correlations_logic(df_raw, df_cb)
    all_additional_tex.update(corr_data)
    
    varregs_data = analyze_varregs_logic(df_raw, df_cb)
    all_additional_tex['varregs'] = varregs_data
    varregs_noFFE_data = analyze_varregs_noFFE_logic(df_raw, df_cb)
    all_additional_tex['varregs_noFFE'] = varregs_noFFE_data
        
    macros_ext = {}
    df_h_mean = df_raw[df_raw['Rater_Type'] == 'human'].groupby(UNIQUE_ID_VAR).mean(numeric_only=True)
    df_l_mean = df_raw[df_raw['Rater_Type'] == 'llm'].groupby(UNIQUE_ID_VAR).mean(numeric_only=True)
    common_ids = df_h_mean.index.intersection(df_l_mean.index)
    
    df_h = df_raw[df_raw['Rater_Type'] == 'human']
    for var in ['tt1_sum_rat_drades', 'tt2_sum_rat_drades', 'tt2_sum_rat_cri']:
        if var in df_h_mean.columns and var in df_l_mean.columns:
            h_vals = df_h_mean.loc[common_ids, var]
            l_vals = df_l_mean.loc[common_ids, var]
            valid = h_vals.notna() & l_vals.notna()
            if valid.sum() > 2:
                r, p = stats.pearsonr(h_vals[valid], l_vals[valid])
                macros_ext[f'rater_human_vs_llm_r_{var}'] = f"{r:.3f}"
                macros_ext[f'rater_human_vs_llm_p_{var}'] = f"{p:.3f}"
                
        if var in df_h.columns:
            h_data = df_h.dropna(subset=[UNIQUE_ID_VAR, var])
            counts = h_data[UNIQUE_ID_VAR].value_counts()
            valid_ids = counts[counts >= 2].index
            
            r1_vals = []
            r2_vals = []
            for uid in valid_ids:
                vals = h_data[h_data[UNIQUE_ID_VAR] == uid][var].values
                r1_vals.append(vals[0])
                r2_vals.append(vals[1])
                
            if len(r1_vals) > 2:
                r, p = stats.pearsonr(r1_vals, r2_vals)
                macros_ext[f'rater_correlations_r_{var}'] = f"{r:.3f}"
                macros_ext[f'rater_correlations_p_{var}'] = f"{p:.3f}"
                
    all_additional_tex['macros'] = macros_ext
        
    with open('Jsons/additional_analysis.json', 'w') as f:
        json.dump(all_additional_tex, f, cls=NpEncoder)
        
    macros = {}
    if 'macros' in all_additional_tex:
        macros.update(all_additional_tex['macros'])
    if 'varregs' in all_additional_tex and 'macros' in all_additional_tex['varregs']:
        macros.update(all_additional_tex['varregs']['macros'])

    for out in ['tt1_sum_rat_drades', 'tt2_sum_rat_drades', 'tt2_sum_rat_cri']:
        if out in df_raw.columns:
            h = df_raw[df_raw['Rater_Type'] == 'human'].groupby(UNIQUE_ID_VAR)[out].mean()
            l = df_raw[df_raw['Rater_Type'] == 'llm'].groupby(UNIQUE_ID_VAR)[out].mean()
            common = h.index.intersection(l.index)
            if len(common) > 1:
                r, p = stats.pearsonr(h.loc[common], l.loc[common])
                macros[f"rater_human_vs_llm_r_{out}"] = f"{r:.3f}"
                macros[f"rater_human_vs_llm_p_{out}"] = f"{p:.3f}"
        
        df_sub = df_raw.dropna(subset=[out, TREATMENT_VAR]).copy()
        if len(df_sub) > 0:
            c = df_sub[df_sub[TREATMENT_VAR]==0][out]
            t = df_sub[df_sub[TREATMENT_VAR]==1][out]
            if len(c) > 0 and len(t) > 0:
                stat, p = stats.levene(c, t)
                out_clean = out.replace("rat_drades", "drades")
                macros[f"violin_levene_stat_{out_clean}_all"] = f"{stat:.3f}"
                macros[f"violin_levene_pval_{out_clean}_all"] = f"{p:.3f}"
                
    return macros


# ==============================================================================
# 11. MASTER REPLICATION ENGINE
# ==============================================================================

def run_all(github_pat=None):
    """
    Executes the complete empirical pipeline:
      1. Loads raw CSV datasets and applies full cleaning and standardization.
      2. Runs all 14 estimation modules across tables and figures.
      3. Exports serialized statistical payloads into Jsons/*.json.
      4. Returns the master macro dictionary for LaTeX document compilation.
    """
    if github_pat is None:
        import auth
        github_pat = auth.get_github_pat()

    df_subject, df_raw, df_cb, global_ref_firm, df_unf = load_and_prep_data(github_pat)
    if df_subject is None:
        raise RuntimeError("Data load failed. Check dataset paths in data/.")

    print("Running analysis...")
    macros = {}
    macros.update(analyze_balance(df_subject))
    macros.update(analyze_firm_summary(df_subject, df_unf))
    macros.update(analyze_attrition(df_subject, df_unf))
    macros.update(analyze_takeup_completion(df_subject))
    macros.update(analyze_summary_table(df_subject, df_raw))
    
    global_largest_firm = df_raw.drop_duplicates(subset=[UNIQUE_ID_VAR])[FIRM_VAR].mode()[0]
    global_ref_firm = f"'{global_largest_firm}'" if isinstance(global_largest_firm, str) else f"{global_largest_firm}"
    
    macros.update(analyze_models(df_subject, df_raw, df_cb, global_ref_firm, global_largest_firm, std_use_pooled=True))
    macros.update(analyze_fisher(df_raw))
    macros.update(analyze_cdf(df_raw))
    macros.update(analyze_histograms(df_raw))
    macros.update(analyze_forest(df_raw, df_cb))
    macros.update(analyze_leave_one_firm_out(df_raw))
    macros.update(analyze_sensitivity(df_raw))
    macros.update(analyze_raters(df_raw))
    macros.update(analyze_usage(df_subject))
    macros.update(analyze_additional(df_raw, df_cb))
    
    print("Analysis complete. Data exported to Jsons/ directory.")
    return macros
    
if __name__ == '__main__':
    run_all()
