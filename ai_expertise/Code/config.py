# ==============================================================================
# config.py
# ------------------------------------------------------------------------------
# Global Configuration & Experimental Parameter Definitions
# "Artificial Intelligence in High-Skill Knowledge Work: Evidence from Patent
#  Drafting and Prosecution"
# ------------------------------------------------------------------------------
# Role & Architecture:
#   This module defines global parameters, variable mappings, evaluation thresholds,
#   standard error configurations, and GitHub synchronization endpoints across
#   the replication package.
# ==============================================================================

import os
from dotenv import load_dotenv

# Load local environment variables (.env file)
load_dotenv()
GITHUB_PAT = os.environ.get("GITHUB_PAT")

# ==============================================================================
# 1. EXPERIMENTAL DESIGN & SAMPLE STRATIFICATION
# ==============================================================================

# Experience cutoff for senior vs junior stratification:
# Seniors: Experience >= 7 years; Juniors: Experience < 7 years
EXPERIENCE_THRESHOLD = 7

# Primary treatment indicator column and treatment groups
TREATMENT_VAR = 'group_binary'
FIRM_VAR = 'firm_num'
UNIQUE_ID_VAR = 'email'
CONTROL_GROUP = 'Group 1'
TREATMENT_GROUP = 'Group 2'

# Standardization distribution choice:
# False = Normalize scores using the Control Group mean and SD (standard academic practice)
# True  = Normalize scores using the full pooled sample
STD_USE_POOLED = False

# Variance regressions firm inclusion:
# True  = Include all firms regardless of size
# False = Exclude micro-firms with <= 2 participants
VARREGS_INCLUDE_LOW_N_FIRMS = True

# Overwrite LaTeX macro definitions on each execution run
WIPE_MACROS_ON_RUN = True

# ==============================================================================
# SAMPLE EXCLUSION TOGGLES & CONFIGURATION
# ==============================================================================

# Individual practitioner exclusion toggle and file path:
# True  = Exclude specific individual email addresses listed in EXCLUSION_INDIVIDUALS_FILE (removing individuals who declined consent to publish)
# False = Include all individual practitioners (default)
EXCLUDE_INDIVIDUALS = True
EXCLUSION_INDIVIDUALS_FILE = 'exclusion_individuals.txt'

# Firm-level exclusion toggle and file path:
# True  = Exclude all observations and usage data from specific firms listed in EXCLUSION_FIRMS_FILE
# False = Include all firms (default)
EXCLUDE_FIRMS = False
EXCLUSION_FIRMS_FILE = 'exclusion_firms.txt'

# ==============================================================================
# 2. GITHUB SYNCHRONIZATION SETTINGS
# ==============================================================================

USE_ALT_REPO = True

if USE_ALT_REPO:
    GITHUB_CONFIG = {
        'OWNER': 'joshmartingoogle',
        'REPO': 'W3-Paper',
        'BRANCH': 'main',
        'TARGET_DIR': 'New',
        'MACRO_FILE_PATH': 'New/macros.tex'
    }
else:
    GITHUB_CONFIG = {
        'OWNER': 'joshmartingoogle',
        'REPO': 'Workstream-3---David-Autor',
        'BRANCH': 'main',
        'TARGET_DIR': 'New',
        'MACRO_FILE_PATH': 'New/macros.tex'
    }

# ==============================================================================
# 3. TASK DIMENSIONS & SECONDARY SURVEY OUTCOMES
# ==============================================================================

# Five canonical patent quality dimensions evaluated by expert and LLM raters:
# enf = Enforceability
# tec = Technical Accuracy
# str = Strategic Ambiguity
# com = Completeness & Alignment
# cla = Clarity
MAIN_OUTCOME_DIMS = ['enf', 'tec', 'str', 'com', 'cla']

# Rater-level outcome variable groupings across tasks
RATER_OUTCOMES = {
    'tt1_drades': [f'tt1_rat_drades_{d}' for d in MAIN_OUTCOME_DIMS],
    'tt2_drades': [f'tt2_rat_drades_{d}' for d in MAIN_OUTCOME_DIMS],
    'tt2_cri': [f'tt2_rat_cri_{d}' for d in MAIN_OUTCOME_DIMS]
}

# Secondary survey and administrative metrics
SECONDARY_OUTCOMES = {
    "TimeOnTask": {
        "vars": ['tt1_sum_dra_tot', 'tt2_sum_dra_tot', 'tt2_sum_cri_tot']
    },
    "DraftingSurvey": {
        "vars": [
            'tt1_sur_dra_spe', 'tt1_sur_dra_qua', 'tt1_sur_dra_sat',
            'tt2_sur_dra_spe', 'tt2_sur_dra_qua', 'tt2_sur_dra_sat'
        ]
    },
    "Patent": {
        "vars": ['pat_num', 'pat_avg_res', 'pat_avg_len', 'pat_avg_rev', 'pat_avg_cor']
    }
}

# Standard error estimation settings
SE_TYPE_SUBJECT = 'HC1'
SE_NOTE_SUBJECT = "robust (HC1) standard errors"
SE_NOTE_RATING = "standard errors clustered at the individual level"
WLS_NOTE_RATING = r" Models are weighted by $w_i=1/n_i$, where $n_i$ is the number of ratings individual $i$ receives so that each individual carries equal total weight."
