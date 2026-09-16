# Replication Package: Does AI Assistance Enhance or Erode Expertise? Evidence from a Three-Month Field Experiment in Patent Drafting

[![Replication Status](https://img.shields.io/badge/Replication-100%25%20Verified-brightgreen)](file:///usr/local/google/home/joshuabmartin/InFlow%202/replicate.sh)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](file:///usr/local/google/home/joshuabmartin/InFlow%202/requirements.txt)
[![LaTeX TeXLive](https://img.shields.io/badge/LaTeX-TeXLive%202022%2B-orange)](file:///usr/local/google/home/joshuabmartin/InFlow%202/draft_2026Aug20.tex)

This repository contains the complete, self-contained replication package for the study:

> **"Does AI Assistance Enhance or Erode Expertise? Evidence from a Three-Month Field Experiment in Patent Drafting"**  


This document combines the full replication guide and data codebook into a single comprehensive manual. It describes the end-to-end empirical pipeline, data anonymization protocols, econometric methodology, and every variable in the experimental datasets so that a replicator can reproduce every table, figure, statistical macro, and the compiled manuscript PDF precisely to the letter.

---

## Table of Contents

1. [Quick Start Replication](#1-quick-start-replication)
2. [System & Software Requirements](#2-system--software-requirements)
3. [Repository Directory & File Architecture](#3-repository-directory--file-architecture)
4. [Comprehensive Data Codebook & Dataset Documentation](#4-comprehensive-data-codebook--dataset-documentation)
   - [4.1 Primary Subject-Level Experimental Dataset (`maindata_2026-08-10.csv`)](#41-primary-subject-level-experimental-dataset-maindata_2026-08-10csv)
   - [4.2 Evaluator Scoring Dataset (`TTs_raters_2026-08-10.csv`)](#42-evaluator-scoring-dataset-tts_raters_2026-08-10csv)
   - [4.3 Metadata Codebook Schema (`codebook_2026-08-10.csv`)](#43-metadata-codebook-schema-codebook_2026-08-10csv)
   - [4.4 Longitudinal Tasking Hub (`Participant-level tasking hub.csv`)](#44-longitudinal-tasking-hub-participant-level-tasking-hubcsv)
   - [4.5 AI Copilot Telemetry & Adoption Data (`All usage measures - Merged.csv`)](#45-ai-copilot-telemetry--adoption-data-all-usage-measures---mergedcsv)
   - [4.6 Telemetry Domain Mapping Key (`All usage measures - Domain Key.csv`)](#46-telemetry-domain-mapping-key-all-usage-measures---domain-keycsv)
   - [4.7 Traceability Keys & Exclusion Files](#47-traceability-keys--exclusion-files)
5. [End-to-End Empirical Pipeline: Step-by-Step Execution](#5-end-to-end-empirical-pipeline-step-by-step-execution)
   - [Step 0: Data Anonymization & Key Verification](#step-0-data-anonymization--key-verification)
   - [Step 1: Econometric Analysis & Macro Generation](#step-1-econometric-analysis--macro-generation)
   - [Step 2: Table Generation](#step-2-generate-all-manuscript-tables)
   - [Step 3: Figure Generation](#step-3-generate-all-empirical-figures)
   - [Step 4: Manuscript PDF Compilation](#step-4-compile-latex-manuscript-pdf)
6. [Mapping of Manuscript Outputs to Code & Data](#6-mapping-of-manuscript-outputs-to-code--data)
   - [6.1 Main Text Tables (Tables 1–9)](#61-main-text-tables-tables-19)
   - [6.2 Main Text Figures (Figures 1–4)](#62-main-text-figures-figures-14)
   - [6.3 Appendix Tables (Tables A1–A8)](#63-appendix-tables-tables-a1a8)
   - [6.4 Appendix Figures (Figures A1–A5)](#64-appendix-figures-figures-a1a5)
7. [Econometric Methodology & Estimation Details](#7-econometric-methodology--estimation-details)
8. [Configuration & Customization Options (`config.py`)](#8-configuration--customization-options-configpy)
9. [Data Privacy, Anonymization & Ethical Compliance](#9-data-privacy-anonymization--ethical-compliance)
10. [Replication Verification & Quality Checklist](#10-replication-verification--quality-checklist)

---

## 1. Quick Start Replication

To replicate the complete empirical analysis—from raw data ingestion, econometric estimations, and JSON serialization to LaTeX table rendering, figure plotting, dynamic macro binding, and compiled manuscript PDF—execute the master replication script from the root directory:

```bash
# 1. (Optional) Create and activate a clean Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install required Python packages
pip install -r requirements.txt

# 3. Execute the automated master replication pipeline
./replicate.sh
```

### What `./replicate.sh` Does Automatically:
1. **[0/4] Verifies Dataset Anonymization**: Checks deterministic UUIDv5 hashes and key mappings ([anonymize.py](file:///usr/local/google/home/joshuabmartin/InFlow%202/anonymize.py)).
2. **[1/4] Runs Econometric Engine & Emits Macros**: Ingests canonical CSVs, normalizes outcomes against control baseline, runs all OLS/WLS regressions with firm fixed effects and clustered standard errors, calculates permutation tests (2,000 permutations), serializes outputs to `Jsons/`, and generates `New/macros.tex` ([macros.py](file:///usr/local/google/home/joshuabmartin/InFlow%202/macros.py) and [analysis.py](file:///usr/local/google/home/joshuabmartin/InFlow%202/analysis.py)).
3. **[2/4] Generates All Tables**: Runs the table generator modules (`tab_*.py`) and saves formatted LaTeX tables to `New/*.tex` ([generate_all_tables.py](file:///usr/local/google/home/joshuabmartin/InFlow%202/generate_all_tables.py)).
4. **[3/4] Generates All Figures**: Runs the figure generator modules (`fig_*.py`) and renders publication-ready 300 DPI PNG figures to `New/*.png` ([generate_all_figures.py](file:///usr/local/google/home/joshuabmartin/InFlow%202/generate_all_figures.py)).
5. **[4/4] Compiles LaTeX Manuscript**: Compiles [draft_2026Aug20.tex](file:///usr/local/google/home/joshuabmartin/InFlow%202/draft_2026Aug20.tex) with `pdflatex` and `bibtex` to produce [draft_2026Aug20.pdf](file:///usr/local/google/home/joshuabmartin/InFlow%202/draft_2026Aug20.pdf).

---

## 2. System & Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+), macOS (12.0+), or Windows 10/11 via WSL2.
- **Python Runtime**: Python 3.10, 3.11, 3.12, or 3.13.
- **Required Python Libraries** (specified in [`requirements.txt`](file:///usr/local/google/home/joshuabmartin/InFlow%202/requirements.txt)):
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `statsmodels>=0.14.0`
  - `scipy>=1.10.0`
  - `matplotlib>=3.7.0`
  - `seaborn>=0.12.0`
  - `python-dotenv>=1.0.0`
  - `requests>=2.31.0`
- **LaTeX Distribution** (for PDF compilation):
  - TeX Live (2022 or newer) or MacTeX containing `pdflatex` and `bibtex`.
  - Required LaTeX packages (standard in TeX Live Full): `amsmath`, `amssymb`, `booktabs`, `tabularx`, `longtable`, `threeparttable`, `caption`, `graphicx`, `natbib`, `siunitx`, `xltabular`, `chngcntr`, `pdfpages`.

---

## 3. Repository Directory & File Architecture

```text
.
├── data/                               # Canonical experimental datasets & metadata
│   ├── maindata_2026-08-10.csv         # Subject-level master dataset (N=999, 136 columns)
│   ├── TTs_raters_2026-08-10.csv       # Multi-rater scoring records (N=1,124, 76 columns)
│   ├── codebook_2026-08-10.csv         # Variable dictionary and metadata schema (N=1,017)
│   ├── CODEBOOK.md                     # Data documentation file
│   ├── Participant-level tasking hub.csv # Longitudinal milestone & completion tracking (N=999)
│   ├── All usage measures - Merged.csv # Weekly AI copilot telemetry data (N=431)
│   └── All usage measures - Domain Key.csv # Domain-to-firm mapping key
│
├── Jsons/                              # Serialized econometric & summary statistics payloads
│   ├── models_data.json                # Main, pooled, and disaggregated regression estimates
│   ├── additional_analysis.json        # Variance regressions (varregs) & inter-item correlation matrices
│   ├── attrition.json                  # Sample attrition linear probability models (Table A2)
│   ├── balance.json                    # Baseline covariate balance tests (Table A1)
│   ├── cell_summary.json               # Standard normal cell means and standard deviations (Table 2, A5)
│   ├── takeup_completion.json          # Overall task completion & experimental takeup rates
│   ├── fig_cdf.json                    # Empirical CDF coordinates & Kolmogorov-Smirnov test statistics
│   ├── fig_fisher.json                 # Permutation test null distributions & exact p-values (Figure 4)
│   ├── fig_forest.json                 # Forest plot coefficients & 95% CIs across quality dimensions (Figure 1)
│   ├── fig_histograms.json             # Density histograms for drafting & redlining (Figures 2 & 3)
│   ├── fig_leave_one_firm_out.json     # Leave-one-firm-out robustness estimates (Figure A4)
│   ├── fig_raters.json                 # Inter-rater agreement scatter coordinates & correlations (Figures A1, A2)
│   ├── fig_human_llm_subscales.json    # Subscale concordance across evaluator modalities
│   ├── fig_sensitivity.json            # Experience cutoff sensitivity analysis (Figure A3)
│   ├── fig_usage_alt.json              # 14-day rolling average active user time series (Figure A5)
│   └── firm_summary.json               # Firm-level sample sizes and task completion rates (Table 1)
│
├── New/                                # Generated LaTeX tables, PNG figures, macros & instruments
│   ├── macros.tex                      # LaTeX macro definitions (\stat{...}) binding numbers in text
│   ├── *.tex                           # 17 publication-ready LaTeX tables (Tables 1-9, Tables A1-A8)
│   ├── *.png                           # 9 publication-ready 300 DPI figures (Figures 1-4, Figures A1-A5)
│   ├── TT1 - Master.pdf                # 10-Day Task Instrument (Drafting Only)
│   └── TT2 - Master.pdf                # 90-Day Task Instrument (Drafting & Redlining)
│
├── anonymize.py                        # Standalone data anonymization engine & key verifier
├── config.py                           # Central configuration & experimental parameters
├── auth.py                             # Local environment & authentication loader
├── utils.py                            # Formatting, label lookups, file I/O, and GitHub sync utilities
├── utils_mock_model.py                 # Deserializer for MockModel and MockTTest regression objects
├── models.py                           # Econometric estimation engine (OLS, WLS, clustering, LaTeX table export)
├── analysis.py                         # Master statistical engine (data prep + 14 empirical analysis modules)
├── macros.py                           # Macro compilation engine (validates draft_2026Aug20.tex and emits New/macros.tex)
├── generate_all_tables.py              # Batch runner for table generator modules
├── generate_all_figures.py             # Batch runner for figure generator modules
│
├── tab_*.py                            # Standalone table generator scripts
├── fig_*.py                            # Standalone figure generator scripts
│
├── exclusion_firms.txt                 # Firm-level exclusion configuration file
├── exclusion_individuals.txt           # Individual practitioner exclusion configuration file
├── anonymization_keys.csv              # Two-way UUIDv5 de-identification key mapping table
├── anonymization_keys.txt              # Formatted human-readable de-identification report
├── draft_2026Aug20.tex                 # Primary LaTeX manuscript source document
├── references.bib                      # BibTeX bibliography database
├── macros.tex                          # Root duplicate copy of New/macros.tex
├── draft_2026Aug20.pdf                 # Publication-ready compiled PDF manuscript
├── requirements.txt                    # Python environment package specifications
├── replicate.sh                        # Master automated replication shell script
└── README.md                           # Combined master replication package & codebook documentation
```

---

## 4. Comprehensive Data Codebook & Dataset Documentation

The replication package includes 7 canonical data files in [`data/`](file:///usr/local/google/home/joshuabmartin/InFlow%202/data), plus anonymization key files and exclusion configurations.

```
Canonical Datasets Overview:
┌──────────────────────────────────────┬───────────────────────┬───────────────────────────────┬────────────────────────┐
│ File Name                            │ Format / Storage      │ Unit of Observation           │ Dimensions             │
├──────────────────────────────────────┼───────────────────────┼───────────────────────────────┼────────────────────────┤
│ maindata_2026-08-10.csv              │ CSV (UTF-8)           │ Practitioner (Individual)     │ 999 rows × 136 cols    │
│ TTs_raters_2026-08-10.csv            │ CSV (UTF-8)           │ Practitioner × Task × Rater   │ 1,124 rows × 76 cols   │
│ codebook_2026-08-10.csv              │ CSV (UTF-8)           │ Variable Metadata             │ 1,017 rows × 6 cols    │
│ Participant-level tasking hub.csv    │ CSV (UTF-8)           │ Practitioner Milestone Record │ 999 rows × 20 cols     │
│ All usage measures - Merged.csv      │ CSV (UTF-8)           │ Firm-Domain × Week            │ 431 rows × 9 cols      │
│ All usage measures - Domain Key.csv  │ CSV (UTF-8)           │ Firm Name × Domain ID         │ 11 rows × 2 cols       │
│ anonymization_keys.csv               │ CSV (UTF-8)           │ Identity Mapping Record       │ 378 rows × 6 cols      │
└──────────────────────────────────────┴───────────────────────┴───────────────────────────────┴────────────────────────┘
```

---

### 4.1 Primary Subject-Level Experimental Dataset (`maindata_2026-08-10.csv`)

[`maindata_2026-08-10.csv`](file:///usr/local/google/home/joshuabmartin/InFlow%202/data/maindata_2026-08-10.csv) is the core subject-level dataset containing randomized experimental assignments, practitioner baseline characteristics, standardized task quality scores, time-on-task metrics, firm administrative patenting behavior, and post-task subjective survey responses for $N = 999$ registered patent practitioners across 19 participating law firms.

#### Identification & Randomization Variables
- **`Email` / `email`**: De-identified unique practitioner identifier (deterministic 36-character UUIDv5 hash).
- **`Firm number` / `firm_num`**: Numeric identifier for the practitioner's firm ($1, 2, \dots, 19$), used for firm fixed effects and clustering.
- **`Firm name` / `firm_name`**: Anonymized firm name identifier (`Firm 1` through `Firm 19`).
- **`Name` / `name`**: Anonymized practitioner name (deterministic UUIDv5 hash).
- **`Group` / `group`**: Experimental arm assignment string:
  - `Group 1`: Control group (completed tasks using standard tools without AI assistance).
  - `Group 2`: Treatment group (granted access to the InFlow generative AI copilot).
- **`Group_binary` / `group_binary`**: Numeric treatment indicator ($1 = \text{Treatment / Group 2}$, $0 = \text{Control / Group 1}$).
- **`Included` / `included`**: Analysis sample indicator ($1.0 = \text{verified patent practitioner with baseline consent}$; $0.0 = \text{ineligible / non-consenting}$).

#### Experience & Seniority Stratification
- **`Experience` / `exp`**: Continuous years of registered patent drafting and prosecution experience.
- **`Junior/expert` / `expert` / `seniority`**: Categorical classification based on the 7-year experience threshold:
  - `Junior`: Experience $< 7$ years.
  - `Senior` / `Expert`: Experience $\ge 7$ years.
- **`junior`**: Binary indicator equal to `1` if `Experience < 7`, and `0` otherwise.
- **`senior`**: Binary indicator equal to `1` if `Experience >= 7`, and `0` otherwise.
- **`group_binary_x_junior`**: Interaction term (`group_binary * junior`), representing the differential treatment effect for junior practitioners.
- **`group_binary_x_senior`**: Interaction term (`group_binary * senior`), representing the treatment effect for senior practitioners.

#### 10-Day Evaluation Task (Task 1: Drafting Only)
- **`TT1_Summary (drafting)` / `tt1_sum_drades` / `tt1_rat_drades_sum`**: Primary composite quality score for the 10-day drafting task (standardized to mean 0, SD 1 on the control baseline).
- **Dependent Claims Sub-Scales** (rated 1–5):
  - `tt1_dra_enf` / `tt1_rat_drades_enf`: Enforceability.
  - `tt1_dra_tec` / `tt1_rat_drades_tec`: Technical accuracy.
  - `tt1_dra_str` / `tt1_rat_drades_str`: Strategic ambiguity.
  - `tt1_dra_com` / `tt1_rat_drades_com`: Completeness and alignment with invention disclosures.
  - `tt1_dra_cla` / `tt1_rat_drades_cla`: Clarity.
- **Detailed Description Sub-Scales** (rated 1–5):
  - `tt1_des_enf`: Detailed description enforceability.
  - `tt1_des_tec`: Detailed description technical accuracy.
  - `tt1_des_str`: Detailed description strategic ambiguity.
  - `tt1_des_com`: Detailed description completeness and alignment.
  - `tt1_des_cla`: Detailed description clarity.
- **Time-on-Task & Survey Variables**:
  - `tt1_sum_dra_tot`: Realized time spent completing the 10-day drafting task (minutes).
  - `tt1_sur_dra_tot`: Survey self-reported time on task (minutes).
  - `tt1_sur_dra_spe`: Self-reported drafting speed perception (1–5 scale, $1 = \text{Much slower}$, $5 = \text{Much faster}$).
  - `tt1_sur_dra_qua`: Self-reported drafting quality perception (1–5 scale, $1 = \text{Very poor}$, $5 = \text{Exceptional}$).
  - `tt1_sur_dra_sat`: Self-reported satisfaction with drafting workflow (1–5 scale, $1 = \text{Very dissatisfied}$, $5 = \text{Very satisfied}$).

#### 90-Day Evaluation Task (Task 2: Drafting & Redlining)
- **Drafting Exercise Outcomes**:
  - `TT2_Summary (drafting)` / `tt2_sum_drades` / `tt2_rat_drades_sum`: Composite standardized drafting quality score.
  - `tt2_dra_enf`, `tt2_dra_tec`, `tt2_dra_str`, `tt2_dra_com`, `tt2_dra_cla`: Dependent claims sub-scale ratings.
  - `tt2_des_enf`, `tt2_des_tec`, `tt2_des_str`, `tt2_des_com`, `tt2_des_cla`: Detailed description sub-scale ratings.
  - `tt2_sum_dra_tot`: Realized time spent completing 90-day drafting (minutes).
  - `tt2_sur_dra_spe`, `tt2_sur_dra_qua`, `tt2_sur_dra_sat`: Subjective speed, quality, and satisfaction ratings (1–5 scale).
- **Redlining / Critique Exercise Outcomes**:
  - `TT2_Summary (critique)` / `tt2_sum_cri` / `tt2_rat_cri_sum`: Composite standardized redlining quality score.
  - `tt2_cri_enf` / `tt2_rat_cri_enf`: Redlining enforceability sub-scale.
  - `tt2_cri_tec` / `tt2_rat_cri_tec`: Redlining technical accuracy sub-scale.
  - `tt2_cri_str` / `tt2_rat_cri_str`: Redlining strategic ambiguity sub-scale.
  - `tt2_cri_com` / `tt2_rat_cri_com`: Redlining completeness and alignment sub-scale.
  - `tt2_cri_cla` / `tt2_rat_cri_cla`: Redlining clarity sub-scale.
  - `tt2_sum_cri_tot`: Realized time spent completing 90-day redlining (minutes).
  - `tt2_sur_cri_spe`, `tt2_sur_cri_qua`, `tt2_sur_cri_sat`: Subjective speed, quality, and satisfaction ratings (1–5 scale).
- **Prompt Leakage & Compliance Checks**:
  - `tt2_cri_noncomp`: Evaluator check for non-compliance / unprompted structural deviation.
  - `tt2_cri_leak`: Indicator for AI prompt artifact leakage into redline submissions.

#### On-the-Job Firm Administrative Patenting Metrics
Administrative metrics tracking real-world patent prosecution and drafting behavior across pre- and post-treatment windows:
- **`pat_num`**: Total count of patent applications drafted and filed with the USPTO.
- **`pat_avg_res`**: Average count of office action responses filed per patent application.
- **`pat_avg_len`**: Average word count of patent detailed description sections.
- **`pat_avg_rev`**: Average revision rounds between junior practitioner and supervising partner.
- **`pat_avg_cor`**: Average count of substantive correction notes on draft patent claims.

---

### 4.2 Evaluator Scoring Dataset (`TTs_raters_2026-08-10.csv`)

[`TTs_raters_2026-08-10.csv`](file:///usr/local/google/home/joshuabmartin/InFlow%202/data/TTs_raters_2026-08-10.csv) contains granular scoring records where each row represents an individual evaluation of a practitioner's submitted task by either an expert human patent attorney or an LLM automated evaluator.

- **`Email`**: De-identified practitioner UUID (links to `maindata_2026-08-10.csv`).
- **`Test Task`**: Task identifier:
  - `TT1`: 10-Day Task (Drafting).
  - `TT2`: 90-Day Task (Drafting and Redlining).
- **`Group`**: Treatment assignment of the evaluated practitioner (`Group 1` or `Group 2`).
- **`Included`**: Inclusion indicator ($1.0 = \text{included}$).
- **`Firm name`**: Anonymized firm identifier (`Firm 1` through `Firm 19`).
- **`Rater ID`**: Unique identifier for the evaluator:
  - Human expert patent attorneys: Anonymized UUIDs (`Rater_1`, `Rater_2`, etc.).
  - LLM automated evaluators: `'llm'`, `'LLM_Rater_1'`, etc.
- **`Rater Type`**: Categorical evaluator type (`Human` vs. `LLM`).
- **Sub-Scale Quality Ratings** (1–5 Likert scores):
  - `In the DRAFTING exercise, please rate the quality of the DEPENDENT CLAIMS section. [Enforceability]` (`enf`)
  - `In the DRAFTING exercise, please rate the quality of the DEPENDENT CLAIMS section. [Technical Accuracy]` (`tec`)
  - `In the DRAFTING exercise, please rate the quality of the DEPENDENT CLAIMS section. [Strategic Ambiguity]` (`str`)
  - `In the DRAFTING exercise, please rate the quality of the DEPENDENT CLAIMS section. [Completeness and Alignment with Objectives]` (`com`)
  - `In the DRAFTING exercise, please rate the quality of the DEPENDENT CLAIMS section. [Clarity ]` (`cla`)
  - Corresponding sub-scale items for Detailed Description and Redlining/Critique.
- **Inverse Variance Weights (`w_i`)**:
  - In pooled regressions using `TTs_raters_2026-08-10.csv`, each evaluation is weighted by $w_i = 1 / n_i$, where $n_i$ is the number of ratings practitioner $i$ received, ensuring every practitioner receives equal weight in the estimation.

---

### 4.3 Metadata Codebook Schema (`codebook_2026-08-10.csv`)

[`codebook_2026-08-10.csv`](file:///usr/local/google/home/joshuabmartin/InFlow%202/data/codebook_2026-08-10.csv) provides a machine-readable schema connecting raw survey headers to standardized econometric variable names:

| Column Header | Data Type | Description | Example |
|:---|:---|:---|:---|
| **`original`** | String | Exact column header in raw survey and export files | `"TT1_Summary (drafting)"` |
| **`varname`** | String | Standardized snake_case variable name in Python scripts | `"tt1_sum_drades"` |
| **`label_shorthand`** | String | Human-readable label for LaTeX table stubs and plot axes | `"10-day drafting"` |
| **`column`** | String | Original spreadsheet column letter index | `"I"` |
| **`type`** | String | Variable classification (`Likert`, `Int`, `Float`, `Dum`/`Dummy`) | `"Likert"` |
| **`highermeans`** | String | Directional interpretation of higher values | `"higher score"`, `"faster"`, `"more satisfied"` |

---

### 4.4 Operational dataset (`Participant-level tasking hub.csv`)

[`Participant-level tasking hub.csv`](file:///usr/local/google/home/joshuabmartin/InFlow%202/data/Participant-level%20tasking%20hub.csv) ($N = 999$ rows) records operational milestones, timestamp logs, and completion status for every practitioner in the study:

- **Identifiers**: `Firm number`, `Firm name`, `Name` (UUID), `Experience`, `Junior/expert`, `Email` (UUID), `Group`, `Included`.
- **Milestone Timestamps & Completion Indicators**:
  - `0. Date of onboarding email`: Date initial welcome and onboarding invitation was dispatched.
  - `1. Onboarding form completed`: Date practitioner completed baseline background survey.
  - `2. Consent form`: Date practitioner executed informed consent.
  - `3. Date TT1 sent`: Date 10-day drafting task was assigned.
  - `4. TT1`: Date 10-day drafting task was completed and submitted.
  - `4a. TT1 review complete`: Date peer review / evaluation scoring was finalized.
  - `5. TT1 survey`: Date 10-day post-task subjective survey was completed.
  - `6. Date TT2 sent`: Date 90-day task was assigned.
  - `7. TT2`: Date 90-day task was submitted.
  - `7a. TT2 review complete`: Date 90-day peer review scoring was finalized.
  - `8. TT2 survey`: Date 90-day post-task subjective survey was completed.
  - `9. Monthly survey`: Date final longitudinal engagement survey was submitted.
- **Analytical Role**: Used to construct attrition models ([Table A2](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_attrition.py)), baseline covariate balance ([Table A1](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_balance.py)), and firm-level completion rates ([Table 1](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_firm_summary.py)).

---

### 4.5 InFlow Telemetry & Adoption Data (`All usage measures - Merged.csv`)

[`All usage measures - Merged.csv`](file:///usr/local/google/home/joshuabmartin/InFlow%202/data/All%20usage%20measures%20-%20Merged.csv) ($N = 431$ weekly records) tracks real-time practitioner interactions with the InFlow AI copilot:

- **`Week Beginning`**: Monday calendar date of the tracking week (YYYY-MM-DD format).
- **`userDomain`**: Anonymized domain identifier (e.g. `firm_1_domain`, `firm_9_domain`).
- **`Event Count`**: Total count of user-initiated AI interaction events.
- **`Event Actions`**: Count of substantive AI actions (drafting prompts, inline completions, suggestions).
- **`Event Values`**: Continuous duration / intensity measure of AI interaction sessions.
- **`Critique Feature Usage`**: Specific count of AI redlining and claim critique tool executions.
- **`Days`**: Number of active working days with logged AI usage during the week.
- **`Active Users`**: Unique practitioner count using the AI tool during the week.
- **`AI Characters`**: Total character volume generated by the AI copilot.
- **Analytical Role**: Used by [`fig_usage_alt.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_usage_alt.py) to construct longitudinal adoption curves, cumulative engagement metrics, and 14-day rolling average active user series ([Figure A5](file:///usr/local/google/home/joshuabmartin/InFlow%202/New/usage_alt.png)).

---

### 4.6 Telemetry Domain Mapping Key (`All usage measures - Domain Key.csv`)

[`All usage measures - Domain Key.csv`](file:///usr/local/google/home/joshuabmartin/InFlow%202/data/All%20usage%20measures%20-%20Domain%20Key.csv) links anonymized firm names to internal telemetry domains:

```csv
firm_name,userDomain
Firm 11,firm_11_domain
Firm 9,firm_9_domain
Firm 5,firm_5_domain
Firm 1,firm_1_domain
Firm 2,firm_2_domain
Firm 12,firm_12_domain
Firm 10,firm_10_domain
Firm 4,firm_4_domain
Firm 3,firm_3_domain
Firm 6,firm_6_domain
Firm 8,firm_8_domain
```

---

### 4.7 Traceability Keys & Exclusion Files

- **`anonymization_keys.csv` & `anonymization_keys.txt`**:
  - Two-way mapping table maintaining exact correspondence between raw practitioner identities, rater IDs, and firm names and their anonymized UUIDv5 hashes.
  - Verified on every run by [anonymize.py](file:///usr/local/google/home/joshuabmartin/InFlow%202/anonymize.py) to ensure 100% data integrity without PII leakage.
- **`exclusion_individuals.txt`**:
  - List of anonymized practitioner UUIDs excluded from empirical estimations (e.g. participants withdrawing consent). Active when `EXCLUDE_INDIVIDUALS = True` in `config.py`.
- **`exclusion_firms.txt`**:
  - List of firm identifiers excluded during sensitivity checks. Active when `EXCLUDE_FIRMS = True` in `config.py`.

---

## 5. End-to-End Empirical Pipeline: Step-by-Step Execution

For complete granular control, replicators can execute each stage of the empirical pipeline individually:

```
Replication Execution Pipeline:
┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  Step 0: Anonymization │ ──> │   Step 1: Econometric  │ ──> │    Step 2: Table       │
│  & Key Verification    │     │   Analysis & Macros    │     │    Generation          │
│  (anonymize.py)        │     │   (macros.py)          │     │    (generate_all_*)    │
└────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                           │
                                                                           ▼
┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  Step 4: Manuscript    │ <── │   Generated Artifacts  │ <── │    Step 3: Figure      │
│  PDF Compilation       │     │   (New/*.tex, New/*.png│     │    Generation          │
│  (pdflatex + bibtex)   │     │    New/macros.tex)     │     │    (generate_all_*)    │
└────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

### Step 0: Data Anonymization & Key Verification
```bash
python3 anonymize.py
```
- **Inputs**: `anonymization_keys.csv`, raw practitioner and rater data files.
- **Actions**:
  1. Verifies deterministic UUIDv5 hashes (DNS namespace) for all practitioners and raters.
  2. Verifies sequential firm pseudonymization (`Firm 1` .. `Firm 19`).
  3. Ensures all materialized CSVs in `data/` match key mappings with zero orphans.
  4. Materializes and verifies `anonymization_keys.txt`.

### Step 1: Econometric Analysis & Macro Generation
```bash
python3 macros.py
```
- **Inputs**: `data/maindata_2026-08-10.csv`, `data/codebook_2026-08-10.csv`, `data/TTs_raters_2026-08-10.csv`.
- **Actions**:
  1. **Data Prep**: Filters analysis samples, applies exclusion lists, normalizes performance scores against control group baseline ($\mu_{\text{Ctrl}} = 0, \sigma_{\text{Ctrl}} = 1$), constructs junior/senior indicators and interaction terms, and computes inverse-variance WLS weights ($w_i = 1 / n_i$).
  2. **Econometric Estimations**: Fits OLS and WLS regression specifications with firm fixed effects and standard errors clustered at the practitioner level (`models.py`, `analysis.py`).
  3. **Nonparametric & Variance Tests**: Computes Levene variance tests, Fisher exact permutation tests (2,000 Monte Carlo permutations), and Kolmogorov-Smirnov CDF statistics.
  4. **Serialization**: Exports 14 intermediate JSON statistical payloads into `Jsons/`.
  5. **Macro Validation**: Emits `New/macros.tex` containing LaTeX macro commands (`\newcommand{\MacroName}{...}`) that bind numbers directly into `draft_2026Aug20.tex`, verifying that every macro referenced in the manuscript is populated.

### Step 2: Generate All Manuscript Tables
```bash
python3 generate_all_tables.py
```
- **Actions**: Discovers and runs the table generator scripts in sequence, reading serialized statistics from `Jsons/` and emitting publication-formatted LaTeX tables to `New/*.tex`.

### Step 3: Generate All Empirical Figures
```bash
python3 generate_all_figures.py
```
- **Actions**: Discovers and runs the figure plotting scripts in sequence, reading serialized statistics from `Jsons/` and rendering 300 DPI publication-quality PNG charts to `New/*.png`.

### Step 4: Compile LaTeX Manuscript PDF
```bash
pdflatex -interaction=nonstopmode draft_2026Aug20.tex
bibtex draft_2026Aug20
pdflatex -interaction=nonstopmode draft_2026Aug20.tex
pdflatex -interaction=nonstopmode draft_2026Aug20.tex
```
- **Actions**: Compiles `draft_2026Aug20.tex` into `draft_2026Aug20.pdf`, incorporating all dynamic tables, figures, statistical macros, and bibliographic citations.

---

## 6. Mapping of Manuscript Outputs to Code & Data

### 6.1 Main Text Tables (Tables 1–9)

| Manuscript Table | Output File | Generator Script | JSON Payload | Core Data Inputs | Econometric Description |
|:---|:---|:---|:---|:---|:---|
| **Table 1** | `New/firm_summary.tex` | [`tab_firm_summary.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_firm_summary.py) | `Jsons/firm_summary.json` | `maindata`, `Participant-level tasking hub` | Firm attributes, practitioner sample counts, randomized treatment allocations, and task completion rates across 19 law firms. |
| **Table 2** | `New/cell_summary_human_ind.tex` | [`tab_cell_summary_human_ind.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_cell_summary_human_ind.py) | `Jsons/cell_summary.json` | `maindata`, `codebook` | Standard normal cell means and standard deviations across experimental arms (Control vs. Treatment) and experience cohorts (Junior vs. Senior) for expert human evaluations. |
| **Table 3** | `New/10dayD_combined_main.tex` | [`tab_10dayD_combined_main.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_10dayD_combined_main.py) | `Jsons/models_data.json` | `maindata`, `TTs_raters` | 10-Day drafting performance regressions (Dependent Claims, Detailed Description, Combined) with firm fixed effects and clustered SEs (Human raters). |
| **Table 4** | `New/90dayD_combined_main.tex` | [`tab_90dayD_combined_main.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_90dayD_combined_main.py) | `Jsons/models_data.json` | `maindata`, `TTs_raters` | 90-Day drafting performance regressions with firm fixed effects and clustered SEs (Human raters). |
| **Table 5** | `New/varregs_drafting.tex` | [`tab_varregs.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_varregs.py) | `Jsons/additional_analysis.json` | `maindata`, `codebook` | Drafting performance variance regressions (absolute residuals and Levene tests) testing AI effects on performance dispersion. |
| **Table 6** | `New/90dayR_combined_main.tex` | [`tab_90dayR_combined_main.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_90dayR_combined_main.py) | `Jsons/models_data.json` | `maindata`, `TTs_raters` | 90-Day redlining/critique performance regressions with firm fixed effects and clustered SEs (Human raters). |
| **Table 7** | `New/varregs_redlining.tex` | [`tab_varregs.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_varregs.py) | `Jsons/additional_analysis.json` | `maindata`, `codebook` | Redlining/critique performance variance regressions testing AI effects on evaluation dispersion. |
| **Table 8** | `New/TimeOnTask.tex` | [`tab_TimeOnTask.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_TimeOnTask.py) | `Jsons/models_data.json` | `maindata`, `codebook` | Realized time on task regressions across 10-day drafting, 90-day drafting, and 90-day redlining exercises. |
| **Table 9** | `New/Patent.tex` | [`tab_Patent.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_Patent.py) | `Jsons/models_data.json` | `maindata`, `codebook` | On-the-job firm administrative patenting regressions (filings, responses, description length, revision rounds, claim correction notes). |

---

### 6.2 Main Text Figures (Figures 1–4)

| Manuscript Figure | Output File | Generator Script | JSON Payload | Core Data Inputs | Description |
|:---|:---|:---|:---|:---|:---|
| **Figure 1** | `New/forest_plot_subelements.png` | [`fig_forest_plot_subelements.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_forest_plot_subelements.py) | `Jsons/fig_forest.json` | `TTs_raters`, `codebook` | Forest plot displaying treatment effect estimates and 95% confidence intervals across the 5 canonical quality sub-elements (Enforceability, Technical Accuracy, Strategic Ambiguity, Completeness, Clarity). |
| **Figure 2** | `New/histograms_drafting.png` | [`fig_histograms.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_histograms.py) | `Jsons/fig_histograms.json` | `maindata`, `codebook` | Overlapping empirical score density distributions for drafting tasks broken down by treatment assignment and experience cohort (Junior vs. Senior). |
| **Figure 3** | `New/histograms_redlining.png` | [`fig_histograms.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_histograms.py) | `Jsons/fig_histograms.json` | `maindata`, `codebook` | Overlapping empirical score density distributions for the 90-day redlining task by treatment assignment and experience cohort. |
| **Figure 4** | `New/fisher.png` | [`fig_fisher.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_fisher.py) | `Jsons/fig_fisher.json` | `TTs_raters`, `maindata` | Nonparametric permutation test distributions (2,000 Monte Carlo permutations under sharp null) and exact two-sided p-values for full sample, juniors, and seniors. |

---

### 6.3 Appendix Tables (Tables A1–A8)

| Appendix Table | Output File | Generator Script | JSON Payload | Core Data Inputs | Description |
|:---|:---|:---|:---|:---|:---|
| **Table A1** | `New/balance.tex` | [`tab_balance.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_balance.py) | `Jsons/balance.json` | `maindata`, `codebook` | Baseline covariate balance tests between Treatment and Control groups across experience, firm size, and baseline patenting. |
| **Table A2** | `New/attrition.tex` | [`tab_attrition.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_attrition.py) | `Jsons/attrition.json` | `Participant-level tasking hub` | Sample attrition linear probability models testing differential dropout across experimental phases. |
| **Table A3** | `New/correlations_human.tex` | [`tab_correlations_human.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_correlations_human.py) | `Jsons/additional_analysis.json` | `TTs_raters` | Inter-item correlation matrix and Cronbach's alpha across human expert evaluator subscale scores. |
| **Table A4** | `New/correlations_llm.tex` | [`tab_correlations_llm.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_correlations_llm.py) | `Jsons/additional_analysis.json` | `TTs_raters` | Inter-item correlation matrix and Cronbach's alpha across LLM evaluator subscale scores. |
| **Table A5** | `New/cell_summary_llm_ind.tex` | [`tab_cell_summary_llm_ind.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_cell_summary_llm_ind.py) | `Jsons/cell_summary.json` | `maindata`, `codebook` | Standard normal cell means and standard deviations under automated LLM evaluation. |
| **Table A6** | `New/90dayR_cheating_ind_human.tex` | [`tab_90dayR_cheating_ind_human.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_90dayR_cheating_ind_human.py) | `Jsons/models_data.json` | `maindata`, `TTs_raters` | Prompt leakage and non-compliance regression checks (Human raters). |
| **Table A7** | `New/90dayR_cheating_ind_llm.tex` | [`tab_90dayR_cheating_ind_llm.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_90dayR_cheating_ind_llm.py) | `Jsons/models_data.json` | `maindata`, `TTs_raters` | Prompt leakage and non-compliance regression checks (LLM raters). |
| **Table A8** | `New/DraftingSurvey.tex` | [`tab_DraftingSurvey.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/tab_DraftingSurvey.py) | `Jsons/models_data.json` | `maindata`, `codebook` | Post-task subjective perceptions regressions (perceived speed, perceived draft quality, task satisfaction). |

---

### 6.4 Appendix Figures (Figures A1–A5)

| Appendix Figure | Output File | Generator Script | JSON Payload | Core Data Inputs | Description |
|:---|:---|:---|:---|:---|:---|
| **Figure A1** | `New/rater1.png` | [`fig_rater1.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_rater1.py) | `Jsons/fig_raters.json` | `TTs_raters` | Inter-rater scoring agreement and distribution comparisons among expert human patent attorneys. |
| **Figure A2** | `New/rater_human_vs_llm.png` | [`fig_rater_human_vs_llm.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_rater_human_vs_llm.py) | `Jsons/fig_raters.json` | `TTs_raters` | Concordance scatter plot and correlation between human expert scores and automated LLM scores. |
| **Figure A3** | `New/sensitivity.png` | [`fig_sensitivity.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_sensitivity.py) | `Jsons/fig_sensitivity.json` | `maindata`, `codebook` | Sensitivity of treatment effect estimates across varying seniority cutoffs from 3 to 15 years of experience. |
| **Figure A4** | `New/forest_plot_leave_one_out.png` | [`fig_leave_one_firm_out.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_leave_one_firm_out.py) | `Jsons/fig_leave_one_firm_out.json` | `maindata`, `codebook` | Leave-one-firm-out robustness check estimating redlining treatment effects omitting each firm iteratively. |
| **Figure A5** | `New/usage_alt.png` | [`fig_usage_alt.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/fig_usage_alt.py) | `Jsons/fig_usage_alt.json` | `All usage measures - Merged` | Longitudinal AI copilot adoption curves and 14-day rolling average active user time series. |

---

## 7. Econometric Methodology & Estimation Details

### 1. Primary Regression Specification
The primary econometric model estimates treatment effects with firm fixed effects and standard errors clustered at the practitioner level:

$$Y_{ij} = \beta_0 + \beta_1 \text{Treated}_i + \beta_2 \text{Junior}_i + \beta_3 (\text{Treated}_i \times \text{Junior}_i) + \alpha_j + \varepsilon_{ij}$$

where:
- $Y_{ij}$ is the standardized outcome metric for practitioner $i$ in firm $j$.
- $\text{Treated}_i \in \{0, 1\}$ indicates random assignment to the InFlow generative AI copilot treatment arm (`group_binary`).
- $\text{Junior}_i \in \{0, 1\}$ indicates experience $< 7$ years (`junior`).
- $\beta_1$ estimates the treatment effect among senior practitioners ($\ge 7$ years experience).
- $\beta_1 + \beta_3$ estimates the treatment effect among junior practitioners.
- $\alpha_j$ denotes firm fixed effects (`firm_num`).
- Standard errors $\varepsilon_{ij}$ are clustered at the individual practitioner level (`email`), with HC1 robust standard error options.

In pooled main effect specifications:

$$Y_{ij} = \beta_0 + \beta_1 \text{Treated}_i + \alpha_j + \varepsilon_{ij}$$

### 2. Standardization Protocol
To ensure scale invariance across tasks and sub-dimensions, all continuous quality scores are standardized relative to the control group baseline:

$$z_{ij} = \frac{x_{ij} - \bar{x}_{\text{Control}}}{s_{\text{Control}}}$$

where $\bar{x}_{\text{Control}}$ and $s_{\text{Control}}$ are the sample mean and sample standard deviation of the control group for that outcome measure (configured via `STD_USE_POOLED = False` in `config.py`).

### 3. Weighted Least Squares (WLS) for Multi-Rater Evaluations
When estimating regressions on multi-rater datasets (`TTs_raters_2026-08-10.csv`), observations are weighted by the inverse count of ratings per practitioner:

$$w_i = \frac{1}{n_i}$$

where $n_i$ is the total number of evaluations received by practitioner $i$, ensuring every practitioner carries identical aggregate weight in the econometric estimation.

### 4. Variance Regressions (Testing Dispersion Effects)
To assess whether AI assistance compresses or widens performance dispersion within and across experience cohorts, we estimate absolute residual regressions:

$$|e_{ij}| = \gamma_0 + \gamma_1 \text{Treated}_i + \gamma_2 \text{Junior}_i + \gamma_3 (\text{Treated}_i \times \text{Junior}_i) + \alpha_j + \nu_{ij}$$

accompanied by formal Levene tests for equality of variance.

### 5. Nonparametric Permutation Inference (Fisher Exact Tests)
To confirm that statistical significance is not an artifact of asymptotic assumptions or clustering, we perform 2,000 Monte Carlo permutations under the sharp null hypothesis of no treatment effect:
1. Treatment assignments are permuted across practitioners within randomization strata.
2. The OLS model is re-estimated for each permutation.
3. The empirical $t$-statistic is compared against the permutation null distribution to compute exact two-sided $p$-values.

---

## 8. Configuration & Customization Options (`config.py`)

All global parameters, variable mappings, and analytical toggles are centrally defined in [`config.py`](file:///usr/local/google/home/joshuabmartin/InFlow%202/config.py):

| Parameter | Default Value | Description |
|:---|:---|:---|
| **`EXPERIENCE_THRESHOLD`** | `7` | Experience cutoff (in years) dividing junior ($<7$) from senior ($\ge 7$) practitioners. |
| **`TREATMENT_VAR`** | `'group_binary'` | Primary binary treatment indicator column ($1 = \text{Treated}, 0 = \text{Control}$). |
| **`FIRM_VAR`** | `'firm_num'` | Firm identifier column for fixed effects and clustering. |
| **`UNIQUE_ID_VAR`** | `'email'` | Subject unique identifier for clustering and merging. |
| **`CONTROL_GROUP`** | `'Group 1'` | String label for control arm. |
| **`TREATMENT_GROUP`** | `'Group 2'` | String label for treatment arm. |
| **`STD_USE_POOLED`** | `False` | Standardization baseline: `False` = Control group mean/SD; `True` = Pooled sample mean/SD. |
| **`VARREGS_INCLUDE_LOW_N_FIRMS`** | `True` | `True` = Include all firms in variance regressions; `False` = Exclude micro-firms with $\le 2$ practitioners. |
| **`EXCLUDE_INDIVIDUALS`** | `True` | `True` = Filter out practitioner UUIDs listed in `exclusion_individuals.txt` (withdrawn consent). |
| **`EXCLUSION_INDIVIDUALS_FILE`** | `'exclusion_individuals.txt'` | Path to individual exclusion file. |
| **`EXCLUDE_FIRMS`** | `False` | `True` = Exclude specific firm IDs listed in `exclusion_firms.txt` during sensitivity checks. |
| **`EXCLUSION_FIRMS_FILE`** | `'exclusion_firms.txt'` | Path to firm exclusion file. |
| **`MAIN_OUTCOME_DIMS`** | `['enf', 'tec', 'str', 'com', 'cla']` | Five canonical quality sub-elements evaluated across tasks. |

---

## 9. Data Privacy, Anonymization & Ethical Compliance

All experimental procedures and data management comply with IRB requirements and practitioner confidentiality agreements:

1. **Deterministic Pseudonymization**: Practitioner names and email addresses are hashed using UUIDv5 under the DNS namespace (`uuid.uuid5(uuid.NAMESPACE_DNS, email.strip().lower())`). This ensures 100% reproducible two-way mapping across longitudinal waves while preventing identity disclosure.
2. **Evaluator Anonymization**: Expert human raters are pseudonymized via UUIDs (`Rater_1`, `Rater_2`, etc.).
3. **Firm Pseudonymization**: Law firms are designated sequentially as `Firm 1` through `Firm 19`.
4. **Traceability Key Management**: The master key mapping is stored in `anonymization_keys.csv` and verified by [anonymize.py](file:///usr/local/google/home/joshuabmartin/InFlow%202/anonymize.py) at the start of every replication run.
5. **Proprietary Patent Disclosures**: All underlying patent drafting texts and invention disclosures have been de-identified to protect unexamined intellectual property.

---

## 10. Replication Verification & Quality Checklist

To confirm complete, bit-for-bit replication fidelity, verify the following checklist after running `./replicate.sh`:

- [x] **Data Ingestion & Anonymization**: `anonymize.py` runs with exit code 0 and verifies 19 firms and 999 practitioners against `anonymization_keys.csv`.
- [x] **Statistical JSON Export**: All 14 JSON files in `Jsons/` are populated and non-empty.
- [x] **Macro Binding**: `New/macros.tex` contains 100% of the statistical macros cited in `draft_2026Aug20.tex` with zero unmapped `\stat{...}` warnings.
- [x] **Table Generation**: All 17 LaTeX tables (`New/*.tex`) are generated and compile cleanly with `booktabs` formatting.
- [x] **Figure Rendering**: All 9 PNG charts (`New/*.png`) are rendered at 300 DPI with publication styling.
- [x] **Manuscript PDF Compilation**: `draft_2026Aug20.pdf` compiles without errors, resolving all cross-references, macro numbers, tables, figures, and bibliographic citations.
