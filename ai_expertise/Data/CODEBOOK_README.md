# InFlow Replication Package: Data Codebook & Documentation

This directory contains the canonical datasets used in the empirical analysis for the study:
**"Does AI Assistance Enhance or Erode Expertise? Evidence from a Three-Month Field Experiment in Patent Drafting"**

---

## 1. Overview of Data Files

The empirical pipeline uses three primary datasets:

| File Name | Description | Unit of Observation | Dimensions |
|:---|:---|:---|:---|
| `maindata_2026-08-10.csv` | Primary experimental dataset containing practitioner demographics, treatment assignments, baseline firm patenting metrics, aggregated task quality scores, time-on-task metrics, and post-task survey responses. | Individual practitioner ($N = 999$) | 999 rows, 136 columns |
| `TTs_raters_2026-08-10.csv` | Granular evaluation dataset containing individual scoring records from both expert human patent practitioners and LLM evaluators across 5 quality dimensions. | Practitioner $\times$ Task $\times$ Evaluator | 1,124 rows, 76 columns |
| `codebook_2026-08-10.csv` | Variable mapping metadata linking raw survey/evaluation column strings to standardized econometric identifiers, LaTeX labels, data types, and scale directionality. | Variable ($N = 1,017$) | 1,017 rows, 6 columns |

---

## 2. Core Experimental Variables (`maindata_2026-08-10.csv`)

### Practitioner Identification & Randomization
* **`Email` / `email`**: De-identified unique practitioner identifier.
* **`Firm number` / `firm_num`**: Unique firm identifier used for firm fixed effects and clustering standard errors.
* **`Firm name` / `firm_name`**: Anonymized firm name identifier.
* **`Group` / `group`**: Experimental arm assignment:
  * `Group 1`: Control group (practitioners completed tasks without AI tool access).
  * `Group 2`: Treatment group (practitioners had access to the InFlow generative AI copilot).
* **`Group_binary` / `group_binary`**: Binary indicator equal to `1` for Treatment (`Group 2`) and `0` for Control (`Group 1`).
* **`Included` / `included`**: Analysis sample indicator (`1` = verified patent practitioner who completed the baseline instruments and consent).

### Experience & Stratification
* **`Experience` / `exp`**: Years of registered patent prosecution and drafting experience.
* **`Junior/expert` / `seniority`**: Experience classification:
  * `Senior`: Practitioners with $\ge 7$ years of patent drafting experience.
  * `Junior`: Practitioners with $< 7$ years of patent drafting experience.
* **`junior`**: Binary indicator (`1` if `Experience < 7`, `0` otherwise).
* **`senior`**: Binary indicator (`1` if `Experience >= 7`, `0` otherwise).

---

## 3. Experimental Task Outcomes

### 10-Day Evaluation Task (Task 1: Drafting Only)
* **`tt1_rat_drades_sum`**: Overall quality score on the 10-day drafting task (standardized to mean 0, SD 1 on the control baseline).
* **`tt1_rat_drades_enf`**: Enforceability sub-scale score for 10-day drafting.
* **`tt1_rat_drades_acc`**: Technical accuracy sub-scale score for 10-day drafting.
* **`tt1_rat_drades_amb`**: Strategic ambiguity sub-scale score for 10-day drafting.
* **`tt1_rat_drades_com`**: Completeness & alignment with invention disclosure sub-scale score for 10-day drafting.
* **`tt1_rat_drades_cla`**: Clarity sub-scale score for 10-day drafting.
* **`tt1_sum_dra_tot`**: Realized time spent completing the 10-day drafting task (minutes).

### 90-Day Evaluation Task (Task 2: Drafting & Redlining)
* **`tt2_rat_drades_sum`**: Overall quality score on the 90-day drafting task (standardized to mean 0, SD 1 on the control baseline).
* **`tt2_rat_drades_enf`**: Enforceability sub-scale score for 90-day drafting.
* **`tt2_rat_drades_acc`**: Technical accuracy sub-scale score for 90-day drafting.
* **`tt2_rat_drades_amb`**: Strategic ambiguity sub-scale score for 90-day drafting.
* **`tt2_rat_drades_com`**: Completeness & alignment sub-scale score for 90-day drafting.
* **`tt2_rat_drades_cla`**: Clarity sub-scale score for 90-day drafting.
* **`tt2_sum_dra_tot`**: Realized time spent completing the 90-day drafting task (minutes).
* **`tt2_rat_cri_sum`**: Overall quality score on the 90-day redlining/critique task (standardized to mean 0, SD 1 on the control baseline).
* **`tt2_rat_cri_enf`**: Enforceability sub-scale score for 90-day redlining.
* **`tt2_rat_cri_acc`**: Technical accuracy sub-scale score for 90-day redlining.
* **`tt2_rat_cri_amb`**: Strategic ambiguity sub-scale score for 90-day redlining.
* **`tt2_rat_cri_com`**: Completeness & alignment sub-scale score for 90-day redlining.
* **`tt2_rat_cri_cla`**: Clarity sub-scale score for 90-day redlining.
* **`tt2_sum_cri_tot`**: Realized time spent completing the 90-day redlining task (minutes).

---

## 4. On-the-Job Firm Patent Drafting Metrics

Firm administrative metrics tracking actual drafting behavior and patent filings across pre- and post-treatment windows:
* **`pat_num`**: Total number of patent applications drafted and filed.
* **`pat_avg_res`**: Average number of office action responses filed.
* **`pat_avg_len`**: Average word count / length of detailed description sections.
* **`pat_avg_rev`**: Average number of revision rounds between junior and supervising senior practitioner.
* **`pat_avg_cor`**: Average number of substantive correction notes on draft patent claims.

---

## 5. Subjective Survey & Perceptions Data

Post-task subjective evaluations measured on 1–5 Likert scales:
* **`tt1_sur_dra_spe` / `tt2_sur_dra_spe`**: Practitioner self-reported perception of drafting speed (1 = Much slower, 5 = Much faster).
* **`tt1_sur_dra_qua` / `tt2_sur_dra_qua`**: Practitioner self-reported perception of draft quality (1 = Very poor, 5 = Exceptional).
* **`tt1_sur_dra_sat` / `tt2_sur_dra_sat`**: Practitioner satisfaction with task workflow and output (1 = Very dissatisfied, 5 = Very satisfied).
* **`tt2_sur_cri_spe` / `tt2_sur_cri_qua` / `tt2_sur_cri_sat`**: Practitioner self-reported speed, quality, and satisfaction on the redlining exercise (1–5 scale).

---

## 6. Raters Evaluation Dataset (`TTs_raters_2026-08-10.csv`)

Contains individual evaluations from expert human raters and LLM evaluators:
* **`Email`**: De-identified subject identifier.
* **`Test Task`**: Task identifier (`TT1` for 10-day drafting, `TT2` for 90-day drafting & redlining).
* **`Group`**: Treatment assignment of the subject.
* **`Rater ID`**: Unique rater identifier:
  * Human expert patent attorneys: `Rater_1`, `Rater_2`, etc.
  * LLM automated evaluators: `LLM_Rater_1`, etc.
* **Quality Sub-Scales**: Individual ratings (1–5 scale) across `Enforceability`, `Technical Accuracy`, `Strategic Ambiguity`, `Completeness and Alignment with Objectives`, and `Clarity`.

---

## 7. Metadata Codebook (`codebook_2026-08-10.csv`)

A structured schema file containing:
* **`original`**: Exact raw column header in the raw export.
* **`varname`**: Standardized snake_case variable name used in Python analysis scripts.
* **`label_shorthand`**: Concise text label for table stubs and axis titles.
* **`column`**: Original column letter index.
* **`type`**: Variable classification (`Demographic`, `Rating`, `Time`, `Survey`, `Administrative`).
* **`highermeans`**: Directional interpretation of scale (e.g., `higher score`, `longer duration`).
