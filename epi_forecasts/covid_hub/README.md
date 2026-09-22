# COVID-19 Forecast Hub

> **Problem Statement:** Refer to the [COVID-19 Forecast Hub Problem Statement](/problem_statements/covid19_problem_statement.txt) for task definitions and submission guidelines.

## Model & Metadata Mapping

The table below maps the model names referenced in the paper and figures to
their corresponding metadata files (`model_metadata/*.ylm`), historical
submission scripts (`model_py/`), ensemble inclusion status, and output
directories:

| Paper / Figure Name | External Model ID (`model_id`) | In Ensemble | Metadata File (`model_metadata/`) | Code Script (`model_py/`) | Output Directory (`model_output/`) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `*G-UM-DeepOutbreak` | `Google_SAI-Adapted_1` | **Yes** | [`Google_SAI-Adapted_1.ylm`](model_metadata/Google_SAI-Adapted_1.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_1/`](model_output/Google_SAI-Adapted_1) |
| `G-CMU-TimeSeries` | `Google_SAI-Adapted_10` | No | [`Google_SAI-Adapted_10.ylm`](model_metadata/Google_SAI-Adapted_10.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_10/`](model_output/Google_SAI-Adapted_10) |
| `G-JHU_CSSE-CSSE_Ensemble` | `Google_SAI-Adapted_11` | No | [`Google_SAI-Adapted_11.ylm`](model_metadata/Google_SAI-Adapted_11.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_11/`](model_output/Google_SAI-Adapted_11) |
| `G-CMU-climate_baseline` | `Google_SAI-Adapted_12` | No | [`Google_SAI-Adapted_12.ylm`](model_metadata/Google_SAI-Adapted_12.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_12/`](model_output/Google_SAI-Adapted_12) |
| `G-CEPH-Rtrend_covid` | `Google_SAI-Adapted_13` | No | [`Google_SAI-Adapted_13.ylm`](model_metadata/Google_SAI-Adapted_13.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_13/`](model_output/Google_SAI-Adapted_13) |
| `*G-MOBS-GLEAM_COVID` | `Google_SAI-Adapted_2` | **Yes** | [`Google_SAI-Adapted_2.ylm`](model_metadata/Google_SAI-Adapted_2.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_2/`](model_output/Google_SAI-Adapted_2) |
| `*G-NEU_ISI-AdaptiveEnsemble` | `Google_SAI-Adapted_3` | **Yes** | [`Google_SAI-Adapted_3.ylm`](model_metadata/Google_SAI-Adapted_3.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_3/`](model_output/Google_SAI-Adapted_3) |
| `*G-UMass-gbqr` | `Google_SAI-Adapted_4` | **Yes** | [`Google_SAI-Adapted_4.ylm`](model_metadata/Google_SAI-Adapted_4.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_4/`](model_output/Google_SAI-Adapted_4) |
| `*G-CADPH-CovidCAT_Ensemble` | `Google_SAI-Adapted_5` | **Yes** | [`Google_SAI-Adapted_5.ylm`](model_metadata/Google_SAI-Adapted_5.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_5/`](model_output/Google_SAI-Adapted_5) |
| `G-UMass-ar6_pooled` | `Google_SAI-Adapted_6` | No | [`Google_SAI-Adapted_6.ylm`](model_metadata/Google_SAI-Adapted_6.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_6/`](model_output/Google_SAI-Adapted_6) |
| `G-CFA-EpiAutoGP` | `Google_SAI-Adapted_7` | No | [`Google_SAI-Adapted_7.ylm`](model_metadata/Google_SAI-Adapted_7.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_7/`](model_output/Google_SAI-Adapted_7) |
| `G-Metaculus-cp` | `Google_SAI-Adapted_8` | No | [`Google_SAI-Adapted_8.ylm`](model_metadata/Google_SAI-Adapted_8.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_8/`](model_output/Google_SAI-Adapted_8) |
| `G-UGA_flucast-INFLAenza` | `Google_SAI-Adapted_9` | No | [`Google_SAI-Adapted_9.ylm`](model_metadata/Google_SAI-Adapted_9.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_9/`](model_output/Google_SAI-Adapted_9) |
| `*G-CMU_TimeSeries-UMass_gbqr` | `Google_SAI-Hybrid_1` | **Yes** | [`Google_SAI-Hybrid_1.ylm`](model_metadata/Google_SAI-Hybrid_1.ylm) | [`model_py/Google_SAI-Hybrid_1.py`](model_py/Google_SAI-Hybrid_1.py) | [`model_output/Google_SAI-Hybrid_1/`](model_output/Google_SAI-Hybrid_1) |
| `*G-CMU_climate_baseline-UMass_ar6_pooled` | `Google_SAI-Hybrid_2` | **Yes** | [`Google_SAI-Hybrid_2.ylm`](model_metadata/Google_SAI-Hybrid_2.ylm) | [`model_py/Google_SAI-Hybrid_2.py`](model_py/Google_SAI-Hybrid_2.py) | [`model_output/Google_SAI-Hybrid_2/`](model_output/Google_SAI-Hybrid_2) |
| `*G-CEPH_Rtrend_covid-CMU_climate_baseline` | `Google_SAI-Hybrid_3` | **Yes** | [`Google_SAI-Hybrid_3.ylm`](model_metadata/Google_SAI-Hybrid_3.ylm) | [`model_py/Google_SAI-Hybrid_3.py`](model_py/Google_SAI-Hybrid_3.py) | [`model_output/Google_SAI-Hybrid_3/`](model_output/Google_SAI-Hybrid_3) |
| `*G-DeepResearch_RegimeSwitchingDetection` | `Google_SAI-Novel_1` | **Yes** | [`Google_SAI-Novel_1.ylm`](model_metadata/Google_SAI-Novel_1.ylm) | [`model_py/Google_SAI-Novel_1.py`](model_py/Google_SAI-Novel_1.py) | [`model_output/Google_SAI-Novel_1/`](model_output/Google_SAI-Novel_1) |
| `G-DeepResearch_CounterfactualSimulation` | `Google_SAI-Novel_2` | No | [`Google_SAI-Novel_2.ylm`](model_metadata/Google_SAI-Novel_2.ylm) | [`model_py/Google_SAI-Novel_2.py`](model_py/Google_SAI-Novel_2.py) | [`model_output/Google_SAI-Novel_2/`](model_output/Google_SAI-Novel_2) |

> **Note:** Code scripts are not yet available for all models. Links will be
> added as they become available.

## References & Acknowledgments

The ERA prompts and model metadata in this directory are synthesized from public
literature and open model documentation provided by forecasting teams
participating in the CDC COVID-19 Forecast Hub:

- **COVID-19 Forecast Hub Repository**: [https://github.com/CDCgov/covid19-forecast-hub](https://github.com/CDCgov/covid19-forecast-hub)
