# FluSight Hub

> **Problem Statement:** Refer to the [FluSight Hub Problem Statement](/problem_statements/flu_problem_statement.txt) for task definitions and submission guidelines.

## Model & Metadata Mapping

The table below maps the model names referenced in the paper and figures to
their corresponding metadata files (`model_metadata/*.ylm`), historical
submission scripts (`model_py/`), ensemble inclusion status, and output
directories:

| Paper / Figure Name | External Model ID (`model_id`) | In Ensemble | Metadata File (`model_metadata/`) | Code Script (`model_py/`) | Output Directory (`model_output/`) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `G-CFA_Pyrenew_Pyrenew_H_Flu` | `Google_SAI-Adapted_1` | No | [`Google_SAI-Adapted_1.ylm`](model_metadata/Google_SAI-Adapted_1.ylm) | [`model_py/Google_SAI-Adapted_1.py`](model_py/Google_SAI-Adapted_1.py) | [`model_output/Google_SAI-Adapted_1/`](model_output/Google_SAI-Adapted_1) |
| `*G-PSI-PROF_MOA` | `Google_SAI-Adapted_10` | **Yes** | [`Google_SAI-Adapted_10.ylm`](model_metadata/Google_SAI-Adapted_10.ylm) | [`model_py/Google_SAI-Adapted_10.py`](model_py/Google_SAI-Adapted_10.py) | [`model_output/Google_SAI-Adapted_10/`](model_output/Google_SAI-Adapted_10) |
| `*G-UMass_Flusion` | `Google_SAI-Adapted_11` | **Yes** | [`Google_SAI-Adapted_11.ylm`](model_metadata/Google_SAI-Adapted_11.ylm) | [`model_py/Google_SAI-Adapted_11.py`](model_py/Google_SAI-Adapted_11.py) | [`model_output/Google_SAI-Adapted_11/`](model_output/Google_SAI-Adapted_11) |
| `G-UVA_Gaussian_processes` | `Google_SAI-Adapted_12` | No | [`Google_SAI-Adapted_12.ylm`](model_metadata/Google_SAI-Adapted_12.ylm) | [`model_py/Google_SAI-Adapted_12.py`](model_py/Google_SAI-Adapted_12.py) | [`model_output/Google_SAI-Adapted_12/`](model_output/Google_SAI-Adapted_12) |
| `G-CMU_climate_baseline` | `Google_SAI-Adapted_13` | No | [`Google_SAI-Adapted_13.ylm`](model_metadata/Google_SAI-Adapted_13.ylm) | [`model_py/Google_SAI-Adapted_13.py`](model_py/Google_SAI-Adapted_13.py) | [`model_output/Google_SAI-Adapted_13/`](model_output/Google_SAI-Adapted_13) |
| `G-CMU_timeseries` | `Google_SAI-Adapted_14` | No | [`Google_SAI-Adapted_14.ylm`](model_metadata/Google_SAI-Adapted_14.ylm) | [`model_py/Google_SAI-Adapted_14.py`](model_py/Google_SAI-Adapted_14.py) | [`model_output/Google_SAI-Adapted_14/`](model_output/Google_SAI-Adapted_14) |
| `G-UMass_KCDE` | `Google_SAI-Adapted_15` | No | [`Google_SAI-Adapted_15.ylm`](model_metadata/Google_SAI-Adapted_15.ylm) | [`model_py/Google_SAI-Adapted_15.py`](model_py/Google_SAI-Adapted_15.py) | [`model_output/Google_SAI-Adapted_15/`](model_output/Google_SAI-Adapted_15) |
| `G-CU_SIRS` | `Google_SAI-Adapted_16` | No | [`Google_SAI-Adapted_16.ylm`](model_metadata/Google_SAI-Adapted_16.ylm) | [`model_py/Google_SAI-Adapted_16.py`](model_py/Google_SAI-Adapted_16.py) | [`model_output/Google_SAI-Adapted_16/`](model_output/Google_SAI-Adapted_16) |
| `G-PSI-PROF_MOA_v2` | `Google_SAI-Adapted_17` | No | [`Google_SAI-Adapted_17.ylm`](model_metadata/Google_SAI-Adapted_17.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_17/`](model_output/Google_SAI-Adapted_17) |
| `G-Cornell_JHU-hierarchSIR_v3` | `Google_SAI-Adapted_18` | No | [`Google_SAI-Adapted_18.ylm`](model_metadata/Google_SAI-Adapted_18.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_18/`](model_output/Google_SAI-Adapted_18) |
| `G-LANL_DBM_v2` | `Google_SAI-Adapted_19` | No | [`Google_SAI-Adapted_19.ylm`](model_metadata/Google_SAI-Adapted_19.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_19/`](model_output/Google_SAI-Adapted_19) |
| `G-PSI_PROF` | `Google_SAI-Adapted_2` | No | [`Google_SAI-Adapted_2.ylm`](model_metadata/Google_SAI-Adapted_2.ylm) | [`model_py/Google_SAI-Adapted_2.py`](model_py/Google_SAI-Adapted_2.py) | [`model_output/Google_SAI-Adapted_2/`](model_output/Google_SAI-Adapted_2) |
| `G-NU-PGF_FLUH_v4` | `Google_SAI-Adapted_20` | No | [`Google_SAI-Adapted_20.ylm`](model_metadata/Google_SAI-Adapted_20.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_20/`](model_output/Google_SAI-Adapted_20) |
| `G-NU-PGF_FLUH_v3` | `Google_SAI-Adapted_21` | **Yes** | [`Google_SAI-Adapted_21.ylm`](model_metadata/Google_SAI-Adapted_21.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_21/`](model_output/Google_SAI-Adapted_21) |
| `G-NU-PGF_FLUH_v2` | `Google_SAI-Adapted_22` | **Yes** | [`Google_SAI-Adapted_22.ylm`](model_metadata/Google_SAI-Adapted_22.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_22/`](model_output/Google_SAI-Adapted_22) |
| `G-LANL_DBM_v3` | `Google_SAI-Adapted_23` | **Yes** | [`Google_SAI-Adapted_23.ylm`](model_metadata/Google_SAI-Adapted_23.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_23/`](model_output/Google_SAI-Adapted_23) |
| `G-LANL_DBM` | `Google_SAI-Adapted_24` | No | [`Google_SAI-Adapted_24.ylm`](model_metadata/Google_SAI-Adapted_24.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_24/`](model_output/Google_SAI-Adapted_24) |
| `G-Cornell_JHU-hierarchSIR_v4` | `Google_SAI-Adapted_25` | No | [`Google_SAI-Adapted_25.ylm`](model_metadata/Google_SAI-Adapted_25.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_25/`](model_output/Google_SAI-Adapted_25) |
| `G-Cornell_JHU-hierarchSIR_v2` | `Google_SAI-Adapted_26` | No | [`Google_SAI-Adapted_26.ylm`](model_metadata/Google_SAI-Adapted_26.ylm) | `model_py/` | [`model_output/Google_SAI-Adapted_26/`](model_output/Google_SAI-Adapted_26) |
| `*G-UGA_flucast-INFLAenza` | `Google_SAI-Adapted_3` | **Yes** | [`Google_SAI-Adapted_3.ylm`](model_metadata/Google_SAI-Adapted_3.ylm) | [`model_py/Google_SAI-Adapted_3.py`](model_py/Google_SAI-Adapted_3.py) | [`model_output/Google_SAI-Adapted_3/`](model_output/Google_SAI-Adapted_3) |
| `G-UGA_flucast_Copycat` | `Google_SAI-Adapted_4` | No | [`Google_SAI-Adapted_4.ylm`](model_metadata/Google_SAI-Adapted_4.ylm) | [`model_py/Google_SAI-Adapted_4.py`](model_py/Google_SAI-Adapted_4.py) | [`model_output/Google_SAI-Adapted_4/`](model_output/Google_SAI-Adapted_4) |
| `G-UGuelph_CompositeCurve` | `Google_SAI-Adapted_5` | No | [`Google_SAI-Adapted_5.ylm`](model_metadata/Google_SAI-Adapted_5.ylm) | [`model_py/Google_SAI-Adapted_5.py`](model_py/Google_SAI-Adapted_5.py) | [`model_output/Google_SAI-Adapted_5/`](model_output/Google_SAI-Adapted_5) |
| `G-UMass-ar6_pooled` | `Google_SAI-Adapted_6` | No | [`Google_SAI-Adapted_6.ylm`](model_metadata/Google_SAI-Adapted_6.ylm) | [`model_py/Google_SAI-Adapted_6.py`](model_py/Google_SAI-Adapted_6.py) | [`model_output/Google_SAI-Adapted_6/`](model_output/Google_SAI-Adapted_6) |
| `G-UMass-gbqr` | `Google_SAI-Adapted_7` | No | [`Google_SAI-Adapted_7.ylm`](model_metadata/Google_SAI-Adapted_7.ylm) | [`model_py/Google_SAI-Adapted_7.py`](model_py/Google_SAI-Adapted_7.py) | [`model_output/Google_SAI-Adapted_7/`](model_output/Google_SAI-Adapted_7) |
| `G-NU-PGF_FLUH` | `Google_SAI-Adapted_8` | No | [`Google_SAI-Adapted_8.ylm`](model_metadata/Google_SAI-Adapted_8.ylm) | [`model_py/Google_SAI-Adapted_8.py`](model_py/Google_SAI-Adapted_8.py) | [`model_output/Google_SAI-Adapted_8/`](model_output/Google_SAI-Adapted_8) |
| `*G-Cornell_JHU-hierarchSIR` | `Google_SAI-Adapted_9` | **Yes** | [`Google_SAI-Adapted_9.ylm`](model_metadata/Google_SAI-Adapted_9.ylm) | [`model_py/Google_SAI-Adapted_9.py`](model_py/Google_SAI-Adapted_9.py) | [`model_output/Google_SAI-Adapted_9/`](model_output/Google_SAI-Adapted_9) |
| `G-LANL_DBM x LANL_Inferno` | `Google_SAI-Hybrid_1` | **Yes** | [`Google_SAI-Hybrid_1.ylm`](model_metadata/Google_SAI-Hybrid_1.ylm) | [`model_py/Google_SAI-Hybrid_1.py`](model_py/Google_SAI-Hybrid_1.py) | [`model_output/Google_SAI-Hybrid_1/`](model_output/Google_SAI-Hybrid_1) |
| `G-CMU_climate_baseline x UGA_flucast_Copycat` | `Google_SAI-Hybrid_2` | **Yes** | [`Google_SAI-Hybrid_2.ylm`](model_metadata/Google_SAI-Hybrid_2.ylm) | [`model_py/Google_SAI-Hybrid_2.py`](model_py/Google_SAI-Hybrid_2.py) | [`model_output/Google_SAI-Hybrid_2/`](model_output/Google_SAI-Hybrid_2) |
| `G-CU_SIRS x UVA_Gaussian_processes` | `Google_SAI-Hybrid_3` | **Yes** | [`Google_SAI-Hybrid_3.ylm`](model_metadata/Google_SAI-Hybrid_3.ylm) | [`model_py/Google_SAI-Hybrid_3.py`](model_py/Google_SAI-Hybrid_3.py) | [`model_output/Google_SAI-Hybrid_3/`](model_output/Google_SAI-Hybrid_3) |
| `G-CMU_climate_baseline x UGuelph_CompositeCurve` | `Google_SAI-Hybrid_4` | No | [`Google_SAI-Hybrid_4.ylm`](model_metadata/Google_SAI-Hybrid_4.ylm) | [`model_py/Google_SAI-Hybrid_4.py`](model_py/Google_SAI-Hybrid_4.py) | [`model_output/Google_SAI-Hybrid_4/`](model_output/Google_SAI-Hybrid_4) |
| `G-CU_SIRS x CMU_climate_baseline` | `Google_SAI-Hybrid_5` | **Yes** | [`Google_SAI-Hybrid_5.ylm`](model_metadata/Google_SAI-Hybrid_5.ylm) | [`model_py/Google_SAI-Hybrid_5.py`](model_py/Google_SAI-Hybrid_5.py) | [`model_output/Google_SAI-Hybrid_5/`](model_output/Google_SAI-Hybrid_5) |
| `*G-LANL_DBM x UMass_Flusion` | `Google_SAI-Hybrid_6` | **Yes** | [`Google_SAI-Hybrid_6.ylm`](model_metadata/Google_SAI-Hybrid_6.ylm) | `model_py/` | [`model_output/Google_SAI-Hybrid_6/`](model_output/Google_SAI-Hybrid_6) |
| `*G-multi-layer-SE` | `Google_SAI-Novel_1` | **Yes** | [`Google_SAI-Novel_1.ylm`](model_metadata/Google_SAI-Novel_1.ylm) | [`model_py/Google_SAI-Novel_1.py`](model_py/Google_SAI-Novel_1.py) | [`model_output/Google_SAI-Novel_1/`](model_output/Google_SAI-Novel_1) |
| `G-Time_Series_to_Vision_Transfer_Learning` | `Google_SAI-Novel_2` | No | [`Google_SAI-Novel_2.ylm`](model_metadata/Google_SAI-Novel_2.ylm) | [`model_py/Google_SAI-Novel_2.py`](model_py/Google_SAI-Novel_2.py) | [`model_output/Google_SAI-Novel_2/`](model_output/Google_SAI-Novel_2) |

> **Note:** Code scripts are not yet available for all models. Links will be
> added as they become available.

## References & Acknowledgments

The ERA prompts and model metadata in this directory are synthesized from public
literature and open model documentation provided by forecasting teams
participating in the CDC FluSight Hub:

- **FluSight Hub Repository**: [https://github.com/cdcepi/FluSight-forecast-hub](https://github.com/cdcepi/FluSight-forecast-hub)
