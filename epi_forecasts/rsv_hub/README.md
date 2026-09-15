# RSV Forecast Hub

> **Problem Statement:** Refer to the [RSV Forecast Hub Problem Statement](/problem_statements/rsv_problem_statement.txt) for task definitions and submission guidelines.

## Model & Metadata Mapping

The table below maps the model names referenced in the paper and figures to
their corresponding metadata files (`model_metadata/*.ylm`), historical
submission scripts (`model_py/`), ensemble inclusion status, and output
directories:

| Paper / Figure Name | External Model ID (`model_id`) | In Ensemble | Metadata File (`model_metadata/`) | Code Script (`model_py/`) | Output Directory (`model_output/`) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `G-general_instruction_lgbm` | `Google_SAI-Novel_1` | **Yes** | [`Google_SAI-Novel_1.ylm`](model_metadata/Google_SAI-Novel_1.ylm) | [`model_py/Google_SAI-Novel_1.py`](model_py/Google_SAI-Novel_1.py) | [`model_output/Google_SAI-Novel_1/`](model_output/Google_SAI-Novel_1) |
| `G-general_instruction` | `Google_SAI-Novel_2` | **Yes** | [`Google_SAI-Novel_2.ylm`](model_metadata/Google_SAI-Novel_2.ylm) | `model_py/` | [`model_output/Google_SAI-Novel_2/`](model_output/Google_SAI-Novel_2) |

> **Note:** Code scripts are not yet available for all models. Links will be
> added as they become available.

## References & Acknowledgments

The task formulation and evaluation framework are based on the CDC RSV Forecast
Hub:

- **RSV Forecast Hub Repository**: [https://github.com/CDCgov/rsv-forecast-hub](https://github.com/CDCgov/rsv-forecast-hub)
