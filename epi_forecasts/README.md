# Google Research Respiratory Infections Forecasting Hub

This repository serves as a public archive for prospective epidemiological forecasts used to create the `Google-SAI` ensemble submissions to CDC's [FluSight](https://github.com/cdcepi/FluSight-forecast-hub), [COVIDHub](https://github.com/CDCgov/covid19-forecast-hub), and [RSVHub](https://github.com/CDCgov/rsv-forecast-hub).
Forecasts cover three public health targets: US jurisdiction-level quantile
forecasts for weekly influenza, COVID-19, and RSV hospitalizations.

> **Companion Paper:** *[Prospective multi-pathogen disease forecasting using autonomous LLM-guided tree search](https://arxiv.org/abs/2605.16238)* (arXiv:2605.16238)

## Forecast Model Directories

Metadata, forecasts, and model code are organized by pathogen target:

- **[FluSight Hub (`flu_hub/`)](flu_hub/README.md)**: Models, metadata, and forecasts for influenza hospitalizations.
- **[COVIDHub (`covid_hub/`)](covid_hub/README.md)**: Models, metadata, and forecasts for COVID-19 hospitalizations.
- **[RSVHub (`rsv_hub/`)](rsv_hub/README.md)**: Models, metadata, and forecasts for RSV hospitalizations.

Each pathogen directory contains a `README.md` mapping model names to their
corresponding model metadata files (`model_metadata/*.ylm`), model code
(`model_py/`), and predictions (`model_output/`).
All forecasts are provided in a standardized CSV format, consistent with the
requirements of CDC's forecasting hubs.

The metadata files (`model_metadata/*.ylm`) in `covid_hub` and `flu_hub` contain
the full prompt (specific instruction) text used to seed Empirical Research
Assistance ([ERA](https://research.google/blog/empirical-research-assistance-era-from-nature-publication-to-catalyzing-computational-discovery/)) runs to generate corresponding forecasts.
For `rsv_hub` models, forecasts were found by unconstrained ERA searches (see
the paper for details).

## Problem Statement Prompts

Broader problem statements used to seed ERA searches for each pathogen are also
included. These are intended to provide context on the specific forecasting
tasks. Specific instructions for each ERA search can be found in the metadata
files.

- [Problem Statements](problem_statements/README.md): Task definitions, output specs, and evaluation metrics for Flu, COVID-19, and RSV.
  - [Influenza Problem Statement](problem_statements/flu_problem_statement.txt)
  - [COVID-19 Problem Statement](problem_statements/covid19_problem_statement.txt)
  - [RSV Problem Statement](problem_statements/rsv_problem_statement.txt)

## References & Acknowledgments

The structure and implementation of this repository are based on open-source
public health forecasting infrastructure provided by the US CDC:

- [FluSight Forecast Hub](https://github.com/cdcepi/FluSight-forecast-hub)
- [COVID-19 Forecast Hub](https://github.com/CDCgov/covid19-forecast-hub)
- [RSV Forecast Hub](https://github.com/CDCgov/rsv-forecast-hub)

Some model descriptions used as specific instructions to seed ERA runs for influenza and COVID-19 forecasts were taken and adapted from teams submitting models to [FluSight](https://github.com/cdcepi/FluSight-forecast-hub) and/or [COVIDHub](https://github.com/CDCgov/covid19-forecast-hub).
These are embedded directly inside the model metadata files in [`flu_hub/model_metadata/`](flu_hub/model_metadata/) and [`covid_hub/model_metadata/`](covid_hub/model_metadata/)—links to the original model metadata written by the authors and citations pointing to original works are included in each file.

We thank all authors involved in creating this source material--without it, much
of this study would not have been possible.
We additionally thank all coauthors of the original [ERA paper](https://www.nature.com/articles/s41586-026-10658-6) for system development, input and feedback.

Please refer to the bibliography in [our paper](https://arxiv.org/abs/2605.16238) for citations of all contributing models and other relevant sources.

## Citation
If you use this repository, prompts, or model code in your research, please cite
our paper:

```
@misc{martinson2026prospectivemultipathogendiseaseforecasting,
      title={Prospective multi-pathogen disease forecasting using autonomous LLM-guided tree search},
      author={Sarah Martinson and Michael P. Brenner and Martyna Plomecka and Brian P. Williams and Nicholas G. Reich and Zahra Shamsi},
      year={2026},
      eprint={2605.16238},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.16238},
}
```
