# SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models

*This is not an official Google product*

## Introduction

This repository contains the gridding library and a demo colab to use the SEEDS
models in the paper https://arxiv.org/abs/2306.14066.

## Model checkpoints and data

The associated data to reproduce all the results in the paper can be found in
the GCS bucket [gs://gresearch/seeds](https://console.cloud.google.com/storage/browser/gresearch/seeds). In particular,

- `model_checkpoints` contains the 22 models in the paper in the Tensorflow
  SavedModel format. See the demo colab for its usage.
- `data`
  - `gefs_forecast_2022_cubedsphere.zarr`: The GEFS forecasts for 2022 regridded
    and processed to be used as the model input. See the demo colab for the
    complete processing procedure from operational GEFS forecasts.
  - `era5_2022_cubedsphere.zarr`: The ERA5 reanalysis for 2022 regridded as the
    label for evaluation.
  - `climatology_cubedsphere.zarr`: The static climatology as part of the model
    input.