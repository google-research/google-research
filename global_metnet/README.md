# Global MetNet: Precipitation Nowcasting Model

This directory contains the open-sourced code for our global precipitation
nowcasting model.

For more details on the model architecture, training, and evaluation, please
refer to our paper available at: <https://arxiv.org/pdf/2510.13050>

## Global MetNet Precipitation Nowcasting Dataset

The complete Global MetNet precipitation nowcasting data is made publicly
available for researchers. The dataset offers:

-   **Historical Reforecasts**: High-resolution precipitation nowcasts
    available from 2020 to present. These reforecasts have a cadence of 3
    hours (8 initializations per day).
-   **Real-time Forecasts**: Real-time production forecasts are also available
    and initialized every 30 minutes.

This data product provides high-resolution (5km spatial, 15 min temporal, up
to 12 hr ahead) probabilistic precipitation forecasts, including optimal
probability thresholds for conversion to categorical values. The dataset is
accessible via a [Google Earth Engine Catalog][gee-catalog] provided the user
agrees to the terms and conditions outlined in this [Google form][google-form].

## System Requirements
This code is tested on Linux. The python dependencies are listed in
`requirements.txt`.

* absl-py
* flax
* jax
* ml_collections
* numpy
* optax
* tensorflow
* tensorflow-probability

Specific versions are not pinned, and you might need to adjust them based on
your environment. We recommend using a virtual environment to manage
dependencies. The current model was trained on TPUs though a GPU may also
work by reducing the dimensions of each training example than what we used
which was 3600X7200.

## Installation Guide
1. Clone or download this repository.
2. Navigate to the `global_metnet` directory of the repository.
3. Install dependencies using pip:
```
pip install -r requirements.txt
```
Installation should take a few minutes on a standard desktop computer
depending on network speed and already installed packages.

## Demo and Instructions for Use
This repository provides the code for the Global MetNet model architecture,
including preprocessors and normalization utilities. It does not include
scripts for running inference or training, or pre-trained model weights.
To run train with this model:
1. Create a training dataset and modify config.py to set the dimensions of
    your training dataset.
2. Implement a data loader for samples from the dataset and add that to the
    `dataset_preprocessor.DatasetPreprocessor` class.
3. Use the model definition in `model.py` and configuration in `config.py` to
    run train on your data using JAX/Flax. `trainer.py` is an example of a
    training setup to call all the relevant training functionalities.

We train up to 100k steps and depending on hardware and size of the data it
could take 1-5 days to train. Once the model is trained running inference
should be a matter of a few mins.

To ensure the trained model results are reproducible, we encourage training a
few times to make sure the evaluation results are similar across several
training runs.

The expected output of the model with the current config is a probabilistic
precipitation forecast for up to 12 hours ahead, at 5km spatial and 15 min
temporal resolution.

[gee-catalog]: https://code.earthengine.google.com/?asset=projects/global-precipitation-nowcast/assets/metnet_nowcast
[google-form]: https://docs.google.com/forms/d/e/1FAIpQLSeObgf53sXtaZaim08YZYiP_KFTbiIiWYzzs1_LXCeQIKuqlw/viewform
