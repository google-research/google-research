# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file to specify hyperparameters."""

#  HYPERPARAMETERS HELD CONSTANT
static_args_dict = {
    # relevant only for "obvious" masking
    "continuous_mask_values": -100,
    # For masked value reconstruction, whether to also reconstruct padding
    "include_padding": False,
    # Gradient clipping hyperparameter
    "clipnorm": None,
    # Whether to normalize raw times to be 0-mean, unit-variance
    "raw_times": False,
    "batch_size": 256,
    # Learning rate for Adam optimizer
    "lr": 0.0005,
    # Patience for early stopping callback
    "patience": 5,
    # Hidden dimensions for neural network representation of time series
    "nn_d": 50,
    # Number of transformer blocks
    "nn_N": 2,
    # Number of heads in each transformer block
    "nn_he": 4,
    "dropout": 0.2,
    # Maximum number of epochs to run pretraining
    "max_pt_eps": 200,
    # For test results, percentages of labeled data to test
    "lds": [100, 50, 20, 10],
    # For each labeled data rate, number of replicates to test
    "repeats": {100: 5, 50: 10, 20: 10, 10: 10},
    # Path to saved geometric masks
    "geo_masks_dir": "geometric_masks_cached/",
    # Desired path for results to be saved
    "results_dir": "results/",
    # fraction to downsample when for testing code on smaller subset of data
    "downsampled_frac": None,
}

## Changing default hyperparameters:
## Uncomment this for very small/fast test example:
# static_args_dict["max_pt_eps"]=5
# static_args_dict["lds"] = [100,50,20,10]
# static_args_dict["repeats"] = {100: 2, 50: 2, 20: 2, 10: 2}
# static_args_dict["downsampled_frac"] = .1


## SETTING UP STRATEGIES TO SWEEP OVER
# Set hyperparameters to sweep over. Here, we experiment with pretraining
# task weightings, and several augmentation options
sweep_dict_list = {
    # Forecasting vs reconstruction loss weight
    "PT_task": [(0, 0), (0, 1), (1, 0), (1, 1), (1, 10), (10, 1)],
    "PT_recon_decoder": ["3hid"],
    # Gaussian noise standard deviation
    "aug_jitter_std": [0, 0.1],
    # Masking rate
    "aug_maskrate": [0, 0.3, 0.5, 0.8],
    # Way of sampling masks
    "aug_masksampling": ["random", "geometric"],
    # Part of the triplet to mask (all or just values)
    "aug_maskpart": ["all", "v"],
    # When masking a value, the value to replace it with
    "aug_maskval": ["obvious", "padding"],
    # Whether to use the same augmentation for finetuning or no augmentation
    "FT_aug": ["none", "same"],
}


# List of arguments that should be saved as the model for pretraining
PT_args_to_save = [
    "PT_task",
    "PT_recon_decoder",
    "aug_jitter_std",
    "aug_maskrate",
    "aug_masksampling",
    "aug_maskpart",
    "aug_maskval",
]
# List of arguments that should be saved as the model for finetuning
FT_args_to_save = [
    "PT_task",
    "PT_recon_decoder",
    "aug_jitter_std",
    "aug_maskrate",
    "aug_masksampling",
    "aug_maskpart",
    "aug_maskval",
    "FT_aug",
]

# When reading a string representation of the strategy setting, whether it
# should be converted to a numerical value (True) or left as a string (False)
args_dict_literaleval = {
    "FT_aug": False,
    "PT_recon_decoder": False,
    "PT_task": True,
    "aug_jitter_std": True,
    "aug_maskpart": False,
    "aug_maskrate": True,
    "aug_masksampling": False,
    "aug_maskval": False,
}
