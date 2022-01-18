# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Caltrain library."""
import seaborn as sns

SAVEFIG_FORMAT = 'pdf'

TRUE_DATASETS = [
    'logistic', 'logistic_beta', 'polynomial', 'flip_polynomial',
    'two_param_polynomial', 'two_param_flip_polynomial', 'logistic_log_odds',
    'logistic_two_param_flip_polynomial'
]

dataset_mlmodel_imageset_map = {
    'resnet110_c10': ('resnet', 'c10'),
    'densenet40_c10': ('densenet', 'c10'),
    'resnet110_SD_c10': ('resnet_SD', 'c10'),
    'resnet_wide32_c10': ('wide_resnet', 'c10'),
    'resnet110_c100': ('resnet', 'c100'),
    'densenet40_c100': ('densenet', 'c100'),
    'resnet152_imgnet': ('resnet', 'imgnet'),
    'densenet161_imgnet': ('densenet', 'imgnet'),
    'resnet110_SD_c100': ('resnet_SD', 'c100'),
    'resnet_wide32_c100': ('wide_resnet', 'c100'),
}

mlmodel_linestyle_map = {
    'resnet': '-',
    'densenet': '--',
    'resnet_SD': '-.',
    'wide_resnet': ':'
}

mlmodel_marker_map = {
    'resnet': '*',
    'densenet': '^',
    'resnet_SD': 'o',
    'wide_resnet': 'd'
}

clrs = sns.color_palette('husl', n_colors=3)

imageset_color_map = {
    'c10': clrs[0],
    'c100': clrs[1],
    'imgnet': clrs[2],
}

cetype_color_map = {
    'em_ece_bin': 'blue',
    'ew_ece_bin': 'navy',
    'em_ece_sweep': 'red',
    'ew_ece_sweep': 'darkred'
}

ce_type_paper_name_map = {
    'em_ece_bin': 'EM',
    'ew_ece_bin': 'EW',
    'em_ece_sweep': 'EMsweep',
    'ew_ece_sweep': 'EWsweep'
}

ml_model_name_map = {
    'resnet110_c10': 'ResNet',
    'densenet40_c10': 'DenseNet',
    'resnet110_SD_c10': 'ResNet_SD',
    'resnet_wide32_c10': 'Wide_ResNet',
    'resnet110_c100': 'ResNet',
    'densenet40_c100': 'DenseNet',
    'resnet152_imgnet': 'ResNet',
    'densenet161_imgnet': 'DenseNet',
    'resnet110_SD_c100': 'ResNet_SD',
    'resnet_wide32_c100': 'Wide_ResNet'
}

ml_data_name_map = {
    'imgnet': 'ImageNet',
    'c10': 'CIFAR-10',
    'c100': 'CIFAR-100'
}
