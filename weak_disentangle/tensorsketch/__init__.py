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

"""Tensorsketch API.
"""

# pylint: disable=g-bad-import-order
from weak_disentangle.tensorsketch.modules import Module
from weak_disentangle.tensorsketch.modules import ModuleList
from weak_disentangle.tensorsketch.modules import Sequential

from weak_disentangle.tensorsketch.modules import Affine
from weak_disentangle.tensorsketch.modules import Dense
from weak_disentangle.tensorsketch.modules import Conv2d
from weak_disentangle.tensorsketch.modules import ConvTranspose2d

from weak_disentangle.tensorsketch.modules import Flatten
from weak_disentangle.tensorsketch.modules import Reshape

from weak_disentangle.tensorsketch.modules import ReLU
from weak_disentangle.tensorsketch.modules import LeakyReLU
from weak_disentangle.tensorsketch.modules import Sigmoid

from weak_disentangle.tensorsketch.normalization import BatchNorm
from weak_disentangle.tensorsketch.normalization import SpectralNorm
from weak_disentangle.tensorsketch.normalization import WeightNorm
from weak_disentangle.tensorsketch.normalization import RunningNorm

from weak_disentangle.tensorsketch.utils import advanced_function
from weak_disentangle.tensorsketch.utils import reset_tf_function
