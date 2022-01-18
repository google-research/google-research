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

# python3
"""Modules API.
"""

# pylint: disable=g-bad-import-order
from weak_disentangle.tensorsketch.modules.base import Module
from weak_disentangle.tensorsketch.modules.base import ModuleList
from weak_disentangle.tensorsketch.modules.base import Sequential

from weak_disentangle.tensorsketch.modules.shape import Flatten
from weak_disentangle.tensorsketch.modules.shape import Reshape

from weak_disentangle.tensorsketch.modules.affine import Affine
from weak_disentangle.tensorsketch.modules.affine import Dense
from weak_disentangle.tensorsketch.modules.affine import Conv2d
from weak_disentangle.tensorsketch.modules.affine import ConvTranspose2d

from weak_disentangle.tensorsketch.modules.activation import ReLU
from weak_disentangle.tensorsketch.modules.activation import LeakyReLU
from weak_disentangle.tensorsketch.modules.activation import Sigmoid
