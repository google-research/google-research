# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Cluttered MNIST Constants.

The cluttered MNIST dataset was generated using the following config:
ORG_SHP = [28, 28]
OUT_SHP = [40, 40]
NUM_DISTORTIONS = 8
dist_size = (5, 5)
n = 1 for the funcion sample_digits

References:
[1] Paper:
https://github.com/daviddao/spatial-transformer-tensorflow
[2] Code:
https://github.com/skaae/recurrent-spatial-transformer-code/blob/master/MNIST_SEQUENCE/create_mnist_sequence.py
"""


NUM_OUTPUTS = 10
IMAGE_EDGE_LENGTH = 40
NUM_FLATTEN_FEATURES = IMAGE_EDGE_LENGTH * IMAGE_EDGE_LENGTH
