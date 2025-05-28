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

"""Hyperparameters for Sequential Attention."""

LEARNING_RATE = {
    'mice': 1,
    'mnist': 6e-3,
    'fashion': 0.4,
    'isolet': 2e-5,
    'coil': 2e-3,
    'activity': 1e-5,
}

DECAY_RATE = {
    'mice': 0.63,
    'mnist': 0.37,
    'fashion': 0.84,
    'isolet': 0.95,
    'coil': 0.52,
    'activity': 1,
}

DECAY_STEPS = {
    'mice': 0,
    'mnist': 330,
    'fashion': 0,
    'isolet': 500,
    'coil': 242,
    'activity': 500,
}

BATCH = {
    'mice': 16,
    'mnist': 349,
    'fashion': 391,
    'isolet': 23,
    'coil': 451,
    'activity': 183,
}

EPOCHS = {
    'mice': 337,
    'mnist': 50,
    'fashion': 1000,
    'isolet': 50,
    'coil': 454,
    'activity': 624,
}

EPOCHS_FIT = {
    'mice': 250,
    'mnist': 50,
    'fashion': 54,
    'isolet': 80,
    'coil': 166,
    'activity': 134,
}
