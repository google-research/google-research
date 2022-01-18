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

"""Configuration of model parameters."""

_is_training = None


# Helper functions for setting train/test flags in the model.
def set_training(x):
  global _is_training
  _is_training = x


def is_training():
  assert _is_training is not None
  return _is_training

USE_SPECTRAL_NORMALIZATION = False
DIM_OF_STYLE_EMBEDDING = 256
REFINEMENT_CHANNEL_SIZE = 32
