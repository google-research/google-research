# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Default constants used in this library."""

# Exponential Coulomb interaction.
#
# v(x) = amplitude * exp(-abs(x) * kappa)
#
# 1d interaction described in
# One-dimensional mimicking of electronic structure: The case for exponentials.
# Physical Review B 91.23 (2015): 235141.
# https://arxiv.org/pdf/1504.05620.pdf

EXPONENTIAL_COULOMB_AMPLITUDE = 1.071295
EXPONENTIAL_COULOMB_KAPPA = 1 / 2.385345

# Soft Coulomb interaction.
SOFT_COULOMB_SOFTEN_FACTOR = 1.

# Chemical accuracy 0.0016 Hartree = 1 kcal/mol
CHEMICAL_ACCURACY = 0.0016
