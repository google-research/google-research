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

# pylint: skip-file
from factoring_sqif import number_theory


def test_integer_to_lattice_dimension():
  assert number_theory.integer_to_lattice_dimension(1961) == 3
  assert number_theory.integer_to_lattice_dimension(48567227) == 5
  assert number_theory.integer_to_lattice_dimension(261980999226229) == 8
