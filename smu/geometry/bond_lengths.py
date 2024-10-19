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
"""Extract bond lengths from SMU molecules."""

import apache_beam as beam

import utilities

from smu import dataset_pb2
from smu.parser import smu_utils_lib

MAX_DIST = 2.0


class GetBondLengthDistribution(beam.DoFn):
  """Generates a bond length distribution."""

  def process(self, molecule):
    bt = molecule.bond_topo[0]
    geom = molecule.opt_geo

    bonded = utilities.bonded(bt)

    natoms = len(bt.atom)

    if molecule.prop.calc.fate != dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW:
      return

    for a1 in range(0, natoms):
      atomic_number1 = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[bt.atom[a1]]
      for a2 in range(a1 + 1, natoms):
        atomic_number2 = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[bt.atom[a2]]
        # Do not process H-H pairs
        if atomic_number1 == 1 and atomic_number2 == 1:
          continue

        d = utilities.distance_between_atoms(geom, a1, a2)
        if d > MAX_DIST:
          continue

        discretized = int(d * utilities.DISTANCE_BINS)
        yield (min(atomic_number1, atomic_number2), int(bonded[a1, a2]),
               max(atomic_number1, atomic_number2), discretized), 1
