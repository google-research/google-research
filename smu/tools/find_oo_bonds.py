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
"""Script to look for OO bonds of a particular type."""

import itertools

from absl import app
from absl import logging
import numpy as np

from smu import dataset_pb2
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.geometry import topology_from_geom
from smu.geometry import topology_molecule
from smu.parser import smu_utils_lib


def main(argv):
  del argv  # Unused.

  db = smu_sqlite.SMUSQLite('20220621_complete.sqlite')

  count_processed = 0
  count_single_o = 0
  count_multiple_o = 0
  count_right_len = 0
  count_matching = 0

  bl = bond_length_distribution.make_csd_dists()
  target_min = bl[(dataset_pb2.BondTopology.ATOM_O,
                   dataset_pb2.BondTopology.ATOM_O
                  )][dataset_pb2.BondTopology.BOND_SINGLE].max()
  target_max = bl[(dataset_pb2.BondTopology.ATOM_O,
                   dataset_pb2.BondTopology.ATOM_O
                  )][dataset_pb2.BondTopology.BOND_UNDEFINED].min()
  bl[(dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_O)].add(
      dataset_pb2.BondTopology.BOND_UNDEFINED,
      bond_length_distribution.FixedWindow(
          target_min, target_min + 0.1,
          bond_length_distribution.STANDARD_UNBONDED_RIGHT_TAIL_MASS))

  matching_parameters = topology_molecule.MatchingParameters()

  for mol in db:
    count_processed += 1

    if count_processed % 25000 == 0:
      logging.info('Read %d, tried %d, found %d', count_processed,
                   count_right_len, count_matching)

    o_indices = [
        i for i in range(len(mol.bond_topo[0].atom))
        if mol.bond_topo[0].atom[i] == dataset_pb2.BondTopology.ATOM_O
    ]
    if not o_indices:
      continue
    elif len(o_indices) == 1:
      count_single_o += 1
      continue

    count_multiple_o += 1

    if (mol.prop.calc.status < 0 or mol.prop.calc.status >= 512 or
        not mol.HasField('opt_geo')):
      continue

    for aidx0, aidx1 in itertools.combinations(o_indices, 2):
      pos0 = mol.opt_geo.atompos[aidx0]
      pos1 = mol.opt_geo.atompos[aidx1]

      bond_len = (
          smu_utils_lib.BOHR_TO_ANGSTROMS * np.linalg.norm(
              np.array([pos0.x, pos0.y, pos0.z]) -
              np.array([pos1.x, pos1.y, pos1.z])))
      if bond_len > target_min and bond_len < target_max:
        count_right_len += 1

        matches = topology_from_geom.bond_topologies_from_geom(
            mol, bond_lengths=bl, matching_parameters=matching_parameters)

        if matches.bond_topology:
          count_matching += 1

        print('%s, %d, %s', mol.mol_id, bond_len,
              True if matches.bond_topology else False)

  logging.info(
      'Read %d, found %d single, %d multiple, %d matching length, %d with valid topo',
      count_processed, count_single_o, count_multiple_o, count_right_len,
      count_matching)


if __name__ == '__main__':
  app.run(main)
