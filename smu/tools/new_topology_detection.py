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
"""Script to sample differences in topology detection with different methods."""

import collections

from absl import app

from smu import dataset_pb2

from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.geometry import topology_from_geom


def main(unused_argv):
  db = smu_sqlite.SMUSQLite('20220128_standard_v2.sqlite')

  bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
  bond_lengths.add_from_sparse_dataframe_file(
      '20220128_bond_lengths.csv',
      bond_length_distribution.STANDARD_UNBONDED_RIGHT_TAIL_MASS,
      bond_length_distribution.STANDARD_SIG_DIGITS)
  fake_smiles_id_dict = collections.defaultdict(lambda: -1)

  print('mol_id, count_all, count_smu, count_covalent, count_allen')
  for molecule in db:
    if abs(hash(str(molecule.mol_id))) % 1000 != 1:
      continue

    topology_from_geom.standard_topology_sensing(molecule, bond_lengths,
                                                 fake_smiles_id_dict)

    count_all = len(molecule.bond_topo)
    count_smu = sum(bt.info & dataset_pb2.BondTopology.SOURCE_DDT != 0
                    for bt in molecule.bond_topo)
    count_covalent = sum(bt.info & dataset_pb2.BondTopology.SOURCE_MLCR != 0
                         for bt in molecule.bond_topo)
    count_allen = sum(bt.info & dataset_pb2.BondTopology.SOURCE_CSD != 0
                      for bt in molecule.bond_topo)

    print(
        f'{molecule.mol_id}, {count_all}, {count_smu}, {count_covalent}, {count_allen}'
    )


if __name__ == '__main__':
  app.run(main)
