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
"""Examples showing important features about multiple bond topologies."""

from smu import dataset_pb2
from smu import smu_sqlite


def print_bond_topologies(mol):
  print('Molecule', mol.molecule_id, 'has', len(mol.bond_topologies),
        'bond topologies')
  for bt in mol.bond_topologies:
    print('    Topology with id', bt.bond_topology_id, 'and SMILES', bt.smiles)
    source_string = ''
    if bt.source & dataset_pb2.BondTopology.SOURCE_STARTING:
      source_string += 'STARTING '
    if bt.source & dataset_pb2.BondTopology.SOURCE_ITC:
      source_string += 'ITC '
    if bt.source & dataset_pb2.BondTopology.SOURCE_MLCR:
      source_string += 'MLCR '
    if bt.source & dataset_pb2.BondTopology.SOURCE_CSD:
      source_string += 'CSD '
    print('        Sources: ', source_string)


db = smu_sqlite.SMUSQLite('20220621_standard.sqlite')

print('Each Molecule can have multiple bond topologies associated with it')

print()
print('Most Molecules (~96%) have exactly one bond topology like this one')
print_bond_topologies(db.find_by_molecule_id(57429002))

print()
print('Some Molecules have multiple bond topologies like this one')
print_bond_topologies(db.find_by_molecule_id(8400001))

print()
print('The "source" field gives information about these multiple topologies')
print('"source" is an integer which is a bit field of several values from')
print('dataset_pb2.BondTopology.SourceType')

print()
print('The most important bit to check is SOURCE_STARTING')
print('This bit is set on the topology used during initial geometry generation')
print('You check this bit with code like')
print('bt.source & dataset_pb2.BondTopology.SOURCE_STARTING')

print()
print('The other three bits are related to the three methods we have for ')
print('matching a topology to the geometry of the Molecule.')
print('Please see the manuscript for details on these methods.')
print('The three bits are:')
print('Initial Topology Criteria: dataset_pb2.BondTopology.SOURCE_ITC')
print('Meng Lewis Covalent Radii: dataset_pb2.BondTopology.SOURCE_MLCR')
print('Cambridge Structural Database: dataset_pb2.BondTopology.SOURCE_CSD')

print()
print('One further note, the same topology id can be present multiple times')
print('For example, consider good old benzene')
benzene = db.find_by_molecule_id(79488001)
print_bond_topologies(benzene)
print('These are the two kekulized forms of benzene')
print(
    'While the final graphs are isomorphic, for a given pair of carbons, the bond types are switched'
)
print('The first bond topology has')
print(benzene.bond_topologies[0].bonds[3], end='')
print('and the second bond topology has')
print(benzene.bond_topologies[1].bonds[3], end='')

print()
print(
    'There are also some cases with a mix of same and different ids, like this')
print_bond_topologies(db.find_by_molecule_id(3177001))

# TODO(pfr): add examples using iterate_bond_topologies
