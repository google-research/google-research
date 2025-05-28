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
"""Examples showing important features about multiple bond topologies."""

from smu import dataset_pb2
from smu import smu_sqlite
from smu.parser import smu_utils_lib


def print_bond_topo(mol):
  """Print bond topology.

  Args:
    mol:
  """
  print('Molecule', mol.mol_id, 'has', len(mol.bond_topo), 'bond topologies')
  for topo in mol.bond_topo:
    print('    Topology with id', topo.topo_id, 'and SMILES', topo.smiles)
    source_string = ''
    if topo.info & dataset_pb2.BondTopology.SOURCE_STARTING:
      source_string += 'STARTING '
    if topo.info & dataset_pb2.BondTopology.SOURCE_DDT:
      source_string += 'DDT '
    if topo.info & dataset_pb2.BondTopology.SOURCE_MLCR:
      source_string += 'MLCR '
    if topo.info & dataset_pb2.BondTopology.SOURCE_CSD:
      source_string += 'CSD '
    print('        Sources: ', source_string)


db = smu_sqlite.SMUSQLite('20220621_standard_v4.sqlite')

print('Each Molecule can have multiple bond topologies associated with it')

print()
print('Most Molecules (~96%) have exactly one bond topology like this one')
print_bond_topo(db.find_by_mol_id(57429002))

print()
print('Some Molecules have multiple bond topologies like this one')
print_bond_topo(db.find_by_mol_id(8400001))

print()
print('The "source" field gives information about these multiple topologies')
print('"source" is an integer which is a bit field of several values from')
print('dataset_pb2.BondTopology.SourceType')

print()
print('The most important bit to check is SOURCE_STARTING')
print('This bit is set on the topology used during initial geometry generation')
print('You check this bit with code like')
print('bt.info & dataset_pb2.BondTopology.SOURCE_STARTING')

print()
print('The other three bits are related to the three methods we have for ')
print('matching a topology to the geometry of the Molecule.')
print('Please see the manuscript for details on these methods.')
print('The three bits are:')
print('Initial Topology Criteria: dataset_pb2.BondTopology.SOURCE_DDT')
print('Meng Lewis Covalent Radii: dataset_pb2.BondTopology.SOURCE_MLCR')
print('Cambridge Structural Database: dataset_pb2.BondTopology.SOURCE_CSD')

print()
print('One further note, the same topology id can be present multiple times')
print('For example, consider good old benzene')
benzene = db.find_by_mol_id(79488001)
print_bond_topo(benzene)
print('These are the two kekulized forms of benzene')
print(
    'While the final graphs are isomorphic, for a given pair of carbon, bond types are switched'
)
print('The first bond topology has')
print(benzene.bond_topo[0].bond[3], end='')
print('and the second bond topology has')
print(benzene.bond_topo[1].bond[3], end='')

print()
print(
    'There are also some cases with a mix of same and different ids, like this')
print_bond_topo(db.find_by_mol_id(3177001))

print()
print(
    'The easiest and most reliable way to select the desired topologies is the function'
)
print(
    'iterate_bond_topologies. The "which" parameter controls which bond topologies are returned'
)

molecule = db.find_by_mol_id(8400001)
print_bond_topo(molecule)

print()
print('Passing ALL as the which parameter gives')
for bt_idx, bt in smu_utils_lib.iterate_bond_topologies(
    molecule, smu_utils_lib.WhichTopologies.ALL):
  print('    Got bond topology with position', bt_idx, 'and id', bt.topo_id)

print()
print('Passing STARTING as the which parameter gives')
for bt_idx, bt in smu_utils_lib.iterate_bond_topologies(
    molecule, smu_utils_lib.WhichTopologies.STARTING):
  print('    Got bond topology with position', bt_idx, 'and id', bt.topo_id)
print(
    'Note that STARTING deals correctly with special cases present in the complete database'
)
print('See special_cases.py for some details.')
print('This is one of the reasons this function is the recommended method')

print()
print(
    'The last 3 which values select topologies based on which methods produced them'
)
print('which of DDT gives')
for bt_idx, bt in smu_utils_lib.iterate_bond_topologies(
    molecule, smu_utils_lib.WhichTopologies.DDT):
  print('    Got bond topology with position', bt_idx, 'and id', bt.topo_id)
print('which of MLCR gives')
for bt_idx, bt in smu_utils_lib.iterate_bond_topologies(
    molecule, smu_utils_lib.WhichTopologies.MLCR):
  print('    Got bond topology with position', bt_idx, 'and id', bt.topo_id)
print('which of CSD gives')
for bt_idx, bt in smu_utils_lib.iterate_bond_topologies(
    molecule, smu_utils_lib.WhichTopologies.CSD):
  print('    Got bond topology with position', bt_idx, 'and id', bt.topo_id)
print('    It is correct that nothing was printed here!')
print('    This topology has no topology matching the CSD criteria')
