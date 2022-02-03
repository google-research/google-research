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

"""Examples showing important features about multiple bond topologies."""

from smu import smu_sqlite

def print_bond_topologies(conf):
  print('Conformer', conf.conformer_id, 'has', len(conf.bond_topologies),
        'bond topologies')
  for bt in conf.bond_topologies:
    print('    Topology with id', bt.bond_topology_id,
          'and SMILES', bt.smiles)
    if bt.is_starting_topology:
      print('        is_starting_topology: True')

db = smu_sqlite.SMUSQLite('20220128_standard.sqlite')

print('Each Conformer can have multiple bond topologies associated with it')

print()
print('Most Conformers (~96%) have exactly one bond topology like this one')
print_bond_topologies(db.find_by_conformer_id(57429002))

print()
print('Some Conformers have multiple possible bond topologies like this one')
print_bond_topologies(db.find_by_conformer_id(8400001))
print('For ones with multiple topologies one will generally be marked')
print('with is_starting_topology, indicating that this is the topology')
print('that was used during the initial geometry generation')

print()
print('However, the same topology id can be present multiple times for a given conformer')
print('For example, consider good old benzene')
benzene = db.find_by_conformer_id(79488001)
print_bond_topologies(benzene)
print('These are the two kekulized forms of benzene')
print('While the final graphs are isomorphic, for a given pair of carbons, the bond types are switched')
print('The first bond topology has')
print(benzene.bond_topologies[0].bonds[3], end='')
print('and the second bond topology has')
print(benzene.bond_topologies[1].bonds[3], end='')

print()
print('There are also some cases with a mix of same and different ids, like this')
print_bond_topologies(db.find_by_conformer_id(3177001))

print()
print('There are also a handful of special cases that have no topology marked')
print('with is_starting_topology. These have exactly one bond topology and')
print('1 or 2 heavy atoms')
print_bond_topologies(db.find_by_conformer_id(899650001))
print_bond_topologies(db.find_by_conformer_id(899651001))
print_bond_topologies(db.find_by_conformer_id(899652001))
print('Those are all the cases in the standard database')
print('There are a couple more in the complete database')
