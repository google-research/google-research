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
"""Use the various indices in the database for fast lookups."""

from smu import smu_sqlite
from smu.parser import smu_utils_lib

db = smu_sqlite.SMUSQLite('20220621_standard_v4.sqlite')

print('There are several ways to efficiently get specific sets of molecules')

print()
print('First is a lookup by molecule id')
mid_molecule = db.find_by_mol_id(57001)
print('Looking up 57001 returns molecule with id', mid_molecule.mol_id,
      'and bond topology with SMILES', mid_molecule.bond_topo[0].smiles)

try:
  db.find_by_mol_id(999999)
except KeyError:
  print('Looking up a molecule id not in the DB raises a KeyError')

print()
print('Looking up by bond topology id will return zero or more molecules')
bt_molecules = list(
    db.find_by_topo_id_list([7984],
                            which_topologies=smu_utils_lib.WhichTopologies.ALL))
print('Querying for bond topology id 7984 returned', len(bt_molecules),
      'molecules')

print('Note that the molecules returned may have multiple bond topologies,'
      'and may or may not have the requested bond topology first')
for mol in bt_molecules:
  print('    Result with mol_id', mol.mol_id)
  for bt in mol.bond_topo:
    print('        has bond topology with id', bt.topo_id, 'and SMILES',
          bt.smiles)

print()
print(
    'Finding by SMILES is essentially equivalent to finding by bond topology id'
)
smiles_molecules = list(
    db.find_by_smiles_list(['O=NONNNO'],
                           which_topologies=smu_utils_lib.WhichTopologies.ALL))
print('With query O=NONNNO', 'we found', len(smiles_molecules), 'results')

print('Note that the SMILES are canonicalized internally, you do not need to')
print(
    'So the equivalent SMILES query ONNNON=O returns the same',
    len(
        list(
            db.find_by_smiles_list(
                ['ONNNON=O'],
                which_topologies=smu_utils_lib.WhichTopologies.ALL))),
    'results')

print()
print('You can also find all the molecules with a given stoichiometry')
stoich_molecules = list(db.find_by_stoichiometry('cn2o3'))
print('For example, "cn2o3" finds', len(stoich_molecules), 'results')
print('The first couple of molecule ids are:',
      [c.mol_id for c in stoich_molecules[0:5]])

print()
print('You may note that there is a "find_by_expanded_stoichiometry" method',
      'in smu_sqlite')
print('This is primarily intended to support "topology queries"')
print('See topology_query.py')
