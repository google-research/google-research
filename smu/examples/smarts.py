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

"""Shows how to use a SMARTS pattern to choose bond topologies."""

from smu import smu_sqlite

db = smu_sqlite.SMUSQLite('20220128_standard_v2.sqlite')

print('SMARTS patterns are convenient ways to select a subset of bond topologies')
print('You can read about SMARTS here: https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html')

print()
smarts = 'C1CCCCC1'
print('The first step is to find the set of bond topology ids that match the SMARTS')
print('We will use a SMARTS query that finds 6 membered, singly bonded carbon rings')
print(smarts)

print()
print('We use find_bond_topology_id_by_smarts to get the bt_ids')
print('This typically takes 20-40 seconds')
bt_ids = list(db.find_bond_topology_id_by_smarts(smarts))
print('In this case we find', len(bt_ids), 'matching bond topologies')

print()
print('You can then use find_by_bond_topology_id_list (just like indices.py) to get the conformers')
conformers = list(db.find_by_bond_topology_id_list(bt_ids))
print('In this case we find', len(conformers), 'matching conformers')
print('As you can see, not all bond topology ids in the standard database will find a matching conformer')
print('You can always find descriptions of all bond topologies in bond_topology.csv')
