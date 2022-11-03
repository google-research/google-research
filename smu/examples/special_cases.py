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
"""The complete database has a special cases to be aware of."""

from smu import smu_sqlite

#-----------------------------------------------------------------------------
# Note that we are loading the *complete* database
#-----------------------------------------------------------------------------
db = smu_sqlite.SMUSQLite('20220128_complete_v2.sqlite')

print(
    'When processing the complete database, there are a few special kinds of molecules to look out for'
)

DUPLICATE_MOLECULE_SAME_ID = 253949008
DUPLICATE_MOLECULE_DIFF_ID = 95603043

print()
print('The first are duplicate records')
molecule_dup_same = db.find_by_molecule_id(DUPLICATE_MOLECULE_SAME_ID)
print('Consider', molecule_dup_same.molecule_id)
print('It has status', molecule_dup_same.properties.errors.status,
      'and duplicated_by', molecule_dup_same.duplicated_by)
print('This means this molecule was considered a duplciated of another')
print('That other molecule proceeded to full calculation',
      'and this molecule did not')
print('A molecule should have status=-1 if and only if',
      'duplicated_by is not 0')
print('Note that this molecule started with bond topology',
      molecule_dup_same.molecule_id // 1000,
      'which is the same as the duplicate',
      molecule_dup_same.duplicated_by // 1000)
print(
    'Therefore, the initial_geometry from', molecule_dup_same.molecule_id,
    'was copied to the list of initial_geometries for',
    molecule_dup_same.duplicated_by, 'which has',
    len(
        db.find_by_molecule_id(
            molecule_dup_same.duplicated_by).initial_geometries),
    'initial_geometries')

molecule_dup_diff = db.find_by_molecule_id(DUPLICATE_MOLECULE_DIFF_ID)
print('Compare this to molecule', molecule_dup_diff.molecule_id)
print('It is duplicated to', molecule_dup_diff.duplicated_by,
      'which has bond topology', molecule_dup_diff.duplicated_by // 1000,
      'which is different than the bond topology for this molecule',
      molecule_dup_diff.molecule_id // 1000)
print(
    'Therefore, the initial geometry from', molecule_dup_diff.molecule_id,
    'is NOT copied to', molecule_dup_diff.duplicated_by, 'which has only',
    len(
        db.find_by_molecule_id(
            molecule_dup_diff.duplicated_by).initial_geometries),
    'inital_geometries')
print('We do this because it is not obvious what the atom matching across',
      'the duplicates should be in this case')
print('If you have a good idea how to do that, you will need these records')

print()
print('The other important cases to watch for are status >= 512')
molecule_stage1 = db.find_by_molecule_id(146002)
print('Consider molecule', molecule_stage1.molecule_id)
print('The status is', molecule_stage1.properties.errors.status,
      'and the full record is:')
print(molecule_stage1)
print('You can see that there is very little information there')
print('Anything with status variable >=512 has nothing other than an',
      'attempted geometry optimization, which failed for some reason')
print('There may not even be an optimized_geometry stored')
print('Unless you are interested in the geometry optimization process,',
      'you probably want to ignore these records')
