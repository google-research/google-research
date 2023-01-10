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
"""Shows how to check for missing fields."""

from smu import smu_sqlite

#-----------------------------------------------------------------------------
# Note that we are loading the *complete* database
#-----------------------------------------------------------------------------
db = smu_sqlite.SMUSQLite('20220621_complete_v4.sqlite')

#-----------------------------------------------------------------------------
# We'll grab a couple of molecules with different amount of information
# stored.
#-----------------------------------------------------------------------------
PARTIAL_MOLECULE_ID = 35004068
MINIMAL_MOLECULE_ID = 35553043
partial_molecule = db.find_by_mol_id(PARTIAL_MOLECULE_ID)
minimal_molecule = db.find_by_mol_id(MINIMAL_MOLECULE_ID)

print('When you process the *complete* database, you have to be careful to',
      'check what data is available')

print('We will examine molecules', PARTIAL_MOLECULE_ID, 'and',
      MINIMAL_MOLECULE_ID)

print(
    'In general, you need to consider the status field and various warning flags'
)
print(
    'in order to understand what fields are available and what you should trust'
)

print('If you ask for the vib_freq for both you get sensible values')
print(PARTIAL_MOLECULE_ID, partial_molecule.prop.vib_freq)
print(MINIMAL_MOLECULE_ID, minimal_molecule.prop.vib_freq)

print()
print('But if you ask for vib_zpe, the second gives a 0')
print(PARTIAL_MOLECULE_ID, partial_molecule.prop.vib_zpe.val)
print(MINIMAL_MOLECULE_ID, minimal_molecule.prop.vib_zpe.val)

print()
print('And if you ask for at2_std_b6_hf298, both give 0')
print(PARTIAL_MOLECULE_ID, partial_molecule.prop.at2_std_b6_hf298.val)
print(MINIMAL_MOLECULE_ID, minimal_molecule.prop.at2_std_b6_hf298.val)

print()
print('These are cases of missing values.')
print(
    'If you request a value which is actually missing, you will silently get a default value '
    '(0.0 for floats)')
print(
    'Therefore, in addition to checking the status field, we recommend you also'
)
print('check whether a Molecule has a value with the HasField method')
print('Calling HasField for vib_freq:')
print(PARTIAL_MOLECULE_ID, partial_molecule.prop.HasField('vib_freq'))
print(MINIMAL_MOLECULE_ID, minimal_molecule.prop.HasField('vib_freq'))
print('Calling HasField for vib_zpe:')
print(PARTIAL_MOLECULE_ID, partial_molecule.prop.HasField('vib_zpe'))
print(MINIMAL_MOLECULE_ID, minimal_molecule.prop.HasField('vib_zpe'))
print('Calling HasField for at2_std_b6_hf298:')
print(PARTIAL_MOLECULE_ID, partial_molecule.prop.HasField('at2_std_b6_hf298'))
print(MINIMAL_MOLECULE_ID, minimal_molecule.prop.HasField('at2_std_b6_hf298'))

print()
print('The one field that is different is vib_mode')
print(
    'Since vib_mode is a list of composite values, missing just means the list is length 0'
)
print('You cannot call HasField on vib_mode')
print('The length of vib_mode in our two molecules are:')
print(PARTIAL_MOLECULE_ID, len(partial_molecule.prop.vib_mode))
print(MINIMAL_MOLECULE_ID, len(minimal_molecule.prop.vib_mode))

print()
print('In summary, when processing the complete database:')
print('1. Always check the status field and warning flags.')
print('2. Always check HasField before accessing properties.')
