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
db = smu_sqlite.SMUSQLite('20220128_complete_v2.sqlite')

#-----------------------------------------------------------------------------
# We'll grab a couple of molecules with different amount of information
# stored.
#-----------------------------------------------------------------------------
PARTIAL_MOLECULE_ID = 35004068
MINIMAL_MOLECULE_ID = 35553043
partial_molecule = db.find_by_molecule_id(PARTIAL_MOLECULE_ID)
minimal_molecule = db.find_by_molecule_id(MINIMAL_MOLECULE_ID)

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

print(
    'If you ask for the harmonic_frequencies for both you get sensible values')
print(PARTIAL_MOLECULE_ID, partial_molecule.properties.harmonic_frequencies)
print(MINIMAL_MOLECULE_ID, minimal_molecule.properties.harmonic_frequencies)

print()
print('But if you ask for zpe_unscaled, the second gives a 0')
print(PARTIAL_MOLECULE_ID, partial_molecule.properties.zpe_unscaled.value)
print(MINIMAL_MOLECULE_ID, minimal_molecule.properties.zpe_unscaled.value)

print()
print('And if you ask for enthalpy_of_formation_298k_atomic_b6, both give 0')
print(PARTIAL_MOLECULE_ID,
      partial_molecule.properties.enthalpy_of_formation_298k_atomic_b6.value)
print(MINIMAL_MOLECULE_ID,
      minimal_molecule.properties.enthalpy_of_formation_298k_atomic_b6.value)

print()
print('These are cases of missing values.')
print(
    'If you request a value which is actually missing, you will silently get a default value (0.0 for floats)'
)
print(
    'Therefore, in addition to checking the status field, we recommend you also'
)
print('check whether a Molecule has a value with the HasField method')
print('Calling HasField for harmonic_frequencies:')
print(PARTIAL_MOLECULE_ID,
      partial_molecule.properties.HasField('harmonic_frequencies'))
print(MINIMAL_MOLECULE_ID,
      minimal_molecule.properties.HasField('harmonic_frequencies'))
print('Calling HasField for zpe_unscaled:')
print(PARTIAL_MOLECULE_ID, partial_molecule.properties.HasField('zpe_unscaled'))
print(MINIMAL_MOLECULE_ID, minimal_molecule.properties.HasField('zpe_unscaled'))
print('Calling HasField for enthalpy_of_formation_298k_atomic_b6:')
print(
    PARTIAL_MOLECULE_ID,
    partial_molecule.properties.HasField(
        'enthalpy_of_formation_298k_atomic_b6'))
print(
    MINIMAL_MOLECULE_ID,
    minimal_molecule.properties.HasField(
        'enthalpy_of_formation_298k_atomic_b6'))

print()
print('The one field that is different is normal_modes')
print(
    'Since normal_modes is a list of composite values, missing just means the list is length 0'
)
print('You cannot call HasField on normal_modes')
print('The length of normal_modes in our two molecules are:')
print(PARTIAL_MOLECULE_ID, len(partial_molecule.properties.normal_modes))
print(MINIMAL_MOLECULE_ID, len(minimal_molecule.properties.normal_modes))

print()
print('In summary, when processing the complete database:')
print('1. Always check the status field and warning flags.')
print('2. Always check HasField before accessing properties.')
