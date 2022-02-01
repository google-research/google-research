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

"""generates a Pandas DataFrame of some field values."""

import pandas as pd

from smu import smu_sqlite

db = smu_sqlite.SMUSQLite('20220104_standard.sqlite', 'r')

count = 0
data_dict = {
    'conformer_id': [],
    'energy': [],
    'homo': [],
    'lumo': [],
    'first important frequency': [],
}
# This iteration will go through all conformers in the database.
for conformer in db:

  data_dict['conformer_id'].append(conformer.conformer_id)
  data_dict['energy'].append(
      conformer.properties.single_point_energy_atomic_b5.value)
  data_dict['homo'].append(conformer.properties.homo_pbe0_6_311gd.value)
  data_dict['lumo'].append(conformer.properties.lumo_pbe0_6_311gd.value)
  data_dict['first important frequency'].append(
      conformer.properties.harmonic_frequencies.value[6])

  # This breaks out of the loop after a couple of records just so this
  # examples runs quickly. If you want process the whole dataset,
  # remove this.
  count += 1
  if count == 5:
    break

df = pd.DataFrame(data_dict)
print('This example creates a Pandas dataframe, which is often a useful',
      'starting point for importing data into other python modules.')
print('We are just printing the dataframe here as an example.')
print('See dataframe.py for details')

print(df)
