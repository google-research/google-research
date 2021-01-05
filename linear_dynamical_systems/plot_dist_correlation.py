# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Script to plot correlation between distances."""

import lds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='whitegrid')

num_pairs = 100
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
hidden_state_dim = 2
lds_pairs = [(lds.generate_linear_dynamical_system(hidden_state_dim),
              lds.generate_linear_dynamical_system(hidden_state_dim))
             for i in xrange(num_pairs)]
lds_distances = [
    lds.eig_dist(system1, system2) for (system1, system2) in lds_pairs
]
expected_ar_distances = [
    np.linalg.norm(system1.get_expected_arparams() -
                   system2.get_expected_arparams())
    for (system1, system2) in lds_pairs
]
print(np.corrcoef(lds_distances, expected_ar_distances)[0, 1])
ax = sns.regplot(x=lds_distances, y=expected_ar_distances)
ax.set(
    xlabel='l-2 distance b/w eigenvalues',
    ylabel='l-2 distance b/w '
    'corresponding AR params',
    title='Hidden dim = 2')
plt.subplot(1, 2, 2)
hidden_state_dim = 3
lds_pairs = [(lds.generate_linear_dynamical_system(hidden_state_dim),
              lds.generate_linear_dynamical_system(hidden_state_dim))
             for i in xrange(num_pairs)]
lds_distances = [
    lds.eig_dist(system1, system2) for (system1, system2) in lds_pairs
]
expected_ar_distances = [
    np.linalg.norm(system1.get_expected_arparams() -
                   system2.get_expected_arparams())
    for (system1, system2) in lds_pairs
]
print(np.corrcoef(lds_distances, expected_ar_distances)[0, 1])
ax = sns.regplot(x=lds_distances, y=expected_ar_distances)
ax.set(xlabel='l-2 distance b/w eigenvalues', ylabel='', title='Hidden dim = 3')
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()
