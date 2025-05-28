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

"""Common utility functions."""

import collections

import numpy as np


PRETTY_NAMES = {
    'bisimulation': r'$d^{\sim}_{\pi}$',
    'mico': r'$U^{\pi}$',
    'reduced_mico': r'$d_{ksme}$',
}


MDPStats = collections.namedtuple(
    'MDPStats',
    ['time', 'num_iterations', 'avg_gap', 'min_gap', 'max_gap'])


VAL_DIFF_STR = r'|V^{\pi}(x) - V^{\pi}(y)|$'
Y_LABELS = {
    'time': 'Time to convergence (log scale)',
    'num_iterations': 'Number of iterations to convergence',
    'avg_gap': r'Avg. $d(x, y) - ' + VAL_DIFF_STR,
    'min_gap': r'Min $d(x, y) - ' + VAL_DIFF_STR,
    'max_gap': r'Max $d(x, y) - ' + VAL_DIFF_STR,
}


def build_auxiliary_mdp(p, r):
  """This method constructs an auxiliary MDP.

  It uses an independent coupling of the transition probabilities, and the
  differences in reward.

  For example, take the following transition matrix:
     [ a b ]
     [ c d ]
  This method will construct a new transition matrix as follows:
     [ aa ab ba bb ]
     [ ac ad bc bd ]
     [ ca cb da db ]
     [ cc cd dc dd ]

  Args:
    p: numpy array, transition matrix.
    r: numpy array, rewards.

  Returns:
    Auxiliary p and r matrices.
  """
  num_states = p.shape[0]
  num_aux_states = num_states**2
  aux_p = np.einsum('xu,yv->xyuv', p, p)
  aux_p = np.reshape(aux_p, (num_aux_states, num_aux_states))
  tiled_r = np.tile(r, [num_states, 1])
  aux_r = np.reshape(abs(tiled_r - np.transpose(tiled_r)), (num_states**2,))
  return aux_p, aux_r


def compute_value(p, r, gamma):
  """This method computes the value function."""
  num_states = p.shape[0]
  discounted_p = p * gamma
  return np.matmul(np.linalg.inv(np.eye(num_states) - discounted_p), r)
