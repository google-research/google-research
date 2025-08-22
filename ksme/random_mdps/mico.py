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

"""Implementation of the MICo distance."""

from absl import logging
import numpy as np
from ksme.random_mdps import metric
from ksme.random_mdps import utils


class MICo(metric.Metric):
  """Implementation of the MICo distance.

  See Castro et al., 2021: "MICo: Improved representations via sampling-based
  state similarity for Markov decision processes"
  https://arxiv.org/abs/2106.08229
  """

  def __init__(self, name, label, env, base_dir, normalize=False,
               reduced=False, run_number=0):
    self._reduced = reduced
    super().__init__(
        name, label, env, base_dir, normalize=normalize,
        run_number=run_number)

  def _compute(self, tolerance=0.0, verbose=False):
    """Compute exact/online mico distance up to the specified tolerance.

    Args:
      tolerance: float, unused here as we compute the metric exactly.
      verbose: bool, whether to print verbose messages.
    """
    del tolerance
    # We compute this metric by solving the value function of an auxiliary MDP.
    # See original paper for details.
    aux_p, aux_r = utils.build_auxiliary_mdp(self.env.policy_transition_probs,
                                             self.env.policy_rewards)
    if verbose:
      logging.info('Computing MICo using an auxiliary MDP.')
    aux_v = utils.compute_value(aux_p, aux_r, self.env.gamma)
    curr_metric = np.reshape(aux_v, [self.num_states, self.num_states])
    if self._reduced:
      if verbose:
        logging.info('Compute reduced version of MICo')
      # If MICo is $U$, reduced MICo is:
      # \Pi U(x, y) = U(x, y) - 1/2 U(x, x) - 1/2 U(y, y)
      # See Section 5 in paper for details.
      diag = np.diag(curr_metric)
      curr_metric -= (diag[None, :] + diag[:, None]) / 2.0
    self.metric = curr_metric
