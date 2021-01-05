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

"""Value-at-Risk (VaR) and Conditional Value-At-Risk (CVaR)."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from typing import List, MutableMapping, Optional, Tuple, Union

TensorOrFloat = Union[tf.Tensor, float]


def var_and_cvar(portfolio_returns,
                 var_levels,
                 key_placeholder,
                 feed_dict,
                 num_batches = 1,
                 tf_session = None
                ):
  """Compute VaR and CVaR given a portfoolio return simulation graph.

  Given an initial position and value for a financial portfolio, Value-at-Risk
  (VaR) at level alpha is the opposite of the level 1 - alpha percentile of
  returns. In other words, it is the smallest loss that occurs with a
  probability less or equal to alpha. The Condition-VaR is the expected value
  of the loss conditioned on the fact that the loss is greater than the VaR.

  Args:
    portfolio_returns: a [num_samples] tensor of scalars, the simulated
      distribution of scalars for a given portfolio.
    var_levels: a [num_vars] list of scalars in [0, 1], the VaR levels to
      compute, typically [0.95, 0.98, 0.999].
    key_placeholder: the placeholder entailing the key for the sub-stream of
      pseudo or quasi random numbers.
    feed_dict: an optional dictionary of values to put in placeholders for the
      portfolio_returns graph to use.
    num_batches: the number of portfolio simulations to run. The estimated
      distribution will have num_samples * num_batches samples.
    tf_session: (optional) tensorflow session.

  Returns:
    a [num_vars] array of var estimates, a [num_vars] array of cvar estimates.
  """

  close_session = False
  if tf_session is None:
    tf_session = tf.Session()
    close_session = True

  return_distribution = []
  for batch_idx in range(num_batches):
    feed_dict.update({key_placeholder: batch_idx})
    return_distribution.extend(
        tf_session.run(portfolio_returns, feed_dict=feed_dict))
  return_distribution = np.asarray(return_distribution)

  percentiles = [100.0 * (1 - p) for p in var_levels]
  var_ests = [np.percentile(return_distribution, p) for p in percentiles]
  cvar_ests = [
      np.mean(return_distribution[return_distribution <= var])
      for var in var_ests
  ]

  if close_session:
    tf_session.close()

  # Flip the sign to abide with the convention that a positive VaR is a loss.
  return -np.asarray(var_ests), -np.asarray(cvar_ests)
