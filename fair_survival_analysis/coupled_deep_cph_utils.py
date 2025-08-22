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

"""Utils for coupled Deep CPH module.

"""

import numpy as np

import tensorflow as tf

dtype = tf.float32

FLOAT_64 = False
if FLOAT_64:
  tf.keras.backend.set_floatx('float64')
  dtype = tf.float64


def partial_ll_loss(model, xb, tb, eb, ab, l2=0.001, eps=0.001):

  """Cox Partial Likelihood loss.

  Args:
    model:
      instance of CoupledDeepCPH/VAE class.
    xb:
      a numpy array of input features (Training Data).
    tb:
      a numpy vector of event times (Training Data).
    eb:
      a numpy vector of event indicators (1 if event occured, 0 otherwise)
      (Training Data).
    ab:
      a numpy vector of the protected group membership (Training Data).
    l2:
      l2 penalty on the weights.
    eps:
      small float to resolve ties.

  Returns:
    A tuple consisting of the baseline survival rates for each demographic.

  """

  tb = tb+ eps*np.random.random(len(tb))

  sindex = np.argsort(-tb)

  xb = xb[sindex]
  tb = tb[sindex]
  eb = eb[sindex]
  ab = ab[sindex]

  lrisksp, lrisksn = model(xb)

  lrisksp = lrisksp[ab == 1]
  lrisksn = lrisksn[ab == 0]

  ebp = eb[ab == 1]
  ebn = eb[ab == 0]

  lrisksdenomp = tf.math.cumulative_logsumexp(lrisksp)
  lrisksdenomn = tf.math.cumulative_logsumexp(lrisksn)

  pllsp = lrisksp - lrisksdenomp
  pllsn = lrisksn - lrisksdenomn

  pllp = pllsp[ebp == 1]
  plln = pllsn[ebn == 1]

  penalty = tf.reduce_mean(model.prot.weights[0]**2)
  penalty += tf.reduce_mean(model.nprot.weights[0]**2)

  pll = tf.reduce_mean(pllp) + tf.reduce_mean(plln) + l2*penalty

  return pll
