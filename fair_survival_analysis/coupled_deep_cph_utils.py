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

"""Utils for coupled Deep CPH module.

"""

import numpy as np

from sksurv.linear_model.coxph import BreslowEstimator

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


def train_breslow(model, xt, tt, et, at, xv, tv, ev, av, groups):

  """Trains a Breslow Estimator from a learnt survival model.

  Args:
    model:
      instance of CoupledDeepCPH/VAE class.
    xt:
      a numpy array of input features (Training Data).
    tt:
      a numpy vector of event times (Training Data).
    et:
      a numpy vector of event indicators (1 if event occured, 0 otherwise)
      (Training Data).
    at:
      a numpy vector of the protected group membership (Training Data).
    xv:
      a numpy array of input features (Validation Data).
    tv:
      a numpy vector of event times (Validation Data).
    ev:
      a numpy vector of event indicators (1 if event occured, 0 otherwise)
      (Validation Data).
    av:
      a numpy vector of the protected group membership (Validation Data).
    groups:
      List of the demographics to adjust for.

  Returns:
    A tuple consisting of the baseline survival rates for each demographic.

  """

  prot, nprot = groups[0], groups[1]

  xall = np.vstack([xt, xv])
  tall = np.concatenate([tt, tv])
  eall = np.concatenate([et, ev])
  aall = np.concatenate([at, av])

  prot_risks, nprot_risks = model(xall)
  prot_risks, nprot_risks = prot_risks.numpy()[:, 0], nprot_risks.numpy()[:, 0]

  blsurvivalp = BreslowEstimator().fit(prot_risks[aall == prot],
                                       eall[aall == prot], tall[aall == prot])
  blsurvivaln = BreslowEstimator().fit(prot_risks[aall == nprot],
                                       eall[aall == nprot], tall[aall == nprot])

  blsurvivalp = blsurvivalp.baseline_survival_
  blsurvivaln = blsurvivaln.baseline_survival_

  return blsurvivalp, blsurvivaln


def predict_survival(trained_model, x, t, a, groups):

  """Returns the survival probability given a model and the breslow's estimator.

  Args:
    trained_model:
      tuple consisting of an instance of a "trained" CoupledDeepCPHVAE class
      and the corresponding breslow's estimator.
    x:
      a numpy array of input features (Test Data).
    t:
      a numpy vector of event times (Test Data).
    a:
      a numpy vector of the protected group membership (Test Data).
    groups:
      List of the demographics to adjust for.

  Returns:
    a numpy vector of the survival probabilities.

  """

  prot, nprot = groups[0], groups[1]

  model, blsurvival = trained_model

  blsurvivalp, blsurvivaln = blsurvival

  survivals = -np.ones_like(a)

  lrisksp, lrisksn = model(x)

  risksp, risksn = np.exp(lrisksp), np.exp(lrisksn)

  s0p = np.log(float(blsurvivalp.T(t)))
  s0n = np.log(float(blsurvivaln.T(t)))

  stp = np.exp(risksp * s0p)[:, 0]
  stn = np.exp(risksn * s0n)[:, 0]

  survivals[a == prot] = stp[a == prot]
  survivals[a == nprot] = stn[a == nprot]

  return survivals
