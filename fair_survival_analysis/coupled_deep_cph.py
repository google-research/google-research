# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Definition of the proposed Coupled Deep Cox.

This module has the tensorflow definitions of the proposed Coupled Deep Cox
model and utility functions to train and evaluate the model.

The module depends on tensorflow 2.

Not designed to be called directly, would be called when running a function from
fair_survival_analysis.fair_survival_analysis

"""

from coupled_deep_cph_utils import partial_ll_loss
from coupled_deep_cph_utils import train_breslow

import lifelines
import numpy as np

from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

tf.keras.backend.set_floatx('float64')


class CoupledDeepCPH(Model):

  """Tensorflow model definition of the Coupled Deep CPH Survival Model.

  The Coupled Deep CPH model involves learning shared representations for
  the each demographic in the dataset.The representation then interacts with
  multiple output heads to determine the log-partial hazard for an individual
  in each group.

  """

  def __init__(self, hidden):
    super(CoupledDeepCPH, self).__init__()

    self.rep1 = Dense(hidden, activation='relu', use_bias=False)
    self.rep2 = Dense(hidden, activation='sigmoid', use_bias=False)

    self.prot = Dense(1, use_bias=False, kernel_initializer='zeros')
    self.nprot = Dense(1, use_bias=False, kernel_initializer='zeros')

  def call(self, x):

    x = self.rep2(self.rep1(x))
    return self.prot(x), self.nprot(x)


class CPH(Model):

  """Tensorflow model definition for a standard CPH model.

  The CPH model involves learning separate CPH model for each demographic in the
  dataset.

  """

  def __init__(self):
    super(CPH, self).__init__()

    self.prot = Dense(1, use_bias=False, kernel_initializer='zeros')
    self.nprot = Dense(1, use_bias=False, kernel_initializer='zeros')

  def call(self, x):
    return self.prot(x), self.nprot(x)


class DeepCPH(Model):

  """Tensorflow model definition of the Deep CPH Survival Model.

  Involves learning a separate Deep Surv/Faraggi-Simon network for each
  demographic.

  """

  def __init__(self, hidden):
    super(DeepCPH, self).__init__()

    self.rep1 = Dense(100, use_bias=False, activation='relu')
    self.rep2 = Dense(100, use_bias=False, activation='relu')

    self.prot = Dense(1, use_bias=False, kernel_initializer='zeros')
    self.nprot = Dense(1, use_bias=False, kernel_initializer='zeros')

  def call(self, x):
    return self.prot(self.rep1(x)), self.nprot(self.rep2(x))


def train_step(model, x, t, e, a, optimizer, bs=256, lambd=1.0, seed=0):

  """Optimizes the model for one epoch.

  Args:
    model:
      instance of CoupledDeepCPH class.
    x:
      a numpy array of input features (Training Data).
    t:
      a numpy vector of event times (Training Data).
    e:
      a numpy vector of event indicators (1 if event occured, 0 otherwise)
      (Training Data).
    a:
      a numpy vector of the protected group membership (Training Data).
    optimizer:
      instance of tf.keras.optimizers (default is Adam)
    bs:
      int minibatch size.
    lambd (float):
       l2 penaly on the last layer.
    seed:
      random seed.

  Returns:
    None. Trains the model inplace.

  """

  x, t, e, a = shuffle(x, t, e, a, random_state=seed)
  n = x.shape[0]

  batches = (n // bs) + 1

  for i in range(batches):

    xb = x[i * bs:(i + 1) * bs]
    tb = t[i * bs:(i + 1) * bs]
    eb = e[i * bs:(i + 1) * bs]
    ab = a[i * bs:(i + 1) * bs]

    with tf.GradientTape() as tape:
      loss = partial_ll_loss(model, xb, tb, eb, ab, lambd)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test_step(model, x, t, e, a, loss='concordance', lambd=1.0):

  """Test the model and compute validation metric.

  Args:
    model:
      instance of CoupledDeepCPH class.
    x:
      a numpy array of input features (Val/Test Data).
    t:
      a numpy vector of event times (Val/Test Data).
    e:
      a numpy vector of event indicators (1 if event occured, 0 otherwise)
      (Val/Test Data).
    a:
      a numpy vector of the protected group membership (Val/Test Data).
    loss (str):
      string the loss metric to compute. one of 'concordance' or 'pll'.
    lambd (float):
       l2 penaly on the last layer.

  Returns:
    a float loss.

  """

  if loss == 'concordance':

    risks = np.zeros_like(a)

    lrisksp, lrisksn = model(x)

    lrisksp, lrisksn = lrisksp[:, 0], lrisksn[:, 0]

    risks[a == 1] = lrisksp[a == 1]
    risks[a == 0] = lrisksn[a == 0]

    pci = lifelines.utils.concordance_index(t[a == 1], -risks[a == 1],
                                            e[a == 1])
    nci = lifelines.utils.concordance_index(t[a == 0], -risks[a == 0],
                                            e[a == 0])
    return 0.5 * (nci + pci)

  if loss == 'pll':
    loss = partial_ll_loss(model, x, t, e, a, lambd)

    return float(loss)


def train(model,
          xt,
          tt,
          et,
          at,
          xv,
          tv,
          ev,
          av,
          groups,
          lambd=1,
          epochs=200,
          patience=2,
          vloss='pll'):

  """The function used to train the Coupled Deep CPH VAE.

  Trains the model and corresponding breslow's estimator given some training and
  validation examples for a fixed number of epochs and learning rate.

  Args:
    model:
      instance of CoupledDeepCPH class.
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
    lambd:
      float Strength of the VAE loss term.
    epochs:
      int Number of Training epochs to run.
    patience:
      number of training epochs to wait before stopping optimization.
    vloss:
      validation metric to optimize for. One of "pll" or "concordance".

  Returns:
    a trained survival analysis model and a breslow estimator.

  """

  prot, nprot = groups[0], groups[1]

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  valc = 0
  patience_ = 0

  # Convert A to a binary Indicator!

  at_ = at.copy()
  at_[at_ == prot] = 1
  at_[at_ == nprot] = 0

  av_ = av.copy()
  av_[av_ == prot] = 1
  av_[av_ == nprot] = 0

  for epoch in range(epochs):

    train_step(model, xt, tt, et, at_, optimizer, lambd=lambd, seed=epoch)
    valcn = test_step(model, xv, tv, ev, av_, loss=vloss, lambd=lambd)

    if epoch % 1 == 0:
      print(patience_, epoch, valcn)

    if valcn < valc:
      patience_ += 1

    if patience_ >= patience:
      return (model, train_breslow(model, xt, tt, et, at, xv, tv, ev, av,
                                   groups))

    valc = valcn

  return (model, train_breslow(model, xt, tt, et, at, xv, tv, ev, av, groups))

