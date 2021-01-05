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

"""Utilitites to train the proposed "fair" survival analysis models.

This module has utility functions to train and evaluate proposed survival
analysis models including,

1) Coupled Deep-Cox: Involves sharing representations between demographics
   followed by a separate Breslow Estimator for each demographic.

2) Coupled Deep-Cox VAE: Involves learning shared representations via the use
   of a Variational Auto-Encoder. The parameters of the VAE and the Cox layers
   are learnt jointly.

The module depends on tensorflow 2.

Not designed to be called directly, would be called when running a function from
fair_survival_analysis.fair_survival_analysis

"""

import copy

import coupled_deep_cph as coupled_dcph
import coupled_deep_cph_utils as coupled_dcph_util
import coupled_deep_cph_vae as coupled_dcph_vae


import numpy as np


def train_model(x, t, e, a, folds, groups, model='coupled_deep_cph'):
  """The function used to train a survival analysis model.

  Trains and returns a proposed survival analysis model.

  Args:
    x:
      a numpy array of input features.
    t:
      a numpy vector of event times.
    e:
      a numpy vector of event indicators (1 if event occured, 0 otherwise).
    a:
      a numpy vector of the protected group membership.
    folds:
      a numpy vector of cv fold.
    groups:
      List of the demogrpahics to adjust for.
    model:
      choice of baseline model. One of "coupled_deep_cph",
      "coupled_deep_cph_vae"
  Returns:
    a trained survival analysis model.

  """
  np.random.seed(0)
  print('Training ', model, ' model... Please be Patient...!')

  if model == 'coupled_deep_cph':

    fold_model = {}

    for f in set(folds):

      xf = x[folds != f]
      tf = t[folds != f]
      ef = e[folds != f]
      af = a[folds != f]

      validx = sorted(np.random.choice(len(xf), size=(int(0.15*len(xf))),\
                                       replace=False))
      vidx = np.array([False] * len(xf))
      vidx[validx] = True

      trained_model = coupled_dcph.CoupledDeepCPH(100)

      trained_model = coupled_dcph.train(trained_model, xf[~vidx], tf[~vidx],
                                         ef[~vidx], af[~vidx], xf[vidx],
                                         tf[vidx], ef[vidx], af[vidx], groups)

      fold_model[f] = copy.deepcopy(trained_model)

      return fold_model

  if model == 'coupled_deep_cph_vae':

    fold_model = {}

    x = x.astype('float32')
    t = t.astype('float32')
    e = e.astype('float32')

    for f in set(folds):

      xf = x[folds != f]
      tf = t[folds != f]
      ef = e[folds != f]
      af = a[folds != f]

      validx = sorted(np.random.choice(len(xf), size=(int(0.15*len(xf))),
                                       replace=False))

      vidx = np.array([False] * len(xf))
      vidx[validx] = True

      trained_model = coupled_dcph_vae.CoupledDeepCPHVAE(100, xf.shape[1])
      trained_model = coupled_dcph_vae.train(trained_model, xf[~vidx],
                                             tf[~vidx], ef[~vidx], af[~vidx],
                                             xf[vidx], tf[vidx], ef[vidx],
                                             af[vidx], groups)

      fold_model[f] = copy.deepcopy(trained_model)
      return fold_model


def predict_scores(trained_model, groups, x, a, folds, quant):
  """Used to evaluate risk at an event horizon from a trained survival model.

  Accepts a trained survival analysis model, features and horizon of interest
  and returns a numpy array of risks at the horizon.

  Args:
    trained_model:
      a trained survival analysis model
      (output of fair_survival_analysis.models.train_model).
    groups:
      List of the demogrpahics to adjust for. Must be same as what was used to
      originally train the model.
    x:
      a numpy array of input features.
    a:
      a numpy vector of protected attributes.
    folds:
      a numpy vector of cv fold.
    quant:
      float/int of the event time at which risk is to be evaluated.
  Returns:
    a numpy vector of risks P(T>t) at the horizon "quant".

  """

  scores = {}

  for fold in set(folds):
    xf = x[folds == fold]
    af = a[folds == fold]

    out = coupled_dcph_util.predict_survival(trained_model[fold], xf,
                                             quant, af, groups)
    scores[fold] = out

  return scores
