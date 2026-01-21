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

"""Utilitites to train the proposed "Deep Cox Mixture" models.

This module has utility functions to train and evaluate proposed survival
analysis models including,

1) Cox Mixtures: Involves learning
2) Deep Cox Mixtures: Involves learning shared representations via the use
   of a Multilayer Perceptron, followed by a Gating function that determines
   latent Z as well as the group (Z) specific Cox model.
3) Deep Cox Mixtures-VAE: Similar to Deep Cox Mixtures, but uses a VAE to
   learn the latent representation instead of an MLP.s

The module depends on tensorflow 2.

Not designed to be called directly, would be called when running
dcm.deep_cox_mixture.experiment

"""

import copy

from dcm import dcm_tf as dcmt
import numpy as np

import logging


def train_model(x, t, e, a, folds, groups, params):
  """The function used to train a DCM model.

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
  Returns:
    a trained survival analysis model.

  """
  np.random.seed(0)


  fold_model = {}
    
  if params is None:
    params = {}
  
  k = params.get('k', 3)
  h = params.get('HIDDEN', 100)
  lr = params.get('lr', 1e-3)
  bs = params.get('bs', 128)
  epochs = params.get('epochs', 50)
  model = params.get('model', 'deep_cox_mixture')
  use_posteriors = params.get('use_posteriors', False)


  logging.info('Training ' + str(model) + ' model... Please be Patient...!')

  for f in set(folds):

    logging.info('On fold: ' + str(f+1))

    xf = x[folds != f]
    tf = t[folds != f]
    ef = e[folds != f]
    af = a[folds != f]

    validx = sorted(np.random.choice(len(xf), size=(int(0.15*len(xf))),\
                                       replace=False))
    vidx = np.array([False] * len(xf))
    vidx[validx] = True
    
    if model == 'deep_cox_mixture_vae':
      trained_model = dcmt.DeepCoxMixtureVAE(k, h, xf.shape[1])
      trained_model = dcmt.train(trained_model, xf[~vidx],
                                 tf[~vidx], ef[~vidx], af[~vidx],
                                 xf[vidx], tf[vidx], ef[vidx],
                                 af[vidx], epochs=epochs, lr=lr, bs=bs,
                                 use_posteriors=use_posteriors)

    if model == 'deep_cox_mixture':
      trained_model = dcmt.DeepCoxMixture(k, h)
      trained_model = dcmt.train(trained_model, xf[~vidx],
                                 tf[~vidx], ef[~vidx], af[~vidx],
                                 xf[vidx], tf[vidx], ef[vidx],
                                 af[vidx], epochs=epochs, lr=lr, bs=bs,
                                 use_posteriors=use_posteriors)
    
    if model == 'cox_mixture':
      trained_model = dcmt.CoxMixture(k)
      trained_model = dcmt.train(trained_model, xf[~vidx],
                                 tf[~vidx], ef[~vidx], af[~vidx],
                                 xf[vidx], tf[vidx], ef[vidx],
                                 af[vidx], epochs=epochs, lr=lr, bs=bs,
                                 use_posteriors=use_posteriors)

    fold_model[f] = copy.copy(trained_model)
     
  return fold_model


def predict_scores(trained_model, groups, x, a, folds, quant):
  """Used to evaluate risk at an event horizon from a trained survival model.

  Accepts a trained survival analysis model, features and horizon of interest
  and returns a numpy array of risks at the horizon.

  Args:
    trained_model:
      a trained survival analysis model
      (output of deep_cox_mixtures.models.train_model).
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
        
    if type(trained_model[0][0]).__name__ in ['DeepCoxMixture',
                                              'DeepCoxMixtureVAE',
                                              'CoxMixture'] :
      out = dcmt.predict_scores(trained_model[fold], xf,
                               quant)
    else:
      raise Exception("Unsupported Model!")

    scores[fold] = out

  return scores
