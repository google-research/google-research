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

"""Utilitites to train and predict with baseline survival analysis models.

This module has utility functions to train and evaluate baseline survival
analysis models including,

Semi-Parametric:
  1) Deep Cox/Faraggi-Simon (pycox)
  2) Linear Cox CPH (lifelines)
Parametric:
  3) Deep Survival Machines (dsm, dsm_loss, dsm_utilities)
  4) Weibull AFT (lifelines)
Non-Parametric
  5) Random Survival Forest (pysurvival)

The module depends on various open-source packages which are indicated in
paranthesis above.

Not designed to be called directly, would be called when running a function from
fair_survival_analysis.fair_survival_analysis

"""

import copy

import dsm
import dsm_loss
import dsm_utilites

from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter

import numpy as np
import pandas as pd

from pycox.models import CoxPH
from pysurvival.models.survival_forest import RandomSurvivalForestModel

import torch
import torchtuples as ttup


def convert_to_data_frame(x, t, e):

  df = pd.DataFrame(data=x, columns=['X' + str(i) for i in range(x.shape[1])])
  df['T'] = pd.DataFrame(data=t.reshape(-1, 1), columns=['T'])
  df['E'] = pd.DataFrame(data=e.reshape(-1, 1), columns=['E'])

  return df


def structure_for_eval(t, e):

  struct = np.array([[e[i], t[i]] for i in range(len(e))])
  return struct


def structure_for_eval_(t, e):

  struct = np.array([(e[i], t[i]) for i in range(len(e))],
                    dtype=[('e', bool), ('t', int)])
  return struct


def _train_cph(x, t, e, folds, l2):

  fold_model = {}

  for f in set(folds):
    df = convert_to_data_frame(x[folds != f], t[folds != f], e[folds != f])
    cph = CoxPHFitter(penalizer=l2).fit(df, duration_col='T', event_col='E')
    fold_model[f] = copy.deepcopy(cph)

  return fold_model


def _train_aft(x, t, e, folds, l2):

  fold_model = {}

  for f in set(folds):
    df = convert_to_data_frame(x[folds != f], t[folds != f], e[folds != f])
    aft = WeibullAFTFitter(penalizer=l2).fit(df, duration_col='T',
                                             event_col='E')
    fold_model[f] = copy.deepcopy(aft)
  return fold_model


def _train_dcph(x, t, e, folds):

  """Helper Function to train a deep-cox model (DeepSurv, Faraggi-Simon).

  Args:
    x:
      a numpy array of input features (Training Data).
    t:
      a numpy vector of event times (Training Data).
    e:
      a numpy vector of event indicators (1 if event occured, 0 otherwise)
      (Training Data).
    folds:
       vector of the training cv folds.

  Returns:
    Trained pycox.CoxPH model.

  """

  in_features = x.shape[1]
  num_nodes = [100, 100]
  out_features = 1
  batch_norm = False
  dropout = 0.0
  output_bias = False

  fold_model = {}

  for f in set(folds):

    xf = x[folds != f]
    tf = t[folds != f]
    ef = e[folds != f]

    validx = sorted(
        np.random.choice(len(xf), size=(int(0.15 * len(xf))), replace=False))

    vidx = np.array([False] * len(xf))
    vidx[validx] = True

    net = ttup.practical.MLPVanilla(
        in_features,
        num_nodes,
        out_features,
        batch_norm,
        dropout,
        output_bias=output_bias).double()

    model = CoxPH(net, torch.optim.Adam)

    y_train = (tf[~vidx], ef[~vidx])
    y_val = (tf[vidx], ef[vidx])
    val = xf[vidx], y_val

    batch_size = 256
    model.optimizer.set_lr(0.001)
    epochs = 20
    callbacks = [ttup.callbacks.EarlyStopping()]

    model.fit(
        xf[~vidx],
        y_train,
        batch_size,
        epochs,
        callbacks,
        True,
        val_data=val,
        val_batch_size=batch_size)
    model.compute_baseline_hazards()

    fold_model[f] = model

  return fold_model


def _train_dsm(x, t, e, folds):

  """Helper Function to train a deep survival machines model.

  Args:
    x:
      a numpy array of input features (Training Data).
    t:
      a numpy vector of event times (Training Data).
    e:
      a numpy vector of event indicators (1 if event occured, 0 otherwise)
      (Training Data).
    folds:
       vector of the training cv folds.

  Returns:
    Trained dsm.DeepSurvivalMachines model.

  """

  xt = torch.from_numpy(x).double()
  tt = torch.from_numpy(t).double()
  et = torch.from_numpy(e).double()

  d = x.shape[1]

  fold_model = {}

  for f in set(folds):

    xf = xt[folds != f]
    tf = tt[folds != f]
    ef = et[folds != f]

    validx = sorted(
        np.random.choice(len(xf), size=(int(0.15 * len(xf))), replace=False))

    vidx = np.array([False] * len(xf))
    vidx[validx] = True

    model_dsm = dsm.DeepSurvivalMachines(
        inputdim=d, k=4, mlptype=2, HIDDEN=[100], dist='Weibull').float()
    model_dsm.double()

    model_dsm, _ = dsm_utilites.trainDSM(
        model_dsm,
        xf[~vidx],
        tf[~vidx],
        ef[~vidx],
        xf[vidx],
        tf[vidx],
        ef[vidx],
        n_iter=75,
        lr=1e-4,
        bs=10,
        alpha=1.0)

    fold_model[f] = copy.deepcopy(model_dsm)

  return fold_model


def _train_rsf(x, t, e, folds):

  fold_model = {}

  for f in set(folds):
    rsf = RandomSurvivalForestModel()
    rsf.fit(x[folds != f], t[folds != f], e[folds != f])
    fold_model[f] = copy.deepcopy(rsf)

  return fold_model


def train_model(x, t, e, folds, model='cph'):
  """The function used to train a survival analysis model.

  Trains and returns a trained baseline survival analysis model.

  Args:
    x:
      a numpy array of input features.
    t:
      a numpy vector of event times.
    e:
      a numpy vector of event indicators (1 if event occured, 0 otherwise).
    folds:
      a numpy vector of cv fold.
    model:
      choice of baseline model. One of "cph", "dcph", "dsm", "rsf", "aft".

  Returns:
    a trained survival analysis model.

  """
  print('Training ', model, ' model... Please be Patient...!')

  if model[:4] == 'dcph':
    fold_model = _train_dcph(x, t, e, folds)

  if model[:3] == 'cph':

    if len(model) > 3:
      l2 = 0.1
    else:
      l2 = 0.001

    fold_model = _train_cph(x, t, e, folds, l2)

  if model[:3] == 'aft':

    if len(model) > 3:
      l2 = 0.1
    else:
      l2 = 0.001

    fold_model = _train_aft(x, t, e, folds, l2)

  if model[:3] == 'dsm':
    fold_model = _train_dsm(x, t, e, folds)

  if model[:3] == 'rsf':
    fold_model = _train_rsf(x, t, e, folds)

  return fold_model


def _predict_cph(trained_model, x, quant, folds):

  scores = {}
  for fold in set(folds):

    scores[fold] = trained_model[fold].predict_survival_function(
        x[folds == fold], times=[quant]).T[quant].values

  return scores


def _predict_rsf(trained_model, x, quant, folds):

  scores = {}
  for fold in set(folds):

    dm = np.argmin(np.abs(quant - trained_model[fold].times))
    scores[fold] = trained_model[fold].predict_survival(
        x[folds == fold]).T[dm]

  return scores


def _predict_aft(trained_model, x, quant, folds):

  scores = {}
  for fold in set(folds):
    df = pd.DataFrame(
        data=x[folds == fold],
        columns=['X' + str(i) for i in range(x.shape[1])])
    scores[fold] = trained_model[fold].predict_survival_function(
        df, times=[quant]).T[quant].values
  return scores


def _predict_dsm(trained_model, x, quant, folds):

  scores = {}
  for fold in set(folds):
    xf = torch.from_numpy(x[folds == fold]).double()
    out = dsm_loss.predict_cdf(trained_model[fold], xf, [quant])
    out = out[0].data.exp().numpy()
    scores[fold] = out

  return scores


def _predict_dcph(trained_model, x, quant, folds):

  scores = {}
  for fold in set(folds):
    scores[fold] = trained_model[fold].predict_surv_df(
        x[folds == fold]).T[quant].values

  return scores


def predict_scores(trained_model, model, fair_strategy, x, a, folds, quant):
  """Used to evaluate risk at an event horizon from a trained survival model.

  Accepts a trained survival analysis model, features and horizon of interest
  and returns a numpy array of risks at the horizon.

  Args:
    trained_model:
      a trained survival analysis model
      (output of fair_survival_analysis.baseline_models.train_model).
    model:
      choice of baseline model. One of "cph", "dcph", "dsm", "rsf", "aft". Must
      be same as what was used to originally train the model.
    fair_strategy:
      strategy used to make the model "fair". One of None, "unawareness",
      "coupled". Must be same as what was used to originally train the model.
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

  if fair_strategy == 'coupled':

    scores_ = {}
    for group in trained_model:
      scores_[group] = predict_scores(trained_model[group], model, None, x, a,
                                      folds, quant)

    for fold in set(folds):

      scoresf = np.zeros_like(a[folds == fold])
      af = a[folds == fold]

      for group in trained_model:
        scoresf[af == group] = scores_[group][fold][af == group]

      scores[fold] = scoresf

    return scores

  else:

    if model[:3] == 'cph':
      scores = _predict_cph(trained_model, x, quant, folds)

    if model[:3] == 'aft':
      scores = _predict_aft(trained_model, x, quant, folds)

    if model[:3] == 'dsm':
      scores = _predict_dsm(trained_model, x, quant, folds)

    if model[:3] == 'rsf':
      scores = _predict_rsf(trained_model, x, quant, folds)

    if model[:4] == 'dcph':
      scores = _predict_dcph(trained_model, x, quant, folds)

  return scores
