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
dcm.deep_cox_mixture.baseline_experiment

"""

import copy

from deep_survival_machines import dsm
from deep_survival_machines import dsm_loss
from deep_survival_machines import dsm_utilites

from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter

import numpy as np
import pandas as pd

from pycox.models import CoxPH
from pycox.models import DeepHitSingle

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
                    dtype=[('e', bool), ('t', float)])
  return struct


def _train_cph(x, t, e, folds, params):
    
  if params is None:
    l2 = 1e-3
  else:
    l2 = params['l2']

  fold_model = {}

  for f in set(folds):
    df = convert_to_data_frame(x[folds != f], t[folds != f], e[folds != f])
    cph = CoxPHFitter(penalizer=l2).fit(df, duration_col='T', event_col='E')
    fold_model[f] = copy.deepcopy(cph)

  return fold_model

def _train_cph_sgd(x, t, e, folds, params):
  return _train_dcph(x, t, e, folds, params)


def _train_aft(x, t, e, folds, params):
    
  if params is None:
    l2 = 1e-3
  else:
    l2 = params['l2']

  fold_model = {}

  for f in set(folds):
    df = convert_to_data_frame(x[folds != f], t[folds != f], e[folds != f])
    aft = WeibullAFTFitter(penalizer=l2).fit(df, duration_col='T',
                                             event_col='E')
    fold_model[f] = copy.deepcopy(aft)
  return fold_model

def _train_dht(x, t, e, folds, params):

  """Helper Function to train a deep-hit model (van der schaar et. al).

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
    Trained pycox.DeepHitSingle model.

  """
  if params is None:
    num_nodes = [100,100]
    lr = 1e-3
    bs = 128
  else:
    num_nodes = params['num_nodes']
    lr =  params['lr'] 
    bs = params['bs']
    
  x = x.astype('float32')
  t = t.astype('float32')
  e = e.astype('int32')
    
#   num_durations = int(0.5*max(t))
#   print ("num_durations:", num_durations)

  num_durations = int(max(t))
  #num_durations = int(30)

  print ("num_durations:", num_durations)


  labtrans = DeepHitSingle.label_transform(num_durations, scheme='quantiles')
  #labtrans = DeepHitSingle.label_transform(num_durations,)

  #print (labtrans)

  in_features = x.shape[1]
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

    
    y_train = labtrans.fit_transform(tf[~vidx], ef[~vidx])
    y_val = labtrans.transform(tf[vidx], ef[vidx])
    out_features = labtrans.out_features

    net = ttup.practical.MLPVanilla(in_features, 
                                    num_nodes,
                                    out_features,
                                    batch_norm,
                                    dropout)
    
    model = DeepHitSingle(net, ttup.optim.Adam, alpha=0.5, sigma=1, duration_index=labtrans.cuts)

    
    y_train = y_train[0].astype('int64'), y_train[1].astype('float32')
    y_val = y_val[0].astype('int64'), y_val[1].astype('float32')

    val = xf[vidx], y_val
    train = xf[~vidx], y_train

    batch_size = bs
    model.optimizer.set_lr(lr)
    epochs = 10
    callbacks = [ttup.callbacks.EarlyStopping()]

    model.fit(
        xf[~vidx],
        y_train,
        batch_size,
        epochs,
        callbacks,
        True,
        val_data=val,)
    
    fold_model[f] = model

  return fold_model


def _train_dcph(x, t, e, folds, params):

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
  if params is None:
    num_nodes = [100,100]
    lr = 1e-3
    bs = 128
  
  else:
    
    num_nodes = params['num_nodes']
    lr =  params['lr'] 
    bs = params['bs']
    
  x = x.astype('float32')
  t = t.astype('float32')
  e = e.astype('int32')

  in_features = x.shape[1]
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
        output_bias=output_bias)

    model = CoxPH(net, torch.optim.Adam)

    y_train = (tf[~vidx], ef[~vidx])
    y_val = (tf[vidx], ef[vidx])
    
    y_train = y_train[0].astype('int64'), y_train[1].astype('float32')
    y_val = y_val[0].astype('int64'), y_val[1].astype('float32')

    val = xf[vidx], y_val

    batch_size = bs
    model.optimizer.set_lr(lr)
    epochs = 40
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


def _train_dsm(x, t, e, folds, params):

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
  if params is None:
    dist = 'Weibull'
    mlptyp, HIDDEN = 2, [100]
    lr = 1e-3
    bs = 128
    k = 4
    
  else:
    dist = params['dist']
    mlptyp, HIDDEN = params['HIDDEN']
    lr = params['lr']
    bs = params['bs']
    k = params['k']
    
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
        inputdim=d, k=k, mlptyp=mlptyp, HIDDEN=HIDDEN, dist=dist).float()
    model_dsm.double()

    print (xf[~vidx].shape, xf[vidx].shape)
    print (tf[~vidx].shape, tf[vidx].shape)
    print (ef[~vidx].shape, ef[vidx].shape)
 
    model_dsm, _ = dsm_utilites.trainDSM(
        model_dsm,
        [0.25, 0.5, 0.75],
        xf[~vidx],
        tf[~vidx],
        ef[~vidx],
        xf[vidx],
        tf[vidx],
        ef[vidx],
        n_iter=75,
        lr=lr,
        bs=bs,
        alpha=1.0)

    fold_model[f] = copy.deepcopy(model_dsm)

  return fold_model


def _train_rsf(x, t, e, folds, params):
    
  if params is None:
        
    num_trees = 50
    max_depth = 4
    
  else:
    
    num_trees = params['num_trees']
    max_depth = params['max_depth']
    
  xt = torch.from_numpy(x).double()
  tt = torch.from_numpy(t).double()
  et = torch.from_numpy(e).double()

  d = x.shape[1]

  fold_model = {}

  for f in set(folds):
    print ("Starting Fold:", f)
    rsf = RandomSurvivalForestModel(num_trees = num_trees)
    rsf.fit(x[folds != f], t[folds != f], e[folds != f], max_depth=max_depth)
    fold_model[f] = copy.copy(rsf)
    print ("Trained Fold:", f)
  return fold_model


def train_model(x, t, e, folds, model='cph', params=None):
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

  if model == 'dcph':
    fold_model = _train_dcph(x, t, e, folds, params)

  if model == 'cph':
    fold_model = _train_cph(x, t, e, folds, params)

  if model == 'cph_sgd':
    fold_model = _train_cph_sgd(x, t, e, folds, params)   
    
  if model == 'aft':
    fold_model = _train_aft(x, t, e, folds, params)

  if model == 'dsm':
    fold_model = _train_dsm(x, t, e, folds, params)

  if model == 'rsf':
    fold_model = _train_rsf(x, t, e, folds, params)
    
  if model == 'dht':
    fold_model = _train_dht(x, t, e, folds, params)

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


def _predict_dht(trained_model, x, quant, folds):
  
  x = x.astype('float32')
  scores = {}
  for fold in set(folds):
        
    scores_ = trained_model[fold].predict_surv_df(x[folds == fold]).T
    times = scores_.columns.values
    dm = np.argmin(np.abs(quant - times))
#     if times[dm]>quant:
#         scores[fold] = (scores_[times[dm]].values+scores_[times[dm-1]].values)/2
#     else: 
#         scores[fold] = (scores_[times[dm+1]].values+scores_[times[dm]].values)/2

    scores[fold] = scores_[times[dm]]

  return scores

def _predict_cph_sgd(trained_model, x, quant, folds):
    
  return _predict_dcph(trained_model, x, quant, folds)

def _predict_dcph(trained_model, x, quant, folds):
    
  return _predict_dht(trained_model, x, quant, folds)


def predict_scores(trained_model, model, fair_strategy, x, a, folds, quant):
  """Used to evaluate risk at an event horizon from a trained survival model.

  Accepts a trained survival analysis model, features and horizon of interest
  and returns a numpy array of risks at the horizon.

  Args:
    trained_model:
      a trained baseline survival analysis model
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
            
    if model == 'cph':
      scores = _predict_cph(trained_model, x, quant, folds)

    if model == 'cph_sgd':
      scores = _predict_cph_sgd(trained_model, x, quant, folds)
    
    if model == 'aft':
      scores = _predict_aft(trained_model, x, quant, folds)

    if model == 'dsm':
      scores = _predict_dsm(trained_model, x, quant, folds)

    if model == 'dht':
      scores = _predict_dht(trained_model, x, quant, folds)

    if model == 'rsf':
      scores = _predict_rsf(trained_model, x, quant, folds)

    if model == 'dcph':
      scores = _predict_dcph(trained_model, x, quant, folds)

  return scores
