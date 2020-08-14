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

"""Top Level interface with helper functions to run baselines and experiments.

This is the top level module that contains two functions, baseline_experiment()
and experiment() that run the survival analysis baselines and methods on the
SEER and SUPPORT datasets in a cross validation fashion. The module then outputs
the Expected Calibration Error and ROC characteristic at various event time
quantiles.

  Typical usage example:
  from fair_survival_analysis import fair_survival_analysis
  fair_survival_analysis.experiment(data='SUPPORT')
  fair_survival_analysis.baseline_experiment(data='SUPPORT')

"""

import pickle as pkl

import fair_survival_analysis.baseline_models as baseline_models
import fair_survival_analysis.data_utils as data_utils
import fair_survival_analysis.models as models
import fair_survival_analysis.plots as plots

import numpy as np
import pandas as pd


def baseline_experiment(data='SUPPORT', quantiles=(0.25, 0.5, 0.75),
                        prot_att='race', groups=('black', 'white'),
                        model='cph', fair_strategy=None, cv_folds=5):

  """Top level interface to train and evaluate a baseline survival model.

  This is the top level function that is designed to be called directly from
  inside a jupyter notebook kernel. This function allows the user to run
  several baselines survival analysis models on the SEER and SUPPORT datasets in
  a cross validation fashion. The function then plots and outputs the Expected
  Calibration Error and ROC characteristic at various event time quantiles.

  Args:
    data:
      a string that determines the dataset to run experiments on. One of "SEER"
      or "SUPPORT".
    quantiles:
      a list of event time quantiles at which the models are to be evaluated.
    prot_att:
      a string that specifies the column in the dataset that is to be treated
      as a protected attribute.
    groups:
      a list of two groups on which the survival analysis models are to be
      evaluated vis a vis accuracy and fairness.
    model:
      the choice of the baseline survival analysis model. One of "cph", "dcph",
      "dsm", "aft", "rsf"
    fair_strategy:
      string that specifies the fairness strategy to be used while running the
      experiment. One of None, "unawareness", "coupled".
    cv_folds:
      int that determines the number of Cross Validation folds.

  Returns:
    a Matplotlib figure with the ROC Curves and Reliability (Calibration) curves
    at various event quantiles.

  """

  if data == 'SUPPORT':

    data = pd.read_csv('data/support2.csv')
    x, t, e, a = data_utils.preprocess_support(data, prot=prot_att,
                                               fair_strategy=fair_strategy)

  elif data == 'SEER':

    data = pkl.load(open('data/SEER/preprocessed_SEER.pkl', 'rb'))
    x, t, e, a = data_utils.preprocess_seer(data, prot=prot_att,
                                            fair_strategy=fair_strategy)

  folds = np.array((list(range(cv_folds)) * (len(a) // cv_folds + 1))[:len(a)])

  quantiles = np.quantile(t[e == 1], quantiles)

  if fair_strategy == 'coupled':

    trained_model = {}
    for grp in groups:
      trained_model[grp] = baseline_models.train_model(x[a == grp],
                                                       t[a == grp],
                                                       e[a == grp],
                                                       folds=folds[a == grp],
                                                       model=model)

  else:
    trained_model = baseline_models.train_model(x, t, e, folds, model=model)

  plots.plotResults(trained_model, model, fair_strategy, x, e, t, a, folds,
                    groups, quantiles, strat='quantile', adj='IPCWpop')


def experiment(data='SUPPORT', quantiles=(0.25, 0.5, 0.75), prot_att='race',
               groups=('black', 'white'), model='coupled_deep_cph',
               cv_folds=5, seed=100):

  """Top level interface to train and evaluate proposed survival models.

  This is the top level function that is designed to be called directly from
  inside a jupyter notebook kernel. This function allows the user to run
  one of the proposed survival analysis models on the SEER and SUPPORT datasets
  in a cross validation fashion. The function then plots and outputs the
  Expected Calibration Error and ROC characteristic at various event time
  quantiles.

  Args:
    data:
      a string that determines the dataset to run experiments on. One of "SEER"
      or "SUPPORT".
    quantiles:
      a list of event time quantiles at which the models are to be evaluated.
    prot_att:
      a string that specifies the column in the dataset that is to be treated
      as a protected attribute.
    groups:
      a list of two groups on which the survival analysis models are to be
      evaluated vis a vis accuracy and fairness.
    model:
      the choice of the proposed survival analysis model. One of
      "coupled_deep_cph", "coupled_deep_cph_vae".
    cv_folds:
      int that determines the number of Cross Validation folds.
    seed:
      numpy random seed.
  Returns:
    a Matplotlib figure with the ROC Curves and Reliability (Calibration) curves
    at various event quantiles.

  """

  np.random.seed(seed)

  if data == 'SUPPORT':

    data = pd.read_csv('data/support2.csv')
    x, t, e, a = data_utils.preprocess_support(data, prot=prot_att,
                                               fair_strategy='unawareness')

  elif data == 'SEER':

    data = pkl.load(open('data/SEER/preprocessed_SEER.pkl', 'rb'))
    x, t, e, a = data_utils.preprocess_seer(data, prot=prot_att,
                                            fair_strategy='unawareness')

  folds = np.array((list(range(cv_folds)) * (len(a) // cv_folds + 1))[:len(a)])

  quantiles = np.quantile(t[e == 1], quantiles)

  trained_model = models.train_model(x, t, e, a, folds, groups, model=model)

  print('TRAINED:', trained_model.keys())

  plots.plotResults(trained_model, model, None, x, t, e, a, folds, groups,
                    quantiles, strat='quantile', adj='IPCWpop')
