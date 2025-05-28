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

"""A minimal implementation of the latent shift adaptation algorithm with known U using scikit-learn and numpy.

This corresponds to equation (1) in https://arxiv.org/abs/2212.11254.
The TF/Keras equivalent of this algorithm is implemented in
latent_shift_adaptation/methods/bbsez.py.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def latent_shift_adaptation(
    x_source,
    y_source,
    u_source,
    x_target,
    u_is_one_hot = False,
    u_weights = None,
    model_type = 'mlp',
):
  """A minimal implementation of latent shift adaptation.

  Computes p_t(Y | X) propto sum_u [p_s(Y | X, U) p_s(U | X) p_t(U) / p_s(U)]

  Arguments:
    x_source: the features X in the source domain. Shape: (n_samples,
      n_features)
    y_source: the categorical labels Y in the source domain. Shape: (n_samples,
      )
    u_source: the categorical unobserved confounder U in the source domain.
      Shape (n_samples, ) if u_is_one_hot is True, (n_samples, n_categories)
      otherwise
    x_target: the features X in the target domain. Shape: (n_samples,
      n_features)
    u_is_one_hot: flag for whether u_source is provided as a one hot encoding or
      a 1-d array of indices
    u_weights: optional label shift weights p_t(U) / p_s(U)
    model_type: str indicator for the model class used for auxiliary models.
      Valid inputs include 'logistic', 'spline', 'gradient_boosting', 'mlp'

  Returns:
    an array of predicted probabilities in the target domain p_t(Y | X)
  """

  model_p_u_x = get_classifier(model_type)
  model_p_y_u_x = get_classifier(model_type)

  # If U is one hot, extract categories
  if u_is_one_hot:
    u_source = u_source.argmax(axis=-1)

  # Pre-process U, dropping unobserved categories
  u_label_encoder = LabelEncoder()
  u_source = u_label_encoder.fit_transform(u_source)

  # Get one hot U over observed categories
  u_one_hot_encoder = OneHotEncoder(sparse=False)
  u_source_one_hot = u_one_hot_encoder.fit_transform(u_source.reshape(-1, 1))
  num_categories = len(u_one_hot_encoder.categories_[0])

  # Fit p_s(U | X)
  model_p_u_x = model_p_u_x.fit(x_source, u_source)

  # Fit p_s(Y | U, X)
  model_p_y_u_x.fit(
      np.concatenate((x_source, u_source_one_hot), axis=1), y_source
  )

  # Label shift correction to estimate weights proportional to p_t(U) / p_s(U)
  if u_weights is None:
    u_predicted = model_p_u_x.predict(x_source)
    conf_mat = confusion_matrix(u_source, u_predicted, normalize='all')
    u_predicted_target = model_p_u_x.predict(x_target)
    u_predicted_frac = (
        u_one_hot_encoder.transform(u_predicted_target.reshape(-1, 1))
        .mean(axis=0)
        .reshape(-1, 1)
    )
    weights = np.maximum(np.linalg.pinv(conf_mat.T) @ u_predicted_frac, 0.0)
  else:
    weights = u_weights

  # p_t(Y | X) \propto \sum_u [p_s(Y | X, U) p_s(U | X) p_t(U) / p_s(U)]
  result_temp = np.zeros(
      (x_target.shape[0], len(model_p_y_u_x.classes_), num_categories)
  )
  for i, category in enumerate(u_one_hot_encoder.categories_[0]):
    result_temp[:, :, i] = (
        model_p_y_u_x.predict_proba(
            np.concatenate(
                (
                    x_target,
                    u_one_hot_encoder.transform(
                        category * np.ones(x_target.shape[0]).reshape(-1, 1)
                    ),
                ),
                axis=1,
            )
        )
        * model_p_u_x.predict_proba(x_target)[:, i].reshape(-1, 1)
        * weights[i]
    )

  # Sum over U
  result_temp = result_temp.sum(axis=-1)
  # Normalize
  pred_probs_target = result_temp / result_temp.sum(axis=-1, keepdims=True)
  return pred_probs_target


def get_classifier(model_type = 'mlp', **kwargs):
  """Return a classifier based on provided model type.

  Arguments:
    model_type: 'logistic', 'gradient_boosting', or 'mlp'
    **kwargs: additional keyword arguments passed to the model constructor

  Returns:
    sklearn estimator
  """
  if model_type == 'logistic':
    model = LogisticRegression(penalty='none', **kwargs)
  elif model_type == 'gradient_boosting':
    model = HistGradientBoostingClassifier(**kwargs)
  elif model_type == 'mlp':
    model = MLPClassifier(**kwargs)
  else:
    raise ValueError('Invalid model_type')
  return model
