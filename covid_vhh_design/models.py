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

"""CoVID modeling code."""

import os
from typing import Sequence, Type

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import base as sk_base
from sklearn import multioutput as sk_multioutput
from tensorflow.io import gfile

from covid_vhh_design import covid
from covid_vhh_design import helper
from covid_vhh_design import utils


LGB_MODEL_DIR = os.path.join(covid.DATA_DIR, 'lgb_model')
AAINDEX_FILENAME = os.path.join(covid.DATA_DIR, 'aaindex_components.csv')
AMINO_ACIDS = tuple(sorted(utils.AMINO_ACIDS))

# Output targets that where used for model training
TARGET_NAMES = (
    'SARS-CoV2_RBD_S477N',
    'SARS-CoV2_RBD_N439K',
    'SARS-CoV2_RBD_V367F',
    'SARS-CoV2_RBD_R408I',
    'SARS-CoV2_RBD_N501F',
    'SARS-CoV2_RBD_N501D',
    'SARS-CoV2_RBD_G502D',
    'SARS-CoV1_RBD',
    'SARS-CoV2_RBD',
    'SARS-CoV2_RBD_N501Y',
    'SARS-CoV2_RBD_N501Y+K417N+E484K',
    'SARS-CoV2_RBD_N501Y+K417T+E484K',
)


class MultiOutputClassifier(sk_multioutput.MultiOutputClassifier):
  """Wrapper that reshapes output prob. of a binary MultiOutputClassifier."""

  def predict_proba(
      self, x, class_idx = 1, **kwargs
  ):
    probas = np.stack(super().predict_proba(x, **kwargs))
    probas = np.rollaxis(probas, 1, 0)
    if class_idx is not None:
      probas = probas[:, :, class_idx]
    return probas


class LGBMBoosterRegressor(sk_base.RegressorMixin, sk_base.BaseEstimator):
  """Sklearn estimator that holds a fitted LGBM booster regressor."""

  def __init__(self, booster):
    self.booster = booster

  def __sklearn_is_fitted__(self):
    return True

  def predict(self, *args, **kwargs):
    return self.booster.predict(*args, **kwargs)  # pytype: disable=bad-return-type  # scipy


class LGBMBoosterClassifier(sk_base.ClassifierMixin, sk_base.BaseEstimator):
  """Sklearn estimatar that holds a fitted LGBM booster classifier."""

  def __init__(self, booster):
    self.booster = booster

  def __sklearn_is_fitted__(self):
    return True

  def predict_proba(self, *args, **kwargs):
    y = self.booster.predict(*args, **kwargs)
    return np.stack((1.0 - y, y)).T

  def predict(self, *args, **kwargs):
    return self.predict_proba(*args, **kwargs).argmax(axis=1)


def _load_lgbm_booster(filename):
  """Loads an LGBM booster from a JSON file."""
  with gfile.GFile(filename, 'r') as f:
    return lgb.Booster(model_str=f.read())


def _get_multi_output_lgbm_filenames(dirname):
  return sorted(gfile.glob(os.path.join(dirname, '[0-9]' * 4 + '.txt')))


MultiOutputRegressor = sk_multioutput.MultiOutputRegressor
MultiOutputEstimator = MultiOutputRegressor | MultiOutputClassifier
LGBMBoosterCls = Type[LGBMBoosterRegressor] | Type[LGBMBoosterClassifier]
MultiOutputEstimatorCls = (
    Type[MultiOutputRegressor] | Type[MultiOutputClassifier]
)


def _load_multi_output_lgbm_estimator(
    dirname,
    estimator_cls,
    multioutput_cls,
):
  """Loads a multi-output LGBM estimator from JSON files."""
  estimators = []
  for filename in _get_multi_output_lgbm_filenames(dirname):
    estimators.append(estimator_cls(_load_lgbm_booster(filename)))
  if not estimators:
    raise ValueError(f'No model txt files found in {dirname}!')
  else:
    helper.tprint(
        'Building %s with %d outputs', multioutput_cls.__name__, len(estimators)
    )
  # A sklearn MultiOutputEstimator is constructor with a single, unfitted
  # estimator. To construct a MultiOutputEstimator with fitted LGBM estimators,
  # we first pass a single estimator to the constructor and then modify the
  # `estimators_` attribute.
  multi_output_estimator = multioutput_cls(estimators[0])
  multi_output_estimator.estimators_ = estimators
  return multi_output_estimator


def load_multi_output_lgbm_regressor(
    dirname,
):
  """Loads a multi-output LGBM regressor."""
  return _load_multi_output_lgbm_estimator(
      dirname, LGBMBoosterRegressor, sk_multioutput.MultiOutputRegressor
  )


def load_multi_output_lgbm_classifier(dirname):
  """Loads a multi-output LGBM classifier."""
  return _load_multi_output_lgbm_estimator(
      dirname, LGBMBoosterClassifier, MultiOutputClassifier
  )


class CombinedModel:
  """A combined regressor-classifier model.

  Multiplies the predictions of a regressor with the predictions of a
  classifier.
  """

  def __init__(
      self, regressor, classifier
  ):
    """Creates and instance of this class.

    Args:
      regressor: A pre-trained regressor.
      classifier: A pre-trained classifier.
    """
    self._regressor = regressor
    self._classifier = classifier

  @property
  def regressor(self):
    """Returns the regressor."""
    return self._regressor

  @property
  def classifier(self):
    """Returns the classifier."""
    return self._classifier

  @classmethod
  def load(cls, dirname = LGB_MODEL_DIR, **kwargs):
    """Loads a combined LGBM regressor/classifier model."""
    return load_combined_lgbm_model(dirname, **kwargs)

  def predict(self, x):
    """Predicts binding scores for the given featurized sequences.

    Args:
      x: An array [num_seqs, num_features] with featurized sequences.

    Returns:
      An array [num_seqs, num_targets] with output binding scores for each
      sequence and target.
    """
    scores = np.asarray(self._regressor.predict(x))
    probs = np.asarray(self._classifier.predict_proba(x))
    return scores * probs


class SequenceEncoder:
  """Encodes (featurizes) an amino acid sequence for making predictions.

  Onehot- and AAIndex encodes each sequence, and concatenates the resulting
  encodings.
  """

  def __init__(
      self,
      numbering = None,
      aaindex_features = None,
      num_aaindex_features = 10,
      positions = covid.ALLOWED_POS,
  ):
    """Creates an instance of this class.

    Args:
      numbering: IMGT numbering table.  Will be read from disk if `None`.
      aaindex_features: AAIndex matrix. Will be read from disk if `None`.
      num_aaindex_features: The number of AAIndex features to be used for
        encoding.
      positions: The IMGT positions that were used for model training.
    """
    if numbering is None:
      numbering = covid.load_aligned_parent_seq()
    if aaindex_features is None:
      aaindex_features = helper.read_csv(AAINDEX_FILENAME)
    self._numbering = numbering

    self._indices = numbering.loc[
        numbering['pos'].isin(positions), 'index'
    ].values
    self._token_to_aaindex = {
        token: features.values
        for token, features in aaindex_features.set_index('token')
        .iloc[:, :num_aaindex_features]
        .iterrows()
    }
    self._token_to_onehot = {}
    for index, token in enumerate(AMINO_ACIDS):
      self._token_to_onehot[token] = np.zeros(len(AMINO_ACIDS))
      self._token_to_onehot[token][index] = 1

  def encode_token(self, token):
    """Encodes an amino acid."""
    onehot_features = self._token_to_onehot[token]
    aaindex_features = self._token_to_aaindex[token]
    return np.hstack([onehot_features, aaindex_features])

  def encode_sequence(self, sequence):
    """Encodes an amino acid sequence."""
    if len(sequence) != len(self._numbering):
      raise ValueError(
          f'Sequence must be {len(self._numbering)} characters long but is'
          f' {len(sequence)} characters long! {sequence}'
      )

    return np.vstack([self.encode_token(sequence[i]) for i in self._indices])

  def encode_sequences(
      self, sequences, flatten = True
  ):
    """Encodes multiple amino acid sequences."""
    encoded = np.stack(
        [self.encode_sequence(sequence) for sequence in sequences]
    )
    if flatten:
      onehot_features = encoded[:, :, : len(AMINO_ACIDS)].reshape(
          len(encoded), -1
      )
      aaindex_features = encoded[:, :, len(AMINO_ACIDS) :].reshape(
          len(encoded), -1
      )
      return np.hstack([onehot_features, aaindex_features])
    else:
      return encoded


def load_combined_lgbm_model(dirname, **kwargs):
  """Loads a combined LGBM regressor/classifier model."""
  regressor = load_multi_output_lgbm_regressor(
      os.path.join(dirname, 'regressor')
  )
  classifier = load_multi_output_lgbm_classifier(
      os.path.join(dirname, 'classifier')
  )
  return CombinedModel(regressor=regressor, classifier=classifier, **kwargs)


def score_sequences(
    model,
    encoder,
    sequences,
    proba = False,
):
  """Scores sequences with the provided model."""
  encoded_seqs = encoder.encode_sequences(sequences)
  predict_fn = model.predict_proba if proba else model.predict
  return pd.DataFrame(predict_fn(encoded_seqs), columns=TARGET_NAMES)


def score_labeled_sequences(
    model,
    encoder,
    sequences,
    proba = False,
    label_column = 'label',
):
  """Scores a dict of labeled sequences with the provided model."""
  return score_sequences(
      model, encoder, list(sequences.values()), proba=proba
  ).set_index(pd.Index(list(sequences), name=label_column))
