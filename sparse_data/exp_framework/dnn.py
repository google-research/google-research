# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Trains and evaluates deep neural network classifiers.

Trains and evaluates deep neural network classification models using Keras.
Performs parameter tuning with grid search and randomized search.
"""

from keras import backend as K
from keras import regularizers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ThresholdedReLU
from keras.models import Model
from keras.optimizers import Adam
# from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

import numpy as np
from scipy import sparse

from sparse_data.exp_framework.utils import generic_pipeline


def pseudo_partial(func, **kwargs):
  """Does the same thing as functool.partial but returns a function.

  Useful if an API (e.g., Keras) uses getargspec which doesn't handle functions.

  Arguments:
    func: function
    **kwargs: additional keyword arguments

  Returns:
    new_func: function
      a function which behaves like func(**kwargs)
  """

  def new_func():
    return func(**kwargs)

  return new_func


def keras_build_fn(num_feature,
                   num_output,
                   is_sparse,
                   embedding_dim=-1,
                   num_hidden_layer=2,
                   hidden_layer_dim=512,
                   activation='elu',
                   learning_rate=1e-3,
                   dropout=0.5,
                   l1=0.0,
                   l2=0.0,
                   loss='categorical_crossentropy'):
  """Initializes and compiles a Keras DNN model using the Adam optimizer.

  Args:
    num_feature: number of features
    num_output: number of outputs (targets, e.g., classes))
    is_sparse: boolean whether input data is in sparse format
    embedding_dim: int number of nodes in embedding layer; if value is <= 0 then
      no embedding layer will be present in the model
    num_hidden_layer: number of hidden layers
    hidden_layer_dim: int number of nodes in the hidden layer(s)
    activation: string
      activation function for hidden layers; see https://keras.io/activations/
    learning_rate: float learning rate for Adam
    dropout: float proportion of nodes to dropout; values in [0, 1]
    l1: float strength of L1 regularization on weights
    l2: float strength of L2 regularization on weights
    loss: string
      loss function; see https://keras.io/losses/

  Returns:
    model: Keras.models.Model
      compiled Keras model
  """
  assert num_hidden_layer >= 1

  inputs = Input(shape=(num_feature,), sparse=is_sparse)

  activation_func_args = ()
  if activation.lower() == 'prelu':
    activation_func = PReLU
  elif activation.lower() == 'leakyrelu':
    activation_func = LeakyReLU
  elif activation.lower() == 'elu':
    activation_func = ELU
  elif activation.lower() == 'thresholdedrelu':
    activation_func = ThresholdedReLU
  else:
    activation_func = Activation
    activation_func_args = (activation)

  if l1 > 0 and l2 > 0:
    reg_init = lambda: regularizers.l1_l2(l1, l2)
  elif l1 > 0:
    reg_init = lambda: regularizers.l1(l1)
  elif l2 > 0:
    reg_init = lambda: regularizers.l2(l2)
  else:
    reg_init = lambda: None

  if embedding_dim > 0:
    # embedding layer
    e = Dense(embedding_dim)(inputs)

    x = Dense(hidden_layer_dim, kernel_regularizer=reg_init())(e)
    x = activation_func(*activation_func_args)(x)
    x = Dropout(dropout)(x)
  else:
    x = Dense(hidden_layer_dim, kernel_regularizer=reg_init())(inputs)
    x = activation_func(*activation_func_args)(x)
    x = Dropout(dropout)(x)

  # add additional hidden layers
  for _ in range(num_hidden_layer - 1):
    x = Dense(hidden_layer_dim, kernel_regularizer=reg_init())(x)
    x = activation_func(*activation_func_args)(x)
    x = Dropout(dropout)(x)

  x = Dense(num_output)(x)
  preds = Activation('softmax')(x)

  model = Model(inputs=inputs, outputs=preds)
  model.compile(optimizer=Adam(lr=learning_rate), loss=loss)

  return model


def pipeline(x_train,
             y_train,
             x_test,
             y_test,
             param_dict=None,
             problem='classification'):
  """Trains and evaluates a DNN classifier.

  Args:
    x_train: np.array or scipy.sparse.*matrix array of features of training data
    y_train: np.array 1-D array of class labels of training data
    x_test: np.array or scipy.sparse.*matrix array of features of test data
    y_test: np.array 1-D array of class labels of the test data
    param_dict: {string: ?} dictionary of parameters of their values
    problem: string type of learning problem; values = 'classification',
      'regression'

  Returns:
    model: Keras.models.Model
      trained Keras model
    metrics: {str: float}
      dictionary of metric scores
  """
  assert problem in ['classification', 'regression']

  if param_dict is None:
    param_dict = {'epochs': 10, 'batch_size': 256}

  num_feature = x_train.shape[1]
  is_sparse = sparse.issparse(x_train)

  param_dict = param_dict.copy()
  num_epoch = param_dict.pop('epochs')
  batch_size = param_dict.pop('batch_size')

  if problem == 'regression':
    num_output = 1
    loss = 'mean_squared_error'
    model_init = KerasRegressor
  else:
    num_output = len(set(y_train))
    loss = 'categorical_crossentropy'
    model_init = FunctionalKerasClassifier

  build_fn = pseudo_partial(
      keras_build_fn,
      num_feature=num_feature,
      num_output=num_output,
      is_sparse=is_sparse,
      loss=loss,
      **param_dict)
  model = model_init(
      build_fn=build_fn,
      epochs=num_epoch,
      batch_size=batch_size,
      shuffle=True,
      verbose=False)

  return generic_pipeline(
      model, x_train, y_train, x_test, y_test, problem=problem)


class FunctionalKerasClassifier(KerasClassifier):
  """Helper scikit-learn wrapper for a Keras model.

  The default KerasClassifier's predict() method does not work for functional
  Keras models (https://github.com/fchollet/keras/issues/2524); this breaks
  using this wrapper with the scikit-learn framework (e.g., search methods).
  """

  def predict_proba(self, x, **kwargs):
    """Predict classes from features.

    Args:
      x: np.array or scipy.sparse.*matrix array of features
      **kwargs: additional keyword arguments

    Returns:
      y_pred: np.array
        2-D array of class predicted probabilities
    """
    kwargs = self.filter_sk_params(Model.predict, kwargs)
    probas = self.model.predict(x, **kwargs)
    return probas

  def predict(self, x, **kwargs):
    """Predict classes from features.

    Args:
      x: np.array or scipy.sparse.*matrix array of features
      **kwargs: additional keyword arguments

    Returns:
      y_pred: np.array
        1-D array of class predictions (not probabilities)
    """
    kwargs = self.filter_sk_params(Model.predict, kwargs)
    probas = self.model.predict(x, **kwargs)
    return np.argmax(probas, axis=1)


def clear_keras_session():
  """Clears Keras session."""
  K.clear_session()
