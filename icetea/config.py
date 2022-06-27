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

"""config!

config files creation of objects, and organization

"""

import itertools
from typing import Dict
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics as sk_metrics
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from icetea import estimators

IMAGE_SIZE = [587, 587]
TensorDict = Dict[str, tf.Tensor]


class MakeParameters:
  """Make Parameters Objects.

  Args:
    - configs_data: dictionary with data parameters from flags.
    - configs_methods: dictionary with methods parameters from flags.
  Return:
    parameters: methods parameteres as objects.
  """

  def __init__(self, configs_data, configs_methods):
    super(MakeParameters, self).__init__()

    self.config_data = _create_configs(configs_data)
    self.config_methods = _create_configs(configs_methods)

    self.parameters_data = _read_data_configs(self.config_data)
    self.parameters_method = _read_method_configs(
        self.config_methods,
        self.parameters_data[0]['data_name'],
        configs_data['setting'][0],
        self.parameters_data[0].get('data_low_dimension', True),
        )


def _create_configs(dict_):
  """Creates a list of dictionaries with the parameters.

  Expands all possible combinations between lists inside dictionary.
  Example: dict1 = {a=['data1','data2'], b=['model1','model2']}
  output= [{a='data1',b='model1'}, {a='data1',b='model2'},
  {a='data2',b='model1'}, {a='data2',b='model2'}].

  Args:
    dict_: dictionary with the lists.
  Returns:
    list of config files.
  """
  keys = dict_.keys()
  vals = dict_.values()
  configs = []
  for instance in itertools.product(*vals):
    configs.append(dict(zip(keys, instance)))
  return configs


def _read_method_configs(config_methods,
                         data_name,
                         setting,
                         data_low_dimension):
  """Creates list of dictionaries.

  Each dict is a set of config parameters for the methods.
  Args:
    config_methods: list of dictionaries
    data_name: string with data name
    setting: quick (for testing), samples or covariates analysies
    data_low_dimension: bool, only used when data is ACIC
  Returns:
     list with all config parameters, some of them are objects
  """
  parameters_method = []
  for config in config_methods:
    parameters = {}
    parameters['estimator'] = _estimator_function(
        config['name_estimator'])
    parameters['name_estimator'] = config['name_estimator']
    parameters['base_model'] = _base_learner(
        config['name_base_model'], data_name)
    parameters['name_base_model'] = config['name_base_model']
    parameters['metric'] = _metric_function(config['name_metric'])
    parameters['name_metric'] = config['name_metric']
    parameters['prop_score'] = _prop_score_function(
        config['name_prop_score'])
    parameters['name_prop_score'] = config['name_prop_score']
    parameters['param_grid'] = _create_param_grid(
        config['name_base_model'],
        data_name, setting, data_low_dimension)
    parameters_method.append(parameters)
  return parameters_method


def _read_data_configs(config_data):
  """Creates list of dictionaries.

  Each dict is a set of config parameters for the data.
  Args:
    config_data: dictionary with data parameters
  Returns:
    list with all config parameters.
  """
  parameters_data = []
  for config in config_data:
    parameters = {}
    if config['data_name'] == 'simple_linear':
      parameters['data_name'] = config['data_name']
      parameters['sample_size'] = config['data_n']
      parameters['num_covariates'] = config['data_num_covariates']
      parameters['noise'] = config['data_noise']
      parameters['linear'] = config['data_linear']
    else:
      parameters['data_name'] = config['data_name']
      parameters['data_path'] = config['data_path']
      parameters['data_low_dimension'] = config['data_low_dimension']
    parameters_data.append(parameters)
  return parameters_data


def _create_param_grid(name_model, data_name, setting, data_low_dimension):
  """Creates parameters for GridSearchCV.

  Args:
    name_model: name of the base model.
    data_name: str, synthethic, ACIC, IHDP
    setting: str, quick (for testing), samples or covariates (synthetic)
    data_low_dimension: bool, only when data_name == ACIC
  Returns:
    dictionary with parameters.
  """
  if name_model == 'Lasso':
    return {'alpha': [0.01, 0.1, 1]}
  elif name_model == 'GBoost':
    return {'n_estimators': [30], 'min_samples_split': [3, 5, 15]}
  elif name_model == 'RandomForest':
    return {'n_estimators': [30], 'min_samples_leaf': [3, 5, 15]}
  elif name_model == 'ElasticNet':
    return {'alpha': [0.01, 0.1, 1]}
  elif name_model == 'NN_regression':
    if data_name == 'simple_linear':
      if setting == 'covariates':
        return {
            'batch_size': [200],
            'epochs': [25],
            'h_units': [1, 10, 50, 100, 500],
            'learning_rate': [0.01],
            'l1': [0.01],
        }
      elif setting == 'quick':
        return {
            'batch_size': [200],
            'epochs': [25],
            'h_units': [5],
            'learning_rate': [0.01],
            'l1': [0.01],
        }
      else:
        return {
            'batch_size': [10, 25],
            'epochs': [100],
            'h_units': [1, 5],
            'learning_rate': [0.01],
            'l1': [0.01, 0.1],
        }
    elif data_name == 'IHDP':
      return {
          'batch_size': [75],
          'epochs': [75],
          'h_units': [5, 10],
          'l1': [0.1],
          'learning_rate': [0.01, 0.1]
      }
    elif data_name == 'ACIC':
      if data_low_dimension:
        return {
            'batch_size': [200],
            'epochs': [150],
            'h_units': [5, 10, 15],
            'l1': [0.0001, 0.01],
            'learning_rate': [0.01]
        }
      else:
        return {
            'batch_size': [200],
            'epochs': [150],
            'h_units': [50, 100],
            'l1': [0.0001],
            'learning_rate': [0.01]
        }
    else:
      return None
  else:
    return None


def _base_learner(name_model, data_name):
  """Creates base-model object.

  Args:
    name_model: name of the base-model.
    data_name: name of the dataset
  Raises:
    Exception: Not Implemented Error.
  Returns:
    model object.
  """
  if name_model == 'LinearRegression':
    return linear_model.LinearRegression()
  elif name_model == 'Lasso':
    return linear_model.Lasso()
  elif name_model == 'RandomForest':
    return ensemble.RandomForestRegressor()
  elif name_model == 'ElasticNet':
    return linear_model.ElasticNet()
  elif name_model == 'XGBoost':
    return ensemble.GradientBoostingRegressor()
  elif name_model == 'NN_regression':
    if data_name == 'ACIC':
      return KerasRegressor(build_fn=create_nn_regression_acic, verbose=0)
    elif data_name == 'IHDP':
      return KerasRegressor(build_fn=create_nn_regression_ihdp, verbose=0)
    else:
      return KerasRegressor(build_fn=create_nn_regression, verbose=0)
  elif name_model == 'MeanDiff':
    return _MeanDiff()
  else:
    if data_name == 'ukb':
      model_config = {}
      model_config['weights'] = 'imagenet'
      model_config['input_shape'] = (587, 587, 3)
      if name_model == 'resnet50':
        model_config['name_base_model'] = 'resnet50'
        return image_model_construction(model_config)
      elif name_model == 'inceptionv3':
        model_config['name_base_model'] = 'inceptionv3'
        return image_model_construction(model_config)
      else:
        model_config['name_base_model'] = 'image_regression'
        return image_model_construction(model_config)

    raise NotImplementedError(
        'Estimator not supported:{}'.format(name_model))


def _metric_function(name_metric):
  if name_metric == 'mse':
    return sk_metrics.mean_squared_error
  else:
    raise NotImplementedError(
        'Estimator not supported:{}'.format(name_metric))


def _prop_score_function(name_prop_score):
  if name_prop_score == 'LogisticRegression':
    return linear_model.LogisticRegression()
  elif name_prop_score == 'prop':
    return _Prop()
  elif name_prop_score == 'LogisticRegression_NN':
    return _LogisticRegressionNN()
  else:
    raise NotImplementedError(
        'Estimator not supported:{}'.format(name_prop_score))


def _estimator_function(name_estimator):
  if name_estimator == 'oaxaca':
    return estimators.estimator_oaxaca
  elif name_estimator == 'aipw':
    return estimators.estimator_aipw
  else:
    raise NotImplementedError(
        'Estimator not supported:{}'.format(name_estimator))


def create_nn_regression_acic(h_units, l1, learning_rate=0.01):
  """Make Neural Network Model.

  It defines a two dense layers without activation, plus an output layer with
  linear activation. This architecture was defined according to preliminary
  tests.
  Args:
    h_units: hidden units
    l1: l1 regularization
    learning_rate: learning rate
  Returns:
    model: keras object
  """
  model = Sequential()
  model.add(
      Dense(h_units, use_bias=True, kernel_regularizer=regularizers.l1(l1=l1)))
  model.add(
      Dense(h_units, use_bias=True, kernel_regularizer=regularizers.l1(l1=l1)))
  model.add(Dense(1, activation='linear', use_bias=True))
  opt = optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss='mse', optimizer=opt, loss_weights=[1.])
  return model


def create_nn_regression_ihdp(h_units, l1, learning_rate=0.01):
  """Make Neural Network Model.

  Defines a two dense layers with relu, plus an output layer with linear
  activation. This architecture was defined according to preliminary
  tests.
  Args:
    h_units: hidden units
    l1: l1 regularization
    learning_rate: learning rate
  Returns:
    model: keras object
  """
  model = Sequential()
  model.add(
      Dense(h_units, use_bias=True, activation='relu',
            kernel_regularizer=regularizers.l1(l1=l1)))
  model.add(
      Dense(h_units, use_bias=True, activation='relu',
            kernel_regularizer=regularizers.l1(l1=l1)))
  model.add(Dense(1, activation='linear', use_bias=True))
  opt = optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss='mse', optimizer=opt, loss_weights=[1.])
  return model


def create_nn_regression(h_units, l1, learning_rate=0.01):
  """Make Neural Network Model.

  Defines a one layer without activation plus another dense layer with linear
  activation function. This architecture was defined according to preliminary
  tests.
  Args:
    h_units: hidden units
    l1: l1 regularization
    learning_rate: learning rate
  Returns:
    model: keras object
  """
  model = Sequential()
  model.add(
      Dense(h_units, use_bias=True, kernel_regularizer=regularizers.l1(l1=l1)))
  model.add(Dense(1, activation='linear', use_bias=True))
  opt = optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss='mse', optimizer=opt, loss_weights=[1.])
  return model


class _LogisticRegressionNN:
  """Make NN version of Logistic Regression.

  Used as Propensity Score Model
  """

  def __init__(self):
    super(_LogisticRegressionNN, self).__init__()
    self.model = self._logistic_regression_architecture()

  def fit(self, data):
    """Fits a Classification Model.

    Args:
      data: prefetch batches 16 [B, H, W, C], not repeated, not shuffled.
    """
    self.model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 20
    steps = 20

    self.model.summary()
    self.model.fit(data, epochs=epochs, steps_per_epoch=steps, verbose=2)

  def predict_proba(self, data, quick=True):
    """Predict Probability of each class.

    Args:
      data: tf.data.Dataset
      quick: subset of the images
    Returns:
      predict: predictions array
    """
    t_pred = []
    t = []
    for i, (batch_x, batch_t) in enumerate(data):
      t_pred.append(self.model.predict_on_batch(batch_x))
      t.append(batch_t.numpy())
      if quick and i > 2000:
        break

    t_pred = np.concatenate(t_pred).ravel().reshape(-1, 2)
    return t_pred

  def _logistic_regression_architecture(self):
    """Implements of Propensity Score.

    It takes as input tensors of shape [B, H, W, C] and outputs [B,Y]
    Returns:
      model: NN object
    """
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(587, 587, 3))

    # A simple logistic regression implemented as NN.
    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='LogisticRegression')
    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', use_bias=True,
                                    kernel_regularizer=regularizers.l1_l2(
                                        l1=1e-5,
                                        l2=1e-4),
                                    bias_regularizer=regularizers.l2(1e-4),
                                    activity_regularizer=regularizers.l2(1e-5)
                                    )(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='LogisticRegression')
    return model


class _Prop:
  """Make Proportion Object.

  Used as Propensity Score Model, it returns the proportion of treated.
  """

  def fit(self, x):
    self.n = x.shape[0]

  def predict_proba(self, x):
    del x
    return np.repeat(0.5, 2 * self.n).reshape(self.n, 2)


class _MeanDiff:
  """Make Difference of Mean Base Model.

  """

  def fit(self, x, y):
    self.y = y
    self.x = x

  def predict(self, x):
    return np.repeat(self.y.mean(), x.shape[0])


def image_model_regression(model_config):
  """Implements a one-layer NN that mimics a Linear Regression.

  Args:
    model_config: dictionary with parameters
  Returns:
    model: Model object
  """
  # A simple linear regression implemented as NN.
  last_activation = model_config.get('activation', 'linear')

  inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='image_regression')
  x = tf.keras.layers.Flatten()(inputs)
  outputs = tf.keras.layers.Dense(1, activation=last_activation, use_bias=True,
                                  kernel_regularizer=regularizers.l1_l2(
                                      l1=1e-5,
                                      l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(x)
  model = tf.keras.Model(
      inputs=inputs, outputs=outputs, name='image_regression')
  return model


def image_model_inceptionv3(model_config):
  """Implements inceptionV3 NN model.

  Args:
    model_config: dictionary with parameters
  Returns:
    model: Model object
  """
  last_activation = model_config.get('activation', 'linear')
  backbone = tf.keras.applications.InceptionV3(
      include_top=False,
      weights=model_config.get('weights', 'imagenet'),
      input_shape=(*IMAGE_SIZE, 3),
      pooling=model_config.get('pooling', 'avg'))

  backbone_drop_rate = model_config.get('backbone_drop_rate', 0.2)

  inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='image')
  hid = backbone(inputs)
  hid = tf.keras.layers.Dropout(backbone_drop_rate)(hid)
  outputs = tf.keras.layers.Dense(1, activation=last_activation,
                                  use_bias=True)(hid)

  model = tf.keras.Model(
      inputs=inputs, outputs=outputs, name='inceptionv3')

  return model


def image_model_resnet50(model_config):
  """Implements Resnet NN model.

  Reference:
  https://keras.io/api/applications/#usage-examples-for-image-classification-models

  Args:
    model_config: dictionary with parameters
  Returns:
    model: Model object
  """
  last_activation = model_config.get('activation', 'linear')
  backbone = tf.keras.applications.ResNet50(
      include_top=False,
      weights='imagenet',
      input_shape=(587, 587, 3)
      )

  inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='image')
  x = backbone(inputs)
  x = tf.keras.layers.Dropout(0.4)(x)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  outputs = tf.keras.layers.Dense(1, activation=last_activation,
                                  use_bias=True)(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet50')
  return model


def image_model_construction(model_config):
  """Constructs the image base model.

  Args:
    model_config: dicstionary with parameters
  Returns:
    model: Model object
  """
  name_base_model = model_config.get('name_base_model', 'inceptionv3')
  if name_base_model == 'inceptionv3':
    model = image_model_inceptionv3(model_config)
    initial_learning_rate = 0.001
  elif name_base_model == 'resnet50':
    model = image_model_resnet50(model_config)
    initial_learning_rate = 0.001
  elif name_base_model == 'image_regression':
    model = image_model_regression(model_config)
    initial_learning_rate = 0.01
  else:
    raise NotImplementedError(
        'Estimator not supported:{}'.format(name_base_model))

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate, decay_steps=30, decay_rate=0.9, staircase=True)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
      loss='mean_squared_error',
      metrics=['mse', 'mae'])
  return model
