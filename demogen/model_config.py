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

"""Utilities for loading the model dataset.

This file contains all necessary utilities for loading a model
from the margin distribution generalization dataset.

  Typical usage example:

  root_dir = # directory where the model dataset is at
  model_config = ModelConfig()
  model_path = model_config.get_model_dir_name(root_dir)
  model_fn = model_config.get_model_fn()

  sess = tf.Session()
  image = # input tensor
  logits = model_fn(image, False)
  model_config.load_parameters(model_path, sess)
"""

import os

import tensorflow as tf
from demogen.models.get_model import get_model


CKPT_NAME = 'model.ckpt-150000'
# all available settings in the dataset
ALL_MODEL_SPEC = {
    'NIN_CIFAR10': {
        'wide_multiplier': [1.0, 1.5, 2.0],
        'batchnorm': [True, False],
        'dropout_prob': [0.0, 0.2, 0.5],
        'augmentation': [True, False],
        'decay_fac': [0.0, 0.001, 0.005],
        'copy': [1, 2]
    },
    'RESNET_CIFAR10': {
        'wide_multiplier': [1.0, 2.0, 4.0],
        'normalization': ['batch', 'group'],
        'augmentation': [True, False],
        'decay_fac': [0.0, 0.02, 0.002],
        'learning_rate': [0.01, 0.001],
        'copy': [1, 2, 3]
    },
    'RESNET_CIFAR100': {
        'wide_multiplier': [1.0, 2.0, 4.0],
        'normalization': ['batch', 'group'],
        'augmentation': [True, False],
        'decay_fac': [0.0, 0.02, 0.002],
        'learning_rate': [0.1, 0.01, 0.001],
        'copy': [1, 2, 3]
    }
}


class ModelConfig(object):
  """A class for easy use of the margin distribution model dataset.

  A model config contains all the relevant information for building a model
  in the margin distribution model dataset. Some attributes only apply
  for specific architecture and changing for the other architecture does
  not have any effects.

  Attributes:
    model_type: The overall topology of the model.
    dataset: The name of the dataset the model uses.
    wide_multiplier: How much wider is the model compared to the base model.
    dropout_prob: Probability of dropping random unit (only for nin).
    augmentation: If data augmentation is used.
    decay_fac: Coefficient for l2 weight decay.
    batchnorm: If batchnorm is used (only for nin).
    normalization: Type of normalization (only for resnet).
    learning_rate: Initial learning rate (only for resnet).
    copy: Index of the copy.
    num_class: Number of classes in the dataset.
    data_format: What kind of data the model expects (e.g. channel first/last).
  """

  def __init__(
      self,
      model_type='nin',
      dataset='cifar10',
      wide_multiplier=1.0,
      batchnorm=False,
      dropout_prob=0.0,
      data_augmentation=False,
      l2_decay_factor=0.0,
      normalization='batch',
      learning_rate=0.01,
      copy=1):

    assert model_type == 'nin' or model_type == 'resnet'
    assert dataset == 'cifar10' or dataset == 'cifar100'
    experiment_type = (model_type + '_' + dataset).upper()
    candidate_params = ALL_MODEL_SPEC[experiment_type]
    assert wide_multiplier in candidate_params['wide_multiplier']
    assert l2_decay_factor in candidate_params['decay_fac']
    assert copy in candidate_params['copy']
    self.model_type = model_type
    self.dataset = dataset
    self.wide_multiplier = wide_multiplier
    self.dropout_prob = dropout_prob
    self.augmentation = data_augmentation
    self.decay_fac = l2_decay_factor
    self.batchnorm = batchnorm
    self.normalization = normalization
    self.learning_rate = learning_rate
    self.copy = copy
    self.num_class = 10 if dataset == 'cifar10' else 100
    self.data_format = 'HWC' if model_type == 'nin' else 'CHW'

  def get_model_dir_name(self, root_dir):
    """Get the name of the trained model's directory.

    Generates the name of the directory that contain a trained model
    specified by the ModelConfig object. The name of the directory is
    generaly indicative of the hyperparameter settings of the model.

    Args:
      root_dir: Root directory where experiment directory is located.

    Returns:
      A string that contains the checkpoint containing weights and
      training/test accuracy of a model.

    Raises:
      ValueError: The model type is not in the dataset
    """
    if self.model_type == 'nin':
      data_dir = 'NIN_'
      data_dir += self.dataset
      model_parent_dir = os.path.join(root_dir, data_dir.upper())
      model_path = [
          self.model_type,
          'wide_{}x'.format(self.wide_multiplier),
          'bn' if self.batchnorm else '',
          'dropout_{}'.format(self.dropout_prob),
          'aug' if self.augmentation else '',
          'decay_{}'.format(self.decay_fac),
          str(self.copy)
      ]
      model_dir = os.path.join(model_parent_dir, '_'.join(model_path))
    elif self.model_type == 'resnet':
      data_dir = 'RESNET_'
      data_dir += self.dataset
      model_parent_dir = os.path.join(root_dir, data_dir.upper())
      model_path = [
          self.model_type,
          'wide_{}x'.format(self.wide_multiplier),
          '{}norm'.format(self.normalization),
          'aug' if self.augmentation else '',
          'decay_{}'.format(self.decay_fac)
      ]
      if self.learning_rate != 0.01:
        model_path.append('lr_{}'.format(self.learning_rate))
      model_path.append(str(self.copy))
      model_dir = os.path.join(model_parent_dir, '_'.join(model_path))
    else:
      raise ValueError('model type {} is not available'.format(self.model_type))
    return os.path.join(model_dir, CKPT_NAME)

  def get_model_fn(self):
    """Get a model function of the model specified by a model configuration.

    Generates a callable function that can build a model specified
    by self. The function is meant to be called on tensors of
    input images.

    Returns:
      A callable model function built according to the hyper parameters of self.
    """
    config = tf.contrib.training.HParams(
        wide=self.wide_multiplier,
        dropout=self.dropout_prob,
        batchnorm=self.batchnorm,
        weight_decay=True,
        decay_fac=self.decay_fac,
        normalization=self.normalization,
        num_class=self.num_class,
        spatial_dropout=False,
    )
    return get_model(self.model_type, config)

  def load_parameters(self, model_dir, tf_session):
    """Load trained parameter from checkpoint into the model.

    Load model parameters from model_dir into the appropriate variables.
    This should be called after the model variables are created which is
    done by calling the model function produced by get_model_fn.

    Args:
      model_dir: Directory where the model checkpoint resides.
      tf_session: A tf session where the model is located

    Returns:
      A callable model function built according to the hyper parameters of self.
    """
    model_var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_type
    )
    saver = tf.train.Saver(model_var_list)
    saver.restore(tf_session, model_dir)
