# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Architecture utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ieg import utils
from ieg.models.custom_ops import decay_weights

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class StrategyNetBase(object):
  """Base class for supporting model training under tf.distribute.strategy."""

  def __init__(self):

    self.replica_updates_set = []
    self.regularization_losses_set = []
    self.created = False

  def count_parameters(self, name):
    """Count parameters.

    Args:
      name: variable scope name
    """
    total_parameters = 0
    for t_var in tf.trainable_variables(scope=name):
      # shape is an array of tf.Dimension
      shape = t_var.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      # print(variable.name, variable_parameters)
      total_parameters += variable_parameters

    tf.logging.info('Total number of parameters (M) {}'.format(
        total_parameters / 1000000.0))

  def get_update_ops(self, name, var_refer):
    """Returns batchnorm update ops.

    Args:
      name: name of variable
      var_refer: var reference to check if it is copies from different devices

    Returns:
      a list of update variables
    """
    scope_name = var_refer.name[:var_refer.name.index(name) + len(name) + 1]
    all_updates = []
    for update in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
      # MetaImage will copy ops by adding <scope>_<1/2/3>/ to the name,
      # we want to exclude them.
      # The design is only for meta learning design
      if name + '_' not in update.name:
        all_updates.append(update)
    name = all_updates[0].name if all_updates else ''
    tf.logging.info('Get scope: {} {} updates (e.g. {})'.format(
        scope_name, len(all_updates), name))

    return all_updates

  def get_regularization_loss(self, scope_name, name='regularization_loss'):
    losses = self.get_regularization_losses(scope_name)

    return tf.add_n(losses, name=name)

  def get_regularization_losses(self, scope_name):
    losses = tf.losses.get_regularization_losses(scope=scope_name)
    tf.logging.info('Get {} regularization losses (e.g. {})'.format(
        len(losses), losses[0].name))
    return losses

  def convert_var_type(self, variables, dtype):
    for var in variables:
      var = tf.cast(var, dtype)

  def init(self, name, with_name=None, outputs=None):
    """Init model class.

    Args:
      name: name of model scope
      with_name: variable with prefix_name
      outputs: an output variable of the model
    """
    self.name = name
    self.updates = self.get_update_ops(name, outputs)
    assert self.updates

    self.trainable_variables = utils.get_var(tf.trainable_variables(), name)
    # batch_norm variables (moving mean and moving average)
    self.updates_variables = [
        a for a in utils.get_var(
            tf.global_variables(), name, with_name=with_name)
        if a not in self.trainable_variables
    ]
    self.total_vars = len(self.trainable_variables) + len(
        self.updates_variables)

    tf.logging.info(
        '[StrategyNetBase] {}: Find {} trainable_variables and {} update_variables {} updates'
        .format(name, len(self.trainable_variables),
                len(self.updates_variables), len(self.updates)))

    self.created = True


class MetaImage(object):
  """Meta network mirroring image class.

  This is used to support meta learning that works with
  arbitrary tensorflow models architectures.
  It uses custom_getter function to replace variabels of models without the need
  to build new graph.
  """

  def __init__(self, net, name):
    tf.logging.info('Create meta image {}'.format(net.__class__.__name__))
    self.trainable_variables = []
    self.updates_variables = []
    self.variables = {}  # variable index
    self.net = net
    self.name = name

  def verbose(self):
    tf.logging.info(
        '[Meta Image] {} variables and {} update_variables are mirrored'.format(
            len(self.trainable_variables), len(self.updates_variables)))

  def get_regularization_loss(self, decay_rate):
    tf.logging.info(
        '[Meta Image] add regularization decay {} to {} variables'.format(
            decay_rate, len(self.trainable_variables)))
    return decay_weights(decay_rate, self.trainable_variables)

  def add_variable_alias(self, var, var_name, var_type='trainable_variables'):
    """Add variable alias.

    Args:
      var: a single variable, e.g. layer weights
      var_name: name of the variable
      var_type: either trainable_variables or updates_variables (for BN)
    """
    self.variables[var_name] = var
    if var_type == 'trainable_variables':
      self.trainable_variables.append(var)
    elif var_type == 'updates_variables':
      self.updates_variables.append(var)

  def metaweight_getter(self, getter, name, *args, **kwargs):
    """Getter function for variable_scope.

    See
    https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/get_variable.

    Args:
      getter: original getter function.
      name: name of variable to getter
      *args: args
      **kwargs: kwargs

    Returns:
      the variable in self.variables
    """
    var = getter(name, *args, **kwargs)
    meta_var = self.variables[var.name]
    return meta_var

  def __call__(self, inputs, name, training=True, reuse=True):
    return self.net(
        inputs,
        self.net.name,
        training=training,
        reuse=reuse,
        custom_getter=self.metaweight_getter)
