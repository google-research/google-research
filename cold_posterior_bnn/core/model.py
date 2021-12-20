# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utility methods to handle the creation of tf.keras.models.Model instances.

We use a number of custom classes with tf.keras.models.Model, and when cloning
models we need to make sure Keras is aware of all our classes in order to
serialize and deserialize them properly.  This file contains utility methods to
this end.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from cold_posterior_bnn.core import prior


def bnn_scope():
  """Create a scope that is aware of BNN library objects.

  Returns:
    scope: tf.keras.utils.CustomObjectScope with object/class mapping.
  """
  scope_dict = {
      'NormalRegularizer': prior.NormalRegularizer,
      'StretchedNormalRegularizer': prior.StretchedNormalRegularizer,
      'HeNormalRegularizer': prior.HeNormalRegularizer,
      'GlorotNormalRegularizer': prior.GlorotNormalRegularizer,
      'LaplaceRegularizer': prior.LaplaceRegularizer,
      'CauchyRegularizer': prior.CauchyRegularizer,
      'SpikeAndSlabRegularizer': prior.SpikeAndSlabRegularizer,
      'EmpiricalBayesNormal': prior.EmpiricalBayesNormal,
      'HeNormalEBRegularizer': prior.HeNormalEBRegularizer,
      'ShiftedNormalRegularizer': prior.ShiftedNormalRegularizer,
  }
  scope_dict.update(tf.keras.utils.get_custom_objects())
  scope = tf.keras.utils.CustomObjectScope(scope_dict)

  return scope


def clone_model(model):
  """Clone a model.

  We add information necessary to serialize/deserialize the `bnn` classes.

  Args:
    model: tf.keras.models.Model to be cloned.

  Returns:
    model_cloned: tf.keras.models.Model having the same structure.
  """
  with bnn_scope():
    if isinstance(model, tf.keras.Sequential) or issubclass(tf.keras.Model,
                                                            type(model)):
      model_cloned = tf.keras.models.clone_model(model)
    elif isinstance(model, tf.keras.Model):

      model_cloned = model.__class__.from_config(model.get_config())
    else:
      raise ValueError('Unknown model type, cannot clone.')

    return model_cloned


def clone_model_and_weights(model, input_shape):
  """Clone a model including weights.

  The model will be build with the given input_shape.

  Args:
    model: tf.keras.models.Model to be cloned.
    input_shape: same parameter as in tf.keras.models.Model.build.
      For example, for MNIST this would typically be (1,784).

  Returns:
    model_cloned: tf.keras.models.Model having the same structure and weights.
  """
  model_cloned = clone_model(model)
  model_cloned.build(input_shape)
  model_cloned.set_weights(model.get_weights())

  return model_cloned

