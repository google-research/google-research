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

"""Interface to Keras layers, that extracts the activated trainable variables."""

from lingvo import compat as tf
from lingvo.core import base_layer


class KerasInterfaceLayer(base_layer.BaseLayer):
  """Interface to Keras layers.

  Allows for adding variables created in the Keras layers to BabelFish's tables
  of trainable variables. Also allows declaration and adding of other variable.
  """

  def __init__(self, params):
    super().__init__(params)
    self._variables = []
    self._activated_var_names = set()

  def AddVariable(self, variable, input_shape=None, keras_scope=None):
    self._variables.append(variable)
    if "keras.layers" in variable.__repr__():
      if keras_scope:
        with tf.variable_scope(keras_scope):
          variable.build(input_shape=input_shape)
      else:
        variable.build(input_shape=input_shape)
      tf.logging.info("Built keras variable {} with variables {}".format(
          variable, variable.trainable_variables))
      self.__AddKerasVariable(variable)
    else:
      self.__AddVariable(variable)
    return variable

  @property
  def activated_var_names(self):
    return self._activated_var_names

  @property
  def trainable_variables(self):
    return self._private_vars

  def get_name(self, var):
    return var.name.split(":")[0]

  def __AddVariable(self, var):
    name = self.get_name(var)
    if name in self._activated_var_names:
      tf.logging.info(
          "Warning, already activated variable with name {}".format(name))
      return
    tf.logging.info("Adding variable {}".format(name))
    self._private_vars[name] = var
    self._private_theta[name] = tf.identity(var)
    self._activated_var_names.add(name)

  def __AddKerasVariable(self, activated_keras_layer):
    for var in activated_keras_layer.trainable_variables:
      self.__AddVariable(var)
