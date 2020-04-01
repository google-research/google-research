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

"""A component independent MLP optimizer.

This optimizer is a function of various useful features from the optimization
community such as RMS averages, momentum, and weight values and whose outputs
are updates to the base model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import base_arch
import device_utils
import gin
import py_utils
import sonnet as snt
import tensorflow.compat.v1 as tf
import utils_arch as utils

nest = tf.contrib.framework.nest

LearnerState = collections.namedtuple("LearnerState", [
    "phi_var_dict",
    "training_step",
    "initial_loss",
    "rolling_features",
])


@gin.configurable("FastRollingMLPLearner")
class Learner(base_arch.BaseLearner):
  """Encapsulate a single learning step.

  The learned optimizer architecture is split between this class and the
  corresponding ThetaMod.
  """

  def __init__(self, name="FastRollingMLPLearner", **kwargs):
    super(Learner, self).__init__(name=name, **kwargs)

    with self._enter_variable_scope():
      self.rolling_features = utils.RollingFeatures(
          decays=[0.5, 0.9, 0.99, 0.999, 0.9999], include_rms=False)

  @snt.reuse_variables
  def loss_and_next_state(self, current_state, loss_state=None):
    params = current_state.phi_var_dict
    loss = self.inner_loss(current_state, loss_state)

    grads_dict = self.loss_module.gradients(loss, params)
    grads = grads_dict.values()

    next_rolling_features = self.rolling_features.next_state(
        current_state.rolling_features, grads)

    next_phi_vars = \
      self.theta_mod.compute_update_and_next_state(next_rolling_features,
                                                   py_utils.eqzip(grads, params.values()),
                                                   current_state.training_step)

    new_phi_var_dict = collections.OrderedDict(
        py_utils.eqzip(current_state.phi_var_dict.keys(), next_phi_vars))

    next_state = LearnerState(
        phi_var_dict=new_phi_var_dict,
        rolling_features=next_rolling_features,
        training_step=current_state.training_step + 1,
        initial_loss=current_state.initial_loss)

    next_state = nest.map_structure(tf.identity, next_state)

    return loss, next_state

  @snt.reuse_variables
  def initial_state(self):
    """Initial state for a learning process."""
    value_dict = self.loss_module.initial_state()
    shapes = [v.shape.as_list() for v in value_dict.values()]
    b = super(Learner, self).initial_state()
    return base_arch.merged_namedtuple(
        LearnerState,
        b,
        rolling_features=self.rolling_features.initial_state(shapes))

  @snt.reuse_variables
  def current_state(self):
    """State stored on local tf.Variable for this Learner."""
    var_dict = self.loss_module.current_state()
    shapes = [v.shape.as_list() for v in var_dict.values()]
    b = super(Learner, self).current_state()
    return base_arch.merged_namedtuple(
        LearnerState,
        b,
        rolling_features=self.rolling_features.current_state(shapes),
    )


@gin.configurable("FastRollingMLPThetaMod")
class ThetaMod(base_arch.BaseThetaMod):
  """Variables shared across workers.

  This sonnet module contains the computation, and variables for the learned
  optimizer.
  See the Learner for more info about the rms values below.
  """

  def __init__(self,
               name="ThetaModStatelessMultiFeatures",
               step_multiplier=0.001,
               magnitude_rate=0.001,
               hidden_size=32,
               hidden_layer=1,
               **kwargs):
    """Initializer.

    Args:
      name: str
      step_multiplier: float multipler on step length to control how large
        initial steps are.
      magnitude_rate: float multipler on log step length to control how large
        initial steps are.
      hidden_size: int
      hidden_layer: int
      **kwargs: other kwargs
    """
    super(ThetaMod, self).__init__(name=name, **kwargs)
    self.step_multiplier = step_multiplier
    self.magnitude_rate = magnitude_rate
    self.hidden_size = hidden_size
    self.hidden_layer = hidden_layer

  @device_utils.tf_device_wrap
  @snt.reuse_variables
  def compute_update_and_next_state(self, rolling_features, grads_and_vars,
                                    training_step):
    new_vars = []

    normalizer = utils.SecondMomentNormalizer(name="Normalizer")
    mod = snt.nets.MLP([self.hidden_size] * self.hidden_layer + [2], name="MLP")

    for (g, v), m, rms in py_utils.eqzip(grads_and_vars,
                                         rolling_features.ms,
                                         rolling_features.rms):

      def do_update(g, flat_v, m, rms):
        """Do a single tensor's update."""
        flat_g = tf.reshape(g, [-1, 1])

        rsqrt = tf.rsqrt(rms + 1e-6)
        norm_g = m * rsqrt

        inp = tf.concat([flat_g, norm_g, flat_v, m, rms, rsqrt], 1)

        inp = normalizer(inp, is_training=True)

        step = utils.tanh_embedding(training_step)
        stack_step = tf.tile(
            tf.reshape(step, [1, -1]), tf.stack([tf.shape(flat_g)[0], 1]))

        inp = tf.concat([inp, stack_step], axis=1)

        output = mod(inp)

        direction = output[:, 0:1]
        magnitude = output[:, 1:2]

        step = direction * tf.exp(
            magnitude * self.magnitude_rate) * self.step_multiplier

        new_flat_v = flat_v - step
        return new_flat_v,

      flat_v = tf.reshape(v, [-1, 1])
      if isinstance(g, tf.IndexedSlices):
        new_flat_v, = utils.indexed_slices_apply_dense2(do_update,
                                                        v.shape.as_list(), g,
                                                        [flat_v, m, rms], 1)
      else:
        new_flat_v, = do_update(g, flat_v, m, rms)

      new_vars.append(tf.reshape(new_flat_v, v.shape))

    return new_vars


@gin.configurable
def fast_rolling_mlp_learner(loss_module, remote_device=""):
  with tf.device(remote_device):
    theta_mod = ThetaMod(device=remote_device)

  learner = Learner(loss_module=loss_module, theta_mod=theta_mod)

  return learner, theta_mod
