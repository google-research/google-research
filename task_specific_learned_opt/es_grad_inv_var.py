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

"""Evolutionary strategies + reparameterization gradient based trainer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import base_trainer as base
import common_trainer as common
import custom_getters
#### NOTINCLUDED
# This is a internal tensorflow database of sorts written for this project.
# This is used to place gradients, as well as fetching graphs.
import data_store
# NOTINCLUDED
import gin
import py_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.framework import tensor_spec
import truncation_strategy

nest = tf.contrib.framework.nest

# Named tuple representing state for a gradient update.
# These values are going to either single (nested) tensors, or batches of
# (nested) tensors.
ESGradData = collections.namedtuple(
    "ESGradData", ["grads", "perturbation", "meta_loss", "antith_meta_loss"])


@gin.configurable
class ESGradInvVarTrainer(base.TruncatedTrainer):
  """Meta-optimization with gradient estimation done via antithetical ES."""

  def __init__(
      self,
      batch_size=128,
      truncation_strategy_fn=truncation_strategy.DefaultTruncationStrategy,
      meta_opt_fn=lambda: tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5),
      name="ESAntithTrainer",
      independent_sample=False,
      gradient_clip_by_value=0.1,
      **kwargs):
    super(ESGradInvVarTrainer, self).__init__(name=name, **kwargs)
    self.batch_size = batch_size
    self.gradient_clip_by_value = gradient_clip_by_value

    # Even if independent, its not truely independent as they share the
    # initial conditions..
    self.independent_sample = independent_sample

    with self._enter_variable_scope():
      self.meta_opt = meta_opt_fn()
      self.truncation_strategy = truncation_strategy_fn()

      # Construct the datastore
      self.ds = data_store.AsyncBatchDataStore(
          batch_size=self.batch_size,
          buffer_size=self.batch_size * 5,
          staleness=-1,  # off,
          sync_updates=False,
          local_device=self.local_device,
          remote_device=self.remote_device,
          index_remote_device=self.index_remote_device,
          name="ESGradTrainer/indexdatastore",
      )

      with tf.device(self.remote_device):
        self.failed_push = tf.get_variable(
            name="failed_push",
            dtype=tf.int32,
            shape=[],
            initializer=tf.zeros_initializer())

      self.custom_getter = custom_getters.ESCustomGetter()
      self.custom_getter_independent = custom_getters.ESCustomGetter()

  def get_saved_remote_variables(self):
    saved_remote_vars = [self.failed_push]
    saved_remote_vars += self.ds.get_saveable_variables()
    saved_remote_vars += self.meta_opt.variables()
    saved_remote_vars += [self.accumulate_var]
    return saved_remote_vars

  def get_not_saved_remote_variables(self):
    return self.ds.get_not_saveable_variables()

  def worker_compute_op(self):
    # Compute the meta loss
    unroll_n_steps = self.truncation_strategy.unroll_n_steps(self.learner)

    loss_state_fn = common.DeterministicMetaLossEvaluator(
        unroll_n_steps=unroll_n_steps,
        inner_loss_state_fn=self.learner.inner_loss_state,
        meta_loss_state_fn=self.learner.meta_loss_state,
    )

    with self.learner.theta_mod.custom_getter.use_getter(self.custom_getter):
      meta_loss, final_state = loss_state_fn(self.learner)

      with self.custom_getter.antithetic_sample():
        antith_meta_loss, _ = loss_state_fn(self.learner)

    total_meta_loss = meta_loss + antith_meta_loss

    # compute gradient of meta loss
    if self.independent_sample:
      with self.learner.theta_mod.custom_getter.use_getter(
          self.custom_getter_independent):
        grad_meta_loss, _ = loss_state_fn(self.learner)
    else:
      grad_meta_loss = total_meta_loss

    # combine all pieces needed for outer-trainer

    perturbation = self.custom_getter.get_perturbations(
        self.learner.theta_mod.get_variables())

    trainable_theta_vars = self.learner.theta_mod.get_variables()
    grads = tf.gradients(grad_meta_loss, trainable_theta_vars)

    to_push = ESGradData(
        grads=grads,
        perturbation=perturbation,
        meta_loss=meta_loss,
        antith_meta_loss=antith_meta_loss)

    # push to parameter server

    with tf.device(self.remote_device):
      with self._enter_variable_scope():
        self.ds.setup_memory_from_spec(
            nest.map_structure(tensor_spec.TensorSpec.from_tensor, to_push))

    should_push = tf.constant(True)

    pre_step_index = self.ds.get_step_index()
    worker_compute_op = common.make_push_op(self.learner, self.ds,
                                            self.failed_push,
                                            should_push,
                                            to_push, final_state,
                                            pre_step_index)

    return worker_compute_op

  def maybe_train_op(self):
    """If there are enough examples, run the training op.

    Otherwise do nothing.
    """
    default_fn = tf.no_op
    maybe_train_op = self.ds.take_tensors_and_perform(self.train_op, default_fn)
    return tf.group(maybe_train_op, name="maybe_train_op")

  def should_do_truncation_op(self):
    return self.truncation_strategy.should_do_truncation(self.learner)

  def ps_was_reset_op(self):
    return self.ds.clear_memory_and_index()

  def init_learner_state(self):
    learner_init_op = tf.initialize_variables(
        self.learner.learner.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))
    local_inits = tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP)
    with tf.control_dependencies(local_inits + [learner_init_op]):
      return self.learner.assign_state(self.learner.initial_state())

  def _did_use_getter_on_all_variables(self):
    # Sanity check to ensure that all variables where fetched with get_variables
    # pylint: disable=protected-access
    num_noise = len(self.custom_getter._common_noise)
    num_vars = len(self.learner.theta_mod.get_variables())
    # pylint: disable=bad-continuation
    assert num_noise == num_vars, ("noise not equal to vars (%d, %d)" % (
        num_noise, num_vars) + "\n This probably means that you cached a " +
        "instead of calling get_variable for theta_mod.")

  def train_op(self, ds_state):
    """Train with ES + Grads."""

    perturbs = ds_state.perturbation
    rp_grads = ds_state.grads
    meta_loss = ds_state.meta_loss
    antith_meta_loss = ds_state.antith_meta_loss

    # convert the [bs] shaped tensors to something like [bs, 1, 1, ...].
    broadcast_loss = [
        tf.reshape(meta_loss, [-1] + [1] * (len(p.shape.as_list()) - 1))
        for p in perturbs
    ]
    broadcast_antith_loss = [
        tf.reshape(antith_meta_loss, [-1] + [1] * (len(p.shape.as_list()) - 1))
        for p in perturbs
    ]

    # ES gradient:
    # f(x+s) * d/ds(log(p(s))) = f(x+s) * s / (std**2)
    # for antith:
    # (f(x+s) - f(x-s))*s/(2 * std**2)
    es_grads = []
    for pos_loss, neg_loss, perturb in py_utils.eqzip(broadcast_loss,
                                                      broadcast_antith_loss,
                                                      perturbs):
      # this is the same as having 2 samples.
      es_grads.append(
          (pos_loss - neg_loss) * perturb / (self.custom_getter.std**2))

    def mean_and_var(g):
      mean = tf.reduce_mean(g, axis=0, keep_dims=True)
      square_sum = tf.reduce_sum(tf.square((g - mean)), axis=0)
      var = square_sum / (g.shape.as_list()[0] - 1)
      return tf.squeeze(mean, 0), var + 1e-8

    def combine(es, rp):
      """Do inverse variance rescaling."""
      mean_es, var_es = mean_and_var(es)
      mean_rp, var_rp = mean_and_var(rp)

      es_var_inv = 1. / var_es
      rp_var_inv = 1. / var_rp

      den = es_var_inv + rp_var_inv
      combine_g = (mean_es * es_var_inv + mean_rp * rp_var_inv) / den

      weight_es = es_var_inv / den

      return combine_g, weight_es

    combine_grads, _ = zip(
        *[combine(es, rp) for es, rp in py_utils.eqzip(es_grads, rp_grads)])

    grads_vars = py_utils.eqzip(combine_grads,
                                self.learner.theta_mod.get_variables())

    grads_vars = common.clip_grads_vars(grads_vars, self.gradient_clip_by_value)
    grads_vars = common.assert_grads_vars_not_nan(grads_vars)

    self._did_use_getter_on_all_variables()

    with tf.device(self.remote_device):
      train_op = self.meta_opt.apply_gradients(grads_vars)

    with tf.control_dependencies([train_op]):
      op = common.assert_post_update_not_nan(grads_vars)
      return tf.group(train_op, op, name="train_op")

  def get_local_variables(self):
    return list(
        self.truncation_strategy.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))
