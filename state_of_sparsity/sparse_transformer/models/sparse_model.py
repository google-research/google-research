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

"""Base class for sparse model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import t2t_model
import tensorflow.compat.v1 as tf
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import training as contrib_training

from tensorflow.contrib.model_pruning.python import pruning as magnitude_pruning


def pruning_hparams(hparams, use_tpu, random):  # pylint: disable=unused-argument
  """Helper to get hparams for pruning library."""
  weight_sparsity_map = [""]
  if hparams.get("embedding_sparsity") >= 0.0:
    weight_sparsity_map = [
        "transformer/symbol_modality_33288_512/shared/:{}"
        .format(hparams.get("embedding_sparsity"))
    ]
    tf.logging.info(
        "Pruning embedding matrix to {}% sparsity"
        .format(hparams.get("embedding_sparsity") * 100))

  hparams = contrib_training.HParams(
      name="model_pruning",
      begin_pruning_step=hparams.get("begin_pruning_step"),
      end_pruning_step=hparams.get("end_pruning_step"),
      weight_sparsity_map=weight_sparsity_map,
      threshold_decay=hparams.get("threshold_decay"),
      pruning_frequency=hparams.get("pruning_frequency"),
      nbins=hparams.get("nbins"),
      block_height=1,
      block_width=1,
      block_pooling_function="AVG",
      initial_sparsity=0.0,  # always start at sparsity 0
      target_sparsity=hparams.get("target_sparsity"),
      sparsity_function_begin_step=hparams.get("begin_pruning_step"),
      sparsity_function_end_step=hparams.get("end_pruning_step"),
      sparsity_function_exponent=hparams.get("sparsity_function_exponent"),
      use_tpu=use_tpu)
  # TODO(tgale): Fix the need to keep this commented out.
  # random pruning currently does not work.
  # random=random)
  return hparams


def check_global_sparsity():
  """Add a summary for the weight sparsity."""
  weight_masks = magnitude_pruning.get_masks()
  weights_per_layer = []
  nonzero_per_layer = []
  for mask in weight_masks:
    nonzero_per_layer.append(tf.reduce_sum(mask))
    weights_per_layer.append(tf.size(mask))
    total_nonzero = tf.add_n(nonzero_per_layer)
    total_weights = tf.add_n(weights_per_layer)
  sparsity = (1.0 - (tf.cast(total_nonzero, tf.float32) /
                     tf.cast(total_weights, tf.float32)))
  tf.summary.scalar("global_weight_sparsity", sparsity)


class SparseModel(t2t_model.T2TModel):
  """T2T model with weight sparsity."""

  def initialize_masks_from_ckpt(self, checkpoint):
    model_dir = self._hparams.get("model_dir", None)
    already_has_ckpt = (
        model_dir and tf.train.latest_checkpoint(model_dir) is not None)
    if already_has_ckpt:
      tf.logging.info("Checkpoint exists in model_dir, not loading variables.")
      return

    # Create a list of mask variables to load
    reader = tf.train.NewCheckpointReader(checkpoint)
    mask_names = reader.get_variable_to_shape_map().keys()
    mask_names = [x for x in mask_names if x.endswith("mask")]

    variable_map = {}
    for var in tf.global_variables():
      var_name = var.name.split(":")[0]
      if var_name in mask_names:
        tf.logging.info("Loading mask variable from checkpoint: %s", var_name)
        variable_map[var_name] = var
      elif "mask" in var_name:
        tf.logging.info(
            "Cannot find mask variable in checkpoint, skipping: %s", var_name)
    tf.train.init_from_checkpoint(checkpoint, variable_map)

  def initialize_non_masks_from_ckpt(self, checkpoint):
    model_dir = self._hparams.get("model_dir", None)
    already_has_ckpt = (
        model_dir and tf.train.latest_checkpoint(model_dir) is not None)
    if already_has_ckpt:
      tf.logging.info("Checkpoint exists in model_dir, not loading variables.")
      return

    # Create a list of non-mask variables to load
    reader = tf.train.NewCheckpointReader(checkpoint)
    non_mask_names = reader.get_variable_to_shape_map().keys()
    non_mask_names = [x for x in non_mask_names if not x.endswith("mask")]

    variable_map = {}
    for var in tf.global_variables():
      var_name = var.name.split(":")[0]
      if var_name in non_mask_names:
        tf.logging.info(
            "Loading non-mask variable from checkpoint: %s", var_name)
        variable_map[var_name] = var
      elif "mask" not in var_name:
        tf.logging.info(
            "Cannot find non-mask variable in checkpoint, skipping: %s",
            var_name)
    tf.train.init_from_checkpoint(checkpoint, variable_map)

  def estimator_spec_train(self, loss, num_async_replicas=1, use_tpu=False):
    """Constructs `tf.estimator.EstimatorSpec` for TRAIN (training) mode."""
    train_op = self.optimize(
        loss,
        num_async_replicas=num_async_replicas,
        use_tpu=use_tpu)

    sparsity_technique = self._hparams.get("sparsity_technique")
    if "pruning" in sparsity_technique:
      if not self._hparams.load_masks_from:
        # If we are loading trained masks, don't add the mask update
        # step to the training process and keep the masks static
        with tf.control_dependencies([train_op]):
          mp_hparams = pruning_hparams(
              self._hparams,
              use_tpu,
              sparsity_technique == "random_pruning")
          p = magnitude_pruning.Pruning(
              mp_hparams,
              global_step=tf.train.get_global_step())
          mask_update_op = p.conditional_mask_update_op()
          train_op = mask_update_op
      check_global_sparsity()

    if use_tpu:
      if self._hparams.warm_start_from:
        def scaffold_fn():
          self.initialize_from_ckpt(
              self._hparams.warm_start_from)
          return tf.train.Scaffold()
      elif self._hparams.load_masks_from and self._hparams.load_weights_from:
        def scaffold_fn():
          self.initialize_masks_from_ckpt(
              self._hparams.load_masks_from)
          self.initialize_non_masks_from_ckpt(
              self._hparams.load_weights_from)
          return tf.train.Scaffold()
      elif self._hparams.load_masks_from:
        def scaffold_fn():
          self.initialize_masks_from_ckpt(
              self._hparams.load_masks_from)
          return tf.train.Scaffold()
      else:
        scaffold_fn = None

      # Note: important to call this before remove_summaries()
      if self.hparams.tpu_enable_host_call:
        host_call = t2t_model.create_host_call(self.hparams.model_dir)
      else:
        host_call = None

      t2t_model.remove_summaries()

      return contrib_tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.TRAIN,
          loss=loss,
          train_op=train_op,
          host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      if self._hparams.warm_start_from:
        self.initialize_from_ckpt(
            self._hparams.warm_start_from)
      elif self._hparams.load_masks_from:
        self.initialize_masks_from_ckpt(
            self._hparams.load_masks_from)

      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.TRAIN,
          loss=loss,
          train_op=train_op)
