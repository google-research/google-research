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

"""Common flags for trainer and decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS

# Flags for hparam sweeps
flags.DEFINE_float("l0_norm_weight", None, "weight for l0 norm.")
flags.DEFINE_integer("l0_weight_start", None, "weight start for l0 norm.")
flags.DEFINE_integer("l0_weight_diff", None, "weight diff for l0 norm.")
flags.DEFINE_float("dkl_weight", None, "weight for dkl norm.")
flags.DEFINE_integer("dkl_weight_start", None, "weight start for dkl norm.")
flags.DEFINE_integer("dkl_weight_diff", None, "weight diff for dkl norm.")
flags.DEFINE_string("dkl_weight_fn", None, "dkl weight curve.")
flags.DEFINE_float("target_sparsity", None, "sparsity for mp.")
flags.DEFINE_integer("begin_pruning_step", None, "start step for mp.")
flags.DEFINE_integer("end_pruning_step", None, "end step for mp.")
flags.DEFINE_integer("pruning_frequency", None, "frequency of mp steps.")
flags.DEFINE_string("regularization", None, "what regularization to use.")
flags.DEFINE_float("clip_log_alpha", None, "clip limit for log alphas.")
flags.DEFINE_integer("nbins", None, "number of bins for mp histogram.")

# For scratch-e, scratch-b, and lottery ticket experiments
flags.DEFINE_string(
    "load_masks_from",
    None,
    "Checkpoint to load trained mask from.")
flags.DEFINE_string(
    "load_weights_from",
    None,
    "Checkpoint to load trained non-mask from.")
flags.DEFINE_float(
    "initial_sparsity",
    None,
    "Initial sparsity for scratch-* experiments.")

# For constant parameter curves
flags.DEFINE_integer(
    "hidden_size",
    None,
    "Hidden size of the Transformer.")
flags.DEFINE_integer(
    "filter_size",
    None,
    "Filter size of the Transformer.")
flags.DEFINE_integer(
    "num_heads",
    None,
    "Number of heads in the Transformer.")

# For imbalanced pruning experiments
flags.DEFINE_float(
    "embedding_sparsity",
    None,
    "Sparsity fraction for embedding matrix",
)


def update_argv(argv):
  """Update the arguments."""
  if FLAGS.l0_norm_weight is not None:
    argv.append("--hp_l0_norm_weight")
    argv.append("{}".format(FLAGS.l0_norm_weight))
  if FLAGS.l0_weight_start is not None:
    argv.append("--hp_l0_weight_start")
    argv.append("{}".format(FLAGS.l0_weight_start))
  if FLAGS.l0_weight_diff is not None:
    argv.append("--hp_l0_weight_diff")
    argv.append("{}".format(FLAGS.l0_weight_diff))
  if FLAGS.dkl_weight is not None:
    argv.append("--hp_dkl_weight")
    argv.append("{}".format(FLAGS.dkl_weight))
  if FLAGS.dkl_weight_start is not None:
    argv.append("--hp_dkl_weight_start")
    argv.append("{}".format(FLAGS.dkl_weight_start))
  if FLAGS.dkl_weight_diff is not None:
    argv.append("--hp_dkl_weight_diff")
    argv.append("{}".format(FLAGS.dkl_weight_diff))
  if FLAGS.dkl_weight_fn is not None:
    argv.append("--hp_dkl_weight_fn")
    argv.append("{}".format(FLAGS.dkl_weight_fn))
  if FLAGS.target_sparsity is not None:
    argv.append("--hp_target_sparsity")
    argv.append("{}".format(FLAGS.target_sparsity))
  if FLAGS.begin_pruning_step is not None:
    argv.append("--hp_begin_pruning_step")
    argv.append("{}".format(FLAGS.begin_pruning_step))
  if FLAGS.end_pruning_step is not None:
    argv.append("--hp_end_pruning_step")
    argv.append("{}".format(FLAGS.end_pruning_step))
  if FLAGS.pruning_frequency is not None:
    argv.append("--hp_pruning_frequency")
    argv.append("{}".format(FLAGS.pruning_frequency))
  if FLAGS.regularization is not None:
    if FLAGS.regularization == "none":
      argv.append("--hp_layer_prepostprocess_dropout")
      argv.append("0.0")
      argv.append("--hp_attention_dropout")
      argv.append("0.0")
      argv.append("--hp_relu_dropout")
      argv.append("0.0")
      argv.append("--hp_label_smoothing")
      argv.append("0.0")
    elif FLAGS.regularization == "label_smoothing":
      argv.append("--hp_layer_prepostprocess_dropout")
      argv.append("0.0")
      argv.append("--hp_attention_dropout")
      argv.append("0.0")
      argv.append("--hp_relu_dropout")
      argv.append("0.0")
    elif FLAGS.regularization == "dropout+label_smoothing":
      # Don't need to do anything
      pass
    elif FLAGS.regularization == "moredropout+label_smoothing":
      # crank up the prepostprocess dropout (like transformer_big)
      argv.append("--hp_layer_prepostprocess_dropout")
      argv.append("0.3")
    elif FLAGS.regularization == "muchmoredropout+label_smoothing":
      # crank up the prepostprocess dropout a lot
      argv.append("--hp_layer_prepostprocess_dropout")
      argv.append("0.5")
    else:
      raise ValueError("Invalid value of regularization flags: {}"
                       .format(FLAGS.regularization))
  if FLAGS.clip_log_alpha is not None:
    argv.append("--hp_clip_log_alpha")
    argv.append("{}".format(FLAGS.clip_log_alpha))
  if FLAGS.nbins is not None:
    argv.append("--hp_nbins")
    argv.append("{}".format(FLAGS.nbins))
  if FLAGS.load_masks_from is not None:
    argv.append("--hp_load_masks_from")
    argv.append("{}".format(FLAGS.load_masks_from))
  if FLAGS.load_weights_from is not None:
    argv.append("--hp_load_weights_from")
    argv.append("{}".format(FLAGS.load_weights_from))
  if FLAGS.initial_sparsity is not None:
    argv.append("--hp_initial_sparsity")
    argv.append("{}".format(FLAGS.initial_sparsity))
  if FLAGS.hidden_size is not None:
    argv.append("--hp_hidden_size")
    argv.append("{}".format(FLAGS.hidden_size))
  if FLAGS.filter_size is not None:
    argv.append("--hp_filter_size")
    argv.append("{}".format(FLAGS.filter_size))
  if FLAGS.num_heads is not None:
    argv.append("--hp_num_heads")
    argv.append("{}".format(FLAGS.num_heads))
  if FLAGS.embedding_sparsity is not None:
    argv.append("--hp_embedding_sparsity")
    argv.append("{}".format(FLAGS.embedding_sparsity))
  return argv
