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

# Copyright 2022 The Google Research Authors.
"""Training script for gradient-based hyper-parameter tuning in JAX."""

import functools
import json
import os
import textwrap
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import serialization
from flax.deprecated import nn
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
import models
import sentencepiece as spm
import tensorflow.compat.v2 as tf
import utils
# pylint: disable=g-import-not-at-top
try:
  import data
  import guided_parameters
except ImportError:
  from gradient_based_tuning import data
  from gradient_based_tuning import guided_parameters
# pylint: enable=g-import-not-at-top

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')

flags.DEFINE_string(
    'vocab_path',
    default=None,
    help='Path to load or store word piece vocab file.',
    required=True)

flags.DEFINE_integer(
    'jax_random_seed', default=0, help='Integer for PRNG random seed.')

flags.DEFINE_string(
    'jax_backend_target',
    default=None,
    help=('TPU grpc target for use with cloud TPUs.'
          ' e.g. grpc://192.168.0.2:8470'))

# Training flags.
flags.DEFINE_string(
    'training_dataset',
    default='lang8',
    help='Which dataset to use for training.')

flags.DEFINE_string(
    'training_dataset_path',
    default=None,
    help='Path to prepacked TFRecords for training.',
    required=True)

flags.DEFINE_string(
    'guidance_dataset_path',
    default='',
    help='Path to prepacked TFRecords for guidance.')

flags.DEFINE_string(
    'eval_dataset_path',
    default='',
    help='Path to prepacked TFRecords for evaluation.')

flags.DEFINE_integer(
    'batch_size', default=256, help='Per host batch size for training.')

flags.DEFINE_integer(
    'num_train_steps',
    default=160000,
    help='Number of train steps. Total train '
    'steps = min(num_train_steps, max_train_epochs * <steps per epoch>).')

flags.DEFINE_integer(
    'max_train_epochs',
    default=100,
    help='Maximum bound of how many epochs to train, will cut training short if'
    ' this number is reached before num_train_steps.')

flags.DEFINE_float('learning_rate', default=0.4, help='Base learning rate.')

flags.DEFINE_string(
    'learning_rate_schedule',
    default='constant * linear_warmup * rsqrt_decay',
    help='Schedule for learning rate of model.')

flags.DEFINE_float(
    'warmup_steps_ratio',
    default=0.1,
    help='Proportion of FLAGS.num_train_steps to spend on warmup. '
    'Overridden by FLAGS.warmup_steps.')

flags.DEFINE_integer(
    'warmup_steps',
    default=0,
    help='How many steps to run linear learning rate warmup. If non-zero, '
    'overrides warmup_steps_ratio.')

flags.DEFINE_bool(
    'guide_with_train_loss',
    default=False,
    help='If True, use train batch for guide batch')

flags.DEFINE_integer(
    'reset_guidance_per_steps',
    default=0,
    help='if nonzero, reset all guided vars every X steps')

flags.DEFINE_integer(
    'max_target_length',
    default=256,
    help='Maximum length cutoff for training/guide/eval examples.')

flags.DEFINE_bool(
    'use_model_lr_for_guided',
    default=False,
    help='If True, use the default model LR schedule for guided vars.')

flags.DEFINE_bool(
    'save_checkpoints', default=True, help='Whether to save model checkpoints.')

flags.DEFINE_integer(
    'keep_checkpoints_count',
    default=10,
    help='How many model checkpoints to keep around at a time.')

flags.DEFINE_bool(
    'save_checkpoint_at_init',
    default=True,
    help='Whether to save model checkpoint 0 at initialization.')

flags.DEFINE_bool(
    'clip_nan_grads',
    default=True,
    help='If True, clip NaNs in the hyper-parameter gradient.')

flags.DEFINE_string(
    'init_checkpoint',
    default=None,
    help='Pretrained checkpoint to initialize training.')

flags.DEFINE_string(
    'guided_hparam_types',
    default='learning_rate',
    help='Only used when "param" in model_optimizer_type, and train_with_guided_parameters is true.'
)

flags.DEFINE_integer(
    'model_ckpt_min_freq',
    default=10000,
    help='Checkpoint model every X steps during training. If 0, ignore.')

flags.DEFINE_integer(
    'train_metrics_per_eval_metric',
    default=10,
    help='Emit train metrics this many times per emitting eval metrics.')

flags.DEFINE_bool(
    'save_ckpt_per_epoch',
    default=False,
    help='If True, save a ckpt per epoch. Supersedes model_ckpt_min_freq.')

flags.DEFINE_float(
    'label_smoothing', default=0.0, help='Cross entropy loss label smoothing.')

flags.DEFINE_float('dropout_rate', default=0.1, help='Dropout rate.')

flags.DEFINE_float(
    'attention_dropout_rate', default=0.1, help='Attention dropout rate.')

# Eval flags.
flags.DEFINE_boolean(
    'do_eval', default=True, help='Whether or not to perform evaluation.')

flags.DEFINE_list(
    'eval_dataset_list',
    default=[
        'bea_dev',
        'conll14',
        'fce_test',
    ],
    help=textwrap.dedent("""Which datasets to evaluate on. Options:
        bea_dev,
        bea_dev_abc_half1,
        bea_dev_abc_half2,
        bea_dev_n_half1,
        bea_dev_n_half2,
        bea_train,
        conll13,
        conll14,
        fce_test,
        fce_train,
        uxr."""),
)

flags.DEFINE_integer(
    'eval_batch_size', default=16, help='Per host batch size for eval.')

flags.DEFINE_integer(
    'min_eval_freq',
    default=500,
    help='Minimum bound of evaluation frequency during model training.')

flags.DEFINE_integer(
    'max_metric_freq',
    default=250,
    help='Minimum bound of how many evaluations to make during model training.')

flags.DEFINE_integer(
    'min_train_metric_freq',
    default=250,
    help='Minimum bound of how many evaluations to make during model training.')

flags.DEFINE_integer(
    'num_eval_steps',
    default=1024,
    help='Number of steps to take during evaluation.')

flags.DEFINE_integer(
    'total_evals',
    default=200,
    help='Eval at least this many times over the course of training. '
    'Can be overridden by FLAGS.min_eval_freq.')

# Guided learning flags.
flags.DEFINE_bool(
    'train_with_guided_parameters',
    default=True,
    help='If True, train with guided parameters. Else, vanilla training.')

flags.DEFINE_float(
    'guided_weight_decay',
    default=0,
    help='weight_decay decay for guided optimizers')

flags.DEFINE_integer(
    'guided_ckpt_min_freq',
    default=0,
    help='Checkpoint guided optimizers every X steps during training. Upper-'
    'bounded by model_ckpt_min_freq. If 0, fall back to model_ckpt_min_freq.'
    'The only reason to have this at greater frequency that model_ckpt_min_freq'
    ' is for guided learning visualization purposes.')

flags.DEFINE_enum(
    'guide_batch_update_freq', 'NEVER',
    ['NEVER', 'PER_EPOCH', 'PER_TRAINING_STEP', 'PER_GUIDANCE_STEP'],
    'How often to update the guidance batch.')

flags.DEFINE_integer(
    'guide_batch_size', default=256, help='Per host batch size for guidance.')

flags.DEFINE_float(
    'learning_rate_guided_vars',
    default=3e-5,
    help='Base learning rate for guided parameters.')

flags.DEFINE_string(
    'learning_rate_schedule_guided_vars',
    default='constant',
    help='Schedule for learning rate of guided parameters.')

flags.DEFINE_float(
    'warmup_steps_ratio_guided_vars',
    default=0.1,
    help='Linear learning rate warmup for guided parameters.')

flags.DEFINE_float(
    'grad_clip_limit',
    default=1.0,
    help='0.0 means no clip limit. 10 is reasonable, if NaN, try lower.')

flags.DEFINE_bool(
    'use_grad_clip',
    default=False,
    help='If True, apply FLAGS.grad_clip_limit to guided parameter gradients.')

# Decode/predict flags.

flags.DEFINE_integer(
    'jax_beam_size', default=4, help='Beam size for inference.')

flags.DEFINE_integer(
    'max_predict_length',
    default=64,
    help='Maximum length cutoff for predicted tokens. Predicted examples are '
    'not packed so this is max length for an individual example.')

flags.DEFINE_bool(
    'do_predict',
    default=True,
    help='Whether to run beam predictions on the eval set.')

flags.DEFINE_bool(
    'do_score',
    default=False,
    help='Whether to run beam predictions on the eval set.')

flags.DEFINE_integer(
    'num_decoding_iterations', default=2, help='Number of decoding iterations.')

flags.DEFINE_float(
    'identity_penalty', default=1.0, help='Identity penalty factor.')

flags.DEFINE_list('testsets', default=[], help='Which testsets to run.')

flags.DEFINE_list(
    'identity_penalty_list',
    default=[0.8, 1, 2.0, 3.0],
    help='which identity_penalty values to grid search, override with csv')

# Model flags

flags.DEFINE_enum('model_optimizer_type', 'lamb', [
    'lamb',
    'gd',
    'adam',
    'sgd',
    'mom',
    'momentum',
    'adagrad',
    'adafactor',
], 'Which optimizer to use for model optimization.')

flags.DEFINE_enum(
    'guided_params_optimizer_type', 'adam', [
        'lamb',
        'gd',
        'adam',
        'sgd',
        'mom',
        'momentum',
        'adagrad',
        'adafactor',
    ], 'Which optimizer to use for guided parameter meta-optimization.')

flags.DEFINE_bool(
    'share_embeddings',
    default=True,
    help='Inputs and targets share embedding.')

flags.DEFINE_bool(
    'logits_via_embedding',
    default=True,
    help='Final logit transform uses embedding matrix transpose.')

flags.DEFINE_integer(
    'num_layers', default=6, help='Number of transformer layers.')

flags.DEFINE_integer(
    'qkv_dim', default=1024, help='Size of query/key/value for attention.')

flags.DEFINE_integer('emb_dim', default=1536, help='Size of embeddings.')

flags.DEFINE_integer('mlp_dim', default=4096, help='Size of the MLP.')

flags.DEFINE_integer('num_heads', default=4, help='Number of attention heads.')

flags.DEFINE_bool(
    'use_bfloat16',
    default=True,
    help=('Use bfloat16 mixed precision training instead of float32.'))

flags.DEFINE_bool(
    'take_current_run_step_from_init',
    default=False,
    help=('If True, when init from ckpt_X, current run step set to X. Else 0.'))

# Model optimizer flags - only those for FLAGS.model_optimizer_type will be used

flags.DEFINE_float('adagrad_eps', default=1e-8, help='Adagrad epsilon.')

flags.DEFINE_float('momentum_beta1', default=0.9, help='Momentum beta1.')
flags.DEFINE_float(
    'momentum_weight_decay', default=0.0, help='Momentum weight_decay.')
flags.DEFINE_bool(
    'momentum_nesterov', default=True, help='Momentum use nesterov.')

flags.DEFINE_bool(
    'adafactor_factored',
    default=True,
    help='Adafactor adafactor_multiply_by_parameter_scale')
flags.DEFINE_bool(
    'adafactor_multiply_by_parameter_scale',
    default=True,
    help='Adafactor adafactor_multiply_by_parameter_scale')
flags.DEFINE_float(
    'adafactor_beta1', default=None, help='Adafactor optimizer beta1.')
flags.DEFINE_float(
    'adafactor_epsilon1', default=1e-30, help='Adafactor optimizer eps1.')
flags.DEFINE_float(
    'adafactor_epsilon2', default=1e-3, help='Adafactor optimizer eps2.')
flags.DEFINE_float(
    'adafactor_decay_rate', default=0.8, help='Adafactor optimizer decay rate.')
flags.DEFINE_integer(
    'adafactor_step_offset',
    default=0,
    help='Adafactor adafactor_multiply_by_parameter_scale')
flags.DEFINE_float(
    'adafactor_clipping_threshold',
    default=1.0,
    help='Adafactor adafactor_multiply_by_parameter_scale')
flags.DEFINE_float(
    'adafactor_weight_decay_rate',
    default=None,
    help='Adafactor adafactor_multiply_by_parameter_scale')
flags.DEFINE_integer(
    'adafactor_min_dim_size_to_factor',
    default=128,
    help='Adafactor adafactor_multiply_by_parameter_scale')

flags.DEFINE_float('adam_beta1', default=0.9, help='Adam optimizer beta1.')
flags.DEFINE_float('adam_beta2', default=0.999, help='Adam optimizer beta2.')
flags.DEFINE_float(
    'adam_beta3', default=0.1, help='smAdam decay optimizer beta3.')
flags.DEFINE_float('adam_eps', default=1e-6, help='Adam optimizer epsilon.')
flags.DEFINE_float(
    'adam_weight_decay', default=0.0, help='Adam optimizer weight_decay.')

flags.DEFINE_float('lamb_beta1', default=0.9, help='LAMB optimizer beta1.')
flags.DEFINE_float('lamb_beta2', default=0.999, help='LAMB optimizer beta2.')
flags.DEFINE_float('lamb_eps', default=1e-6, help='LAMB optimizer epsilon.')
flags.DEFINE_float(
    'lamb_weight_decay', default=0.0, help='LAMB optimizer weight_decay.')

# pylint: enable=line-too-long
flags.DEFINE_float(
    'weight_decay',
    default=0,
    help='If the model_optimizer_type supports weight decay, it '
    'will be initialized by this value. May be overridden by '
    'guided hparam.')

flags.DEFINE_float(
    'ghp_lrs_beta1',
    default=1.0,
    help='Scale the meta-optimizer guided learning rate for beta1.')
flags.DEFINE_float(
    'ghp_lrs_eps',
    default=1.0,
    help='Scale the meta-optimizer guided learning rate for epsilon.')
flags.DEFINE_float(
    'ghp_lrs_lr',
    default=1.0,
    help='Scale the meta-optimizer guided learning rate for learning_rate.')
flags.DEFINE_float(
    'ghp_lrs_wd',
    default=1.0,
    help='Scale the meta-optimizer guided learning rate for weight_decay.')
flags.DEFINE_float(
    'ghp_lrs_ls',
    default=1.0,
    help='Scale the meta-optimizer guided learning rate for label_smoothing.')

flags.DEFINE_float(
    'guided_update_beta1',
    default=None,
    help='Use to override meta-optimizer default beta1, if applicable.')
flags.DEFINE_float(
    'guided_update_beta2',
    default=None,
    help='Use to override meta-optimizer default beta2, if applicable.')
flags.DEFINE_float(
    'guided_update_eps',
    default=None,
    help='Use to override meta-optimizer default epsilon, if applicable.')
flags.DEFINE_float(
    'guided_update_wd',
    default=None,
    help='Use to override meta-optimizer default weight_decay, if applicable.')

flags.DEFINE_bool(
    'print_for_colab',
    default=False,
    help='If true, print to console as well as output logging.')

flags.DEFINE_integer(
    'max_eval_steps',
    default=256,
    help='Max number of steps to take during evaluation.')

_TRAIN_KEYS_BASE = [
    'inputs',
    'targets',
    'inputs_position',
    'targets_position',
    'inputs_segmentation',
    'targets_segmentation',
]
_TRAINING_DONE_FILENAME = 'training_done.txt'


def _get_default_opt_kwargs(model_optimizer_type, lr):
  """Returns a dict of all default hparams of the given optimizer."""
  opt_kwargs = {'learning_rate': lr}
  model_optimizer_type = model_optimizer_type.split('_')
  if 'adam' in model_optimizer_type:
    opt_kwargs.update({
        'beta1': FLAGS.adam_beta1,
        'beta2': FLAGS.adam_beta2,
        'eps': FLAGS.adam_eps,
        'weight_decay': FLAGS.adam_weight_decay,
    })
  elif 'lamb' in model_optimizer_type:
    opt_kwargs.update({
        'beta1': FLAGS.lamb_beta1,
        'beta2': FLAGS.lamb_beta2,
        'eps': FLAGS.lamb_eps,
        'weight_decay': FLAGS.lamb_weight_decay,
    })
  elif 'mom' in model_optimizer_type:
    opt_kwargs.update({
        'beta': FLAGS.momentum_beta1,
        'weight_decay': FLAGS.momentum_weight_decay,
        'nesterov': FLAGS.momentum_nesterov,
    })
  elif 'adagrad' in model_optimizer_type:
    opt_kwargs.update({
        'eps': FLAGS.adagrad_eps,
    })
  elif 'adafactor' in model_optimizer_type:
    opt_kwargs.update({
        'beta1':
            FLAGS.adafactor_beta1,
        'decay_rate':
            FLAGS.adafactor_decay_rate,
        'epsilon1':
            FLAGS.adafactor_epsilon1,
        'epsilon2':
            FLAGS.adafactor_epsilon2,
        'factored':
            FLAGS.adafactor_factored,
        'multiply_by_parameter_scale':
            FLAGS.adafactor_multiply_by_parameter_scale,
        'step_offset':
            FLAGS.adafactor_step_offset,
        'clipping_threshold':
            FLAGS.adafactor_clipping_threshold,
        'weight_decay_rate':
            FLAGS.adafactor_weight_decay_rate,
        'min_dim_size_to_factor':
            FLAGS.adafactor_min_dim_size_to_factor,
    })
  elif 'sgd' in model_optimizer_type or 'gd' in model_optimizer_type:
    pass
  else:
    raise ValueError('Unrecognized model_optimizer_type: %s' %
                     model_optimizer_type)
  return opt_kwargs


def get_step_from_opt(model_optimizer):
  if not hasattr(model_optimizer, 'state'):
    raise ValueError(
        'model optimizer does not have state attribute! Cannot retrieve step.')
  if isinstance(model_optimizer.state, tuple):
    if hasattr(model_optimizer.state[0], 'count'):
      return model_optimizer.state[0].count
  if hasattr(model_optimizer.state, 'count'):
    return model_optimizer.state.count
  return model_optimizer.state.step


def update_model_and_guidance_step(
    optimizer_dict,
    train_batch,
    guided_vars_dict,
    learning_rate_fn,
    learning_rate_fn_guided_vars,
    guide_batch,
    grad_clip_limit=0.0,
    use_bfloat16=False,
    dropout_rng=None,
    output_metrics=None,
):
  """Updates all guided optimizers and the model, with a single forward pass.

  If multiple guided optimizers are in use, they are updated simultaneously.

  Args:
    optimizer_dict: a dictionary holding all optimizers
    train_batch: the batch of training data to be applied
    guided_vars_dict: specifies the guided parameter configuration
    learning_rate_fn: model learning rate function
    learning_rate_fn_guided_vars: guided vars learning rate function
    guide_batch: the batch of guidance data to be applied
    grad_clip_limit: if nonzero, clip all gradients to abs magnitude <= limit
    use_bfloat16: bool, If True, use lower-precision bfloat16
    dropout_rng: the JAX rng for dropout
    output_metrics: dict of metrics to write out to tensorboard

  Returns:
    optimizer_dict: updated with steps taken for guided parameter optimizers
    metrics: metrics dict keeping track of tensorboard-reported summary values
    new_dropout_rng: updated RNG
  """
  metrics_joint = {}  # for summarizing
  model_optimizer = optimizer_dict['model']  # for readability
  dropout_rng, new_dropout_rng = random.split(dropout_rng)
  current_step = get_step_from_opt(model_optimizer)

  # Extract fields of interest from the training batch.
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [train_batch.get(k, None) for k in _TRAIN_KEYS_BASE]

  # Extract fields from the guidance batch.
  (guide_inputs, guide_targets, guide_inputs_positions, guide_targets_positions,
   guide_inputs_segmentation, guide_targets_segmentation) = [
       guide_batch.get(k, None) for k in _TRAIN_KEYS_BASE
   ]

  # Organize raw vars and activation functions for easy access.
  raw_vars_dict, act_fn_dict = guided_parameters.get_raw_vars_and_act_fns(
      optimizer_dict, guided_vars_dict)

  # Learning rate for the guided variables.
  lr_guided_vars = learning_rate_fn_guided_vars(current_step)
  metrics_joint['meta_hparams/learning_rate_for_guided_vars'] = lr_guided_vars
  for x in guided_vars_dict:
    metrics_joint[
        'meta_hparams/learning_rate_for_guided_vars__%s' %
        x.split('-')[-1].split('__')
        [-1]] = lr_guided_vars * guided_vars_dict[x]['learning_rate_scalar']

  # Loss fn for model step lookahead, used within loss_fn_guidance_step.
  # Applies example weighting, smoothing, etc, to the lookahead model.
  def loss_fn_model_lookahead(
      model,
      raw_label_smoothing,
  ):
    """Loss fn for model step lookahead, used within loss_fn_guidance_step."""

    # Default to FLAGS value if not set within guided_vars_dict.
    dropout_rate = FLAGS.dropout_rate
    attention_dropout_rate = FLAGS.attention_dropout_rate
    label_smoothing = FLAGS.label_smoothing if FLAGS.label_smoothing else 0.0

    # It is necessary to do None comparison rather than ternary operator here
    # because JAX will have tracer errors with the ternary operator.
    if raw_label_smoothing is not None:
      label_smoothing = act_fn_dict['hp-label_smoothing'](raw_label_smoothing)

    with nn.stochastic(dropout_rng):
      train_logits = model(
          inputs=inputs,
          targets=targets,
          use_bfloat16=use_bfloat16,
          inputs_positions=inputs_positions,
          targets_positions=targets_positions,
          inputs_segmentation=inputs_segmentation,
          targets_segmentation=targets_segmentation,
          train=True,
          cache=None,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate)

    # Initialize weights using all available guided parameters.
    weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

    # Now that all dp-vars have been applied to weights, calculate loss.
    loss, weight_sum = utils.compute_weighted_cross_entropy(
        train_logits,
        targets,
        weights,
        label_smoothing,
    )
    full_train_loss = loss / jnp.where(weight_sum != 0, weight_sum, 1)

    return full_train_loss, (train_logits,)

  # Loss fn for guided variables, contains a call to loss_fn_model_lookahead.
  def loss_fn_guidance_step(
      model_opt,
      map_guided_names_to_raw_indices,
      *per_param_raw_values,
  ):
    """Loss function used for guided parameter variables.

    Summary:
      this returns validation loss of the next-step model as a function of the
      current model and the current weights.

      The partial derivative of this function with respect to the raw_vars gives
      the gradient direction for the updates to the raw vars, as determined by
      their marginal contribution to the loss on the validation set. It is
      necessary to pass the raw vars (as opposed to the optimizer that wraps
      them) in order to take the gradient of this function with respect to the
      vars themselves.

    Steps:
      calculate current model next train step gradient using current weights
      apply gradient to get 'next-step' model
      calculate validation loss for 'next-step' model
      (model_opt_lookahead.target)
      return mean_val_loss

    Args:
      model_opt: optimizer that wraps Transformer model. the model is accessed
        via model_opt.target. It is necessary to pass the optimizer in (rather
        than just the model itself) so that the model may be updated via the
        optimizer's model_opt.apply_gradient(...) fn.
      map_guided_names_to_raw_indices: dict mapping guided names to raw value
        indices within per_param_raw_values
      *per_param_raw_values: raw values of guided params, with indices specified
        by map_guided_names_to_raw_indices

    Returns:
      mean_val_loss
      guide_logits
    """
    global_guided_vars_count = 0

    raw_label_smoothing = None
    if 'hp-label_smoothing' in map_guided_names_to_raw_indices:
      raw_label_smoothing = per_param_raw_values[
          map_guided_names_to_raw_indices['hp-label_smoothing']]
      global_guided_vars_count += 1

    # All raw guided vars need to be activated.
    model_learning_rate = learning_rate_fn(current_step)
    if 'hp-learning_rate' in map_guided_names_to_raw_indices:
      activated_lr = act_fn_dict['hp-learning_rate'](
          per_param_raw_values[
              map_guided_names_to_raw_indices['hp-learning_rate']])
      model_learning_rate *= activated_lr
      global_guided_vars_count += 1

    # calculate current model next train step gradient using current weights
    (full_train_loss,
     (train_logits,)), model_train_step_grad = jax.value_and_grad(
         loss_fn_model_lookahead, argnums=0, has_aux=True)(
             model_opt.target,
             raw_label_smoothing=raw_label_smoothing,
         )
    model_train_step_grad = jax.lax.pmean(model_train_step_grad, 'batch')

    opt_kwargs_dict = _get_default_opt_kwargs(
        model_optimizer_type=FLAGS.model_optimizer_type,
        lr=model_learning_rate,
    )
    # Override defaults with learned values.
    for hparam in opt_kwargs_dict:
      if 'hp-%s' % hparam in map_guided_names_to_raw_indices:
        if hparam == 'learning_rate':  # Skip LR as it is already set above.
          continue
        opt_kwargs_dict[hparam] = act_fn_dict['hp-%s' % hparam](
            per_param_raw_values[map_guided_names_to_raw_indices['hp-%s' %
                                                                 hparam]])
    apply_grad_ret_list = model_opt.apply_gradient(
        model_train_step_grad,
        **opt_kwargs_dict,
    )
    # Some optimizers report param_norms and update_norms, so check if the
    # returned value is a tuple which contains those or not.
    model_opt_out = apply_grad_ret_list
    # Note: uses validation inputs/targets and model_opt
    with nn.stochastic(dropout_rng):
      guide_logits = model_opt_out.target(
          inputs=guide_inputs,
          targets=guide_targets,
          inputs_positions=guide_inputs_positions,
          targets_positions=guide_targets_positions,
          inputs_segmentation=guide_inputs_segmentation,
          targets_segmentation=guide_targets_segmentation,
          use_bfloat16=use_bfloat16,
          dropout_rate=0,
          attention_dropout_rate=0,
          train=False,
          cache=None)

    guide_loss, guide_weight_sum = utils.compute_weighted_cross_entropy(
        guide_logits,
        guide_targets,
        weights=jnp.where(guide_targets > 0, 1, 0).astype(jnp.float32))

    total_guide_loss = guide_loss / guide_weight_sum
    return total_guide_loss, (guide_logits, full_train_loss, train_logits,
                              model_opt_out, model_learning_rate)

  # Calculate gradient for all guided vars.
  guided_hparam_names = [x for x in optimizer_dict.keys() if x != 'model'
                        ]  # Exclude the model itself.
  # Sort by the key (lp-<abbreviated param name>__<hparam> for param strings).
  guided_hparam_names.sort()

  # Extract values only.
  guided_hparam_raw_values = [raw_vars_dict[k] for k in guided_hparam_names]

  # This allows us to know which of the guided_hparam_raw_values corresponds to
  # which name, and apply the appropriate logic, within the
  # loss_fn_guidance_step.
  map_guided_names_to_raw_indices = {
      name: i for i, name in enumerate(guided_hparam_names)
  }
  # We have to use args list instead of kwargs dict for jax.grad to be able to
  # pick out the correct arg for differentiation.
  guide_step_args = [
      optimizer_dict['model'],
      map_guided_names_to_raw_indices,
      *guided_hparam_raw_values,
  ]

  # The count of explicit args in the signature of loss_fn_guidance_step
  # preceding the catch-all *per_param_raw_values.
  # The specific mapping is:
  # def loss_fn_guidance_step(
  #   model_opt, -- index 0
  #   map_guided_names_to_raw_indices, -- index 1
  #   *per_param_raw_values,
  #   )
  # TODO(lichtarge): set this with the inspect.signature fn instead of manually.
  max_index_explicit_args = 1
  # Add 1 because enumerate starts at 0 and otherwise we'd overlap the index of
  # map_guided_names_to_raw_indices.
  guided_hparam_arg_indices = [
      map_guided_names_to_raw_indices[x] + 1 + max_index_explicit_args
      for x in guided_hparam_names
  ]

  # We do not know how many learned opts there will be so we unpack into a list.
  # https://www.python.org/dev/peps/pep-3132/
  # The unused vars are unregularized_loss and regularization_loss respectively.
  (_, (guide_logits, _, train_logits, model_opt_out,
       model_learning_rate)), *learned_grads = jax.value_and_grad(
           loss_fn_guidance_step, guided_hparam_arg_indices,
           has_aux=True)(*guide_step_args)

  metrics_joint.update(
      utils.compute_metrics(
          train_logits,
          targets,
          weights=jnp.where(targets > 0, 1, 0).astype(jnp.float32),
          tag='dataset_train'))
  optimizer_dict['model'] = model_opt_out

  # These are always used.
  metrics_joint['hparams/guided_lr_schedule'] = model_learning_rate

  if output_metrics is not None:
    for guided_hparam in guided_hparam_names:
      if guided_hparam == 'hp-learning_rate':
        metrics_joint['guided_hparams/guided_lr_schedule'] = model_learning_rate
      activated_vars = act_fn_dict[guided_hparam](raw_vars_dict[guided_hparam])
      metrics_joint['guided_hparams/%s' % guided_hparam] = activated_vars

  metrics_joint['hparams/default_lr_schedule'] = learning_rate_fn(current_step)

  for i, guided_opt_key in enumerate(guided_hparam_names):
    guided_optimizer = optimizer_dict[guided_opt_key]
    # learned_grads is a List of size 1, the single element is a Tuple
    raw_vars_grad = learned_grads[0][i]
    if output_metrics is not None:
      metrics_joint['guided_hparams_grads/%s' % guided_opt_key] = raw_vars_grad
    if FLAGS.clip_nan_grads:
      raw_vars_grad = jnp.nan_to_num(raw_vars_grad, posinf=10.0, neginf=-10.0)

    # Deal with parallelization.
    raw_vars_grad = jax.lax.pmean(raw_vars_grad, 'batch')
    var_type = guided_parameters.get_var_type_from_opt_key(guided_opt_key)

    # Grad clipping.
    grad_clip_limit = FLAGS.grad_clip_limit
    if grad_clip_limit != 0.0 and FLAGS.use_grad_clip:
      metrics_joint['model_update/%s_grad_clip_limit' %
                    guided_opt_key] = grad_clip_limit
      metrics_joint['model_update/%s_num_grad_clipped_total' %
                    guided_opt_key] = jnp.count_nonzero(
                        raw_vars_grad > grad_clip_limit) + jnp.count_nonzero(
                            raw_vars_grad < -grad_clip_limit)
      raw_vars_grad = jax.tree.map(
          lambda g_: jnp.clip(  # pylint: disable=g-long-lambda
              g_,
              min=-grad_clip_limit,
              max=grad_clip_limit),
          raw_vars_grad)

    guided_update_kwargs = {'learning_rate': lr_guided_vars}
    if guided_vars_dict[var_type]['optimizer_type'] == 'adam':
      if FLAGS.guided_update_beta1 is not None:
        guided_update_kwargs.update({'beta1': FLAGS.guided_update_beta1})
      if FLAGS.guided_update_beta2 is not None:
        guided_update_kwargs.update({'beta2': FLAGS.guided_update_beta2})
      if FLAGS.guided_update_eps is not None:
        guided_update_kwargs.update({'eps': FLAGS.guided_update_eps})
      if FLAGS.guided_update_wd is not None:
        guided_update_kwargs.update({'weight_decay': FLAGS.guided_update_wd})
    elif guided_vars_dict[var_type]['optimizer_type'] in ['momentum', 'mom']:
      if FLAGS.guided_update_beta1 is not None:
        guided_update_kwargs.update({'beta': FLAGS.guided_update_beta1})
      guided_update_kwargs.update({'nesterov': True})
    elif guided_vars_dict[var_type]['optimizer_type'] in ['adagrad']:
      if FLAGS.guided_update_eps is not None:
        guided_update_kwargs.update({'eps': FLAGS.guided_update_eps})
    elif guided_vars_dict[var_type]['optimizer_type'] in ['adafactor']:
      if FLAGS.guided_update_eps is not None:
        guided_update_kwargs.update({'eps': FLAGS.guided_update_eps})
      if FLAGS.guided_update_beta1 is not None:
        guided_update_kwargs.update({'beta1': FLAGS.guided_update_beta1})
    # apply raw vars gradient to dp optimizer, updating raw_vars

    guided_optimizer = guided_optimizer.apply_gradient(raw_vars_grad,
                                                       **guided_update_kwargs)
    # update the optimizer_dict
    optimizer_dict[guided_opt_key] = guided_optimizer
  if output_metrics is not None:
    metrics_joint.update(
        utils.compute_metrics(
            guide_logits,
            guide_targets,
            weights=jnp.where(guide_targets > 0, 1, 0).astype(jnp.float32),
            tag='dataset_guide'))

  return optimizer_dict, metrics_joint, new_dropout_rng


def update_model_step(
    optimizer_dict,
    train_batch,
    learning_rate_fn,
    use_bfloat16=False,
    dropout_rng=None,
    output_metrics=None,
):
  """Updates the model only, without using guided parameters.

  Args:
    optimizer_dict: a dictionary holding all optimizers
    train_batch: the batch of training data to be applied
    learning_rate_fn: model learning rate function
    use_bfloat16: bool, If True, use lower-precision bfloat16
    dropout_rng: the JAX rng for dropout
    output_metrics: metrics to be passed through and reported

  Returns:
    optimizer_dict: updated with steps taken for guided parameter optimizers
    metrics: metrics dict keeping track of tensorboard-reported summary values
    new_dropout_rng: updated RNG
  """
  metrics_train = {}  # for summarizing
  if output_metrics:
    metrics_train.update(output_metrics)
  model_optimizer = optimizer_dict['model']  # for readability
  dropout_rng, new_dropout_rng = random.split(dropout_rng)
  current_step = model_optimizer.state.step

  # Extract fields of interest from the training batch.
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [train_batch.get(k, None) for k in _TRAIN_KEYS_BASE]

  # Loss fn for model step lookahead, used within loss_fn_guidance_step.
  # Applies example weighting, smoothing, etc, to the lookahead model.
  def loss_fn_model_lookahead(model,):
    """Loss fn for model step lookahead, used within loss_fn_guidance_step."""

    # Default to FLAGS value if not set within guided_vars_dict.
    dropout_rate = FLAGS.dropout_rate
    attention_dropout_rate = FLAGS.attention_dropout_rate
    label_smoothing = FLAGS.label_smoothing if FLAGS.label_smoothing else 0.0

    with nn.stochastic(dropout_rng):
      train_logits = model(
          inputs=inputs,
          targets=targets,
          use_bfloat16=use_bfloat16,
          inputs_positions=inputs_positions,
          targets_positions=targets_positions,
          inputs_segmentation=inputs_segmentation,
          targets_segmentation=targets_segmentation,
          train=True,
          cache=None,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate)

    # Ignore loss on padding.
    weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

    loss, weight_sum = utils.compute_weighted_cross_entropy(
        train_logits,
        targets,
        weights,
        label_smoothing,
    )
    full_train_loss = loss / jnp.where(weight_sum != 0, weight_sum, 1)

    return full_train_loss, (train_logits,)

  model_learning_rate = learning_rate_fn(current_step)
  (_, (train_logits,)), model_train_step_grad = jax.value_and_grad(
      loss_fn_model_lookahead, argnums=0, has_aux=True)(model_optimizer.target,)
  model_train_step_grad = jax.lax.pmean(model_train_step_grad, 'batch')

  opt_kwargs_dict = _get_default_opt_kwargs(
      model_optimizer_type=FLAGS.model_optimizer_type,
      lr=model_learning_rate,
  )
  model_opt_out = model_optimizer.apply_gradient(
      model_train_step_grad,
      **opt_kwargs_dict,
  )

  metrics_train.update(
      utils.compute_metrics(
          train_logits,
          targets,
          weights=jnp.where(targets > 0, 1, 0).astype(jnp.float32),
          tag='dataset_train'))
  optimizer_dict['model'] = model_opt_out

  # These are always used.
  metrics_train['hparams/default_lr_schedule'] = learning_rate_fn(current_step)

  return optimizer_dict, metrics_train, new_dropout_rng


def eval_step(model, batch, label_smoothing=0.0, use_bfloat16=False):
  """Calculate evaluation metrics on a batch."""
  del label_smoothing
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [batch.get(k, None) for k in _TRAIN_KEYS_BASE]
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = model(
      inputs,
      targets,
      use_bfloat16=use_bfloat16,
      inputs_positions=inputs_positions,
      targets_positions=targets_positions,
      inputs_segmentation=inputs_segmentation,
      targets_segmentation=targets_segmentation,
      train=False,
      cache=None)
  return utils.compute_metrics(logits, targets, weights, tag='dataset_eval')


def batch_to_numpy(batch):
  return jax.tree.map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access


def do_eval_single(
    step,
    eval_ds,
    eval_ds_name,
    p_eval_step,
    optimizer,
):
  """Computes metrics on a single evaluation dataset."""
  eval_metrics = []
  eval_iter = iter(eval_ds)
  sum_unique_examples = 0
  logging.info('Starting eval for eval set %s at step %s.', eval_ds_name, step)
  if FLAGS.print_for_colab:
    print('Starting eval for eval set %s at step %s.', eval_ds_name, step)

  eval_ds_num_batches = 0
  for eval_batch in eval_iter:
    eval_ds_num_batches += 1
    eval_batch = batch_to_numpy(eval_batch)
    sum_unique_examples += data.get_unique_examples(eval_batch)
    eval_batch = common_utils.shard(eval_batch)
    metrics = p_eval_step(optimizer.target, eval_batch)
    eval_metrics.append(metrics)
  if not eval_metrics:
    raise ValueError('Eval has failed for: %s' % eval_ds_name)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree.map(jnp.sum, eval_metrics)

  eval_denominator = eval_metrics_sums.pop('dataset_eval/denominator')
  eval_summary = jax.tree.map(
      lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
      eval_metrics_sums)
  eval_summary['dataset_eval/unique_examples'] = sum_unique_examples

  logging.info('Finished do_eval (%s batches).', eval_ds_num_batches)
  if FLAGS.print_for_colab:
    print('Finished do_eval (%s batches).', eval_ds_num_batches)
  return eval_summary


def write_finished_training(model_dir):
  """Write a file into CNS marking this training run as complete.."""
  if jax.process_index() != 0:
    return
  logging.info('Writing to CNS that training is done.')
  try:
    with tf.io.gfile.GFile(
        os.path.join(model_dir, _TRAINING_DONE_FILENAME), 'w') as f:
      f.write('Training run is complete.')
  except tf.errors.NotFoundError:
    pass


def maybe_remove_finished_training(model_dir):
  """Remove (if present) the CNS file marking this training run as complete."""
  if jax.process_index() != 0:
    return
  if tf.io.gfile.exists(os.path.join(model_dir, _TRAINING_DONE_FILENAME)):
    logging.info('Removing "%s" while current run ongoing.',
                 _TRAINING_DONE_FILENAME)
    tf.io.gfile.remove(os.path.join(model_dir, _TRAINING_DONE_FILENAME))


def save_all_checkpoints(
    t_start,
    optimizer_dict,
    step,
):
  """Save optimizers from optimizer_dict."""
  if not FLAGS.save_checkpoints:
    return
  # only save once, not once per worker
  if jax.process_index() == 0:
    utils.log_message(t_start, 'Saving optimizers, current step: %s' % step)
    if FLAGS.print_for_colab:
      print('saving optimizers, current step: %s' % step)
    print('sorted(list(optimizer_dict.keys()))')
    print(sorted(list(optimizer_dict.keys())))
    for opt_dict_key in sorted(optimizer_dict.keys()):
      optimizer = optimizer_dict[opt_dict_key]
      utils.log_message(t_start, 'Saving optimizer: %s' % opt_dict_key)
      prefix_string = '%s_' % opt_dict_key.replace('/', '.')
      write_to_file = os.path.join(FLAGS.model_dir,
                                   '%s%s' % (prefix_string, step))
      # It may be necessary to overwrite hp-vars files if the training run
      # was interrupted and we had to go back to the last model ckpt.
      if tf.io.gfile.exists(write_to_file):
        logging.info('Skipping saving ckpt at %s because it already exists.',
                     write_to_file)
        print('Skipping saving ckpt at %s because it already exists.',
              write_to_file)
        continue
      checkpoints.save_checkpoint(
          FLAGS.model_dir,
          jax_utils.unreplicate(optimizer),
          step,
          prefix=prefix_string,
          keep=FLAGS.keep_checkpoints_count)


def maybe_save_ckpts(
    t_start,
    optimizer_dict,
    current_step,
    steps_since_last_model_ckpt,
    steps_since_last_guided_ckpt,
):
  """Save optimizers from optimizer_dict."""
  if not FLAGS.save_checkpoints:
    return steps_since_last_model_ckpt, steps_since_last_guided_ckpt
  # only save once, not once per worker
  if jax.process_index() != 0:
    return steps_since_last_model_ckpt, steps_since_last_guided_ckpt
  # Save a checkpoint on one host after every model_ckpt_min_freq steps.
  # Maybe save guided vars more frequently (every guided_ckpt_min_freq steps).
  save_guided = False
  save_model = False

  if FLAGS.guided_ckpt_min_freq and steps_since_last_guided_ckpt >= FLAGS.guided_ckpt_min_freq:
    save_guided = True
    steps_since_last_guided_ckpt = 0
  else:
    steps_since_last_guided_ckpt += 1
  if FLAGS.model_ckpt_min_freq and steps_since_last_model_ckpt >= FLAGS.model_ckpt_min_freq - 1:
    save_model = True
    steps_since_last_model_ckpt = 0
    save_guided = True
    steps_since_last_guided_ckpt = 0
  else:
    steps_since_last_model_ckpt += 1

  for opt_dict_key, optimizer in optimizer_dict.items():
    if opt_dict_key == 'model' and not save_model:
      continue
    if opt_dict_key != 'model' and not save_guided:
      continue
    utils.log_message(t_start, 'saving optimizer: %s' % opt_dict_key)
    prefix_string = '%s_' % opt_dict_key
    write_to_file = os.path.join(FLAGS.model_dir,
                                 '%s%s' % (prefix_string, current_step))
    # it may be necessary to overwrite guided-vars files if the training run
    # was interrupted and we had to go back to the last model ckpt
    overwrite = False
    if tf.io.gfile.exists(write_to_file):
      if opt_dict_key == 'model':
        raise ValueError('trying to save model but already exists... '
                         'this can happen if two runs are writing to same dir?'
                         ' file: %s' % write_to_file)
      utils.log_message(
          t_start, 'Overwriting old guided vars checkpoint: %s' % write_to_file)
      tf.io.gfile.remove(write_to_file)
      overwrite = True
    checkpoints.save_checkpoint(
        FLAGS.model_dir,
        jax_utils.unreplicate(optimizer),
        current_step,
        prefix=prefix_string,
        overwrite=overwrite,
        keep=FLAGS.keep_checkpoints_count)
  return steps_since_last_model_ckpt, steps_since_last_guided_ckpt


def write_flags_files(output_dir, guided_vars_dict):
  """Write run flags to a human-readable text file and a json dump."""
  if jax.process_index() != 0:
    return

  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.MakeDirs(output_dir)

  flag_dict = FLAGS.flags_by_module_dict()
  jax_module_names = [x for x in flag_dict.keys() if 'jax' in x or 'train' in x]

  jax_flags_str = ''
  jax_train_dict = {}
  for module in jax_module_names:
    jax_flags_str += '\n' + module + '\n'
    jax_flags_str += '\t\n'.join(
        sorted(['--%s=%s' % (f.name, f.value) for f in flag_dict[module]]))
    jax_flags_str += '\n'

    # This isolates train.py flags locally and on borg.
    if 'jax.train' in module or 'train.par' in module:
      jax_train_dict.update({f.name: f.value for f in flag_dict[module]})
  flags_txt = os.path.join(output_dir, 'jax_flags.txt')
  logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())

  logging.info('writing flags into: %s', flags_txt)
  try:
    with tf.io.gfile.GFile(flags_txt, 'w') as f:
      f.write(jax_flags_str)
  except tf.errors.NotFoundError:
    logging.warn('Failed to write flags to CNS: %s', flags_txt)

  guided_flags_json = os.path.join(output_dir, 'train_run_hparams.json')
  if guided_vars_dict:
    flattened_dict = {}
    for gv_key, gv_dict in guided_vars_dict.items():
      flattened_dict.update({
          '%s.%s' % (gv_key, k): v
          for k, v in gv_dict.items()
          if k != 'raw_guided_vars'
      })  # Exclude raw_guided_vars as it is unnecessary, non-JSON-serializable.

    jax_train_dict.update(
        {'guided_vars.%s' % k: v for k, v in flattened_dict.items()})
    jax_train_dict = guided_parameters.make_dict_json_safe(jax_train_dict)
  try:
    with tf.io.gfile.GFile(guided_flags_json, 'w') as f:
      json.dump(jax_train_dict, f)
  except tf.errors.NotFoundError:
    pass


def _load_spm_tokenizer(model_path):
  """Load a spm tokenizer from given model filepath."""
  with tf.io.gfile.GFile(model_path, 'rb') as f:
    spm_model = f.read()
  sp_tokenizer = spm.SentencePieceProcessor()
  sp_tokenizer.LoadFromSerializedProto(spm_model)
  return sp_tokenizer


def restore_or_init_model(t_start):
  """Initializes a model; restores from prior ckpt or init ckpt as needed.

  Randomly initialize a Transformer model with FLAGS-specified parameters.
  If the current job has been pre-empted (or started again after termination),
    restore the model parameters from the most recent checkpoint in the output
    directory.
  If no output dir ckpts exist, initialize from the FLAGS.init_checkpoint
    parameters.

  Args:
    t_start: timestamp used for logging

  Returns:
    model_optimizer: the model optimizer
    optimizer_step: the step of the current optimizer state
    rng: jax PRNG key, an np.ndarray of two ints
  """
  utils.log_message(t_start, 'Initializing model.')
  encoder = _load_spm_tokenizer(FLAGS.vocab_path)
  vocab_size = int(encoder.GetPieceSize())
  current_run_step = 0

  # Build Model and Optimizer
  transformer_kwargs = {
      'vocab_size': vocab_size,
      'output_vocab_size': vocab_size,
      'emb_dim': FLAGS.emb_dim,
      'num_heads': FLAGS.num_heads,
      'num_layers': FLAGS.num_layers,
      'qkv_dim': FLAGS.qkv_dim,
      'mlp_dim': FLAGS.mlp_dim,
      'max_len': FLAGS.max_target_length,
      'share_embeddings': FLAGS.share_embeddings,
      'logits_via_embedding': FLAGS.logits_via_embedding,
  }
  logging.info('Transformer kwargs: %s', transformer_kwargs)

  rng = random.PRNGKey(FLAGS.jax_random_seed)
  rng, init_rng = random.split(rng)

  input_shape = (FLAGS.batch_size, FLAGS.max_target_length)
  target_shape = (FLAGS.batch_size, FLAGS.max_target_length)
  model, cache_def = models.create_model(init_rng, input_shape, target_shape,
                                         transformer_kwargs)
  del cache_def
  optimizer = models.init_optimizer_by_type(model, FLAGS.model_optimizer_type)
  # We access model only from optimizer below via optimizer.target.
  del model
  model = None

  # Remove bad tmp files caused by preemption during save; avoids error on save.
  if jax.process_index() == 0:
    for x in tf.io.gfile.glob(os.path.join(FLAGS.model_dir, '*_tmp')):
      tf.io.gfile.remove(x)
  glob_path = tf.io.gfile.glob(os.path.join(FLAGS.model_dir, 'model_*'))
  if glob_path:
    # If training has already started, and checkpoints exist in the model_dir,
    # take the last checkpoint available (occurs if training job is preempted).
    max_model_step = int(
        checkpoints.natural_sort(glob_path)[-1].split('model_')[-1])
    current_run_step = max_model_step
    # restore_checkpoint checks model_dir for present checkpoints;
    # if no checkpoints in model_dir, it passes the optimizer unchanged
    optimizer = checkpoints.restore_checkpoint(
        FLAGS.model_dir, optimizer, prefix='model_')

    # Grab last step from the optimizer itself.
    # this is progress already made this training run [0 --> num_train_steps]
    optimizer_step = get_step_from_opt(optimizer)
    utils.log_message(
        t_start, 'Restoring ckpt from "model_%s", optimizer step (including '
        'warm-initialization steps): %s, current run step (excluding warm-init)'
        ': %s' % (max_model_step, optimizer_step, current_run_step))
    if FLAGS.print_for_colab:
      print(
          t_start, 'Restoring ckpt from "model_%s", optimizer step (including '
          'warm-initialization steps): %s, current run step (excluding warm-init)'
          ': %s' % (max_model_step, optimizer_step, current_run_step))

  # if training hasn't already started / saved a ckpt, init from init_checkpoint
  elif FLAGS.init_checkpoint:
    utils.log_message(t_start,
                      'Init checkpoint from %s' % FLAGS.init_checkpoint)
    if FLAGS.print_for_colab:
      print(t_start, 'Init checkpoint from %s' % FLAGS.init_checkpoint)
    with tf.io.gfile.GFile(FLAGS.init_checkpoint, 'rb') as fp:
      optimizer = serialization.from_bytes(optimizer, fp.read())
    optimizer_step = get_step_from_opt(optimizer)
    if FLAGS.take_current_run_step_from_init:
      current_run_step = optimizer_step

  return optimizer, current_run_step, rng, model


def remove_stale_guided_vars_ckpts(opt_key, restore_step):
  """Remove any stale checkpoints which may have been left by task failure."""
  glob_paths = tf.io.gfile.glob(os.path.join(FLAGS.model_dir, '%s*' % opt_key))
  past_restore_path = os.path.join(FLAGS.model_dir,
                                   '%s_%s' % (opt_key, restore_step + 1))
  glob_paths.append(past_restore_path)
  # It may be necessary to overwrite guided optimizer ckpts if the training run
  # was interrupted and we had to go back to the last model ckpt.
  sorted_ckpts = checkpoints.natural_sort(glob_paths)
  for x in sorted_ckpts[sorted_ckpts.index(past_restore_path):]:
    if tf.io.gfile.exists(x):
      logging.info(
          'Removing bad ckpt (got ahead of model due to task failure): %s', x)
      tf.io.gfile.remove(x)


def restore_or_init_guided_optimizers(
    t_start,
    guided_vars_dict,
):
  """Initializes and maybe restores guided optimizers.

  Args:
    t_start: timestamp used for logging
    guided_vars_dict: the guided parameters to be applied in this run

  Raises:
    ValueError: if the guided_vars_dict contains an unrecognized key
    ValueError: if the guided opt step is not equal to the expected restore step

  Returns:
    optimizer_dict: contains all guided optimizers

  """
  optimizer_dict = {}
  try:
    if jax.process_index() == 0:
      for x in tf.io.gfile.glob(os.path.join(FLAGS.model_dir, '*_tmp')):
        tf.io.gfile.remove(x)
    glob_path = tf.io.gfile.glob(os.path.join(FLAGS.model_dir, 'model_*'))
  except tf.errors.NotFoundError:
    glob_path = None
  # If there are no models_* files, restore step is 0
  restore_step = -1
  if glob_path:
    last_ckpt = checkpoints.natural_sort(glob_path)[-1]
    if 'model_' in last_ckpt:
      last_ckpt_num = last_ckpt.split('model_')[-1]
      restore_step = int(last_ckpt_num)
  if FLAGS.print_for_colab:
    print('guided_var_types:')
    for guided_var_type in guided_vars_dict:
      print('\t', guided_var_type)
  for guided_var_type in guided_vars_dict:
    guided_var_opt_key = guided_parameters.get_opt_key_from_var_type(
        guided_var_type)
    # First, remove stale checkpoints which may have been left behind by
    # pre-emption.
    remove_stale_guided_vars_ckpts(guided_var_opt_key, restore_step)
    optimizer_hparams = {
        'weight_decay': FLAGS.guided_weight_decay,
    }
    cur_guided_opt = models.init_optimizer_by_type(
        guided_vars_dict[guided_var_type]['raw_guided_vars'],
        guided_vars_dict[guided_var_type]['optimizer_type'],
        optimizer_hparams=optimizer_hparams)

    if restore_step > 0:
      utils.log_message(
          t_start, 'Restoring cur_guided_opt from "%s_%s", ' %
          (guided_var_opt_key, restore_step))

      cur_guided_opt = checkpoints.restore_checkpoint(
          FLAGS.model_dir,
          cur_guided_opt,
          prefix='%s_%s' % (guided_var_opt_key, restore_step))
      cur_opt_step = get_step_from_opt(cur_guided_opt)
      utils.log_message(
          t_start, 'restoring cur_guided_opt from "hp-%s_%s", '
          'optimizer step: %s' % (guided_var_type, restore_step, cur_opt_step))
      if FLAGS.print_for_colab:
        print('restoring cur_guided_opt from "hp-%s_%s", '
              'optimizer step: %s' %
              (guided_var_type, restore_step, cur_opt_step))
    elif 'init_path' in guided_vars_dict[guided_var_type]:
      init_checkpoint_path = guided_vars_dict[guided_var_type]['init_path']
      utils.log_message(
          t_start,
          'initializing cur_guided_opt from "%s", ' % init_checkpoint_path)
      if FLAGS.print_for_colab:
        print('initializing cur_guided_opt from "%s", ' % init_checkpoint_path)
      init_dir, dp_init_step = init_checkpoint_path.split('%s_' %
                                                          guided_var_opt_key)
      if FLAGS.print_for_colab:
        print('init_dir, dp_init_step')
        print(init_dir, dp_init_step)
      cur_guided_opt = checkpoints.restore_checkpoint(
          init_dir,
          cur_guided_opt,
          prefix='%s_%s' % (guided_var_opt_key, dp_init_step))
    cur_guided_opt = jax_utils.replicate(cur_guided_opt)
    utils.log_message(
        t_start, '%s optimizer initialized. optimizer step: %s' %
        (guided_var_opt_key, get_step_from_opt(cur_guided_opt)))
    optimizer_dict[guided_var_opt_key] = cur_guided_opt

  return optimizer_dict


def set_up_summary_writers():
  """Helper fn to set up metric-reporting SummaryWriters."""
  train_sum_writer = tensorboard.SummaryWriter(
      os.path.join(FLAGS.model_dir, 'train-%s' % FLAGS.training_dataset))

  eval_sum_writers_dict = {}
  for eval_ds_name in get_evalsets():
    tb_summary_name = 'eval-%s' % eval_ds_name
    eval_sum_writers_dict[tb_summary_name] = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, tb_summary_name))

  return train_sum_writer, eval_sum_writers_dict


def get_dataset_from_path(data_path, batch_size):
  """Helper to fetch training dataset, specified by FLAGS.training_dataset.

  Args:
    data_path: path to data
    batch_size: size of batch

  Returns:
    tf.data.Dataset
  """
  train_ds_kwargs = {
      'batch_size': batch_size,
      'deterministic': FLAGS.print_for_colab,
      'max_length': FLAGS.max_target_length,
      'pack': True,
      'random_seed': FLAGS.jax_random_seed,
      'repeat': True,
      'vocab_path': FLAGS.vocab_path,
  }
  logging.info('Train args: %s', train_ds_kwargs)
  if FLAGS.print_for_colab:
    print('Training data is deterministic.')
    print('Train dataset args: %s' % train_ds_kwargs)
  else:
    logging.info('training is not deterministic')
  train_ds = data.get_prepacked_examples(
      file_pattern=data_path, **train_ds_kwargs)
  return train_ds


def maybe_get_new_guide_batch(guide_batch, guide_iter, frequency,
                              guide_batch_unq_ex):
  """Update guide batch if FLAGS.guide_batch_update_freq == frequency."""
  if frequency not in ['PER_EPOCH', 'PER_TRAINING_STEP', 'PER_GUIDANCE_STEP']:
    raise ValueError('Invalid frequency: %s' % frequency)
  if FLAGS.guide_batch_update_freq == frequency:
    next_guide_batch = batch_to_numpy(next(guide_iter))
    guide_batch_unq_ex = data.get_unique_examples(next_guide_batch)
    logging.info('guide batch unq ex: %s', guide_batch_unq_ex)
    if FLAGS.print_for_colab:
      print('guide batch unq ex:', guide_batch_unq_ex)
    guide_batch = common_utils.shard(next_guide_batch)
  return guide_batch, guide_iter, guide_batch_unq_ex


def get_evalsets():
  evalsets = []
  if FLAGS.eval_dataset_path:
    evalsets.append('%s_dev' % FLAGS.training_dataset)
  if FLAGS.guidance_dataset_path and not FLAGS.train_with_guided_parameters:
    evalsets.append('%s_guide' % FLAGS.training_dataset)
  return evalsets


def get_learning_rate_fns():
  """Get learning rate fns, specified by relevant FLAGS.

  Returns:
    learning_rate_fn, learning_rate_fn_guided_vars
  """
  warmup_steps = FLAGS.warmup_steps_ratio * FLAGS.num_train_steps
  warmup_steps_guided = FLAGS.warmup_steps_ratio_guided_vars * FLAGS.num_train_steps
  if FLAGS.warmup_steps != 0:
    warmup_steps = FLAGS.warmup_steps
    logging.info(
        'Overriding FLAGS.warmup_steps_ratio with FLAGS.warmup_steps = %s',
        FLAGS.warmup_steps)
  learning_rate_fn = utils.create_learning_rate_scheduler(
      factors=FLAGS.learning_rate_schedule,
      base_learning_rate=FLAGS.learning_rate,
      warmup_steps=warmup_steps)
  guided_factors = FLAGS.learning_rate_schedule_guided_vars
  if FLAGS.use_model_lr_for_guided:
    logging.info('Using model learning rate schedule for guided parameters.')
    guided_factors = FLAGS.learning_rate_schedule
    warmup_steps_guided = warmup_steps
  learning_rate_fn_guided_vars = utils.create_learning_rate_scheduler(
      factors=guided_factors,
      base_learning_rate=FLAGS.learning_rate_guided_vars,
      warmup_steps=warmup_steps_guided)

  return learning_rate_fn, learning_rate_fn_guided_vars


def maybe_reset_guidance_model(
    guided_vars_dict,
    optimizer_dict,
    frequency,
    force_reset=False,
):
  """Reset guidance optimizers that are specified to be reset at <frequency>.

  Args:
    guided_vars_dict: Holds guidance parameter specification.
    optimizer_dict: Holds all optimizers
    frequency: Which frequency to reset, corresponds to the location of this
      method call in the training loop.
    force_reset: if True, force the reset even if the given frequency would not

  Returns:
    guided_vars_dict, optimizer_dict
  """
  if frequency not in ['reset_per_epoch', 'reset_per_batch']:
    raise ValueError('Invalid frequency: %s' % frequency)
  for var_type in guided_vars_dict:
    if guided_vars_dict[var_type][frequency] or force_reset:
      if guided_vars_dict[var_type][frequency]:
        logging.info('Resetting %s to initialization vals, due to %s tag',
                     var_type, frequency)
      else:
        logging.info(
            'Resetting %s to initialization vals, due to reset guidance per steps = %s',
            var_type, FLAGS.reset_guidance_per_steps)
      guided_vars_dict = guided_parameters.reset_subdict_raw_vars(
          guided_vars_dict, var_type)
      # import pdb
      # pdb.set_trace()
      optimizer_dict[guided_parameters.get_opt_key_from_var_type(
          var_type)] = jax_utils.replicate(
              models.init_optimizer_by_type(
                  guided_vars_dict[var_type]['raw_guided_vars'],
                  guided_vars_dict[var_type]['optimizer_type'],
                  guided_parameters.Granularity.GLOBAL))
  return guided_vars_dict, optimizer_dict


def output_train_metrics(
    t_start,
    step,
    train_step_metrics,
    train_sum_writer,
    t_metrics_timer,
    guide_step_metrics,
    extra_metrics,
):
  """Output training step metrics (may include guide step metrics).


  Args:
    t_start: For timekeeping
    step: Current model step
    train_step_metrics: Metrics from the training step.
    train_sum_writer: Summary writer for training loop.
    t_metrics_timer: keeps track of time since eval
    guide_step_metrics: Metrics dict for guide step.
    extra_metrics: Extra info regarding training run (epoch, guide ds size, etc)

  Returns:
    A timestamp, new if metrics were output else t_metrics_timer is passed
    through.
  """
  # Periodic metric handling frequency.
  logging.info('Output metrics at step: %s', step)
  # Training Metrics
  train_step_metrics = common_utils.get_metrics(train_step_metrics)

  summary = extra_metrics
  # Remove already-averaged metrics (learning rate, guidance avgs, etc).
  for k in list(train_step_metrics.keys()):
    if k.split('dataset_train/')[-1] not in [
        'loss', 'accuracy', 'denominator'
    ] and k.split('dataset_guide/')[-1] not in [
        'loss', 'accuracy', 'denominator'
    ]:
      # These metrics do not need to be summed / divided, so take them out.
      # This is 'learning_rate' and various dppl metrics if present.
      try:
        summary[k] = train_step_metrics.pop(k).mean()
      except AttributeError as e:
        logging.warning('Metrics key: %s causes error: %s. Overriding to 0.', k,
                        e)
        summary[k] = 0.0
      pass
  # Take the sums across batch for 'loss', 'accuracy', and 'denominator'.
  metrics_sums = jax.tree.map(jnp.sum, train_step_metrics)
  # Handle weighted loss.
  unweighted_keys = [x for x in metrics_sums.keys() if 'training_set_unw/' in x]
  if unweighted_keys:
    unweighted_metrics_sums = {}
    for unw_k in unweighted_keys:
      unweighted_metrics_sums[unw_k] = metrics_sums.pop(unw_k)
    denominator_unweighted = unweighted_metrics_sums.pop(
        'training_set_unw/denominator')
    summary.update(
        jax.tree.map(lambda x: x / denominator_unweighted,
                     unweighted_metrics_sums))  # pylint: disable=cell-var-from-loop
  guidance_set_keys = [x for x in metrics_sums.keys() if 'dataset_guide/' in x]
  if guidance_set_keys:
    unweighted_metrics_sums = {}
    for unw_k in guidance_set_keys:
      unweighted_metrics_sums[unw_k] = metrics_sums.pop(unw_k)
    denominator_unweighted = unweighted_metrics_sums.pop(
        'dataset_guide/denominator')
    summary.update(
        jax.tree.map(lambda x: x / denominator_unweighted,
                     unweighted_metrics_sums))  # pylint: disable=cell-var-from-loop

  # After this, only loss and accuracy are left in the metrics_sums.
  if 'dataset_train/denominator' in metrics_sums:
    denominator = metrics_sums.pop('dataset_train/denominator')
    summary.update(jax.tree.map(lambda x: x / denominator, metrics_sums))  # pylint: disable=cell-var-from-loop

  steps_per_eval = get_train_metrics_freq() if step != 0 else 1
  steps_per_sec = steps_per_eval / (time.time() - t_metrics_timer)
  if jax.process_index() == 0:
    train_sum_writer.scalar('general/steps_per_second', steps_per_sec, step)
    for key, val in summary.items():
      train_sum_writer.scalar(key, val, step)
      if '/loss' in key:
        train_sum_writer.scalar('loss/%s' % key, val, step)

    train_sum_writer.flush()
  train_step_metrics = []
  if 'loss' in summary:
    utils.log_message(t_start,
                      'train in step: %d, loss: %.4f' % (step, summary['loss']))

  # Guidance Metrics
  if guide_step_metrics:
    guide_step_metrics = common_utils.get_metrics(guide_step_metrics)
    logging.info('guide_step_metrics: %s', guide_step_metrics)
    metrics_summary = {}
    for k in list(guide_step_metrics.keys()):
      if k.split('/')[-1] not in ['loss', 'accuracy', 'denominator']:
        # These metrics do not need to be summed / divided, so take them out.
        # This is 'learning_rate' and various guidance metrics if present.
        metrics_summary[k] = guide_step_metrics.pop(k).mean()
    # Take the sums across batch for 'loss', 'accuracy', and 'denominator'.
    metrics_sums = jax.tree.map(jnp.sum, guide_step_metrics)
    # After this, only loss and accuracy are left in the metrics_sums.
    if 'dataset_guide/denominator' in metrics_sums:
      denominator = metrics_sums.pop('dataset_guide/denominator')
      # metrics_summary['dataset_guide/denominator'] = denominator
      metrics_summary.update(
          jax.tree.map(lambda x: x / denominator, metrics_sums))  # pylint: disable=cell-var-from-loop

    if jax.process_index() == 0:
      train_sum_writer.flush()
  guide_step_metrics = []
  if 'loss' in summary:
    utils.log_message(t_start,
                      'train in step: %d, loss: %.4f' % (step, summary['loss']))
  return time.time()


def get_train_metrics_freq():
  metric_frequency = int(FLAGS.num_train_steps /
                         min(FLAGS.total_evals, FLAGS.num_train_steps))
  if FLAGS.min_eval_freq:
    metric_frequency = FLAGS.min_eval_freq
  train_metric_freq = max(
      FLAGS.max_metric_freq,
      metric_frequency // FLAGS.train_metrics_per_eval_metric)
  train_metric_freq = min(train_metric_freq, FLAGS.min_train_metric_freq)
  return train_metric_freq


def output_train_metrics_this_step(step):
  train_metric_freq = get_train_metrics_freq()
  return step % train_metric_freq == 0


def output_eval_metrics_this_step(step):
  metric_frequency = int(FLAGS.num_train_steps /
                         min(FLAGS.total_evals, FLAGS.num_train_steps))
  if FLAGS.min_eval_freq:
    metric_frequency = FLAGS.min_eval_freq
  return step % metric_frequency == 0


def do_eval_all_eval_sets(
    step,
    p_eval_step,
    eval_ds_dict,
    eval_sum_writers_dict,
    model_optimizer,
):
  """Do eval for all eval datasets.

  Args:
    step: Current model step.
    p_eval_step: Parallelized evaluation step function.
    eval_ds_dict: Contains all eval datasets.
    eval_sum_writers_dict: Maps eval set names to summary writers.
    model_optimizer: The optimizer of the model.

  Returns:
  """
  logging.info('Running eval for all evalsets [%s] at step: %s.',
               list(eval_sum_writers_dict.keys()), step)
  if FLAGS.print_for_colab:
    print('Running eval for all evalsets [%s] at step: %s.' %
          (list(eval_sum_writers_dict.keys()), step))
  for eval_ds_name, eval_summary_writer in eval_sum_writers_dict.items():
    logging.info('Starting eval for: %s.', eval_ds_name)

    if FLAGS.print_for_colab:
      print('Starting eval for: %s.' % eval_ds_name)
    eval_ds = eval_ds_dict[eval_ds_name]
    t_before_eval = time.time()
    eval_summary = do_eval_single(step, eval_ds, eval_ds_name, p_eval_step,
                                  model_optimizer)
    logging.info('Finished eval for: %s.', eval_ds_name)

    eval_duration = time.time() - t_before_eval
    if FLAGS.print_for_colab:
      print('finished eval for %s in duration %s' %
            (eval_ds_name, eval_duration))
    if jax.process_index() == 0:
      eval_summary_writer.scalar('dataset_eval/duration', eval_duration, step)
      for key, val in eval_summary.items():
        # If we slipped the guidance dataset into the evals (we are training
        # without guided parameters), make sure we pull it back out into it's
        # correct category for comparison to guided runs.
        # eval_ds_simple_name = eval_ds_name.split('-')[-1]
        if 'guide-' in eval_ds_name:
          key = key.replace('dataset_eval', 'dataset_guide')
        eval_summary_writer.scalar(key, val, step)
        if '/loss' in key:
          eval_summary_writer.scalar('loss/%s' % eval_ds_name.split('-')[-1],
                                     val, step)

      eval_summary_writer.flush()
  logging.info('Finished eval of all sets at step %s', step)


def jax_synchronize_hosts():
  """Ensure all jax hosts synchronize before exiting."""
  if jax.process_count() > 1:
    # Make sure all hosts stay up until the end of main.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    assert x[0] == jax.device_count()


def main(_):
  maybe_remove_finished_training(FLAGS.model_dir)
  t_start = time.time()

  # Necessary to make it work on GPU without OOM.
  # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html#common-causes-of-oom-failures
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Number of local devices for this host.
  n_devices = jax.local_device_count()
  logging.info('JAX host: process_index %d / process_count %d',
               jax.process_index(), jax.process_count())

  if FLAGS.batch_size % n_devices:
    raise ValueError(
        'Batch size (%s) must be divisible by the number of devices (%s)' %
        (FLAGS.batch_size, n_devices))

  if FLAGS.jax_backend_target:
    jax.config.update('jax_xla_backend', 'tpu_driver')
    jax.config.update('jax_backend_target', FLAGS.jax_backend_target)

  # Output functions (summary writers, writing out flags file) need only be
  # done once.
  train_sum_writer, eval_sum_writers_dict = set_up_summary_writers()

  # Load data-parameters dict.
  utils.log_message(t_start, 'Maybe loading guided_vars_dict.')

  optimizer_dict = {}
  model_optimizer, current_step, rng, model = restore_or_init_model(t_start)
  optimizer_dict['model'] = jax_utils.replicate(model_optimizer)
  # Replicate optimizer.
  del model_optimizer  # Only use through dict.
  del model  # Only reference through dict.

  extra_metrics = {
      'general/num_params_model':
          sum(x.size for x in jax.tree.leaves(optimizer_dict['model'].target)),
  }

  # this allows us to set separate learning_rate_scalar values for each guided
  # parameter via FLAGS. All guided params share a learning rate, but this
  # scalar is multiplied by that learning rate for each one separately, allowing
  # the params to take different learning rates.
  learning_rate_scalar_override = {
      'decay_rate': 1,
      'beta1': FLAGS.ghp_lrs_beta1,
      'eps': FLAGS.ghp_lrs_eps,
      'label_smoothing': FLAGS.ghp_lrs_ls,
      'learning_rate': FLAGS.ghp_lrs_lr,
      'weight_decay': FLAGS.ghp_lrs_wd,
  }

  guided_vars_dict = None

  if FLAGS.train_with_guided_parameters:
    opt_kwargs = _get_default_opt_kwargs(
        model_optimizer_type=FLAGS.model_optimizer_type,
        lr=1,
    )
    guided_vars_dict = guided_parameters.get_guided_vars_dict(
        guided_hparam_types=FLAGS.guided_hparam_types,
        model_opt_type=FLAGS.model_optimizer_type,
        guided_opt_type=FLAGS.guided_params_optimizer_type,
        init_dict=opt_kwargs,
        learning_rate_scalar_override=learning_rate_scalar_override,
    )

    if jax.process_index() == 0:
      guided_parameters.save_guided_vars_dict(guided_vars_dict, FLAGS.model_dir)
  # Write flags/guided_vars_dict state to CNS.
  write_flags_files(FLAGS.model_dir, guided_vars_dict)
  utils.log_message(
      t_start,
      'Loaded model, current training step: %s, model optimizer step (total steps since random init): %s'
      % (current_step, get_step_from_opt(optimizer_dict['model'])))

  # Set learning rate fns.
  learning_rate_fn, learning_rate_fn_hyperparams = get_learning_rate_fns()

  # Load Dataset
  utils.log_message(t_start, 'Initializing dataset.')
  if FLAGS.print_for_colab:
    print('Initializing dataset.')
  if FLAGS.training_dataset_path:
    logging.info('FLAGS.training_dataset_path  %s', FLAGS.training_dataset_path)

    train_ds_kwargs = {
        'batch_size': FLAGS.batch_size,
        'deterministic': FLAGS.print_for_colab,
        'max_length': FLAGS.max_target_length,
        'pack': True,
        'random_seed': FLAGS.jax_random_seed,
        'repeat': True,
        'vocab_path': FLAGS.vocab_path,
    }
    train_ds = data.get_prepacked_examples(
        file_pattern=FLAGS.training_dataset_path, **train_ds_kwargs)
  if FLAGS.guidance_dataset_path:
    logging.info('FLAGS.guidance_dataset_path  %s', FLAGS.guidance_dataset_path)

    eval_ds_dict = {}
    guide_ds_kwargs = {
        'batch_size': FLAGS.guide_batch_size,
        'deterministic': FLAGS.print_for_colab,
        'drop_remainder': False,
        'max_length': FLAGS.max_target_length,
        'pack': True,
        'random_seed': FLAGS.jax_random_seed,
        'repeat': False,
        'vocab_path': FLAGS.vocab_path,
        'shard_data': False,
    }
    guide_ds = data.get_prepacked_examples(
        file_pattern=FLAGS.guidance_dataset_path, **guide_ds_kwargs)
    if not FLAGS.train_with_guided_parameters:
      eval_ds_dict['eval-%s_guide' % FLAGS.training_dataset] = guide_ds
    if FLAGS.guide_batch_update_freq == 'NEVER' and FLAGS.training_dataset != 'guide':
      logging.info('Truncating guidance dataset to a single batch')
      if FLAGS.print_for_colab:
        print('Truncating guidance dataset to a single batch')
      guide_ds = guide_ds.take(jax.local_device_count()).repeat()
  if FLAGS.eval_dataset_path:
    logging.info('FLAGS.eval_dataset_path  %s', FLAGS.eval_dataset_path)
    eval_ds_kwargs = {
        'batch_size': FLAGS.eval_batch_size,
        'deterministic': True,
        'drop_remainder': False,
        'max_length': FLAGS.max_target_length,
        'pack': True,
        'random_seed': FLAGS.jax_random_seed,
        'repeat': False,
        'shard_data': False,
        'vocab_path': FLAGS.vocab_path,
    }
    eval_ds = data.get_prepacked_examples(
        file_pattern=FLAGS.eval_dataset_path,
        **eval_ds_kwargs).take(jax.local_device_count())
    eval_ds_dict['eval-%s_dev' % FLAGS.training_dataset] = eval_ds
  logging.info('train keys: %s')
  if FLAGS.print_for_colab:
    print('train keys:')
    for x in next(iter(train_ds)).keys():
      print('\t', x)
    if FLAGS.train_with_guided_parameters:
      print('guide keys: %s')
      for x in next(iter(guide_ds)).keys():
        print('\t', x)

  guide_batch_unq_ex = 0
  train_batch_unq_ex = 0
  if FLAGS.train_with_guided_parameters:
    guide_iter = iter(guide_ds)  # This is an infinite iterator.
    guide_batch = None
    if FLAGS.guide_batch_update_freq == 'NEVER':
      guide_batch = batch_to_numpy(next(guide_iter))
      # print guide_batch unique examples count
      guide_batch_unq_ex = data.get_unique_examples(guide_batch)
      logging.info('Guide batch unique examples: %s', guide_batch_unq_ex)
      if FLAGS.print_for_colab:
        print('Guide batch unique examples:', guide_batch_unq_ex)
      guide_batch = common_utils.shard(guide_batch)

    if FLAGS.print_for_colab:
      print('Guide batch update freq:', FLAGS.guide_batch_update_freq)

  # Load guided params state.
  if FLAGS.train_with_guided_parameters:
    guided_optimizers = restore_or_init_guided_optimizers(
        t_start, guided_vars_dict)
    optimizer_dict.update(guided_optimizers)

  # set up model pmapped update fns
  if not FLAGS.train_with_guided_parameters:
    p_update_model_step = jax.pmap(
        functools.partial(
            update_model_step,
            learning_rate_fn=learning_rate_fn,
            use_bfloat16=FLAGS.use_bfloat16),
        axis_name='batch')
  # Set up guided vars pmapped step fns.
  elif FLAGS.train_with_guided_parameters:
    utils.log_message(t_start, 'Setting up p_update_model_and_guidance_step.')
    p_update_model_and_guidance_step = jax.pmap(
        functools.partial(
            update_model_and_guidance_step,
            learning_rate_fn=learning_rate_fn,
            learning_rate_fn_guided_vars=learning_rate_fn_hyperparams,
            guided_vars_dict=guided_vars_dict,
            grad_clip_limit=FLAGS.grad_clip_limit,
            use_bfloat16=FLAGS.use_bfloat16,
        ),
        axis_name='batch')

  p_eval_step = jax.pmap(
      functools.partial(eval_step, use_bfloat16=FLAGS.use_bfloat16),
      axis_name='batch')
  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  dropout_rngs = random.split(rng, n_devices)

  utils.log_message(t_start, 'Starting training loop.')
  if FLAGS.print_for_colab:
    print(t_start, 'Starting training loop.')
  train_step_metrics = []
  guide_step_metrics = []
  metrics_train = None
  metrics_joint = None
  metrics_guide = None

  t_metrics_timer = time.time()
  # Training loop
  delta_guide_loss_dict = {}
  steps_since_last_model_ckpt = 0
  steps_since_last_dp_ckpt = 0
  steps_since_preemption = 0
  current_epoch = guided_parameters.load_epoch(FLAGS.model_dir)
  logging.info('Loaded epoch: %s', current_epoch)

  if current_step == 0 and FLAGS.save_checkpoint_at_init:
    # This is not a restart after pre-emption.
    logging.info('saving checkpoint 0: current_step == %s', current_step)
    save_all_checkpoints(t_start, optimizer_dict, current_step)
    # If not random-init, do eval before starting training.
    if FLAGS.init_checkpoint and FLAGS.do_eval:
      logging.info('Doing eval before training.')
      do_eval_all_eval_sets(current_step, p_eval_step, eval_ds_dict,
                            eval_sum_writers_dict, optimizer_dict['model'])

  # Loop through epochs.
  for epoch in range(current_epoch, FLAGS.max_train_epochs):
    logging.info('Starting epoch loop for epoch %s', epoch)
    if FLAGS.print_for_colab:
      print('Starting epoch loop for epoch %s' % epoch)
    if jax.process_index() == 0:
      guided_parameters.save_epoch(epoch, FLAGS.model_dir)
    if FLAGS.num_train_steps > 0 and current_step > FLAGS.num_train_steps:
      logging.info(
          'Breaking out of trianing loop as current step %s > FLAGS.num_train_steps %s',
          current_step, FLAGS.num_train_steps)
      if FLAGS.print_for_colab:
        print(
            f'Breaking out of trianing loop as current step {current_step} > FLAGS.num_train_steps {FLAGS.num_train_steps}'
        )
      break

    steps_per_epoch = 0
    # Refresh the non-repeating (single epoch) train_ds iterator.
    if FLAGS.train_with_guided_parameters:
      guide_batch, guide_iter, guide_batch_unq_ex = maybe_get_new_guide_batch(
          guide_batch, guide_iter, 'PER_EPOCH', guide_batch_unq_ex)

      # maybe_reset_guidance_model_epoch
      # this must go after the filtering! otherwise no filtering will occur due
      # to reset values. Don't filter on the first (0th) epoch.
      if epoch != 0:
        guided_vars_dict, optimizer_dict = maybe_reset_guidance_model(
            guided_vars_dict, optimizer_dict, 'reset_per_epoch')
    train_iter = iter(train_ds)

    # Per epoch loop.
    epoch_step = 0
    finish_epoch = False
    for train_batch in train_iter:
      if FLAGS.num_train_steps > 0 and current_step > FLAGS.num_train_steps:
        break
      if finish_epoch:
        finish_epoch = False
        break
      # Shard data for multiple devices.
      train_batch = batch_to_numpy(train_batch)
      if FLAGS.guide_with_train_loss:
        guide_batch = common_utils.shard(train_batch)
      train_batch_unq_ex = data.get_unique_examples(train_batch)

      epoch_step += 1
      if FLAGS.num_train_steps > 0 and current_step > FLAGS.num_train_steps:
        logging.info('num_train_steps (%s) reached', FLAGS.num_train_steps)
        save_all_checkpoints(t_start, optimizer_dict, current_step)
        logging.info('Saving ckpts and quitting.')
        break
      current_step += 1
      steps_per_epoch += 1
      steps_since_preemption += 1
      if current_step % 1000 == 0:
        utils.log_message(t_start, 'Current_step: %s' % current_step)

      if FLAGS.train_with_guided_parameters:
        if FLAGS.reset_guidance_per_steps and current_step % FLAGS.reset_guidance_per_steps == 0:
          guided_vars_dict, optimizer_dict = maybe_reset_guidance_model(
              guided_vars_dict,
              optimizer_dict,
              'reset_per_epoch',
              force_reset=True)
          if FLAGS.print_for_colab:
            print('resetting guided vars at step: %s' % current_step)
        guide_batch, guide_iter, guide_batch_unq_ex = maybe_get_new_guide_batch(
            guide_batch, guide_iter, 'PER_TRAINING_STEP', guide_batch_unq_ex)

        if current_step % (max(FLAGS.num_train_steps // 10,
                               1)) == 0 or current_step < 10:
          logging.info('train step %s', current_step)
          if FLAGS.print_for_colab:
            print('train step %s' % current_step)
        sharded_train_batch = common_utils.shard(train_batch)

        # Either joint optimization, or run validation and train steps in seq.
        joint_step_kwargs = {
            'train_batch': sharded_train_batch,
            'dropout_rng': dropout_rngs,
            'guide_batch': guide_batch,
        }
        if output_train_metrics_this_step(current_step):
          joint_step_kwargs.update({
              'output_metrics': jnp.array([True] * jax.local_device_count()),
          })
        optimizer_dict, metrics_joint, dropout_rngs = p_update_model_and_guidance_step(
            optimizer_dict, **joint_step_kwargs)
      else:
        if FLAGS.train_with_guided_parameters:
          # using update_model_multiweighted_train_loss_dict()
          optimizer_dict, metrics_train, dropout_rngs = p_update_model_step(
              optimizer_dict,
              train_batch=common_utils.shard(train_batch),
              dropout_rng=dropout_rngs)
        else:
          # using update_model_step()
          optimizer_dict, metrics_train, dropout_rngs = p_update_model_step(
              optimizer_dict,
              train_batch=common_utils.shard(train_batch),
              dropout_rng=dropout_rngs)

      if metrics_train:
        train_step_metrics.append(metrics_train)
      if metrics_guide:
        guide_step_metrics.append(metrics_guide)
      if metrics_joint:
        train_step_metrics.append(metrics_joint)

      extra_metrics.update({
          'general/steps_since_preemption': steps_since_preemption,
          'general/epoch': epoch,
          'dataset_train/train_batch_unq_ex': train_batch_unq_ex,
          'dataset_guide/guide_batch_unq_ex': guide_batch_unq_ex,
          'general/steps_per_epoch': steps_per_epoch,
          'hparams/dropout_rate': FLAGS.dropout_rate,
          'hparams/attention_dropout_rate': FLAGS.attention_dropout_rate,
          'hparams/label_smoothing': FLAGS.label_smoothing,
      })
      # TODO(lichtarge): Calculate change in guidance loss, add to metrics.
      for key, v in delta_guide_loss_dict.items():
        extra_metrics.update({'delta_guide_loss-%s' % key: v})
      if output_train_metrics_this_step(current_step):
        logging.info('Output train metrics on step: %s', current_step)
        t_metrics_timer = output_train_metrics(
            t_start,
            current_step,
            train_step_metrics,
            train_sum_writer,
            t_metrics_timer,
            guide_step_metrics,
            extra_metrics,
        )
        extra_metrics = {}
      if output_eval_metrics_this_step(current_step):
        if FLAGS.do_eval:
          do_eval_all_eval_sets(current_step, p_eval_step, eval_ds_dict,
                                eval_sum_writers_dict, optimizer_dict['model'])
      train_step_metrics, guide_step_metrics = [], []
      steps_since_last_model_ckpt, steps_since_last_dp_ckpt = maybe_save_ckpts(
          t_start, optimizer_dict, current_step, steps_since_last_model_ckpt,
          steps_since_last_dp_ckpt)

    logging.info('Epoch finished! current_step: %s, epoch %s, steps per %s',
                 current_step, epoch, steps_per_epoch)
    if FLAGS.print_for_colab:
      print('EPOCH! current_step: %s, epoch %s, steps per %s' %
            (current_step, epoch, steps_per_epoch))
    if FLAGS.save_ckpt_per_epoch:
      logging.info('saving all ckpts at end of epoch')
      save_all_checkpoints(t_start, optimizer_dict, current_step)
      steps_since_last_model_ckpt = 0

  # Done with training loop.
  write_finished_training(FLAGS.model_dir)

  if FLAGS.print_for_colab:
    print('Finished training loop.')
  jax_synchronize_hosts()


if __name__ == '__main__':
  app.run(main)
