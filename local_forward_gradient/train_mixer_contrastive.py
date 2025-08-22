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

# pylint: skip-file
"""A basic CIFAR example using Numpy and JAX.

The primary aim here is simplicity and minimal dependencies.
"""
import math
import os
import time
import pickle as pkl
import jax
import tensorflow as tf
from mixer_lib import (fa_group_linear, fa_linear, depthwise_conv, get_blk,
                       get_blk_idx, get_dataset_metadata,
                       get_layer_sizes, get_param_scale, init_random_params,
                       linear, group_linear, normalize, preprocess, get_num_layers)
from mixer_lib import NFIRST, NLAYER

# Ask TF to not occupy GPU memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import augmentations
from dataset import (_decode_and_center_crop, _decode_and_random_crop)
from repeated_dot_product_v2_pmap import (
    spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_jvp,
    spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_vjp)
from train_utils import (save_checkpoint, last_checkpoint)

from absl import app
from absl import flags
from jax import grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import optax
from clu import metric_writers
import tensorflow_datasets as tfds

flags.DEFINE_string('data_root', None, 'where to store datasets')
flags.DEFINE_string('exp', 'all', 'which experiment to run')
flags.DEFINE_string('workdir', '', 'experiment directory')
flags.DEFINE_float('mom', 0.0, 'momentum')
flags.DEFINE_string('lr', '0.1', 'learning rate')
flags.DEFINE_string('optimizer', 'sgd', 'optimizer name')
flags.DEFINE_float('wd', 0.0001, 'weight decay')
flags.DEFINE_integer('warmup_epochs', 1, 'number of warmup epochs')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('num_epochs', 40, 'number of epochs')
flags.DEFINE_string('dataset', 'cifar-10', 'dataset to run')
flags.DEFINE_string('init_scheme', 'kaiming',
                    'kaiming or lecun or constant or list')
flags.DEFINE_integer('num_blocks', 6, 'number of mixer blocks')
flags.DEFINE_integer('num_channel_mlp_units', 256, 'number of units')
flags.DEFINE_integer('num_channel_mlp_hidden_units', -1,
                     'number of hidden units')
flags.DEFINE_integer('num_proj_units', 128, 'number of units')
flags.DEFINE_integer('num_patches', 8, 'number of patches on each side')
flags.DEFINE_integer('num_groups', 1, 'number of groups')
flags.DEFINE_float('last_layer_lr', 0.1, 'last layer learning rate')
flags.DEFINE_integer('num_passes', 1, 'number of forward gradient passes')
flags.DEFINE_bool('layer_norm_all', False, 'entire layer')
flags.DEFINE_integer('stop_every', 1, 'stop gradient every')
flags.DEFINE_integer('stop_remainder', -1, 'stop gradient remainder')
flags.DEFINE_string('schedule', 'linear', 'learning rate schedule')
flags.DEFINE_float('temp', 0.5, 'SimCLR loss temperature')
flags.DEFINE_float('area_lb', 0.08, 'area crop lower bound')
flags.DEFINE_bool('batch_norm', False, 'if we run batch norm')
flags.DEFINE_bool('avgpool_token', True, 'average pool tokens before head')
flags.DEFINE_bool('concat_groups', True, 'whether to concat groups')
flags.DEFINE_float('head_lr', 1.0, 'classifier head lr multiplier')
flags.DEFINE_bool('norm_grad', True, 'normalization layer gradients')
flags.DEFINE_bool('aug', True, 'add random crop augmentation')
flags.DEFINE_bool('simclr_sg', True, 'add stop_gradient in simclr')
flags.DEFINE_bool('gaussian_blur', True, 'whether to perform blurring')
flags.DEFINE_bool('greedy', False, 'whether do greedy training')
flags.DEFINE_bool('same_head', True, 'whether use same head')
flags.DEFINE_bool('begin_ln', False, 'whether to layer norm in the beginning')
flags.DEFINE_bool('middle_ln', False, 'whether to add middle LN')
flags.DEFINE_bool('last_layer_ln', True, 'whether to layer norm in the last')
flags.DEFINE_bool('post_linear_ln', True, 'whether to layer norm post linear')
flags.DEFINE_integer('jax_trace', 1000, 'num epochs per trace')
flags.DEFINE_bool('conv_mixer', False, 'use conv for mixing')
flags.DEFINE_integer('kernel_size', 3, 'conv kernel size')
flags.DEFINE_bool('jax_profiler', False, 'whether to start jax profiler')
flags.DEFINE_bool('stopgrad_input', False, 'whether to stopgrad on the input')
flags.DEFINE_bool('freeze_backbone', False, 'whether freeze')
flags.DEFINE_bool('spatial_loss', True,
                  'whether to keep the spatial dimensions')
flags.DEFINE_bool('modular_loss', True,
                  'whether to have loss functions in each module')
flags.DEFINE_string('gcs_path', 'gs://research-brain-rmy-gcp-xgcp',
                    'cloud storage')
flags.DEFINE_bool('use_gcs', False, 'whether to use cloud storage')
flags.DEFINE_bool('linear_scale', True, 'whether to scale learning rate')

FLAGS = flags.FLAGS


def normalize_layer(x, swap=False):
  return normalize(x,
                   swap=swap,
                   batch_norm=FLAGS.batch_norm,
                   layer_norm_all=FLAGS.layer_norm_all)


def mlp_block(inputs,
              params,
              noise=None,
              name='',
              mask=None,
              stop_every_layer=False):
  # Token mixing.
  conv = FLAGS.conv_mixer
  states = {}
  B, P, G, D = inputs.shape
  H = int(math.sqrt(P))
  if FLAGS.stopgrad_input:
    inputs_ = jax.lax.stop_gradient(inputs)
  else:
    inputs_ = inputs
  outputs = jnp.reshape(inputs_, [B, P, -1])
  if FLAGS.begin_ln:
    outputs = normalize_layer(outputs)
  if conv:
    outputs = jnp.reshape(outputs, [B, H, H, G * D])
    states[f'{name}/pre_0'] = outputs
    if len(params[0]) == 3:  # Feedback alignment
      outputs = depthwise_conv(outputs, params[0][0])
      outputs_bw = depthwise_conv(outputs, params[0][2])
      outputs = jax.lax.stop_gradient(outputs -
                                      outputs_bw) + outputs_bw + params[0][1]
    else:
      outputs = depthwise_conv(outputs, params[0][0]) + params[0][1]
    outputs = jnp.reshape(outputs, [B, P, -1])
  else:
    outputs = jnp.swapaxes(outputs, 1, 2)
    states[f'{name}/pre_0'] = outputs
    if len(params[0]) == 3:  # Feedback alignment
      outputs = fa_linear(outputs, params[0][0], params[0][1], params[0][2])
    else:
      outputs = linear(outputs, params[0][0], params[0][1])
  states[f'{name}/prenorm_0'] = outputs
  if FLAGS.post_linear_ln:
    outputs = normalize_layer(outputs, swap=not conv)

  if noise is not None:
    outputs = outputs + noise[0]
  outputs = jax.nn.relu(outputs)
  if mask is not None:
    outputs = outputs * jnp.swapaxes(mask, 1, 2)
  states[f'{name}/post_0'] = outputs

  if stop_every_layer:
    outputs = jax.lax.stop_gradient(outputs)

  if not conv:
    outputs = jnp.swapaxes(outputs, 1, 2)

  # Channel mixing.
  if FLAGS.middle_ln:
    outputs = normalize_layer(outputs)
  states[f'{name}/pre_1'] = outputs
  if len(params[1]) == 3:  # Feedback alignment
    outputs = fa_linear(outputs, params[1][0], params[1][1], params[1][2])
  else:
    outputs = linear(outputs, params[1][0], params[1][1])
  states[f'{name}/prenorm_1'] = outputs
  if FLAGS.post_linear_ln:
    outputs = normalize_layer(outputs)
  if noise is not None:
    outputs = outputs + noise[1]
  outputs = jax.nn.relu(outputs)
  if mask is not None:
    outputs = outputs * mask
  states[f'{name}/post_1'] = outputs

  if stop_every_layer:
    outputs = jax.lax.stop_gradient(outputs)

  outputs = jnp.reshape(outputs, [B, P, G, D])
  if FLAGS.middle_ln:
    outputs = normalize_layer(outputs)
  states[f'{name}/pre_2'] = outputs
  if len(params[2]) == 3:  # Feedback alignment
    outputs = fa_group_linear(outputs, params[2][0], params[2][1],
                              params[2][2])
  else:
    outputs = group_linear(outputs, params[2][0], params[2][1])
  states[f'{name}/prenorm_2'] = outputs
  if FLAGS.post_linear_ln:
    outputs = normalize_layer(outputs)
  if noise is not None:
    outputs = outputs + noise[2]
  outputs = outputs + inputs
  outputs = jax.nn.relu(outputs)
  if mask is not None:
    outputs = outputs * mask
  states[f'{name}/post_2'] = outputs

  if stop_every_layer:
    outputs = jax.lax.stop_gradient(outputs)
  return outputs, (states, {})


def block0(inputs,
           params,
           num_groups,
           noise=None,
           name='',
           mask=None,
           stop_every_layer=False):
  states = {}
  outputs = inputs
  if FLAGS.begin_ln:
    outputs = normalize_layer(outputs)
  states[f'{name}/pre_0'] = outputs
  if len(params[0]) == 3:  # Feedback alignment
    outputs = fa_linear(outputs, params[0][0], params[0][1], params[0][2])
  else:
    outputs = linear(outputs, params[0][0], params[0][1])
  states[f'{name}/prenorm_0'] = outputs
  if FLAGS.post_linear_ln:
    outputs = normalize_layer(outputs)
  if noise is not None:
    outputs = outputs + noise[0]
  outputs = jax.nn.relu(outputs)
  if mask is not None:
    outputs = outputs * mask
  states[f'{name}/post_0'] = outputs
  if stop_every_layer:
    outputs = jax.lax.stop_gradient(outputs)

  B, P, D = outputs.shape
  G = num_groups
  outputs = jnp.reshape(outputs, [B, P, G, -1])
  if FLAGS.middle_ln:
    outputs = normalize_layer(outputs)
  states[f'{name}/pre_1'] = outputs
  if len(params[1]) == 3:  # Feedback alignment
    outputs = fa_group_linear(outputs, params[1][0], params[1][1],
                              params[1][2])
  else:
    outputs = group_linear(outputs, params[1][0], params[1][1])
  states[f'{name}/prenorm_1'] = outputs
  if FLAGS.post_linear_ln:
    outputs = normalize_layer(outputs)
  if noise is not None:
    outputs = outputs + noise[1]
  outputs = jax.nn.relu(outputs)
  if mask is not None:
    outputs = outputs * mask
  states[f'{name}/post_1'] = outputs
  if stop_every_layer:
    outputs = jax.lax.stop_gradient(outputs)
  return outputs, (states, {})


def run_block(block_idx,
              num_groups,
              inputs,
              block_params,
              block_noise,
              stop_every_layer=False):
  if block_idx == 0:
    outputs, (states, logs) = block0(inputs,
                                     block_params,
                                     num_groups,
                                     block_noise,
                                     name=f'block_{block_idx}',
                                     stop_every_layer=stop_every_layer)
  elif block_idx > 0:
    outputs, (states, logs) = mlp_block(inputs,
                                        block_params,
                                        block_noise,
                                        name=f'block_{block_idx}',
                                        stop_every_layer=stop_every_layer)
  else:
    # Negative number is the final block.
    states = {}
    x = jnp.reshape(inputs, [inputs.shape[0], inputs.shape[1], -1])
    x = jnp.mean(x, axis=1)  # [B, D]
    # For supervised classification readout (unsupervised rep learning).
    x = jax.lax.stop_gradient(x)
    states[f'pre_cls'] = x
    pred_cls = linear(x, block_params[-1][0], block_params[-1][1])
    states['pred_cls'] = pred_cls
    outputs = pred_cls
    logs = {}
  return outputs, (states, logs)


def predict(params,
            inputs,
            noise=None,
            stop_gradient=False,
            readout=False,
            stop_every=1,
            stop_remainder=-1,
            is_training=False,
            stop_every_layer=False):
  """MLP mixer"""
  NBLK = FLAGS.num_blocks
  if stop_remainder < 0:
    stop_remainder = stop_remainder + stop_every

  md = get_dataset_metadata(FLAGS.dataset)
  inputs = preprocess(inputs, md['image_mean'], md['image_std'],
                      FLAGS.num_patches)
  x = inputs
  # We will start with a channel mixing MLP instead of token mixing.
  all_states = {}
  all_logs = {}
  # Build network.
  for blk in range(NBLK):
    start, end = get_blk_idx(blk)
    if noise is not None:
      noise_ = noise[start:end]
    else:
      noise_ = None
    x, (states, logs) = run_block(blk,
                                  FLAGS.num_groups,
                                  x,
                                  params[start:end],
                                  noise_,
                                  stop_every_layer=stop_every_layer)

    for k in states:
      all_states[k] = states[k]
    for k in logs:
      all_logs[k] = logs[k]

    if stop_gradient and not FLAGS.stopgrad_input:
      if blk % stop_every == stop_remainder:
        x = jax.lax.stop_gradient(x)

  if readout:
    x = jax.lax.stop_gradient(x)

  x = jnp.reshape(x, [x.shape[0], x.shape[1], -1])
  x = jnp.mean(x, axis=1)  # [B, D]
  all_states[f'pre_final'] = x
  # [B, K]
  pred = linear(x, params[-2][0], params[-2][1])
  all_states['pred_final'] = pred

  # For supervised classification readout (unsupervised rep learning).
  x = jax.lax.stop_gradient(x)
  if FLAGS.last_layer_ln:
    x = normalize_layer(x)
  all_states[f'pre_cls'] = x
  pred_cls = linear(x, params[-1][0], params[-1][1])
  all_states['pred_cls'] = pred_cls
  return pred_cls, (all_states, all_logs)


def softmax_cross_entropy(logits, targets, reduction="none"):
  logits = logits - logsumexp(logits, axis=-1, keepdims=True)
  loss = -jnp.sum(logits * targets, axis=-1)
  if reduction == "mean":
    loss = jnp.mean(loss)
  return loss


def l2_normalize(x, axis=None, epsilon=1e-12):
  """l2 normalize a tensor on an axis with numerical stability."""
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return x * x_inv_norm


def _simclr_loss(x, y, temp, labels_idx):
  normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
  is_parallel = jax.device_count() > 1
  D = normed_x.shape[1]
  if is_parallel:
    # [N,D] -> [MN,D]
    x_all = jnp.reshape(jax.lax.all_gather(normed_x, axis_name='i'), [-1, D])
    y_all = jnp.reshape(jax.lax.all_gather(normed_y, axis_name='i'), [-1, D])
  else:
    x_all = normed_x
    y_all = normed_y

  # B = x.shape[0]
  Nall = x_all.shape[0]
  LARGE_NUM = 1e9
  # labels_idx = jnp.arange(B)
  labels = jax.nn.one_hot(labels_idx, Nall * 2)
  masks = jax.nn.one_hot(labels_idx, Nall)
  logits_aa = jnp.dot(normed_x,
                      jnp.transpose(x_all)) / temp - masks * LARGE_NUM
  logits_ab = jnp.dot(normed_x, jnp.transpose(y_all)) / temp
  logits_ba = jnp.dot(normed_y, jnp.transpose(x_all)) / temp
  logits_bb = jnp.dot(normed_y,
                      jnp.transpose(y_all)) / temp - masks * LARGE_NUM
  logits_a = jnp.concatenate([logits_ab, logits_aa], axis=1)
  logits_b = jnp.concatenate([logits_ba, logits_bb], axis=1)
  loss_a = softmax_cross_entropy(logits_a, labels, reduction='none')
  loss_b = softmax_cross_entropy(logits_b, labels, reduction='none')
  return loss_a, loss_b


def _simclr_loss_sg(x, y, temp, labels_idx):
  normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
  is_parallel = jax.device_count() > 1
  D = normed_x.shape[1]
  if is_parallel:
    # [N,D] -> [MN,D]
    x_all = jnp.reshape(jax.lax.all_gather(normed_x, axis_name='i'), [-1, D])
    y_all = jnp.reshape(jax.lax.all_gather(normed_y, axis_name='i'), [-1, D])
  else:
    x_all = normed_x
    y_all = normed_y

  # B = x.shape[0]
  LARGE_NUM = 1e9
  Nall = x_all.shape[0]
  # labels_idx = jnp.arange(B)
  labels = jax.nn.one_hot(labels_idx, Nall * 2)
  masks = jax.nn.one_hot(labels_idx, Nall)
  x_all_sg = jax.lax.stop_gradient(x_all)
  y_all_sg = jax.lax.stop_gradient(y_all)
  # [N, D] x [M, D] -> [N, M]
  logits_aa = jnp.dot(normed_x,
                      jnp.transpose(x_all_sg)) / temp - masks * LARGE_NUM
  logits_ab = jnp.dot(normed_x, jnp.transpose(y_all_sg)) / temp
  logits_ba = jnp.dot(normed_y, jnp.transpose(x_all_sg)) / temp
  logits_bb = jnp.dot(normed_y,
                      jnp.transpose(y_all_sg)) / temp - masks * LARGE_NUM
  logits_a = jnp.concatenate([logits_ab, logits_aa], axis=1)
  logits_b = jnp.concatenate([logits_ba, logits_bb], axis=1)
  loss_a = softmax_cross_entropy(logits_a, labels, reduction='none')
  loss_b = softmax_cross_entropy(logits_b, labels, reduction='none')
  return loss_a, loss_b


def _local_simclr_loss_fused(x,
                             y,
                             w,
                             b,
                             temp,
                             labels_idx,
                             custom_forward=False):
  """Use the avg feature of one side to guide the other side."""
  # x: [B, P, G, D]
  # y: [B, P, G, D]
  if custom_forward:
    repeated_dp_ce = spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_jvp
  else:
    repeated_dp_ce = spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_vjp
  loss = repeated_dp_ce(x, y, w, b, temp, labels_idx)
  return loss


def simclr_loss(x, y, temp, labels_idx):
  loss_a, loss_b = _simclr_loss(x, y, temp, labels_idx)
  return loss_a + loss_b


def simclr_loss_sg(x, y, temp, labels_idx):
  loss_a, loss_b = _simclr_loss_sg(x, y, temp, labels_idx)
  return jnp.concatenate([loss_a, loss_b], axis=0)


def local_simclr_loss_sg(x, y, w, b, temp, labels_idx, custom_forward=False):
  loss = _local_simclr_loss_fused(x,
                                  y,
                                  w,
                                  b,
                                  temp,
                                  labels_idx,
                                  custom_forward=custom_forward)
  return loss


def loss_dfa(params, batch, noise=None, key=None):
  md = get_dataset_metadata(FLAGS.dataset)
  key, subkey = jax.random.split(key)
  batch = augmentations.postprocess(batch,
                                    subkey,
                                    add_gaussian_blur=FLAGS.gaussian_blur)
  view1, view2, targets, labels_idx = batch["view1"], batch["view2"], batch[
      "labels"], batch["labels_idx"]
  targets = jax.nn.one_hot(targets, md["num_classes"])
  NBLK = FLAGS.num_blocks
  NL = get_num_layers(NBLK)
  B = view1.shape[0]
  num_patches = FLAGS.num_patches
  P = num_patches**2
  G = FLAGS.num_groups
  _, (states1, logs) = predict(
      params,
      view1,
      noise=noise,
      stop_gradient=True,
      readout=True,
      is_training=True,
      stop_every_layer=True,
  )
  _, (states2, _) = predict(params,
                            view2,
                            noise=noise,
                            stop_gradient=True,
                            readout=True,
                            is_training=True,
                            stop_every_layer=True)

  # Last block avg pool contrastive loss.
  proj1 = states1[f'pred_final']
  proj2 = states2[f'pred_final']
  if FLAGS.simclr_sg:
    loss_ssl = jnp.mean(simclr_loss_sg(proj1, proj2, FLAGS.temp,
                                       labels_idx)) * 2.0
  else:
    loss_ssl = jnp.mean(simclr_loss(proj1, proj2, FLAGS.temp, labels_idx))

  logs['local_loss/final'] = loss_ssl
  loss = loss_ssl
  local_losses = []

  for j, (weights, bias, weights_b, bias_b) in enumerate(params[:NL]):
    blk, layer = get_blk(j)
    x1 = states1[f'block_{blk}/post_{layer}']
    B, P = x1.shape[0], x1.shape[1]
    x1 = jnp.mean(x1, [1])
    x1 = jnp.reshape(x1, [B, -1])
    x2 = states2[f'block_{blk}/post_{layer}']
    x2 = jnp.mean(x2, [1])
    x2 = jnp.reshape(x2, [B, -1])
    x_proj1 = linear(x1, weights_b, bias_b)
    x_proj2 = linear(x2, weights_b, bias_b)
    if FLAGS.simclr_sg:
      local_loss_bw_ = jnp.mean(
          simclr_loss_sg(x_proj1, x_proj2, FLAGS.temp, labels_idx)) * 2.0
    else:
      local_loss_bw_ = jnp.mean(
          simclr_loss(x_proj1, x_proj2, FLAGS.temp, labels_idx))
    local_loss_ = jax.lax.stop_gradient(loss - local_loss_bw_) + local_loss_bw_
    local_losses.append(local_loss_)
    logs[f'local_loss/blk_{blk}'] = local_loss_
  return loss, local_losses, (states1, states2, logs)


def classif_loss(params,
                 batch,
                 noise=None,
                 stop_gradient=False,
                 key=None,
                 stop_every=1,
                 stop_remainder=-1):
  inputs, targets = batch["image"], batch["labels"]
  md = get_dataset_metadata(FLAGS.dataset)
  targets = jax.nn.one_hot(targets, md['num_classes'])
  logits, (states, logs) = predict(params,
                                   inputs,
                                   noise=noise,
                                   stop_gradient=stop_gradient,
                                   readout=True,
                                   stop_every=stop_every,
                                   stop_remainder=stop_remainder,
                                   is_training=True)
  classif_loss = softmax_cross_entropy(logits, targets, reduction="mean")
  logs['loss'] = classif_loss
  return classif_loss, (states, logs)


def local_loss(block_idx,
               block_params,
               batch,
               key=None,
               noise=None,
               custom_forward=False):
  if block_idx == 0:
    key, subkey = jax.random.split(key)
    batch = augmentations.postprocess(batch,
                                      subkey,
                                      add_gaussian_blur=FLAGS.gaussian_blur)
    view1, view2 = batch["view1"], batch["view2"]
    # [B,P,D]
    md = get_dataset_metadata(FLAGS.dataset)
    view1 = preprocess(view1, md['image_mean'], md['image_std'],
                       FLAGS.num_patches)
    view2 = preprocess(view2, md['image_mean'], md['image_std'],
                       FLAGS.num_patches)
  else:
    view1, view2 = batch["view1"], batch["view2"]
  B = view1.shape[0]
  x1, (states1, logs) = run_block(block_idx, FLAGS.num_groups, view1,
                                  block_params, noise)
  x2, (states2, _) = run_block(block_idx, FLAGS.num_groups, view2,
                               block_params, noise)
  local_loss_ = local_simclr_loss_sg(x1,
                                     x2,
                                     block_params[-1][0],
                                     block_params[-1][1],
                                     FLAGS.temp,
                                     batch["labels_idx"],
                                     custom_forward=custom_forward)  # [B, P]
  if not FLAGS.spatial_loss:
    # [B, P, G] -> [B, 1, G]
    local_loss_ = jnp.mean(local_loss_, axis=1, keepdims=True)
  denom = B
  logs[f'local_loss/blk_{block_idx}'] = jnp.mean(local_loss_) * 2.0
  return local_loss_ / float(denom), (x1, x2), (states1, states2, logs)


def loss(params,
         batch,
         noise=None,
         stop_gradient=False,
         readout=False,
         key=None,
         stop_every=1,
         stop_remainder=-1,
         block_start=0,
         block_end=-1,
         custom_forward=False,
         local=True,
         avg_batch=True):
  """Supervised classification loss."""
  md = get_dataset_metadata(FLAGS.dataset)
  key, subkey = jax.random.split(key)
  batch = augmentations.postprocess(batch,
                                    subkey,
                                    add_gaussian_blur=FLAGS.gaussian_blur)
  view1, view2, targets, labels_idx = batch["view1"], batch["view2"], batch[
      "labels"], batch["labels_idx"]
  targets = jax.nn.one_hot(targets, md["num_classes"])
  NBLK = FLAGS.num_blocks
  NL = get_num_layers(NBLK)
  B = view1.shape[0]
  num_patches = FLAGS.num_patches
  P = num_patches**2
  G = FLAGS.num_groups

  _, (states1, logs) = predict(
      params,
      view1,
      noise=noise,
      stop_gradient=stop_gradient,
      readout=readout,
      #  key=key,
      stop_every=stop_every,
      stop_remainder=stop_remainder,
      is_training=True)
  _, (states2, _) = predict(
      params,
      view2,
      noise=noise,
      stop_gradient=stop_gradient,
      readout=readout,
      # key=key,
      stop_every=stop_every,
      stop_remainder=stop_remainder,
      is_training=True)

  # Last block avg pool contrastive loss.
  proj1 = states1[f'pred_final']
  proj2 = states2[f'pred_final']
  if FLAGS.simclr_sg:
    loss_ssl = simclr_loss_sg(proj1, proj2, FLAGS.temp, labels_idx) * 2.0
  else:
    loss_ssl = simclr_loss(proj1, proj2, FLAGS.temp, labels_idx)
  if avg_batch:
    loss_ssl = jnp.mean(loss_ssl)
    logs['local_loss/final'] = loss_ssl
  else:
    logs['local_loss/final'] = jnp.mean(loss_ssl)
  loss = loss_ssl

  local_losses = []
  avgpool_token = FLAGS.avgpool_token

  if block_end < 0:
    block_end = NBLK
  if not local:
    block_start = block_end = 0
  for blk in range(block_start, block_end):
    if blk == 0:
      proj1_ = states1[f'block_{blk}/post_1']  # B, P, D
      proj2_ = states2[f'block_{blk}/post_1']
    else:
      proj1_ = states1[f'block_{blk}/post_2']  # B, P, D
      proj2_ = states2[f'block_{blk}/post_2']
    proj1_ = jnp.reshape(proj1_, [B, P, G, -1])
    proj2_ = jnp.reshape(proj2_, [B, P, G, -1])
    local_loss_ = local_simclr_loss_sg(proj1_,
                                       proj2_,
                                       params[NL + blk][0],
                                       params[NL + blk][1],
                                       FLAGS.temp,
                                       labels_idx,
                                       custom_forward=custom_forward)  # [B, P]
    if avgpool_token:
      denom = B
    else:
      denom = B * P

    if not FLAGS.spatial_loss:
      # [B, P, G] -> [B, 1, G]
      local_loss_ = jnp.mean(local_loss_, axis=1, keepdims=True)

    local_losses.append(local_loss_ / float(denom))
    logs[f'local_loss/blk_{blk}'] = jnp.mean(local_loss_) * 2.0
  if not FLAGS.modular_loss:
    local_losses = local_losses[-1]
  return loss, local_losses, (states1, states2, logs)


def accuracy(params, batch):
  inputs, targets = batch['image'], batch['labels']
  pred, (states, logs) = predict(params, inputs)
  predicted_class = jnp.argmax(pred, axis=-1)
  return jnp.mean(predicted_class == targets)


def update_backprop(params, batch, key):
  wd = FLAGS.wd

  def loss_fn(params, batch):
    final_loss_, _, (states1, states2, logs) = loss(
        params,
        batch,
        noise=None,
        stop_gradient=False,
        key=key,
        local=False,
    )
    return final_loss_, (states1, states2, logs)

  grads_now, (_, _, logs) = grad(loss_fn, has_aux=True)(params, batch)

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb)
                 for (gw, gb), (w, b) in zip(grads_now[:-1], params[:-1])
                ] + grads_now[-1:]
  return grads_now, logs


def update_local_backprop(params, batch, key):
  wd = FLAGS.wd
  avgpool_token = FLAGS.avgpool_token
  NBLK = FLAGS.num_blocks
  NL = get_num_layers(NBLK)
  P = FLAGS.num_patches**2
  G = FLAGS.num_groups

  def local_loss_(params, batch):
    final_loss_, local_loss_, (states1, states2,
                               logs) = loss(params,
                                            batch,
                                            noise=None,
                                            stop_gradient=False,
                                            readout=True,
                                            key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states1, states2, logs)

  grads_now, (_, _, logs) = grad(local_loss_, has_aux=True)(params, batch)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr)

  if FLAGS.same_head:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G), grads_now[i][1] / float(G))

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P), grads_now[i][1] / float(P))

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb)
                 for (gw, gb), (w, b) in zip(grads_now[:-1], params[:-1])
                ] + grads_now[-1:]
  return grads_now, logs


def update_local_stopgrad_backprop(params, batch, key):
  wd = FLAGS.wd
  stop_every = FLAGS.stop_every
  stop_remainder = FLAGS.stop_remainder
  avgpool_token = FLAGS.avgpool_token
  NBLK = FLAGS.num_blocks
  NL = get_num_layers(NBLK)
  P = FLAGS.num_patches**2
  G = FLAGS.num_groups

  def local_loss_(params, batch):
    final_loss_, local_loss_, (states1, states2,
                               logs) = loss(params,
                                            batch,
                                            noise=None,
                                            stop_gradient=True,
                                            key=key,
                                            stop_every=stop_every,
                                            stop_remainder=stop_remainder)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states1, states2, logs)

  grads_now, (_, _, logs) = grad(local_loss_, has_aux=True)(params, batch)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr)

  if FLAGS.same_head:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G), grads_now[i][1] / float(G))

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P), grads_now[i][1] / float(P))

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb)
                 for (gw, gb), (w, b) in zip(grads_now[:-1], params[:-1])
                ] + grads_now[-1:]
  return grads_now, logs


def update_local_stopgrad_overlap_backprop(params, batch, key):
  wd = FLAGS.wd
  # stop_every = FLAGS.stop_every
  # stop_remainder = FLAGS.stop_remainder
  avgpool_token = FLAGS.avgpool_token
  NBLK = FLAGS.num_blocks
  NL = get_num_layers(NBLK)
  P = FLAGS.num_patches**2
  G = FLAGS.num_groups

  def local_loss_(params, batch):
    final_loss1_, local_loss1_, (states1, states2,
                                 logs) = loss(params,
                                              batch,
                                              noise=None,
                                              stop_gradient=True,
                                              key=key,
                                              stop_every=2,
                                              stop_remainder=0)
    final_loss2_, local_loss2_, (_, _, logs) = loss(params,
                                                    batch,
                                                    noise=None,
                                                    stop_gradient=True,
                                                    key=key,
                                                    stop_every=2,
                                                    stop_remainder=1)
    final_loss_ = final_loss1_ + final_loss2_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss1_])) + jnp.sum(
            jnp.stack([jnp.sum(l) for l in local_loss2_]))
    return final_loss_, (states1, states2, logs)

  grads_now, (_, _, logs) = grad(local_loss_, has_aux=True)(params, batch)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr)

  if FLAGS.same_head:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G), grads_now[i][1] / float(G))

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P), grads_now[i][1] / float(P))

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb)
                 for (gw, gb), (w, b) in zip(grads_now[:-1], params[:-1])
                ] + grads_now[-1:]
  return grads_now, logs


def update_backprop_readout(params, batch, key):
  wd = FLAGS.wd

  def loss_fn(readout_params, batch):
    full_params = params[:-1] + [readout_params]
    final_loss_, (states, logs) = classif_loss(full_params,
                                               batch,
                                               noise=None,
                                               stop_gradient=False,
                                               key=key)
    return final_loss_, (states, logs)

  grads, (_, logs) = grad(loss_fn, has_aux=True)(params[-1], batch)
  if wd > 0.0:
    grads = tuple([grads[0] + wd * params[-1][0]] + list(grads[1:]))
  return grads, logs


def get_local_stopgrad_greedy_backprop_fn(block_idx):

  def update_local_stopgrad_greedy_backprop(block_params, batch, key):
    wd = FLAGS.wd

    def local_loss_(block_params, batch):
      loss_, (x1, x2), (states1, states2,
                        logs) = local_loss(block_idx,
                                           block_params,
                                           batch,
                                           key=key,
                                           noise=None,
                                           custom_forward=False)
      loss_ = jnp.sum(loss_)
      return loss_, (x1, x2, states1, states2, logs)

    grads_now, (x1, x2, _, _, logs) = grad(local_loss_,
                                           has_aux=True)(block_params, batch)

    if FLAGS.optimizer == 'sgd' and wd > 0.0:
      grads_now = [(gw + wd * w, gb)
                   for (gw, gb), (w, b) in zip(grads_now, block_params)]
    return grads_now, (x1, x2), logs

  return update_local_stopgrad_greedy_backprop


def update_forward_grad_weights(params, batch, key):
  num_patches = FLAGS.num_patches
  num_groups = FLAGS.num_groups
  num_passes = FLAGS.num_passes
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  wd = FLAGS.wd
  NBLK = FLAGS.num_blocks
  assert FLAGS.simclr_sg
  local_simclr_loss = lambda x, y, w, b: jnp.mean(
      local_simclr_loss_sg(x, y, w, b, FLAGS.temp, batch["labels_idx"]))
  local_simclr_grad = jax.grad(local_simclr_loss, argnums=[2, 3])

  def local_loss(params):
    _, local_loss_, (states1, states2,
                     logs) = loss(params,
                                  batch,
                                  stop_gradient=FLAGS.modular_loss,
                                  readout=True,
                                  key=key,
                                  custom_forward=True)
    return local_loss_, (states1, states2, logs)

  for npass in range(num_passes):
    noise = []
    label = jax.nn.one_hot(batch['labels'], md["num_classes"])
    print(batch['labels'])
    print(label)
    G = num_groups
    M = num_passes
    NL = get_num_layers(NBLK)
    main_params = params[:NL]
    loss_params = params[NL:]

    for i, (weight, bias) in enumerate(main_params):
      blk, layer = get_blk(i)
      key, subkey = jax.random.split(key)
      dw = jax.random.normal(subkey, weight.shape)
      key, subkey = jax.random.split(key)
      db = jax.random.normal(subkey, bias.shape)
      noise.append((dw, db))

    for i, (weight, bias) in enumerate(loss_params):
      noise.append((jnp.zeros_like(weight), jnp.zeros_like(bias)))

    # [L,B,P]
    _, g, (states1, states2, logs) = jax.jvp(local_loss, [params], [noise],
                                             has_aux=True)

    # Main network layers.
    for i, ((weight, bias), (dw, db)) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      if FLAGS.modular_loss:
        g_ = g[blk]
      else:
        g_ = g

      # [B, P, G] -> [G]
      g_ = jnp.sum(g_, axis=[0, 1])

      if blk > 0 and layer == 0:
        g_ = jnp.sum(g_)  # []
        grad_w = g_ * dw
        grad_b = g_ * db
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        dw = jnp.reshape(dw, [weight.shape[0], G, weight.shape[1] // G])
        db = jnp.reshape(db, [G, -1])
        grad_w = g_[None, :, None] * dw
        grad_b = g_[:, None] * db
      elif (blk == 0 and layer == 1) or layer == 2:
        dw = jnp.reshape(dw, [G, weight.shape[0], -1])
        db = jnp.reshape(db, [G, -1])
        grad_w = g_[:, None, None] * dw
        grad_b = g_[:, None] * db
      grad_w = jnp.reshape(grad_w, weight.shape)
      grad_b = jnp.reshape(grad_b, bias.shape)
      idx = i
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      elif i == len(loss_params) - 2:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      else:
        # Intermediate classification layer.
        if blk == 0:
          pre_act1 = states1[f'block_{blk}/post_1']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_1']
        else:
          pre_act1 = states1[f'block_{blk}/post_2']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_2']
        grad_w, grad_b = local_simclr_grad(pre_act1, pre_act2, weight, bias)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def update_global_forward_grad_weights(params, batch, key):
  num_patches = FLAGS.num_patches
  num_groups = FLAGS.num_groups
  num_passes = FLAGS.num_passes
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  wd = FLAGS.wd
  NBLK = FLAGS.num_blocks
  assert FLAGS.simclr_sg
  local_simclr_loss = lambda x, y, w, b: jnp.mean(
      local_simclr_loss_sg(x, y, w, b, FLAGS.temp, batch["labels_idx"]))
  local_simclr_grad = jax.grad(local_simclr_loss, argnums=[2, 3])

  def local_loss(params):
    loss_, local_loss_, (states1, states2, logs) = loss(params,
                                                        batch,
                                                        stop_gradient=False,
                                                        readout=False,
                                                        key=key,
                                                        custom_forward=True)
    return loss_, (states1, states2, logs)

  for npass in range(num_passes):
    noise = []
    label = jax.nn.one_hot(batch['labels'], md["num_classes"])
    print(batch['labels'])
    print(label)
    G = num_groups
    M = num_passes
    NL = get_num_layers(NBLK)
    main_params = params[:NL]
    loss_params = params[NL:]

    for i, (weight, bias) in enumerate(main_params):
      blk, layer = get_blk(i)
      key, subkey = jax.random.split(key)
      dw = jax.random.normal(subkey, weight.shape)
      key, subkey = jax.random.split(key)
      db = jax.random.normal(subkey, bias.shape)
      noise.append((dw, db))

    for i, (weight, bias) in enumerate(loss_params):
      noise.append((jnp.zeros_like(weight), jnp.zeros_like(bias)))

    # [L,B,P]
    _, g, (states1, states2, logs) = jax.jvp(local_loss, [params], [noise],
                                             has_aux=True)

    # Main network layers.
    for i, ((weight, bias), (dw, db)) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      g_ = g

      if blk > 0 and layer == 0:
        grad_w = g_ * dw
        grad_b = g_ * db
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        dw = jnp.reshape(dw, [weight.shape[0], G, weight.shape[1] // G])
        db = jnp.reshape(db, [G, -1])
        grad_w = g_ * dw
        grad_b = g_ * db
      elif (blk == 0 and layer == 1) or layer == 2:
        dw = jnp.reshape(dw, [G, weight.shape[0], -1])
        db = jnp.reshape(db, [G, -1])
        grad_w = g_ * dw
        grad_b = g_ * db
      print(blk, layer, weight.shape, grad_w.shape)
      # assert False
      grad_w = jnp.reshape(grad_w, weight.shape)
      grad_b = jnp.reshape(grad_b, bias.shape)
      idx = i
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      elif i == len(loss_params) - 2:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      else:
        # Intermediate classification layer.
        if blk == 0:
          pre_act1 = states1[f'block_{blk}/post_1']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_1']
        else:
          pre_act1 = states1[f'block_{blk}/post_2']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_2']
        grad_w, grad_b = local_simclr_grad(pre_act1, pre_act2, weight, bias)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def update_forward_grad_activations(params, batch, key):
  num_patches = FLAGS.num_patches
  num_units = FLAGS.num_channel_mlp_units
  num_groups = FLAGS.num_groups
  num_passes = FLAGS.num_passes
  conv = FLAGS.conv_mixer
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  wd = FLAGS.wd
  NBLK = FLAGS.num_blocks
  ln_loss = lambda x, dy: jnp.sum(normalize(x) * dy)
  ln_grad = jax.grad(ln_loss)
  ln_loss1 = lambda x, dy: jnp.sum(normalize(x, swap=not conv) * dy)
  ln_grad1 = jax.grad(ln_loss1)
  assert FLAGS.simclr_sg
  local_simclr_loss = lambda x, y, w, b: jnp.mean(
      local_simclr_loss_sg(x, y, w, b, FLAGS.temp, batch["labels_idx"]))
  local_simclr_grad = jax.grad(local_simclr_loss, argnums=[2, 3])
  conv_loss = lambda w, x, dy: jnp.sum(depthwise_conv(x, w) * dy)
  conv_grad = jax.grad(conv_loss)

  def local_loss_(noise):
    _, local_loss_, (states1, states2,
                     logs) = loss(params,
                                  batch,
                                  noise=noise,
                                  stop_gradient=FLAGS.modular_loss,
                                  readout=True,
                                  key=key,
                                  custom_forward=True)
    return local_loss_, (states1, states2, logs)

  for npass in range(num_passes):
    zeros = []
    noise = []
    num_items = num_patches**2
    P = num_items
    B = batch['labels'].shape[0]
    label = jax.nn.one_hot(batch['labels'], md["num_classes"])
    G = num_groups
    M = num_passes
    NL = get_num_layers(NBLK)
    main_params = params[:NL]
    loss_params = params[NL:]

    for i, (weight, bias) in enumerate(main_params):
      blk, layer = get_blk(i)
      if blk == 0:
        if layer == 0:
          out_shape = [B, num_items, num_units]
        else:
          out_shape = [B, num_items, num_groups, num_units // num_groups]
      else:
        if layer == 0:
          if conv:
            out_shape = [B, num_items, num_units]
          else:
            out_shape = [B, num_units, num_items]
        elif layer == 1:
          out_shape = [B, num_items, num_units]
        else:
          out_shape = [B, num_items, num_groups, num_units // num_groups]
      key, subkey = jax.random.split(key)
      dz = jax.random.normal(subkey, out_shape)
      noise.append(dz)
      zeros.append(jnp.zeros(out_shape))

    # [L,B,P]
    _, g, (states1, states2, logs) = jax.jvp(local_loss_, [zeros], [noise],
                                             has_aux=True)

    # Main network layers.
    for i, ((weight, bias), dz) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      pre_act1 = states1[f'block_{blk}/pre_{layer}']
      prenorm_act1 = states1[f'block_{blk}/prenorm_{layer}']
      post_act1 = states1[f'block_{blk}/post_{layer}']
      mask1 = (post_act1 > 0.0).astype(jnp.float32)
      pre_act2 = states2[f'block_{blk}/pre_{layer}']
      prenorm_act2 = states2[f'block_{blk}/prenorm_{layer}']
      post_act2 = states2[f'block_{blk}/post_{layer}']
      mask2 = (post_act2 > 0.0).astype(jnp.float32)
      dz1 = dz
      dz2 = dz
      if FLAGS.modular_loss:
        g_ = g[blk]
      else:
        g_ = g

      # [B, D] -> [B, D, P] or [B, P] -> [B, P, D]
      if blk > 0 and layer == 0:
        # Token mixing layer, sum groups together
        if conv:
          g_ = jnp.sum(g_, axis=-1, keepdims=True)  # [2B, P, 1]
        else:
          g_ = jnp.sum(g_, axis=-1)[:, None, :]  # [2B, 1, P]
      else:
        # Channel mixing layer
        # [B, P, D] -> [B, P, G, D]
        dz1 = jnp.reshape(dz1, [B, P, G, -1])
        dz2 = jnp.reshape(dz2, [B, P, G, -1])
        mask1 = jnp.reshape(mask1, [B, P, G, -1])
        mask2 = jnp.reshape(mask2, [B, P, G, -1])
        g_ = g_[:, :, :, None]  # [2B, P, G, 1]

      g1_ = g_[:B]
      g2_ = g_[B:]
      grad_z1 = g1_ * dz1 * mask1
      grad_z2 = g2_ * dz2 * mask2

      if FLAGS.norm_grad and FLAGS.post_linear_ln:
        # Backprop through normalization.
        if blk > 0 and layer == 0:
          grad_z1 = jnp.reshape(
              ln_grad1(prenorm_act1, jnp.reshape(grad_z1, prenorm_act1.shape)),
              dz1.shape)
          grad_z2 = jnp.reshape(
              ln_grad1(prenorm_act2, jnp.reshape(grad_z2, prenorm_act2.shape)),
              dz2.shape)
        else:
          grad_z1 = jnp.reshape(
              ln_grad(prenorm_act1, jnp.reshape(grad_z1, prenorm_act1.shape)),
              dz1.shape)
          grad_z2 = jnp.reshape(
              ln_grad(prenorm_act2, jnp.reshape(grad_z2, prenorm_act2.shape)),
              dz2.shape)
      if blk > 0 and layer == 0 and conv:
        H = int(math.sqrt(P))
        grad_z1 = jnp.reshape(grad_z1, [B, H, H, -1])
        grad_z2 = jnp.reshape(grad_z2, [B, H, H, -1])
        grad_w1 = conv_grad(weight, pre_act1, grad_z1)
        grad_w2 = conv_grad(weight, pre_act2, grad_z2)
        grad_b1 = jnp.einsum('nhwd->d', grad_z1)
        grad_b2 = jnp.einsum('nhwd->d', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
      elif blk > 0 and layer == 0 and not conv:
        grad_w1 = jnp.einsum('npc,npd->cd', pre_act1, grad_z1)
        grad_b1 = jnp.einsum('npc->c', grad_z1)
        grad_w2 = jnp.einsum('npc,npd->cd', pre_act2, grad_z2)
        grad_b2 = jnp.einsum('npc->c', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        grad_z1 = jnp.reshape(grad_z1, [B, P, -1])
        grad_z2 = jnp.reshape(grad_z2, [B, P, -1])
        grad_w1 = jnp.einsum('npc,npd->cd', pre_act1, grad_z1)
        grad_b1 = jnp.einsum('npc->c', grad_z1)
        grad_w2 = jnp.einsum('npc,npd->cd', pre_act2, grad_z2)
        grad_b2 = jnp.einsum('npc->c', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
      else:
        grad_w1 = jnp.einsum('npgc,npgd->gcd', pre_act1, grad_z1)
        grad_b1 = jnp.einsum('npgd->gd', grad_z1)
        grad_w2 = jnp.einsum('npgc,npgd->gcd', pre_act2, grad_z2)
        grad_b2 = jnp.einsum('npgd->gd', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
        grad_w = jnp.reshape(grad_w, weight.shape)
        grad_b = jnp.reshape(grad_b, bias.shape)
      idx = i
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      elif i == len(loss_params) - 2:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      else:
        # Intermediate classification layer.
        if blk == 0:
          pre_act1 = states1[f'block_{blk}/post_1']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_1']
        else:
          pre_act1 = states1[f'block_{blk}/post_2']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_2']
        grad_w, grad_b = local_simclr_grad(pre_act1, pre_act2, weight, bias)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def update_global_forward_grad_activations(params, batch, key):
  num_patches = FLAGS.num_patches
  num_units = FLAGS.num_channel_mlp_units
  num_groups = FLAGS.num_groups
  num_passes = FLAGS.num_passes
  conv = FLAGS.conv_mixer
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  wd = FLAGS.wd
  NBLK = FLAGS.num_blocks
  ln_loss = lambda x, dy: jnp.sum(normalize(x) * dy)
  ln_grad = jax.grad(ln_loss)
  ln_loss1 = lambda x, dy: jnp.sum(normalize(x, swap=not conv) * dy)
  ln_grad1 = jax.grad(ln_loss1)
  conv_loss = lambda w, x, dy: jnp.sum(depthwise_conv(x, w) * dy)
  conv_grad = jax.grad(conv_loss)

  assert FLAGS.simclr_sg
  local_simclr_loss = lambda x, y, w, b: jnp.mean(
      local_simclr_loss_sg(x, y, w, b, FLAGS.temp, batch["labels_idx"]))
  local_simclr_grad = jax.grad(local_simclr_loss, argnums=[2, 3])

  def loss_fn(noise):
    # [B, P, G]
    loss_, local_loss_, (states1, states2, logs) = loss(params,
                                                        batch,
                                                        noise=noise,
                                                        stop_gradient=False,
                                                        readout=False,
                                                        key=key,
                                                        custom_forward=True,
                                                        avg_batch=False)
    return loss_, (states1, states2, logs)

  for npass in range(num_passes):
    zeros = []
    noise = []
    num_items = num_patches**2
    P = num_items
    B = batch['labels'].shape[0]
    label = jax.nn.one_hot(batch['labels'], md["num_classes"])
    print(batch['labels'])
    print(label)
    G = num_groups
    M = num_passes
    NL = get_num_layers(NBLK)
    main_params = params[:NL]
    loss_params = params[NL:]

    for i, (weight, bias) in enumerate(main_params):
      blk, layer = get_blk(i)
      if blk == 0:
        if layer == 0:
          out_shape = [B, num_items, num_units]
        else:
          out_shape = [B, num_items, num_groups, num_units // num_groups]
      else:
        if layer == 0:
          if conv:
            out_shape = [B, num_items, num_units]
          else:
            out_shape = [B, num_units, num_items]
        elif layer == 1:
          out_shape = [B, num_items, num_units]
        else:
          out_shape = [B, num_items, num_groups, num_units // num_groups]
      print(i, out_shape)
      key, subkey = jax.random.split(key)
      dz = jax.random.normal(subkey, out_shape)
      noise.append(dz)
      zeros.append(jnp.zeros(out_shape))

    # [L,B,P]
    _, g, (states1, states2, logs) = jax.jvp(loss_fn, [zeros], [noise],
                                             has_aux=True)

    # Main network layers.
    for i, ((weight, bias), dz) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      pre_act1 = states1[f'block_{blk}/pre_{layer}']
      prenorm_act1 = states1[f'block_{blk}/prenorm_{layer}']
      post_act1 = states1[f'block_{blk}/post_{layer}']
      mask1 = (post_act1 > 0.0).astype(jnp.float32)
      pre_act2 = states2[f'block_{blk}/pre_{layer}']
      prenorm_act2 = states2[f'block_{blk}/prenorm_{layer}']
      post_act2 = states2[f'block_{blk}/post_{layer}']
      mask2 = (post_act2 > 0.0).astype(jnp.float32)
      dz1 = dz
      dz2 = dz
      g_ = g

      # [B, D] -> [B, D, P] or [B, P] -> [B, P, D]
      if blk > 0 and layer == 0:
        # Token mixing layer, sum groups together
        g_ = g_[:, None, None]  # [2B, 1, 1, 1]
      else:
        # Channel mixing layer
        # [B, P, D] -> [B, P, G, D]
        dz1 = jnp.reshape(dz1, [B, P, G, -1])
        dz2 = jnp.reshape(dz2, [B, P, G, -1])
        mask1 = jnp.reshape(mask1, [B, P, G, -1])
        mask2 = jnp.reshape(mask2, [B, P, G, -1])
        g_ = g_[:, None, None, None]  # [2B, 1, 1, 1]

      g1_ = g_[:B]
      g2_ = g_[B:]
      grad_z1 = g1_ * dz1 * mask1
      grad_z2 = g2_ * dz2 * mask2

      if FLAGS.norm_grad and FLAGS.post_linear_ln:
        # Backprop through normalization.
        if blk > 0 and layer == 0:
          grad_z1 = jnp.reshape(
              ln_grad1(prenorm_act1, jnp.reshape(grad_z1, prenorm_act1.shape)),
              dz1.shape)
          grad_z2 = jnp.reshape(
              ln_grad1(prenorm_act2, jnp.reshape(grad_z2, prenorm_act2.shape)),
              dz2.shape)
        else:
          grad_z1 = jnp.reshape(
              ln_grad(prenorm_act1, jnp.reshape(grad_z1, prenorm_act1.shape)),
              dz1.shape)
          grad_z2 = jnp.reshape(
              ln_grad(prenorm_act2, jnp.reshape(grad_z2, prenorm_act2.shape)),
              dz2.shape)
      if blk > 0 and layer == 0 and conv:
        H = int(math.sqrt(P))
        grad_z1 = jnp.reshape(grad_z1, [B, H, H, -1])
        grad_z2 = jnp.reshape(grad_z2, [B, H, H, -1])
        grad_w1 = conv_grad(weight, pre_act1, grad_z1)
        grad_w2 = conv_grad(weight, pre_act2, grad_z2)
        grad_b1 = jnp.einsum('nhwd->d', grad_z1)
        grad_b2 = jnp.einsum('nhwd->d', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
      elif blk > 0 and layer == 0 and not conv:
        grad_w1 = jnp.einsum('npc,npd->cd', pre_act1, grad_z1)
        grad_b1 = jnp.einsum('npc->c', grad_z1)
        grad_w2 = jnp.einsum('npc,npd->cd', pre_act2, grad_z2)
        grad_b2 = jnp.einsum('npc->c', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        grad_z1 = jnp.reshape(grad_z1, [B, P, -1])
        grad_z2 = jnp.reshape(grad_z2, [B, P, -1])
        grad_w1 = jnp.einsum('npc,npd->cd', pre_act1, grad_z1)
        grad_b1 = jnp.einsum('npc->c', grad_z1)
        grad_w2 = jnp.einsum('npc,npd->cd', pre_act2, grad_z2)
        grad_b2 = jnp.einsum('npc->c', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
      else:
        grad_w1 = jnp.einsum('npgc,npgd->gcd', pre_act1, grad_z1)
        grad_b1 = jnp.einsum('npgd->gd', grad_z1)
        grad_w2 = jnp.einsum('npgc,npgd->gcd', pre_act2, grad_z2)
        grad_b2 = jnp.einsum('npgd->gd', grad_z2)
        grad_w = (grad_w1 + grad_w2) / 2.0
        grad_b = (grad_b1 + grad_b2) / 2.0
      idx = i
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      elif i == len(loss_params) - 2:
        grad_w, grad_b = jnp.zeros_like(weight), jnp.zeros_like(bias)
      else:
        # Intermediate classification layer.
        if blk == 0:
          pre_act1 = states1[f'block_{blk}/post_1']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_1']
        else:
          pre_act1 = states1[f'block_{blk}/post_2']  # B, P, D
          pre_act2 = states2[f'block_{blk}/post_2']
        grad_w, grad_b = local_simclr_grad(pre_act1, pre_act2, weight, bias)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def get_forward_grad_activations_fn(block_idx):

  def _update_forward_grad_activations(block_params, batch, key):
    num_patches = FLAGS.num_patches
    num_units = FLAGS.num_channel_mlp_units
    num_groups = FLAGS.num_groups
    num_passes = FLAGS.num_passes
    conv = FLAGS.conv_mixer
    md = get_dataset_metadata(FLAGS.dataset)
    grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
                 for (weight, bias) in block_params]
    wd = FLAGS.wd
    ln_loss = lambda x, dy: jnp.sum(normalize(x) * dy)
    ln_grad = jax.grad(ln_loss)
    ln_loss1 = lambda x, dy: jnp.sum(normalize(x, swap=not conv) * dy)
    ln_grad1 = jax.grad(ln_loss1)
    conv_loss = lambda w, x, dy: jnp.sum(depthwise_conv(x, w) * dy)
    conv_grad = jax.grad(conv_loss)

    assert FLAGS.simclr_sg
    local_simclr_loss = lambda x, y, w, b: jnp.mean(
        local_simclr_loss_sg(x, y, w, b, FLAGS.temp, batch["labels_idx"]))
    local_simclr_grad = jax.grad(local_simclr_loss, argnums=[2, 3])

    def local_loss_(noise):
      loss_, (x1, x2), (states1, states2,
                        logs) = local_loss(block_idx,
                                           block_params,
                                           batch,
                                           key=key,
                                           noise=noise,
                                           custom_forward=True)
      return loss_, (x1, x2, states1, states2, logs)

    for npass in range(num_passes):
      zeros = []
      noise = []
      num_items = num_patches**2
      P = num_items
      B = batch['labels'].shape[0]
      label = jax.nn.one_hot(batch['labels'], md["num_classes"])
      G = num_groups
      M = num_passes
      for i, (weight, bias) in enumerate(block_params[:-1]):
        if block_idx == 0:
          if i == 0:
            out_shape = [B, num_items, num_units]
          else:
            out_shape = [B, num_items, num_groups, num_units // num_groups]
        else:
          if i == 0:
            if conv:
              out_shape = [B, num_items, num_units]
            else:
              out_shape = [B, num_units, num_items]
          elif i == 1:
            out_shape = [B, num_items, num_units]
          else:
            out_shape = [B, num_items, num_groups, num_units // num_groups]
        print(i, out_shape)
        key, subkey = jax.random.split(key)
        dz = jax.random.normal(subkey, out_shape)
        noise.append(dz)
        zeros.append(jnp.zeros(out_shape))

      # [L,B,P]
      _, g, (x1, x2, states1, states2, logs) = jax.jvp(local_loss_, [zeros],
                                                       [noise],
                                                       has_aux=True)

      # Main network layers.
      for i, ((weight, bias), dz) in enumerate(zip(block_params[:-1], noise)):
        blk, layer = block_idx, i
        # Forward gradient.
        pre_act1 = states1[f'block_{blk}/pre_{layer}']
        prenorm_act1 = states1[f'block_{blk}/prenorm_{layer}']
        post_act1 = states1[f'block_{blk}/post_{layer}']
        mask1 = (post_act1 > 0.0).astype(jnp.float32)
        pre_act2 = states2[f'block_{blk}/pre_{layer}']
        prenorm_act2 = states2[f'block_{blk}/prenorm_{layer}']
        post_act2 = states2[f'block_{blk}/post_{layer}']
        mask2 = (post_act2 > 0.0).astype(jnp.float32)
        dz1 = dz
        dz2 = dz

        # [B, D] -> [B, D, P] or [B, P] -> [B, P, D]
        if blk > 0 and layer == 0:
          # Token mixing layer
          if conv:
            g_ = jnp.sum(g, axis=-1, keepdims=True)  # [2B, P, 1]
          else:
            g_ = jnp.sum(g, axis=-1)[:, None, :]  # [2B, 1, P]
        else:
          # Channel mixing layer
          dz1 = jnp.reshape(dz1, [B, P, G, -1])
          dz2 = jnp.reshape(dz2, [B, P, G, -1])
          mask1 = jnp.reshape(mask1, [B, P, G, -1])
          mask2 = jnp.reshape(mask2, [B, P, G, -1])
          g_ = g[:, :, :, None]  # [2B, P, G]

        # if FLAGS.simclr_sg:
        g1_ = g_[:B]
        g2_ = g_[B:]
        # else:
        #   g1_ = g_
        #   g2_ = g_
        # print(g1_.shape, dz1.shape, mask1.shape)
        # print(g2_.shape, dz2.shape, mask2.shape)
        grad_z1 = g1_ * dz1 * mask1
        grad_z2 = g2_ * dz2 * mask2
        if FLAGS.norm_grad and FLAGS.post_linear_ln:
          # Backprop through normalization.
          if blk > 0 and layer == 0:
            grad_z1 = jnp.reshape(
                ln_grad1(prenorm_act1,
                         jnp.reshape(grad_z1, prenorm_act1.shape)), dz1.shape)
            grad_z2 = jnp.reshape(
                ln_grad1(prenorm_act2,
                         jnp.reshape(grad_z2, prenorm_act2.shape)), dz2.shape)
          else:
            grad_z1 = jnp.reshape(
                ln_grad(prenorm_act1,
                        jnp.reshape(grad_z1, prenorm_act1.shape)), dz1.shape)
            grad_z2 = jnp.reshape(
                ln_grad(prenorm_act2,
                        jnp.reshape(grad_z2, prenorm_act2.shape)), dz2.shape)
        if blk > 0 and layer == 0 and conv:
          H = int(math.sqrt(P))
          grad_z1 = jnp.reshape(grad_z1, [B, H, H, -1])
          grad_z2 = jnp.reshape(grad_z2, [B, H, H, -1])
          grad_w1 = conv_grad(weight, pre_act1, grad_z1)
          grad_w2 = conv_grad(weight, pre_act2, grad_z2)
          grad_b1 = jnp.einsum('nhwd->d', grad_z1)
          grad_b2 = jnp.einsum('nhwd->d', grad_z2)
          grad_w = (grad_w1 + grad_w2) / 2.0
          grad_b = (grad_b1 + grad_b2) / 2.0
        elif blk > 0 and layer == 0 and not conv:
          grad_w1 = jnp.einsum('npc,npd->cd', pre_act1, grad_z1)
          grad_b1 = jnp.einsum('npc->c', grad_z1)
          grad_w2 = jnp.einsum('npc,npd->cd', pre_act2, grad_z2)
          grad_b2 = jnp.einsum('npc->c', grad_z2)
          grad_w = (grad_w1 + grad_w2) / 2.0
          grad_b = (grad_b1 + grad_b2) / 2.0
        elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
          grad_z1 = jnp.reshape(grad_z1, [B, P, -1])
          grad_z2 = jnp.reshape(grad_z2, [B, P, -1])
          grad_w1 = jnp.einsum('npc,npd->cd', pre_act1, grad_z1)
          grad_b1 = jnp.einsum('npc->c', grad_z1)
          grad_w2 = jnp.einsum('npc,npd->cd', pre_act2, grad_z2)
          grad_b2 = jnp.einsum('npc->c', grad_z2)
          grad_w = (grad_w1 + grad_w2) / 2.0
          grad_b = (grad_b1 + grad_b2) / 2.0
        else:
          grad_w1 = jnp.einsum('npgc,npgd->gcd', pre_act1, grad_z1)
          grad_b1 = jnp.einsum('npgd->gd', grad_z1)
          grad_w2 = jnp.einsum('npgc,npgd->gcd', pre_act2, grad_z2)
          grad_b2 = jnp.einsum('npgd->gd', grad_z2)
          grad_w = (grad_w1 + grad_w2) / 2.0
          grad_b = (grad_b1 + grad_b2) / 2.0
          grad_w = jnp.reshape(grad_w, weight.shape)
          grad_b = jnp.reshape(grad_b, bias.shape)
        idx = i
        grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                          grads_now[idx][1] + grad_b / float(M))

      # Intermediate classification layer.
      weight, bias = block_params[-1]
      blk = block_idx  # Every block has a loss.
      if blk == 0:
        pre_act1 = states1[f'block_{blk}/post_1']
        pre_act2 = states2[f'block_{blk}/post_1']
      else:
        pre_act1 = states1[f'block_{blk}/post_2']
        pre_act2 = states2[f'block_{blk}/post_2']
      grad_w, grad_b = local_simclr_grad(pre_act1, pre_act2, weight, bias)
      grad_w = grad_w * FLAGS.head_lr
      grad_b = grad_b * FLAGS.head_lr
      grads_now[-1] = (grads_now[-1][0] + grad_w / float(M),
                       grads_now[-1][1] + grad_b / float(M))

    if FLAGS.optimizer == 'sgd' and wd > 0.0:
      grads_now = [(gw + wd * w, gb)
                   for (gw, gb), (w, b) in zip(grads_now, block_params)]
    return grads_now, (x1, x2), logs

  return _update_forward_grad_activations


def update_direct_feedback_alignment(params, batch, key):
  wd = FLAGS.wd

  def loss_fn(params, batch):
    final_loss_, local_loss_, (states1, states2, logs) = loss_dfa(params,
                                                                  batch,
                                                                  noise=None,
                                                                  key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states1, states2, logs)

  grads_now, (states1, states2, logs) = grad(loss_fn, has_aux=True)(params,
                                                                    batch)

  # Clear gradient for backward weights.
  grads_now = [
      (gw, gb, 0.0 * gwb, 0.0 * gbb) for (gw, gb, gwb, gbb) in grads_now
  ]
  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb, 0.0 * gwb, 0.0 * gbb)
                 for (gw, gb, gwb, gbb) in grads_now[:-1]] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb, gwb, gbb)
                 for (gw, gb, gwb, gbb), (w, b, wb,
                                          bb) in zip(grads_now, params)]
  return grads_now, logs


def update_feedback_alignment(params, batch, key):
  wd = FLAGS.wd

  def loss_fn(params, batch):
    final_loss_, local_loss_, (states1, states2,
                               logs) = loss(params,
                                            batch,
                                            noise=None,
                                            stop_gradient=False,
                                            readout=True,
                                            key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states1, states2, logs)

  grads_now, (states1, states2, logs) = grad(loss_fn, has_aux=True)(params,
                                                                    batch)

  # Swap backward gradient for representation layers.
  grads_now = [(gwb, gb, 0.0 * gw) for (gw, gb, gwb) in grads_now]

  if FLAGS.freeze_backbone:
    grads_now = [
        (0.0 * gw, 0.0 * gb, 0.0 * gwb) for (gw, gb, gwb) in grads_now[:-1]
    ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb, gwb)
                 for (gw, gb, gwb), (w, b, wb) in zip(grads_now, params)]
  return grads_now, logs


def update_local_feedback_alignment(params, batch, key):
  wd = FLAGS.wd
  NBLK = FLAGS.num_blocks
  avgpool_token = FLAGS.avgpool_token
  G = FLAGS.num_groups
  NL = get_num_layers(NBLK)
  P = FLAGS.num_patches**2

  def local_loss(params, batch):
    final_loss_, local_loss_, (states1, states2,
                               logs) = loss(params,
                                            batch,
                                            noise=None,
                                            stop_gradient=False,
                                            readout=True,
                                            key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states1, states2, logs)

  NL = get_num_layers(NBLK)
  grads_now, (states1, states2, logs) = grad(local_loss, has_aux=True)(params,
                                                                       batch)

  # Swap backward gradient for representation layers.
  grads_now = [(gwb, gb, 0.0 * gw) for (gw, gb, gwb) in grads_now]

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr,
                      grads_now[i][2] * FLAGS.head_lr)

  if FLAGS.same_head:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G), grads_now[i][1] / float(G),
                      grads_now[i][2] / float(G))

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P), grads_now[i][1] / float(P),
                      grads_now[i][2] / float(P))

  if FLAGS.freeze_backbone:
    grads_now = [
        (0.0 * gw, 0.0 * gb, 0.0 * gwb) for (gw, gb, gwb) in grads_now[:NL]
    ] + grads_now[NL:]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb, gwb)
                 for (gw, gb, gwb), (w, b, wb) in zip(grads_now, params)]
  return grads_now, logs


def update_local_stopgrad_feedback_alignment(params, batch, key):
  wd = FLAGS.wd
  NBLK = FLAGS.num_blocks
  avgpool_token = FLAGS.avgpool_token
  G = FLAGS.num_groups
  NL = get_num_layers(NBLK)
  P = FLAGS.num_patches**2

  def local_loss(params, batch):
    final_loss_, local_loss_, (states1, states2,
                               logs) = loss(params,
                                            batch,
                                            noise=None,
                                            stop_gradient=True,
                                            readout=True,
                                            key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states1, states2, logs)

  NL = get_num_layers(NBLK)
  grads_now, (states1, states2, logs) = grad(local_loss, has_aux=True)(params,
                                                                       batch)

  # Swap backward gradient for representation layers.
  grads_now = [(gwb, gb, 0.0 * gw) for (gw, gb, gwb) in grads_now]

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr,
                      grads_now[i][2] * FLAGS.head_lr)

  if FLAGS.same_head:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G), grads_now[i][1] / float(G),
                      grads_now[i][2] / float(G))

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P), grads_now[i][1] / float(P),
                      grads_now[i][2] / float(P))

  if FLAGS.freeze_backbone:
    grads_now = [
        (0.0 * gw, 0.0 * gb, 0.0 * gwb) for (gw, gb, gwb) in grads_now[:NL]
    ] + grads_now[NL:]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb, gwb)
                 for (gw, gb, gwb), (w, b, wb) in zip(grads_now, params)]
  return grads_now, logs


def _to_tfds_split(split):
  """Returns the TFDS split appropriately sharded."""
  if split in ["train", "valid", "train_eval"]:
    return tfds.Split.TRAIN
  elif split == "test":
    return tfds.Split.TEST


def _to_imagenet100_split(split):
  """Returns the TFDS split appropriately sharded."""
  if split in ["train", "train_eval"]:
    return "train"
  elif split == "test":
    return "validation"


def _to_imagenet2012_split(split):
  """Returns the TFDS split appropriately sharded."""
  if split in ["train", "train_eval"]:
    return "train"
  elif split == "test":
    return "validation"


def _random_crop(image):
  """Make a random crop of 224."""
  img_size = (tf.shape(image)[0], tf.shape(image)[1])
  area = tf.cast(img_size[1] * img_size[0], tf.float32)
  target_area = tf.random.uniform([], FLAGS.area_lb, 1.0,
                                  dtype=tf.float32) * area

  log_ratio = (tf.math.log(3 / 4), tf.math.log(4 / 3))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([], *log_ratio, dtype=tf.float32))

  w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
  h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

  w = tf.minimum(w, img_size[1])
  h = tf.minimum(h, img_size[0])

  offset_w = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[1] - w + 1,
                               dtype=tf.int32)
  offset_h = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[0] - h + 1,
                               dtype=tf.int32)
  image = image[offset_h:offset_h + h, offset_w:offset_w + w, :]
  return image


def _center_crop(image, img_size=224, crop_padding=32):
  """Make a random crop of 224."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  padded_center_crop_size = tf.cast(
      ((img_size / (img_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  image = image[offset_height:offset_height + img_size,
                offset_width:offset_width + img_size, :]
  return image


def get_dataset(split, seed=0):
  batch_size = FLAGS.batch_size
  data_root = FLAGS.data_root
  if FLAGS.dataset in ["cifar-10"]:
    ds = tfds.load('cifar10',
                   split=_to_tfds_split(split),
                   data_dir=data_root,
                   shuffle_files=True)
    jpg = False
  elif FLAGS.dataset in ["imagenet-100"]:
    if FLAGS.use_gcs:
      ds = tfds.load(
          'imagenet100',
          split=_to_imagenet100_split(split),
          data_dir=FLAGS.gcs_path,
          shuffle_files=True,
          decoders={'image': {
              'encoded': tfds.decode.SkipDecoding()
          }},
          try_gcs=FLAGS.use_gcs)
    else:
      ds = tfds.load(
          'imagenet100',
          split=_to_imagenet100_split(split),
          data_dir=data_root,
          shuffle_files=True,
          decoders={'image': {
              'encoded': tfds.decode.SkipDecoding()
          }})
      # builder = tfds.ImageFolder(
      #     os.path.join(data_root, 'downloads/imagenet-100'))
      # ds = builder.as_dataset(split=_to_imagenet100_split(split),
      #                         shuffle_files=True,
      #                         decoders={'image': tfds.decode.SkipDecoding()})
    jpg = True
  elif FLAGS.dataset in ["imagenet2012"]:
    splits = tfds.even_splits(_to_imagenet2012_split(split),
                              n=jax.process_count(),
                              drop_remainder=True)
    process_split = splits[jax.process_index()]
    ds = tfds.load('imagenet2012',
                   split=process_split,
                   data_dir=FLAGS.gcs_path if FLAGS.use_gcs else data_root,
                   shuffle_files=True,
                   decoders={'image': tfds.decode.SkipDecoding()},
                   try_gcs=FLAGS.use_gcs)
    jpg = True
  ds = ds.repeat()
  ds = ds.shuffle(buffer_size=10 * batch_size,
                  seed=jax.process_index() * 1234 + seed)
  # Only does data loading for the current process.
  is_parallel = jax.device_count() > 1
  num_parallel = jax.local_device_count()
  md = get_dataset_metadata(FLAGS.dataset)

  def preprocess_image(image):
    if not jpg:
      image = tf.image.convert_image_dtype(image, tf.float32)
      if split == "train" and FLAGS.aug:
        image = _random_crop(image)
        image = tf.image.resize(image, [md['input_height'], md['input_width']],
                                tf.image.ResizeMethod.BICUBIC)
        image = tf.clip_by_value(image, 0.0, 1.0)
      return image
    else:
      if split == "train":
        if FLAGS.aug:
          image = _decode_and_random_crop(image, area_lb=FLAGS.area_lb)
        else:
          image = _decode_and_center_crop(image, md['input_height'])
        image = tf.image.random_flip_left_right(image)
      else:
        image = _decode_and_center_crop(image, md['input_height'])
      image = tf.image.resize(image, [md['input_height'], md['input_width']],
                              tf.image.ResizeMethod.BICUBIC)
      image = tf.clip_by_value(image / 255., 0., 1.)
    return image

  def preprocess(example):
    if 'label' in example:
      label = tf.cast(example['label'], tf.int32)
    else:
      label = tf.cast(example['image']['class']['label'], tf.int32)
    if type(example['image']) is dict:
      image = example['image']['encoded']
    else:
      image = example['image']
    if split == "train":
      view1 = preprocess_image(image)
      view2 = preprocess_image(image)
      return {'view1': view1, 'view2': view2, 'labels': label}
    else:
      image = preprocess_image(image)
      return {'image': image, 'labels': label}

  ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)

  def index(batch):
    batch['labels_idx'] = tf.range(
        FLAGS.batch_size) + jax.process_index() * FLAGS.batch_size
    if is_parallel:
      for k in batch:
        batch[k] = tf.reshape(batch[k],
                              [num_parallel, batch_size // num_parallel] +
                              list(batch[k].shape[1:]))
    return batch

  ds = ds.map(index, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if FLAGS.dataset in ["cifar-10"]:
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  else:
    ds = ds.prefetch(buffer_size=2)
    # ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  yield from tfds.as_numpy(ds)


def get_optimizer(lr,
                  num_epochs,
                  num_ex,
                  batch_size,
                  schedule,
                  weight_decay_mask=None):
  num_proc = jax.process_count()
  num_batches = num_ex // num_proc // batch_size

  if schedule == "linear":
    scheduler = optax.linear_schedule(init_value=lr,
                                      end_value=0.0,
                                      transition_steps=num_epochs *
                                      num_batches,
                                      transition_begin=0)
  elif schedule == "warmup_cosine":
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=FLAGS.warmup_epochs * num_batches,
        decay_steps=num_epochs * num_batches)
  else:
    assert False

  optimizer_name = FLAGS.optimizer
  if optimizer_name in ['sgd']:
    optimizer = optax.sgd(scheduler, FLAGS.mom, nesterov=False)
  elif optimizer_name in ['adam']:
    optimizer = optax.adam(scheduler)
  elif optimizer_name in ['adam', 'adamw'] and FLAGS.wd > 0.0:
    optimizer = optax.adamw(scheduler,
                            weight_decay=FLAGS.wd,
                            mask=weight_decay_mask)
  elif optimizer_name in ['lars']:
    optimizer = optax.lars(scheduler,
                           weight_decay=FLAGS.wd,
                           weight_decay_mask=weight_decay_mask,
                           momentum=FLAGS.mom)
    print(optimizer)
  else:
    raise ValueError(f'Unknown optimizer {optimizer_name}')
  return optimizer, scheduler


# def run_exp_greedy_v2(mode, lr, train_loader, train_eval_loader, test_loader,
#                       num_epochs, log_dir):
#   assert FLAGS.modular_loss, "Greedy learning needs modular_loss"
#   NBLK = FLAGS.num_blocks
#   md = get_dataset_metadata(FLAGS.dataset)
#   num_ex = md['num_examples_train']
#   num_ex_test = md['num_examples_test']
#   num_proc = jax.process_count()
#   optimizer_list, scheduler_list = [], []
#   for blk in range(NBLK):
#     if mode in [
#         "local_feedback_alignment", "local_stopgrad_feedback_alignment"
#     ]:
#       first_segment = (True, False, False)
#       second_segment = (False, False, False)
#     else:
#       first_segment = (True, False)
#       second_segment = (False, False)
#     if blk == 0:
#       wd_mask = [first_segment] * NFIRST + [second_segment]
#     else:
#       wd_mask = [first_segment] * NLAYER + [second_segment]
#     if FLAGS.linear_scale:
#       scaling = FLAGS.batch_size * jax.process_count() / 256
#       lr_scale = lr * scaling
#     else:
#       lr_scale = lr
#     optimizer_, scheduler_ = get_optimizer(lr_scale,
#                                            num_epochs,
#                                            num_ex,
#                                            FLAGS.batch_size,
#                                            FLAGS.schedule,
#                                            weight_decay_mask=wd_mask)
#     optimizer_list.append(optimizer_)
#     scheduler_list.append(scheduler_)
#   readout_optimizer = optax.sgd(0.01, 0.0, nesterov=False)
#   is_parallel = jax.device_count() > 1
#   num_parallel = jax.local_device_count()
#   num_batches = num_ex // num_proc // FLAGS.batch_size
#   num_batches_test = num_ex_test // num_proc // FLAGS.batch_size
#   num_batches_eval = num_batches_test

#   def get_step_fn(update_fn, blk_idx):

#     def step_fn_blk(batch, params, opt_state, key):
#       key, subkey = jax.random.split(key)
#       block_params = get_blk_params(params, NBLK, blk_idx)
#       grads, (x1, x2), logs = update_fn(block_params, batch, subkey)
#       if is_parallel:
#         grads = [(jax.lax.pmean(p[0], axis_name='i'),
#                   jax.lax.pmean(p[1], axis_name='i')) for p in grads]
#       if not FLAGS.freeze_backbone:
#         updates, opt_state = optimizer_list[blk_idx].update(grads, opt_state)
#         block_params = optax.apply_updates(block_params, updates)
#         params = set_blk_params(params, NBLK, blk_idx, block_params)
#       return params, (x1, x2), opt_state, logs, key

#     return step_fn_blk

#   def readout_step_fn(batch, params, opt_state, key):
#     """Only updates last layer parameter."""
#     key, subkey = jax.random.split(key)
#     grads, logs = update_backprop_readout(params, batch, subkey)
#     if is_parallel:
#       grads = (jax.lax.pmean(grads[0], axis_name='i'),
#                jax.lax.pmean(grads[1], axis_name='i'))
#     updates, opt_state = readout_optimizer.update([grads], opt_state)
#     params[-1] = optax.apply_updates([params[-1]], updates)[0]
#     return params, opt_state, logs, key

#   # batch_size = FLAGS.batch_size
#   if mode == "forward_grad_activations":
#     get_update_fn = get_forward_grad_activations_fn
#   elif mode == "local_stopgrad_backprop":
#     get_update_fn = get_local_stopgrad_greedy_backprop_fn
#   else:
#     assert False

#   update_fn = [get_update_fn(blk) for blk in range(NBLK)]
#   step_fn = [get_step_fn(update_fn[blk], blk) for blk in range(NBLK)]

#   if is_parallel:
#     step_fn = [jax.pmap(
#         s,
#         axis_name='i',
#     ) for s in step_fn]
#     readout_step_fn = jax.pmap(
#         readout_step_fn,
#         axis_name='i',
#     )
#     # accuracy_fn = jax.pmap(accuracy, axis_name='i')
#     accuracy_fn = jax.pmap(lambda params, batch: jax.lax.pmean(
#         accuracy(params, batch), axis_name='i'),
#                            axis_name='i')
#   else:
#     step_fn = [jax.jit(s) for s in step_fn]
#     readout_step_fn = jax.jit(readout_step_fn)
#     accuracy_fn = jax.jit(accuracy)

#   layer_sizes = get_layer_sizes(md,
#                                 FLAGS.num_patches,
#                                 FLAGS.num_channel_mlp_units,
#                                 FLAGS.num_blocks,
#                                 FLAGS.num_groups,
#                                 FLAGS.concat_groups,
#                                 FLAGS.same_head,
#                                 FLAGS.conv_mixer,
#                                 FLAGS.kernel_size,
#                                 num_proj_units=FLAGS.num_proj_units)
#   print('layer sizes', layer_sizes)
#   param_scale = get_param_scale(FLAGS.init_scheme, layer_sizes)
#   print("param scale", param_scale)
#   params = init_random_params(param_scale, layer_sizes)
#   if is_parallel:
#     params = jax.pmap(lambda i: init_random_params(param_scale, layer_sizes),
#                       axis_name='i')(jnp.zeros([num_parallel],
#                                                dtype=jnp.int32))
#   else:
#     params = init_random_params(param_scale, layer_sizes)
#   block_params = [get_blk_params(params, NBLK, blk) for blk in range(NBLK)]
#   opt_state = [None] * NBLK
#   for blk in range(NBLK):
#     if is_parallel:
#       opt_state[blk] = jax.pmap(optimizer_list[blk].init,
#                                 axis_name='i')(block_params[blk])
#     else:
#       opt_state[blk] = optimizer_list[blk].init(block_params[blk])
#   if is_parallel:
#     readout_opt_state = jax.pmap(readout_optimizer.init,
#                                  axis_name='i')([params[-1]])
#   else:
#     readout_opt_state = readout_optimizer.init([params[-1]])

#   if is_parallel:
#     key = jax.pmap(lambda i: jax.random.PRNGKey(i),
#                    axis_name='i')(jnp.arange(num_parallel))
#   else:
#     key = jax.random.PRNGKey(0)
#   if jax.process_index() == 0:
#     writer = metric_writers.create_default_writer(log_dir, asynchronous=True)
#   else:
#     writer = metric_writers.MultiWriter([])  # writing is a no-op
#   step = 0

#   for epoch in range(num_epochs):
#     total_loss = 0.0
#     start_time = time.time()
#     # num_batches = num_ex // num_proc // batch_size
#     if epoch % FLAGS.jax_trace == 0:
#       jax.profiler.start_trace(log_dir)
#     for _ in range(num_batches):
#       batch = next(train_loader)
#       logs = []
#       for blk in range(NBLK):
#         params, (x1, x2), opt_state_, logs_, key = step_fn[blk](batch, params,
#                                                                 opt_state[blk],
#                                                                 key)
#         batch['view1'] = x1
#         batch['view2'] = x2
#         opt_state[blk] = opt_state_
#         logs.append(logs_)
#       if is_parallel:
#         total_loss += logs[-1][f'local_loss/blk_{NBLK-1}'][0] / float(
#             num_batches)
#       else:
#         total_loss += logs[-1][f'local_loss/blk_{NBLK-1}'] / float(num_batches)
#       if step % 10 == 0:
#         for logs_ in logs:
#           writer.write_scalars(
#               step, {k: v[0] if is_parallel else v for k, v in logs_.items()})
#       step += 1
#     if jnp.isnan(total_loss):
#       print('total loss exploded', total_loss)
#       break
#     epoch_time = time.time() - start_time
#     if epoch % FLAGS.jax_trace == 0:
#       jax.profiler.stop_trace()

#     # Run readout for one epoch.
#     for _ in range(num_batches):
#       batch = next(train_eval_loader)
#       params, readout_opt_state, logs, key = readout_step_fn(
#           batch, params, readout_opt_state, key)
#       readout_loss = logs['loss']
#     writer.write_scalars(
#         step, {'loss': readout_loss[0] if is_parallel else readout_loss})

#     train_acc = 0.0
#     for _ in range(num_batches_eval):
#       batch = next(train_eval_loader)
#       train_acc_ = accuracy_fn(params, batch)
#       if is_parallel:
#         train_acc_ = train_acc_[0]
#       train_acc += train_acc_ / float(num_batches_eval)

#     test_acc = 0.0
#     for batch in range(num_batches_test):
#       batch = next(test_loader)
#       test_acc_ = accuracy_fn(params, batch)
#       if is_parallel:
#         test_acc_ = test_acc_[0]
#       test_acc += test_acc_ / float(num_batches_test)

#     lr_ = scheduler_list[-1](epoch * num_batches)
#     writer.write_scalars(step, {
#         'acc/train': train_acc,
#         'acc/test': test_acc,
#         'lr': lr_
#     })
#     writer.flush()
#     if jax.process_index() == 0:
#       msg = "Mode {} Host {} LR {:.2e} Epoch {} Time {:.2f}s Train Loss {:.2f} Train Acc {:.2f}% Test Acc {:.2f}%".format(
#           mode, jax.process_index(), lr_, epoch, epoch_time, total_loss,
#           train_acc * 100.0, test_acc * 100.0)
#       print(msg)

#       # Save checkpoint.
#       if is_parallel:
#         params_save = jax.tree_util.tree_map(lambda p: p[0], params)
#         opt_state_save = jax.tree_util.tree_map(lambda p: p[0], opt_state)
#         readout_opt_state_save = jax.tree_util.tree_map(
#             lambda p: p[0], readout_opt_state)
#       else:
#         params_save = params
#         opt_state_save = opt_state
#         readout_opt_state_save = readout_opt_state
#       ckpt = dict(epoch=epoch,
#                   params=params_save,
#                   opt_state=opt_state_save,
#                   readout_opt_state=readout_opt_state_save,
#                   key=key)
#       save_checkpoint(os.path.join(log_dir, "ckpt"), epoch, ckpt)


def run_exp(mode, lr, train_loader, train_eval_loader, test_loader, num_epochs,
            log_dir):
  if mode == "backprop":
    update_fn = update_backprop
  elif mode == "forward_grad_weights":
    update_fn = update_forward_grad_weights
  elif mode == "global_forward_grad_weights":
    update_fn = update_global_forward_grad_weights
  elif mode == "forward_grad_activations":
    update_fn = update_forward_grad_activations
  elif mode == "global_forward_grad_activations":
    update_fn = update_global_forward_grad_activations
  elif mode == "local_backprop":
    update_fn = update_local_backprop
  elif mode == "local_stopgrad_backprop":
    update_fn = update_local_stopgrad_backprop
  elif mode == "local_stopgrad_overlap_backprop":
    update_fn = update_local_stopgrad_overlap_backprop
  elif mode == "feedback_alignment":
    update_fn = update_feedback_alignment
  elif mode == "local_feedback_alignment":
    update_fn = update_local_feedback_alignment
  elif mode == "local_stopgrad_feedback_alignment":
    update_fn = update_local_stopgrad_feedback_alignment
  elif mode == "direct_feedback_alignment":
    update_fn = update_direct_feedback_alignment
  else:
    assert False

  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)

  md = get_dataset_metadata(FLAGS.dataset)
  layer_sizes = get_layer_sizes(md,
                                FLAGS.num_patches,
                                FLAGS.num_channel_mlp_units,
                                FLAGS.num_blocks,
                                FLAGS.num_groups,
                                FLAGS.concat_groups,
                                FLAGS.same_head,
                                FLAGS.conv_mixer,
                                FLAGS.kernel_size,
                                num_proj_units=FLAGS.num_proj_units)
  print('layer sizes', layer_sizes)
  param_scale = get_param_scale(FLAGS.init_scheme, layer_sizes)
  print("param scale", param_scale)
  ckpt_fn_list = last_checkpoint(log_dir)
  if ckpt_fn_list is not None:
    for ckpt_fn_ in ckpt_fn_list:
      print('Load from checkpoint {}'.format(ckpt_fn_))
      try:
        with tf.io.gfile.GFile(ckpt_fn_, 'rb') as f:
          ckpt = pkl.load(f)
      except pkl.UnpicklingError as e:
        continue
      if "epoch" in ckpt:
        epoch_start = ckpt["epoch"] + 1
      else:
        epoch_start = int(os.path.basename(ckpt_fn_).split('-')[-1]) + 1
      break
    params = ckpt["params"]
    opt_state = ckpt["opt_state"]
    readout_opt_state = ckpt["readout_opt_state"]
    key = ckpt["key"]
  else:
    epoch_start = 0
    params = init_random_params(param_scale, layer_sizes)
    if mode in [
        "feedback_alignment", "local_feedback_alignment",
        "local_stopgrad_feedback_alignment"
    ]:
      params_bw = init_random_params(param_scale, layer_sizes)
      params = [(p[0], p[1], pbw[0]) for p, pbw in zip(params, params_bw)]
    elif mode in ["direct_feedback_alignment"]:
      layer_sizes_bw = [[
          l[1] if len(l) == 2 else l[0] * l[2], md["num_classes"]
      ] for l in layer_sizes]
      params_bw = init_random_params(param_scale, layer_sizes_bw)
      params = [
          (p[0], p[1], pbw[0], pbw[1]) for p, pbw in zip(params, params_bw)
      ]
    key = jax.random.PRNGKey(0)

  is_parallel = jax.device_count() > 1
  num_parallel = jax.local_device_count()
  num_ex = md['num_examples_train']
  num_ex_test = md['num_examples_test']
  num_proc = jax.process_count()
  num_batches = num_ex // num_proc // FLAGS.batch_size
  num_batches_test = num_ex_test // num_proc // FLAGS.batch_size
  num_batches_eval = num_batches_test
  if mode in [
      "feedback_alignment", "local_feedback_alignment",
      "local_stopgrad_feedback_alignment"
  ]:
    wd_mask = [(True, False, False)] * NFIRST + [
        (True, False, False)
    ] * (FLAGS.num_blocks - 1) * NLAYER + [(False, False, False)
                                          ] * (FLAGS.num_blocks + 2)
  elif mode in ["direct_feedback_alignment"]:
    wd_mask = [(True, False, False, False)] * NFIRST + [
        (True, False, False, False)
    ] * (FLAGS.num_blocks - 1) * NLAYER + [(False, False, False, False)
                                          ] * (FLAGS.num_blocks + 2)
  else:
    wd_mask = [(True, False)] * NFIRST + [
        (True, False)
    ] * (FLAGS.num_blocks - 1) * NLAYER + [(False, False)
                                          ] * (FLAGS.num_blocks + 2)
  if FLAGS.linear_scale:
    scaling = FLAGS.batch_size * jax.process_count() / 256
    lr_scale = lr * scaling
  else:
    lr_scale = lr
  optimizer, scheduler = get_optimizer(lr_scale,
                                       num_epochs,
                                       num_ex,
                                       FLAGS.batch_size,
                                       FLAGS.schedule,
                                       weight_decay_mask=wd_mask)
  readout_optimizer = optax.sgd(0.1 * FLAGS.last_layer_lr, 0.0, nesterov=False)

  if is_parallel:
    params = jax.pmap(lambda i: params,
                      axis_name='i')(jnp.zeros([num_parallel],
                                               dtype=jnp.int32))

  def step_fn(batch, params, opt_state, key):
    key, subkey = jax.random.split(key)
    if not FLAGS.freeze_backbone:
      grads, logs = update_fn(params, batch, subkey)
      if is_parallel:
        grads = jax.tree_util.tree_map(
            lambda p: jax.lax.pmean(p, axis_name='i'), grads)
      updates, opt_state = optimizer.update(grads, opt_state, params=params)
      params = optax.apply_updates(params, updates)
    return params, opt_state, logs, key

  def readout_step_fn(batch, params, opt_state, key):
    """Only updates last layer parameter."""
    key, subkey = jax.random.split(key)
    grads, logs = update_backprop_readout(params, batch, subkey)
    if is_parallel:
      grads = jax.tree_util.tree_map(lambda p: jax.lax.pmean(p, axis_name='i'),
                                     grads)
    updates, opt_state = readout_optimizer.update([grads], opt_state)
    params[-1] = optax.apply_updates([params[-1]], updates)[0]
    return params, opt_state, logs, key

  if is_parallel:
    step_fn = jax.pmap(
        step_fn,
        axis_name='i',
    )
    readout_step_fn = jax.pmap(
        readout_step_fn,
        axis_name='i',
    )
    accuracy_fn = jax.pmap(lambda params, batch: jax.lax.pmean(
        accuracy(params, batch), axis_name='i'),
                           axis_name='i')
    if epoch_start == 0:
      opt_state = jax.pmap(optimizer.init, axis_name='i')(params)
      key = jax.pmap(lambda i: jax.random.PRNGKey(i),
                     axis_name='i')(jnp.arange(num_parallel))
      readout_opt_state = jax.pmap(readout_optimizer.init,
                                   axis_name='i')([params[-1]])
    else:
      opt_state = jax.pmap(lambda i: opt_state,
                           axis_name='i')(jnp.zeros([num_parallel],
                                                    dtype=jnp.int32))
      key = jax.pmap(lambda i: i, axis_name='i')(key)
      readout_opt_state = jax.pmap(lambda i: readout_opt_state,
                                   axis_name='i')(jnp.zeros([num_parallel],
                                                            dtype=jnp.int32))
  else:
    step_fn = jax.jit(step_fn)
    readout_step_fn = jax.jit(readout_step_fn)
    accuracy_fn = jax.jit(accuracy)
    if epoch_start == 0:
      opt_state = optimizer.init(params)
      key = jax.random.PRNGKey(0)
      readout_opt_state = readout_optimizer.init([params[-1]])

  if jax.process_index() == 0:
    writer = metric_writers.create_default_writer(log_dir, asynchronous=True)
  else:
    writer = metric_writers.MultiWriter([])  # writing is a no-op

  step = epoch_start * num_batches
  lr_ = scheduler(epoch_start * num_batches)
  for epoch in range(epoch_start, num_epochs):
    total_loss = 0.0
    start_time = time.time()
    # if epoch % FLAGS.jax_trace == 0:
    #   jax.profiler.start_trace(log_dir)
    for _ in range(num_batches):
      batch = next(train_loader)
      params, opt_state, logs, key = step_fn(batch, params, opt_state, key)

      if is_parallel:
        total_loss += logs['local_loss/final'][0] / float(num_batches)
      else:
        total_loss += logs['local_loss/final'] / float(num_batches)
      if step % 10 == 0:
        for k in logs:
          writer.write_scalars(
              step, {k: v[0] if is_parallel else v for k, v in logs.items()})
      step += 1
    if jnp.isnan(total_loss):
      print('total loss exploded', total_loss)
      break
    epoch_time = time.time() - start_time
    # if epoch % FLAGS.jax_trace == 0:
    #   jax.profiler.stop_trace()

    # Run readout for one epoch.
    for _ in range(num_batches):
      batch = next(train_eval_loader)
      params, readout_opt_state, logs, key = readout_step_fn(
          batch, params, readout_opt_state, key)
      readout_loss = logs['loss']
    writer.write_scalars(
        step, {'loss': readout_loss[0] if is_parallel else readout_loss})

    train_acc = 0.0
    for _ in range(num_batches_eval):
      batch = next(train_eval_loader)
      train_acc_ = accuracy_fn(params, batch)
      if is_parallel:
        train_acc_ = train_acc_[0]
      train_acc += train_acc_ / float(num_batches_eval)

    test_acc = 0.0
    for batch in range(num_batches_test):
      batch = next(test_loader)
      test_acc_ = accuracy_fn(params, batch)
      if is_parallel:
        test_acc_ = test_acc_[0]
      test_acc += test_acc_ / float(num_batches_test)

    lr_ = scheduler(epoch * num_batches)
    writer.write_scalars(step, {
        'acc/train': train_acc,
        'acc/test': test_acc,
        'lr': lr_
    })
    writer.flush()
    if jax.process_index() == 0:
      msg = "Mode {} Host {} LR {:.2e} Epoch {} Time {:.2f}s Loss {:.2f} Train Acc {:.2f}% Test Acc {:.2f}%".format(
          mode, jax.process_index(), lr_, epoch, epoch_time, total_loss,
          train_acc * 100.0, test_acc * 100.0)
      print(msg)

      # Save checkpoint.
      if is_parallel:
        params_save = jax.tree_util.tree_map(lambda p: p[0], params)
        opt_state_save = jax.tree_util.tree_map(lambda p: p[0], opt_state)
        readout_opt_state_save = jax.tree_util.tree_map(
            lambda p: p[0], readout_opt_state)
      else:
        params_save = params
        opt_state_save = opt_state
        readout_opt_state_save = readout_opt_state
      ckpt = dict(epoch=epoch,
                  params=params_save,
                  opt_state=opt_state_save,
                  readout_opt_state=readout_opt_state_save,
                  key=key)
      save_checkpoint(os.path.join(log_dir, "ckpt"), epoch, ckpt)


def main(_):
  num_epochs = FLAGS.num_epochs
  experiment = FLAGS.exp
  train_loader = get_dataset("train", seed=0)
  train_eval_loader = get_dataset("train_eval", seed=1)
  test_loader = get_dataset("test", seed=0)
  exp_dir = FLAGS.workdir

  if experiment == "all":
    if FLAGS.greedy:
      keys = ['local_stopgrad_backprop', 'forward_grad_activations']
    else:
      keys = [
          'backprop', 'local_backprop', 'local_stopgrad_backprop',
          'forward_grad_activations'
      ]
    lr_list = [float(s) for s in FLAGS.lr.split(',')]
    all_results = {}
    for mode in keys:
      all_results[mode] = {}
      log_dir = f'{exp_dir}/{mode}'
      if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
      for lr in lr_list:
        lr_dir = '{}/{:.2f}'.format(log_dir, lr)
        if not tf.io.gfile.exists(lr_dir):
          tf.io.gfile.makedirs(lr_dir)

        if FLAGS.greedy and mode not in ["backprop", "local_backprop"]:
          # run_exp_greedy_v2(mode, lr, train_loader, train_eval_loader,
          #                   test_loader, num_epochs, lr_dir)
          assert False, "Not supported yet"
        else:
          run_exp(mode, lr, train_loader, train_eval_loader, test_loader,
                  num_epochs, lr_dir)
  else:
    exp_list = experiment.split(',')
    lr_list = [float(s) for s in FLAGS.lr.split(',')]
    for exp in exp_list:
      log_dir = f'{exp_dir}/{exp}'
      if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
      for lr in lr_list:
        print(lr)
        lr_dir = '{}/{:.2f}'.format(log_dir, lr)
        if not tf.io.gfile.exists(lr_dir):
          tf.io.gfile.makedirs(lr_dir)
        if FLAGS.greedy and exp not in ["backprop", "local_backprop"]:
          # run_exp_greedy_v2(exp, lr, train_loader, train_eval_loader,
          #                   test_loader, num_epochs, lr_dir)
          assert False, "Not supported yet"
        else:
          run_exp(exp, lr, train_loader, train_eval_loader, test_loader,
                  num_epochs, lr_dir)


if __name__ == '__main__':
  app.run(main)
