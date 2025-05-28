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

import jax
import math
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import time
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds

from jax.scipy.special import logsumexp
from mixer_lib import (fa_group_linear, fa_linear, get_blk,
                       get_dataset_metadata, get_layer_sizes, get_param_scale,
                       preprocess, normalize, init_random_params, linear,
                       group_linear, depthwise_conv, get_num_layers,
                       get_blk_idx, avg_pooling, max_pooling)
from train_utils import (save_checkpoint, last_checkpoint)
# Ask TF to not occupy GPU memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import augmentations
from spatial_avg_linear import (spatial_avg_group_linear_custom_jvp,
                                spatial_avg_group_linear_custom_vjp)
from spatial_avg_linear_v2 import (
    spatial_avg_group_linear_cross_entropy_custom_jvp,
    spatial_avg_group_linear_cross_entropy_custom_vjp)

from dataset import (_decode_and_random_crop, _decode_and_center_crop)

from absl import app
from absl import flags
from jax import grad
from clu import metric_writers

flags.DEFINE_string('data_root', None, 'where to store datasets')
flags.DEFINE_string('exp', 'all', 'which experiment to run')
flags.DEFINE_string('workdir', '', 'experiment directory')
flags.DEFINE_float('mom', 0.0, 'momentum')
flags.DEFINE_string('lr', '0.1', 'learning rate')
flags.DEFINE_string('optimizer', 'sgd', 'optimizer name')
flags.DEFINE_float('wd', 0.0001, 'weight decay')
flags.DEFINE_integer('warmup_epochs', 1, 'number of warmup epochs')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
flags.DEFINE_string('init_scheme', 'constant',
                    'kaiming or lecun or constant or list')
flags.DEFINE_integer('num_blocks', 6, 'number of mixer blocks')
flags.DEFINE_integer('num_channel_mlp_units', 256, 'number of units')
flags.DEFINE_integer('num_channel_mlp_hidden_units', -1,
                     'number of hidden units')
flags.DEFINE_integer('num_token_mlp_units', 64, 'number of units')
flags.DEFINE_integer('num_patches', 4, 'number of patches on each side')
flags.DEFINE_string('downsample', '1,1,1,1', 'downsample ratio')
flags.DEFINE_string('channel_ratio', '1,1,1,1', 'channel ratio')
flags.DEFINE_string('group_ratio', '1,1,1,1', 'group ratio')
flags.DEFINE_integer('num_channel_group', 1, 'number of channels for a group')
flags.DEFINE_integer('num_groups', 1, 'number of groups')
flags.DEFINE_float('last_layer_lr', 0.1, 'last layer learning rate')
flags.DEFINE_integer('num_passes', 1, 'number of forward gradient passes')
flags.DEFINE_bool('layer_norm_all', False, 'entire layer')
flags.DEFINE_integer('stop_every', 1, 'stop gradient every')
flags.DEFINE_integer('stop_remainder', -1, 'stop gradient remainder')
flags.DEFINE_string('schedule', 'linear', 'learning rate schedule')
flags.DEFINE_bool('aug', False, 'whether to do random cropping')
flags.DEFINE_bool('batch_norm', False, 'if we run batch norm')
flags.DEFINE_bool('avgpool_token', True, 'average pool tokens before head')
flags.DEFINE_bool('concat_groups', True, 'whether to concat groups')
flags.DEFINE_float('head_lr', 1.0, 'classifier head lr multiplier')
flags.DEFINE_bool('norm_grad', True, 'normalization layer gradients')
flags.DEFINE_bool('same_head', True, 'whether use same head')
flags.DEFINE_bool('fuse_cross_entropy', True, 'whether to fuse operation')
flags.DEFINE_string('dataset', 'cifar-10', 'cifar-10 or imagenet-100')
flags.DEFINE_bool('begin_ln', False, 'layer norm in the beginning')
flags.DEFINE_bool('middle_ln', False, 'layer norm in the middle')
flags.DEFINE_bool('last_layer_ln', False, 'layer norm before read out')
flags.DEFINE_bool('post_linear_ln', True, 'whether to layer norm post linear')
flags.DEFINE_bool('inter_ln', False, 'layer norm before intermediate read out')
flags.DEFINE_bool('augcolor', False, 'whether to augment colors')
flags.DEFINE_float('area_lb', 0.08, 'area crop lower bound')
flags.DEFINE_bool('conv_mixer', False, 'use conv for mixing')
flags.DEFINE_integer('kernel_size', 3, 'conv kernel size')
flags.DEFINE_bool('bp_last_lr', True, 'whether to tune last layer LR for BP')
flags.DEFINE_bool('stopgrad_input', False, 'whether to stopgrad on the input')
flags.DEFINE_bool('freeze_backbone', False, 'whether freeze')
flags.DEFINE_bool('spatial_loss', True,
                  'whether to keep the spatial dimensions')
flags.DEFINE_bool('modular_loss', True,
                  'whether to have loss functions in each module')
flags.DEFINE_string('gcs_path', 'gs://research-brain-rmy-gcp-xgcp',
                    'cloud storage')
flags.DEFINE_bool('use_gcs', False, 'whether to use cloud storage')
flags.DEFINE_float('perturb_epsilon', 1e-3, 'perturbation stepsize')
flags.DEFINE_bool('train_eval', False, 'whether to add train eval')
flags.DEFINE_bool('spatial_pooling', False, 'whether to add spatial pooling')
flags.DEFINE_string('pool_fn', 'avg', 'avg or max')

FLAGS = flags.FLAGS


def normalize_layer(x, swap=False):
  return normalize(x,
                   swap=swap,
                   batch_norm=FLAGS.batch_norm,
                   layer_norm_all=FLAGS.layer_norm_all)


def mlp_block(inputs,
              params,
              num_groups,
              noise=None,
              name='',
              mask=None,
              stop_every_layer=False):
  # Token mixing.
  print('inputs', inputs.shape)
  conv = FLAGS.conv_mixer
  states = {}
  B, P = inputs.shape[0], inputs.shape[1]
  G = num_groups
  H = int(math.sqrt(P))
  if FLAGS.stopgrad_input:
    inputs_ = jax.lax.stop_gradient(inputs)
  else:
    inputs_ = inputs
  outputs = jnp.reshape(inputs_, [B, P, -1])
  if FLAGS.begin_ln:
    outputs = normalize_layer(outputs)
  if conv:
    outputs = jnp.reshape(outputs, [B, H, H, -1])
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
    outputs = normalize(outputs)
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

  outputs = jnp.reshape(outputs, [B, P, G, -1])
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

  if params[1][0].shape[0] != params[1][0].shape[1]:
    # Double the channels.
    inputs = jnp.concatenate([inputs, inputs], axis=2)
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
                                        num_groups,
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
  downsample = [int(d) for d in FLAGS.downsample.split(',')]
  group_ratio = [int(d) for d in FLAGS.group_ratio.split(',')]
  if stop_remainder < 0:
    stop_remainder = stop_remainder + stop_every
  md = get_dataset_metadata(FLAGS.dataset)
  inputs = preprocess(inputs, md['image_mean'], md['image_std'],
                      FLAGS.num_patches)
  x = inputs
  # We will start with a channel mixing MLP instead of token mixing.
  all_states = {}
  all_logs = {}
  num_groups_ = FLAGS.num_groups
  # Build network.
  for blk in range(NBLK):
    start, end = get_blk_idx(blk)
    if noise is not None:
      noise_ = noise[start:end]
    else:
      noise_ = None
    x, (states, logs) = run_block(blk,
                                  num_groups_,
                                  x,
                                  params[start:end],
                                  noise_,
                                  stop_every_layer=stop_every_layer)
    x_proj = x
    if FLAGS.inter_ln:
      x_proj = normalize(x_proj)
    states[f'block_{blk}/pre_pred'] = x_proj

    for k in states:
      all_states[k] = states[k]
    for k in logs:
      all_logs[k] = logs[k]

    if stop_gradient and not FLAGS.stopgrad_input:
      if blk % stop_every == stop_remainder:
        x = jax.lax.stop_gradient(x)

    if downsample[blk] > 1:
      # Downsample 2x
      if FLAGS.pool_fn == "avg":
        x = avg_pooling(x, stride=downsample[blk])
      elif FLAGS.pool_fn == "max":
        x = max_pooling(x, stride=downsample[blk])
    num_groups_ = num_groups_ * group_ratio[blk]

  if readout:
    x = jax.lax.stop_gradient(x)
  x = jnp.reshape(x, [x.shape[0], x.shape[1], -1])
  x = jnp.mean(x, axis=1)  # [B, D]
  if FLAGS.last_layer_ln:
    x = normalize(x)
  all_states[f'pre_final'] = x
  # [B, K]
  if len(params[-1]) == 3:
    pred = fa_linear(x, params[-1][0], params[-1][1], params[-1][2])
  else:
    pred = linear(x, params[-1][0], params[-1][1])
  all_states['pred_final'] = pred
  return pred, (all_states, all_logs)


def loss_dfa(params, batch, noise=None, key=None):

  if FLAGS.augcolor:
    key, subkey = jax.random.split(key)
    batch = augmentations.postprocess1(batch, subkey, add_gaussian_blur=False)
  inputs, targets = batch['image'], batch['label']
  md = get_dataset_metadata(FLAGS.dataset)
  targets_onehot = jax.nn.one_hot(targets, md['num_classes'])
  logits, (states, logs) = predict(params,
                                   inputs,
                                   noise=noise,
                                   stop_gradient=True,
                                   stop_every_layer=True,
                                   readout=True,
                                   is_training=True)
  loss = jnp.mean(classif_loss(logits, targets_onehot))
  predicted_class = jnp.argmax(logits, axis=-1)
  logs['acc/train'] = jnp.mean(predicted_class == targets)
  logs['loss'] = loss
  local_losses = []
  NBLK = FLAGS.num_blocks
  NL = get_num_layers(NBLK)
  for j, (weights, bias, weights_b, bias_b) in enumerate(params[:NL]):
    blk, layer = get_blk(j)
    x_proj = states[f'block_{blk}/post_{layer}']
    B, P = x_proj.shape[0], x_proj.shape[1]
    x_proj = jnp.mean(x_proj, [1])
    x_proj = jnp.reshape(x_proj, [B, -1])
    logit_bw_ = linear(x_proj, weights_b, bias_b)
    local_loss_bw_ = jnp.mean(classif_loss(logit_bw_, targets_onehot))
    local_loss_ = jax.lax.stop_gradient(loss - local_loss_bw_) + local_loss_bw_
    local_losses.append(local_loss_)
    logs[f'local_loss/blk_{blk}'] = local_loss_
  return loss, local_losses, (states, logs)


def loss(params,
         batch,
         noise=None,
         stop_gradient=False,
         readout=False,
         key=None,
         stop_every=1,
         stop_remainder=-1,
         custom_forward=False,
         avg_batch=True):
  """Supervised classification loss."""
  if FLAGS.augcolor:
    key, subkey = jax.random.split(key)
    batch = augmentations.postprocess1(batch, subkey, add_gaussian_blur=False)
  inputs, targets = batch['image'], batch['label']
  md = get_dataset_metadata(FLAGS.dataset)
  targets_onehot = jax.nn.one_hot(targets, md['num_classes'])
  logits, (states, logs) = predict(params,
                                   inputs,
                                   noise=noise,
                                   stop_gradient=stop_gradient,
                                   readout=readout,
                                   stop_every=stop_every,
                                   stop_remainder=stop_remainder,
                                   is_training=True)
  loss = classif_loss(logits, targets_onehot)
  if avg_batch:
    loss = jnp.mean(loss)
  predicted_class = jnp.argmax(logits, axis=-1)
  logs['acc/train'] = jnp.mean(predicted_class == targets)
  if not avg_batch:
    logs['loss'] = jnp.mean(loss)
  else:
    logs['loss'] = loss
  local_losses = []
  NBLK = FLAGS.num_blocks
  NL = get_num_layers(NBLK)
  avgpool_token = FLAGS.avgpool_token
  for blk in range(NBLK):
    x_proj = states[f'block_{blk}/pre_pred']
    B, P = x_proj.shape[0], x_proj.shape[1]
    param_ = params[NL + blk]
    if FLAGS.fuse_cross_entropy:
      if custom_forward:
        loss_fn_ = spatial_avg_group_linear_cross_entropy_custom_jvp
      else:
        loss_fn_ = spatial_avg_group_linear_cross_entropy_custom_vjp
      local_loss_ = loss_fn_(x_proj, param_[0], param_[1], targets_onehot)
      if len(param_) == 3:  # Feedback alignment
        local_loss_bw_ = loss_fn_(x_proj, param_[2], param_[1], targets_onehot)
        local_loss_ = jax.lax.stop_gradient(local_loss_ -
                                            local_loss_bw_) + local_loss_bw_
    else:
      assert False

    if not FLAGS.spatial_loss:
      # [B, P, G] -> [B, 1, G]
      local_loss_ = jnp.mean(local_loss_, axis=1, keepdims=True)

    assert avgpool_token
    if avgpool_token:
      denom = B
    else:
      denom = B * P
    local_losses.append(local_loss_ / float(denom))
    logs[f'local_loss/blk_{blk}'] = jnp.mean(local_loss_)
  if not FLAGS.modular_loss:
    local_losses = local_losses[-1]
  return loss, local_losses, (states, logs)


def classif_loss(logits, targets):
  logits = logits - logsumexp(logits, axis=-1, keepdims=True)
  if len(logits.shape) == 3:
    targets = targets[:, None, :]
  elif len(logits.shape) == 4:
    targets = targets[:, None, None, :]
  elif len(logits.shape) == 5:
    targets = targets[:, None, None, None, :]
  loss = -jnp.sum(logits * targets, axis=-1)
  return loss


def accuracy(params, batch):
  inputs, targets = batch['image'], batch['label']
  pred, (states, logs) = predict(params, inputs)
  predicted_class = jnp.argmax(pred, axis=-1)
  return jnp.mean(predicted_class == targets)


def update_backprop(params, batch, key):
  wd = FLAGS.wd

  def loss_fn(params, batch):
    final_loss_, local_loss_, (states, logs) = loss(params,
                                                    batch,
                                                    noise=None,
                                                    stop_gradient=False,
                                                    readout=False,
                                                    key=key)
    return final_loss_, (states, logs)

  grads_now, (states, logs) = grad(loss_fn, has_aux=True)(params, batch)

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.last_layer_lr < 1.0 and FLAGS.bp_last_lr:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr)

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def update_local_backprop(params, batch, key):
  wd = FLAGS.wd
  NBLK = FLAGS.num_blocks
  avgpool_token = FLAGS.avgpool_token
  G = FLAGS.num_groups
  NL = get_num_layers(NBLK)
  P = FLAGS.num_patches**2
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]

  def local_loss(params, batch):
    final_loss_, local_loss_, (states, logs) = loss(params,
                                                    batch,
                                                    noise=None,
                                                    stop_gradient=False,
                                                    readout=True,
                                                    key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states, logs)

  grads_now, (states, logs) = grad(local_loss, has_aux=True)(params, batch)
  if FLAGS.last_layer_lr < 1.0:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr)

  if FLAGS.same_head:
    G_ = G
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G_), grads_now[i][1] / float(G_))
      G_ = G_ * group_ratio[i - NL]

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    P_ = P
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P_), grads_now[i][1] / float(P_))
      P_ = P_ // (downsample[i - NL]**2)

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def update_local_stopgrad_backprop(params, batch, key):
  wd = FLAGS.wd
  stop_every = FLAGS.stop_every
  stop_remainder = FLAGS.stop_remainder
  NBLK = FLAGS.num_blocks
  avgpool_token = FLAGS.avgpool_token
  NL = get_num_layers(NBLK)
  P = FLAGS.num_patches**2
  G = FLAGS.num_groups
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]

  def local_loss(params, batch):
    final_loss_, local_loss_, (states,
                               logs) = loss(params,
                                            batch,
                                            noise=None,
                                            stop_gradient=True,
                                            readout=True,
                                            key=key,
                                            stop_every=stop_every,
                                            stop_remainder=stop_remainder)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states, logs)

  grads_now, (states, logs) = grad(local_loss, has_aux=True)(params, batch)

  if FLAGS.last_layer_lr < 1.0:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr)

  if FLAGS.same_head:
    G_ = G
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G_), grads_now[i][1] / float(G_))
      G_ = G_ * group_ratio[i - NL]

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    P_ = P
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P_), grads_now[i][1] / float(P_))
      P_ = P_ // (downsample[i - NL]**2)

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
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
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]

  def local_loss_(params, batch):
    final_loss1_, local_loss1_, (states, logs) = loss(params,
                                                      batch,
                                                      noise=None,
                                                      stop_gradient=True,
                                                      key=key,
                                                      stop_every=2,
                                                      stop_remainder=0)
    final_loss2_, local_loss2_, (_, logs) = loss(params,
                                                 batch,
                                                 noise=None,
                                                 stop_gradient=True,
                                                 key=key,
                                                 stop_every=2,
                                                 stop_remainder=1)
    final_loss_ = final_loss1_ + final_loss2_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss1_])) + jnp.sum(
            jnp.stack([jnp.sum(l) for l in local_loss2_]))
    return final_loss_, (states, logs)

  grads_now, (_, logs) = grad(local_loss_, has_aux=True)(params, batch)

  if FLAGS.last_layer_lr < 1.0:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr)

  if FLAGS.same_head:
    G_ = G
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G_), grads_now[i][1] / float(G_))
      G_ = G_ * group_ratio[i - NL]

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    P_ = P
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P_), grads_now[i][1] / float(P_))
      P_ = P_ // (downsample[i - NL]**2)

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb)
                 for (gw, gb), (w, b) in zip(grads_now[:-1], params[:-1])
                ] + grads_now[-1:]
  return grads_now, logs


def update_forward_grad_weights(params, batch, key):
  num_patches = FLAGS.num_patches
  num_groups = FLAGS.num_groups
  num_passes = FLAGS.num_passes
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  if FLAGS.fuse_cross_entropy:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, targets))
  else:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        classif_loss(spatial_avg_group_linear_custom_vjp(x, w, b), targets))
  local_classif_grad = jax.grad(local_classif_loss, argnums=[1, 2])
  global_classif_loss = lambda x, w, b, targets: jnp.mean(
      classif_loss(jnp.einsum('nc,cd->nd', x, w) + b, targets))
  global_classif_grad = jax.grad(global_classif_loss, argnums=[1, 2])
  wd = FLAGS.wd

  def local_loss(params):
    _, local_loss_, (states, logs) = loss(params,
                                          batch,
                                          stop_gradient=FLAGS.modular_loss,
                                          readout=True,
                                          key=key,
                                          custom_forward=True)
    return local_loss_, (states, logs)

  for npass in range(num_passes):
    noise = []
    num_items = num_patches**2
    NBLK = FLAGS.num_blocks
    label = jax.nn.one_hot(batch['label'], md['num_classes'])
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
    _, g, (states, logs) = jax.jvp(local_loss, [params], [noise], has_aux=True)

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
        # Last classification layer.
        pre_act = states[f'pre_final']
        grad_w, grad_b = global_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.last_layer_lr
        grad_b = grad_b * FLAGS.last_layer_lr
      else:
        # Intermediate classification layer.
        pre_act = states[f'block_{blk}/pre_pred']
        grad_w, grad_b = local_classif_grad(pre_act, weight, bias, label)
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
  if FLAGS.fuse_cross_entropy:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, targets))
  else:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        classif_loss(spatial_avg_group_linear_custom_vjp(x, w, b), targets))
  local_classif_grad = jax.grad(local_classif_loss, argnums=[1, 2])
  global_classif_loss = lambda x, w, b, targets: jnp.mean(
      classif_loss(jnp.einsum('nc,cd->nd', x, w) + b, targets))
  global_classif_grad = jax.grad(global_classif_loss, argnums=[1, 2])
  wd = FLAGS.wd

  def local_loss(params):
    loss_, local_loss_, (states, logs) = loss(params,
                                              batch,
                                              stop_gradient=False,
                                              readout=False,
                                              key=key,
                                              custom_forward=True)
    return loss_, (states, logs)

  for npass in range(num_passes):
    noise = []
    num_items = num_patches**2
    NBLK = FLAGS.num_blocks
    label = jax.nn.one_hot(batch['label'], md['num_classes'])
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
    _, g, (states, logs) = jax.jvp(local_loss, [params], [noise], has_aux=True)

    # Main network layers.
    for i, ((weight, bias), (dw, db)) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      g_ = g

      if blk > 0 and layer == 0:
        grad_w = g_ * dw
        grad_b = g_ * db
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        # TODO: here needs attention!
        dw = jnp.reshape(dw, [weight.shape[0], G, weight.shape[1] // G])
        db = jnp.reshape(db, [G, -1])
        grad_w = g_ * dw
        grad_b = g_ * db
      elif (blk == 0 and layer == 1) or layer == 2:
        # TODO: here needs attention!
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
        # Last classification layer.
        pre_act = states[f'pre_final']
        grad_w, grad_b = global_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.last_layer_lr
        grad_b = grad_b * FLAGS.last_layer_lr
      else:
        # Intermediate classification layer.
        pre_act = states[f'block_{blk}/pre_pred']
        grad_w, grad_b = local_classif_grad(pre_act, weight, bias, label)
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


def update_perturb_weights(params, batch, key):
  num_patches = FLAGS.num_patches
  num_groups = FLAGS.num_groups
  num_passes = FLAGS.num_passes
  epsilon = FLAGS.perturb_epsilon
  wd = FLAGS.wd
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  if FLAGS.fuse_cross_entropy:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, targets))
  else:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        classif_loss(spatial_avg_group_linear_custom_vjp(x, w, b), targets))
  local_classif_grad = jax.grad(local_classif_loss, argnums=[1, 2])
  global_classif_loss = lambda x, w, b, targets: jnp.mean(
      classif_loss(jnp.einsum('nc,cd->nd', x, w) + b, targets))
  global_classif_grad = jax.grad(global_classif_loss, argnums=[1, 2])

  def local_loss(params):
    _, local_loss_, (states, logs) = loss(params,
                                          batch,
                                          stop_gradient=FLAGS.modular_loss,
                                          readout=True,
                                          key=key,
                                          custom_forward=True)
    return local_loss_, (states, logs)

  for npass in range(num_passes):
    noise = []
    NBLK = FLAGS.num_blocks
    label = jax.nn.one_hot(batch['label'], md['num_classes'])
    G = num_groups
    M = num_passes
    NL = get_num_layers(NBLK)
    main_params = params[:NL]
    loss_params = params[NL:]

    for i, (weight, bias) in enumerate(main_params):
      blk, layer = get_blk(i)
      key, subkey = jax.random.split(key)
      dw = jax.random.normal(subkey, weight.shape) * epsilon
      key, subkey = jax.random.split(key)
      db = jax.random.normal(subkey, bias.shape) * epsilon
      noise.append((dw, db))

    for i, (weight, bias) in enumerate(loss_params):
      noise.append((jnp.zeros_like(weight), jnp.zeros_like(bias)))

    # [L,B,P]
    # _, g, (states, logs) = jax.jvp(local_loss, [params], [noise], has_aux=True)
    local_loss1_, (states, logs) = local_loss(params)
    local_loss2_, (_, _) = local_loss([
        (w + nw, b + nb) for (w, b), (nw, nb) in zip(params, noise)
    ])

    # Main network layers.
    for i, ((weight, bias), (dw, db)) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      if FLAGS.modular_loss:
        g_ = (local_loss2_[blk] - local_loss1_[blk]) / (epsilon**2)
      else:
        g_ = (local_loss2_ - local_loss1_) / (epsilon**2)

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
        # Last classification layer.
        pre_act = states[f'pre_final']
        grad_w, grad_b = global_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.last_layer_lr
        grad_b = grad_b * FLAGS.last_layer_lr
      else:
        # Intermediate classification layer.
        pre_act = states[f'block_{blk}/pre_pred']
        grad_w, grad_b = local_classif_grad(pre_act, weight, bias, label)
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


def sample_activation_noise(params,
                            key,
                            batch_size,
                            num_patches,
                            num_hid_units,
                            num_units,
                            num_groups,
                            num_blocks,
                            conv=False,
                            num_passes=1):
  noise = []
  H = num_hid_units
  D = num_units
  G = num_groups
  P = num_patches
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]
  channel_ratio = [int(c) for c in FLAGS.channel_ratio.split(',')]
  B = batch_size
  NBLK = num_blocks
  NL = get_num_layers(NBLK)
  main_params = params[:NL]
  loss_params = params[NL:]
  group_list = [G]
  token_list = [P]
  hid_unit_list = [H]
  unit_list = [D]
  for blk in range(NBLK - 1):
    group_list.append(group_list[-1] * group_ratio[blk])
    token_list.append(token_list[-1] // (downsample[blk]**2))
    hid_unit_list.append(hid_unit_list[-1] * channel_ratio[blk])
    unit_list.append(unit_list[-1] * channel_ratio[blk])
  for i, p in enumerate(main_params):
    blk, layer = get_blk(i)
    G_ = group_list[blk]
    P_ = token_list[blk]
    H_ = hid_unit_list[blk]
    if blk > 0:
      I_ = unit_list[blk - 1]
    D_ = unit_list[blk]
    if blk == 0:
      if layer == 0:
        out_shape = [B, P_, H_]
      else:
        out_shape = [B, P_, G_, D_ // G_]
    else:
      if layer == 0:
        if conv:
          out_shape = [B, P_, I_]
        else:
          out_shape = [B, I_, P_]
      elif layer == 1:
        out_shape = [B, P_, H_]
      else:
        out_shape = [B, P_, G_, D_ // G_]
    if num_passes > 1:
      out_shape = [num_passes] + out_shape
    key, subkey = jax.random.split(key)
    dz = jax.random.normal(subkey, out_shape)
    noise.append(dz)
  return noise, key


def update_forward_grad_activations(params, batch, key):
  num_patches = FLAGS.num_patches
  num_units = FLAGS.num_channel_mlp_units
  num_hid_units = FLAGS.num_channel_mlp_hidden_units
  num_items = num_patches**2
  NBLK = FLAGS.num_blocks
  if num_hid_units < 0:
    num_hid_units = num_units
  num_groups = FLAGS.num_groups
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]
  channel_ratio = [int(c) for c in FLAGS.channel_ratio.split(',')]

  group_list = [num_groups]
  token_list = [num_patches**2]
  hid_unit_list = [num_hid_units]
  unit_list = [num_units]
  for blk in range(NBLK - 1):
    group_list.append(group_list[-1] * group_ratio[blk])
    token_list.append(token_list[-1] // (downsample[blk]**2))
    hid_unit_list.append(hid_unit_list[-1] * channel_ratio[blk])
    unit_list.append(unit_list[-1] * channel_ratio[blk])

  num_passes = FLAGS.num_passes
  conv = FLAGS.conv_mixer
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  ln_loss = lambda x, dy: jnp.sum(normalize(x) * dy)
  ln_grad = jax.grad(ln_loss)
  ln_loss1 = lambda x, dy: jnp.sum(normalize(x, swap=not conv) * dy)
  ln_grad1 = jax.grad(ln_loss1)
  conv_loss = lambda w, x, dy: jnp.sum(depthwise_conv(x, w) * dy)
  conv_grad = jax.grad(conv_loss)

  if FLAGS.fuse_cross_entropy:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, targets))
  else:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        classif_loss(spatial_avg_group_linear_custom_vjp(x, w, b), targets))
  local_classif_grad = jax.grad(local_classif_loss, argnums=[1, 2])
  global_classif_loss = lambda x, w, b, targets: jnp.mean(
      classif_loss(jnp.einsum('nc,cd->nd', x, w) + b, targets))
  global_classif_grad = jax.grad(global_classif_loss, argnums=[1, 2])
  wd = FLAGS.wd

  def local_loss(noise):
    _, local_loss_, (states, logs) = loss(params,
                                          batch,
                                          noise=noise,
                                          stop_gradient=FLAGS.modular_loss,
                                          readout=True,
                                          key=key,
                                          custom_forward=True)
    return local_loss_, (states, logs)

  for npass in range(num_passes):
    zeros = []
    noise = []
    B = batch['image'].shape[0]
    label = jax.nn.one_hot(batch['label'], md['num_classes'])
    M = num_passes
    NL = get_num_layers(NBLK)
    main_params = params[:NL]
    loss_params = params[NL:]
    noise, key = sample_activation_noise(params,
                                         key,
                                         B,
                                         num_items,
                                         num_hid_units,
                                         num_units,
                                         num_groups,
                                         NBLK,
                                         conv=conv,
                                         num_passes=1)
    zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), noise)

    # [L,B,P]
    _, g, (states, logs) = jax.jvp(local_loss, [zeros], [noise], has_aux=True)

    # Main network layers.
    for i, ((weight, bias), dz) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      pre_act = states[f'block_{blk}/pre_{layer}']
      prenorm_act = states[f'block_{blk}/prenorm_{layer}']
      post_act = states[f'block_{blk}/post_{layer}']
      mask = (post_act > 0.0).astype(jnp.float32)
      G_ = group_list[blk]
      P_ = token_list[blk]

      if FLAGS.modular_loss:
        g_ = g[blk]
      else:
        g_ = g

      # [B, D] -> [B, D, P] or [B, P] -> [B, P, D]
      if blk > 0 and layer == 0:
        # Token mixing layer
        if conv:
          g_ = jnp.sum(g_, axis=-1, keepdims=True)  # [2B, P, 1]
        else:
          g_ = jnp.sum(g_, axis=-1)[:, None, :]  # [2B, 1, P]
      else:
        # Channel mixing layer
        dz = jnp.reshape(dz, [B, P_, G_, -1])
        mask = jnp.reshape(mask, [B, P_, G_, -1])
        g_ = g_[:, :, :, None]

      grad_z = g_ * dz * mask

      # Backprop through normalization.
      if FLAGS.norm_grad and FLAGS.post_linear_ln:
        if blk > 0 and layer == 0:
          grad_z = jnp.reshape(
              ln_grad1(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)
        else:
          grad_z = jnp.reshape(
              ln_grad(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)

      if blk > 0 and layer == 0 and conv:
        # Token mixing conv
        H_ = int(math.sqrt(P_))
        grad_z = jnp.reshape(grad_z, [B, H_, H_, -1])
        grad_w = conv_grad(weight, pre_act, grad_z)
        grad_b = jnp.einsum('nhwd->d', grad_z)
      elif blk > 0 and layer == 0 and not conv:
        # Token mixing FC
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        grad_z = jnp.reshape(grad_z, [B, P_, -1])
        # Channel mixing FC
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      else:
        # Channel mixing group FC
        grad_w = jnp.einsum('npgc,npgd->gcd', pre_act, grad_z)
        grad_b = jnp.einsum('npgd->gd', grad_z)
      idx = i
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        # Last classification layer.
        pre_act = states[f'pre_final']
        grad_w, grad_b = global_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.last_layer_lr
        grad_b = grad_b * FLAGS.last_layer_lr
      else:
        # Intermediate classification layer.
        pre_act = states[f'block_{blk}/pre_pred']
        grad_w, grad_b = local_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def update_global_forward_grad_activations(params, batch, key):
  num_patches = FLAGS.num_patches
  num_units = FLAGS.num_channel_mlp_units
  num_hid_units = FLAGS.num_channel_mlp_hidden_units
  num_items = num_patches**2
  NBLK = FLAGS.num_blocks
  if num_hid_units < 0:
    num_hid_units = num_units
  num_groups = FLAGS.num_groups
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')
                ] if FLAGS.group_ratio is not None else [1] * FLAGS.num_blocks
  downsample = [int(d) for d in FLAGS.downsample.split(',')
               ] if FLAGS.downsample is not None else [1] * FLAGS.num_blocks
  channel_ratio = [
      int(c) for c in FLAGS.channel_ratio.split(',')
  ] if FLAGS.channel_ratio is not None else [1] * FLAGS.num_blocks

  group_list = [num_groups]
  token_list = [num_patches**2]
  hid_unit_list = [num_hid_units]
  unit_list = [num_units]
  for blk in range(NBLK - 1):
    group_list.append(group_list[-1] * group_ratio[blk])
    token_list.append(token_list[-1] // (downsample[blk]**2))
    hid_unit_list.append(hid_unit_list[-1] * channel_ratio[blk])
    unit_list.append(unit_list[-1] * channel_ratio[blk])

  num_passes = FLAGS.num_passes
  conv = FLAGS.conv_mixer
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  ln_loss = lambda x, dy: jnp.sum(normalize(x) * dy)
  ln_grad = jax.grad(ln_loss)
  ln_loss1 = lambda x, dy: jnp.sum(normalize(x, swap=not conv) * dy)
  ln_grad1 = jax.grad(ln_loss1)
  conv_loss = lambda w, x, dy: jnp.sum(depthwise_conv(x, w) * dy)
  conv_grad = jax.grad(conv_loss)

  if FLAGS.fuse_cross_entropy:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, targets))
  else:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        classif_loss(spatial_avg_group_linear_custom_vjp(x, w, b), targets))
  local_classif_grad = jax.grad(local_classif_loss, argnums=[1, 2])
  global_classif_loss = lambda x, w, b, targets: jnp.mean(
      classif_loss(jnp.einsum('nc,cd->nd', x, w) + b, targets))
  global_classif_grad = jax.grad(global_classif_loss, argnums=[1, 2])
  wd = FLAGS.wd

  def local_loss(noise):
    loss_, local_loss_, (states, logs) = loss(params,
                                              batch,
                                              noise=noise,
                                              stop_gradient=False,
                                              readout=False,
                                              key=key,
                                              custom_forward=True,
                                              avg_batch=False)
    return loss_, (states, logs)

  for npass in range(num_passes):
    zeros = []
    noise = []
    B = batch['image'].shape[0]
    label = jax.nn.one_hot(batch['label'], md['num_classes'])
    M = num_passes
    NL = get_num_layers(NBLK)
    main_params = params[:NL]
    loss_params = params[NL:]
    noise, key = sample_activation_noise(params,
                                         key,
                                         B,
                                         num_items,
                                         num_hid_units,
                                         num_units,
                                         num_groups,
                                         NBLK,
                                         conv=conv,
                                         num_passes=1)
    zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), noise)

    # [L,B,P]
    _, g, (states, logs) = jax.jvp(local_loss, [zeros], [noise], has_aux=True)

    # Main network layers.
    for i, ((weight, bias), dz) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      pre_act = states[f'block_{blk}/pre_{layer}']
      prenorm_act = states[f'block_{blk}/prenorm_{layer}']
      post_act = states[f'block_{blk}/post_{layer}']
      mask = (post_act > 0.0).astype(jnp.float32)
      G_ = group_list[blk]
      P_ = token_list[blk]
      g_ = g

      # [B, D] -> [B, D, P] or [B, P] -> [B, P, D]
      if blk > 0 and layer == 0:
        # Token mixing layer
        g_ = g_[:, None, None]  # [2B, 1, 1, 1]
      else:
        # Channel mixing layer
        dz = jnp.reshape(dz, [B, P_, G_, -1])
        mask = jnp.reshape(mask, [B, P_, G_, -1])
        g_ = g_[:, None, None, None]  # [2B, 1, 1, 1]

      grad_z = g_ * dz * mask

      # Backprop through normalization.
      if FLAGS.norm_grad and FLAGS.post_linear_ln:
        if blk > 0 and layer == 0:
          grad_z = jnp.reshape(
              ln_grad1(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)
        else:
          grad_z = jnp.reshape(
              ln_grad(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)

      if blk > 0 and layer == 0 and conv:
        # Token mixing conv
        H_ = int(math.sqrt(P_))
        grad_z = jnp.reshape(grad_z, [B, H_, H_, -1])
        grad_w = conv_grad(weight, pre_act, grad_z)
        grad_b = jnp.einsum('nhwd->d', grad_z)
      elif blk > 0 and layer == 0 and not conv:
        # Token mixing FC
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        grad_z = jnp.reshape(grad_z, [B, P_, -1])
        # Channel mixing FC
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      else:
        # Channel mixing group FC
        grad_w = jnp.einsum('npgc,npgd->gcd', pre_act, grad_z)
        grad_b = jnp.einsum('npgd->gd', grad_z)
      idx = i
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        # Last classification layer.
        pre_act = states[f'pre_final']
        grad_w, grad_b = global_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.last_layer_lr
        grad_b = grad_b * FLAGS.last_layer_lr
      else:
        # Intermediate classification layer.
        pre_act = states[f'block_{blk}/pre_pred']
        grad_w, grad_b = local_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs
  pass


def update_forward_grad_activations_vmap(params, batch, key):
  num_patches = FLAGS.num_patches
  num_units = FLAGS.num_channel_mlp_units
  num_hid_units = FLAGS.num_channel_mlp_hidden_units
  num_items = num_patches**2
  NBLK = FLAGS.num_blocks
  if num_hid_units < 0:
    num_hid_units = num_units
  num_groups = FLAGS.num_groups
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]
  channel_ratio = [int(c) for c in FLAGS.channel_ratio.split(',')]

  group_list = [num_groups]
  token_list = [num_patches**2]
  hid_unit_list = [num_hid_units]
  unit_list = [num_units]
  for blk in range(NBLK - 1):
    group_list.append(group_list[-1] * group_ratio[blk])
    token_list.append(token_list[-1] // (downsample[blk]**2))
    hid_unit_list.append(hid_unit_list[-1] * channel_ratio[blk])
    unit_list.append(unit_list[-1] * channel_ratio[blk])

  num_passes = FLAGS.num_passes
  conv = FLAGS.conv_mixer
  wd = FLAGS.wd
  B = batch['image'].shape[0]
  md = get_dataset_metadata(FLAGS.dataset)
  label = jax.nn.one_hot(batch['label'], md['num_classes'])
  NBLK = FLAGS.num_blocks
  G = num_groups
  P = num_patches**2
  M = num_passes
  NL = get_num_layers(NBLK)
  main_params = params[:NL]
  loss_params = params[NL:]
  ln_loss = lambda x, dy: jnp.sum(normalize(x) * dy)
  ln_grad = jax.grad(ln_loss)
  ln_loss1 = lambda x, dy: jnp.sum(normalize(x, swap=not conv) * dy)
  ln_grad1 = jax.grad(ln_loss1)
  conv_loss = lambda w, x, dy: jnp.sum(depthwise_conv(x, w) * dy)
  conv_grad = jax.grad(conv_loss)

  if FLAGS.fuse_cross_entropy:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, targets))
  else:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        classif_loss(spatial_avg_group_linear_custom_vjp(x, w, b), targets))
  local_classif_grad = jax.grad(local_classif_loss, argnums=[1, 2])
  global_classif_loss = lambda x, w, b, targets: jnp.mean(
      classif_loss(jnp.einsum('nc,cd->nd', x, w) + b, targets))
  global_classif_grad = jax.grad(global_classif_loss, argnums=[1, 2])

  def local_loss(noise):
    _, local_loss_, (states, logs) = loss(params,
                                          batch,
                                          noise=noise,
                                          stop_gradient=True,
                                          readout=True,
                                          key=key,
                                          custom_forward=True)
    return local_loss_, (states, logs)

  def fw_fun(noise):
    grads_now = [(None, None) for p in params]
    zeros = [jnp.zeros_like(n) for n in noise]
    # [L,B,P]
    _, g, (states, logs) = jax.jvp(local_loss, [zeros], [noise], has_aux=True)

    # Main network layers.
    for i, ((weight, bias), dz) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      pre_act = states[f'block_{blk}/pre_{layer}']
      prenorm_act = states[f'block_{blk}/prenorm_{layer}']
      post_act = states[f'block_{blk}/post_{layer}']
      mask = (post_act > 0.0).astype(jnp.float32)
      G_ = group_list[blk]
      P_ = token_list[blk]

      # [B, D] -> [B, D, P] or [B, P] -> [B, P, D]
      if blk > 0 and layer == 0:
        if conv:
          g_ = jnp.sum(g[blk], axis=-1, keepdims=True)  # [2B, P, 1]
        else:
          g_ = jnp.sum(g[blk], axis=-1)[:, None, :]  # [2B, 1, P]
      else:
        # Channel mixing layer
        dz = jnp.reshape(dz, [B, P_, G_, -1])
        mask = jnp.reshape(mask, [B, P_, G_, -1])
        g_ = g[blk][:, :, :, None]  # [B, P, G]

      grad_z = g_ * dz * mask
      if FLAGS.norm_grad and FLAGS.post_linear_ln:
        if blk > 0 and layer == 0:
          grad_z = jnp.reshape(
              ln_grad1(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)
        else:
          grad_z = jnp.reshape(
              ln_grad(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)

      if blk > 0 and layer == 0 and conv:
        H_ = int(math.sqrt(P_))
        grad_z = jnp.reshape(grad_z, [B, H_, H_, -1])
        grad_w = conv_grad(weight, pre_act, grad_z)
        grad_b = jnp.einsum('nhwd->d', grad_z)
      elif blk > 0 and layer == 0 and not conv:
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        grad_z = jnp.reshape(grad_z, [B, P_, -1])
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      elif len(grad_z.shape) == 4:
        grad_w = jnp.einsum('npgc,npgd->gcd', pre_act, grad_z)
        grad_b = jnp.einsum('npgd->gd', grad_z)
      idx = i
      grads_now[idx] = (grad_w, grad_b)

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        # Last classification layer.
        pre_act = states[f'pre_final']
        grad_w, grad_b = global_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.last_layer_lr
        grad_b = grad_b * FLAGS.last_layer_lr
      else:
        # Intermediate classification layer.
        pre_act = states[f'block_{blk}/pre_pred']
        grad_w, grad_b = local_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grad_w, grad_b)

    return grads_now, logs['loss']

  # noise, key = sample_noise(key)
  noise, key = sample_activation_noise(params,
                                       key,
                                       B,
                                       P,
                                       num_hid_units,
                                       num_units,
                                       num_groups,
                                       NBLK,
                                       conv=conv,
                                       num_passes=num_passes)
  # zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), noise)
  in_axes = ([0 for p in params],)
  out_axes = ([(0, 0) for p in params], 0)
  vfw_fun = jax.vmap(fw_fun, in_axes, out_axes)
  grads_now, allloss = vfw_fun(noise)
  grads_now = [(jnp.mean(grad_w, axis=0), jnp.mean(grad_b, axis=0))
               for grad_w, grad_b in grads_now]

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  allloss = allloss[-1]
  return grads_now, {'loss': allloss}


def update_perturb_activations(params, batch, key):
  num_patches = FLAGS.num_patches
  num_units = FLAGS.num_channel_mlp_units
  num_groups = FLAGS.num_groups
  num_passes = FLAGS.num_passes
  conv = FLAGS.conv_mixer
  epsilon = FLAGS.perturb_epsilon
  wd = FLAGS.wd
  md = get_dataset_metadata(FLAGS.dataset)
  grads_now = [(jnp.zeros_like(weight), jnp.zeros_like(bias))
               for (weight, bias) in params]
  ln_loss = lambda x, dy: jnp.sum(normalize(x) * dy)
  ln_grad = jax.grad(ln_loss)
  ln_loss1 = lambda x, dy: jnp.sum(normalize(x, swap=not conv) * dy)
  ln_grad1 = jax.grad(ln_loss1)

  if FLAGS.fuse_cross_entropy:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        spatial_avg_group_linear_cross_entropy_custom_vjp(x, w, b, targets))
  else:
    local_classif_loss = lambda x, w, b, targets: jnp.mean(
        classif_loss(spatial_avg_group_linear_custom_vjp(x, w, b), targets))
  local_classif_grad = jax.grad(local_classif_loss, argnums=[1, 2])
  global_classif_loss = lambda x, w, b, targets: jnp.mean(
      classif_loss(jnp.einsum('nc,cd->nd', x, w) + b, targets))
  global_classif_grad = jax.grad(global_classif_loss, argnums=[1, 2])

  def local_loss(noise):
    _, local_loss_, (states, logs) = loss(params,
                                          batch,
                                          noise=noise,
                                          stop_gradient=FLAGS.modular_loss,
                                          readout=True,
                                          key=key,
                                          custom_forward=True)
    return local_loss_, (states, logs)

  for npass in range(num_passes):
    zeros = []
    noise = []
    num_items = num_patches**2
    NBLK = FLAGS.num_blocks
    B = batch['image'].shape[0]
    label = jax.nn.one_hot(batch['label'], md['num_classes'])
    G = num_groups
    P = num_items
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
      dz = jax.random.normal(subkey, out_shape) * epsilon
      noise.append(dz)
      zeros.append(jnp.zeros_like(dz))

    # [L,B,P]
    # _, g, (states, logs) = jax.jvp(local_loss, [zeros], [noise], has_aux=True)
    local_loss1_, (states, logs) = local_loss(zeros)
    local_loss2_, (_, _) = local_loss(noise)

    # Main network layers.
    for i, ((weight, bias), dz) in enumerate(zip(main_params, noise)):
      blk, layer = get_blk(i)
      # Forward gradient.
      pre_act = states[f'block_{blk}/pre_{layer}']
      prenorm_act = states[f'block_{blk}/prenorm_{layer}']
      post_act = states[f'block_{blk}/post_{layer}']
      mask = (post_act > 0.0).astype(jnp.float32)
      # print(blk, layer, g[blk].shape, dz.shape, mask.shape)

      if FLAGS.modular_loss:
        g_ = (local_loss2_[blk] - local_loss1_[blk]) / (epsilon**2)
      else:
        g_ = (local_loss2_ - local_loss1_) / (epsilon**2)

      # [B, D] -> [B, D, P] or [B, P] -> [B, P, D]
      if blk > 0 and layer == 0:
        # Token mixing layer
        if conv:
          g_ = jnp.sum(g_, axis=-1, keepdims=True)  # [2B, P, 1]
        else:
          g_ = jnp.sum(g_, axis=-1)[:, None, :]  # [2B, 1, P]
      else:
        # Channel mixing layer
        dz = jnp.reshape(dz, [B, P, G, -1])
        mask = jnp.reshape(mask, [B, P, G, -1])
        g_ = g_[:, :, :, None]

      grad_z = g_ * dz * mask

      # Backprop through normalization.
      if FLAGS.norm_grad and FLAGS.post_linear_ln:
        if blk > 0 and layer == 0:
          grad_z = jnp.reshape(
              ln_grad1(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)
        else:
          grad_z = jnp.reshape(
              ln_grad(prenorm_act, jnp.reshape(grad_z, prenorm_act.shape)),
              dz.shape)

      if blk > 0 and layer == 0 and conv:
        H = int(math.sqrt(P))
        grad_z = jnp.reshape(grad_z, [B, H, H, -1])
        ksize = FLAGS.kernel_size
        psize = (ksize - 1) // 2
        x_pad = jnp.pad(pre_act, [(0, 0), (psize, psize), (psize, psize),
                                  (0, 0)],
                        mode="constant",
                        constant_values=0.0)
        grad_w = jax.lax.conv_general_dilated(
            jnp.transpose(x_pad, [3, 1, 2, 0]),
            jnp.transpose(grad_z, [1, 2, 0, 3]),
            window_strides=[1, 1],
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            batch_group_count=x_pad.shape[-1])
        grad_w = jnp.transpose(grad_w, [1, 2, 0, 3])
        grad_b = jnp.einsum('nhwd->d', grad_z)
      elif blk > 0 and layer == 0 and not conv:
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      elif (blk == 0 and layer == 0) or (blk > 0 and layer == 1):
        grad_z = jnp.reshape(grad_z, [B, P, -1])
        grad_w = jnp.einsum('npc,npd->cd', pre_act, grad_z)
        grad_b = jnp.einsum('npc->c', grad_z)
      else:
        grad_w = jnp.einsum('npgc,npgd->gcd', pre_act, grad_z)
        grad_b = jnp.einsum('npgd->gd', grad_z)
        grad_w = jnp.reshape(grad_w, weight.shape)
        grad_b = jnp.reshape(grad_b, bias.shape)
      idx = i
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

    # Classification layers.
    for i, (weight, bias) in enumerate(loss_params):
      blk = i  # Every block has a loss.
      if i == len(loss_params) - 1:
        # Last classification layer.
        pre_act = states[f'pre_final']
        grad_w, grad_b = global_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.last_layer_lr
        grad_b = grad_b * FLAGS.last_layer_lr
      else:
        # Intermediate classification layer.
        pre_act = states[f'block_{blk}/pre_pred']
        grad_w, grad_b = local_classif_grad(pre_act, weight, bias, label)
        grad_w = grad_w * FLAGS.head_lr
        grad_b = grad_b * FLAGS.head_lr
      idx = i + NL
      grads_now[idx] = (grads_now[idx][0] + grad_w / float(M),
                        grads_now[idx][1] + grad_b / float(M))

  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb) for (gw, gb) in grads_now[:-1]
                ] + [grads_now[-1]]

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [
        (gw + wd * w, gb) for (gw, gb), (w, b) in zip(grads_now, params)
    ]
  return grads_now, logs


def update_direct_feedback_alignment(params, batch, key):
  wd = FLAGS.wd

  def loss_fn(params, batch):
    final_loss_, local_loss_, (states, logs) = loss_dfa(params,
                                                        batch,
                                                        noise=None,
                                                        key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states, logs)

  grads_now, (states, logs) = grad(loss_fn, has_aux=True)(params, batch)
  # Clear gradient for backward weights.
  grads_now = [
      (gw, gb, 0.0 * gwb, 0.0 * gbb) for (gw, gb, gwb, gbb) in grads_now
  ]
  if FLAGS.freeze_backbone:
    grads_now = [(0.0 * gw, 0.0 * gb, 0.0 * gwb, 0.0 * gbb)
                 for (gw, gb, gwb, gbb) in grads_now[:-1]] + [grads_now[-1]]

  if FLAGS.last_layer_lr < 1.0 and FLAGS.bp_last_lr:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr,
                     grads_now[-1][2] * FLAGS.last_layer_lr,
                     grads_now[-1][3] * FLAGS.last_layer_lr)

  if FLAGS.optimizer == 'sgd' and wd > 0.0:
    grads_now = [(gw + wd * w, gb, gwb, gbb)
                 for (gw, gb, gwb, gbb), (w, b, wb,
                                          bb) in zip(grads_now, params)]
  return grads_now, logs


def update_feedback_alignment(params, batch, key):
  wd = FLAGS.wd

  def loss_fn(params, batch):
    final_loss_, local_loss_, (states, logs) = loss(params,
                                                    batch,
                                                    noise=None,
                                                    stop_gradient=False,
                                                    readout=True,
                                                    key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states, logs)

  grads_now, (states, logs) = grad(loss_fn, has_aux=True)(params, batch)

  # Swap backward gradient for representation layers.
  grads_now = [(gwb, gb, 0.0 * gw) for (gw, gb, gwb) in grads_now]

  if FLAGS.freeze_backbone:
    grads_now = [
        (0.0 * gw, 0.0 * gb, 0.0 * gwb) for (gw, gb, gwb) in grads_now[:-1]
    ] + [grads_now[-1]]

  if FLAGS.last_layer_lr < 1.0 and FLAGS.bp_last_lr:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr,
                     grads_now[-1][2] * FLAGS.last_layer_lr)

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
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]

  def local_loss(params, batch):
    final_loss_, local_loss_, (states, logs) = loss(params,
                                                    batch,
                                                    noise=None,
                                                    stop_gradient=False,
                                                    readout=True,
                                                    key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states, logs)

  NL = get_num_layers(NBLK)
  grads_now, (states, logs) = grad(local_loss, has_aux=True)(params, batch)

  # Swap backward gradient for representation layers.
  grads_now = [(gwb, gb, 0.0 * gw) for (gw, gb, gwb) in grads_now]

  if FLAGS.last_layer_lr < 1.0:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr,
                     grads_now[-1][2] * FLAGS.last_layer_lr)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr,
                      grads_now[i][2] * FLAGS.head_lr)

  if FLAGS.same_head:
    G_ = G
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G_), grads_now[i][1] / float(G_),
                      grads_now[i][2] / float(G_))
      G_ = G_ * group_ratio[i - NL]

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    P_ = P
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P_), grads_now[i][1] / float(P_),
                      grads_now[i][2] / float(P_))
      P_ = P_ // (downsample[i - NL]**2)

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
  group_ratio = [int(g) for g in FLAGS.group_ratio.split(',')]
  downsample = [int(d) for d in FLAGS.downsample.split(',')]

  def local_loss(params, batch):
    final_loss_, local_loss_, (states, logs) = loss(params,
                                                    batch,
                                                    noise=None,
                                                    stop_gradient=True,
                                                    readout=True,
                                                    key=key)
    final_loss_ = final_loss_ + jnp.sum(
        jnp.stack([jnp.sum(l) for l in local_loss_]))
    return final_loss_, (states, logs)

  NL = get_num_layers(NBLK)
  grads_now, (states, logs) = grad(local_loss, has_aux=True)(params, batch)

  # Swap backward gradient for representation layers.
  grads_now = [(gwb, gb, 0.0 * gw) for (gw, gb, gwb) in grads_now]

  if FLAGS.last_layer_lr < 1.0:
    grads_now[-1] = (grads_now[-1][0] * FLAGS.last_layer_lr,
                     grads_now[-1][1] * FLAGS.last_layer_lr,
                     grads_now[-1][2] * FLAGS.last_layer_lr)

  if FLAGS.head_lr != 1.0:
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] * FLAGS.head_lr,
                      grads_now[i][1] * FLAGS.head_lr,
                      grads_now[i][2] * FLAGS.head_lr)

  if FLAGS.same_head:
    G_ = G
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(G_), grads_now[i][1] / float(G_),
                      grads_now[i][2] / float(G_))
      G_ = G_ * group_ratio[i - NL]

  # Last layer divide by P, since it is after average pooling.
  if avgpool_token:
    P_ = P
    for i in range(NL, len(grads_now) - 1):
      grads_now[i] = (grads_now[i][0] / float(P_), grads_now[i][1] / float(P_),
                      grads_now[i][2] / float(P_))
      P_ = P_ // (downsample[i - NL]**2)

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


def get_dataset_cifar10(split, seed=0):
  batch_size = FLAGS.batch_size
  data_root = FLAGS.data_root
  ds = tfds.load('cifar10',
                 split=_to_tfds_split(split),
                 data_dir=data_root,
                 shuffle_files=True)
  ds = ds.repeat()
  ds = ds.shuffle(buffer_size=10 * batch_size, seed=seed)
  is_parallel = jax.device_count() > 1
  num_parallel = jax.local_device_count()

  def preprocess(example):
    image = tf.image.convert_image_dtype(example['image'], tf.float32)
    if split == "train":
      if FLAGS.aug:
        image = tf.pad(image, [(4, 4), (4, 4), (0, 0)])
        image = tf.image.random_crop(image, (32, 32, 3))
      image = tf.image.random_flip_left_right(image)
    elif split == "train_noaug":
      image = tf.image.random_flip_left_right(image)
    label = tf.cast(example['label'], tf.int32)
    return {'image': image, 'label': label}

  ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)

  def rebatch(batch):
    if is_parallel:
      for k in batch:
        batch[k] = tf.reshape(batch[k],
                              [num_parallel, batch_size // num_parallel] +
                              list(batch[k].shape[1:]))
    return batch

  ds = ds.map(rebatch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  yield from tfds.as_numpy(ds)


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


def get_dataset_imagenet100(split, imagenet2012=False, seed=0):
  batch_size = FLAGS.batch_size
  data_dir = FLAGS.data_root
  if imagenet2012:
    splits = tfds.even_splits(_to_imagenet2012_split(split),
                              n=jax.process_count(),
                              drop_remainder=True)
    process_split = splits[jax.process_index()]
    ds = tfds.load('imagenet2012',
                   split=process_split,
                   data_dir=FLAGS.gcs_path if FLAGS.use_gcs else data_dir,
                   shuffle_files=True,
                   decoders={'image': tfds.decode.SkipDecoding()},
                   try_gcs=FLAGS.use_gcs)
  else:
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
          data_dir=data_dir,
          shuffle_files=True,
          decoders={'image': {
              'encoded': tfds.decode.SkipDecoding()
          }})
      # builder = tfds.ImageFolder(
      #     os.path.join(data_dir, 'downloads', 'imagenet-100'))
      # ds = builder.as_dataset(split=_to_imagenet100_split(split),
      #                         shuffle_files=True,
      #                         decoders={'image': tfds.decode.SkipDecoding()})
  ds = ds.repeat()
  ds = ds.shuffle(buffer_size=10 * batch_size,
                  seed=jax.process_index() * 1234 + seed)
  is_parallel = jax.device_count() > 1
  num_parallel = jax.local_device_count()
  if imagenet2012:
    md = get_dataset_metadata("imagenet2012")
  else:
    md = get_dataset_metadata("imagenet-100")

  def preprocess(example):
    image = example['image']
    if type(image) is dict:
      image = image['encoded']
    if split == "train":
      if FLAGS.aug:
        image = _decode_and_random_crop(image, area_lb=FLAGS.area_lb)
      else:
        image = _decode_and_center_crop(image, md['input_height'])
      image = tf.image.random_flip_left_right(image)
      assert image.dtype == tf.uint8
    else:
      image = _decode_and_center_crop(image, md['input_height'])
    image = tf.image.resize(image, [md['input_height'], md['input_width']],
                            tf.image.ResizeMethod.BICUBIC)
    image = tf.clip_by_value(image / 255., 0., 1.)
    if 'label' in example:
      label = tf.cast(example['label'], tf.int32)
    else:
      label = tf.cast(example['image']['class']['label'], tf.int32)
    return {'image': image, 'label': label}

  ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)

  def rebatch(batch):
    if is_parallel:
      for k in batch:
        batch[k] = tf.reshape(batch[k],
                              [num_parallel, batch_size // num_parallel] +
                              list(batch[k].shape[1:]))
    return batch

  ds = ds.map(rebatch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  yield from tfds.as_numpy(ds)


def get_dataset(split, seed=0):
  if FLAGS.dataset in ['cifar-10']:
    return get_dataset_cifar10(split, seed=seed)
  elif FLAGS.dataset in ['imagenet-100']:
    return get_dataset_imagenet100(split, seed=seed)
  elif FLAGS.dataset in ['imagenet2012']:
    return get_dataset_imagenet100(split, imagenet2012=True, seed=seed)
  else:
    raise ValueError('Dataset not found {}'.format(FLAGS.dataset))


def run_exp(mode, lr, train_loader, train_eval_loader, test_loader,
            param_scale, layer_sizes, num_epochs, log_dir):
  print('layer sizes', layer_sizes)
  if mode == "backprop":
    update_fn = update_backprop
  elif mode == "perturb_weights":
    update_fn = update_perturb_weights
  elif mode == "forward_grad_weights":
    update_fn = update_forward_grad_weights
  elif mode == "global_forward_grad_weights":
    update_fn = update_global_forward_grad_weights
  elif mode == "perturb_activations":
    update_fn = update_perturb_activations
  elif mode == "forward_grad_activations":
    update_fn = update_forward_grad_activations
  elif mode == "global_forward_grad_activations":
    update_fn = update_global_forward_grad_activations
  elif mode == "forward_grad_activations_vmap":
    update_fn = update_forward_grad_activations_vmap
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
  num_proc = jax.process_count()
  num_batches = md['num_examples_train'] // num_proc // FLAGS.batch_size
  num_batches_test = md['num_examples_test'] // num_proc // FLAGS.batch_size
  num_batches_eval = num_batches_test

  if is_parallel:
    params = jax.pmap(lambda i: params,
                      axis_name='i')(jnp.zeros([num_parallel],
                                               dtype=jnp.int32))

  schedule = FLAGS.schedule

  if schedule == "linear":
    scheduler = optax.linear_schedule(init_value=lr,
                                      end_value=0.0,
                                      transition_steps=num_epochs *
                                      num_batches,
                                      transition_begin=0)
  elif schedule == "staircase":
    scheduler = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales={
            int(num_epochs * 0.5 * num_batches): 0.1,
            int(num_epochs * 0.8 * num_batches): 0.1
        })
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
  elif optimizer_name in ['adam'] and FLAGS.wd == 0.0:
    optimizer = optax.adam(scheduler)
  elif optimizer_name in ['adam', 'adamw'] and FLAGS.wd > 0.0:
    optimizer = optax.adamw(scheduler, weight_decay=FLAGS.wd)
  elif optimizer_name in ['lars']:
    optimizer = optax.lars(scheduler, FLAGS.wd, momentum=FLAGS.mom)
    print(optimizer)
  else:
    raise ValueError(f'Unknown optimizer {optimizer_name}')

  loss_curve = []
  train_acc_curve = []
  test_acc_curve = []

  if jax.process_index() == 0:
    writer = metric_writers.create_default_writer(log_dir, asynchronous=True)
  else:
    writer = metric_writers.MultiWriter([])  # writing is a no-op

  def step_fn(batch, params, opt_state, key):
    key, subkey = jax.random.split(key)
    grads, logs = update_fn(params, batch, subkey)
    if is_parallel:
      grads = jax.tree_util.tree_map(lambda p: jax.lax.pmean(p, axis_name='i'),
                                     grads)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, logs, key

  if is_parallel:
    step_fn = jax.pmap(
        step_fn,
        axis_name='i',
    )
    accuracy_fn = jax.pmap(lambda params, batch: jax.lax.pmean(
        accuracy(params, batch), axis_name='i'),
                           axis_name='i')
    if epoch_start == 0:
      opt_state = jax.pmap(optimizer.init, axis_name='i')(params)
      key = jax.pmap(lambda i: jax.random.PRNGKey(i),
                     axis_name='i')(jnp.arange(num_parallel))
    else:
      opt_state = jax.pmap(lambda i: opt_state,
                           axis_name='i')(jnp.zeros([num_parallel],
                                                    dtype=jnp.int32))
      key = jax.pmap(lambda i: i, axis_name='i')(key)
  else:
    step_fn = jax.jit(step_fn)
    accuracy_fn = jax.jit(accuracy)
    if epoch_start == 0:
      opt_state = optimizer.init(params)
      key = jax.random.PRNGKey(0)

  step = epoch_start * num_batches
  lr_ = scheduler(epoch_start * num_batches)
  for epoch in range(epoch_start, num_epochs):
    total_loss = 0.0
    train_acc = 0.0
    start_time = time.time()
    for _ in range(num_batches):
      batch = next(train_loader)
      params, opt_state, logs, key = step_fn(batch, params, opt_state, key)
      if is_parallel:
        total_loss += logs['loss'][0] / float(num_batches)
        train_acc += jnp.mean(logs['acc/train']) / float(num_batches)
      else:
        total_loss += logs['loss'] / float(num_batches)
        train_acc += logs['acc/train'] / float(num_batches)
      del logs['acc/train']
      loss_curve.append(logs['loss'])
      if step % 10 == 0:
        writer.write_scalars(
            step, {k: v[0] if is_parallel else v for k, v in logs.items()})
      step += 1
    if jnp.isnan(total_loss):
      print('total loss exploded', total_loss)
      break
    epoch_time = time.time() - start_time

    # Evaluate
    if FLAGS.train_eval:
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
    train_acc_curve.append(train_acc)
    test_acc_curve.append(test_acc)
    if jax.process_index() == 0:
      msg = "Mode {} LR {:.2e} Epoch {} Time {:.2f}s Train Loss {:.2f} Train Acc {:.2f}% Test Acc {:.2f}%".format(
          mode, lr_, epoch, epoch_time, total_loss, train_acc * 100.0,
          test_acc * 100.0)
      print(msg)

      # Save checkpoint.
      if is_parallel:
        params_save = jax.tree_util.tree_map(lambda p: p[0], params)
        opt_state_save = jax.tree_util.tree_map(lambda p: p[0], opt_state)
      else:
        params_save = params
        opt_state_save = opt_state
      ckpt = dict(epoch=epoch,
                  params=params_save,
                  opt_state=opt_state_save,
                  key=key)
      save_checkpoint(os.path.join(log_dir, "ckpt"), epoch, ckpt)

  return {
      'loss': np.array(jnp.array(loss_curve)),
      'train_acc': np.array(jnp.array(train_acc_curve)),
      'test_acc': np.array(jnp.array(test_acc_curve))
  }


def main(_):
  md = get_dataset_metadata(FLAGS.dataset)
  downsample = [int(d) for d in FLAGS.downsample.split(',')]
  channel_ratio = [int(d) for d in FLAGS.channel_ratio.split(',')]
  group_ratio = [int(d) for d in FLAGS.group_ratio.split(',')]
  layer_sizes = get_layer_sizes(
      md,
      FLAGS.num_patches,
      FLAGS.num_channel_mlp_units,
      FLAGS.num_blocks,
      FLAGS.num_groups,
      FLAGS.concat_groups,
      FLAGS.same_head,
      FLAGS.conv_mixer,
      FLAGS.kernel_size,
      num_channel_mlp_hidden_units=FLAGS.num_channel_mlp_hidden_units,
      downsample=downsample,
      channel_ratio=channel_ratio,
      group_ratio=group_ratio)
  param_scale = get_param_scale(FLAGS.init_scheme, layer_sizes)
  print("param scale", param_scale)
  num_epochs = FLAGS.num_epochs
  experiment = FLAGS.exp
  train_loader = get_dataset("train", seed=0)
  if FLAGS.train_eval:
    train_eval_loader = get_dataset("train_eval", seed=1)
  else:
    train_eval_loader = None
  test_loader = get_dataset("test", seed=0)

  fname = 'cifar_mixer_t1c2_fg'
  exp_dir = FLAGS.workdir
  fname_full = f'{exp_dir}/results.pkl'

  if experiment == "all":
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
        lr_dir = '{}/{:.2e}'.format(log_dir, lr)
        if not tf.io.gfile.exists(lr_dir):
          tf.io.gfile.makedirs(lr_dir)
        results = run_exp(mode, lr, train_loader, train_eval_loader,
                          test_loader, param_scale, layer_sizes, num_epochs,
                          lr_dir)
        all_results[mode][lr] = results
    if jax.process_index() == 0:
      pkl.dump(all_results, tf.io.gfile.GFile(fname_full, 'wb'))
  else:
    exp_list = experiment.split(',')
    lr_list = [float(s) for s in FLAGS.lr.split(',')]
    for exp in exp_list:
      log_dir = f'{exp_dir}/{exp}'
      if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
      for lr in lr_list:
        print(lr)
        lr_dir = '{}/{:.2e}'.format(log_dir, lr)
        if not tf.io.gfile.exists(lr_dir):
          tf.io.gfile.makedirs(lr_dir)
        results = run_exp(exp, lr, train_loader, train_eval_loader,
                          test_loader, param_scale, layer_sizes, num_epochs,
                          lr_dir)
        if tf.io.gfile.exists(fname_full):
          all_results = pkl.load(tf.io.gfile.GFile(fname_full, 'rb'))
        else:
          all_results = {}
        if exp not in all_results:
          all_results[exp] = {}
        all_results[exp][lr] = results
        if jax.process_index() == 0:
          pkl.dump(all_results, tf.io.gfile.GFile(fname_full, 'wb'))


if __name__ == '__main__':
  app.run(main)
