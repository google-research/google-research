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

"""CIFAR-10 example.

This script trains a ResNet-50 on the CIFAR-10 dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import os
import pickle
import time

from . import util
from .architectures import model_pyramidnet
from .architectures import model_resnet
from .architectures import model_wrn
from .architectures import model_wrn_shakeshake

from .data_sources import imagenet_data_source
from .data_sources import small_image_data_source

from flax import jax_utils
from flax import optim
import flax.deprecated.nn
from flax.metrics import tensorboard
import jax

import jax.nn
import jax.numpy as jnp

from .masking import regularizers

import numpy as onp

import tensorflow.compat.v2 as tf


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def create_model(key, arch, batch_size, image_size, n_classes):
  """Create model with the specified architecture.

  Architectures:
  wrn20_10: 20 layer wide ResNet 10x width, for 32x32 images
  wrn26_10: 26 layer wide ResNet 10x width, for 32x32 images
  wrn26_2: 26 layer wide ResNet 2x width, for 32x32 images
  wrn20_6_shakeshake: 20 layer wide ResNet 96x2d, shake-shake, for 32x32 images
  wrn26_6_shakeshake: 26 layer wide ResNet 96x2d, shake-shake, for 32x32 images
  wrn26_2_shakeshake: 26 layer wide ResNet 32x2d, shake-shake, for 32x32 images
  pyramid: 272 layer PyramidNet with ShakeDrop, for 32x32 images
  resnet50: 50 layer ResNet, for ImageNet
  resnet101: 101 layer ResNet, for ImageNet
  resnet152: 152 layer ResNet, for ImageNet
  resnet50x2: 50 layer ResNet, 2x width, for ImageNet
  resnet101x2: 101 layer ResNet, 2x width, for ImageNet
  resnet152x2: 152 layer ResNet, 2x width, for ImageNet
  resnext50_32x4d: 50 layer ResNeXt, 32x4d, for ImageNet
  resnext101_32x4d: 101 layer ResNeXt, 32x4d, for ImageNet
  resnext152_32x4d: 152 layer ResNeXt, 32x4d, for ImageNet

  Args:
    key: PRNG key for initialization
    arch: architecture as a string that identifies the network to be built
    batch_size: training batch size
    image_size: images will be image_size x image_size pixels
    n_classes: number of classes to predict

  Returns:
    (model, init_state): tuple of model built and state (batch stats)
  """
  input_shape = (batch_size, image_size, image_size, 3)
  if arch == 'wrn20_10':
    model_def = model_wrn.WideResnet.partial(
        blocks_per_group=3, channel_multiplier=10, num_outputs=n_classes,
        dropout_rate=0.3)
  elif arch == 'wrn26_10':
    model_def = model_wrn.WideResnet.partial(
        blocks_per_group=4, channel_multiplier=10, num_outputs=n_classes,
        dropout_rate=0.3)
  elif arch == 'wrn26_2':
    model_def = model_wrn.WideResnet.partial(
        blocks_per_group=4, channel_multiplier=2, num_outputs=n_classes,
        dropout_rate=0.3)
  elif arch == 'wrn20_6_shakeshake':
    model_def = model_wrn_shakeshake.WideResnetShakeShake.partial(
        blocks_per_group=3, channel_multiplier=6, num_outputs=n_classes)
  elif arch == 'wrn26_6_shakeshake':
    model_def = model_wrn_shakeshake.WideResnetShakeShake.partial(
        blocks_per_group=4, channel_multiplier=6, num_outputs=n_classes)
  elif arch == 'wrn26_2_shakeshake':
    model_def = model_wrn_shakeshake.WideResnetShakeShake.partial(
        blocks_per_group=4, channel_multiplier=2, num_outputs=n_classes)
  elif arch == 'pyramid':
    model_def = model_pyramidnet.PyramidNetShakeDrop.partial(
        num_outputs=n_classes)
  elif arch == 'resnet50':
    model_def = model_resnet.ResNet50.partial(num_outputs=n_classes)
  elif arch == 'resnet101':
    model_def = model_resnet.ResNet101.partial(num_outputs=n_classes)
  elif arch == 'resnet152':
    model_def = model_resnet.ResNet152.partial(num_outputs=n_classes)
  elif arch == 'resnet50x2':
    model_def = model_resnet.ResNet50x2.partial(num_outputs=n_classes)
  elif arch == 'resnet101x2':
    model_def = model_resnet.ResNet101x2.partial(num_outputs=n_classes)
  elif arch == 'resnet152x2':
    model_def = model_resnet.ResNet152x2.partial(num_outputs=n_classes)
  elif arch == 'resnext50_32x4d':
    model_def = model_resnet.ResNext50_32x4d.partial(num_outputs=n_classes)
  elif arch == 'resnext101_32x8d':
    model_def = model_resnet.ResNext101_32x8d.partial(num_outputs=n_classes)
  elif arch == 'resnext152_32x4d':
    model_def = model_resnet.ResNext152_32x4d.partial(num_outputs=n_classes)
  else:
    raise ValueError('Unknown architecture \'{}\''.format(arch))
  with flax.deprecated.nn.stateful() as init_state:
    with flax.deprecated.nn.stochastic(jax.random.PRNGKey(0)):
      _, model = model_def.create_by_shape(
          key, [(input_shape, jnp.float32)])
  return model, init_state


def cross_entropy_loss(logits, labels):
  """Compute cross entropy loss.

  Args:
    logits: logits as (batch_size, n_classes) array
    labels: labels as (batch_size) integer array

  Returns:
    mean cross entropy loss
  """
  log_softmax_logits = jax.nn.log_softmax(logits)
  loss_sum = -jnp.sum(util.onehot(labels, logits.shape[1]) * log_softmax_logits)
  return loss_sum / labels.size


def compute_train_metrics(logits, labels):
  """Compute training metrics (loss and error rate).

  Args:
    logits: logits as (batch_size, n_classes) array
    labels: labels as (batch_size) integer array

  Returns:
    metrics as a dict
  """
  loss = cross_entropy_loss(logits, labels)
  error_rate = jnp.mean(jnp.argmax(logits, -1) != labels)
  metrics = {
      'loss': loss,
      'error_rate': error_rate,
  }
  metrics = util.pmean(metrics)
  return metrics


def compute_eval_metrics(logits, labels, eval_top_5):
  """Compute evaluation metrics.

  Eval metrics consists of loss sum, error count and sample count,
  and top-5 error count if eval_top_5 is True.

  Args:
    logits: logits as (batch_size, n_classes) array
    labels: labels as (batch_size) integer array
    eval_top_5: if True, compute top 5 error count

  Returns:
    metrics as a dict
  """
  mask = (labels != -1).astype(jnp.float32)

  # `onehot` will generate all zeros for samples that are labelled -1
  # So no need to multiply the per-sample loss by the mask
  log_softmax_logits = jax.nn.log_softmax(logits)
  ce_loss = -jnp.sum(util.onehot(labels, logits.shape[1]) * log_softmax_logits)

  error_rate = ((jnp.argmax(logits, -1) != labels) * mask).sum(-1)
  metrics = {
      'loss_sum': ce_loss,
      'error_count': error_rate,
      'sample_count': mask.sum(-1),
  }

  if eval_top_5:
    top5_pred = jnp.argsort(logits, axis=-1)[..., -5:]
    top5_hits = (top5_pred == labels[..., None]).any(axis=-1)
    top5_errs = (~top5_hits).astype(jnp.float32) * mask
    metrics['top5_error_count'] = top5_errs.sum(-1)

  metrics = util.psum(metrics)
  return metrics


def avg_eval_metrics(metrics):
  """Average evaluation metrics (divide values by sample count).

  Args:
    metrics: evaluation metrics

  Returns:
    averaged metrics as a dict
  """
  n = metrics['sample_count']
  metrics['loss'] = metrics['loss_sum'] / n
  metrics['error_rate'] = metrics['error_count'] / n
  if 'top5_error_count' in metrics:
    metrics['top5_error_rate'] = metrics['top5_error_count'] / n
  return metrics


def piecewise_constant(boundaries, values, t):
  """Piecewise constant.

  Helper function for stepped learning rate.

  Args:
    boundaries: boundaries at which value changes
    values: values start at corresponding boundary and finish at next
      boundary
    t: value to sample at

  Returns:
    value
  """
  index = jnp.sum(boundaries < t)
  return jnp.take(values, index)


def create_constant_learning_rate_fn(base_learning_rate):
  """Create a constant learning rate function.

  Args:
    base_learning_rate: learning rate that will always be returned

  Returns:
    function of the form f(step) -> learning_rate
  """
  def step_fn(step):  ## pylint: disable=unused-argument
    return base_learning_rate
  return step_fn


def create_stepped_learning_rate_fn(base_learning_rate, steps_per_epoch,
                                    lr_sched_steps, warmup_length=0.0):
  """Create a stepped learning rate function.

  Args:
    base_learning_rate: base learning rate
    steps_per_epoch: number of steps per epoch
    lr_sched_steps: learning rate schedule as a list of pairs where each
      pair is `[step, lr_factor]`
    warmup_length: linear LR warmup length; 0 for no warmup

  Returns:
    function of the form f(step) -> learning_rate
  """
  boundaries = [step[0] for step in lr_sched_steps]
  decays = [step[1] for step in lr_sched_steps]
  boundaries = onp.array(boundaries) * steps_per_epoch
  boundaries = onp.round(boundaries).astype(int)
  values = onp.array([1.0] + decays) * base_learning_rate

  def step_fn(step):
    lr = piecewise_constant(boundaries, values, step)
    if warmup_length > 0.0:
      lr = lr * jnp.minimum(1., step / float(warmup_length) / steps_per_epoch)
    return lr
  return step_fn


def create_cosine_learning_rate_fn(base_learning_rate, steps_per_epoch,
                                   halfcoslength_epochs, warmup_length=0.0):
  """Create a cosline annealing learning rate function.

  Args:
    base_learning_rate: base learning rate
    steps_per_epoch: number of steps per epoch
    halfcoslength_epochs: number of epochs over which a half cosine wave
      will transition from 1 to 0
    warmup_length: linear LR warmup length; 0 for no warmup

  Returns:
    function of the form f(step) -> learning_rate
  """
  halfwavelength_steps = halfcoslength_epochs * steps_per_epoch

  def step_fn(step):
    f = jnp.cos(step * jnp.pi / halfwavelength_steps) * 0.5 + 0.5
    lr = base_learning_rate * f
    if warmup_length > 0.0:
      lr = lr * jnp.minimum(1., step / float(warmup_length) / steps_per_epoch)
    return lr
  return step_fn


def confidence_thresholding(confidence, conf_thresh, conf_avg):
  """Confidence thresholding helper function.

  Args:
    confidence: per-sample confidence as a (batch_size,) array
    conf_thresh: confidence threshold
    conf_avg: if True, return the mean of the confidence masks

  Returns:
    (mask, conf_rate) tuple of the masks to use and the proportion of samples
      that pass the threshold
  """
  if conf_thresh > 0.0:
    conf_mask = (confidence > conf_thresh).astype(jnp.float32)
  else:
    conf_mask = jnp.ones_like(confidence)
  conf_mask_rate = conf_mask.mean()
  if conf_avg:
    unsup_loss_mask = conf_mask_rate
  else:
    unsup_loss_mask = conf_mask
  return unsup_loss_mask, conf_mask_rate


def train_step(optimizer_stu, state_stu, model_tea, state_tea,
               batch, learning_rate_fn,
               l2_reg, weight_decay,
               teacher_alpha_fn,
               unsup_reg=None, cons_weight=0.0,
               conf_thresh=0.97, conf_avg=False,
               mix_reg=None, mix_aug_separately=False, mix_logits=True,
               mix_weight=0.0, mix_conf_thresh=0.97, mix_conf_avg=True,
               mix_conf_mode='mix_prob'):
  """Perform a single training step."""

  state_tea = util.pmean(state_tea)

  def loss_fn(model_stu):
    """loss function used for training."""

    # Get data from batch
    sup_x = batch['sup_image']
    rng_key = batch['rng_key']
    (unsup_reg_stu_rng, model_rng_sup, model_rng_unsup_stu,
     model_rng_unsup_tea, mix_rng, model_rng_mix_stu,
     model_rng_mix_tea) = jax.random.split(rng_key, num=7)

    with flax.deprecated.nn.stateful(state_stu) as new_state_stu:
      with flax.deprecated.nn.stochastic(model_rng_sup):
        sup_logits = model_stu(sup_x, train=True)
    sup_loss = cross_entropy_loss(sup_logits, batch['sup_label'])
    loss = sup_loss

    new_state_tea = state_tea

    if unsup_reg is not None:
      #
      # UNSUPERVISED PATH
      #

      unsup_x0 = batch['unsup_image0']
      unsup_x1 = batch['unsup_image1']

      # Apply unsupervised reg
      unsup_x1 = unsup_reg.perturb_sample(unsup_x1, unsup_reg_stu_rng)

      with flax.deprecated.nn.stateful(new_state_tea):
        with flax.deprecated.nn.stochastic(model_rng_unsup_tea):
          unsup_logits_tea = model_tea(unsup_x0, train=False)
      unsup_logits_tea = jax.lax.stop_gradient(unsup_logits_tea)
      with flax.deprecated.nn.stateful(new_state_stu) as new_state_stu:
        with flax.deprecated.nn.stochastic(model_rng_unsup_stu):
          unsup_logits_stu = model_stu(unsup_x1, train=True)

      # Logits -> probs
      unsup_prob_tea = jax.nn.softmax(unsup_logits_tea)
      unsup_prob_stu = jax.nn.softmax(unsup_logits_stu)

      # Confidence thresholding
      unsup_loss_mask, conf_mask_rate = confidence_thresholding(
          unsup_prob_tea.max(axis=-1, keepdims=True), conf_thresh, conf_avg
      )

      # Unsupervised loss
      unsup_var_loss = ((unsup_prob_stu - unsup_prob_tea)**2) * unsup_loss_mask
      cons_loss = unsup_var_loss.sum(axis=-1).mean()
      loss = loss + cons_loss * cons_weight
    else:
      cons_loss = jnp.array(0.0, dtype=jnp.float32)
      conf_mask_rate = jnp.array(0.0, dtype=jnp.float32)
      unsup_logits_tea = None
      unsup_prob_tea = None

    if mix_reg is not None:
      #
      # MIX PATH
      #

      if mix_aug_separately:
        x0_mix_tea = batch['unsup_image1']
      else:
        x0_mix_tea = batch['unsup_image0']

      x0_mix = batch['unsup_image0']
      x1_mix = jnp.concatenate([x0_mix[1:, ...], x0_mix[:1, ...]], axis=0)
      x_mix, mix_blend_facs = mix_reg.mix_images(x0_mix, x1_mix, mix_rng)

      if unsup_reg is not None:
        # We can use the logits generated for unsupervised samples using the
        # teacher model because:
        # - the unsupervised loss path as used
        # - the teacher inputs wer *not* perturbed/masked
        # The unsupervised loss was computed so the unsupervised logits from
        # the teacher model are available
        # Furthermore, the teacher input was *not* perturbed
        logits0_mix_tea = unsup_logits_tea
      else:
        # unsup_logits_tea is the result of applying the teacher model to a
        # perturbed sample, so re-compute
        with flax.deprecated.nn.stateful(new_state_tea):
          with flax.deprecated.nn.stochastic(model_rng_mix_tea):
            logits0_mix_tea = model_tea(x0_mix_tea, train=False)
        logits0_mix_tea = jax.lax.stop_gradient(logits0_mix_tea)

      # Logits -> probs, using temperature
      prob0_mix_tea = jax.nn.softmax(logits0_mix_tea)
      prob0_mix_tea_conf = jax.nn.softmax(logits0_mix_tea)

      logits1_mix_tea = jnp.concatenate(
          [logits0_mix_tea[1:, ...], logits0_mix_tea[:1, ...]], axis=0)
      prob1_mix_tea = jnp.concatenate(
          [prob0_mix_tea[1:, ...], prob0_mix_tea[:1, ...]], axis=0)
      prob1_mix_tea_conf = jnp.concatenate(
          [prob0_mix_tea_conf[1:, ...], prob0_mix_tea_conf[:1, ...]], axis=0)

      # Apply mix
      if mix_logits:
        logits_mix_tea = logits0_mix_tea + \
            (logits1_mix_tea - logits0_mix_tea) * mix_blend_facs[:, None]
        prob_mix_tea = jax.nn.softmax(logits_mix_tea)
        prob_mix_tea_conf = jax.nn.softmax(logits_mix_tea)
      else:
        prob_mix_tea = prob0_mix_tea + \
            (prob1_mix_tea - prob0_mix_tea) * mix_blend_facs[:, None]
        prob_mix_tea_conf = prob0_mix_tea_conf + \
            (prob1_mix_tea_conf - prob0_mix_tea_conf) * mix_blend_facs[:, None]

      with flax.deprecated.nn.stateful(new_state_stu) as new_state_stu_mix:
        with flax.deprecated.nn.stochastic(model_rng_mix_stu):
          logits_mix_stu = model_stu(x_mix, train=True)
      prob_mix_stu = jax.nn.softmax(logits_mix_stu)

      if unsup_reg is not None:
        pass
      else:
        new_state_stu = new_state_stu_mix

      if mix_conf_mode == 'mix_prob':
        conf_tea = prob_mix_tea_conf.max(axis=-1, keepdims=True)
      elif mix_conf_mode == 'mix_conf':
        conf0_tea = prob0_mix_tea_conf.max(axis=-1, keepdims=True)
        conf1_tea = prob1_mix_tea_conf.max(axis=-1, keepdims=True)
        conf_tea = conf0_tea + \
            (conf1_tea - conf0_tea) * mix_blend_facs[:, None]
      else:
        raise RuntimeError

      # Confidence thresholding
      mix_loss_mask, mix_conf_mask_rate = confidence_thresholding(
          conf_tea, mix_conf_thresh, mix_conf_avg
      )

      # Mix loss
      mix_var_loss = ((prob_mix_stu - prob_mix_tea) ** 2) * mix_loss_mask
      mix_loss = mix_var_loss.sum(axis=-1).mean()
      loss = loss + mix_loss * mix_weight
    else:
      mix_loss = jnp.array(0.0, dtype=jnp.float32)
      mix_conf_mask_rate = jnp.array(0.0, dtype=jnp.float32)

    if l2_reg > 0:
      weight_penalty_params = jax.tree_leaves(model_stu.params)
      weight_l2 = sum([jnp.sum(x ** 2)
                       for x in weight_penalty_params
                       if x.ndim > 1])
      weight_penalty = l2_reg * 0.5 * weight_l2
      loss = loss + weight_penalty
    return loss, (new_state_stu, new_state_tea, sup_logits, cons_loss,
                  conf_mask_rate, mix_loss, mix_conf_mask_rate)

  step = optimizer_stu.state.step
  lr = learning_rate_fn(step)
  new_optimizer_stu, _, (new_state_stu, new_state_tea, logits,
                         cons_loss, conf_mask_rate, mix_loss,
                         mix_conf_mask_rate) = \
      optimizer_stu.optimize(loss_fn, learning_rate=lr)

  if weight_decay > 0.0:
    decayed_params = jax.tree_map(
        lambda p: p * (1.0 - weight_decay * lr),
        new_optimizer_stu.target.params
    )
    new_model_stu = new_optimizer_stu.target.replace(params=decayed_params)
    new_optimizer_stu = new_optimizer_stu.replace(target=new_model_stu)

  tea_alpha = teacher_alpha_fn(step)

  model_tea_params = jax.tree_multimap(
      lambda p_tea, p_stu: p_tea * tea_alpha + p_stu * (1.0 - tea_alpha),
      model_tea.params, new_optimizer_stu.target.params
  )
  new_state_tea = jax.tree_multimap(
      lambda p_tea, p_stu: p_tea * tea_alpha + p_stu * (1.0 - tea_alpha),
      state_tea, new_state_stu
  )

  model_tea = model_tea.replace(params=model_tea_params)

  metrics = compute_train_metrics(logits, batch['sup_label'])
  metrics['learning_rate'] = lr
  metrics['cons_loss'] = cons_loss
  metrics['conf_rate'] = conf_mask_rate
  metrics['mix_loss'] = mix_loss
  metrics['mix_conf_rate'] = mix_conf_mask_rate

  return new_optimizer_stu, new_state_stu, metrics, model_tea, new_state_tea


def eval_step(model, state, batch, eval_top_5=False):
  state = util.pmean(state)
  with flax.deprecated.nn.stateful(state, mutable=False):
    logits = model(batch['image'], train=False)
  return compute_eval_metrics(logits, batch['label'], eval_top_5=eval_top_5)


def shard(xs, rng=None):
  local_device_count = jax.local_device_count()
  sharded_xs = jax.tree_map(
      lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs)
  if rng is not None:
    keys = jax.random.split(rng, num=local_device_count)
    sharded_xs['rng_key'] = keys
  return sharded_xs


def build_pert_reg(unsupervised_regularizer, cut_backg_noise=1.0,
                   cut_prob=1.0, box_reg_scale_mode='fixed',
                   box_reg_scale=0.25, box_reg_random_aspect_ratio=False,
                   cow_sigma_range=(4.0, 8.0), cow_prop_range=(0.0, 1.0),):
  """Build perturbation regularizer."""
  if unsupervised_regularizer == 'none':
    unsup_reg = None
    augment_twice = False
  elif unsupervised_regularizer == 'mt':
    unsup_reg = regularizers.IdentityRegularizer()
    augment_twice = False
  elif unsupervised_regularizer == 'aug':
    unsup_reg = regularizers.IdentityRegularizer()
    augment_twice = True
  elif unsupervised_regularizer == 'cutout':
    unsup_reg = regularizers.BoxMaskRegularizer(
        cut_backg_noise, cut_prob, box_reg_scale_mode, box_reg_scale,
        box_reg_random_aspect_ratio)
    augment_twice = False
  elif unsupervised_regularizer == 'aug_cutout':
    unsup_reg = regularizers.BoxMaskRegularizer(
        cut_backg_noise, cut_prob, box_reg_scale_mode, box_reg_scale,
        box_reg_random_aspect_ratio)
    augment_twice = True
  elif unsupervised_regularizer == 'cowout':
    unsup_reg = regularizers.CowMaskRegularizer(
        cut_backg_noise, cut_prob, cow_sigma_range, cow_prop_range)
    augment_twice = False
  elif unsupervised_regularizer == 'aug_cowout':
    unsup_reg = regularizers.CowMaskRegularizer(
        cut_backg_noise, cut_prob, cow_sigma_range, cow_prop_range)
    augment_twice = True
  else:
    raise ValueError('Unknown supervised_regularizer \'{}\''.format(
        unsupervised_regularizer))
  return unsup_reg, augment_twice


def build_mix_reg(mix_regularizer, ict_alpha=0.1,
                  box_reg_scale_mode='fixed', box_reg_scale=0.25,
                  box_reg_random_aspect_ratio=False,
                  cow_sigma_range=(4.0, 8.0), cow_prop_range=(0.0, 1.0)):
  """Build mix regularizer."""
  if mix_regularizer == 'none':
    mix_reg = None
  elif mix_regularizer == 'ict':
    mix_reg = regularizers.ICTRegularizer(ict_alpha)
  elif mix_regularizer == 'cutmix':
    mix_reg = regularizers.BoxMaskRegularizer(
        0.0, 1.0, box_reg_scale_mode, box_reg_scale,
        box_reg_random_aspect_ratio)
  elif mix_regularizer == 'cowmix':
    mix_reg = regularizers.CowMaskRegularizer(
        0.0, 1.0, cow_sigma_range, cow_prop_range)
  else:
    raise ValueError('Unknown supervised_regularizer \'{}\''.format(
        mix_regularizer))
  return mix_reg


def experiment(model_dir='.',  # pylint: disable=dangerous-default-value
               imagenet_subset_dir=None,
               dataset='cifar10',
               batch_size=256,
               eval_batch_size=1024,
               num_epochs=200,
               learning_rate=0.1,
               aug_imagenet_apply_colour_jitter=False,
               aug_imagenet_greyscale_prob=0.0,
               sgd_momentum=0.9,
               sgd_nesterov=True,
               lr_schedule='stepped',
               lr_sched_steps=[[60, 0.2], [120, 0.04], [160, 0.008]],
               lr_sched_halfcoslength=400.0,
               lr_sched_warmup=5.0,
               l2_reg=0.0005,
               weight_decay=0.0,
               architecture='wrn22_10',
               n_val=5000,
               n_sup=1000,
               teacher_alpha=0.999,
               anneal_teacher_alpha=False,
               unsupervised_regularizer='none',
               cons_weight=1.0,
               conf_thresh=0.97,
               conf_avg=False,
               cut_backg_noise=1.0,
               cut_prob=1.0,
               box_reg_scale_mode='fixed',
               box_reg_scale=0.25,
               box_reg_random_aspect_ratio=False,
               cow_sigma_range=(4.0, 8.0),
               cow_prop_range=(0.25, 1.0),
               mix_regularizer='none',
               mix_aug_separately=False,
               mix_logits=True,
               mix_weight=1.0,
               mix_conf_thresh=0.97,
               mix_conf_avg=True,
               mix_conf_mode='mix_prob',
               ict_alpha=0.1,
               mix_box_reg_scale_mode='fixed',
               mix_box_reg_scale=0.25,
               mix_box_reg_random_aspect_ratio=False,
               mix_cow_sigma_range=(4.0, 8.0),
               mix_cow_prop_range=(0.0, 1.0),
               subset_seed=12345,
               val_seed=131,
               run_seed=None,
               log_fn=print,
               checkpoints='on',
               on_epoch_finished_fn=None,
               debug=False):
  """Run experiment."""
  if checkpoints not in {'none', 'on', 'retain'}:
    raise ValueError('checkpoints should be one of (none|on|retain)')

  if checkpoints != 'none':
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pkl')
    checkpoint_new_path = os.path.join(model_dir, 'checkpoint.pkl.new')
  else:
    checkpoint_path = None
    checkpoint_new_path = None

  if dataset not in {'svhn', 'cifar10', 'cifar100', 'imagenet'}:
    raise ValueError('Unknown dataset \'{}\''.format(dataset))

  if architecture not in {'wrn20_10', 'wrn26_10', 'wrn26_2',
                          'wrn20_6_shakeshake', 'wrn26_6_shakeshake',
                          'wrn26_2_shakeshake', 'pyramid',
                          'resnet50', 'resnet101', 'resnet152',
                          'resnet50x2', 'resnet101x2', 'resnet152x2',
                          'resnet50x4', 'resnet101x4', 'resnet152x4',
                          'resnext50_32x4d', 'resnext101_32x8d',
                          'resnext152_32x4d'}:
    raise ValueError('Unknown architecture \'{}\''.format(architecture))

  if lr_schedule not in {'constant', 'stepped', 'cosine'}:
    raise ValueError('Unknown LR schedule \'{}\''.format(lr_schedule))

  if mix_conf_mode not in {'mix_prob', 'mix_conf'}:
    raise ValueError('Unknown mix_conf_mode \'{}\''.format(mix_conf_mode))

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(model_dir)
  else:
    summary_writer = None

  unsup_reg, augment_twice = build_pert_reg(
      unsupervised_regularizer, cut_backg_noise=cut_backg_noise,
      cut_prob=cut_prob, box_reg_scale_mode=box_reg_scale_mode,
      box_reg_scale=box_reg_scale,
      box_reg_random_aspect_ratio=box_reg_random_aspect_ratio,
      cow_sigma_range=cow_sigma_range, cow_prop_range=cow_prop_range)

  mix_reg = build_mix_reg(
      mix_regularizer, ict_alpha=ict_alpha,
      box_reg_scale_mode=mix_box_reg_scale_mode,
      box_reg_scale=mix_box_reg_scale,
      box_reg_random_aspect_ratio=mix_box_reg_random_aspect_ratio,
      cow_sigma_range=mix_cow_sigma_range, cow_prop_range=mix_cow_prop_range)

  if run_seed is None:
    run_seed = subset_seed << 32 | n_val
  train_rng = jax.random.PRNGKey(run_seed)
  init_rng, train_rng = jax.random.split(train_rng)

  if batch_size % jax.device_count() > 0:
    raise ValueError('Train batch size must be divisible by the number of '
                     'devices')
  if eval_batch_size % jax.device_count() > 0:
    raise ValueError('Eval batch size must be divisible by the number of '
                     'devices')
  local_batch_size = batch_size // jax.host_count()
  local_eval_batch_size = eval_batch_size // jax.host_count()
  device_batch_size = batch_size // jax.device_count()

  if dataset == 'svhn':
    image_size = 32
    top5_err_required = False
    data_source = small_image_data_source.SVHNDataSource(
        n_val=n_val, n_sup=n_sup, train_batch_size=local_batch_size,
        eval_batch_size=local_eval_batch_size,
        augment_twice=augment_twice, subset_seed=subset_seed,
        val_seed=val_seed)
  elif dataset == 'cifar10':
    image_size = 32
    top5_err_required = False
    data_source = small_image_data_source.CIFAR10DataSource(
        n_val=n_val, n_sup=n_sup, train_batch_size=local_batch_size,
        eval_batch_size=local_eval_batch_size, augment_twice=augment_twice,
        subset_seed=subset_seed, val_seed=val_seed)
  elif dataset == 'cifar100':
    image_size = 32
    top5_err_required = False
    data_source = small_image_data_source.CIFAR100DataSource(
        n_val=n_val, n_sup=n_sup, train_batch_size=local_batch_size,
        eval_batch_size=local_eval_batch_size, augment_twice=augment_twice,
        subset_seed=subset_seed, val_seed=val_seed)
  elif dataset == 'imagenet':
    image_size = 224
    top5_err_required = True
    if imagenet_subset_dir is None:
      raise ValueError('Please provide a directory to the imagenet_subset_dir '
                       'command line arg to specify where the ImageNet '
                       'subsets are stored')
    data_source = imagenet_data_source.ImageNetDataSource(
        imagenet_subset_dir, n_val, n_sup, local_batch_size,
        local_eval_batch_size, augment_twice,
        apply_colour_jitter=aug_imagenet_apply_colour_jitter,
        greyscale_prob=aug_imagenet_greyscale_prob,
        load_test_set=(n_val == 0), image_size=image_size,
        subset_seed=subset_seed, val_seed=val_seed)
  else:
    raise RuntimeError

  n_train = data_source.n_train
  train_ds = data_source.train_semisup_ds

  if n_val == 0:
    eval_ds = data_source.test_ds
    n_eval = data_source.n_test
  else:
    eval_ds = data_source.val_ds
    n_eval = data_source.n_val

  log_fn('DATA: |train|={}, |sup|={}, |eval|={}, (|val|={}, |test|={})'.format(
      data_source.n_train, data_source.n_sup, n_eval, data_source.n_val,
      data_source.n_test))

  log_fn('Loaded dataset')

  steps_per_epoch = n_train // batch_size
  steps_per_eval = n_eval // eval_batch_size
  if n_eval % eval_batch_size > 0:
    steps_per_eval += 1
  num_steps = steps_per_epoch * num_epochs

  # Create model
  model_stu, state_stu = create_model(
      init_rng, architecture, device_batch_size, image_size,
      data_source.n_classes)
  state_stu = jax_utils.replicate(state_stu)
  log_fn('Built model')

  # Create optimizer
  optimizer_def = optim.Momentum(learning_rate=learning_rate,
                                 beta=sgd_momentum,
                                 nesterov=sgd_nesterov)

  optimizer_stu = optimizer_def.create(model_stu)
  optimizer_stu = optimizer_stu.replicate()
  del model_stu  # don't keep a copy of the initial model

  # Create learning rate function
  base_learning_rate = learning_rate * batch_size / 256.
  if lr_schedule == 'constant':
    learning_rate_fn = create_constant_learning_rate_fn(base_learning_rate)
  elif lr_schedule == 'stepped':
    learning_rate_fn = create_stepped_learning_rate_fn(
        base_learning_rate, steps_per_epoch, lr_sched_steps=lr_sched_steps,
        warmup_length=lr_sched_warmup)
  elif lr_schedule == 'cosine':
    learning_rate_fn = create_cosine_learning_rate_fn(
        base_learning_rate, steps_per_epoch,
        halfcoslength_epochs=lr_sched_halfcoslength,
        warmup_length=lr_sched_warmup)
  else:
    raise RuntimeError

  if anneal_teacher_alpha:
    if lr_schedule == 'constant':
      one_minus_alpha_fn = create_constant_learning_rate_fn(1.0 - teacher_alpha)
    elif lr_schedule == 'stepped':
      one_minus_alpha_fn = create_stepped_learning_rate_fn(
          1.0 - teacher_alpha, steps_per_epoch, lr_sched_steps=lr_sched_steps)
    elif lr_schedule == 'cosine':
      one_minus_alpha_fn = create_cosine_learning_rate_fn(
          1.0 - teacher_alpha, steps_per_epoch,
          halfcoslength_epochs=lr_sched_halfcoslength)
    else:
      raise RuntimeError
    teacher_alpha_fn = lambda step: 1.0 - one_minus_alpha_fn(step)
  else:
    teacher_alpha_fn = lambda step: teacher_alpha

  log_fn('Built optimizer')

  # Teacher model is just the student as we duplicate it when we modify it
  model_tea = optimizer_stu.target
  # Replicate batch stats
  state_tea = jax.tree_map(lambda x: x, state_stu)

  # Set up epoch and step counter
  # Load existing checkpoint if available
  epoch = 1
  step = 0

  if checkpoints != 'none':
    if tf.io.gfile.exists(checkpoint_path):
      with tf.io.gfile.GFile(checkpoint_path, 'rb') as f_in:
        check = pickle.load(f_in)

        # Student optimizer and batch stats
        optimizer_stu = util.restore_state_list(
            optimizer_stu, check['optimizer_stu'])

        state_stu = util.restore_state_list(
            state_stu, check['state_stu'])

        # Teacher model and batch stats
        model_tea = util.restore_state_list(
            model_tea, check['model_tea'])

        state_tea = util.restore_state_list(
            state_tea, check['state_tea'])

        epoch = check['epoch']
        step = check['step']

        log_fn('Loaded checkpoint from {}'.format(checkpoint_path))

  #
  # Training and evaluation step functions
  #
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn,
                        l2_reg=l2_reg, weight_decay=weight_decay,
                        teacher_alpha_fn=teacher_alpha_fn,
                        unsup_reg=unsup_reg, cons_weight=cons_weight,
                        conf_thresh=conf_thresh,
                        conf_avg=conf_avg,
                        mix_reg=mix_reg, mix_aug_separately=mix_aug_separately,
                        mix_logits=mix_logits, mix_weight=mix_weight,
                        mix_conf_thresh=mix_conf_thresh,
                        mix_conf_avg=mix_conf_avg,
                        mix_conf_mode=mix_conf_mode),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(eval_step, eval_top_5=top5_err_required),
      axis_name='batch')

  # Create dataset batch iterators
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  #
  # Training loop
  #

  log_fn('Training...')
  epoch_metrics_stu = []
  t1 = time.time()
  while step < num_steps:
    train_rng, iter_rng = jax.random.split(train_rng)
    batch = next(train_iter)
    batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
    batch = shard(batch, iter_rng)

    optimizer_stu, state_stu, metrics_stu, model_tea, state_tea = p_train_step(
        optimizer_stu, state_stu, model_tea, state_tea, batch)

    if debug:
      log_fn('Step {} time {}'.format(step, time.time()-t1))

    epoch_metrics_stu.append(metrics_stu)
    if (step + 1) % steps_per_epoch == 0:
      epoch_metrics_stu = util.get_metrics(epoch_metrics_stu)
      train_epoch_metrics = jax.tree_map(lambda x: x.mean(), epoch_metrics_stu)
      if summary_writer is not None:
        for key, vals in epoch_metrics_stu.items():
          tag = 'train_%s' % key
          for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

      epoch_metrics_stu = []
      eval_stu_metrics = []
      eval_tea_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        # TF to NumPy
        eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
        # Pad short batches
        eval_batch = util.pad_classification_batch(
            eval_batch, local_eval_batch_size)
        # Shard across local devices
        eval_batch = shard(eval_batch)
        metrics_stu = p_eval_step(optimizer_stu.target, state_stu, eval_batch)
        metrics_tea = p_eval_step(model_tea, state_tea, eval_batch)
        eval_stu_metrics.append(metrics_stu)
        eval_tea_metrics.append(metrics_tea)
      eval_stu_metrics = util.get_metrics(eval_stu_metrics)
      eval_tea_metrics = util.get_metrics(eval_tea_metrics)
      eval_stu_epoch_metrics = jax.tree_map(lambda x: x.sum(), eval_stu_metrics)
      eval_tea_epoch_metrics = jax.tree_map(lambda x: x.sum(), eval_tea_metrics)
      eval_stu_epoch_metrics = avg_eval_metrics(eval_stu_epoch_metrics)
      eval_tea_epoch_metrics = avg_eval_metrics(eval_tea_epoch_metrics)

      t2 = time.time()

      if top5_err_required:
        log_fn('EPOCH {} (took {:.3f}s): Train loss={:.6f}, err={:.3%}, '
               'cons loss={:.6f}, conf rate={:.3%}, mix loss={:.6f}, '
               'mix conf rate={:.3%}; STU Eval loss={:.6f}, err={:.3%}, '
               'top-5-err={:.3%}, TEA Eval loss={:.6f}, err={:.3%}, '
               'top-5-err={:.3%}'.format(
                   epoch, t2 - t1, train_epoch_metrics['loss'],
                   train_epoch_metrics['error_rate'],
                   train_epoch_metrics['cons_loss'],
                   train_epoch_metrics['conf_rate'],
                   train_epoch_metrics['mix_loss'],
                   train_epoch_metrics['mix_conf_rate'],
                   eval_stu_epoch_metrics['loss'],
                   eval_stu_epoch_metrics['error_rate'],
                   eval_stu_epoch_metrics['top5_error_rate'],
                   eval_tea_epoch_metrics['loss'],
                   eval_tea_epoch_metrics['error_rate'],
                   eval_tea_epoch_metrics['top5_error_rate'],))
      else:
        log_fn('EPOCH {} (took {:.3f}s): Train loss={:.6f}, err={:.3%}, '
               'cons loss={:.6f}, conf rate={:.3%}, mix loss={:.6f}, '
               'mix conf rate={:.3%}; STU Eval loss={:.6f}, err={:.3%}, '
               'TEA Eval loss={:.6f}, err={:.3%}'.format(
                   epoch, t2 - t1, train_epoch_metrics['loss'],
                   train_epoch_metrics['error_rate'],
                   train_epoch_metrics['cons_loss'],
                   train_epoch_metrics['conf_rate'],
                   train_epoch_metrics['mix_loss'],
                   train_epoch_metrics['mix_conf_rate'],
                   eval_stu_epoch_metrics['loss'],
                   eval_stu_epoch_metrics['error_rate'],
                   eval_tea_epoch_metrics['loss'],
                   eval_tea_epoch_metrics['error_rate'],))

      if on_epoch_finished_fn is not None:
        if top5_err_required:
          on_epoch_finished_fn(
              epoch,
              eval_stu_err=eval_stu_epoch_metrics['error_rate'],
              eval_tea_err=eval_tea_epoch_metrics['error_rate'],
              eval_stu_top5_err=eval_stu_epoch_metrics['top5_error_rate'],
              eval_tea_top5_err=eval_tea_epoch_metrics['top5_error_rate'],
          )
        else:
          on_epoch_finished_fn(
              epoch,
              eval_stu_err=eval_stu_epoch_metrics['error_rate'],
              eval_tea_err=eval_tea_epoch_metrics['error_rate'],
          )

      t1 = t2

      if summary_writer is not None:
        summary_writer.scalar(
            'eval_stu_loss', eval_stu_epoch_metrics['loss'], epoch)
        summary_writer.scalar(
            'eval_stu_error_rate', eval_stu_epoch_metrics['error_rate'], epoch)
        summary_writer.scalar(
            'eval_tea_loss', eval_tea_epoch_metrics['loss'], epoch)
        summary_writer.scalar(
            'eval_tea_error_rate', eval_tea_epoch_metrics['error_rate'], epoch)
        if top5_err_required:
          summary_writer.scalar(
              'eval_stu_top5_error_rate',
              eval_stu_epoch_metrics['top5_error_rate'], epoch)
          summary_writer.scalar(
              'eval_tea_top5_error_rate',
              eval_tea_epoch_metrics['top5_error_rate'], epoch)
        summary_writer.flush()

        epoch += 1

        if checkpoints != 'none':
          if jax.host_id() == 0:
            # Write to new checkpoint file so that we don't immediately
            # overwrite the old one
            with tf.io.gfile.GFile(checkpoint_new_path, 'wb') as f_out:
              check = dict(
                  optimizer_stu=util.to_state_list(optimizer_stu),
                  state_stu=util.to_state_list(state_stu),
                  model_tea=util.to_state_list(model_tea),
                  state_tea=util.to_state_list(state_tea),
                  epoch=epoch,
                  step=step + 1,
              )
              pickle.dump(check, f_out)
              del check
            # Remove old checkpoint and rename
            if tf.io.gfile.exists(checkpoint_path):
              tf.io.gfile.remove(checkpoint_path)
            tf.io.gfile.rename(checkpoint_new_path, checkpoint_path)

    step += 1

  if checkpoints == 'on':
    if jax.host_id() == 0:
      if tf.io.gfile.exists(checkpoint_path):
        tf.io.gfile.remove(checkpoint_path)
