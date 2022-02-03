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

"""Diffusion model training and distillation."""

# pylint: disable=g-long-lambda,g-complex-comprehension,g-long-ternary
# pylint: disable=invalid-name,logging-format-interpolation

import functools
from typing import Any, Dict, Union

from . import checkpoints
from . import datasets
from . import dpm
from . import schedules
from . import unet
from . import utils
from absl import logging
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as onp


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: Union[flax.optim.Optimizer, None]
  ema_params: Any
  num_sample_steps: int


class Model:
  """Diffusion model."""

  def __init__(self, config, dataset=None):
    self.config = config

    if dataset is not None:
      self.dataset = dataset
    else:
      self.dataset = getattr(datasets, config.dataset.name)(
          **config.dataset.args)

    self._eval_step = None

    # infer number of output channels for UNet
    x_ch = self.dataset.data_shape[-1]
    out_ch = x_ch
    if config.model.mean_type == 'both':
      out_ch += x_ch
    if 'learned' in config.model.logvar_type:
      out_ch += x_ch

    self.model = unet.UNet(
        num_classes=self.dataset.num_classes,
        out_ch=out_ch,
        **config.model.args)

  @property
  def current_num_steps(self):
    if hasattr(self.config, 'distillation'):
      assert hasattr(self, 'teacher_state')
      return int(self.teacher_state.num_sample_steps // 2)
    else:
      return self.config.model.train_num_steps

  def make_init_params(self, global_rng):
    init_kwargs = dict(
        x=jnp.zeros((1, *self.dataset.data_shape), dtype=jnp.float32),
        y=jnp.zeros((1,), dtype=jnp.int32),
        logsnr=jnp.zeros((1,), dtype=jnp.float32),
        train=False,
    )
    return self.model.init({'params': global_rng}, **init_kwargs)['params']

  def make_init_state(self):
    """Make an initial TrainState."""
    # Init model params (same rng across hosts)
    init_params = self.make_init_params(
        global_rng=jax.random.PRNGKey(self.config.seed))
    logging.info('Param shapes: {}'.format(
        jax.tree_map(lambda a: a.shape, init_params)))
    logging.info('Number of trainable parameters: {:,}'.format(
        utils.count_params(init_params)))

    # Make the optimizer
    optimizer_def = self.make_optimizer_def()

    # For ema_params below, copy so that pmap buffer donation doesn't donate the
    # same buffer twice
    return TrainState(
        step=0,
        optimizer=optimizer_def.create(init_params),
        ema_params=utils.copy_pytree(init_params),
        num_sample_steps=self.config.model.train_num_steps)

  def load_teacher_state(self, ckpt_path=None):
    """Load teacher state and fix flax version incompatibilities."""
    teacher_state = jax.device_get(
        self.make_init_state().replace(optimizer=None))
    if ckpt_path is None:
      ckpt_path = self.config.distillation.teacher_checkpoint_path
    loaded_state = checkpoints.restore_from_path(ckpt_path, target=None)
    teacher_params = loaded_state['ema_params']
    teacher_params = flax.core.unfreeze(teacher_params)
    teacher_params = jax.tree_map(
        lambda x, y: onp.reshape(x, y.shape) if hasattr(y, 'shape') else x,
        teacher_params,
        flax.core.unfreeze(teacher_state.ema_params))
    teacher_params = flax.core.freeze(teacher_params)
    if ('num_sample_steps' in loaded_state and
        loaded_state['num_sample_steps'] > 0):
      num_sample_steps = loaded_state['num_sample_steps']
    else:
      num_sample_steps = self.config.distillation.start_num_steps
    self.teacher_state = TrainState(
        step=0,  # reset number of steps
        optimizer=None,
        ema_params=teacher_params,
        num_sample_steps=num_sample_steps,
        )

  def loss_fn(self, rng, train, batch, params):
    """Training/distillation loss for diffusion model."""
    rng = utils.RngGen(rng)

    # Input: image
    img = batch['image']
    assert img.dtype == jnp.float32
    img = utils.normalize_data(img)  # scale image to [-1, 1]

    # Input: label
    label = batch.get('label', None)
    if label is not None:
      assert label.shape == (img.shape[0],)
      assert label.dtype == jnp.int32

    def model_fn(x, logsnr):
      return self.model.apply(
          {'params': params}, x=x, logsnr=logsnr, y=label, train=train,
          rngs={'dropout': next(rng)} if train else None)

    if hasattr(self.config, 'distillation'):
      def target_model_fn(x, logsnr):
        return self.model.apply(
            {'params': self.teacher_state.ema_params}, x=x, logsnr=logsnr,
            y=label, train=False, rngs=None)
    else:
      target_model_fn = None

    logging.info(
        f'train_logsnr_schedule: {self.config.model.train_logsnr_schedule}')
    model = dpm.Model(
        model_fn=model_fn,
        target_model_fn=target_model_fn,
        mean_type=self.config.model.mean_type,
        logvar_type=self.config.model.logvar_type,
        logvar_coeff=self.config.model.get('logvar_coeff', 0.))
    loss_dict = model.training_losses(
        x=img,
        rng=next(rng),
        logsnr_schedule_fn=schedules.get_logsnr_schedule(
            **self.config.model.train_logsnr_schedule),
        num_steps=self.current_num_steps,
        mean_loss_weight_type=self.config.model.mean_loss_weight_type)

    assert all(v.shape == (img.shape[0],) for v in loss_dict.values())
    loss_dict = {k: v.mean() for (k, v) in loss_dict.items()}
    return loss_dict['loss'], loss_dict

  def step_fn(self, base_rng, train, state,
              batch, learning_rate_mult=1.):
    """One training/eval step."""
    config = self.config

    # RNG for this step on this host
    step = state.step
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, step)
    rng = utils.RngGen(rng)

    # Loss and gradient
    loss_fn = functools.partial(self.loss_fn, next(rng), train, batch)

    if train:
      # Training mode
      (_, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(
          state.optimizer.target)

      # Average grad across shards
      grad, metrics['gnorm'] = utils.clip_by_global_norm(
          grad, clip_norm=config.train.grad_clip)
      grad = jax.lax.pmean(grad, axis_name='batch')

      # Learning rate decay/warmup
      learning_rate = config.train.learning_rate
      if learning_rate_mult != 1:
        learning_rate *= learning_rate_mult

      # Update optimizer and EMA params
      new_optimizer = state.optimizer.apply_gradient(
          grad, learning_rate=learning_rate)
      if hasattr(config.train, 'ema_decay'):
        ema_decay = config.train.ema_decay
      elif config.train.avg_type == 'ema':
        ema_decay = 1. - (1. / config.train.avg_steps)
      elif config.train.avg_type == 'aa':
        t = step % config.train.avg_steps
        ema_decay = t / (t + 1.)
      elif config.train.avg_type is None:
        ema_decay = 0.
      else:
        raise NotImplementedError(config.train.avg_type)
      if ema_decay == 0:
        new_ema_params = new_optimizer.target
      else:
        new_ema_params = utils.apply_ema(
            decay=jnp.where(step == 0, 0.0, ema_decay),
            avg=state.ema_params,
            new=new_optimizer.target)
      new_state = state.replace(  # pytype: disable=attribute-error
          step=step + 1,
          optimizer=new_optimizer,
          ema_params=new_ema_params)
      if config.train.get('enable_update_skip', True):
        # Apply update if the new optimizer state is all finite
        ok = jnp.all(jnp.asarray([
            jnp.all(jnp.isfinite(p)) for p in jax.tree_leaves(new_optimizer)]))
        new_state_no_update = state.replace(step=step + 1)
        state = jax.tree_multimap(
            lambda a, b: jnp.where(ok, a, b), new_state, new_state_no_update)
      else:
        logging.info('Update skipping disabled')
        state = new_state

    else:
      # Eval mode with EMA params
      _, metrics = loss_fn(state.ema_params)

    # Average metrics across shards
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    assert all(v.shape == () for v in metrics.values())
    metrics = {  # prepend prefix to names of metrics
        f"{'train' if train else 'eval'}/{k}": v for k, v in metrics.items()
    }
    return (state, metrics) if train else metrics

  def samples_fn(self,
                 *,
                 rng,
                 params,
                 labels=None,
                 batch=None,
                 num_steps=None,
                 num_samples=8):
    """Sample from the model."""
    rng = utils.RngGen(rng)
    if labels is not None:
      y = labels
    elif batch is not None:
      y = batch.get('label', None)
    else:
      y = None
    if y is not None:
      num_samples = len(y)
    if batch is not None and 'image' in batch:
      dummy_x = batch['image']
    else:
      dummy_x = jnp.zeros((num_samples, *self.dataset.data_shape),
                          dtype=jnp.float32)

    model_fn = lambda x, logsnr: self.model.apply(
        {'params': params}, x=x, logsnr=logsnr, y=y, train=False)

    if num_steps is None:
      num_steps = self.config.model.eval_sampling_num_steps
    logging.info(
        f'eval_sampling_num_steps: {num_steps}'
    )
    logging.info(
        f'eval_logsnr_schedule: {self.config.model.eval_logsnr_schedule}')

    init_x = jax.random.normal(
        next(rng), shape=dummy_x.shape, dtype=dummy_x.dtype)

    model = dpm.Model(
        model_fn=model_fn,
        mean_type=self.config.model.mean_type,
        logvar_type=self.config.model.logvar_type,
        logvar_coeff=self.config.model.get('logvar_coeff', 0.))
    samples = model.sample_loop(
        rng=next(rng),
        init_x=init_x,
        num_steps=num_steps,
        logsnr_schedule_fn=schedules.get_logsnr_schedule(
            **self.config.model.eval_logsnr_schedule),
        sampler=self.config.sampler,
        clip_x=self.config.model.eval_clip_denoised)

    unnormalized_samples = jnp.clip(utils.unnormalize_data(samples), 0, 255)
    assert unnormalized_samples.shape == dummy_x.shape
    return unnormalized_samples

  def make_optimizer_def(self):
    """Make the optimizer def."""
    config = self.config

    optimizer_kwargs = {}
    if config.train.weight_decay > 0.:
      optimizer_kwargs['weight_decay'] = config.train.weight_decay

    if config.train.optimizer == 'adam':
      optimizer_def = flax.optim.Adam(
          **optimizer_kwargs,
          beta1=config.train.get('adam_beta1', 0.9),
          beta2=config.train.get('adam_beta2', 0.999))
    elif config.train.optimizer == 'momentum':
      optimizer_def = flax.optim.Momentum(
          **optimizer_kwargs,
          beta=config.train.get('optimizer_beta', 0.9))
    elif config.train.optimizer == 'nesterov':
      optimizer_def = flax.optim.Momentum(
          **optimizer_kwargs,
          beta=config.train.get('optimizer_beta', 0.9),
          nesterov=True)
    else:
      raise NotImplementedError(f'Unknown optimizer: {config.train.optimizer}')

    return optimizer_def


