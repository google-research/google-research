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

"""CIFAR10 diffusion model."""

from typing import Dict
import jax.numpy as jnp
import ml_collections

from d3pm.images import entry_point
from d3pm.images import gm
from d3pm.images import model
from d3pm.images import utils
from d3pm.images.diffusion_categorical import make_diffusion


class Cifar10DiffusionModel(gm.TrainableModel):
  """CIFAR10 diffusion model."""

  def __init__(self, config):
    super().__init__(config)

    assert config.dataset.name in {'CIFAR10', 'MockCIFAR10'}
    self.num_bits = 8

    self.model_class = {'unet0': model.UNet}[config.model.name]
    # Ensure that max_time in model and num_timesteps in the betas are the same.
    self.model = self.model_class(
        num_classes=self.dataset.num_classes,
        max_time=config.model.diffusion_betas.num_timesteps,
        num_pixel_vals=2**self.num_bits,
        **config.model.args)
    self.num_timesteps = config.model.diffusion_betas.num_timesteps

    assert self.config.train.num_train_steps is not None
    assert self.config.train.num_train_steps % self.config.train.substeps == 0
    assert self.config.train.retain_checkpoint_every_steps % self.config.train.substeps == 0

  def make_init_params(self, global_rng):
    init_kwargs = dict(
        x=jnp.zeros((1, *self.dataset.data_shape), dtype=jnp.int32),
        t=jnp.zeros((1,), dtype=jnp.int32),
        y=jnp.zeros((1,), dtype=jnp.int32),
        train=False,
    )
    return self.model.init({'params': global_rng}, **init_kwargs)['params']

  def step_fn(self, base_rng, train, state,
              batch):
    """Converts x_start input data to int32 and then does regular step_fn."""
    batch['image'] = batch['image'].astype(jnp.int32)
    return super().step_fn(base_rng, train, state, batch)

  def loss_fn(self, rng, train, batch, params):
    rng = utils.RngGen(rng)

    # Input: image
    img = batch['image']
    assert img.dtype == jnp.int32

    # Input: label
    label = batch.get('label', None)
    if label is not None:
      assert label.shape == (img.shape[0],)
      assert label.dtype == jnp.int32

    def model_fn(x, t):
      return self.model.apply({'params': params},
                              x=x,
                              t=t,
                              y=label,
                              train=train,
                              rngs={'dropout': next(rng)} if train else None)

    dif = make_diffusion(self.config.model, num_bits=self.num_bits)
    loss = dif.training_losses(model_fn, x_start=img, rng=next(rng)).mean()
    if not train:
      loss_dict = dif.calc_bpd_loop(model_fn, x_start=img, rng=next(rng))
      total_bpd = jnp.mean(loss_dict['total'], axis=0)  # scalar
      # vb_terms = jnp.mean(loss_dict['vbterms'], axis=0)  # vec: num_timesteps
      prior_bpd = jnp.mean(loss_dict['prior'], axis=0)
      return loss, {
          'loss': loss,
          'prior_bpd': prior_bpd,
          'total_bpd': total_bpd
      }
    else:
      prior_bpd = dif.prior_bpd(img).mean()
      return loss, {
          'loss': loss,
          'prior_bpd': prior_bpd,
      }

  def samples_fn(self, params, rng, samples_shape):
    y = None

    def model_fn(x, t):
      return self.model.apply({'params': params}, x=x, t=t, y=y, train=False)

    samples = make_diffusion(self.config.model, self.num_bits).p_sample_loop(
        model_fn=model_fn, shape=samples_shape, rng=rng)
    # Samples are integer values in range [0, 255]
    assert samples.shape == samples_shape
    return samples.astype(jnp.float32)


def run_train(*, config, experiment_dir,
              work_unit_dir, rng):
  return Cifar10DiffusionModel(config).run_train(
      experiment_dir=experiment_dir, work_unit_dir=work_unit_dir, rng=rng)


if __name__ == '__main__':
  entry_point.run(train=run_train)
