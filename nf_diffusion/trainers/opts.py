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

"""Optimization related logics."""

from absl import logging
import flax
import flax.optim
import jax.numpy as jnp


def get_opt(config, params, opt_cfg_key=None, return_def=False):
  """Get optimizer according to configuration."""
  cfg_opt = config.opt
  if opt_cfg_key is not None:
    cfg_opt = cfg_opt.get(opt_cfg_key)

  if cfg_opt.type.lower() == "adam":
    opt_def = flax.optim.Adam(
        beta1=cfg_opt.get("beta1", 0.9),
        beta2=getattr(cfg_opt, "beta2", 0.999)
    )
  elif cfg_opt.type.lower() == "sgd":
    opt_def = flax.optim.Momentum(
        beta=cfg_opt.sgd_momentum)
  else:
    raise ValueError(f"Optimizer {config.opt.type} not supported.")

  if return_def:
    return opt_def
  else:
    optimizer = opt_def.create(params)
    return optimizer


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def constant(lr, *_):
  return lr


def get_learning_rate(step,
                      *,
                      base_learning_rate,
                      steps_per_epoch,
                      num_epochs,
                      warmup_epochs = 5,
                      lr_schedule = "cosine"):
  """Cosine learning rate schedule."""
  logging.info(
      (
          "get_learning_rate(step=%s, base_learning_rate=%s,"
          " steps_per_epoch=%s, num_epochs=%s"
      ),
      step,
      base_learning_rate,
      steps_per_epoch,
      num_epochs,
  )
  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch should be a positive integer but was "
                     f"{steps_per_epoch}.")
  if warmup_epochs >= num_epochs:
    raise ValueError(f"warmup_epochs should be smaller than num_epochs. "
                     f"Currently warmup_epochs is {warmup_epochs}, "
                     f"and num_epochs is {num_epochs}.")
  epoch = step / steps_per_epoch
  if lr_schedule == "cosine":
    lr = cosine_decay(base_learning_rate, epoch - warmup_epochs,
                      num_epochs - warmup_epochs)
  elif lr_schedule == "constant":
    lr = constant(base_learning_rate, epoch - warmup_epochs,
                  num_epochs - warmup_epochs)
  else:
    raise NotImplementedError
  if warmup_epochs > 0:
    warmup = jnp.minimum(1., epoch / warmup_epochs)
  else:
    warmup = 1
  return lr * warmup
