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

"""Utilities for training."""


from clu import metrics
import flax
import jax
import ml_collections
import numpy as np
import optax


def prepare_example_batch(example_batch):
  """Function to get rid of extra dimension in batch due to pmap."""
  # Get rid of the pmapping dimension as intialization is done on main process
  example_batch = jax.tree.map(lambda x: np.asarray(x[0]), example_batch)
  return example_batch


def create_optimizer(config, learning_rate_fn):
  """Create training optimizer."""
  optimizer = optax.adamw(
      learning_rate=learning_rate_fn,
      b1=0.9,
      b2=0.98,
      eps=1e-9,
      weight_decay=config.train.weight_decay,
  )

  if config.train.grad_max_norm > 0:
    tx = optax.chain(
        optax.clip_by_global_norm(config.train.grad_max_norm), optimizer
    )
  elif config.train.grad_max_val > 1:
    tx = optax.chain(optax.clip(config.train.grad_max_val), optimizer)
  else:
    tx = optimizer
  return tx


def create_learning_rate_fn(
    config,
):
  """Create learning rate schedule."""
  # Linear warmup
  warmup_fn = optax.linear_schedule(
      init_value=0.0,
      end_value=config.train.lr_init,
      transition_steps=config.train.warmup_steps,
  )

  if config.train.scheduler == "linear":
    decay_fn = optax.linear_schedule(
        init_value=config.train.lr_init,
        end_value=0.0,
        transition_steps=config.train.max_steps - config.train.warmup_steps,
    )
  elif config.train.scheduler == "cosine":
    cosine_steps = max(config.train.max_steps - config.train.warmup_steps, 1)
    decay_fn = optax.cosine_decay_schedule(
        init_value=config.train.lr_init, decay_steps=cosine_steps
    )
  elif config.train.scheduler == "step":

    def schedule(count):
      return config.train.lr_init * (0.5 ** (count // 50000))

    decay_fn = schedule

  else:
    raise NotImplementedError

  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, decay_fn], boundaries=[config.train.warmup_steps]
  )
  return schedule_fn


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """To store the train metrics."""

  total_loss: metrics.Average.from_output("total_loss")
  # disp
  disp_layer_loss: metrics.Average.from_output("disp_layer_loss")
  disp_layer_alpha: metrics.Average.from_output("disp_layer_alpha")
  disp_smooth_loss: metrics.Average.from_output("disp_smooth_loss")

  shadow_smooth_loss: metrics.Average.from_output("shadow_smooth_loss")
  fg_alpha_reg_l0_loss: metrics.Average.from_output("fg_alpha_reg_l0_loss")
  fg_alpha_reg_l0_alpha: metrics.Average.from_output("fg_alpha_reg_l0_alpha")
  fg_alpha_reg_l1_loss: metrics.Average.from_output("fg_alpha_reg_l1_loss")
  fg_alpha_reg_l1_alpha: metrics.Average.from_output("fg_alpha_reg_l1_alpha")
  fg_mask_loss: metrics.Average.from_output("fg_mask_loss")
  fg_mask_alpha: metrics.Average.from_output("fg_mask_alpha")

  # ---------------------------------------------------------------------
  # src reconstruction for rgb.
  # ---------------------------------------------------------------------
  src_rgb_recon_loss: metrics.Average.from_output("src_rgb_recon_loss")
  src_rgb_recon_alpha: metrics.Average.from_output("src_rgb_recon_alpha")
  # ---------------------------------------------------------------------
  # projection.
  # ---------------------------------------------------------------------
  proj_far_rgb_loss: metrics.Average.from_output("proj_far_rgb_loss")
  proj_far_rgb_alpha: metrics.Average.from_output("proj_far_rgb_alpha")
  # misc.
  learining_rate: metrics.LastValue.from_output("learning_rate")
  weight_l2: metrics.Average.from_output("weight_l2")


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  eval_src_rgb_recon_loss: metrics.Average.from_output("src_rgb_recon_loss")
