# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Adds DP via jax_privacy to tunix's PeftTrainer."""

import dataclasses

from absl import logging
from flax import nnx
import jax
from jax_privacy import clipping
from jax_privacy import noise_addition
import optax
import tunix
from typing_extensions import override


class DPPeftTrainer(tunix.sft.peft_trainer.PeftTrainer):
  """PeftTrainer with DP.

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use. To monitor the learning rate at each step,
      use `optax.schedules.inject_hyperparams` to inject learning rate as a
      hyperparameter. For example: ``optimizer =
      optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)``
    training_config: The training config.
    clip_norm: The clip norm to use for DP.
    noise_multiplier: The noise multiplier to use for DP (not accounting for the
      sensitivity of the clipped gradient function).
    microbatch_size: The microbatch size to use for DP. Smaller microbatches
      require less memory, but more sequential computation. So ideally, this
      should be set as large as possible without causing memory errors.
    expected_batch_size: The expected batch size (or an approximation thereof),
      used to normalize the sum-of-clipped-gradients before adding noise.
    clip_axis_name: The name of the mesh axis over which the batch is sharded
      for per-example gradient clipping. Passed as `spmd_axis_name` to
      `jax.vmap` within `clipping.clipped_fun`. If None, no SPMD axis is used.
  """

  def __init__(
      self,
      model,
      optimizer,
      training_config,
      *,
      clip_norm,
      noise_multiplier,
      microbatch_size = 64,
      seed,
      expected_batch_size = None,
      clip_axis_name = None,
  ):
    # Pass a no-checkpoint config to super().__init__ so that the parent does
    # not call maybe_restore against the plain optimizer's pytree structure.
    # The real checkpoint manager is reinstated below, after self.optimizer is
    # set to the noisy_optimizer, so that maybe_restore is called exactly once
    # with the correct optimizer structure.
    no_ckpt_config = dataclasses.replace(
        training_config, checkpoint_root_directory=None
    )
    super().__init__(model, optimizer, no_ckpt_config)
    self._clip_norm = clip_norm
    self.noise_multiplier = noise_multiplier
    self._seed = seed
    self.expected_batch_size = expected_batch_size
    self._clip_axis_name = clip_axis_name
    if self._lora_enabled:
      wrt = nnx.LoRAParam
    else:
      wrt = nnx.Param
    self._microbatch_size = microbatch_size

    _, privatizer = self._get_dp_components(nnx.split(model)[0])
    noisy_optimizer = optax.chain(privatizer, optimizer)
    self.optimizer = nnx.Optimizer(self.model, noisy_optimizer, wrt=wrt)

    # Reinstantiate the checkpoint manager with the real config now that
    # self.optimizer has the correct (noisy) structure.
    self.checkpoint_manager = tunix.CheckpointManager(
        root_directory=training_config.checkpoint_root_directory,
        options=training_config.checkpointing_options,
    )
    try:
      self._train_steps, self._restored_custom_metadata = (
          self.checkpoint_manager.maybe_restore(
              self.model,
              self.optimizer,
              restore_only_lora_params=self._lora_enabled,
          )
      )
    except Exception as e:
      logging.exception(
          "Failed to restore from checkpoint: %s", e, exc_info=True
      )
      raise e

  def _get_dp_components(self, graphdef):
    """Returns the clipped grad function and privatizer."""

    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )

    # wrapped_grad_fn has two purposes: It gives the grads as a function
    # of the params, rather than the whole model, which makes it 'jax-like',
    # and clipped_fun expects the grad to be the first output, but
    # nnx.value_and_grad returns the value + aux first, so we swap the outputs.
    def wrapped_grad_fn(s, i):
      out, grads = grad_fn(nnx.merge(graphdef, s), **i)
      return grads, out

    clipped_grad_fn = clipping.clipped_fun(
        fun=wrapped_grad_fn,
        has_aux=True,
        l2_clip_norm=self._clip_norm,
        batch_argnums=1,
        microbatch_size=self._microbatch_size,
        dtype=jax.numpy.float32,
        normalize_by=self.expected_batch_size or 1,
        spmd_axis_name=self._clip_axis_name,
    )
    privatizer = noise_addition.gaussian_privatizer(
        stddev=self.noise_multiplier * clipped_grad_fn.sensitivity(),
        dtype=jax.numpy.float32,
        prng_key=self._seed,
    )
    return clipped_grad_fn, privatizer

  @override
  def _train_step(self, model, optimizer, inputs):
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The nnx Optimizer that adds noise and updates the model.
      inputs: The training input.

    Returns:
      The loss and auxiliary data if has_aux is True, otherwise the loss.
    """
    batch_size = jax.tree.leaves(inputs)[0].shape[0]
    logging.info("Compiling train_step for batch size: %s", batch_size)
    inputs = self.gen_model_input_fn(inputs)

    graphdef, state = nnx.split(model)
    clipped_grad_fn, _ = self._get_dp_components(graphdef)

    grads, out = clipped_grad_fn(state, inputs)
    # JAX Privacy returns aux on a per-example basis.
    logging.info("Aux structure: %s", jax.tree.structure(out))
    out = jax.tree.map(lambda x: x.mean(axis=0), out)
    grad_norm = optax.tree.norm(grads)

    optimizer.update(model, grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux, grad_norm
    else:
      loss = out
      return loss, None, grad_norm
