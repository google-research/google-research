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

"""Transformer for Layout Trainer."""

# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order

import sys
sys.path.append("..")

import functools
from typing import Any, Dict, Optional, Tuple

from absl import logging
from clu import metrics
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
from flax.training import common_utils
import jax
import jax.numpy as jnp
from layout_blt import input_pipeline
from layout_blt.nets import transformer
from layout_blt.trainers import base_trainer
from layout_blt.utils import layout_fast_decode
from layout_blt.utils import task_manager
import numpy as np


@flax.struct.dataclass
class TrainState:
  """Data structure for checkpoint the model."""
  step: int
  optimizer: flax.optim.Optimizer
  model_state: Optional[Any]


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """Metrics during training process."""
  loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  """Metrics during evaluation process."""
  eval_loss: metrics.Average.from_output("eval_loss")


class TransformerTrainer(base_trainer.LayoutBaseTrainer):
  """Transformer for Layout Trainer."""

  def preprocess_batch(self, batch, batch_size, dataset, use_vertical=False):
    label = None
    # When we reach to the end of the dataset iter, the batch size may not be
    # be our expected one. In the case, we simply skip it.
    if batch.shape[0] != batch_size:
      return None, None
    batch = common_utils.shard(batch)
    return batch, label

  def create_train_state(
      self,
      rng,
      inputs,
  ):
    model = functools.partial(
        transformer.TransformerDecoder, config=self.config)
    param_rng, latent_rng = jax.random.split(rng, 2)
    model_variables = model(deterministic=True).init(param_rng,
                                                     inputs["inputs"],
                                                     inputs["labels"],
                                                     latent_rng)

    model_state = dict(model_variables)
    model_params = model_state.pop("params")
    logging.info("logging model parameters")
    parameter_overview.log_parameter_overview(model_params)
    optimizer = self.create_optimizer().create(model_params)
    model_dict = dict(model=model)
    train_state = TrainState(
        step=0,
        optimizer=optimizer,
        model_state=model_state)
    return model_dict, train_state

  def train_step(
      self,
      rng,
      state,
      batch,
      label,
      learning_rate_fn,
      model_dict,
      logits_mask
  ):
    """Perform a single training step.

    Args:
      rng: The random seed,
      state: State of the model (optimizer and state).
      batch: Training inputs for this step.
      label: Training input vectical info (always None for now).
      learning_rate_fn: The learning scheduler.
      model_dict: The model used in training.
      logits_mask: Logits mask for each step.

    Returns:
      The new model state and dictionary with metrics
    """
    logging.info("train_step(batch=%s)", batch)
    step = state.step + 1
    lr = learning_rate_fn(state.step)
    model = model_dict["model"]
    dec_target = batch[:, 1:]
    if logits_mask is not None:
      logits_mask = logits_mask[:, :-1, :]
    pad_mask = jnp.where(dec_target > 0, 1, 0).astype(jnp.float32)
    def loss_fn(params):
      dropout_rng, latent_rng = jax.random.split(rng)
      variables = {"params": params}
      variables.update(state.model_state)
      (logits, _), new_variables = model().apply(
          {"params": params},
          batch,
          label,
          latent_rng,
          rngs={"dropout": dropout_rng},
          mutable=True)
      recon_loss, num_tokens = self.compute_weighted_cross_entropy(
          logits, dec_target, pad_mask, self.config.label_smoothing,
          logits_mask)
      recon_loss = recon_loss / num_tokens
      loss = recon_loss
      new_model_state = dict(new_variables)
      new_model_state.pop("params")
      return loss, (recon_loss, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon_loss,
            new_model_state)), grad = grad_fn(state.optimizer.target)
    del recon_loss
    grad = jax.lax.pmean(grad, "batch")
    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_state = state.replace(
        step=step, optimizer=new_optimizer, model_state=new_model_state)
    metrics_update = TrainMetrics.gather_from_model_output(loss=loss)
    return new_state, metrics_update

  def eval_step(self, rng, state, batch, label, model_dict, logits_mask):
    model = model_dict["model"]
    dec_target = batch[:, 1:]
    logits_mask = logits_mask[:, :-1, :]
    pad_mask = jnp.where(dec_target > 0, 1, 0).astype(jnp.float32)
    (logits, _) = model(deterministic=True).apply(
        {"params": state.optimizer.target},
        batch,
        label,
        rng)
    recon_loss, num_tokens = self.compute_weighted_cross_entropy(
        logits, dec_target, pad_mask, self.config.label_smoothing, logits_mask)
    recon_loss = recon_loss / num_tokens
    loss = recon_loss
    metrics_update = EvalMetrics.gather_from_model_output(eval_loss=loss)
    return metrics_update

  def test(self,
           sampling_method="topp",
           conditional="none",
           eos_id=2,
           batch_size=1,
           sample_step_num=1,
           max_decode_len=128,
           use_vertical=False,
           vertical_idx=0):
    """Runs a test run.

    Args:
      sampling_method: str: how to generate the current token.
      conditional: str: none: uncondtional generation, a: asset condtional
        generation, a+s: asset + size condtional generation.
      eos_id: int: the index of eos symbol.
      batch_size: int: batch size of generation at one time.
      sample_step_num: int: how many batches to generate.
      max_decode_len: int: the maximum number of tokens during generation.
      use_vertical: bool: whether use vertical information (always False).
      vertical_idx: int: vertical index.
    Returns:
      generated_samples: [sample_step_num*batch_size, max_decode_len]:
        generated layouts.
      real_samples: [sample_step_num*batch_size, max_decode_len]: real layouts.
    """
    assert batch_size % jax.local_device_count() == 0
    rng = jax.random.PRNGKey(self.config.seed)
    # Make sure each host uses a different RNG.
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, model_rng, data_rng = jax.random.split(rng, 3)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    dataset = self.config.dataset

    test_ds, vocab_size, pos_info = input_pipeline.get_dataset(
        batch_size,
        self.config.dataset_path,
        jax.local_device_count(),
        "test.json",
        max_length=max_decode_len,
        dataset_name=dataset)

    init_batch = jnp.ones((batch_size, self.config.max_length))
    init_label = jnp.ones((batch_size, 1))
    init_batch = dict(inputs=init_batch, labels=init_label)
    model_dict, state = self.create_train_state(model_rng, init_batch)
    ckpt_path = self.config.test_checkpoint_dir
    state = task_manager.restore_checkpoint(state, ckpt_path)
    state = flax_utils.replicate(state)
    sample_one_batch_fn = functools.partial(
        self.sample_one_batch,
        pos_info=pos_info,
        batch_size=batch_size//jax.local_device_count(),
        conditional=conditional,
        eos_id=eos_id,
        max_decode_len=max_decode_len,
        sampling_method=sampling_method)
    p_generate_batch = jax.pmap(
        functools.partial(
            sample_one_batch_fn,
            model_dict=model_dict,
        ),
        axis_name="batch")

    test_iter = iter(test_ds)  # pytype: disable=wrong-arg-types
    def tohost(x):
      """Collect batches from all devices to host and flatten batch dimensions."""
      n_device, n_batch, *remaining_dims = x.shape
      return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))

    _, sample_offset = self.make_mask(vocab_size, pos_info, max_decode_len,
                                      self.layout_dim)
    generated_sample_list, real_sample_list = [], []
    for idx, test_batch in enumerate(test_iter):
      if idx == sample_step_num:
        break
      rng, sample_rng = jax.random.split(rng, 2)
      p_rng = jax.random.split(sample_rng, jax.local_device_count())
      test_batch = jax.tree.map(lambda x: x._numpy(), test_batch)  # pylint: disable=protected-access
      test_batch, test_label = self.preprocess_batch(test_batch, batch_size,
                                                     dataset, use_vertical)
      # For uncondtional generation, we stop the process according to the
      # sampel_step_num, otherwise, we use the whole test set.
      if test_batch is None or (conditional == "none" and
                                idx == sample_step_num):
        break

      if conditional == "none":
        if use_vertical:
          test_label = jnp.full_like(test_label, vertical_idx)
        sample_layouts = p_generate_batch(None, p_rng, state, label=test_label)
      else:
        sample_layouts = p_generate_batch(test_batch[Ellipsis, 1:], p_rng, state,
                                          label=test_label)
      # We do not need bos symbol.
      sample_layouts = tohost(sample_layouts)[Ellipsis, 1:]
      real_layouts = None
      if test_batch is not None:
        real_layouts = tohost(test_batch)[Ellipsis, 1:]
        _, real_offset = self.make_mask(self.config.vocab_size, pos_info,
                                        real_layouts.shape[-1], self.layout_dim)
        real_layouts = real_layouts - real_offset
      generated_sample_list.append(sample_layouts - sample_offset[Ellipsis, :-1])
      real_sample_list.append(real_layouts)
    generated_samples = jnp.concatenate(generated_sample_list, axis=0)
    real_samples = jnp.concatenate(real_sample_list, axis=0)

    return generated_samples, real_samples

  def sample_step(self, rng, state, model_dict, pos_info):
    """Samples layouts just for visualization during training."""
    # TODO(xiang): sample some images during training.
    return None

  def fast_decode(self,
                  rng,
                  variables,
                  model,
                  pos_info,
                  label=None,
                  batch=None,
                  batch_size=1,
                  conditional="none",
                  eos_id=2,
                  max_decode_len=100,
                  sampling_method="topp"):
    """Fast layout generation deocoding method.

    Args:
      rng: jax random state.
      variables: model parameters.
      model: layouu generation model.
      pos_info: vocabulary segmentation infomation.
      label: vertical information (always None for now).
      batch: real layouts batch for conditional generation.
      batch_size: number of layouts to generate one time.
      conditional: conditioanl type.
      eos_id: index of eos symbol.
      max_decode_len: maximum number of tokens to generate.
      sampling_method: sampling method during generation (argmax or sampling).
    Returns:
      seqs: generated layouts.
    """
    eval_model = model(deterministic=True, is_train=False)
    init_rng, rng, latent_rng = jax.random.split(rng, 3)
    init_batch = jnp.ones((batch_size, max_decode_len))
    init_label = jnp.ones((batch_size, 1))
    initial_vars = eval_model.init(init_rng, init_batch, init_label, latent_rng)
    cache_dict, _ = initial_vars.pop("params")

    def tokens_to_logits(xi, cache_dict, decode_step, initial_z):
      logits, cache_dict = eval_model.apply(
          {
              **variables,
              **cache_dict
          },
          xi,
          label,
          initial_z,
          decode_step,
          mutable=["cache"],
          method=transformer.TransformerDecoder.decode)
      return logits, cache_dict

    logit_masks, _ = self.make_mask(self.config.vocab_size, pos_info,
                                    self.total_dim, self.layout_dim)
    # BOS symbol.
    initial_z = jax.random.normal(rng, (batch_size, eval_model.config.emb_dim))
    tokens_to_logits_fn = functools.partial(
        tokens_to_logits, initial_z=initial_z)
    batch = init_batch if batch is None else batch

    seqs = layout_fast_decode.decode(
        batch,
        cache_dict,
        tokens_to_logits_fn,
        max_decode_len=max_decode_len,
        sampling_method=sampling_method,
        rng=rng,
        logit_masks=logit_masks,
        conditional=conditional)
    return seqs

  def sample_one_batch(self, batch, rng, state, model_dict, pos_info, label,
                       batch_size, conditional, eos_id, max_decode_len,
                       sampling_method):
    """Samples one batch for eval."""
    model = model_dict["model"]
    variables = {"params": state.optimizer.target}
    variables.update(state.model_state)

    x = self.fast_decode(rng, variables, model, pos_info, label, batch,
                         batch_size, conditional, eos_id, max_decode_len,
                         sampling_method)
    return x
