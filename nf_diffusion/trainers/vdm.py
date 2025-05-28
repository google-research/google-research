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

"""Methods for training Variational Diffusion Model."""

import copy
import functools
from typing import Any, Dict, Optional, Tuple, Union

from absl import logging
from clu import metrics
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

import nf_diffusion.models.vdm as vdm_model
from nf_diffusion.trainers.utils import vdm_utils


# from clu import metric_writers
def get_learning_rate_scheduler(config, data_info):
  """Get Learning rate schedule."""
  cfg_opt = config.opt
  num_steps = int(data_info.num_train_steps)
  warmup_ratio = cfg_opt.get("warmup_ratio", 0.1)
  warmup_iters = int(num_steps * warmup_ratio)
  learning_rate = config.opt.get("learning_rate", 2e-4)

  if config.opt.get("lr_decay", True):
    end_value = config.opt.get("end_lr", 0.0)
  else:
    end_value = learning_rate
  schedule_fn = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=learning_rate,
      warmup_steps=warmup_iters,
      decay_steps=num_steps - warmup_iters,
      end_value=end_value,
  )
  return schedule_fn


def get_opt(
    config, data_info, opt_cfg_key=None
):
  """Get an optax optimizer factory."""
  cfgopt = config.opt
  if opt_cfg_key is not None:
    cfgopt = config.opt.get(opt_cfg_key)

  lr_schedule = get_learning_rate_scheduler(config, data_info)

  def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params)
    )
    flat_mask = {  # pylint: disable=g-complex-comprehension
        path: path[-1] != "bias" and path[-2:] not in [
            ("layer_norm", "scale"),
            ("final_layer_norm", "scale"),
        ]
        for path in flat_params
    }
    return flax.core.frozen_dict.FrozenDict(
        flax.traverse_util.unflatten_dict(flat_mask)
    )

  if cfgopt.name == "adamw":
    optimizer = optax.adamw(
        learning_rate=lr_schedule, mask=decay_mask_fn, **cfgopt.args
    )
    if cfgopt.get("gradient_clip_norm", -1) > 0:
      clip = optax.clip_by_global_norm(cfgopt.gradient_clip_norm)
      optimizer = optax.chain(clip, optimizer)
  elif cfgopt.name == "dummy":
    optimizer = optax.sgd(0.0)
  else:
    raise Exception("Unknow optimizer.")

  return optimizer


class TrainState(flax.struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer."""

  step: int
  params: Union[flax.core.frozen_dict.FrozenDict[str, Any], Dict[str, Any]]
  ema_params: Union[flax.core.frozen_dict.FrozenDict[str, Any], Dict[str, Any]]
  opt_state: optax.OptState

  @classmethod
  def create(cls, *, params, opt_state, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    ema_params = copy.deepcopy(params)
    return cls(
        step=0,
        params=params,
        ema_params=ema_params,
        opt_state=opt_state,
        **kwargs,
    )


def ema_apply_gradients(
    state, opt, grads, ema_rate, **kwargs
):
  """EMD Gradients Update."""
  updates, new_opt_state = opt.update(grads, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)
  new_ema_params = jax.tree.map(
      lambda x, y: x + (1.0 - ema_rate) * (y - x),
      state.ema_params,
      new_params,
  )

  return state.replace(
      step=state.step + 1,
      params=new_params,
      ema_params=new_ema_params,
      opt_state=new_opt_state,
      **kwargs,
  )


def create_train_state(
    config, rng, data_info
):
  """Create train state."""
  logging.warning("=== Initializing model ===")
  model = vdm_model.VDM(vdm_model.VDMConfig(**config.model))
  resolution = config.data.get("resolution", 32)
  channels = config.data.get("channels", 1)
  batch_size = config.data.per_device_batch_size
  inputs = {
      "images": jnp.zeros(
          (batch_size, resolution, resolution, channels), "float32"
      ),
      "conditioning": jnp.zeros((batch_size,)),
  }
  rng1, rng2 = jax.random.split(rng)
  variables = model.init({"params": rng1, "sample": rng2}, **inputs)
  parameter_overview.log_parameter_overview(variables)

  logging.info("=== Initializing train state ===")
  opt = get_opt(config, data_info)
  params = variables["params"]
  opt_state = opt.init(params)
  train_state = TrainState.create(params=params, opt_state=opt_state)

  return model, opt, train_state


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")

  # From loss function
  bpd: metrics.LastValue.from_output("bpd")
  bpd_latent: metrics.LastValue.from_output("bpd_latent")
  bpd_recon: metrics.LastValue.from_output("bpd_recon")
  bpd_diff: metrics.LastValue.from_output("bpd_diff")
  var: metrics.LastValue.from_output("var")
  var0: metrics.LastValue.from_output("var0")


def make_loss_fn(model):
  """Make VDM loss function."""

  def loss_fn(params, batch, rng, is_train):
    inputs = {"images": batch["image"], "conditioning": batch["label"]}
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if is_train:
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng

    # sample time steps, with antithetic sampling
    outputs = model.apply(
        variables={"params": params},
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
    )

    rescale_to_bpd = 1.0 / (np.prod(inputs["images"].shape[1:]) * np.log(2.0))
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff
    scalar_dict = {
        "bpd": bpd,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
    }
    img_dict = {"inputs": inputs["images"]}
    output = {"scalars": scalar_dict, "images": img_dict}

    return bpd, output

  return loss_fn


def train_step(
    config,
    model,
    state,
    opt,
    # opt,
    learning_rate_fn,
    batch,
    rng,
):
  """Train step."""
  rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
  rng = jax.random.fold_in(rng, state.step)

  curr_loss_fn = make_loss_fn(model)
  grad_fn = jax.value_and_grad(curr_loss_fn, has_aux=True)
  (loss_val, metrics_dict), grads = grad_fn(
      state.params, batch, rng=rng, is_train=True
  )
  grads = jax.lax.pmean(grads, "batch")
  new_state = ema_apply_gradients(
      state, opt=opt, grads=grads, ema_rate=config.opt.ema_rate
  )

  metrics_dict["scalars"] = jax.tree.map(
      lambda x: jax.lax.pmean(x, axis_name="batch"), metrics_dict["scalars"]
  )

  lr = learning_rate_fn(state.step)
  if config.get("multi"):
    metrics_update = TrainMetrics.gather_from_model_output(
        loss=loss_val, learning_rate=lr, **metrics_dict["scalars"]
    )
  else:
    metrics_update = TrainMetrics.single_from_model_output(
        loss=loss_val, learning_rate=lr, **metrics_dict["scalars"]
    )

  train_info = {f"image/{k}": v for k, v in metrics_dict["images"].items()}
  return new_state, metrics_update, train_info


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  # From loss function
  bpd: metrics.LastValue.from_output("bpd")
  bpd_latent: metrics.LastValue.from_output("bpd_latent")
  bpd_recon: metrics.LastValue.from_output("bpd_recon")
  bpd_diff: metrics.LastValue.from_output("bpd_diff")
  var: metrics.LastValue.from_output("var")
  var0: metrics.LastValue.from_output("var0")


def make_eval_step(loss_fn):
  """Function that return the eval step."""

  def eval_step_fn(base_rng, params, batch, eval_step=0):
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index("batch"))
    rng = jax.random.fold_in(rng, eval_step)

    _, metrics_dict = loss_fn(params, batch, rng=rng, is_train=False)

    # summarize metrics
    metrics_dict["scalars"] = jax.lax.pmean(
        metrics_dict["scalars"], axis_name="batch"
    )
    return metrics_dict

  return eval_step_fn


def evaluate(
    config,
    model,
    state,
    eval_ds,
    rng,
    num_eval_steps = -1,
):
  """Perform one evaluation."""
  logging.info("=== Experiment.evaluate() ===")
  assert config.get("multi")

  # TODO(guandao)
  eval_rng, sample_rng = jax.random.split(rng, 2)
  loss_fn = make_loss_fn(model)
  p_eval_step = functools.partial(make_eval_step(loss_fn), eval_rng)
  p_eval_step = jax.pmap(p_eval_step, "batch")
  pparams = state.params  # This is already replicated outside

  # Evaluating test time losses
  logging.info("=== Eval set loss ===")
  eval_metrics = []
  for curr_eval_step, batch in enumerate(eval_ds):
    batch = jax.tree.map(jnp.asarray, batch)
    metrics_dict = p_eval_step(
        pparams, batch, flax_utils.replicate(curr_eval_step)
    )
    eval_metrics.append(metrics_dict["scalars"])
    if curr_eval_step >= num_eval_steps:
      break

  # average over eval metrics
  eval_metrics = vdm_utils.get_metrics(eval_metrics)
  eval_metrics = jax.tree.map(jnp.mean, eval_metrics)
  eval_metrics = EvalMetrics.single_from_model_output(**eval_metrics)

  # sample a batch of images
  logging.info("=== Sample ===")
  p_sample = functools.partial(
      vdm_utils.make_sample_fn(model),
      dummy_inputs=next(iter(eval_ds))["image"][0],
      rng=sample_rng,
  )
  p_sample = vdm_utils.dist(p_sample, accumulate="concat", axis_name="batch")
  samples = p_sample(params=state.params)
  samples = vdm_utils.generate_image_grids(samples)[None, :, :, :]
  eval_info = {"samples": samples.astype(np.uint8)}

  return eval_metrics, eval_info


def eval_visualize(
    unused_config,
    # writer: metric_writers.MetricWriter,
    writer,
    step,
    unused_model,
    unused_state,
    eval_info,
    unused_eval_dir,
):
  writer.write_images(step, eval_info)


def train_visualize(
    config,
    # writer: metric_writers.MetricWriter,
    writer,
    step,
    unused_model,
    unused_state,
    train_info,
    unused_workdir,
):
  """Trainer visualization routine."""
  train_info_img = {
      "train/{0}".format(key): val
      for key, val in train_info.items()
      if key.startswith("image")
  }
  if config.get("multi"):
    train_info_img = {
        key: val.reshape(-1, *val.shape[2:])
        for key, val in train_info_img.items()
    }
  writer.write_images(step, train_info_img)
  writer.write_histograms(step, train_info_img)
