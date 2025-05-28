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

"""Training scripts."""
import enum
import functools
import multiprocessing
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Union, Optional

from absl import logging
from clu import metric_writers
import flax
from flax import struct
from flax.core import frozen_dict
from flax.core.scope import DenyList
from flax.linen import initializers
import gin
import jax
from jax.experimental import maps
from jax.experimental import pjit
import jax.numpy as jnp
import numpy
import optax
import tensorflow.compat.v2 as tf
from vmoe import partitioning
from vmoe import utils
from vmoe.checkpoints import base as checkpoints_base
from vmoe.checkpoints import partitioned as checkpoints_partitioned
from vmoe.data import pjit_utils
from vmoe.train import schedule
from vmoe.train import trainer

from moe_mtl.main import evaluator


gin.external_configurable(optax.sgd)
gin.external_configurable(optax.adamw)
gin.external_configurable(initializers.lecun_uniform, "lecun_uniform")
gin.external_configurable(initializers.lecun_normal, "lecun_normal")
gin.external_configurable(initializers.he_uniform, "he_uniform")
gin.external_configurable(initializers.he_normal, "he_normal")
gin.external_configurable(initializers.xavier_uniform, "xavier_uniform")
gin.external_configurable(initializers.xavier_normal, "xavier_normal")
gin.constant("jnp.bfloat16", jnp.bfloat16)
PRNGKey = Union[jax.numpy.ndarray, jax.Array]


class ExecutionMode(enum.Enum):
  """Defines the model execution mode."""
  TRAIN = 1
  EVAL = 2
  PREDICT = 3


@struct.dataclass
class TrainState:
  step: int
  optimizer: Any
  model_state: Any
  rngs: Dict[str, PRNGKey]
  ema_target: Any


@struct.dataclass
class TrainStateDynamic:
  step: int
  optimizer: Any
  model_state: Any
  rngs: Dict[str, PRNGKey]
  ema_target: Any
  current_k_det: int
  current_k_cls: int
  prev_best_val_loss_cls: float
  prev_best_val_loss_det: float


Mesh = partitioning.Mesh
PartitionSpec = partitioning.PartitionSpec

Array = jnp.ndarray
ArrayDict = Dict[str, Array]
LossArray = Union[Array, ArrayDict]
TrainStateAxisResources = TrainState
Batch = Union[Tuple[Any, Any], List[Any]]
FilterVars = Tuple[Tuple[str, str], Ellipsis]
Dataset = tf.data.Dataset
ThreadPool = multiprocessing.pool.ThreadPool
KwArgs = Mapping[str, Any]

"""
def save_file(workdir, name, data):
  f = tf.io.gfile.GFile(f"{workdir}/{name}", "w")
  f.write(pickle.dumps(data))
  f.flush()
  f.close()
"""


def get_random_bounding_box(image_shape,
                            ratio,
                            rng,
                            margin = 0.):
  """Returns a random bounding box for Cutmix.

  Based on the implementation in timm:
  https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py

  Args:
    image_shape: The shape of the image, specified as [height, width].
    ratio: Ratio of the input height/width to use as the maximum dimensions of
      the randomly sampled bounding box.
    rng: JAX rng key.
    margin: Percentage of bounding box dimension to enforce as the margin. This
      reduces the amount of the bounding box outside the image.

  Returns:
    The bounding box parameterised as y_min, y_max, x_min, x_max.
  """
  img_h, img_w = image_shape
  cut_h, cut_w = (img_h * ratio).astype(int), (img_w * ratio).astype(int)
  margin_y, margin_x = (margin * cut_h).astype(int), (margin *
                                                      cut_w).astype(int)
  rngx, rngy = jax.random.split(rng)
  cy = jax.random.randint(rngy, [1], 0 + margin_y, img_h - margin_y)
  cx = jax.random.randint(rngx, [1], 0 + margin_x, img_w - margin_x)

  y_min = jnp.clip(cy - cut_h // 2, 0, img_h)[0]
  y_max = jnp.clip(cy + cut_h // 2, 0, img_h)[0]
  x_min = jnp.clip(cx - cut_w // 2, 0, img_w)[0]
  x_max = jnp.clip(cx + cut_w // 2, 0, img_w)[0]
  return y_min, y_max, x_min, x_max


def wrap_mixup_and_cutmix(
    images,
    labels,
    mixup_rngs,
    mixup_alpha,
    cutmix_alpha,
    switch_prob=0.5,
    label_smoothing=0.0):
  """Performs Mixup or Cutmix within a single batch."""

  if labels.shape[-1] == 1:
    raise ValueError("Mixup requires one-hot targets.")

  batch_size = labels.shape[0]

  def perform_mixup(rng):
    _, mixup_weight_rng = jax.random.split(rng)
    weight = jax.random.beta(mixup_weight_rng, mixup_alpha, mixup_alpha)
    weight *= jnp.ones((batch_size, 1))

    # Mixup inputs.
    # Shape calculations use np to avoid device memory fragmentation:
    image_weight_shape = numpy.ones((images.ndim))
    image_weight_shape[0] = batch_size
    image_weight = jnp.reshape(weight, image_weight_shape.astype(numpy.int32))
    reverse = tuple(
        slice(images.shape[i]) if i > 0 else slice(-1, None, -1)
        for i in range(images.ndim))
    image = (image_weight * images + (1.0 - image_weight) * images[reverse])
    return image, weight

  def perform_cutmix(rng):
    rng, lambda_rng = jax.random.split(rng)
    cutmix_lambda = jax.random.beta(lambda_rng, cutmix_alpha, cutmix_alpha)

    rng, bounding_box_rng = jax.random.split(rng)
    y_min, y_max, x_min, x_max = get_random_bounding_box(
        images.shape[1:3], cutmix_lambda, bounding_box_rng)
    y_indices = jnp.arange(images.shape[1])[:, jnp.newaxis]
    x_indices = jnp.arange(images.shape[2])[jnp.newaxis, :]
    image_mask = jnp.logical_or(
        jnp.logical_or(y_indices < y_min, y_indices >= y_max),
        jnp.logical_or(x_indices < x_min, x_indices >= x_max))
    image_mask = image_mask[jnp.newaxis, :, :, jnp.newaxis].astype(images.dtype)

    output_image = (
        images * image_mask + jnp.flip(images, axis=0) * (1.0 - image_mask))
    box_area = (y_max - y_min) * (x_max - x_min)
    label_weight = 1.0 - box_area / float(images.shape[1] * images.shape[2])
    label_weight = jnp.broadcast_to(
        jnp.array(label_weight).reshape(1, 1), (images.shape[0], 1))
    return output_image, label_weight

  mixup_rngs, switch_rngs = jax.random.split(mixup_rngs)
  if cutmix_alpha > 0 and mixup_alpha <= 0:
    mixup_images, label_weight = perform_cutmix(mixup_rngs)
  elif mixup_alpha > 0 and cutmix_alpha <= 0:
    mixup_images, label_weight = perform_mixup(mixup_rngs)
  elif cutmix_alpha > 0 and mixup_alpha > 0:
    mixup_images, label_weight = jax.lax.cond(
        jax.random.uniform(switch_rngs) < switch_prob, perform_cutmix,
        perform_mixup, mixup_rngs)
  else:
    return images, labels

  # Mixup label
  if label_smoothing > 0:
    average = label_smoothing / labels.shape[-1]
    labels = labels * (1.0 - label_smoothing) + average

  mixup_labels = label_weight * labels + (1.0 - label_weight) * labels[::-1]

  return mixup_images, mixup_labels


@gin.configurable
def warmup_polynomial_decay_schedule_wrapper(
    peak_value,
    end_value,
    power,
    warmup_steps,
    decay_steps):
  """Linear warmup followed by polynomial decay."""
  return schedule.warmup_polynomial_decay_schedule(
      peak_value,
      end_value,
      power,
      warmup_steps,
      decay_steps
  )


@gin.configurable
def warmup_cosine_decay_schedule(
    peak_value,
    end_value,
    warmup_steps,
    decay_steps,
    offset_steps = 0):
  """Linear warmup followed by cosine decay."""
  return optax.join_schedules([
      optax.constant_schedule(0),
      optax.linear_schedule(
          init_value=0.0, end_value=peak_value, transition_steps=warmup_steps),
      optax.cosine_decay_schedule(
          init_value=peak_value,
          decay_steps=decay_steps - warmup_steps,
          alpha=end_value)
  ], [offset_steps, warmup_steps])


@gin.configurable
def my_create_learning_rate_scheduler(
    global_step,
    factors = "constant * linear_warmup * cosine_decay",
    base_learning_rate = 0.5,
    warmup_steps = 1000,
    decay_factor = 0.5,
    step_offset = 0,
    steps_per_decay = 20000,
    steps_per_cycle = 100000):
  """Creates learning rate schedule."""
  factors = [n.strip() for n in factors.split("*")]
  global_step = jnp.maximum(0, global_step - step_offset)
  global_step = jnp.array(global_step, jnp.float32)
  ret = 1.0
  for name in factors:
    if name == "constant":
      ret *= base_learning_rate
    elif name == "linear_warmup":
      ret *= jnp.minimum(1.0, global_step / warmup_steps)
    elif name == "rsqrt_decay":
      ret /= jnp.sqrt(jnp.maximum(global_step, warmup_steps))
    elif name == "rsqrt_normalized_decay":
      ret *= jnp.sqrt(warmup_steps)
      ret /= jnp.sqrt(jnp.maximum(global_step, warmup_steps))
    elif name == "decay_every":
      ret *= (decay_factor**(global_step // steps_per_decay))
    elif name == "linear_decay":
      progress = jnp.maximum(0.0, (global_step - warmup_steps) /
                             float(steps_per_cycle - warmup_steps))
      ret *= jnp.maximum(0., 1.0 - progress)
    elif name == "cosine_decay":
      progress = jnp.maximum(0.0, (global_step - warmup_steps) /
                             float(steps_per_cycle - warmup_steps))
      ret *= jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * progress)))
    else:
      raise ValueError("Unknown factor %s." % name)
  return jnp.asarray(ret, dtype=jnp.float32)


def create_input_generator(input_fn):  # pylint: disable=g-bare-generic
  """Prefetches data to local devices and returns a wrapped data generator."""
  dataset, as_numpy_dataset = input_fn()
  samples = (d for d in as_numpy_dataset)
  return dataset, next(samples)


def init_state(
    model_fn,
    input_sample_det,
    real_bs_det,
    input_sample_cls,
    real_bs_cls,
    optimizer_def,
    input_axis_resources_det,
    input_axis_resources_cls,
    mode,
    dynamic=False):
  """Initialize the TrainState."""
  def init_fn(rngs):
    if isinstance(input_sample_det, tuple):
      images_det, labels_det = input_sample_det
      sample_det = {"images_det": images_det, "labels_det": labels_det}
    else:
      sample_det = {}
      for key in input_sample_det:
        sample_det[key + "_det"] = input_sample_det[key]

    if isinstance(input_sample_cls, tuple):
      images_cls, labels_cls = input_sample_cls
      sample_cls = {"images_cls": images_cls, "labels_cls": labels_cls}
    else:
      sample_cls = {}
      for key in input_sample_cls:
        sample_cls[key + "_cls"] = input_sample_cls[key]

    sample_det = jax.tree.map(
        lambda x: jnp.zeros((real_bs_det, *x.shape[1:]), dtype=x.dtype),
        sample_det)
    sample_cls = jax.tree.map(
        lambda x: jnp.zeros((real_bs_cls, *x.shape[1:]), dtype=x.dtype),
        sample_cls)

    sample_det = partitioning.with_sharding_constraint(
        sample_det, input_axis_resources_det)
    sample_cls = partitioning.with_sharding_constraint(
        sample_cls, input_axis_resources_cls)

    for key in list(sample_cls.keys()):
      if not ("label" in key or "image" in key):
        del sample_cls[key]

    model_state = model_fn(mode=mode).init(  # pylint: disable=g-long-lambda
        rngs,
        _do_remap=True,
        **sample_det,
        **sample_cls)

    rngs = dict(**rngs)
    rngs.pop("params")
    model_state = dict(model_state)
    params = model_state.pop("params")
    model_state = frozen_dict.freeze(model_state)

    opt = optimizer_def.create(params)
    if not dynamic:
      return TrainState(
          model_state=model_state,
          optimizer=opt,
          step=0,
          rngs=rngs,
          ema_target=params)
    else:
      return TrainStateDynamic(
          model_state=model_state,
          optimizer=opt,
          step=0,
          rngs=rngs,
          ema_target=params,
          current_k_det=2,
          current_k_cls=2,
          prev_best_val_loss_cls=100.0,
          prev_best_val_loss_det=100.0,
      )

  return init_fn


def init_cls_state(
    model_fn,
    input_sample_cls,
    real_bs_cls,
    optimizer_def,
    input_axis_resources_cls,
    mode):
  """Init train states for classification."""
  def init_fn(rngs):
    if isinstance(input_sample_cls, tuple):
      images_cls, labels_cls = input_sample_cls
      sample_cls = {"images": images_cls, "labels": labels_cls}
    else:
      sample_cls = {}
      for key in input_sample_cls:
        sample_cls[key] = input_sample_cls[key]

    sample_cls = jax.tree.map(
        lambda x: jnp.zeros((real_bs_cls, *x.shape[1:]), dtype=x.dtype),
        sample_cls)

    sample_cls = partitioning.with_sharding_constraint(
        sample_cls, input_axis_resources_cls)

    if "_fake_cls" in sample_cls:
      del sample_cls["_fake_cls"]
    model_state = model_fn(mode=mode).init(  # pylint: disable=g-long-lambda
        rngs,
        _do_remap=True,
        **sample_cls)

    rngs = dict(**rngs)
    rngs.pop("params")
    model_state = dict(model_state)
    params = model_state.pop("params")
    model_state = frozen_dict.freeze(model_state)
    logging.info("model state keys: %s", model_state.keys())
    opt = optimizer_def.create(params)
    return TrainState(
        model_state=model_state,
        optimizer=opt,
        step=0,
        rngs=rngs,
        ema_target=params)

  return init_fn


def create_train_state(
    state_init_fn,
    rngs,
    mesh,
    axis_resources):
  """Create TrainState object."""
  mesh = mesh or maps.thread_resources.env.physical_mesh
  with jax.sharding.Mesh(mesh.devices, mesh.axis_names):
    train_state = pjit.pjit(
        state_init_fn, in_shardings=(None,), out_shardings=axis_resources
    )(rngs)
    return train_state


def make_dataset_iterator(dataset):
  """Returns an iterator over a TF Dataset."""

  def to_numpy(data):

    def trans(x):
      try:
        a = memoryview(x)
      except MemoryError:
        a = x
      return a

    return jax.tree.map(lambda x: jnp.asarray(trans(x)), data)

  ds_iter = iter(dataset)
  ds_iter = map(to_numpy, ds_iter)
  return ds_iter


@gin.configurable(denylist=["output_dir"])
def train_vmoe_mtl(
    output_dir,
    num_expert_partitions = gin.REQUIRED,
    input_fn_det = gin.REQUIRED,
    input_fn_cls = gin.REQUIRED):
  """Train MTL."""
  mesh = partitioning.get_auto_logical_mesh(num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)
  with mesh:
    _train_vmoe_mtl(
        output_dir, mesh, input_fn_det=input_fn_det, input_fn_cls=input_fn_cls)


def tree_shape_dtype_struct(tree):
  """Converts a PyTree with array-like objects to jax.ShapeDtypeStruct."""

  def fn(x):
    shape, dtype = x.shape, x.dtype
    # Useful to convert Tensorflow Tensors.
    dtype = dtype.as_numpy_dtype if hasattr(dtype, "as_numpy_dtype") else dtype
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)

  return jax.tree.map(fn, tree)


@gin.configurable(denylist=["output_dir", "mesh"])
def _train_vmoe_mtl(
    output_dir,
    mesh,
    input_fn_det = gin.REQUIRED,
    input_fn_cls = gin.REQUIRED,
    model_fn=gin.REQUIRED,
    optimizer_def = gin.REQUIRED,
    loss_fn = gin.REQUIRED,
    det_learning_rate = gin.REQUIRED,
    cls_learning_rate = gin.REQUIRED,
    total_train_steps = gin.REQUIRED,
    real_bs_det = gin.REQUIRED,
    real_bs_cls = gin.REQUIRED,
    rng_seed = 0,
    checkpoint_to_load = None,
    profile_config = None,
    progress_config = None,
    checkpoint_config = None,
    fixed_loss_ratio = None,
    cls_model_checkpoints = None,
    cls_model_fn = None,
    mixup_cutmix_config = None):
  """Train the model."""
  if not output_dir:
    raise ValueError("output_dir should be a non-empty path.")
  dataset_det, batch_det = create_input_generator(input_fn_det)
  dataset_cls, batch_cls = create_input_generator(input_fn_cls)

  if isinstance(batch_det, tuple):
    images_det, labels_det = batch_det
    sample_det = {"images_det": images_det, "labels_det": labels_det}
  else:
    sample_det = {}
    for key in batch_det:
      sample_det[key + "_det"] = batch_det[key]

  if isinstance(batch_cls, tuple):
    images_cls, labels_cls = batch_cls
    sample_cls = {"images_cls": images_cls, "labels_cls": labels_cls}
  else:
    sample_cls = {}
    for key in batch_cls:
      sample_cls[key + "_cls"] = batch_cls[key]
  def generate_partition_spec(_):
    return partitioning.parse_partition_spec((mesh.axis_names,))
  resources_det = jax.tree.map(
      generate_partition_spec, sample_det)
  resources_cls = jax.tree.map(
      generate_partition_spec, sample_cls)
  train_state_rngs = utils.make_rngs(("params", "gating_cls", "gating_det",
                                      "dropout_cls", "dropout_det", "rng",
                                      "gating", "dropout"),
                                     rng_seed)
  train_state_rngs, _ = utils.tree_rngs_split(train_state_rngs)
  state_init_fn = init_state(
      model_fn,
      batch_det,
      real_bs_det,
      batch_cls,
      real_bs_cls,
      optimizer_def,
      resources_det,
      resources_cls,
      mode=ExecutionMode.TRAIN)

  params_axis_resources = [("Moe/Mlp/.*", ("expert",))]
  train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=jax.eval_shape(state_init_fn, train_state_rngs),
      axis_resources_regexes=params_axis_resources)
  logging.info("Creating train state.")
  if cls_model_checkpoints is not None:
    cls_train_state_rngs = utils.make_rngs(
        ("params", "gating", "dropout", "rng"), rng_seed)
    resources_cls_load = jax.tree.map(
        generate_partition_spec, batch_cls)
    cls_state_init_fn = init_cls_state(
        cls_model_fn,
        batch_cls,
        real_bs_cls,
        optimizer_def,
        resources_cls_load,
        mode=ExecutionMode.TRAIN
    )
    cls_train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
        tree=jax.eval_shape(cls_state_init_fn, cls_train_state_rngs),
        axis_resources_regexes=params_axis_resources)

  if checkpoint_to_load is None:
    prefix = os.path.join(output_dir, "ckpt")
    train_state = restore_or_create_train_state(
        prefix=prefix,
        path=checkpoint_to_load,
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
    init_step = train_state.step
  else:
    train_state = restore_or_create_train_state(
        prefix="",
        path=checkpoint_to_load,
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
    init_step = 0

  if (cls_model_checkpoints is not None) and (init_step == 0):
    cls_train_state = restore_or_create_train_state(
        prefix="",
        path=cls_model_checkpoints,
        state_init_fn=cls_state_init_fn,
        axis_resources=cls_train_state_axis_resources,
        rngs=cls_train_state_rngs,
        thread_pool=ThreadPool())
    params_to_load = cls_train_state.ema_target
    params = train_state.optimizer.target
    is_frozen_dict = isinstance(params, flax.core.FrozenDict)
    if is_frozen_dict:
      params = flax.core.unfreeze(params)

    params_flat = flax.traverse_util.flatten_dict(params)
    ckpt_params = flax.core.unfreeze(params_to_load)
    ckpt_params_flat = flax.traverse_util.flatten_dict(ckpt_params)
    ckpt_params_flat = {"/".join(k): v for k, v in ckpt_params_flat.items()}
    ckpt_params_used = set()

    for key_tuple, value in params_flat.items():
      key = "/".join(key_tuple)
      logging.info(key)

      if "dense1" in key:
        ckpt_key = key.replace("dense1", "Dense_0")
      elif "dense2" in key:
        ckpt_key = key.replace("dense2", "Dense_1")
      elif "Router_cls" in key:
        ckpt_key = key.replace("Router_cls", "Router")
      elif "cls_cls" in key:
        ckpt_key = key.replace("cls_cls", "cls")
      elif "embedding_cls" in key:
        ckpt_key = key.replace("embedding_cls", "embedding")
      else:
        ckpt_key = key
      if "VisionTransformerMoeMTL_0" in ckpt_key:
        ckpt_key = ckpt_key.replace(
            "VisionTransformerMoeMTL_0", "VisionTransformerMoe_0")

      ckpt_params_used.add(ckpt_key)
      if ckpt_key in ckpt_params_flat:
        ckpt_value = ckpt_params_flat[ckpt_key]

        if value.shape == ckpt_value.shape:
          params_flat[key_tuple] = ckpt_value
        else:
          raise ValueError(f"Parameter {key!r} was mapped to {ckpt_key!r}, but "
                           f"their shapes are not equal: {value.shape} vs "
                           f"{ckpt_value.shape}.")
      else:
        logging.info("Missing %s", ckpt_key)
    params = flax.traverse_util.unflatten_dict(params_flat)

    if is_frozen_dict:
      freeze_params = flax.core.freeze(params)
    else:
      freeze_params = params
    train_state = train_state.replace(
        optimizer=optimizer_def.create(freeze_params))
    # train_state = train_state.replace(
    #    ema_target=train_state.optimizer.target
    # )
    del ckpt_params, ckpt_params_flat, params_flat, cls_state_init_fn, cls_train_state
    def sync_ema_and_weight(train_state):
      old_ema_target = train_state.ema_target
      optimizer = train_state.optimizer
      model_state = train_state.model_state
      rngs = train_state.rngs
      new_ema_target = jax.tree.map(lambda x, y: 1 * x + 0 * y,
                                    optimizer.target, old_ema_target)
      new_train_state = train_state.replace(  # pytype: disable=attribute-error
          step=init_step,
          optimizer=optimizer,
          model_state=model_state,
          ema_target=new_ema_target,
          rngs=rngs)
      return new_train_state

    sync_pjit = pjit.pjit(
        fun=sync_ema_and_weight,
        in_shardings=(train_state_axis_resources,),
        out_shardings=train_state_axis_resources,
    )
    train_state = sync_pjit(train_state)
    logging.info("sync complete")
  if init_step == total_train_steps:
    return

  def train_step(
      train_state,
      data_det,
      data_cls):
    """A single training step."""
    step = train_state.step
    optimizer = train_state.optimizer
    model_state = train_state.model_state
    det_lr = det_learning_rate(step) if callable(
        det_learning_rate) else det_learning_rate
    cls_lr = cls_learning_rate(step) if callable(
        cls_learning_rate) else cls_learning_rate
    # det_lr = 0
    rngs = train_state.rngs
    if mixup_cutmix_config is not None:
      mixup_rngs, rngs = utils.tree_rngs_split(rngs)
      mixup_rng = mixup_rngs["rng"]
      images = data_cls["images_cls"]
      labels = data_cls["labels_cls"]
      mixup_cutmix_image, mixup_cutmix_labels = wrap_mixup_and_cutmix(
          images,
          labels,
          mixup_rng,
          mixup_alpha=mixup_cutmix_config.get("mixup_alpha", 0),
          cutmix_alpha=mixup_cutmix_config.get("cutmix_alpha", 0))
      data_cls["images_cls"] = mixup_cutmix_image
      data_cls["labels_cls"] = mixup_cutmix_labels

    rngs, next_rngs = utils.tree_rngs_split(rngs)

    def loss_step(
        params,
        det_lr,
        cls_lr,
        base_lr):
      """Loss step that is used to compute the gradients."""

      variables = {"params": params}
      variables.update(model_state)

      outputs, new_variables = model_fn(mode=ExecutionMode.TRAIN).apply(
          variables,
          _do_remap=True,
          **data_det,
          **data_cls,
          mutable=DenyList("params"),
          rngs=rngs)
      metrics = outputs["metrics"]["metrics"]

      losses = loss_fn(
          outputs, **data_det, **data_cls, _do_remap=True, params=params)
      aux_output = {"new_model_state": new_variables, "losses": losses}
      if fixed_loss_ratio is None:
        det_lr_ratio = det_lr / (base_lr + 1e-8)  # calculate balance factor
        cls_lr_ratio = cls_lr / (base_lr + 1e-8)
      else:
        det_lr_ratio = fixed_loss_ratio
        cls_lr_ratio = 1
      model_loss = det_lr_ratio * losses["det_loss"] + cls_lr_ratio * losses[
          "cls_loss"]
      aux_output["lr"] = {}
      aux_output["lr"]["det_lr"] = det_lr
      aux_output["lr"]["cls_lr"] = cls_lr
      metrics["auxiliary_loss_det"] = jnp.mean(metrics["auxiliary_loss_det"])
      metrics["auxiliary_loss_cls"] = jnp.mean(metrics["auxiliary_loss_cls"])
      model_loss += det_lr_ratio * metrics["auxiliary_loss_det"]
      model_loss += cls_lr_ratio * metrics["auxiliary_loss_cls"]
      aux_output["losses"]["auxiliary_loss_cls"] = metrics["auxiliary_loss_cls"]
      aux_output["losses"]["auxiliary_loss_det"] = metrics["auxiliary_loss_det"]
      if "xent_loss_det" in metrics:
        metrics["xent_loss_det"] = jnp.mean(metrics["xent_loss_det"])
        metrics["xent_loss_cls"] = jnp.mean(metrics["xent_loss_cls"])
        aux_output["losses"]["xent_loss_det"] = metrics["xent_loss_det"]
        aux_output["losses"]["xent_loss_cls"] = metrics["xent_loss_cls"]
      aux_output["losses"]["model_loss"] = model_loss
      return model_loss, aux_output

    grad_fn = jax.value_and_grad(loss_step, has_aux=True)
    base_lr = cls_lr

    (_, aux_output), grads = grad_fn(optimizer.target, det_lr, cls_lr, base_lr)

    new_optimizer = optimizer.apply_gradient(grads, learning_rate=base_lr)

    new_model_state = aux_output["new_model_state"]
    losses = aux_output["losses"]
    lrs = aux_output["lr"]
    losses.update(lrs)
    old_ema_target = train_state.ema_target
    new_ema_target = jax.tree.map(lambda x, y: 0.1 * x + 0.9 * y,
                                  new_optimizer.target, old_ema_target)
    new_train_state = train_state.replace(  # pytype: disable=attribute-error
        step=step + 1,
        optimizer=new_optimizer,
        model_state=new_model_state,
        ema_target=new_ema_target,
        rngs=next_rngs)
    return new_train_state, losses

  train_step_pjit = pjit.pjit(
      fun=functools.partial(train_step),
      in_shardings=(
          train_state_axis_resources,  # train_state
          resources_det,  # images
          resources_cls,
      ),
      out_shardings=(
          train_state_axis_resources,  # train_state
          None,  # metrics
      ),
      donate_argnums=(0, 1, 2),
  )

  writer = metric_writers.create_default_writer(
      logdir=output_dir, just_logging=jax.process_index() > 0)
  profile_hook = trainer.create_profile_hook(
      workdir=output_dir, **profile_config)
  progress_hook = trainer.create_progress_hook(
      writer=writer,
      first_step=int(init_step) + 1,
      train_steps=total_train_steps,
      **progress_config)
  checkpoint_hook = trainer.create_checkpoint_hook(  # pytype: disable=module-attr
      workdir=output_dir,
      progress_hook=progress_hook,
      train_state_axis_resources=train_state_axis_resources,
      train_steps=total_train_steps,
      **checkpoint_config)
  # config_model_eval = config.model.copy_and_resolve_references()
  with metric_writers.ensure_flushes(writer):
    # Explicitly compile train_step here and report the compilation time.
    t0 = time.time()
    # logging.info(jax.tree.map(lambda x: x.shape, batch))
    train_step_pjit = train_step_pjit.lower(
        *tree_shape_dtype_struct((train_state, sample_det, sample_cls))
    ).compile()

    t1 = time.time()
    writer.write_scalars(init_step + 1, {"train/compile_secs": t1 - t0})
    # Create iterator over the train dataset.
    logging.info("create dataloader.")
    tr_iter_cls = pjit_utils.prefetch_to_device(
        iterator=make_dataset_iterator(dataset_cls),
        size=2,
        mesh=mesh)
    tr_iter_det = pjit_utils.prefetch_to_device(
        iterator=make_dataset_iterator(dataset_det),
        size=2,
        mesh=mesh)

    logging.info("before training.")
    for step, r_batch_det, r_batch_cls in zip(
        range(init_step + 1, total_train_steps + 1),
        tr_iter_det, tr_iter_cls):
      profile_hook(step)
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        if isinstance(r_batch_cls, tuple):
          images_cls, labels_cls = r_batch_cls
          r_input_batch_cls = {
              "images_cls": images_cls,
              "labels_cls": labels_cls
          }
        else:
          r_input_batch_cls = {}
          for key in batch_cls:
            r_input_batch_cls[key + "_cls"] = r_batch_cls[key]
        if isinstance(r_batch_det, tuple):
          images_det, labels_det = r_batch_det
          r_input_batch_det = {
              "images_det": images_det,
              "labels_det": labels_det
          }
        else:
          r_input_batch_det = {}
          for key in batch_det:
            r_input_batch_det[key + "_det"] = r_batch_det[key]

        train_state, metrics_value = train_step_pjit(train_state,
                                                     r_input_batch_det,
                                                     r_input_batch_cls)
        progress_hook(
            step,
            scalar_metrics={
                f"train/{k}": v for k, v in metrics_value.items()
            })
        checkpoint_hook(step, state=train_state)


def create_evaluation_hook(
    *,
    writer,
    progress_hook,
    input_fn_det,
    input_fn_cls,
    model_fn,
    metrics_fn,
    train_state_axis_resources,
    input_axis_resources_det,
    input_axis_resources_cls,
    rngs,
    use_ema = True,
    **kwargs):
  """Returns a hook to evaluate a model async."""

  # Always evaluate on the first and last step.
  # on_steps.update([int(first_step), int(train_steps)])
  return evaluator.EvaluateMultipleDatasetsMTL(
      model_fn=model_fn,
      input_fn_det=input_fn_det,
      input_fn_cls=input_fn_cls,
      metrics_fn=metrics_fn,
      train_state_axis_resources=train_state_axis_resources,
      input_axis_resources_det=input_axis_resources_det,
      input_axis_resources_cls=input_axis_resources_cls,
      metric_writer=writer,
      rngs=rngs,
      report_progress=progress_hook,
      report_progress_name="eval",
      use_ema=use_ema,
      **kwargs)


@gin.configurable(denylist=["output_dir"])
def evaluate_vmoe_mtl(
    output_dir,
    num_expert_partitions = gin.REQUIRED,
    input_fn_cls = gin.REQUIRED,
    input_fn_det = gin.REQUIRED):
  """Evaluate."""
  mesh = partitioning.get_auto_logical_mesh(num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)

  with mesh:
    _evaluate_vmoe_mtl(
        output_dir, mesh, input_fn_det=input_fn_det, input_fn_cls=input_fn_cls)


@gin.configurable(denylist=["output_dir"])
def evaluate_vmoe_mtl_with_aes(
    output_dir,
    num_expert_partitions = gin.REQUIRED,
    input_fn_cls = gin.REQUIRED,
    input_fn_det = gin.REQUIRED):
  """Evaluate."""
  mesh = partitioning.get_auto_logical_mesh(num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)

  with mesh:
    _evaluate_vmoe_mtl_with_aes(
        output_dir, mesh, input_fn_det=input_fn_det, input_fn_cls=input_fn_cls)


@gin.configurable(denylist=["output_dir", "mesh"])
def _evaluate_vmoe_mtl(
    output_dir,
    mesh,
    input_fn_det = gin.REQUIRED,
    input_fn_cls = gin.REQUIRED,
    model_fn=gin.REQUIRED,
    optimizer_def = gin.REQUIRED,
    total_train_steps = gin.REQUIRED,
    real_bs_det = gin.REQUIRED,
    real_bs_cls = gin.REQUIRED,
    eval_metrics=flax.core.FrozenDict({}),
    rng_seed = 0,
    use_ema = True,
    model_to_eval = None,
    model_to_eval_start = None,
    model_to_eval_end = None,
    model_to_eval_interval = None,
    model_to_eval_prefix = None,
    progress_config = None,
    evaluate_config = None):
  """Evaluate the model."""
  if not output_dir:
    raise ValueError("output_dir should be a non-empty path.")
  logging.info("Starting train function.")
  logging.info("Creating input generator.")
  _, batch_det = create_input_generator(input_fn_det)
  _, batch_cls = create_input_generator(input_fn_cls)
  logging.info("Finished creating input generator.")
  if isinstance(batch_det, tuple):
    images_det, labels_det = batch_det
    sample_det = {"images_det": images_det, "labels_det": labels_det}
  else:
    sample_det = {}
    for key in batch_det:
      sample_det[key + "_det"] = batch_det[key]

  if isinstance(batch_cls, tuple):
    images_cls, labels_cls = batch_cls
    sample_cls = {"images_cls": images_cls, "labels_cls": labels_cls}
  else:
    sample_cls = {}
    for key in batch_cls:
      sample_cls[key + "_cls"] = batch_cls[key]
  def generate_partition_spec(_):
    return partitioning.parse_partition_spec((mesh.axis_names,))
  input_axis_resources_cls = jax.tree.map(
      generate_partition_spec, sample_cls)
  input_axis_resources_det = jax.tree.map(
      generate_partition_spec, sample_det)

  logging.info("Building model.")
  train_state_rngs = utils.make_rngs(("params", "gating_cls", "gating_det",
                                      "dropout_cls", "dropout_det", "rng",
                                      "gating", "dropout"),
                                     rng_seed)
  train_state_rngs, eval_state_rngs = utils.tree_rngs_split(train_state_rngs)
  state_init_fn = init_state(
      model_fn,
      batch_det,
      real_bs_det,
      batch_cls,
      real_bs_cls,
      optimizer_def,
      input_axis_resources_det=input_axis_resources_det,
      input_axis_resources_cls=input_axis_resources_cls,
      mode=ExecutionMode.EVAL, dynamic=False)
  params_axis_resources = [("Moe/Mlp/.*", ("expert",))]
  train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=jax.eval_shape(state_init_fn, train_state_rngs),
      axis_resources_regexes=params_axis_resources)

  logging.info("Creating train state.")
  # separate out params and model state. This is needed for only computing
  # gradients w.r.t. the params.

  writer = metric_writers.create_default_writer(
      logdir=output_dir, just_logging=jax.process_index() > 0)
  progress_hook = trainer.create_progress_hook(
      writer=writer,
      first_step=1,
      train_steps=total_train_steps,
      **progress_config)
  if model_to_eval:
    train_state = restore_or_create_train_state(
        prefix="",
        path=model_to_eval,
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
  elif model_to_eval_start:
    train_state = restore_or_create_train_state(
        prefix="",
        path=f"{model_to_eval_prefix}/ckpt_{model_to_eval_start}",
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
    step = model_to_eval_start
  else:
    train_state = restore_or_create_train_state(
        prefix=os.path.join(output_dir, "ckpt"),
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs)
  init_step = int(train_state.step * 1)
  logging.info(init_step)
  logging.info(total_train_steps)
  evaluation_hook = create_evaluation_hook(
      writer=writer,
      progress_hook=progress_hook,
      input_fn_det=input_fn_det,
      input_fn_cls=input_fn_cls,
      model_fn=model_fn,
      metrics_fn=eval_metrics,
      train_state_axis_resources=train_state_axis_resources,
      input_axis_resources_det=input_axis_resources_det,
      input_axis_resources_cls=input_axis_resources_cls,
      rngs=eval_state_rngs,
      use_ema=use_ema,
      **evaluate_config)
  with metric_writers.ensure_flushes(writer):
    # Explicitly compile train_step here and report the compilation time.
    previous_step = 0
    while True:
      writer.write_scalars(1, {"eval/steps": init_step})
      if init_step > 0 and init_step != previous_step:
        # for i in range(previous_step + 1, init_step):
        # evaluation_hook(i, train_state=train_state)
        logging.info(init_step)
        evaluation_hook(init_step, t=None, train_state=train_state)
        previous_step = init_step
      else:
        time.sleep(30)
      train_state = restore_or_create_train_state(
          prefix=os.path.join(output_dir, "ckpt"),
          state_init_fn=state_init_fn,
          axis_resources=train_state_axis_resources,
          rngs=train_state_rngs,
          thread_pool=ThreadPool())
      init_step = train_state.step  # pytype: disable=attribute-error

      if model_to_eval:
        break
      elif model_to_eval_interval:
        step = int(step) + int(model_to_eval_interval)
        logging.info(step)
        if step > model_to_eval_end:
          break
        else:
          train_state = restore_or_create_train_state(
              path=f"{model_to_eval_prefix}/ckpt_{step}",
              prefix="",
              state_init_fn=state_init_fn,
              axis_resources=train_state_axis_resources,
              rngs=train_state_rngs,
              thread_pool=ThreadPool())
          init_step = train_state.step


@gin.configurable(denylist=["output_dir", "mesh"])
def _evaluate_vmoe_mtl_with_aes(
    output_dir,
    mesh,
    input_fn_det = gin.REQUIRED,
    input_fn_cls = gin.REQUIRED,
    model_fn=gin.REQUIRED,
    optimizer_def = gin.REQUIRED,
    total_train_steps = gin.REQUIRED,
    real_bs_det = gin.REQUIRED,
    real_bs_cls = gin.REQUIRED,
    eval_metrics=flax.core.FrozenDict({}),
    rng_seed = 0,
    use_ema = True,
    model_to_eval = None,
    model_to_eval_start = None,
    model_to_eval_end = None,
    model_to_eval_interval = None,
    model_to_eval_prefix = None,
    progress_config = None,
    evaluate_config = None):
  """Evaluate the model."""
  if not output_dir:
    raise ValueError("output_dir should be a non-empty path.")
  logging.info("Starting train function.")
  logging.info("Creating input generator.")
  _, batch_det = create_input_generator(input_fn_det)
  _, batch_cls = create_input_generator(input_fn_cls)
  logging.info("Finished creating input generator.")

  if isinstance(batch_det, tuple):
    images_det, labels_det = batch_det
    sample_det = {"images_det": images_det, "labels_det": labels_det}
  else:
    sample_det = {}
    for key in batch_det:
      sample_det[key + "_det"] = batch_det[key]

  if isinstance(batch_cls, tuple):
    images_cls, labels_cls = batch_cls
    sample_cls = {"images_cls": images_cls, "labels_cls": labels_cls}
  else:
    sample_cls = {}
    for key in batch_cls:
      sample_cls[key + "_cls"] = batch_cls[key]
  def generate_partition_spec(_):
    return partitioning.parse_partition_spec((mesh.axis_names,))
  input_axis_resources_cls = jax.tree.map(
      generate_partition_spec, sample_cls)
  input_axis_resources_det = jax.tree.map(
      generate_partition_spec, sample_det)

  logging.info("Building model.")
  train_state_rngs = utils.make_rngs(("params", "gating_cls", "gating_det",
                                      "dropout_cls", "dropout_det", "rng",
                                      "gating", "dropout"),
                                     rng_seed)
  train_state_rngs, eval_state_rngs = utils.tree_rngs_split(train_state_rngs)
  state_init_fn = init_state(
      model_fn,
      batch_det,
      real_bs_det,
      batch_cls,
      real_bs_cls,
      optimizer_def,
      input_axis_resources_det=input_axis_resources_det,
      input_axis_resources_cls=input_axis_resources_cls,
      mode=ExecutionMode.EVAL, dynamic=True)
  params_axis_resources = [("Moe/Mlp/.*", ("expert",))]
  train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=jax.eval_shape(state_init_fn, train_state_rngs),
      axis_resources_regexes=params_axis_resources)

  logging.info("Creating train state.")
  # separate out params and model state. This is needed for only computing
  # gradients w.r.t. the params.

  writer = metric_writers.create_default_writer(
      logdir=output_dir, just_logging=jax.process_index() > 0)
  progress_hook = trainer.create_progress_hook(
      writer=writer,
      first_step=1,
      train_steps=total_train_steps,
      **progress_config)
  if model_to_eval:
    train_state = restore_or_create_train_state(
        prefix="",
        path=model_to_eval,
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
  elif model_to_eval_start:
    train_state = restore_or_create_train_state(
        prefix="",
        path=f"{model_to_eval_prefix}/ckpt_{model_to_eval_start}",
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
    step = model_to_eval_start
  else:
    train_state = restore_or_create_train_state(
        prefix=os.path.join(output_dir, "ckpt"),
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs)
  init_step = int(train_state.step * 1)
  logging.info(init_step)
  logging.info(total_train_steps)
  encoder_config = gin.query_parameter(
      "get_uvit_vmoe_mtl_model_fn.encoder_config")
  encoder_config["moe"]["router"]["num_selected_experts_cls"] = int(
      train_state.current_k_cls)  # pytype: disable=attribute-error
  encoder_config["moe"]["router"]["num_selected_experts_det"] = int(
      train_state.current_k_det)  # pytype: disable=attribute-error
  with gin.unlock_config():
    gin.bind_parameter("get_uvit_vmoe_model_fn.encoder_config", encoder_config)

  evaluation_hook = create_evaluation_hook(
      writer=writer,
      progress_hook=progress_hook,
      input_fn_det=input_fn_det,
      input_fn_cls=input_fn_cls,
      model_fn=model_fn,
      metrics_fn=eval_metrics,
      train_state_axis_resources=train_state_axis_resources,
      input_axis_resources_det=input_axis_resources_det,
      input_axis_resources_cls=input_axis_resources_cls,
      rngs=eval_state_rngs,
      use_ema=use_ema,
      **evaluate_config)
  with metric_writers.ensure_flushes(writer):
    # Explicitly compile train_step here and report the compilation time.
    previous_step = 0
    while True:
      writer.write_scalars(1, {"eval/steps": init_step})
      if init_step > 0 and init_step != previous_step:
        # for i in range(previous_step + 1, init_step):
        # evaluation_hook(i, train_state=train_state)
        logging.info(init_step)
        evaluation_hook(init_step, t=None, train_state=train_state)
        previous_step = init_step
      else:
        time.sleep(30)
      train_state = restore_or_create_train_state(
          prefix=os.path.join(output_dir, "ckpt"),
          state_init_fn=state_init_fn,
          axis_resources=train_state_axis_resources,
          rngs=train_state_rngs,
          thread_pool=ThreadPool())
      init_step = train_state.step  # pytype: disable=attribute-error

      if model_to_eval:
        break
      elif model_to_eval_interval:
        step = int(step) + int(model_to_eval_interval)
        logging.info(step)
        if step > model_to_eval_end:
          break
        else:
          train_state = restore_or_create_train_state(
              path=f"{model_to_eval_prefix}/ckpt_{step}",
              prefix="",
              state_init_fn=state_init_fn,
              axis_resources=train_state_axis_resources,
              rngs=train_state_rngs,
              thread_pool=ThreadPool())
          init_step = train_state.step


def restore_or_create_train_state(
    *,
    prefix,
    state_init_fn,
    axis_resources,
    rngs,
    path = None,
    mesh = None,
    thread_pool=None,
):
  """Restores a TrainState from the latest complete checkpoint or creates one."""
  mesh = mesh or maps.thread_resources.env.physical_mesh

  if path:
    prefix = path
  else:
    prefix = checkpoints_base.find_latest_complete_checkpoint_for_prefix(
        prefix=prefix, suffixes=(".index", ".data"))
  logging.info("Prefix: %s", prefix)
  if prefix:
    # Restore train_state from checkpoints to CPU memory.
    train_state = checkpoints_partitioned.restore_checkpoint(
        prefix=prefix,
        tree=jax.eval_shape(state_init_fn, rngs),
        thread_pool=thread_pool)
    # Copy TrainState to device memory, and return.
    with jax.sharding.Mesh(mesh.devices, mesh.axis_names):
      return pjit.pjit(
          fun=lambda x: x,
          in_shardings=(axis_resources,),
          out_shardings=axis_resources,
      )(train_state)
  # If no complete checkpoints exist with the given prefix, create a train
  # state from scratch on device.
  return create_train_state(
      state_init_fn=state_init_fn,
      axis_resources=axis_resources,
      rngs=rngs,
      mesh=mesh)


@gin.configurable(denylist=["output_dir"])
def train_and_validate_vmoe_mtl(
    output_dir,
    num_expert_partitions = gin.REQUIRED):
  mesh = partitioning.get_auto_logical_mesh(num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)
  with mesh:
    _train_and_validate_vmoe_mtl(output_dir, mesh)


@gin.configurable(denylist=["output_dir"])
def _train_and_validate_vmoe_mtl(
    output_dir,
    mesh,
    input_fn_det = gin.REQUIRED,
    input_fn_cls = gin.REQUIRED,
    val_input_fn_det = gin.REQUIRED,
    val_input_fn_cls = gin.REQUIRED,
    model_fn=gin.REQUIRED,
    optimizer_def = gin.REQUIRED,
    loss_fn = gin.REQUIRED,
    det_learning_rate = gin.REQUIRED,
    cls_learning_rate = gin.REQUIRED,
    total_train_steps = gin.REQUIRED,
    real_bs_det = gin.REQUIRED,
    real_bs_cls = gin.REQUIRED,
    rng_seed = 0,
    checkpoint_to_load = None,
    profile_config = None,
    progress_config = None,
    checkpoint_config = None,
    fixed_loss_ratio = None,
    cls_model_checkpoints = None,
    cls_model_fn = None,
    mixup_cutmix_config = None):
  """Train the model."""
  if not output_dir:
    raise ValueError("output_dir should be a non-empty path.")
  dataset_det, batch_det = create_input_generator(input_fn_det)
  dataset_cls, batch_cls = create_input_generator(input_fn_cls)
  val_dataset_det, _ = create_input_generator(val_input_fn_det)
  val_dataset_cls, _ = create_input_generator(val_input_fn_cls)
  if isinstance(batch_det, tuple):
    images_det, labels_det = batch_det
    sample_det = {"images_det": images_det, "labels_det": labels_det}
  else:
    sample_det = {}
    for key in batch_det:
      sample_det[key + "_det"] = batch_det[key]

  if isinstance(batch_cls, tuple):
    images_cls, labels_cls = batch_cls
    sample_cls = {"images_cls": images_cls, "labels_cls": labels_cls}
  else:
    sample_cls = {}
    for key in batch_cls:
      sample_cls[key + "_cls"] = batch_cls[key]
  def generate_partition_spec(_):
    return partitioning.parse_partition_spec((mesh.axis_names,))
  resources_det = jax.tree.map(
      generate_partition_spec, sample_det)
  resources_cls = jax.tree.map(
      generate_partition_spec, sample_cls)
  train_state_rngs = utils.make_rngs(("params", "gating_cls", "gating_det",
                                      "dropout_cls", "dropout_det", "rng",
                                      "gating", "dropout"),
                                     rng_seed)
  train_state_rngs, _ = utils.tree_rngs_split(train_state_rngs)
  state_init_fn = init_state(
      model_fn,
      batch_det,
      real_bs_det,
      batch_cls,
      real_bs_cls,
      optimizer_def,
      resources_det,
      resources_cls,
      mode=ExecutionMode.TRAIN, dynamic=True)

  params_axis_resources = [("Moe/Mlp/.*", ("expert",))]
  train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
      tree=jax.eval_shape(state_init_fn, train_state_rngs),
      axis_resources_regexes=params_axis_resources)
  logging.info("Creating train state.")
  if cls_model_checkpoints is not None:
    cls_train_state_rngs = utils.make_rngs(
        ("params", "gating", "dropout", "rng"), rng_seed)
    resources_cls_load = jax.tree.map(
        generate_partition_spec, batch_cls)
    cls_state_init_fn = init_cls_state(
        cls_model_fn,
        batch_cls,
        real_bs_cls,
        optimizer_def,
        resources_cls_load,
        mode=ExecutionMode.TRAIN
    )
    cls_train_state_axis_resources = partitioning.tree_axis_resources_from_regexes(
        tree=jax.eval_shape(cls_state_init_fn, cls_train_state_rngs),
        axis_resources_regexes=params_axis_resources)

  if checkpoint_to_load is None:
    prefix = os.path.join(output_dir, "ckpt")
    train_state = restore_or_create_train_state(
        prefix=prefix,
        path=checkpoint_to_load,
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
    init_step = train_state.step
  else:
    train_state = restore_or_create_train_state(
        prefix="",
        path=checkpoint_to_load,
        state_init_fn=state_init_fn,
        axis_resources=train_state_axis_resources,
        rngs=train_state_rngs,
        thread_pool=ThreadPool())
    init_step = 0

  if (cls_model_checkpoints is not None) and (init_step == 0):
    cls_train_state = restore_or_create_train_state(
        prefix="",
        path=cls_model_checkpoints,
        state_init_fn=cls_state_init_fn,
        axis_resources=cls_train_state_axis_resources,
        rngs=cls_train_state_rngs,
        thread_pool=ThreadPool())
    params_to_load = cls_train_state.ema_target
    params = train_state.optimizer.target
    is_frozen_dict = isinstance(params, flax.core.FrozenDict)
    if is_frozen_dict:
      params = flax.core.unfreeze(params)

    params_flat = flax.traverse_util.flatten_dict(params)
    ckpt_params = flax.core.unfreeze(params_to_load)
    ckpt_params_flat = flax.traverse_util.flatten_dict(ckpt_params)
    ckpt_params_flat = {"/".join(k): v for k, v in ckpt_params_flat.items()}
    ckpt_params_used = set()

    for key_tuple, value in params_flat.items():
      key = "/".join(key_tuple)
      logging.info(key)

      if "dense1" in key:
        ckpt_key = key.replace("dense1", "Dense_0")
      elif "dense2" in key:
        ckpt_key = key.replace("dense2", "Dense_1")
      elif "Router_cls" in key:
        ckpt_key = key.replace("Router_cls", "Router")
      elif "cls_cls" in key:
        ckpt_key = key.replace("cls_cls", "cls")
      elif "embedding_cls" in key:
        ckpt_key = key.replace("embedding_cls", "embedding")
      else:
        ckpt_key = key
      if "VisionTransformerMoeMTL_0" in ckpt_key:
        ckpt_key = ckpt_key.replace(
            "VisionTransformerMoeMTL_0", "VisionTransformerMoe_0")

      ckpt_params_used.add(ckpt_key)
      if ckpt_key in ckpt_params_flat:
        ckpt_value = ckpt_params_flat[ckpt_key]

        if value.shape == ckpt_value.shape:
          params_flat[key_tuple] = ckpt_value
        else:
          raise ValueError(f"Parameter {key!r} was mapped to {ckpt_key!r}, but "
                           f"their shapes are not equal: {value.shape} vs "
                           f"{ckpt_value.shape}.")
      else:
        logging.info("Missing %s", ckpt_key)
    params = flax.traverse_util.unflatten_dict(params_flat)

    if is_frozen_dict:
      freeze_params = flax.core.freeze(params)
    else:
      freeze_params = params
    train_state = train_state.replace(
        optimizer=optimizer_def.create(freeze_params))
    # train_state = train_state.replace(
    #    ema_target=train_state.optimizer.target
    # )
    del ckpt_params, ckpt_params_flat, params_flat, cls_state_init_fn, cls_train_state
    def sync_ema_and_weight(train_state):
      old_ema_target = train_state.ema_target
      optimizer = train_state.optimizer
      model_state = train_state.model_state
      rngs = train_state.rngs
      new_ema_target = jax.tree.map(lambda x, y: 1 * x + 0 * y,
                                    optimizer.target, old_ema_target)
      new_train_state = train_state.replace(  # pytype: disable=attribute-error
          step=init_step,
          optimizer=optimizer,
          model_state=model_state,
          ema_target=new_ema_target,
          rngs=rngs)
      return new_train_state

    sync_pjit = pjit.pjit(
        fun=sync_ema_and_weight,
        in_shardings=(train_state_axis_resources,),
        out_shardings=train_state_axis_resources,
    )
    train_state = sync_pjit(train_state)
    logging.info("sync complete")
  if init_step == total_train_steps:
    return
  logging.info(init_step)

  def train_step(
      train_state,
      data_det,
      data_cls,
      second_stage=False):
    """A single training step."""
    step = train_state.step
    optimizer = train_state.optimizer
    model_state = train_state.model_state
    det_lr = det_learning_rate(step) if callable(
        det_learning_rate) else det_learning_rate
    cls_lr = cls_learning_rate(step) if callable(
        cls_learning_rate) else cls_learning_rate
    # det_lr = 0
    rngs = train_state.rngs
    if mixup_cutmix_config is not None:
      mixup_rngs, rngs = utils.tree_rngs_split(rngs)
      mixup_rng = mixup_rngs["rng"]
      images = data_cls["images_cls"]
      labels = data_cls["labels_cls"]
      mixup_cutmix_image, mixup_cutmix_labels = wrap_mixup_and_cutmix(
          images,
          labels,
          mixup_rng,
          mixup_alpha=mixup_cutmix_config.get("mixup_alpha", 0),
          cutmix_alpha=mixup_cutmix_config.get("cutmix_alpha", 0))
      data_cls["images_cls"] = mixup_cutmix_image
      data_cls["labels_cls"] = mixup_cutmix_labels

    rngs, next_rngs = utils.tree_rngs_split(rngs)

    def loss_step(
        params,
        det_lr,
        cls_lr,
        base_lr):
      variables = {"params": params}
      variables.update(model_state)

      outputs, new_variables = model_fn(mode=ExecutionMode.TRAIN).apply(
          variables,
          _do_remap=True,
          **data_det,
          **data_cls,
          second_stage=second_stage,
          mutable=DenyList("params"),
          rngs=rngs)
      metrics = outputs["metrics"]["metrics"]

      losses = loss_fn(
          outputs, **data_det, **data_cls, _do_remap=True, params=params)
      aux_output = {"new_model_state": new_variables, "losses": losses}
      if fixed_loss_ratio is None:
        det_lr_ratio = det_lr / (base_lr + 1e-8)  # calculate balance factor
        cls_lr_ratio = cls_lr / (base_lr + 1e-8)
      else:
        det_lr_ratio = fixed_loss_ratio
        cls_lr_ratio = 1
      model_loss = det_lr_ratio * losses["det_loss"]
      model_loss += cls_lr_ratio * losses["cls_loss"]
      aux_output["lr"] = {}
      aux_output["lr"]["det_lr"] = det_lr
      aux_output["lr"]["cls_lr"] = cls_lr
      metrics["auxiliary_loss_det"] = jnp.mean(metrics["auxiliary_loss_det"])
      metrics["auxiliary_loss_cls"] = jnp.mean(metrics["auxiliary_loss_cls"])
      model_loss += det_lr_ratio * metrics["auxiliary_loss_det"]
      model_loss += cls_lr_ratio * metrics["auxiliary_loss_cls"]
      aux_output["losses"]["auxiliary_loss_cls"] = metrics["auxiliary_loss_cls"]
      aux_output["losses"]["auxiliary_loss_det"] = metrics["auxiliary_loss_det"]
      if "xent_loss_det" in metrics:
        metrics["xent_loss_det"] = jnp.mean(metrics["xent_loss_det"])
        metrics["xent_loss_cls"] = jnp.mean(metrics["xent_loss_cls"])
        aux_output["losses"]["xent_loss_det"] = metrics["xent_loss_det"]
        aux_output["losses"]["xent_loss_cls"] = metrics["xent_loss_cls"]
      aux_output["losses"]["model_loss"] = model_loss
      return model_loss, aux_output

    grad_fn = jax.value_and_grad(loss_step, has_aux=True)
    base_lr = cls_lr

    (_, aux_output), grads = grad_fn(optimizer.target, det_lr, cls_lr, base_lr)

    new_optimizer = optimizer.apply_gradient(grads, learning_rate=base_lr)

    new_model_state = aux_output["new_model_state"]
    losses = aux_output["losses"]
    lrs = aux_output["lr"]
    losses.update(lrs)
    old_ema_target = train_state.ema_target
    new_ema_target = jax.tree.map(lambda x, y: 0.1 * x + 0.9 * y,
                                  new_optimizer.target, old_ema_target)
    new_train_state = train_state.replace(  # pytype: disable=attribute-error
        step=step + 1,
        optimizer=new_optimizer,
        model_state=new_model_state,
        ema_target=new_ema_target,
        rngs=next_rngs)
    return new_train_state, losses

  def val_step(train_state, data_det, data_cls):
    """A single validation step."""
    model_state = train_state.model_state
    rngs = train_state.rngs
    rngs, _ = utils.tree_rngs_split(rngs)

    def loss_step(params):
      """Loss step that is used to compute the gradients."""
      variables = {"params": params}
      variables.update(model_state)
      outputs, _ = model_fn(mode=ExecutionMode.TRAIN).apply(
          variables,
          _do_remap=True,
          **data_det,
          **data_cls,
          mutable=DenyList("params"),
          rngs=rngs)
      losses = loss_fn(
          outputs, **data_det, **data_cls, _do_remap=True, params=params)
      return losses

    return loss_step(train_state.optimizer.target)
  def assign_func(
      train_state,
      current_k_det,
      current_k_cls,
      prev_best_val_loss_det,
      prev_best_val_loss_cls):
    return train_state.replace(
        current_k_det=current_k_det,
        current_k_cls=current_k_cls,
        prev_best_val_loss_cls=prev_best_val_loss_cls,
        prev_best_val_loss_det=prev_best_val_loss_det
    )
  train_step_pjit = pjit.pjit(
      fun=functools.partial(train_step),
      in_shardings=(
          train_state_axis_resources,  # train_state
          resources_det,  # images
          resources_cls,
      ),
      out_shardings=(
          train_state_axis_resources,  # train_state
          None,  # metrics
      ),
      donate_argnums=(0, 1, 2),
  )

  val_step_pjit = pjit.pjit(
      fun=val_step,
      in_shardings=(
          train_state_axis_resources,  # train_state
          resources_det,  # images
          resources_cls,
      ),
      out_shardings=None,
      donate_argnums=(1, 2),
  )
  assign_pjit = pjit.pjit(
      fun=assign_func,
      in_shardings=(train_state_axis_resources, None, None, None, None),
      out_shardings=train_state_axis_resources,
  )
  writer = metric_writers.create_default_writer(
      logdir=output_dir, just_logging=jax.process_index() > 0)
  profile_hook = trainer.create_profile_hook(
      workdir=output_dir, **profile_config)
  progress_hook = trainer.create_progress_hook(
      writer=writer,
      first_step=int(init_step) + 1,
      train_steps=int(total_train_steps),
      **progress_config)
  checkpoint_hook = trainer.create_checkpoint_hook(  # pytype: disable=module-attr
      workdir=output_dir,
      progress_hook=progress_hook,
      train_state_axis_resources=train_state_axis_resources,
      train_steps=total_train_steps,
      **checkpoint_config)
  # config_model_eval = config.model.copy_and_resolve_references()
  with metric_writers.ensure_flushes(writer):
    # Explicitly compile train_step here and report the compilation time.
    t0 = time.time()
    # logging.info(jax.tree.map(lambda x: x.shape, batch))
    train_step_pjit = train_step_pjit.lower(
        *tree_shape_dtype_struct((train_state, sample_det, sample_cls))
    ).compile()
    val_step_pjit = val_step_pjit.lower(
        *tree_shape_dtype_struct((train_state, sample_det, sample_cls))
    ).compile()

    t1 = time.time()
    writer.write_scalars(init_step + 1, {"train/compile_secs": t1 - t0})
    tr_iter_cls = pjit_utils.prefetch_to_device(
        iterator=make_dataset_iterator(dataset_cls),
        size=2,
        mesh=mesh)
    tr_iter_det = pjit_utils.prefetch_to_device(
        iterator=make_dataset_iterator(dataset_det),
        size=2,
        mesh=mesh)
    val_iter_cls = pjit_utils.prefetch_to_device(
        iterator=make_dataset_iterator(val_dataset_cls),
        size=2,
        mesh=mesh)
    val_iter_det = pjit_utils.prefetch_to_device(
        iterator=make_dataset_iterator(val_dataset_det),
        size=2,
        mesh=mesh)
    encoder_config = gin.query_parameter(
        "get_uvit_vmoe_mtl_model_fn.encoder_config")

    current_k_det = int(train_state.current_k_det)  # pytype: disable=attribute-error
    current_k_cls = int(train_state.current_k_cls)  # pytype: disable=attribute-error
    prev_best_val_loss_cls = float(train_state.prev_best_val_loss_cls)  # pytype: disable=attribute-error
    prev_best_val_loss_det = float(train_state.prev_best_val_loss_det)  # pytype: disable=attribute-error
    dynamic_k = True
    stop_changing = False
    flag = False
    cls_best_step = det_best_step = 40000
    for step, r_batch_det, r_batch_cls in zip(
        range(init_step + 1, total_train_steps + 1), tr_iter_det, tr_iter_cls):
      profile_hook(step)
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        if isinstance(r_batch_cls, tuple):
          images_cls, labels_cls = r_batch_cls
          r_input_batch_cls = {
              "images_cls": images_cls,
              "labels_cls": labels_cls
          }
        else:
          r_input_batch_cls = {}
          for key in batch_cls:
            r_input_batch_cls[key + "_cls"] = r_batch_cls[key]
        if isinstance(r_batch_det, tuple):
          images_det, labels_det = r_batch_det
          r_input_batch_det = {
              "images_det": images_det,
              "labels_det": labels_det
          }
        else:
          r_input_batch_det = {}
          for key in batch_det:
            r_input_batch_det[key + "_det"] = r_batch_det[key]

        train_state, metrics_value = train_step_pjit(train_state,
                                                     r_input_batch_det,
                                                     r_input_batch_cls)
        scalar_metrics = {f"train/{k}": v for k, v in metrics_value.items()}
      scalar_metrics.update({"train/current_k_det": current_k_det})
      scalar_metrics.update({"train/current_k_cls": current_k_cls})
      scalar_metrics.update({"train/best_val_loss_det": prev_best_val_loss_det})
      scalar_metrics.update({"train/best_val_loss_cls": prev_best_val_loss_cls})

      if step % 2000 == 0 and dynamic_k:
        val_loss_cls = 0
        val_loss_det = 0
        recompile = False

        for val_step_number, (r_batch_det, r_batch_cls) in enumerate(
            zip(val_iter_det, val_iter_cls)):
          logging.info(val_step_number)
          if isinstance(r_batch_det, tuple):
            images_det, labels_det = r_batch_det
            r_input_batch_det = {
                "images_det": images_det,
                "labels_det": labels_det
            }
          else:
            r_input_batch_det = {}
            for key in batch_det:
              r_input_batch_det[key + "_det"] = r_batch_det[key]

          if isinstance(r_batch_cls, tuple):
            images_cls, labels_cls = r_batch_cls
            r_input_batch_cls = {
                "images_cls": images_cls,
                "labels_cls": labels_cls
            }
          else:
            r_input_batch_cls = {}
            for key in batch_cls:
              r_input_batch_cls[key + "_cls"] = r_batch_cls[key]

          losses = val_step_pjit(
              train_state, r_input_batch_det, r_input_batch_cls)
          logging.info(losses)
          val_loss_det = val_loss_det + losses["det_loss"]
          val_loss_cls = val_loss_cls + losses["cls_loss"]

          if val_step_number >= 10:
            break

        scalar_metrics.update({"val/loss_det": val_loss_det})
        scalar_metrics.update({"val/loss_cls": val_loss_cls})
        if val_loss_cls < prev_best_val_loss_cls:
          prev_best_val_loss_cls = val_loss_cls
          cls_best_step = step
          flag = False
        elif step - cls_best_step > 5000 and current_k_cls < 8 and not stop_changing:
          if flag:
            current_k_cls -= 1
            stop_changing = True
          else:
            current_k_cls += 1
          router_config = encoder_config["moe"]["router"]
          router_config["num_selected_experts_cls"] = current_k_cls
          encoder_config["moe"]["router"] = router_config
          recompile = True

        if val_loss_det < prev_best_val_loss_det:
          prev_best_val_loss_det = val_loss_det
          det_best_step = step
          flag = False
        elif step - det_best_step > 5000 and current_k_det < 8 and not stop_changing:
          if flag:
            current_k_det -= 1
            stop_changing = True
          else:
            current_k_det += 1
          router_config = encoder_config["moe"]["router"]
          router_config["num_selected_experts_det"] = current_k_det
          encoder_config["moe"]["router"] = router_config
          recompile = True

        if recompile:
          with gin.unlock_config():
            gin.bind_parameter("get_uvit_vmoe_model_fn.encoder_config",
                               encoder_config)

          del train_step_pjit
          del val_step_pjit
          train_step_pjit = pjit.pjit(
              fun=train_step,
              in_shardings=(
                  train_state_axis_resources,  # train_state
                  resources_det,  # images
                  resources_cls,
              ),
              out_shardings=(
                  train_state_axis_resources,  # train_state
                  None,  # metrics
              ),
              donate_argnums=(1, 2),
          )

          val_step_pjit = pjit.pjit(
              fun=val_step,
              in_shardings=(
                  train_state_axis_resources,  # train_state
                  resources_det,  # images
                  resources_cls,
              ),
              out_shardings=None,
              donate_argnums=(1, 2),
          )
          flag = True

      train_state = assign_pjit(train_state, current_k_det, current_k_cls,
                                prev_best_val_loss_det, prev_best_val_loss_cls)
      progress_hook(step, scalar_metrics=scalar_metrics)
      checkpoint_hook(step, state=train_state)
