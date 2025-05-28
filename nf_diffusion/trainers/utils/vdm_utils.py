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

import functools

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np


def generate_image_grids(images):
  """Simple helper to generate a single image from a mini batch."""

  def image_grid(nrow, ncol, imagevecs, imshape):
    images = iter(imagevecs.reshape((-1,) + imshape))
    return jnp.squeeze(
        jnp.vstack([
            jnp.hstack([next(images)
                        for _ in range(ncol)][::-1])
            for _ in range(nrow)
        ]))

  batch_size = images.shape[0]
  grid_size = int(np.floor(np.sqrt(batch_size)))

  image_shape = images.shape[1:]
  return image_grid(
      nrow=grid_size,
      ncol=grid_size,
      imagevecs=images[0:grid_size**2],
      imshape=image_shape,
  )


def make_sample_fn(model, total_t=1000):
  """VDM Sample function."""

  def sample_fn(dummy_inputs, rng, params, conditioning=None):
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

    if model.config.sm_n_timesteps > 0:
      total_t = model.config.sm_n_timesteps

    if conditioning is None:
      conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype="uint8")

    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    z_init = jax.random.normal(sample_rng, dummy_inputs.shape)

    def body_fn(i, z_t):
      return model.apply(
          variables={"params": params},
          i=i,
          T=total_t,
          z_t=z_t,
          conditioning=conditioning,
          rng=rng,
          method=model.sample_step,
      )

    z_0 = jax.lax.fori_loop(
        lower=0, upper=total_t, body_fun=body_fn, init_val=z_init)

    samples = model.apply(
        variables={"params": params},
        z_0=z_0,
        method=model.decode,
    )
    return samples

  return sample_fn


def allgather_and_reshape(x, axis_name="batch"):
  """Allgather and merge the newly inserted axis w/ the original batch axis."""
  y = jax.lax.all_gather(x, axis_name=axis_name)
  assert y.shape[1:] == x.shape
  return y.reshape(y.shape[0] * x.shape[0], *x.shape[1:])


def dist(fn, accumulate, axis_name="batch"):
  """Wrap a function in pmap and device_get(unreplicate(.)) its return value."""

  if accumulate == "concat":
    accumulate_fn = functools.partial(
        allgather_and_reshape, axis_name=axis_name)
  elif accumulate == "mean":
    accumulate_fn = functools.partial(
        jax.lax.pmean, axis_name=axis_name)
  elif accumulate == "none":
    accumulate_fn = None
  else:
    raise NotImplementedError(accumulate)

  @functools.partial(jax.pmap, axis_name=axis_name)
  def pmapped_fn(*args, **kwargs):
    out = fn(*args, **kwargs)
    return out if accumulate_fn is None else jax.tree.map(accumulate_fn, out)

  def wrapper(*args, **kwargs):
    return jax.device_get(
        flax.jax_utils.unreplicate(pmapped_fn(*args, **kwargs)))

  return wrapper


def stack_forest(forest):
  stack_args = lambda *args: np.stack(args)
  # return jax.tree_multimap(stack_args, *forest)
  return jax.tree_util.tree_map(stack_args, *forest)


def get_metrics(device_metrics):
  # We select the first element of x in order to get a single copy of a
  # device-replicated metric.
  device_metrics = jax.tree.map(lambda x: x[0], device_metrics)
  metrics_np = jax.device_get(device_metrics)
  return stack_forest(metrics_np)
