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

"""Jax Utils."""
import chex
import jax
import jax.dlpack
import jax.numpy as jnp
import tensorflow as tf


def truncated_dtype():
  dtype = jnp.float32
  if is_tpu_backend():
    dtype = jnp.bfloat16
  elif is_gpu_backend():
    dtype = jnp.float16
  return dtype


def maybe_truncate_dtype(x):
  if x.dtype == jnp.float32:
    if is_tpu_backend():
      x = x.astype(jnp.bfloat16)
    elif is_gpu_backend():
      x = x.astype(jnp.float16)
  return x


def tf2jax(tensor):
  """Tensorflow to Jax using the same underlying buffer."""
  return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(tensor))


def is_tpu_backend():
  return jax.local_devices()[0].platform == 'tpu'


def is_gpu_backend():
  return jax.local_devices()[0].platform == 'gpu'


def is_cpu_backend():
  return jax.local_devices()[0].platform == 'cpu'
