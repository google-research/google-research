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

# python3
"""Fixed Non Volume Preserving flow based tasks."""
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
from task_set.tasks import nvp


def get_loss_fn(num_bijectors, layers):
  """Helper for constructing a NVP based loss.

  Args:
    num_bijectors: int the number of bijectors to use.
    layers: list list with number of units per layer for each bijector.

  Returns:
    callable that returns a sonnet module representing the loss.
  """

  def _fn(batch):
    dist = nvp.distribution_with_nvp_bijectors(
        batch["image"], num_bijectors=num_bijectors, layers=layers)
    return nvp.neg_log_p(dist, batch["image"])

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed("FixedNVP_mnist_2layer_bs64")
def _():
  base_model_fn = get_loss_fn(2, layers=(2048, 2048))
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedNVP_mnist_5layer_bs64")
def _():
  base_model_fn = get_loss_fn(5, layers=(2048, 2048))
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedNVP_mnist_5layer_thin_bs64")
def _():
  base_model_fn = get_loss_fn(5, layers=(128, 128))
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedNVP_mnist_3layer_thin_bs64")
def _():
  base_model_fn = get_loss_fn(3, layers=(128, 128))
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedNVP_mnist_9layer_thin_bs16")
def _():
  base_model_fn = get_loss_fn(9, layers=(128, 128))
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)
