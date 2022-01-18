# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Fixed masked autoregressive flows tasks.

See: https://arxiv.org/abs/1705.07057 for more information.
"""
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
from task_set.tasks import maf


def get_loss_fn(num_bijectors, layers):

  def _fn(batch):
    dist = maf.dist_with_maf_bijectors(
        batch["image"], num_bijectors=num_bijectors, layers=layers)
    return maf.neg_log_p(dist, batch["image"])

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed("FixedMAF_mnist_2layer_bs64")
def _():
  base_model_fn = get_loss_fn(2, (2048, 2048))
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedMAF_mnist_3layer_thin_bs64")
def _():
  base_model_fn = get_loss_fn(3, (128, 128))
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedMAF_cifar10_3layer_bs64")
def _():
  base_model_fn = get_loss_fn(3, (1024, 1024))
  dataset = datasets.get_image_datasets("cifar10", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)
