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

"""Fixed MLP autoencoder tasks."""
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
import tensorflow.compat.v1 as tf


def fc_ae_loss_fn(hidden_units, activation):
  """Helper for a fully connected autoencoder.

  Args:
    hidden_units: list list of integers containing the hidden layers.
    activation: callable the activation function used by the MLP.

  Returns:
    A callable that returns a sonnet module representing the task loss.
  """

  def _fn(batch):
    net = snt.BatchFlatten()(batch["image"])
    feats = net.shape.as_list()[-1]
    logits = snt.nets.MLP(hidden_units + [feats], activation=activation)(net)

    loss_vec = tf.reduce_mean(tf.square(net - tf.nn.sigmoid(logits)), [1])
    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed("FixedMLPAE_mnist_32x32x32_bs128")
def _():
  base_model_fn = fc_ae_loss_fn([32, 32, 32], tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedMLPAE_mnist_128x32x128_bs128")
def _():
  base_model_fn = fc_ae_loss_fn([128, 32, 128], tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedMLPAE_cifar10_128x32x128_bs128")
def _():
  base_model_fn = fc_ae_loss_fn([128, 32, 128], tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)
