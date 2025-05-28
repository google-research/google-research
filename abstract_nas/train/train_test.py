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

"""Tests for train."""

import random

from jax import numpy as jnp
import ml_collections as mlc

from absl.testing import absltest as test
from abstract_nas.abstract.shape import ShapeProperty
from abstract_nas.evolution.mutator.random_sequential import RandomSequentialMutator
from abstract_nas.synthesis.enum_sequential import EnumerativeSequentialSynthesizer
from abstract_nas.train import train
from abstract_nas.train.config import Config
from abstract_nas.zoo import cnn
from abstract_nas.zoo import resnetv1
from abstract_nas.zoo import utils as zoo_utils
import tensorflow.io.gfile as gfile  # pylint: disable=consider-using-from-import


def get_cifar_config():
  config = mlc.ConfigDict()

  config.seed = 0

  ###################
  # Train Config

  config.train = mlc.ConfigDict()
  config.train.seed = 0
  config.train.epochs = 90
  config.train.device_batch_size = 64
  config.train.log_epochs = 1

  # Dataset section
  config.train.dataset_name = "cifar10"
  config.train.dataset = mlc.ConfigDict()
  config.train.dataset.train_split = "train"
  config.train.dataset.val_split = "test"
  config.train.dataset.num_classes = 10
  config.train.dataset.input_shape = (32, 32, 3)

  # Model section
  config.train.model_name = "resnet18"
  config.train.model = mlc.ConfigDict()

  # Optimizer section
  config.train.optim = mlc.ConfigDict()
  config.train.optim.optax_name = "trace"  # momentum
  config.train.optim.optax = mlc.ConfigDict()
  config.train.optim.optax.decay = 0.9
  config.train.optim.optax.nesterov = False

  config.train.optim.wd = 1e-4
  config.train.optim.wd_mults = [(".*", 1.0)]
  config.train.optim.grad_clip_norm = 1.0

  # Learning rate section
  config.train.optim.lr = 0.1
  config.train.optim.schedule = mlc.ConfigDict()
  config.train.optim.schedule.warmup_epochs = 5
  config.train.optim.schedule.decay_type = "cosine"
  # Base batch-size being 256.
  config.train.optim.schedule.scale_with_batchsize = True

  return config


class TrainTest(test.TestCase):

  def test_train_resnet(self):
    graph = resnetv1.ResNet18(num_classes=10, input_resolution="small")
    config_dict = get_cifar_config()
    config_dict.train.epochs = 3
    config_dict.train.optim.schedule.warmup_epochs = 1
    config = Config(
        config_dict=config_dict,
        graph=graph,
        output_dir=None,
        inherit_weights=False)
    metrics = train.train_and_eval(config)[0]
    print(f"final test acc: {metrics.acc}")
    print(flush=True)

  def test_train_resnet_long(self):
    graph = resnetv1.ResNet18(num_classes=10, input_resolution="small")
    config = Config(
        config_dict=get_cifar_config(),
        graph=graph,
        output_dir=None,
        inherit_weights=False)
    metrics = train.train_and_eval(config)[0]
    print(f"final test acc: {metrics.acc}")
    print(flush=True)

    self.assertGreater(metrics.acc, .2)


if __name__ == "__main__":
  test.main()
