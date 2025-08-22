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

"""An Early Stopping Module"""

import numpy as np
import tensorflow.compat.v1 as tf

from fm4tlp.models import model_template


class EarlyStopMonitor(object):

  def __init__(
      self,
      save_model_dir,
      save_model_id,
      tolerance = 1e-10,
      patience = 5,
      higher_better = True,
  ):
    r"""Early Stopping Monitor

    save_model_dir: where to save the model
    save_model_id: an id to save the model with
    tolerance: float, the amount of tolerance of the early stopper
    patience: int, how many round to wait
    higher_better: whether higher_value of the a metric is better
    """
    self.tolerance = tolerance
    self.patience = patience
    self.higher_better = higher_better
    self.counter = 0
    self.best_sofar = None
    self.best_epoch = 0
    self.epoch_idx = 1

    self.save_model_dir = save_model_dir
    if not tf.io.gfile.isdir(self.save_model_dir):
      print("INFO: Create directory {}".format(save_model_dir))
      tf.io.gfile.makedirs(self.save_model_dir)
    self.save_model_id = save_model_id

  def get_best_model_path(self):
    r"""return the path of the best model"""
    return self.save_model_dir + "/{}.pth".format(self.save_model_id)

  def step_check(self, curr_metric, model):
    r"""execute the early stop strategy

    curr_metric: a metric to evaluate the early stopping on.
    model: model to check.
    """
    if not self.higher_better:
      curr_metric *= -1

    if (self.best_sofar is None) or (
        (curr_metric - self.best_sofar) / np.abs(self.best_sofar)
        > self.tolerance
    ):
      # first iteration or observing an improvement
      self.best_sofar = curr_metric
      print("INFO: save a checkpoint...")
      self.save_checkpoint(model)
      self.counter = 0
      self.best_epoch = self.epoch_idx
    else:
      # no improvement observed
      self.counter += 1

    self.epoch_idx += 1

    return self.counter >= self.patience

  def save_checkpoint(self, model):
    r"""save models as a checkpoint

    model: model to be saved.
    """
    model_path = self.get_best_model_path()
    print("INFO: save the model to {}".format(model_path))
    model.save_model(model_path)

  def load_checkpoint(self, model):
    r"""save models from the checkpoint

    model: model to be loaded.
    """
    model_path = self.get_best_model_path()
    print(
        "INFO: load the model of epoch {} from {}".format(
            self.best_epoch, model_path
        )
    )
    model.load_model(model_path)
