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

"""Implementation of common functions."""
import os
import random

import numpy as np
import torch


def seed_torch(seed=42):
  """Defines a function to fix the seed of different random variables for reproducability.

  Args:
    seed: an integer
  """

  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def add_gaussian_noise(tensor, mean=0.1, std=1., device='cpu'):
  """Function that add noise to a given tensor.

  Args:
    tensor: An input tensor
    mean: Gaussian noise mean
    std: Gaussian noise std
    device: device used to store tensor

  Returns:
    tensor: A new tensor with added noise

  """

  return tensor + torch.randn(tensor.size()).to(device) * std + mean


class EarlyStopping:
  """Class to montior the progress of the model and stop early if no improvement on validation set."""

  def __init__(self, patience=7, verbose=False, delta=0):
    """Initializes parameters for EarlyStopping class.

    Args:
      patience: an integer
      verbose: a boolean
      delta: a float
    """
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta

  def __call__(self, val_loss, model, path):
    """Checks if the validation loss is better than the best validation loss.

       If so model is saved.
       If not the EarlyStopping  counter is increased
    Args:
      val_loss: a float representing validation loss
      model: the trained model
      path: a string representing the path to save the model
    """
    score = -val_loss
    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(val_loss, model, path)
    elif score < self.best_score + self.delta:
      self.counter += 1
      print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
      else:
        self.early_stop = False
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, model, path)
      self.counter = 0

  def save_checkpoint(self, val_loss, model, path):
    """Saves the model and updates the best validation loss.

    Args:
      val_loss: a float representing validation loss
      model: the trained model
      path: a string representing the path to save the model
    """
    if self.verbose:
      print(
          f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
      )
    torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
    self.val_loss_min = val_loss

