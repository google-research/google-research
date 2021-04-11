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

# pylint: skip-file
from absl import logging
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, MaxPool, Relu, Flatten, LogSoftmax
from jax.experimental import optimizers
from . import equations
from . import meshes
from . import gmres
import os
import functools
import jax
import flax
from jax import lax
import jax.numpy as np
import numpy as onp
import jax.ops
from jax.tree_util import Partial

randn = stax.randn
glorot = stax.glorot


# CNN definition

class CNN_old_linear(flax.nn.Module):
  def apply(self, x, inner_channels=8):
    x = flax.nn.Conv(x, features=1, kernel_size=(3, 3), bias=False,
                     strides=(2, 2),
                     padding='VALID')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='VALID', strides=(2,2))
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     input_dilation=(2,2),padding=[(2, 2), (2, 2)])
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=1, kernel_size=(3, 3), bias=False,
                     input_dilation=(2,2),padding=[(2, 2), (2, 2)])
    return x

class Relaxation(flax.nn.Module):
  def apply(self, x, inner_channels=8):
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    return x

class Cycle(flax.nn.Module):
  def apply(self, x, num_cycles=3, inner_channels=8):
    x = Relaxation(x, inner_channels=inner_channels)
    if num_cycles > 0:
      x1 = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3),
                        bias=False,
                        strides=(2, 2),
                        padding='VALID')
      x1 = Cycle(x1, num_cycles=num_cycles-1, inner_channels=inner_channels)
      x1 = flax.nn.Conv(x1, features=1, kernel_size=(3, 3), bias=False,
                       input_dilation=(2,2),padding=[(2, 2), (2, 2)])
      x = x + x1
      x = Relaxation(x, inner_channels=inner_channels)
    return x

class new_CNN(flax.nn.Module):
  def apply(self, x, inner_channels=8):
    x = Cycle(x, 3, inner_channels)
    x = flax.nn.Conv(x, features=1, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    return x

class NonLinearRelaxation(flax.nn.Module):
  def apply(self, x, inner_channels=8):
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.relu(x)
    #x = flax.nn.BatchNorm(x)
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.relu(x)
    #x = flax.nn.BatchNorm(x)
    x = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.relu(x)
    #x = flax.nn.BatchNorm(x)
    return x

class NonLinearCycle(flax.nn.Module):
  def apply(self, x, num_cycles=3, inner_channels=8):
    x = NonLinearRelaxation(x, inner_channels=inner_channels)
    if num_cycles > 0:
      x1 = flax.nn.Conv(x, features=inner_channels, kernel_size=(3, 3),
                        bias=False,
                        strides=(2, 2),
                        padding='VALID')
      x1 = NonLinearCycle(x1, num_cycles=num_cycles-1, inner_channels=inner_channels)
      x1 = flax.nn.Conv(x1, features=1, kernel_size=(3, 3), bias=False,
                       input_dilation=(2,2),padding=[(2, 2), (2, 2)])
      x = x + x1
      #x = np.concatenate((x,x1), axis=3)
      #print(x.shape)
      x = NonLinearRelaxation(x, inner_channels=inner_channels)
    return x

class new_NonLinearCNN(flax.nn.Module):
  def apply(self, x, inner_channels=8):
    x = NonLinearCycle(x, 4, inner_channels)
    x = flax.nn.Conv(x, features=1, kernel_size=(3, 3), bias=False,
                     padding='SAME')
    x = flax.nn.relu(x)
    return x


