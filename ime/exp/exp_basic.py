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

"""Parent class for all experiments."""
import os
import torch


class  ExpBasic(object):
  """A basic experiment class that will be inherited by all other experiments."""

  def __init__(self, args):
    """Initializes a  ExpBasic instance.

    Args:
     args: parser arguments
    """

    self.args = args
    self.device = self._acquire_device()
    self.model_type = self.args.model
    self.data = self._get_dataset()
    self.model = self._build_model().to(self.device)

  def _build_model(self):
    raise NotImplementedError

  def _acquire_device(self):
    if self.args.use_gpu:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(
          self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
      device = torch.device('cuda:{}'.format(self.args.gpu))
      print('Use GPU: cuda:{}'.format(self.args.gpu))
    else:
      device = torch.device('cpu')
      print('Use CPU')
    return device

  def _get_dataset(self):
    raise NotImplementedError

  def _get_data(self):
    raise NotImplementedError

  def vali(self):
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  def test(self):
    raise NotImplementedError
