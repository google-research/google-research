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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Warmup Scheduler."""
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
  """Gradual Warmup Scheduler."""

  def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
    self.multiplier = multiplier
    if self.multiplier < 1.:
      raise ValueError('multiplier should be greater thant or equal to 1.')
    self.total_epoch = total_epoch
    self.after_scheduler = after_scheduler
    self.finished = False
    super().__init__(optimizer)

  def get_lr(self):
    if self.last_epoch > self.total_epoch:
      if self.after_scheduler:
        if not self.finished:
          self.after_scheduler.base_lrs = [
              base_lr * self.multiplier for base_lr in self.base_lrs
          ]
          self.finished = True
        return self.after_scheduler.get_lr()
    return [
        base_lr *
        ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
        for base_lr in self.base_lrs
    ]

  def step_ReduceLROnPlateau(self, metrics, epoch=None):
    if epoch is None:
      epoch = self.last_epoch + 1
    self.last_epoch = epoch if epoch != 0 else 1
    if self.last_epoch <= self.total_epoch:
      warmup_lr = [
          base_lr *
          ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
          for base_lr in self.base_lrs
      ]
      for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
        param_group['lr'] = lr
    else:
      if epoch is None:
        self.after_scheduler.step(metrics, None)
      else:
        self.after_scheduler.step(metrics, epoch - self.total_epoch)

  def step(self, epoch=None, metrics=None):
    if not isinstance(self.after_scheduler, ReduceLROnPlateau):
      if self.finished and self.after_scheduler:
        if epoch is None:
          self.after_scheduler.step(None)
        else:
          self.after_scheduler.step(epoch - self.total_epoch)
      else:
        return super().step(epoch)
    else:
      self.step_ReduceLROnPlateau(metrics, epoch)
