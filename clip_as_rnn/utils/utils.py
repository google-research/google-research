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

"""Utility functions for the project."""

from __future__ import print_function
# pylint: disable=g-importing-member
from collections import defaultdict
from collections import deque
from copy import deepcopy
import datetime
import errno
import os
import sys
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import yaml

# pylint: disable=g-bad-import-order
from data.voc import CLASS2ID
from data.voc import VOC_CLASSES


_MB = 1024.0 * 1024.0

DINO_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class Config:

  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      if isinstance(value, dict):
        setattr(self, key, Config(**value))
      else:
        setattr(self, key, value)


def load_yaml(filename):
  with open(filename) as file:
    try:
      data = yaml.safe_load(file)
      return data
    except yaml.YAMLError as e:
      print(f"Error while loading YAML file: {e}")


def normalize(x, dim=None, eps=1e-15):
  if dim is None:
    return (x - x.min()) / (x.max() - x.min())
  # Normalize to [0, 1].
  numerator = x - x.min(axis=dim, keepdims=True)[0]
  denominator = (
      x.max(axis=dim, keepdims=True)[0]
      - x.min(axis=dim, keepdims=True)[0]
      + eps
  )
  return numerator / denominator


class SmoothedValue(object):
  """Track a series of values and provide access to smoothed values over a window or the global series average."""

  def __init__(self, window_size=20, fmt=None):
    if fmt is None:
      fmt = "{median:.4f} ({global_avg:.4f})"
    self.deque = deque(maxlen=window_size)
    self.total = 0.0
    self.count = 0
    self.fmt = fmt

  def update(self, value, n=1):
    self.deque.append(value)
    self.count += n
    self.total += value * n

  # def synchronize_between_processes(self):
  #     """
  #     Warning: does not synchronize the deque!
  #     """
  #     if not is_dist_avail_and_initialized():
  #         return
  #     t = torch.tensor([self.count, self.total],
  #                      dtype=torch.float64, device='cuda')
  #     dist.barrier()
  #     dist.all_reduce(t)
  #     t = t.tolist()
  #     self.count = int(t[0])
  #     self.total = t[1]

  @property
  def median(self):
    d = torch.tensor(list(self.deque))
    return d.median().item()

  @property
  def avg(self):
    d = torch.tensor(list(self.deque), dtype=torch.float32)
    return d.mean().item()

  @property
  def global_avg(self):
    return self.total / self.count

  @property
  def max(self):
    return max(self.deque)

  @property
  def value(self):
    return self.deque[-1]

  def __str__(self):
    return self.fmt.format(
        median=self.median,
        avg=self.avg,
        global_avg=self.global_avg,
        max=self.max,
        value=self.value,
    )


class MetricLogger(object):
  """Log the metrics."""

  def __init__(self, delimiter="\t"):
    self.meters = defaultdict(SmoothedValue)
    self.delimiter = delimiter

  def update(self, **kwargs):
    for k, v in kwargs.items():
      if isinstance(v, torch.Tensor):
        v = v.item()
      assert isinstance(v, (float, int))
      self.meters[k].update(v)

  def __getattr__(self, attr):
    if attr in self.meters:
      return self.meters[attr]
    if attr in self.__dict__:
      return self.__dict__[attr]
    raise AttributeError(
        "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
    )

  def __str__(self):
    loss_str = []
    for name, meter in self.meters.items():
      loss_str.append("{}: {}".format(name, str(meter)))
    return self.delimiter.join(loss_str)

  def synchronize_between_processes(self):
    for meter in self.meters.values():
      meter.synchronize_between_processes()

  def add_meter(self, name, meter):
    self.meters[name] = meter

  def log_every(self, iterable, print_freq, header=None):
    """Log every `print_freq` times."""
    i = 0
    if not header:
      header = ""
    start_time = time.time()
    end = time.time()
    iter_time = SmoothedValue(fmt="{avg:.4f}")
    data_time = SmoothedValue(fmt="{avg:.4f}")
    space_fmt = ":" + str(len(str(len(iterable)))) + "d"
    log_msg = self.delimiter.join([
        header,
        "[{0" + space_fmt + "}/{1}]",
        "eta: {eta}",
        "{meters}",
        "time: {time}",
        "data: {data}",
        "max mem: {memory:.0f}",
    ])
    for obj in iterable:
      data_time.update(time.time() - end)
      yield obj
      iter_time.update(time.time() - end)
      if i % print_freq == 0:
        eta_seconds = iter_time.global_avg * (len(iterable) - i)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        print(
            log_msg.format(
                i,
                len(iterable),
                eta=eta_string,
                meters=str(self),
                time=str(iter_time),
                data=str(data_time),
                memory=torch.cuda.max_memory_allocated() / _MB,
            )
        )
        sys.stdout.flush()

      i += 1
      end = time.time()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("{} Total time: {}".format(header, total_time_str))


def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def pad_to_square(im):
  """Pad the images to square shape."""
  im = deepcopy(im)
  width, height = im.size
  top_pad = (max(width, height) - height) // 2
  bot_pad = max(width, height) - height - top_pad
  left_pad = (max(width, height) - width) // 2
  right_pad = max(width, height) - width - left_pad

  if len(im.mode) == 3:
    color = (0, 0, 0)
  elif len(im.mode) == 1:
    color = 0
  else:
    raise ValueError(f"Image mode not supported. Image has {im.mode} channels.")

  return add_margin(im, top_pad, right_pad, bot_pad, left_pad, color=color)


def add_margin(pil_img, top, right, bottom, left, color=(0, 0, 0)):
  """Ref: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/."""
  width, height = pil_img.size
  new_width = width + right + left
  new_height = height + top + bottom
  result = Image.new(pil_img.mode, (new_width, new_height), color)
  result.paste(pil_img, (left, top))

  # 1 represents the image, 0 represents the padding
  pad = [left, top, width, height]
  return result, pad


def process_sentence(sentence, ds_name):
  """Dataset specific sentence processing."""
  if "refcoco" in ds_name:
    sentence = sentence[0].lower()
    # get rid of special characters
    sentence = sentence.replace('"', "")
    sentence = sentence.replace("/", "")
  if ds_name == "voc":
    if sentence in list(CLASS2ID.keys()):
      label_id = CLASS2ID[sentence] - 1
      sentence = VOC_CLASSES[label_id]

  if not isinstance(sentence, str):
    sentence = sentence[0]
  return sentence
