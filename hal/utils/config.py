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

# Lint as: python3
"""An experiment configuration object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class Config:
  """Contains configurations of the experiments."""

  def __init__(self, content=None):
    content_dict = collections.OrderedDict()
    if content:
      content_dict.update(content)
    self._content = content_dict

  def __getattr__(self, item):
    if item in self._content:
      return self.__getattribute__('_content')[item]
    else:
      return self.__getattribute__(item)

  def __repr__(self):
    rt = ''
    for k, v in self._content.items():
      rt += '{}: {} \n'.format(k, v)
    return rt

  def as_dict(self):
    return dict(self._content)

  def update(self, new_cfg):
    self.__getattribute__('_content').update(new_cfg.as_dict())

  def keys(self):
    return self._content.keys()
