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

"""File utilities."""
import os
from os import path

BASE_DIR = "light_field_neural_rendering"
INTERNAL = False


def open_file(pth, mode="r"):

  return open(pth, mode=mode)


def file_exists(pth):
  return path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return path.isdir(pth)


def makedirs(pth):
  os.makedirs(pth)
