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

"""Simple test to see if all the modules load."""

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import tensorflow as tf

from keypose import utils

try:
  import cv2  # pylint: disable=g-import-not-at-top
except ImportError as e:
  print(e)

print(cv2)
print(plt)
print(np)
print(ski)
print(tf)
print(utils)


def main():
  pass


if __name__ == '__main__':
  main()
