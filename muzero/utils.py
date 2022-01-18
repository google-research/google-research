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
# pylint: disable=logging-format-interpolation
# pylint: disable=missing-docstring
# pylint: disable=g-complex-comprehension
r"""Utils for the MuZero SEED RL implementation."""

import tensorflow as tf


def write_flags(flags_dict, file_name):
  important_kwords = [
      # Add flags here for their values to be written out to file.
  ]
  flags_str = "\n".join([
      "--{}\n{}".format(k, v.value)
      for k, v in flags_dict.items()
      if k in important_kwords
  ])
  with tf.io.gfile.GFile(file_name, "w") as f:
    f.write(flags_str)
