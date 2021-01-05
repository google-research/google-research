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

"""tensor2tensor data generator incorporating sparsity problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_datagen
import tensorflow.compat.v1 as tf
from routing_transformer import problems  # pylint: disable=unused-import


def main(argv):
  t2t_datagen.main(argv)


if __name__ == "__main__":
  tf.app.run()
