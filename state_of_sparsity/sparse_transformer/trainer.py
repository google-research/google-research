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

"""Train and eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem

import tensorflow.compat.v1 as tf

from state_of_sparsity.sparse_transformer import common_flags
from state_of_sparsity.sparse_transformer.models import sparse_transformer  # pylint: disable=unused-import

flags = tf.flags
FLAGS = flags.FLAGS


def main(argv):

  argv = common_flags.update_argv(argv)
  return t2t_trainer.main(argv)


if __name__ == "__main__":
  tf.app.run()
