# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Decode the test dataset with a trained model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
from absl import logging

from tensor2tensor.bin import t2t_decoder
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib

from state_of_sparsity.sparse_transformer import common_flags
from state_of_sparsity.sparse_transformer.models import sparse_transformer  # pylint: disable=unused-import

FLAGS = flags.FLAGS


def create_hparams(argv):
  t2t_trainer.set_hparams_from_args(argv[1:])
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)


def main(argv):

  # HACK: redirect the create_hparams function to setup the hparams
  # using the passed in command-line args
  argv = common_flags.update_argv(argv)
  t2t_decoder.create_hparams = functools.partial(create_hparams, argv)
  t2t_decoder.main(None)

if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
