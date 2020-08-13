# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A binary for training depth and egomotion."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app

from depth_and_motion_learning import depth_motion_field_model
from depth_and_motion_learning import training_utils


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  training_utils.train(depth_motion_field_model.input_fn,
                       depth_motion_field_model.loss_fn,
                       depth_motion_field_model.get_vars_to_restore_fn)


if __name__ == '__main__':
  app.run(main)
