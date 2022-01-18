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

"""Binary for downloading waymo open dataset frames tfrecords."""

import os

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'target_dir', '',
    'Path to the local folder that the tfrecords will be downloaded to.')


def main(_):
  data_shard_info = {'train': 1212, 'val': 471, 'test': 528}

  if not os.path.exists(FLAGS.target_dir):
    os.mkdir(FLAGS.target_dir)

  for key, value in data_shard_info.items():
    for i in range(value):
      file_url = (
          'gs://waymo_open_dataset_tf_example_tf3d/original_tfrecords/%s-%05d-of-%05d.tfrecords'
          % (key, i, value))
      logging.info(file_url)
      assert os.system(
          'gsutil cp ' + file_url + ' ' +
          FLAGS.target_dir) == 0, 'Failed to download %s' % file_url


if __name__ == '__main__':
  app.run(main)
