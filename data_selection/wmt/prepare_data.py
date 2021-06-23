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

"""Download and prepare TFDS data."""

from absl import app
from absl import flags
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir', default=None,
    help='Directory to store data.')


def main(_):
  newscommentary_config = tfds.translate.wmt.WmtConfig(
      version='1.0.0',
      language_pair=('de', 'en'),
      subsets={
          tfds.Split.TRAIN: ['newscommentary_v13'],
          tfds.Split.VALIDATION: ['newscommentary_v13'],
      },
      name='newscommentary')
  paracrawl_config = tfds.translate.wmt.WmtConfig(
      version='1.0.0',
      language_pair=('de', 'en'),
      subsets={
          tfds.Split.TRAIN: ['paracrawl_v1'],
      },
      name='paracrawl')

  nc_builder = tfds.builder(
      'wmt_translate',
      config=newscommentary_config,
      data_dir=FLAGS.data_dir)
  para_builder = tfds.builder(
      'wmt_translate',
      config=paracrawl_config,
      data_dir=FLAGS.data_dir)
  nc_builder.download_and_prepare()
  para_builder.download_and_prepare()


if __name__ == '__main__':
  app.run(main)
