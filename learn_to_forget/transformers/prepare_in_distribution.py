# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Prepares in-distribution examples from the wmt dataset validation set.

"""

from collections.abc import Sequence

from absl import app
import tensorflow as tf
import tensorflow_datasets as tfds

from learn_to_forget.transformers import constants


features = tfds.features.FeaturesDict({
    'inputs': tfds.features.Text(),
    'targets': tfds.features.Text(),
})


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # TODO(teobaluta) read config from somewhere and pick the same setting as the
  # out-of-distribution canaries
  num_examples = [512, 512, 512]
  num_repeats = [1, 10, 100]

  val_dataset = tfds.load(
      name='wmt_t2t_translate/de-en:1.0.0',
      split='validation',
      shuffle_files=False,
      data_dir=constants.TFDS_DATA_DIR)

  for num, repeats in zip(num_examples, num_repeats):
    records = val_dataset.take(num).as_numpy_iterator()

    # Train, validation and test splits all the same
    with tf.io.TFRecordWriter(
        constants.WMT_TFDS_DATA_PATTERN.format(repeats)
    ) as writer:
      for record in records:
        data = {
            'inputs':
                'translate German to English: {}'.format(
                    record['de'].decode('utf-8')),
            'targets':
                record['en'].decode('utf-8'),
        }
        for _ in range(repeats):
          serialized_record = features.serialize_example(data)
          writer.write(serialized_record)

if __name__ == '__main__':
  app.run(main)
