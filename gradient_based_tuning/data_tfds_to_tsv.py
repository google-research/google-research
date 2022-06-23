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
"""Writes WMT training data from tf.datasets to a TSV table."""
import csv
import os

from absl import app
from absl import flags
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_string('tfds_name', 'wmt19_translate/de-en', 'TFDS dataset name.')
flags.DEFINE_string('output_dir', '/tmp/tsv_data', 'Path to the output TSV.')
flags.DEFINE_string('source_language', 'de', 'Source language identifier.')
flags.DEFINE_string('target_language', 'en', 'Target language identifier.')
flags.DEFINE_float(
    'split_for_guidance_data',
    0.01,
    help='Proportion of training data to set aside for guidance dataset. Default to 1%.'
)

LOGGING_STEPS = 100000


def _read_string_tensor(tensor):
  raw_text = tensor.numpy().decode('utf-8')
  return ' '.join(raw_text.strip().split())


def write_data_to_tsv(
    output_dir,
    source_language,
    target_language,
    tfds_name,
    split_for_guidance_data,
):
  """Download data and write it to plain tsv for train, dev, and guide datasets.

  Args:
    output_dir: The dir to which the data will be written. Dirs will be
      recursively created if not already present.
    source_language: Source language of the translation data.
    target_language: Target language of the translation data.
    tfds_name: The name of the desired dataset in tfds. (ie.
      wmt19_translate/de-en). See
      https://www.tensorflow.org/datasets/catalog/wmt19_translate for more
        details.
    split_for_guidance_data: How much of the training data to set aside for
      guidance dataset. Defaults to 1%. Set to 0 to produce a full training
      split.
  """
  lang_pair = source_language + target_language

  if not 0 < split_for_guidance_data < 1:
    raise ValueError('split_for_guidance_data must be between 0 and 1: (%s)' %
                     split_for_guidance_data)
  output_file_train = os.path.join(
      output_dir, '{}_train_{:.0f}percent.tsv'.format(
          lang_pair, 100 * (1.0 - split_for_guidance_data)))
  output_file_guide = os.path.join(
      output_dir,
      '{}_guide_{:.0f}percent.tsv'.format(lang_pair,
                                          100 * (split_for_guidance_data)))

  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  with open(output_file_train, 'w') as outfile_train:
    with open(output_file_guide, 'w') as outfile_guide:
      csv_writer_train = csv.writer(outfile_train, delimiter='\t')
      csv_writer_guide = csv.writer(outfile_guide, delimiter='\t')
      for num_done_examples, example in enumerate(
          tfds.load(tfds_name, split='train')):
        if num_done_examples % LOGGING_STEPS == 0:
          print('%d train examples done.' % num_done_examples)
        if split_for_guidance_data > 0 and num_done_examples % (
            1 / split_for_guidance_data) == 0:
          csv_writer_guide.writerow([
              _read_string_tensor(example[source_language]),
              _read_string_tensor(example[target_language])
          ])
        else:
          csv_writer_train.writerow([
              _read_string_tensor(example[source_language]),
              _read_string_tensor(example[target_language])
          ])

  output_file_dev = os.path.join(output_dir, '{}_dev.tsv'.format(lang_pair))

  with open(output_file_dev, 'w') as outfile_dev:
    csv_writer_dev = csv.writer(outfile_dev, delimiter='\t')
    for num_done_examples, example in enumerate(
        tfds.load(tfds_name, split='validation')):
      csv_writer_dev.writerow([
          _read_string_tensor(example[source_language]),
          _read_string_tensor(example[target_language])
      ])
    print('%d validation examples done.' % num_done_examples)


def main(unused_args):
  write_data_to_tsv(
      output_dir=FLAGS.output_dir,
      source_language=FLAGS.source_language,
      target_language=FLAGS.target_language,
      tfds_name=FLAGS.tfds_name,
      split_for_guidance_data=FLAGS.split_for_guidance_data,
  )


if __name__ == '__main__':
  app.run(main)
