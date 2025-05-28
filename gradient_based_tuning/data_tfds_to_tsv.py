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
"""Writes WMT training data from tfds to a TSV file, and generates spm model."""
import csv
import os
import tempfile

from absl import app
from absl import flags
from sentencepiece import SentencePieceTrainer
import tensorflow.compat.v2 as tf
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

  tf.io.gfile.makedirs(os.path.dirname(output_dir))
  guide_example_count = 0
  train_example_count = 0
  print('Writing train output to: %s' % output_file_train)
  print('Writing guide output to: %s' % output_file_guide)
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
          guide_example_count += 1
        else:
          csv_writer_train.writerow([
              _read_string_tensor(example[source_language]),
              _read_string_tensor(example[target_language])
          ])
          train_example_count += 1

  print('Num train examples: %d' % train_example_count)
  print('Num guide examples: %d' % guide_example_count)
  output_file_dev = os.path.join(output_dir, '{}_dev.tsv'.format(lang_pair))

  with open(output_file_dev, 'w') as outfile_dev:
    csv_writer_dev = csv.writer(outfile_dev, delimiter='\t')
    for num_done_examples, example in enumerate(
        tfds.load(tfds_name, split='validation')):
      csv_writer_dev.writerow([
          _read_string_tensor(example[source_language]),
          _read_string_tensor(example[target_language])
      ])
    print('Num validation examples: %d' % num_done_examples)


def generate_vocab(
    output_dir,
    source_language,
    target_language,
    tfds_name,
):
  """Train a sentencepiece vocab on a portion of the data.

  Args:
    output_dir: The dir to which the data will be written. Dirs will be
      recursively created if not already present.
    source_language: Source language of the translation data.
    target_language: Target language of the translation data.
    tfds_name: The name of the desired dataset in tfds. (ie.
      wmt19_translate/de-en). See
      https://www.tensorflow.org/datasets/catalog/wmt19_translate for more
        details.
  """

  tf.io.gfile.makedirs(os.path.dirname(output_dir))
  train_ds = tfds.load(tfds_name, split='train')

  vocab_file = os.path.join(
      output_dir, '{}.32k.spm.model'.format(source_language + target_language))
  print('vocab_file %s' % vocab_file)
  _train_sentencepiece(
      dataset=train_ds,
      model_path=vocab_file,
      vocab_size=2**15,
      data_keys=(source_language, target_language))


def _dump_chars_to_textfile(dataset,
                            maxchars=int(1e7),
                            data_keys=('inputs', 'targets')):
  """Write part of a TFDS sentence dataset to lines in a text file.

  Args:
    dataset: tf.dataset containing string-data.
    maxchars: int: approximate number of characters to save from dataset.
    data_keys: Tuple[str]: what keys in dataset to dump from.

  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  char_count = 0
  ds_iter = dataset.as_numpy_iterator()
  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/ds_chars') as outfp:
    while char_count < maxchars:
      example = next(ds_iter)
      for k in data_keys:
        line = example[k] + b'\n'
        char_count += len(line)
        outfp.write(line)
  return outfp.name, char_count


def _train_sentencepiece(dataset,
                         model_path,
                         vocab_size=2**15,
                         maxchars=int(1e7),
                         model_type='unigram',
                         character_coverage=1.0,
                         data_keys=('inputs', 'targets')):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    dataset: tf.dataset
    model_path: str: path of model file to save vocab model to.
    vocab_size: int: size of vocab tokens to train.
    maxchars: int: number of characters to use for sentencepiece training.
    model_type: str: type of sentencepiece vocab to train.
    character_coverage: amount of characters covered by the model, good defaults
      are 0.9995 for languages with rich character set like Japanese or Chinese
      and 1.0 for other languages with small character set.
    data_keys: Tuple[str]: keys of dataset to use for training.

  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  if model_path.startswith('gs://'):
    abs_model_path = model_path
  else:
    abs_model_path = os.path.abspath(os.path.expanduser(model_path))
  fname, _ = _dump_chars_to_textfile(
      dataset, maxchars=maxchars, data_keys=data_keys)
  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/sp_tmp') as model_fp:
    pass  # we just want a prefix'd tmp-filename
  argstr = ' '.join([
      f'--input={fname}', f'--vocab_size={vocab_size}',
      f'--character_coverage={character_coverage}',
      f'--model_prefix={model_fp.name}', f'--model_type={model_type}'
  ])
  SentencePieceTrainer.Train(argstr)
  # Use an intermediate filename that is renamed to the target name to address
  # create and fill delays.
  copy_rename_path = abs_model_path + '.rntmp'
  tf.io.gfile.copy(model_fp.name + '.model', copy_rename_path, overwrite=True)
  tf.io.gfile.rename(copy_rename_path, abs_model_path, overwrite=True)
  print('copied %s to %s' % (model_fp.name + '.model', abs_model_path))
  return abs_model_path


def main(unused_args):
  tf.io.gfile.makedirs(FLAGS.output_dir)
  print('Generating vocab.')
  generate_vocab(
      output_dir=FLAGS.output_dir,
      source_language=FLAGS.source_language,
      target_language=FLAGS.target_language,
      tfds_name=FLAGS.tfds_name,
  )
  print('Saving data.')
  write_data_to_tsv(
      output_dir=FLAGS.output_dir,
      source_language=FLAGS.source_language,
      target_language=FLAGS.target_language,
      tfds_name=FLAGS.tfds_name,
      split_for_guidance_data=FLAGS.split_for_guidance_data,
  )


if __name__ == '__main__':
  app.run(main)
