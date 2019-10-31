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

# Lint as: python3
r"""Generation tfrecords for pie and non-pie images.


"""

import os
import time
from absl import app
from absl import flags
import pandas as pd
import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.utils import data_input

flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_enum('mode', 'eval', ('eval', 'train'),
                  'Mode designated as train or eval.')
flags.DEFINE_string('predictions_file', '',
                    'Input file to pull aggregate prediction measures from.')
flags.DEFINE_string('output_file', '', 'File to save the tfrecords to.')
flags.DEFINE_float('end_sparsity', 0.5,
                   'Target sparsity desired by end of training.')
flags.DEFINE_string('subdirectory',
                    'imagenet-with-bbox-validation-00017-of-00128',
                    'Input filename')
flags.DEFINE_enum('dataset_name', 'imagenet', ('imagenet'),
                  'name of dataset used')
flags.DEFINE_enum('is_pie', 'pie', ('pie', 'non_pie'),
                  'whether to store PIE or non_pie images')
flags.DEFINE_string('data_directory', '',
                    'The location of the sstable used for training.')
flags.DEFINE_integer('num_cores', default=8, help=('Number of cores.'))
# set this flag to true to do a test run of this code with synthetic data
flags.DEFINE_bool('test_small_sample', False,
                  'Boolean for whether to test internally.')

FLAGS = flags.FLAGS


class GenNewImageNet(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, dataset_name, subdirectory):
    # Create a single Session to run all image coding calls.
    self._dataset_name = dataset_name
    self._subdirectory = subdirectory

  def produce_data_set(self, data_path, df, output_file):
    """produces pie or non-pie tfrecords."""

    self._graph = tf.Graph()
    with self._graph.as_default():
      params = {}
      params['batch_size'] = 1
      params['data_dir'] = data_path
      params['num_cores'] = FLAGS.num_cores
      params['is_training'] = False
      params['task'] = 'pie_dataset_gen'
      params['mode'] = 'eval'
      if FLAGS.test_small_sample:
        update_params = {
            'train_batch_size': 2,
            'eval_batch_size': 2,
            'num_train_steps': 10,
            'num_images': 2,
            'num_train_images': 10,
            'num_eval_images': 10,
            'batch_size': 2
        }
        params['test_small_sample'] = True
        params.update(update_params)
      else:
        params['test_small_sample'] = False

      data = data_input.input_fn(params)
      image_raw = data['image_raw']
      label = data['label']
      key_image = data['key_']

      session_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=True,
          intra_op_parallelism_threads=1,
          inter_op_parallelism_threads=1)

      self._sess = tf.Session(graph=self._graph, config=session_config)

      self._coord = tf.Coordinator()
      output_file = output_file + self._subdirectory

      writer = tf.python_io.TFRecordWriter(output_file)
      max_value = df['image_index'].max()
      example_count = 0
      count = 0
      pie_count = 0
      pruned_correct_count = 0

      threads = tf.train.start_queue_runners(self._sess, self._coord)
      tf.logging.info('Started queue runners')

      try:
        with self._coord.stop_on_exception():
          while not self._coord.should_stop() and (example_count <
                                                   max_value + 1):
            raw_img_out, label_out, key_image_out = self._sess.run(
                [image_raw, label, key_image])

            stored_class_label = df['true_class_label'][example_count].max()
            image_index = df['image_index'][example_count].max()
            variant_mode_label = df['predicted_mode_variant_label'][
                example_count].max()
            baseline_mode_label = df['baseline_mode_base_label'][
                example_count].max()
            baseline_n = df['baseline_number_observ'][example_count].max()
            variant_n = df['variant_number_observ'][example_count].max()
            baseline_mean_probability = df['baseline_mean_probability'][
                example_count].max()
            variant_mean_probability = df['variant_mean_probability'][
                example_count].max()
            pruning_fraction = df['pruning_fraction'][example_count].max()

            if variant_mode_label != baseline_mode_label:
              is_pie = 'pie'
            else:
              is_pie = 'non_pie'

            if (variant_mode_label != baseline_mode_label) and (
                variant_mode_label == stored_class_label):
              pruned_correct_count += 1

            if variant_mode_label != baseline_mode_label:
              pie_count += 1

            # flags is_pie indicates to create tfrecords for pie or non-pie
            if is_pie == FLAGS.is_pie:
              example = data_input.image_to_tfexample(
                  key=key_image_out[0],
                  raw_image=raw_img_out[0],
                  image_index=image_index,
                  label=label_out,
                  stored_class_label=stored_class_label,
                  variant_mode_label=variant_mode_label,
                  baseline_mode_label=baseline_mode_label,
                  pruning_fraction=pruning_fraction,
                  baseline_n=baseline_n,
                  variant_n=variant_n,
                  baseline_mean_probability=baseline_mean_probability,
                  variant_mean_probability=variant_mean_probability,
                  test_small_sample=FLAGS.test_small_sample)
              writer.write(example.SerializeToString())
              count += 1
            example_count += 1

        tf.logging.info('Finished number of images:', count)
        writer.close()
        return example_count, pie_count, pruned_correct_count
      finally:
        self._coord.request_stop()
        self._coord.join(threads)
        writer.close()


def generate_dataset(data_directory, predictions_dir, dataset_name,
                     output_directory, subdirectory, file_path):
  """Generates dataset."""

  predictions_path = os.path.join(predictions_dir, file_path)
  with tf.gfile.Open(predictions_path) as f:
    df = pd.read_csv(f)

  output_dir = ('%s/%s/%s/%s/%s/' %
                (output_directory, FLAGS.dataset_name, FLAGS.mode, FLAGS.is_pie,
                 str(FLAGS.end_sparsity)))

  data_gen = GenNewImageNet(
      dataset_name=dataset_name, subdirectory=subdirectory)
  example_count, pie_count, pruned_correct_count = data_gen.produce_data_set(
      data_path=data_directory, df=df, output_file=output_dir)

  output_dir_stats = ('%s/%s/%s/' %
                      (output_directory, FLAGS.dataset_name, 'stats'))

  timestamp = str(time.time())
  filename = FLAGS.mode + '_' + str(subdirectory) + '_' + timestamp + '_' + str(
      FLAGS.end_sparsity)
  df = pd.DataFrame()
  df['file'] = subdirectory
  df['pie_count'] = pie_count
  df['pruned_correct_count'] = pruned_correct_count
  df['total_images_count'] = example_count
  df['sparsity'] = FLAGS.end_sparsity
  df['mode'] = FLAGS.mode

  with tf.gfile.Open(os.path.join(output_dir_stats, filename), 'wb') as f:
    df.to_csv(f, sep='\t', header=True, index=False)


def main(argv):
  del argv  # Unused.

  data_directory = os.path.join(FLAGS.data_directory, FLAGS.subdirectory)
  predictions_directory = os.path.join(FLAGS.predictions_file,
                                       FLAGS.dataset_name, FLAGS.mode,
                                       FLAGS.subdirectory)

  filenames = tf.gfile.Glob(predictions_directory + '/' +
                            str(FLAGS.end_sparsity) + '*.csv')

  generate_dataset(
      data_directory=data_directory,
      predictions_dir=predictions_directory,
      dataset_name=FLAGS.dataset_name,
      output_directory=FLAGS.output_file,
      subdirectory=FLAGS.subdirectory,
      file_path=filenames[0])


if __name__ == '__main__':
  app.run(main)
