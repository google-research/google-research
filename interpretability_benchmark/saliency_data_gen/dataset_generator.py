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

# Lint as: python3
r"""This script allows generation of tfrecords.


"""

import os
from absl import app
from absl import flags
import numpy as np
from scipy import ndimage
from six.moves import range
import tensorflow.compat.v1 as tf
from interpretability_benchmark.saliency_data_gen.data_helper import DataIterator
from interpretability_benchmark.saliency_data_gen.data_helper import image_to_tfexample
from interpretability_benchmark.saliency_data_gen.data_helper import SALIENCY_BASELINE
from interpretability_benchmark.saliency_data_gen.saliency_helper import generate_saliency_image
from interpretability_benchmark.saliency_data_gen.saliency_helper import get_saliency_image
from interpretability_benchmark.utils import resnet_model
tf.disable_v2_behavior()

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('output_dir', '/tmp/saliency/',
                    'output directory for tfrecords')

flags.DEFINE_string('data_path', '', 'Pathway to the input tfrecord dataset.')

flags.DEFINE_string('ckpt_path', '', 'Pathway to the trained checkpoint.')

flags.DEFINE_enum(
    'split', 'validation', ('training', 'validation'),
    'Specifies whether to create saliency maps for'
    'training or test set.')

flags.DEFINE_enum('dataset_name', 'imagenet',
                  ('food_101', 'imagenet', 'birdsnap'),
                  'What dataset is the model trained on.')

flags.DEFINE_enum('saliency_method', 'SH_SG',
                  ('SH_SG', 'IG_SG', 'GB_SG', 'SH_SG_2', 'IG_SG_2', 'GB_SG_2',
                   'GB', 'IG', 'SH', 'SOBEL'),
                  'saliency method dataset to produce.')
flags.DEFINE_bool('test_small_sample', True,
                  'Boolean for whether to test internally.')
FLAGS = flags.FLAGS

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

N_CLASSES = {'imagenet': 1000, 'food_101': 101, 'birdsnap': 500}


class ProcessSaliencyMaps(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, dataset_name, saliency_method, ckpt_directory,
               num_label_classes):
    # Create a single Session to run all image coding calls.

    self._dataset_name = dataset_name
    self._saliency_method = saliency_method
    self._ckpt_directory = ckpt_directory
    self._num_label_classes = num_label_classes

  def produce_saliency_map(self, data_path, writer):
    """produces a saliency map."""

    self._dataset = DataIterator(
        data_path,
        self._dataset_name,
        preprocessing=False,
        test_small_sample=FLAGS.test_small_sample)

    self._graph = tf.Graph()
    with self._graph.as_default():
      image_raw, image_processed, label = self._dataset.input_fn()

      image_processed -= tf.constant(
          MEAN_RGB, shape=[1, 1, 3], dtype=image_processed.dtype)
      image_processed /= tf.constant(
          STDDEV_RGB, shape=[1, 1, 3], dtype=image_processed.dtype)

      network = resnet_model.resnet_50(
          num_classes=self._num_label_classes,
          data_format='channels_last',
      )

      logits = network(inputs=image_processed, is_training=False)

      prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)

      self._neuron_selector = tf.placeholder(tf.int32)

      y = logits[0][self._neuron_selector]

      self._sess = tf.Session(graph=self._graph)
      saver = tf.train.Saver()

      saver.restore(self._sess, self._ckpt_directory)

      self._gradient_placeholder = get_saliency_image(
          self._graph, self._sess, y, image_processed, 'gradient')
      self._back_prop_placeholder = get_saliency_image(
          self._graph, self._sess, y, image_processed, 'guided_backprop')
      self._integrated_gradient_placeholder = get_saliency_image(
          self._graph, self._sess, y, image_processed, 'integrated_gradients')

      baseline = SALIENCY_BASELINE['resnet_50']

      self._coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=self._sess, coord=self._coord)

      example_count = 0
      try:
        while True:
          img_out, raw_img_out, label_out, prediction_out = self._sess.run(
              [image_processed, image_raw, label, prediction])
          if img_out.shape[3] == 3:
            img_out = np.squeeze(img_out)

            feed_dict = {self._neuron_selector: prediction_out[0]}
            if self._saliency_method != 'SOBEL':
              saliency_map = generate_saliency_image(
                  self._saliency_method, img_out, feed_dict,
                  self._gradient_placeholder, self._back_prop_placeholder,
                  self._integrated_gradient_placeholder, baseline)
            else:
              saliency_map = ndimage.sobel(img_out, axis=0)

          saliency_map = saliency_map.astype(np.float32)
          saliency_map = np.reshape(saliency_map, [-1])
          example = image_to_tfexample(
              raw_image=raw_img_out[0], maps=saliency_map, label=label_out)
          writer.write(example.SerializeToString())
          example_count += 1

          if FLAGS.test_small_sample:
            if example_count == 2:
              break

      except tf.errors.OutOfRangeError:
        print('Finished number of images:', example_count)
      finally:
        self._coord.request_stop()
        self._coord.join(threads)
        writer.close()


def generate_dataset(data_directory, dataset_name, num_shards, output_directory,
                     ckpt_directory, num_label_classes, filenames,
                     saliency_method):
  """Generate a dataset."""

  data_gen = ProcessSaliencyMaps(
      dataset_name=dataset_name,
      ckpt_directory=ckpt_directory,
      num_label_classes=num_label_classes,
      saliency_method=saliency_method)

  counter = 0
  for i in range(num_shards):
    filename = filenames[i]
    data_path = data_directory + filename
    output_file = os.path.join(output_directory, filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    _ = data_gen.produce_saliency_map(data_path, writer)
    counter += 1
    print('Finished shard number:', counter)

  print('Finished outputting all records to the directory.')


def main(argv):
  del argv  # Unused.

  if FLAGS.test_small_sample:
    filenames = ['test_small_sample']
    num_shards = 1
    output_dir = FLAGS.output_dir
  else:
    output_dir = ('%s/%s/%s/%s' % (FLAGS.output_dir, FLAGS.dataset_name,
                                   'resnet_50', FLAGS.saliency_method))
    filenames = tf.gfile.ListDirectory(FLAGS.data_path)
    num_shards = len(filenames)

  generate_dataset(
      data_directory=FLAGS.data_path,
      output_directory=output_dir,
      num_shards=num_shards,
      dataset_name=FLAGS.dataset_name,
      ckpt_directory=FLAGS.ckpt_directory,
      num_label_classes=N_CLASSES[FLAGS.dataset_name],
      filenames=filenames,
      saliency_method=FLAGS.saliency_method)


if __name__ == '__main__':
  app.run(main)
