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

r"""Script to colorize or recolorize a directory of images.

Instructions
------------
1. Download pretrained models from
https://storage.cloud.google.com/gresearch/coltran/coltran.zip

2. Set the following variables:

* LOGDIR    - Checkpoint Directory to the corresponding checkpoints.
* IMG_DIR   - Directory with ground-truth grayscale or colored images.
* STORE_DIR - Directory to store generated images.
* MODE      - "colorize" if IMG_DIR consists of grayscale images
              "recolorize" if IMG_DIR consists of colored images.

2. Run the colorizer to get a coarsely colorized image. Set as follows:

python -m coltran.custom_colorize --config=configs/colorizer.py \
--logdir=$LOGDIR/colorizer --img_dir=$IMG_DIR --store_dir=$STORE_DIR \
--mode=$MODE

The generated images will be stored in $STORE_DIR/stage1

3. Run the color upsampler to upsample the coarsely colored image.

python -m coltran.custom_colorize --config=configs/color_upsampler.py \
--logdir=$LOGDIR/color_upsampler --img_dir=$IMG_DIR --store_dir=$STORE_DIR \
--gen_data_dir=$STORE_DIR/stage1 --mode=$MODE

The generated images will be stored in $STORE_DIR/stage2

4. Run the spatial upsampler to super-resolve into the final output.

python -m coltran.custom_colorize --config=configs/spatial_upsampler.py \
--logdir=$LOGDIR/spatial_upsampler --img_dir=$IMG_DIR --store_dir=$STORE_DIR \
--gen_data_dir=$STORE_DIR/stage2 --mode=$MODE

Notes
-----
* The model is pre-trained on ImageNet. Colorized images may reflect the biases
present in the ImageNet dataset.
* Once in a while, there can be artifacts or anomalous colorizations
due to accumulation of errors.
See Section M of https://openreview.net/pdf?id=5NA1PinlGFu
* Legacy images may have a different distribution as compared to the
grayscale images used to train the model. This might reflect in difference in
colorization fidelity between colorizing legacy images and our reported results.
* Setting "mode" correctly is important.
If img_dir consists of grayscale images, it should be set to "colorize"
if img_dir consists of colored images , it should be set to "recolorize".

"""
import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
from ml_collections import config_flags
import numpy as np

import tensorflow.compat.v2 as tf

from coltran import datasets
from coltran.models import colorizer
from coltran.models import upsampler
from coltran.utils import base_utils
from coltran.utils import datasets_utils
from coltran.utils import train_utils


flags.DEFINE_string('img_dir', None,
                    'Path for images needed to be colorized / recolorized.')
flags.DEFINE_string('logdir', '/tmp/svt', 'Checkpoint directory.')
flags.DEFINE_string('gen_data_dir', None,
                    'Path to images generated from the previous stages. '
                    'Has to be set if the model is the color or spatial '
                    'upsampler.')
flags.DEFINE_string('store_dir', None, 'Path to store generated images.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('tpu_worker_name', 'tpu_worker', 'Name of the TPU worker.')
flags.DEFINE_enum('accelerator_type', 'GPU', ['CPU', 'GPU', 'TPU'],
                  'Hardware type.')
flags.DEFINE_enum('mode', 'colorize', ['colorize', 'recolorize'],
                  'Whether to colorizer or recolorize images.')
flags.DEFINE_integer('steps_per_summaries', 100, 'Steps per summaries.')
flags.DEFINE_integer('batch_size', None,
                     'Batch size. If not provided, use the optimal batch-size '
                     'for each model.')
config_flags.DEFINE_config_file(
    'config',
    default='test_configs/colorizer.py',
    help_string='Training configuration file.')
FLAGS = flags.FLAGS


def create_grayscale_dataset_from_images(image_dir, batch_size):
  """Creates a dataset of grayscale images from the input image directory."""
  def load_and_preprocess_image(path, child_path):
    image_str = tf.io.read_file(path)
    num_channels = 1 if FLAGS.mode == 'colorize' else 3
    image = tf.image.decode_image(image_str, channels=num_channels)

    # Central crop to square and resize to 256x256.
    image = datasets.resize_to_square(image, resolution=256, train=False)

    # Resize to a low resolution image.
    image_64 = datasets_utils.change_resolution(image, res=64)
    if FLAGS.mode == 'recolorize':
      image = tf.image.rgb_to_grayscale(image)
      image_64 = tf.image.rgb_to_grayscale(image_64)
    return image, image_64, child_path

  child_files = tf.io.gfile.listdir(image_dir)
  files = [os.path.join(image_dir, file) for file in child_files]
  files = tf.convert_to_tensor(files, dtype=tf.string)
  dataset = tf.data.Dataset.from_tensor_slices((files, child_files))
  dataset = dataset.map(load_and_preprocess_image)
  return dataset.batch(batch_size=batch_size)


def build_model(config):
  """Builds model."""
  name = config.model.name
  optimizer = train_utils.build_optimizer(config)

  zero_64 = tf.zeros((1, 64, 64, 3), dtype=tf.int32)
  zero_64_slice = tf.zeros((1, 64, 64, 1), dtype=tf.int32)
  zero = tf.zeros((1, 256, 256, 3), dtype=tf.int32)
  zero_slice = tf.zeros((1, 256, 256, 1), dtype=tf.int32)

  if name == 'coltran_core':
    model = colorizer.ColTranCore(config.model)
    model(zero_64, training=False)
  elif name == 'color_upsampler':
    model = upsampler.ColorUpsampler(config.model)
    model(inputs=zero_64, inputs_slice=zero_64_slice, training=False)
  elif name == 'spatial_upsampler':
    model = upsampler.SpatialUpsampler(config.model)
    model(inputs=zero, inputs_slice=zero_slice, training=False)

  ema_vars = model.trainable_variables
  ema = train_utils.build_ema(config, ema_vars)
  return model, optimizer, ema


def get_batch_size(name):
  """Gets optimal batch-size based on model."""
  if FLAGS.batch_size is not None:
    return FLAGS.batch_size
  elif 'upsampler' in name:
    return 5
  return 20


def get_store_dir(name, store_dir):
  store_dict = {
      'coltran_core': 'stage1',
      'color_upsampler': 'stage2',
      'spatial_upsampler': 'final'}
  store_dir = os.path.join(store_dir, store_dict[name])
  tf.io.gfile.makedirs(store_dir)
  return store_dir


def main(_):
  config, store_dir, img_dir = FLAGS.config, FLAGS.store_dir, FLAGS.img_dir
  assert store_dir is not None
  assert img_dir is not None
  model_name, gen_data_dir = config.model.name, FLAGS.gen_data_dir
  needs_gen = model_name in ['color_upsampler', 'spatial_upsampler']

  batch_size = get_batch_size(model_name)
  store_dir = get_store_dir(model_name, store_dir)
  num_files = len(tf.io.gfile.listdir(img_dir))

  if needs_gen:
    assert gen_data_dir is not None
    gen_dataset = datasets.create_gen_dataset_from_images(gen_data_dir)
    gen_dataset = gen_dataset.batch(batch_size)
    gen_dataset_iter = iter(gen_dataset)

  dataset = create_grayscale_dataset_from_images(FLAGS.img_dir, batch_size)
  dataset_iter = iter(dataset)

  model, optimizer, ema = build_model(config)
  checkpoints = train_utils.create_checkpoint(model, optimizer=optimizer,
                                              ema=ema)
  train_utils.restore(model, checkpoints, FLAGS.logdir, ema)
  num_steps_v = optimizer.iterations.numpy()
  logging.info('Producing sample after %d training steps.', num_steps_v)

  num_epochs = int(np.ceil(num_files / batch_size))
  logging.info(num_epochs)

  for _ in range(num_epochs):
    gray, gray_64, child_paths = next(dataset_iter)

    if needs_gen:
      prev_gen = next(gen_dataset_iter)

    if model_name == 'coltran_core':
      out = model.sample(gray_64, mode='sample')
      samples = out['auto_sample']
    elif model_name == 'color_upsampler':
      prev_gen = base_utils.convert_bits(prev_gen, n_bits_in=8, n_bits_out=3)
      out = model.sample(bit_cond=prev_gen, gray_cond=gray_64)
      samples = out['bit_up_argmax']
    else:
      prev_gen = datasets_utils.change_resolution(prev_gen, 256)
      out = model.sample(gray_cond=gray, inputs=prev_gen, mode='argmax')
      samples = out['high_res_argmax']

    child_paths = child_paths.numpy()
    child_paths = [child_path.decode('utf-8') for child_path in child_paths]
    logging.info(child_paths)

    for sample, child_path in zip(samples, child_paths):
      write_path = os.path.join(store_dir, child_path)
      logging.info(write_path)
      sample = sample.numpy().astype(np.uint8)
      logging.info(sample.shape)
      with tf.io.gfile.GFile(write_path, 'wb') as f:
        plt.imsave(f, sample)


if __name__ == '__main__':
  app.run(main)
