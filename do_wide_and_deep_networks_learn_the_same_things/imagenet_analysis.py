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

"""Compute internal CKA for ImageNet models."""

import functools
from itertools import combinations
import json
import os
import pickle
import random
import re
import sys
from absl import app
from absl import flags
from absl import logging
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import entropy
import tensorflow.compat.v2 as tf
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from do_wide_and_deep_networks_learn_the_same_things.efficient_CKA import *
from do_wide_and_deep_networks_learn_the_same_things.large_scale_training import alt_resnet
from do_wide_and_deep_networks_learn_the_same_things.large_scale_training.pc_imagenet_train import load_test_data
from tensorflow_models.official.vision.image_classification import preprocessing

tf.enable_v2_behavior()

FLAGS = flags.FLAGS
flags.DEFINE_integer('cka_batch', 256, 'Batch size used to approximate CKA')
flags.DEFINE_integer('cka_iter', 10,
                     'Number of iterations to run minibatch CKA approximation')
flags.DEFINE_string('model_dir', '', 'Path to where the trained model is saved')
flags.DEFINE_integer('model_depth', 14,
                     'Only run analysis for models of this depth')
flags.DEFINE_float('model_width', 1,
                   'Only run analysis for models of this width multiplier')
flags.DEFINE_string('experiment_dir', None,
                    'Path to where the trained model is saved')
flags.DEFINE_boolean('scale_all', False, 'whether all layers are scaled')
flags.DEFINE_integer('num_classes', 1000, 'number of classes')


def preprocess_data_imagenet(data, is_training):
  """ImageNet data preprocessing"""
  if is_training:
    image = preprocessing.preprocess_for_train(data['image'], image_size=224)
  else:
    image = preprocessing.preprocess_for_eval(data['image'], image_size=224)
  return (image, data['label'])


def load_test_data_imagenet(batch_size, shuffle=False):
  """Load ImageNet test data"""
  builder = tfds.builder('imagenet2012', version='5.1.0')
  decoders = {'image': tfds.decode.SkipDecoding()}
  test_dataset = builder.as_dataset(tfds.Split.VALIDATION, decoders=decoders)
  test_dataset = test_dataset.map(
      functools.partial(preprocess_data_imagenet, is_training=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
  if shuffle:
    test_dataset = test_dataset.shuffle(buffer_size=10000)
  test_dataset = test_dataset.batch(batch_size)
  test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  test_dataset_size = builder.info.splits[tfds.Split.VALIDATION].num_examples
  return test_dataset


def count_layers(model):
  """Count the total number of layers in an ImageNet model"""
  res = model({'input_1': tf.zeros((1, 64, 64, 3))}, return_layers=True)
  count = 0
  for sublist in res:
    if isinstance(sublist, list):  #block group
      for subsublist in sublist:  #each block
        count += len(subsublist)
    else:
      count += 1
  return count


def get_activations_imagenet(images, model):
  """Return a list of activations obtained from a model on a set of ImageNet images."""
  activations = model({'input_1': images}, return_layers=True)
  output = []

  def remove_nestings(l):
    for i in l:
      if type(i) == list:
        remove_nestings(i)
      else:
        output.append(i)

  remove_nestings(activations)
  return output


@tf.function(experimental_compile=True)
def process_batch(model, images, cka):
  cka.update_state(get_activations_imagenet(images, model))


def compute_cka_internal_imagenet(model, cka, data_path=None, use_batch=True):
  """Compute CKA score of each layer in a model to every other layer in the same model."""
  if use_batch:
    for _ in range(FLAGS.cka_iter):
      dataset = load_test_data(
          FLAGS.cka_batch
      )  #load_test_data_imagenet(FLAGS.cka_batch, shuffle=True)
      for images in dataset:
        process_batch(model, images['input_1'], cka)

  else:
    dataset = load_test_data_imagenet(FLAGS.cka_batch)
    all_images = tf.concat([x[0] for x in dataset], 0)
    process_batch(model, all_images, cka)

  return cka.result()


def main(argv):
  filename = 'cka_within_model_%d.pkl' % FLAGS.cka_batch
  out_dir = os.path.join(FLAGS.experiment_dir, filename)
  if tf.io.gfile.exists(out_dir):
    return

  DEPTH = FLAGS.model_depth
  WIDTH = FLAGS.model_width

  if FLAGS.scale_all:
    #Models with all layers scaled:
    LAYERS = [int(round(DEPTH * x)) for x in [3, 4, 6, 3]]
    WIDTH_MULTIPLIERS = [1] + [WIDTH] * 4
  else:
    # Models with scaled stage 3:
    LAYERS = [3, 4, 6 * DEPTH, 3]
    WIDTH_MULTIPLIERS = [1, 1, 1, WIDTH, 1]

  model = alt_resnet.Resnet(
      block_fn=alt_resnet.BottleneckBlock,
      layers=LAYERS,
      width_multipliers=WIDTH_MULTIPLIERS,
      num_classes=FLAGS.num_classes,
      kernel_regularizer=tf.keras.regularizers.l2(5e-5))
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_path = os.path.join(FLAGS.experiment_dir, 'checkpoints')
  ckpt = [f for f in tf.io.gfile.listdir(checkpoint_path) if f.startswith('ckpt-')][0]
  ckpt = ckpt.split('.')[0]
  checkpoint_path = os.path.join(checkpoint_path, ckpt)

  n_layers = count_layers(model)
  cka = MinibatchCKA(n_layers)
  heatmap = compute_cka_internal_imagenet(model, cka)
  heatmap = heatmap.numpy()
  logging.info(out_dir)
  with tf.io.gfile.GFile(out_dir, 'wb') as f:
    pickle.dump(heatmap, f)


if __name__ == '__main__':
  app.run(main)
