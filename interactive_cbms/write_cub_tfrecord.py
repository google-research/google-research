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

"""Writes train and test TFRecords using splits provided by [1].

[1] Pang Wei Koh*, Thao Nguyen*, Yew Siang Tang*, Stephen Mussmann,
Emma Pierson, Been Kim, and Percy Liang. Concept Bottleneck Models, ICML 2020.
"""

import os
import pickle
import random
import tempfile
from typing import Any, Dict, Sequence

from absl import app
from absl import logging
from PIL import Image
import tensorflow as tf
import tqdm

from interactive_cbms import network
from interactive_cbms.datasets import preproc_util

# pylint: disable=g-bad-import-order

tfk = tf.keras

_WRITE_DIR = 'interactive_cbms/datasets/CUB_certainty_TFRecord_test'
_READ_DIR = ('interactive_cbms/datasets/CUB_processed/'
             'class_attr_data_10')
_PRETRAINED_MODEL_PATH = ('ICBM_checkpoints/XtoCtoY/'
                          'sgd_lr-0.01_wd-4e-05/checkpoint')
os.makedirs(_WRITE_DIR)


def load_pretrained_model():
  """Loads the pretrained model for feature extraction."""
  model = network.InteractiveBottleneckModel(arch='XtoCtoY')
  model.compile(optimizer='sgd')
  model.build([None, 299, 299, 3])
  model.load_weights(_PRETRAINED_MODEL_PATH)
  return model


def convert_to_tf_example(example,
                          tempdir, model,
                          extract_features):
  """Converts an instance from the CUB dataset to a tf.train.Example proto."""
  example['img_path'] = example['img_path'][example['img_path']
                                            .find('CUB_200_2011'):]
  image = tf.convert_to_tensor(
      Image.open(f'{tempdir}/{example["img_path"]}').convert('RGB'))
  feature = {
      'image':
          tf.train.Feature(
              bytes_list=tf.train.BytesList(
                  value=[tf.io.serialize_tensor(image).numpy()])),
      'img_path':
          tf.train.Feature(
              bytes_list=tf.train.BytesList(
                  value=[example['img_path'].encode('utf-8')])),
      'attribute_label':
          tf.train.Feature(
              int64_list=tf.train.Int64List(
                  value=example['attribute_label'])),
      'attribute_certainty':
          tf.train.Feature(
              int64_list=tf.train.Int64List(
                  value=example['attribute_certainty'])),
      'class_label':
          tf.train.Feature(
              int64_list=tf.train.Int64List(
                  value=[example['class_label']])),
  }

  if extract_features:
    cropped = preproc_util.center_crop(
        image, height=299, width=299, crop_proportion=1)
    feat = model.gap(model.base_model(cropped[None, :, :, :]))[0]
    feature['feature'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=feat.numpy().tolist()))

  tf_example = tf.train.Example(features=tf.train.Features(
      feature=feature)).SerializeToString()
  return tf_example


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  random.seed(0)

  tempdir = 'datasets/'
  model = load_pretrained_model()

  for split in ['train', 'val', 'train_and_val', 'test']:
    logging.info('Writing %s split', split)

    if split == 'train_and_val':
      with open(os.path.join(_READ_DIR, 'train.pkl'), 'rb') as f:
        data = pickle.load(f)
      with open(os.path.join(_READ_DIR, 'val.pkl'), 'rb') as f:
        data.extend(pickle.load(f))
    else:
      with open(os.path.join(_READ_DIR, f'{split}.pkl'), 'rb') as f:
        data = pickle.load(f)

    if split != 'test':
      random.shuffle(data)


    with tf.io.TFRecordWriter(os.path.join(_WRITE_DIR, f'{split}.tfrecord')
                              ) as writer:
      for example in tqdm.tqdm(data):
        tf_example = convert_to_tf_example(example, tempdir, model,
                                           extract_features=False)
        writer.write(tf_example)


if __name__ == '__main__':
  app.run(main)
