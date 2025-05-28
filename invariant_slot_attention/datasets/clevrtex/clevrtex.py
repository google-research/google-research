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

"""CLEVRTex dataset."""
import functools
import os
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


_DESCRIPTION = """
The CLEVRTex dataset with training, validation and testing splits.
In addition, there are six variants: camo (camouflaged objects),
grassbg (grass background), pbg (plain background), vbg (varied background)
and outd (unseen materials).
"""

_CITATION = """
@inproceedings{karazija21clevrtex,
  author    = {Laurynas Karazija and
               Iro Laina and
               Christian Rupprecht},
  editor    = {Joaquin Vanschoren and
               Sai{-}Kit Yeung},
  title     = {ClevrTex: {A} Texture-Rich Benchmark for Unsupervised Multi-Object
               Segmentation},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on
               Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December
               2021, virtual},
  year      = {2021}
}
"""

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('ZLIB')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = [
    'image', 'segmentations', 'depth', 'albedo', 'normal', 'shadow']

BASE_DIR = 'tfrecords'

TRAIN_TF_RECORDS_PATH = os.path.join(
    BASE_DIR, 'clevrtex_full/clevrtex_full_240x320.tfrecords')
CAMO_TF_RECORDS_PATH = os.path.join(
    BASE_DIR, 'clevrtex_camo/clevrtex_camo_240x320.tfrecords')
GRASSBG_TF_RECORDS_PATH = os.path.join(
    BASE_DIR, 'clevrtex_grassbg/clevrtex_grassbg_240x320.tfrecords')
PBG_TF_RECORDS_PATH = os.path.join(
    BASE_DIR, 'clevrtex_pbg/clevrtex_pbg_240x320.tfrecords')
VBG_TF_RECORDS_PATH = os.path.join(
    BASE_DIR, 'clevrtex_vbg/clevrtex_vbg_240x320.tfrecords')
OUTD_TF_RECORDS_PATH = os.path.join(
    BASE_DIR, 'clevrtex_outd/clevrtex_outd_240x320.tfrecords')


# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image':
        tf.io.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
    'segmentations':
        tf.io.FixedLenFeature(IMAGE_SIZE, tf.string),
    'depth':
        tf.io.FixedLenFeature(IMAGE_SIZE, tf.string),
    'albedo':
        tf.io.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
    'normal':
        tf.io.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
    'shadow':
        tf.io.FixedLenFeature(IMAGE_SIZE, tf.string),
    'num_objects':
        tf.io.FixedLenFeature([1], tf.int64),
    'image_index':
        tf.io.FixedLenFeature([1], tf.int64),
    'object_3d_coords':
        tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'object_pixel_coords':
        tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'object_rotations':
        tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32)
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


class ClevrTex(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for CLEVRTex dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self):
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(240, 320, 3)),
            'segmentations': tfds.features.Image(shape=(240, 320, 1)),
            'depth': tfds.features.Image(shape=(240, 320, 1)),
            'albedo': tfds.features.Image(shape=(240, 320, 3)),
            'normal': tfds.features.Image(shape=(240, 320, 3)),
            'shadow': tfds.features.Image(shape=(240, 320, 1)),
            'instances': tfds.features.Sequence(feature={
                'bboxes': tfds.features.BBoxFeature(),
                'positions': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
                # "(px, py, pz): px and py give 2D image-space coordinates;
                # pz gives depth"
                # https://github.com/karazijal/clevrtex-generation/blob/main/clevrtex-gen/blender_utils.py
                'image_positions': tfds.features.Tensor(
                    shape=(3,), dtype=tf.float32),
                'rotation': tf.float32
            }),
            'num_instances': tf.uint16,
            'image_index': tf.int64,
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://www.robots.ox.ac.uk/~vgg/data/clevrtex/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return {
        'train': self._generate_examples(
            TRAIN_TF_RECORDS_PATH, skip=10000),
        'validation': self._generate_examples(
            TRAIN_TF_RECORDS_PATH, skip=5000, take=5000),
        'testing': self._generate_examples(
            TRAIN_TF_RECORDS_PATH, take=5000),
        'camo': self._generate_examples(CAMO_TF_RECORDS_PATH),
        'grassbg': self._generate_examples(GRASSBG_TF_RECORDS_PATH),
        'pbg': self._generate_examples(PBG_TF_RECORDS_PATH),
        'vbg': self._generate_examples(VBG_TF_RECORDS_PATH),
        'outd': self._generate_examples(OUTD_TF_RECORDS_PATH),
    }

  def _generate_examples(self, tfrecords_path, skip=None, take=None):
    """Yields examples."""

    def _compute_bboxes(example):
      """Return bounding box from segment belonging to each visible object."""
      bboxes = []
      segmentations = example['segmentations']
      for k in range(1, example['num_objects'][0] + 1):
        idxs = np.array(np.where(segmentations == k), dtype=np.float32)
        if idxs.size > 0:
          y_min = float(idxs[0].min() / segmentations.shape[0])
          x_min = float(idxs[1].min() / segmentations.shape[1])
          y_max = float((idxs[0].max() + 1) / segmentations.shape[0])
          x_max = float((idxs[1].max() + 1) / segmentations.shape[1])
        else:
          y_min, x_min, y_max, x_max = 0, 0, 0, 0
        bboxes.append(tfds.features.BBox(y_min, x_min, y_max, x_max))
      return bboxes

    file_path = tfrecords_path + '*'
    buffer_size = 2*(2**20)
    files = tf.data.Dataset.list_files(file_path)
    fc = functools.partial(
        tf.data.TFRecordDataset,
        compression_type=COMPRESSION_TYPE,
        buffer_size=buffer_size)
    raw_dataset = files.interleave(
        fc, cycle_length=64, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = raw_dataset.map(
        _decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Index starts at zero, hence ">=".
    if skip is not None:
      ds = ds.filter(lambda x: x['image_index'][0] >= skip)

    if take is not None:
      limit = take
      if skip is not None:
        limit += skip
      ds = ds.filter(lambda x: x['image_index'][0] < limit)

    ds_iter = ds.as_numpy_iterator()
    for i, example in enumerate(ds_iter):
      yield i, {
          'image': example['image'],
          'segmentations': example['segmentations'][Ellipsis, None],
          'depth': example['depth'][Ellipsis, None],
          'albedo': example['albedo'],
          'normal': example['normal'],
          'shadow': example['shadow'][Ellipsis, None],
          'instances': {
              'bboxes':
                  _compute_bboxes(example),
              'positions':
                  example['object_3d_coords'][:example['num_objects'][0]],
              'image_positions':
                  example['object_pixel_coords'][:example['num_objects'][0]],
              'rotation':
                  example['object_rotations'][:example['num_objects'][0]],
          },
          'num_instances': example['num_objects'][0],
          'image_index': example['image_index'][0]
      }
