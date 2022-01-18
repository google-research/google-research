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

"""Convert Spherical MNIST dataset to tensorflow_datasets (tfds) format.

This module converts the dataset from the format used in
"Spin-Weighted Spherical CNNs", NeurIPS'20 to tensorflow_datasets
(tfds).

To build the dataset, run the following from directory containing this file:
$ tfds build
"""

from typing import Any, Dict, Iterable, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


_DESCRIPTION = """\
Spherical MNIST consists of MNIST digits projected on the sphere.

In "canonical" mode, projections are centered at the south pole.

In "rotated" mode, the spherical image is randomly rotated after
projection. Instead of sampling one rotation per image, each sampled rotation
is applied to 500 images (chunk_size=500 in the original code).
"""

_CITATION = """\
@inproceedings{DBLP:conf/iclr/CohenGKW18,
  author    = {Taco S. Cohen and
               Mario Geiger and
               Jonas K{\"{o}}hler and
               Max Welling},
  title     = {Spherical CNNs},
  booktitle = {6th International Conference on Learning Representations,
               {ICLR} 2018, Vancouver, BC, Canada, April 30 - May 3, 2018,
               Conference Track Proceedings},
  publisher = {OpenReview.net},
  year      = {2018},
  url       = {https://openreview.net/forum?id=Hkbd5xZRb},
  timestamp = {Thu, 21 Jan 2021 17:36:45 +0100},
  biburl    = {https://dblp.org/rec/conf/iclr/CohenGKW18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_HOMEPAGE = 'https://github.com/jonas-koehler/s2cnn'

# This is the dataset in the format used in the "Spin-Weighted
# Spherical CNNs", linked in
# https://github.com/daniilidis-group/swscnn.
_DOWNLOAD_URL = 'https://drive.google.com/uc?id=1h7JwdjWalXZFoXCU8Ez1rLscWih8PcZ7'

_IMAGE_DIMENSIONS = (64, 64, 1)
_VALIDATION_SET_SIZE = 10_000


class SphericalMnist(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for spherical_mnist dataset. See superclass for details."""

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
            'image': tfds.features.Image(shape=_IMAGE_DIMENSIONS),
            'label': tfds.features.ClassLabel(num_classes=10),
        }),
        # These are returned if `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(
      self, dl_manager):
    """Returns SplitGenerators. See superclass method for details."""
    dataset_directory = dl_manager.download_and_extract(_DOWNLOAD_URL)
    dataset_files = {
        'train_rotated': dataset_directory / 'rr/train0.tfrecord',
        'validation_rotated': dataset_directory / 'rr/train0.tfrecord',
        'test_rotated': dataset_directory / 'rr/test0.tfrecord',
        'train_canonical': dataset_directory / 'nrnr/train0.tfrecord',
        'validation_canonical': dataset_directory / 'nrnr/train0.tfrecord',
        'test_canonical': dataset_directory / 'nrnr/test0.tfrecord'}

    return {split: self._generate_examples(filename, split)
            for split, filename in dataset_files.items()}

  def _generate_examples(self,
                         path,
                         split):
    """Dataset generator. See superclass method for details."""
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP')

    for image_id, datapoint in enumerate(dataset):
      # The validation set is obtained from train, but we must make sure the ids
      # are different since one might want to combine both during training.
      if split.startswith('train') and image_id < _VALIDATION_SET_SIZE:
        continue
      if split.startswith('validation') and image_id >= _VALIDATION_SET_SIZE:
        break

      parsed = tf.train.Example.FromString(datapoint.numpy())
      image = np.frombuffer(parsed.features.feature['x'].bytes_list.value[0],
                            dtype=np.float32).reshape(*_IMAGE_DIMENSIONS)
      label = parsed.features.feature['y'].int64_list.value[0]

      yield image_id, {
          'image': image.astype('uint8'),
          'label': label,
      }
