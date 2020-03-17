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

"""Street View House Numbers (SVHN) Dataset, cropped version."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets.public_api as tfds

URL = "http://ufldl.stanford.edu/housenumbers/"

_CITATION = """\
@article{Netzer2011,
author = {Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Ng, Andrew Y},
booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
title = {Reading Digits in Natural Images with Unsupervised Feature Learning},
year = {2011}
}
"""


class SvhnCroppedSmall(tfds.core.GeneratorBasedBuilder):
  """Street View House Numbers (SVHN) Dataset, cropped version."""

  VERSION = tfds.core.Version(
      "1.0.0", experiments={tfds.core.Experiment.S3: False})
  SUPPORTED_VERSIONS = [
      tfds.core.Version(
          "3.0.0", "New split API (https://tensorflow.org/datasets/splits)"),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=(
            "The Street View House Numbers (SVHN) Dataset is an image digit "
            "recognition dataset of over 600,000 digit images coming from "
            "real world data. Images are cropped to 32x32."),
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=(32, 32, 3)),
            "label": tfds.features.ClassLabel(num_classes=10),
        }),
        supervised_keys=("image", "label"),
        urls=[URL],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):

    output_files = {
        "train": "svhn_data/train_small_32x32.mat",
        "val": "svhn_data/val_small_32x32.mat",
        "test": "svhn_data/test_small_32x32.mat",
    }

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,
            gen_kwargs=dict(filepath=output_files["train"],)),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=10,
            gen_kwargs=dict(filepath=output_files["val"],)),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=1,
            gen_kwargs=dict(filepath=output_files["test"],)),
    ]

  def _generate_examples(self, filepath):
    """Generate examples as dicts.

    Args:
      filepath: `str` path of the file to process.

    Yields:
      Generator yielding the next samples
    """
    with tf.io.gfile.GFile(filepath, "rb") as f:
      print(filepath)
      data = tfds.core.lazy_imports.scipy.io.loadmat(f)

    for i, (image,
            label) in enumerate(zip(np.rollaxis(data["X"], -1), data["y"])):
      label = label.reshape(())
      record = {
          "image": image,
          "label": label % 5,  # digit 0 is saved as 0 (instead of 10)
      }
      yield i, record
