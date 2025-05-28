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

"""CINIC-10 dataset."""

import os
import random
from typing import Any, Dict, List, Tuple, Iterator

import tensorflow_datasets.public_api as tfds


_CITATION = """
@article{darlow2018cinic,
  title={Cinic-10 is not imagenet or cifar-10},
  author={Darlow, Luke N and Crowley, Elliot J and Antoniou, Antreas and Storkey, Amos J},
  journal={arXiv preprint arXiv:1810.03505},
  year={2018}
}
"""

_DESCRIPTION = """
CINIC-10 is a drop-in replacement for CIFAR-10.
We compiled it as a benchmarking datset because CIFAR-10 can be too small/too easy and ImageNet is often too large/difficult.
ImageNet32 and ImageNet64 are smaller than ImageNet but even more difficult. CINIC-10 fills this benchmarking gap.
CINIC-10 has a total of 270,000 images equally split amonst three subsets: train, validate, and test.
In each subset (90,000 images) there are ten classes (identical to CIFAR-10 classes).
There are 9,000 images per class per subset. Using the suggested data split (an equal three-way split).
CINIC-10 has 1.8 times as many training samples than CIFAR-10.
CINIC-10 is designed to be directly swappable with CIFAR-10.
"""

_NUM_CLASSES = 10
_CLASS_MAP = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


class Cinic10(tfds.core.GeneratorBasedBuilder):
  """CINIC-10 dataset.

  Input (x):
    32 x 32 x 3 RGB digit image.
  Label (y):
    y is one of 10 classes.
  """

  VERSION = tfds.core.Version("0.1.0")
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
    place data in manual dir
  """

  def __init__(self, shuffle_seed = 100, **kwargs):
    super().__init__(**kwargs)
    self.shuffle_seed = shuffle_seed

  def _info(self):
    # Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(num_classes=_NUM_CLASSES),
            "file_name": tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("image", "label"),
        # Homepage of the dataset for documentation
        homepage="https://github.com/BayesWatch/cinic-10",
        citation=_CITATION,
    )

  def _split_generators(
      self,
      dl_manager
  ):
    """Returns SplitGenerators."""
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    extracted_path = dl_manager.manual_dir

    return [
        tfds.core.SplitGenerator(
            name="train",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "train",
            },
        ),
        tfds.core.SplitGenerator(
            name="valid",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "valid",
            },
        ),
        tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "test",
            },
        ),
    ]

  def _generate_examples(
      self,
      data_dir,
      split
  ):
    """Yields examples."""
    # Yields (key, example) tuples from the dataset
    all_inputs = []
    for k in range(_NUM_CLASSES):
      class_dir = os.path.join(data_dir, split, _CLASS_MAP[k])
      class_inputs = [
          (os.path.join(class_dir, filename), k)
          for filename in os.listdir(class_dir)
          if os.path.isfile(os.path.join(class_dir, filename))
      ]
      all_inputs.extend(class_inputs)
    random.seed(self.shuffle_seed)
    random.shuffle(all_inputs)
    for img_filename, label in all_inputs:
      features = {
          "file_name": img_filename,
          "image": img_filename,
          "label": label,
      }
      yield img_filename, features
