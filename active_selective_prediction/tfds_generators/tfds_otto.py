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

"""Otto Group Product Classification dataset."""

import os

from typing import Any, Dict, List, Tuple, Iterator
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_CITATION = """
@misc{titericz2020otto,
  title={Otto Group Product Classification Challenge: Classify products into the correct category},
  author={TITERICZ, G and SEMENOV, S},
  year={2020},
  publisher={URL: https://www.kaggle.com/competitions/otto-group-product-classification-challenge}
}
"""

_DESCRIPTION = """
Otto Group Product Classification dataset.

The Otto Group is one of the world's biggest e-commerce companies, with subsidiaries in more than 20 countries, including Crate & Barrel (USA), Otto.de (Germany) and 3 Suisses (France).
We are selling millions of products worldwide every day, with several thousand products being added to our product line.

A consistent analysis of the performance of our products is crucial. However, due to our diverse global infrastructure, many identical products get classified differently.
Therefore, the quality of our product analysis depends heavily on the ability to accurately cluster similar products.
The better the classification, the more insights we can generate about our product range.
"""


class Otto(tfds.core.GeneratorBasedBuilder):
  """The Otto Group Product Classification dataset.

  Input (x):
      93 numerical features

  Label (y):
      9 categories for all products.
      Each target category represents one of our most important product
      categories (like fashion, electronics, etc.).
  """

  VERSION = tfds.core.Version("0.1.0")
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
    place data in manual dir
  """

  def __init__(
      self,
      shuffle_seed = 100,
      train_frac = 0.7,
      test_frac = 0.2,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.shuffle_seed = shuffle_seed
    self.train_frac = train_frac
    self.test_frac = test_frac

  def _info(self):
    # Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features like input_features, labels.
            "input_feature": tfds.features.Tensor(shape=(93,), dtype=tf.int64),
            "label": tfds.features.ClassLabel(num_classes=9),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("input_feature", "label"),
        # Homepage of the dataset for documentation
        homepage="https://www.kaggle.com/competitions/otto-group-product-classification-challengel",
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
    df = pd.read_csv(os.path.join(extracted_path, "train.csv"), index_col="id")
    input_features = np.array(df.iloc[:, :-1])
    labels = np.array([int(y[-1:]) - 1 for y in df.iloc[:, -1]])
    num_samples = input_features.shape[0]
    clf = LocalOutlierFactor(
        contamination=self.test_frac,
    )
    scores = clf.fit_predict(input_features)
    ood_cond = scores == -1
    non_ood_cond = scores == 1
    non_test_indices = np.where(non_ood_cond)[0]
    np.random.seed(self.shuffle_seed)
    np.random.shuffle(non_test_indices)
    split_index = int(self.train_frac * num_samples)
    train_indices = non_test_indices[:split_index]
    val_indices = non_test_indices[split_index:]
    test_indices = np.where(ood_cond)[0]
    train_features = input_features[train_indices]
    train_labels = labels[train_indices]
    val_features = input_features[val_indices]
    val_labels = labels[val_indices]
    test_features = input_features[test_indices]
    test_labels = labels[test_indices]
    data_dict = {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }
    return [
        tfds.core.SplitGenerator(
            name="train",
            gen_kwargs={
                "data_dict": data_dict,
                "split": "train",
            },
        ),
        tfds.core.SplitGenerator(
            name="val",
            gen_kwargs={
                "data_dict": data_dict,
                "split": "val",
            },
        ),
        tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "data_dict": data_dict,
                "split": "test",
            },
        ),
    ]

  def _generate_examples(
      self, data_dict, split
  ):
    """Yields examples."""
    # Yields (key, example) tuples from the dataset
    input_features = data_dict[f"{split}_features"]
    labels = data_dict[f"{split}_labels"]
    num_samples = labels.shape[0]
    for idx in range(num_samples):
      features = {
          "input_feature": input_features[idx],
          "label": labels[idx],
      }
      yield f"{split}_{idx}", features
