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

"""Amazon Review dataset."""

import csv
import os
from typing import Any, Dict, List, Tuple, Iterator

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from transformers import AutoTokenizer
from transformers import TFRobertaModel
from wilds.common.utils import map_to_id_array


_CITATION = """
@inproceedings{ni2019justifying,
  author = {J. Ni and J. Li and J. McAuley},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  pages = {188--197},
  title = {Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
  year = {2019},
}
"""

_DESCRIPTION = """
Amazon Review dataset.

This is a modified version of the 2018 Amazon Reviews dataset. Extract embeddings for classification using RoBerta.
We consider a hybrid domain generalization and subpopulation problem where the domains correspond to different reviewers.
The task is multi-class sentiment classification, where the input x is the text of a review,
the label y is a corresponding star rating from 1 to 5, and the domain d is the identifier of the reviewer who wrote the review.
Our goal is to perform consistently well across a wide range of reviewers,
i.e., to achieve high tail performance on different subpopulations of reviewers in addition to high average performance.
In addition, we consider disjoint set of reviewers between training and test time.
"""

BATCH_SIZE = 16


class Amazon_review(tfds.core.GeneratorBasedBuilder):  # pylint: disable=invalid-name
  """The Amazon Review dataset.

  Input (x):
      Review text of maximum token length of 512

  Label (y):
      y is the star rating (0,1,2,3,4 corresponding to 1-5 stars)

  Metadata:
      reviewer: reviewer ID
      year: year in which the review was written
      category: product category
      product: product ID
  """

  _NOT_IN_DATASET: int = -1
  VERSION = tfds.core.Version("0.1.0")
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
    place data in manual dir
  """

  def _info(self):
    # Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like texts, labels ...
            "text": tfds.features.Text(),
            "label": tfds.features.ClassLabel(num_classes=5),
            "meta_data": tfds.features.Tensor(shape=(5,), dtype=tf.int64),
            "embedding": tfds.features.Tensor(shape=(768,), dtype=tf.float32),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("text", "label"),
        # Homepage of the dataset for documentation
        homepage="https://nijianmo.github.io/amazon/index.html",
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
            name="id_val",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "id_val",
            },
        ),
        tfds.core.SplitGenerator(
            name="id_test",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "id_test",
            },
        ),
        tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "test",
            },
        ),
        tfds.core.SplitGenerator(
            name="val",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "val",
            },
        ),
    ]

  def load_metadata(
      self,
      data_df,
      split_array
  ):
    # Get metadata
    columns = ["reviewerID", "asin", "category", "reviewYear", "overall"]
    metadata_fields = ["user", "product", "category", "year", "y"]
    metadata_df = data_df[columns].copy()
    metadata_df.columns = metadata_fields

    sort_idx = np.argsort(split_array)
    ordered_maps = {}
    for field in ["user", "product", "category"]:
      # map to IDs in the order of split values
      ordered_maps[field] = pd.unique(metadata_df.iloc[sort_idx][field])
    ordered_maps["y"] = range(1, 6)
    ordered_maps["year"] = range(
        metadata_df["year"].min(), metadata_df["year"].max() + 1
    )
    metadata_map, metadata = map_to_id_array(metadata_df, ordered_maps)
    metadata = metadata.astype(np.int64)
    return metadata_fields, metadata, metadata_map

  def _generate_examples(
      self,
      data_dir,
      split
  ):
    """Yields examples."""
    # Yields (key, example) tuples from the dataset

    # The official split is to split by users
    split_scheme = "user"

    # Load data
    data_df = pd.read_csv(
        os.path.join(data_dir, "reviews.csv"),
        dtype={
            "reviewerID": str,
            "asin": str,
            "reviewTime": str,
            "unixReviewTime": int,
            "reviewText": str,
            "summary": str,
            "verified": bool,
            "category": str,
            "reviewYear": int,
        },
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONNUMERIC,
    )
    split_df = pd.read_csv(
        os.path.join(data_dir, "splits", f"{split_scheme}.csv")
    )
    is_in_dataset = split_df["split"] != Amazon_review._NOT_IN_DATASET
    split_df = split_df[is_in_dataset]
    data_df = data_df[is_in_dataset]
    # Get arrays
    split_array = split_df["split"].values
    input_array = np.array(list(data_df["reviewText"]))
    # Get metadata
    (
        metadata_fields,
        metadata_array,
        _,
    ) = self.load_metadata(data_df, split_array)
    # Get y from metadata
    y_array = metadata_array[:, metadata_fields.index("y")]
    # Set split info
    if split_scheme in ("user", "time") or split_scheme.endswith(
        "_generalization"
    ):
      split_dict = {
          "train": 0,
          "val": 1,
          "id_val": 2,
          "test": 3,
          "id_test": 4,
      }
    elif split_scheme == "category_subpopulation" or split_scheme.endswith(
        "_baseline"
    ):
      # Use defaults
      pass
    else:
      raise ValueError(f"Split scheme {split_scheme} is not recognized.")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = TFRobertaModel.from_pretrained("roberta-base")
    indices = np.where(split_array == split_dict[split])[0]
    n = indices.shape[0]
    batch_size = BATCH_SIZE
    num_batch = (n + batch_size - 1) // batch_size
    embeddings = []
    for k in range(num_batch):
      batch_index = indices[k*batch_size:(k+1)*batch_size]
      batch_text = list(input_array[batch_index])
      batch_encoded_inputs = tokenizer(
          batch_text,
          max_length=512,
          truncation=True,
          padding="max_length",
          return_tensors="tf",
      )
      batch_outputs = model(batch_encoded_inputs, training=False)
      batch_embeddings = batch_outputs.last_hidden_state[:, 0, :]
      embeddings.extend(batch_embeddings.numpy())
    embeddings = np.array(embeddings)
    for i, idx in enumerate(indices):
      features = {
          "text": input_array[idx],
          "label": int(y_array[idx]),
          "meta_data": metadata_array[idx],
          "embedding": embeddings[i],
      }
      yield str(idx), features
