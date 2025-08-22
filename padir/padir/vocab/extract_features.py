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

"""Script to extract sentences from a TFDS WMT dataset."""

from collections.abc import Sequence

from absl import app
from absl import flags
import seqio


_DATASET = flags.DEFINE_string(
    "dataset",
    default="wmt14_translate/en-de:1.0.0",
    help="TFDS name of WMT dataset.",
)

_FEATURES = flags.DEFINE_multi_string(
    "features",
    default=["German", "English"],
    help="Name of features to extract.",
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "Path to output txt file containing one feature value per line.",
)


def open_file(filepattern, mode):
  return open(filepattern, mode)  # pylint: disable=unreachable


def extract_sentences(dataset, features, output_file):
  train_split = seqio.TfdsDataSource(dataset).get_dataset("train")
  with open_file(output_file, "w") as fp:
    for datum in train_split:
      for feature in features:
        feature_value = datum[feature].numpy().decode("utf8")
        fp.write(f"{feature_value}\n")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  assert _OUTPUT_FILE.value is not None, "--output_file must be set."
  extract_sentences(
      _DATASET.value,
      _FEATURES.value,
      _OUTPUT_FILE.value,
  )


if __name__ == "__main__":
  app.run(main)
