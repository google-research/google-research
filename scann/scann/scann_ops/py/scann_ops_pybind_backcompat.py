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

"""Back-compatibility shim with old-style scann_ops_pybind serialization."""

import os

from scann.scann_ops import scann_assets_pb2


def path_exists(path):
  """Wrapper around Google/OSS check for if file/directory exists."""
  return os.path.exists(path)


def populate_and_save_assets_proto(
    artifacts_dir):
  """Populate and write a ScannAssets proto listing assets in `artifacts_dir`.

  Args:
    artifacts_dir: the directory for which this function finds ScaNN assets. The
      resulting proto is written in plaintext format to
      `artifacts_dir/scann_assets.pbtxt`.
  Returns:
    The ScannAssets proto object.
  """
  assets = scann_assets_pb2.ScannAssets()

  def add_if_exists(filename, asset_type):
    file_path = os.path.join(artifacts_dir, filename)
    if path_exists(file_path):
      assets.assets.append(
          scann_assets_pb2.ScannAsset(
              asset_path=file_path, asset_type=asset_type))

  add_if_exists("ah_codebook.pb", scann_assets_pb2.ScannAsset.AH_CENTERS)
  add_if_exists("serialized_partitioner.pb",
                scann_assets_pb2.ScannAsset.PARTITIONER)
  add_if_exists("datapoint_to_token.npy",
                scann_assets_pb2.ScannAsset.TOKENIZATION_NPY)
  add_if_exists("hashed_dataset.npy",
                scann_assets_pb2.ScannAsset.AH_DATASET_NPY)
  add_if_exists("int8_dataset.npy",
                scann_assets_pb2.ScannAsset.INT8_DATASET_NPY)
  add_if_exists("int8_multipliers.npy",
                scann_assets_pb2.ScannAsset.INT8_MULTIPLIERS_NPY)
  add_if_exists("dp_norms.npy", scann_assets_pb2.ScannAsset.INT8_NORMS_NPY)
  add_if_exists("dataset.npy", scann_assets_pb2.ScannAsset.DATASET_NPY)

  with open(os.path.join(artifacts_dir, "scann_assets.pbtxt"), "w") as f:
    f.write(str(assets))
  return assets
