# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for data utilities."""

import json
import os
from unittest import mock

from absl.testing import absltest
import datasets
from etils import epath
import ml_collections
from PIL import Image

from Uboreshaji_Modeli.common import data


class LoadCocoJsonTest(absltest.TestCase):

  def test_loads_correct_json(self):
    temp_dir = epath.Path(self.create_tempdir().full_path)
    json_path = temp_dir / "test.json"
    dummy_data = {"images": [{"id": 1, "file_name": "img1.jpg"}]}
    with json_path.open("w") as f:
      json.dump(dummy_data, f)

    result = data._load_coco_json(json_path)
    self.assertEqual(result, dummy_data)


class CocoToHfDictTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = epath.Path(self.create_tempdir().full_path)
    self.image_dir = self.temp_dir / "images"
    self.image_dir.mkdir()
    self.annotation_file = self.temp_dir / "annotations.json"

    # Create a dummy image
    self.image_path = self.image_dir / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(self.image_path)

  def test_converts_normal_coco(self):
    coco_data = {
        "categories": [{"id": 1, "name": "cat"}],
        "images": [{"id": 1, "file_name": "test_image.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}
        ],
    }
    with self.annotation_file.open("w") as f:
      json.dump(coco_data, f)

    hf_data, features = data._coco_to_hf_dict(
        self.image_dir, self.annotation_file
    )

    self.assertEqual(hf_data["image_id"], [1])
    self.assertLen(hf_data["image"], 1)
    self.assertEqual(
        hf_data["objects"], [{"category": [1], "bbox": [[10, 10, 20, 20]]}]
    )
    self.assertIn("image", features)
    self.assertIn("objects", features)

  def test_handles_missing_categories(self):
    coco_data = {"images": [], "annotations": []}
    with self.annotation_file.open("w") as f:
      json.dump(coco_data, f)
    hf_data, _ = data._coco_to_hf_dict(self.image_dir, self.annotation_file)
    self.assertEqual(hf_data, {"image_id": [], "image": [], "objects": []})

  def test_handles_missing_image_file(self):
    coco_data = {
        "categories": [{"id": 1, "name": "cat"}],
        "images": [{"id": 1, "file_name": "missing.jpg"}],
        "annotations": [],
    }
    with self.annotation_file.open("w") as f:
      json.dump(coco_data, f)
    hf_data, _ = data._coco_to_hf_dict(self.image_dir, self.annotation_file)
    self.assertEqual(hf_data["image_id"], [])

  def test_handles_bbox_dict(self):
    coco_data = {
        "categories": [{"id": 1, "name": "cat"}],
        "images": [{"id": 1, "file_name": "test_image.jpg"}],
        "annotations": [{
            "image_id": 1,
            "category_id": 1,
            "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
        }],
    }
    with self.annotation_file.open("w") as f:
      json.dump(coco_data, f)

    hf_data, _ = data._coco_to_hf_dict(
        self.image_dir, self.annotation_file
    )
    self.assertEqual(hf_data["objects"][0]["bbox"], [[10.0, 10.0, 20.0, 20.0]])


class ConvertCocoFolderToHfTest(absltest.TestCase):

  def test_end_to_end_conversion(self):
    root_dir = epath.Path(self.create_tempdir().full_path)
    image_dir = root_dir / "images"
    image_dir.mkdir()
    output_path = root_dir / "hf_dataset"

    # Create dummy image
    img_path = image_dir / "train_img.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    # Create dummy train.json
    train_json = {
        "categories": [{"id": 1, "name": "cat"}],
        "images": [{"id": 1, "file_name": "train_img.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}
        ],
    }
    with (root_dir / "train.json").open("w") as f:
      json.dump(train_json, f)

    data.convert_coco_folder_to_hf(root_dir, output_path)

    self.assertTrue(output_path.exists())
    dataset_dict = datasets.load_from_disk(output_path)
    self.assertIn("train", dataset_dict)
    self.assertLen(dataset_dict["train"], 1)


class GetDatasetTest(absltest.TestCase):

  def test_returns_loaded_dataset(self):
    temp_dir = self.create_tempdir().full_path
    fake_dataset_dict = {"col1": [1, 2], "col2": ["a", "b"]}
    fake_dataset = datasets.Dataset.from_dict(fake_dataset_dict)
    fake_dataset.save_to_disk(temp_dir)
    cfg = ml_collections.ConfigDict()
    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.dataset_path = temp_dir
    result = data.get_dataset(cfg)
    self.assertEqual(list(result), list(fake_dataset))

  def test_returns_loaded_dataset_with_trailing_slash(self):
    temp_dir = self.create_tempdir().full_path
    fake_dataset_dict = {"col1": [1, 2], "col2": ["a", "b"]}
    fake_dataset = datasets.Dataset.from_dict(fake_dataset_dict)
    fake_dataset.save_to_disk(temp_dir)
    cfg = ml_collections.ConfigDict()
    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.dataset_path = temp_dir + "/"
    result = data.get_dataset(cfg)
    self.assertEqual(list(result), list(fake_dataset))

  def test_propagates_error(self):
    temp_dir = self.create_tempdir().full_path
    cfg = ml_collections.ConfigDict()
    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.dataset_path = os.path.join(temp_dir, "nonexistent")
    with self.assertRaises(FileNotFoundError):
      data.get_dataset(cfg)

  def test_parquet_fallback_streaming(self):
    temp_dir = self.create_tempdir().full_path
    cfg = ml_collections.ConfigDict()
    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.dataset_path = os.path.join(temp_dir, "parquet_dir")

    with mock.patch.object(
        datasets, "load_dataset", autospec=True
    ) as mock_load_dataset:
      mock_load_dataset.return_value = "mocked_dataset"
      result = data.get_dataset(cfg)

      mock_load_dataset.assert_called_once()
      _, kwargs = mock_load_dataset.call_args
      self.assertTrue(kwargs.get("streaming"))
      self.assertEqual(result, "mocked_dataset")


if __name__ == "__main__":
  absltest.main()
