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

"""FMOW-WILDS dataset."""

import datetime
import os
from typing import Any, Dict, List, Tuple, Iterator

import numpy as np
import pandas as pd
import pytz
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_CITATION = """
@inproceedings{fmow2018,
  title={Functional Map of the World},
  author={Christie, Gordon and Fendley, Neil and Wilson, James and Mukherjee, Ryan},
  booktitle={CVPR},
  year={2018}
}
"""

_DESCRIPTION = """
The Functional Map of the World land use / building classification dataset.
This is a processed version of the Functional Map of the World dataset originally sourced from https://github.com/fMoW/dataset.
We consider a hybrid domain generalization and subpopulation shift problem,
where the input x is a RGB satellite image (resized to 224 x 224 pixels),
the label y is one of 62 building or land use categories,
and the domain d represents both the year the image was taken as well as its geographical region (Africa, the Americas, Oceania, Asia, or Europe).
"""

CATEGORIES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]


class Fmow(tfds.core.GeneratorBasedBuilder):
  """The Functional Map of the World land use / building classification dataset.

  Input (x):
    224 x 224 x 3 RGB satellite image.
  Label (y):
    y is one of 62 land use / building classes.
  Metadata:
    each image is annotated with a location coordinate, timestamp, country code.
    This dataset computes region as a derivative of country code.
  """

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
            # These are the features of your dataset like images, labels ...
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(num_classes=62),
            "meta_data": tfds.features.Tensor(shape=(3,), dtype=tf.int64),
            "file_name": tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("image", "label"),
        # Homepage of the dataset for documentation
        homepage="https://github.com/fMoW/dataset",
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
            name="val",
            gen_kwargs={
                "data_dir": extracted_path,
                "split": "val",
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
    split_dict = {"train": 0, "id_val": 1, "id_test": 2, "val": 3, "test": 4}
    split_scheme = "time_after_2016"
    category_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    metadata = pd.read_csv(os.path.join(data_dir, "rgb_metadata.csv"))
    country_codes_df = pd.read_csv(
        os.path.join(data_dir, "country_code_mapping.csv")
    )
    countrycode_to_region = {
        k: v
        for k, v in zip(country_codes_df["alpha-3"], country_codes_df["region"])
    }
    regions = [
        countrycode_to_region.get(code, "Other")
        for code in metadata["country_code"].to_list()
    ]
    metadata["region"] = regions
    if split_scheme.startswith("time_after"):
      year = int(split_scheme.split("_")[2])
      year_dt = datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)  # pylint: disable=g-tzinfo-datetime
      test_ood_mask = np.asarray(
          pd.to_datetime(metadata["timestamp"]) >= year_dt
      )
      # use 3 years of the training set as validation
      year_minus_3_dt = datetime.datetime(year - 3, 1, 1, tzinfo=pytz.UTC)  # pylint: disable=g-tzinfo-datetime
      val_ood_mask = (
          np.asarray(pd.to_datetime(metadata["timestamp"]) >= year_minus_3_dt)
          & ~test_ood_mask
      )
      ood_mask = test_ood_mask | val_ood_mask
    else:
      raise ValueError(f"Not supported: split_scheme = {split_scheme}")

    split_array = -1 * np.ones(len(metadata))
    for curr_split in split_dict:
      idxs = np.arange(len(metadata))
      if curr_split == "test":
        test_mask = np.asarray(metadata["split"] == "test")
        idxs = idxs[test_ood_mask & test_mask]
      elif curr_split == "val":
        val_mask = np.asarray(metadata["split"] == "val")
        idxs = idxs[val_ood_mask & val_mask]
      elif curr_split == "id_test":
        test_mask = np.asarray(metadata["split"] == "test")
        idxs = idxs[~ood_mask & test_mask]
      elif curr_split == "id_val":
        val_mask = np.asarray(metadata["split"] == "val")
        idxs = idxs[~ood_mask & val_mask]
      else:
        split_mask = np.asarray(metadata["split"] == curr_split)
        idxs = idxs[~ood_mask & split_mask]

      split_array[idxs] = split_dict[curr_split]

    # filter out sequestered images from full dataset
    seq_mask = np.asarray(metadata["split"] == "seq")
    # take out the sequestered images
    split_array = split_array[~seq_mask]
    y_array = np.asarray(
        [category_to_idx[y] for y in list(metadata["category"])]
    )
    metadata["y"] = y_array
    y_array = y_array[~seq_mask]

    # convert region to idxs
    all_regions = list(metadata["region"].unique())
    region_to_region_idx = {region: i for i, region in enumerate(all_regions)}
    metadata_map = {"region": all_regions}
    region_idxs = [
        region_to_region_idx[region] for region in metadata["region"].tolist()
    ]
    metadata["region"] = region_idxs

    # make a year column in metadata
    year_array = -1 * np.ones(len(metadata))
    ts = pd.to_datetime(metadata["timestamp"])
    for year in range(2002, 2018):
      year_mask = np.asarray(
          ts >= datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)  # pylint: disable=g-tzinfo-datetime
      ) & np.asarray(ts < datetime.datetime(year + 1, 1, 1, tzinfo=pytz.UTC))  # pylint: disable=g-tzinfo-datetime
      year_array[year_mask] = year - 2002
    metadata["year"] = year_array
    metadata_map["year"] = list(range(2002, 2018))

    metadata_fields = ["region", "year", "y"]
    metadata_array = metadata[metadata_fields].astype(int).to_numpy()[~seq_mask]

    indices = np.where(split_array == split_dict[split])[0]
    for idx in indices:
      img_filename = os.path.join(data_dir, "images", f"rgb_img_{idx}.png")
      features = {
          "file_name": img_filename,
          "image": img_filename,
          "label": int(y_array[idx]),
          "meta_data": metadata_array[idx],
      }
      yield img_filename, features
