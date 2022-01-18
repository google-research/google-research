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

r"""TFDS databuilder for fmow_v1.0."""

import datetime as dt
import os

import numpy as onp
import pandas as pd
import pytz
import tensorflow as tf
import tensorflow_datasets as tfds


def filter_date(date, start, end):
  return date >= start and date < end


class Fmow(tfds.core.BeamBasedBuilder):
  """TFDS builder for fmow_v1.

  The Functional Map of the World land use / building classification dataset.
    This is a processed version of the Functional Map of the World dataset
    originally sourced from https://github.com/fMoW/dataset.

  This dataset is part of the WILDS benchmark.
  """

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You must manually download and extract fmow_v1.0 data from
  (https://worksheets.codalab.org/rest/bundles/0xc59ea8261dfe4d2baa3820866e33d781/contents/blob/)
  and place them in `manual_dir`.
  """

  VERSION = tfds.core.Version('1.0.0')
  _SPLITS = ['train', 'val_id', 'val_ood', 'test_id', 'test_ood']
  _CLASSES = [
      'airport', 'airport_hangar', 'airport_terminal', 'amusement_park',
      'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint',
      'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam',
      'debris_or_rubble', 'educational_institution', 'electric_substation',
      'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain',
      'gas_station', 'golf_course', 'ground_transportation_station', 'helipad',
      'hospital', 'impoverished_settlement', 'interchange', 'lake_or_pond',
      'lighthouse', 'military_facility', 'multi-unit_residential',
      'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park',
      'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port',
      'prison', 'race_track', 'railway_bridge', 'recreational_facility',
      'road_bridge', 'runway', 'shipyard', 'shopping_mall',
      'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility',
      'stadium', 'storage_tank', 'surface_mine', 'swimming_pool', 'toll_booth',
      'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility',
      'wind_farm', 'zoo'
  ]

  # pylint: disable=g-tzinfo-datetime
  # pylint: disable=g-long-lambda
  _DOMAIN_FILTERS = {
      'train':
          lambda date: filter(date, dt.datetime(2002, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2013, 1, 1, tzinfo=pytz.UTC)),
      'val_id':
          lambda date: filter(date, dt.datetime(2002, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2013, 1, 1, tzinfo=pytz.UTC)),
      'val_ood':
          lambda date: filter(date, dt.datetime(2013, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2016, 1, 1, tzinfo=pytz.UTC)),
      'test_id':
          lambda date: filter(date, dt.datetime(2002, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2013, 1, 1, tzinfo=pytz.UTC)),
      'test_ood':
          lambda date: filter(date, dt.datetime(2016, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2018, 1, 1, tzinfo=pytz.UTC)),
  }

  # pylint: enable=g-long-lambda
  # pylint: enable=g-tzinfo-datetime

  def _info(self):

    return tfds.core.DatasetInfo(
        builder=self,
        description=('fmow:'),
        features=tfds.features.FeaturesDict({
            'image':
                tfds.features.Image(
                    shape=(None, None, 3), encoding_format='jpeg'),
            'label':
                tfds.features.ClassLabel(names=self._CLASSES)
        }),
        supervised_keys=('image', 'label'),
        homepage='https://github.com/fMoW/dataset',
        citation=r"""@inproceedings{fmow2018,
                    title={Functional Map of the World},
                    author={Christie, Gordon and Fendley, Neil and Wilson, James and Mukherjee, Ryan},
                    booktitle={CVPR},
                    year={2018}
                  }""",
    )

  def _split_generators(self, dl_manager):
    """Download data and define the splits."""
    image_dirs = os.path.join(dl_manager.manual_dir)
    meta_dirs = os.path.join(dl_manager.manual_dir, 'rgb_metadata.csv')

    splits = []
    for split in self._SPLITS:
      gen_kwargs = {
          'data_dir': image_dirs,
          'meta_dir': meta_dirs,
          'split': split,
      }
      splits.append(
          tfds.core.SplitGenerator(name=f'{split}', gen_kwargs=gen_kwargs))

    return splits

  def _build_pcollection(self, pipeline, data_dir, meta_dir, split):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    with tf.io.gfile.GFile(meta_dir) as meta_file:
      meta_lines = meta_file.readlines()
      header = meta_lines[0].split(',')
      examples_descriptions = meta_lines[1:]

    total_examples = len(examples_descriptions)
    examples_descriptions = enumerate(examples_descriptions)
    split_index = header.index('split')
    date_index = header.index('timestamp')

    def _process_example(example_description):
      (idx, features) = example_description
      (unused_split, unused_img_filename, unused_img_path,
       unused_spatial_reference, unused_epsg, category, unused_visible,
       unused_img_width, unused_img_height, unused_country_code,
       unused_cloud_cover, unused_timestamp, unused_lat,
       unused_lon) = features.split(',')
      chunk_size = total_examples // 100
      batch_indx = int(idx) // chunk_size
      img_indx = int(idx) % chunk_size
      image = onp.load(
          os.path.join(data_dir, f'rgb_all_imgs_{batch_indx}.npy'),
          mmap_mode='r')[img_indx]
      return idx, {'image': image, 'label': category}

    def _filter_example(example_description):
      time_condition = self._DOMAIN_FILTERS[split](
          pd.to_datetime(example_description[1].split(',')[date_index]))
      split_condition = (
          example_description[1].split(',')[split_index] == split.split('_')[0])
      return time_condition and split_condition

    return pipeline | beam.Create(
        (examples_descriptions
        )) | beam.Filter(_filter_example) | beam.Map(_process_example)
