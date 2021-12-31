# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Library of Earth Engine utility functions."""

import enum
import math
import os
import random
from typing import List, Text, Dict

import ee

# Included to make the shuffling in split_days_into_train_eval_test
# deterministic.
random.seed(123)


class DataType(enum.Enum):
  ELEVATION_SRTM = 1
  VEGETATION_VIIRS = 2
  DROUGHT_GRIDMET = 3
  WEATHER_ERA5 = 4
  WEATHER_GRIDMET = 5
  FIRE_MODIS = 6
  POPULATION = 7


DATA_SOURCES = {
    DataType.ELEVATION_SRTM: 'USGS/SRTMGL1_003',
    DataType.VEGETATION_VIIRS: 'NOAA/VIIRS/001/VNP13A1',
    DataType.DROUGHT_GRIDMET: 'GRIDMET/DROUGHT',
    DataType.WEATHER_ERA5: 'ECMWF/ERA5/DAILY',
    DataType.WEATHER_GRIDMET: 'IDAHO_EPSCOR/GRIDMET',
    DataType.FIRE_MODIS: 'MODIS/006/MOD14A1',
    DataType.POPULATION: 'CIESIN/GPWv411/GPW_Population_Density'
}

DATA_BANDS = {
    DataType.ELEVATION_SRTM: ['elevation'],
    DataType.VEGETATION_VIIRS: ['NDVI'],
    DataType.DROUGHT_GRIDMET: ['pdsi'],
    DataType.WEATHER_ERA5: [
        'mean_2m_air_temperature',
        'total_precipitation',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m',
    ],
    DataType.WEATHER_GRIDMET: [
        'pr',
        'sph',
        'th',
        'tmmn',
        'tmmx',
        'vs',
        'erc',
    ],
    DataType.FIRE_MODIS: ['FireMask'],
    DataType.POPULATION: ['population_density']
}

# The time unit is 'days'.
DATA_TIME_SAMPLING = {
    DataType.VEGETATION_VIIRS: 8,
    DataType.DROUGHT_GRIDMET: 5,
    DataType.WEATHER_ERA5: 0.25,
    DataType.WEATHER_GRIDMET: 2,
    DataType.FIRE_MODIS: 1,
}

RESAMPLING_SCALE = {DataType.WEATHER_GRIDMET: 20000}

DETECTION_BAND = 'detection'
DEFAULT_KERNEL_SIZE = 128
DEFAULT_SAMPLING_RESOLUTION = 1000  # Units: meters
DEFAULT_EVAL_SPLIT = 0.2
DEFAULT_LIMIT_PER_EE_CALL = 60
DEFAULT_SEED = 123

COORDINATES = {
    # Used as input to ee.Geometry.Rectangle().
    'US': [-124, 24, -73, 49]
}


def get_image(data_type):
  """Gets an image corresponding to `data_type`.

  Args:
    data_type: A specifier for the type of data.

  Returns:
    The EE image correspoding to the selected `data_type`.
  """
  return ee.Image(DATA_SOURCES[data_type]).select(DATA_BANDS[data_type])


def get_image_collection(data_type):
  """Gets an image collection corresponding to `data_type`.

  Args:
    data_type: A specifier for the type of data.

  Returns:
    The EE image collection corresponding to `data_type`.
  """
  return ee.ImageCollection(DATA_SOURCES[data_type]).select(
      DATA_BANDS[data_type])


def remove_mask(image):
  """Removes the mask from an EE image.

  Args:
    image: The input EE image.

  Returns:
    The EE image without its mask.
  """
  mask = ee.Image(1)
  return image.updateMask(mask)


def export_feature_collection(
    feature_collection,
    description,
    bucket,
    folder,
    bands,
    file_format = 'TFRecord',
):
  """Starts an EE task to export `feature_collection` to TFRecords.

  Args:
    feature_collection: The EE feature collection to export.
    description: The filename prefix to use in the export.
    bucket: The name of the Google Cloud bucket.
    folder: The folder to export to.
    bands: The list of names of the features to export.
    file_format: The output file format. 'TFRecord' and 'GeoTIFF' are supported.

  Returns:
    The EE task associated with the export.
  """
  task = ee.batch.Export.table.toCloudStorage(
      collection=feature_collection,
      description=description,
      bucket=bucket,
      fileNamePrefix=os.path.join(folder, description),
      fileFormat=file_format,
      selectors=bands)
  task.start()
  return task


def convert_features_to_arrays(
    image_list,
    kernel_size = DEFAULT_KERNEL_SIZE,
):
  """Converts a list of EE images into `(kernel_size x kernel_size)` tiles.

  Args:
    image_list: The list of EE images.
    kernel_size: The size of the tiles (kernel_size x kernel_size).

  Returns:
    An EE image made of (kernel_size x kernel_size) tiles.
  """
  feature_stack = ee.Image.cat(image_list).float()
  kernel_list = ee.List.repeat(1, kernel_size)  # pytype: disable=attribute-error
  kernel_lists = ee.List.repeat(kernel_list, kernel_size)  # pytype: disable=attribute-error
  kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_lists)
  return feature_stack.neighborhoodToArray(kernel)


def get_detection_count(
    detection_image,
    geometry,
    sampling_scale = DEFAULT_SAMPLING_RESOLUTION,
    detection_band = DETECTION_BAND,
):
  """Counts the total number of positive pixels in the detection image.

  Assumes that the pixels in the `detection_band` of `detection_image` are
  zeros and ones.

  Args:
    detection_image: An EE image with a detection band.
    geometry: The EE geometry over which to count the pixels.
    sampling_scale: The sampling scale used to count pixels.
    detection_band: The name of the image band to use.

  Returns:
    The number of positive pixel counts or -1 if EE throws an error.
  """
  detection_stats = detection_image.reduceRegion(
      reducer=ee.Reducer.sum(), geometry=geometry, scale=sampling_scale)
  try:
    detection_count = int(detection_stats.get(detection_band).getInfo())
  except ee.EEException:
    # If the number of positive pixels cannot be counted because of a server-
    # side error, return -1.
    detection_count = -1
  return detection_count


def extract_samples(
    image,
    detection_count,
    geometry,
    sampling_ratio,
    detection_band = DETECTION_BAND,
    sampling_limit_per_call = DEFAULT_LIMIT_PER_EE_CALL,
    resolution = DEFAULT_SAMPLING_RESOLUTION,
    seed = DEFAULT_SEED,
):
  """Samples an EE image for positive and negative samples.

  Extracts `detection_count` positive examples and (`sampling_ratio` x
  `detection_count`) negative examples. Assumes that the pixels in the
  `detection_band` of `detection_image` are zeros and ones.

  Args:
    image: The EE image to extract samples from.
    detection_count: The number of positive samples to extract.
    geometry: The EE geometry over which to sample.
    sampling_ratio: If sampling negatives examples, samples (`sampling_ratio` x
      `detection_count`) negative examples. When extracting only positive
      examples, set this to zero.
    detection_band: The name of the image band to use to determine sampling
      locations.
    sampling_limit_per_call: The limit on the size of EE calls. Can be used to
      avoid memory errors on the EE server side. To disable this limit, set it
      to `detection_count`.
    resolution: The resolution in meters at which to scale.
    seed: The number used to seed the random number generator. Used when
      sampling less than the total number of pixels.

  Returns:
    An EE feature collection with all the extracted samples.
  """
  feature_collection = ee.FeatureCollection([])
  num_per_call = sampling_limit_per_call // (sampling_ratio + 1)

  # The sequence of sampling calls is deterministic, so calling stratifiedSample
  # multiple times never returns samples with the same center pixel.
  for _ in range(math.ceil(detection_count / num_per_call)):
    samples = image.stratifiedSample(
        region=geometry,
        numPoints=0,
        classBand=detection_band,
        scale=resolution,
        seed=seed,
        classValues=[0, 1],
        classPoints=[num_per_call * sampling_ratio, num_per_call],
        dropNulls=True)
    feature_collection = feature_collection.merge(samples)
  return feature_collection


def split_days_into_train_eval_test(
    start_date,
    end_date,
    split_ratio = DEFAULT_EVAL_SPLIT,
    window_length_days = 8,
):
  """Splits the days into train / eval / test sets.

  Splits the interval between  `start_date` and `end_date` into subintervals of
  duration `window_length` days, and divides them into train / eval / test sets.

  Args:
    start_date: The start date.
    end_date: The end date.
    split_ratio: The split ratio for the divide between sets, such that the
      number of eval time chunks and test time chunks are equal to the total
      number of time chunks x `split_ratio`. All the remaining time chunks are
      training time chunks.
    window_length_days: The length of the time chunks (in days).

  Returns:
    A dictionary containing the list of start day indices of each time chunk for
    each set.
  """
  num_days = int(
      ee.Date.difference(end_date, start_date, unit='days').getInfo())  # pytype: disable=attribute-error
  days = list(range(num_days))
  days = days[::window_length_days]
  random.shuffle(days)
  num_eval = int(len(days) * split_ratio)
  split_days = {}
  split_days['train'] = days[:-2 * num_eval]
  split_days['eval'] = days[-2 * num_eval:-num_eval]
  split_days['test'] = days[-num_eval:]
  return split_days
