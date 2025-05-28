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

"""Script to combine ISMN data with the data released along with the paper."""

import datetime
import itertools
import multiprocessing.pool
import os
from typing import Any, Tuple

from absl import app
from absl import flags
from google.cloud import storage
from ismn.interface import ISMN_Interface
import tensorflow as tf

_OUTPUT_DIR_FLAG = flags.DEFINE_string(
    "output_dir", "",
    "Output data directory where merged data is written to. You would need ~1TB of storage to download, process and save the entire data."
)
_ISMN_DATA_PATH_FLAG = flags.DEFINE_string(
    "ismn_data_path", "", "Path to where the ISMN data zip file is stored.")
_TEMP_STORAGE_DIR_FLAG = flags.DEFINE_string(
    "temp_storage_dir", "/tmp", "Directory used for temporary storage.")
_NUM_WORKERS_FLAG = flags.DEFINE_integer(
    "num_workers", 1,
    "Number of workers to use in the multiprocessing pool to process data in parallel."
)

_TOTAL_FILES = 500
_EOSCIENCE_PUBLIC_BUCKET = "eoscience-public"


def get_ismn_sm_5cm(sensor_id: str, timestamp: int,
                    ismn_data: ISMN_Interface) -> float:
  """Fetches the in-situ soil moisture reading at 5cm from the ISMN data.

  Args:
    sensor_id: A unique string determining the sensor at which we want to
      extract soil moisture readings at. Must be of the format
      <SENSOR_NETWORK>__<STATION_NAME>__<SENSOR_TYPE>.
    timestamp: A timestamp in millis determining the time at which the in-situ
      soil moisture will be fetched at.
    ismn_data: An instance of the ISMN_Interface which is used to fetch ISMN
      data.

  Returns:
    A float corresponding to the soil moisture reading at 5cm.

  Raises:
    LookupError: When either of the sensor network, in-situ station or soil
      moisture timeseries/datapoint at timestamp cannot be found.
  """
  sensor_network, sensor_name, _ = sensor_id.split("__")

  if sensor_network in ismn_data.networks:
    sensor_network_data = ismn_data[sensor_network]
  else:
    raise LookupError(
        f"Unable to find sensor network for {sensor_id}, skipping.")

  station_data = None
  for station_name in sensor_network_data.stations:
    matches = True
    for sensor_name_part in sensor_name.split("_"):
      if (sensor_name_part
          not in station_name) and (sensor_name_part
                                    not in station_name.replace("-", "")):
        matches = False
        break
    if matches:
      station_data = sensor_network_data[station_name]
      break
  if not station_data:
    raise LookupError(
        f"Unable to find in-situ station for sensor {sensor_id}, skipping.")

  sensor_ts = None
  for key in station_data.sensors:
    if "0.05" in key and "soil_moisture" in key:
      sensor_ts = station_data[key].read_data()
      break
  if sensor_ts is None:
    raise LookupError(
        f"Unable to find soil moisture timeseries at 5cm depth for sensor {sensor_id}, skipping."
    )

  # ISMN uses the UTC timezone.
  timestamp = int(timestamp)
  date_time = datetime.datetime.utcfromtimestamp(
      timestamp / 1000).replace(microsecond=timestamp % 1000 * 1000)
  # We use a small buffer of +/- 3 mins to match datapoints temporally.
  date_time_low = date_time + datetime.timedelta(minutes=-3)
  date_time_high = date_time + datetime.timedelta(minutes=3)
  date_time_low = datetime.datetime.strftime(date_time_low, "%Y-%m-%d %H:%M:%S")
  date_time_high = datetime.datetime.strftime(date_time_high,
                                              "%Y-%m-%d %H:%M:%S")

  sensor_ts = sensor_ts[date_time_low:date_time_high]

  if sensor_ts.empty:
    raise LookupError(
        f"Unable to find SM reading {timestamp} between timestamps {date_time_low} and {date_time_high} for sensor {sensor_id}, skipping."
    )

  sm_readings = sensor_ts[date_time_low:date_time_high]["soil_moisture"].values
  if len(sm_readings) > 1:
    raise ValueError(
        f"Multiple soil moisture readings found between timestamps {date_time_low} and {date_time_high} for sensor {sensor_id}. There should have only been one, skipping."
    )
  sm_reading = sm_readings[0]

  return sm_reading


def download_file_from_gcs(file_path: str, save_path: str,
                           gcs_client: storage.Client):
  """Downloads the dataset provided with the paper from Google Cloud Storage.

  Args:
    file_path: Path to the file on GCS (without the bucket) to be downloaded.
    save_path: Path where the dataset should be downloaded to. Can point to a
      /tmp path in order for it to be cleaned up automatically.
    gcs_client: GCS client.
  """
  save_dir = save_path.rsplit("/", 1)[0]
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  bucket = gcs_client.bucket(_EOSCIENCE_PUBLIC_BUCKET)
  blob = bucket.blob(file_path)
  blob.download_to_filename(save_path)


def add_sm_labels_to_data(input_file: str, output_dir: str,
                          temp_storage_dir: str) -> Tuple[int, int, int]:
  """Add soil moisture labels to data released with the paper.

  Args:
    input_file: Path to the input TFRecord on GCS (without the bucket). Data
      present in this file will be downloaded, processed and saved.
    output_dir: Directory where processed data is saved to. The output filenames
      remain the same as the input file names.
    temp_storage_dir: Directory where temporary data is stored during
      processing.

  Returns:
    Tuple containing counters for (num_success, num_failures)
  """
  print(f"Processing: {input_file}", flush=True)

  # Setup counters.
  num_lookup_failures = 0
  num_value_failures = 0
  num_success = 0

  temp_download_path = f"{temp_storage_dir}/{input_file}"
  download_file_from_gcs(input_file, temp_download_path,
                         add_sm_labels_to_data.gcs_client)
  dataset = tf.data.TFRecordDataset(temp_download_path, compression_type="GZIP")
  output_file = os.path.join(
      output_dir,
      input_file.split("/")[-1].split(".")[0] + ".tfrecord.gz")
  with tf.io.TFRecordWriter(output_file, "GZIP") as writer:
    for buf in dataset.as_numpy_iterator():
      try:
        tf_example = tf.train.Example.FromString(buf)
        new_feature = {}
        sensor_id = None
        timestamp = None
        sm_present = False
        for key, feature in tf_example.features.feature.items():
          new_feature[key] = feature
          if "sensor_id" == key:
            sensor_id = feature.bytes_list.value[0].decode("utf-8")
          if "timestamp" == key:
            timestamp = feature.float_list.value[0]
          if "sm_0_5" == key or "sm_5_5" == key:
            sm_present = True
        if not sm_present:
          sm_value = get_ismn_sm_5cm(sensor_id, timestamp,
                                     add_sm_labels_to_data.ismn_data)
          new_feature["sm_0_5"] = tf.train.Feature(
              float_list=tf.train.FloatList(value=[sm_value]))
        new_example = tf.train.Example(
            features=tf.train.Features(feature=new_feature))
        writer.write(new_example.SerializeToString())
        num_success += 1
      except LookupError as e:
        print(e, flush=True)
        num_lookup_failures += 1
      except ValueError as e:
        print(e, flush=True)
        num_value_failures += 1

    writer.close()
  # Clean up temporary storage.
  os.remove(temp_download_path)
  return (num_success, num_lookup_failures, num_value_failures)


def init_worker(function: Any):
  """Sets up a worker with the GCS client and the ISMN Data."""
  function.gcs_client = storage.Client.create_anonymous_client()
  function.ismn_data = ISMN_Interface(_ISMN_DATA_PATH_FLAG.value, parallel=True)


def main(_) -> None:
  # Make a call to the ISMN_Interface to create metadata required once.
  ISMN_Interface(_ISMN_DATA_PATH_FLAG.value, parallel=True)

  # Soil moisture data released with the paper is stored under
  # soil_moisture_retrieval_data/* under the eoscience-public GCS bucket.
  input_files = [
      f"soil_moisture_retrieval_data/data-{idx:05d}-of-{_TOTAL_FILES:05d}.tfrecord.gz"
      for idx in range(_TOTAL_FILES)
  ]

  with multiprocessing.pool.Pool(
      _NUM_WORKERS_FLAG.value,
      initializer=init_worker,
      initargs=(add_sm_labels_to_data,)) as pool:
    stats_list = pool.starmap(
        add_sm_labels_to_data,
        zip(input_files, itertools.repeat(_OUTPUT_DIR_FLAG.value),
            itertools.repeat(_TEMP_STORAGE_DIR_FLAG.value)))
    pool.close()
    pool.join()

    print("Finished execution!")
    total_stats = list(map(sum, zip(*stats_list)))
    print(
        f"Total success: {total_stats[0]}, Total lookup failures: {total_stats[1]}, Total value failures: {total_stats[2]}"
    )


if __name__ == "__main__":
  app.run(main)
