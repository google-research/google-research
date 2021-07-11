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

"""Utilities for dealing with model output including Tensorboard."""
import json
import logging
import os
import random
import time

from dataclasses import dataclass
from google import api_core
from google.cloud import storage

from covid_epidemiology.src import constants


@dataclass
class TensorboardConfig:
  """Class for holding Tensorboard configuration information."""
  mode: str
  gcs_directory: str = 'unknown_experiment/unknown_run'
  gcs_bucket_name: str = constants.TENSORBOARD_BUCKET
  local_directory: str = constants.TENSORBOARD_LOCAL_DIR
  log_iterations: int = constants.TENSORBOARD_LOG_ITERATIONS

  # The profiler only avilable in TensorFlow >= 2.2
  use_profiler: bool = True
  profiler_start: int = 10
  profiler_end: int = 15

  def bucket(self, project=None):
    client = storage.Client(project)
    return client.bucket(self.gcs_bucket_name)

  @property
  def log_location(self):
    return self.local_directory if self.mode == 'LOCAL' else self.gcs_url

  @property
  def enabled(self):
    return self.mode != constants.TENSORBOARD_OFF

  @property
  def gcs_url(self):
    return os.path.join(
        'gs://',
        self.gcs_bucket_name,
        self.gcs_directory,
    )


def upload_local_directory_to_gcs(local_directory, bucket,
                                  gcs_path):
  """Uploads all files in a directory to GCS recursively.

  Args:
    local_directory: The base directory.
    bucket: The GCS bucket object to upload the data
    gcs_path: The base "directory" to upload the files to.
  """
  for entry in os.scandir(local_directory,):
    local_path = entry.path
    local_subdir = local_path[1 + len(local_directory):]
    if entry.is_file():
      remote_path = os.path.join(gcs_path, local_subdir)
      blob = bucket.blob(remote_path)
      blob.upload_from_filename(local_path)
    elif entry.is_dir():
      upload_local_directory_to_gcs(local_path, bucket,
                                    os.path.join(gcs_path, local_subdir))


def write_tensorboard_metadata(tensorboard_config,
                               output_filename):
  """Writes the metadata for tensorboard if it is enabled."""

  if tensorboard_config.enabled:
    if tensorboard_config.mode == 'LOCAL':
      upload_local_directory_to_gcs(
          tensorboard_config.local_directory,
          tensorboard_config.bucket(),
          tensorboard_config.gcs_directory,
      )

    # This only works with a GCS path. It does not work with a local path.
    metadata = {
        'outputs': [{
            'type': 'tensorboard',
            'source': tensorboard_config.gcs_url,
        }]
    }
    with open(output_filename, 'w') as f:
      json.dump(metadata, f)


def upload_string_to_gcs(
    output_data,
    output_bucket,
    output_filename,
    gcs_client = None,
    num_retries = 5,
    base_retry_delay = 4.0,
    max_retry_delay = 120,
):
  """Uploads a string to GCS with retries.

  Retries using exponential backoff with full jitter when a handled exception is
  thrown.

  Args:
    output_data: The string to be uploaded.
    output_bucket: The bucket where the string will be uploaded.
    output_filename: The file's name.
    gcs_client: The storage client for the project.
    num_retries: The maximum number of retries
    base_retry_delay: The maximum initial time to wait before retrying.
    max_retry_delay: The maximum total time to wait before retrying.
  """

  if gcs_client is None:
    gcs_client = storage.Client(project=constants.PROJECT_ID_MODEL_TRAINING)

  for retry in range(num_retries + 1):
    try:
      output_bucket = gcs_client.get_bucket(output_bucket)
      output_blob = output_bucket.blob(output_filename)
      output_blob.upload_from_string(output_data)
      break
    except api_core.exceptions.Forbidden as e:
      if retry >= num_retries:
        raise
      logging.warning('Retrying GCS upload: %s', e)
      time.sleep(
          random.uniform(0, min(max_retry_delay, base_retry_delay * 2**retry)))
