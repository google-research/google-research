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

import unittest
from unittest import mock

from google import api_core
from google.cloud import storage
from covid_epidemiology.src.models.shared import output_utils


class TestOutputUtils(unittest.TestCase):

  def test_uploads_with_no_errors(self):
    mock_client = mock.Mock(spec=storage.Client)
    mock_bucket = mock.Mock(spec=storage.Bucket)
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.Mock(spec=storage.Blob)
    mock_bucket.blob.return_value = mock_blob

    output_utils.upload_string_to_gcs("test_data", "bucket", "filename",
                                      mock_client)

    mock_client.get_bucket.assert_called_once_with("bucket")
    mock_bucket.blob.assert_called_once_with("filename")
    mock_blob.upload_from_string.assert_called_once_with("test_data")

  def test_retries_the_specified_number_of_times_and_then_raises(self):
    mock_client = mock.Mock(spec=storage.Client)
    mock_bucket = mock.Mock(spec=storage.Bucket)
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.Mock(spec=storage.Blob)
    mock_bucket.blob.return_value = mock_blob

    mock_blob.upload_from_string.side_effect = api_core.exceptions.Forbidden(
        "test")

    num_retries = 4
    num_calls = num_retries + 1
    with mock.patch("time.sleep") as sleep_mock:
      with self.assertRaises(api_core.exceptions.Forbidden):
        output_utils.upload_string_to_gcs(
            "test_data",
            "bucket",
            "filename",
            mock_client,
            num_retries=num_retries)
      self.assertEqual(sleep_mock.call_count, num_retries)

    self.assertEqual(mock_client.get_bucket.call_count, num_calls)
    self.assertEqual(mock_bucket.blob.call_count, num_calls)
    self.assertEqual(mock_blob.upload_from_string.call_count, num_calls)


if __name__ == "__main__":
  unittest.main()
