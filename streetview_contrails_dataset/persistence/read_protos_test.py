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

"""Demonstration of how to read the dataset."""

import os

from absl.testing import absltest
import tensorflow as tf

from streetview_contrails_dataset.persistence import streetview_dataset_pb2


class ReadProtosTest(absltest.TestCase):

  def test_write_and_read_proto(self):
    formation_info = streetview_dataset_pb2.FormationInfo(
        flight_id=1234,
        nearby_flight_waypoints=streetview_dataset_pb2.Waypoints(points=[
            streetview_dataset_pb2.Location(
                timestamp=1582676608,
                latitude=37.2612,
                longitude=-121.5602,
                altitude_meters=10000,
                era5_data=streetview_dataset_pb2.EcmwfData(
                    u=100., v=100., w=-0.5))
        ]))
    persistence_labels = [
        streetview_dataset_pb2.PersistenceLabel(
            timestamp=1582676608,
            detected=streetview_dataset_pb2.DETECTED_SINGLETON,
            satellite_contrail=streetview_dataset_pb2.LinearContrail(
                lat1=37.2612,
                lng1=-121.5602,
                lat2=37.3417,
                lng2=-121.5273,
                timestamp=1582676608)),
        streetview_dataset_pb2.PersistenceLabel(
            timestamp=1582679608,
            detected=streetview_dataset_pb2.DETECTED_MULTIPLE_CANDIDATES,
            satellite_contrail=streetview_dataset_pb2.LinearContrail(
                lat1=37.2612,
                lng1=-121.5602,
                lat2=37.3417,
                lng2=-121.5273,
                timestamp=1582679608)),
    ]
    dataset_entry = streetview_dataset_pb2.DatasetEntry(
        formation_info=formation_info, persistence_labels=persistence_labels)

    temp_dir = self.create_tempdir()
    file_path = os.path.join(temp_dir, 'dataset.tfrecord')
    with tf.io.TFRecordWriter(file_path) as writer:
      writer.write(dataset_entry.SerializeToString())

    tfrecord_dataset = tf.data.TFRecordDataset([file_path])
    for buf in tfrecord_dataset.take(1):
      recovered_dataset_entry = streetview_dataset_pb2.DatasetEntry()
      recovered_dataset_entry.ParseFromString(buf.numpy())

    self.assertEqual(dataset_entry, recovered_dataset_entry)


if __name__ == '__main__':
  absltest.main()
