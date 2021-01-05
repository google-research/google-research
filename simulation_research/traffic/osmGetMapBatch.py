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

r"""Download a large map batch by batch.

Example usage:
Check the batches without downloading:
blaze-bin/research/simulation/traffic/osmGetMapBatch \
    -b 37.3445,-122.1863,37.4554,-121.9963 -s 0.05

Download and save files:
blaze-bin/research/simulation/traffic/osmGetMapBatch \
    -b 37.3445,-122.1863,37.4554,-121.9963 -s 0.05 --store_files

Download ignoring existing files:
blaze-bin/research/simulation/traffic/osmGetMapBatch \
    -b 37.3445,-122.1863,37.4554,-121.9963 -s 0.05 --store_files -f
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import six
from six.moves import range
import six.moves.http_client
import six.moves.urllib.parse as urlparse

from simulation_research.traffic import file_util

FLAGS = flags.FLAGS

flags.DEFINE_string("prefix", "osm",
                    "prefix of the output file",
                    short_name="p")
flags.DEFINE_string("output_dir", None,
                    "output folder",
                    short_name="o")
flags.DEFINE_list("bbox", "37.3445,-122.1863,37.4554,-121.9963",
                  "geo coordinates: south,west,north,east",
                  short_name="b")
flags.DEFINE_float("batch_size", 0.05,
                   "download batchsize",
                   short_name="s")
flags.DEFINE_float("overlap_ratio", 0.1,
                   "overlap ratio between adjacent batches",
                   short_name="l")
flags.DEFINE_bool("store_files", False,
                  "download and save files",
                  short_name="d")
flags.DEFINE_bool("force_download", False,
                  "force downloading ignoring existing files",
                  short_name="f")
flags.DEFINE_integer("reconnect_times", 5,
                     "how many times for reconnecting server")

# List of hosts (https://wiki.openstreetmap.org/wiki/Overpass_API#Introduction):
# 1) "www.overpass-api.de"
# 2) "z.overpass-api.de"
# 3) "lz4.overpass-api.de"
# 4) "overpass.osm.rambler.ru"

host = "lz4.overpass-api.de"
port = 443
conn = None


def SetUpConnection():
  """Sets up a connection."""
  global conn
  if conn is None:
    print("setting up connection")
    if os.environ.get("https_proxy") is not None:
      proxy_url = os.environ.get("https_proxy")
      url = urlparse.urlparse(proxy_url)
      conn = six.moves.http_client.HTTPSConnection(url.hostname, url.port)
      conn.set_tunnel(host, port, {})
    else:
      conn = six.moves.http_client.HTTPConnection(host)
  return conn


def ReadCompressed(query, file_path):
  """Reads compressed file."""
  global conn
  query_token = """
  <osm-script timeout="300" element-limit="1073741824">
  <union>
     %s
     <recurse type="node-relation" into="rels"/>
     <recurse type="node-way"/>
     <recurse type="way-relation"/>
  </union>
  <union>
     <item/>
     <recurse type="way-node"/>
  </union>
  <print mode="body"/>
  </osm-script>""" % query

  for try_count in range(FLAGS.reconnect_times):
    try:
      conn = SetUpConnection()
      print("Sending request...")
      conn.request("POST", "/api/interpreter", query_token)
      response = conn.getresponse()
      print(response.status, response.reason)
      if response.status == 200:
        break
      if try_count >= FLAGS.reconnect_times-1:
        print("Quitting after failing to reconnect %s times" %
              FLAGS.reconnect_times)
        return -1
    except six.moves.http_client.CannotSendRequest:
      print("Retrying connecting to server...")
      conn.close()
      conn = None

  out = open(file_path, "wb")
  out.write(six.ensure_binary(response.read()))
  out.close()


def Get(_):
  """Gets the map tile by tile from south to north, then west to east."""
  coordinate_list = [float(coordinate) for coordinate in FLAGS.bbox]
  south, west, north, east = coordinate_list
  logging.info("Boundaries: south %s, west %s, north %s, east %s.",
               south, west, north, east)
  if south > north or west > east:
    raise flags.Error("""Invalid geocoordinates in bbox.
        Make sure the coordinates are in order: south, west, north, east.""")

  north_south_span = north - south
  east_west_span = east - west
  num_batches_north_south = int(round(north_south_span / FLAGS.batch_size))
  num_batches_east_west = int(round(east_west_span / FLAGS.batch_size))
  num_batches_north_south = (1 if num_batches_north_south == 0
                             else num_batches_north_south)
  num_batches_east_west = (1 if num_batches_east_west == 0
                           else num_batches_east_west)
  batch_step_size_north_south = north_south_span / num_batches_north_south
  batch_step_size_east_west = east_west_span / num_batches_east_west
  batch_margin_size_north_south = (batch_step_size_north_south *
                                   FLAGS.overlap_ratio)
  batch_margin_size_east_west = (batch_step_size_east_west *
                                 FLAGS.overlap_ratio)

  logging.info("Map span [degree]: NS(%s), EW(%s).",
               north_south_span, east_west_span)
  logging.info("Number of steps: NS(%s) X EW(%s).",
               num_batches_north_south, num_batches_east_west)
  logging.info("Map batch size: NS(%s) X EW(%s).",
               batch_step_size_north_south, batch_step_size_east_west)
  logging.info("Overlap margin size [degree]: NS(%s) X EW(%s).",
               batch_margin_size_north_south, batch_margin_size_east_west)

  bottom = south
  left = west
  for north_south_bathc_index in range(num_batches_north_south):
    for east_west_batch_index in range(num_batches_east_west):
      top = bottom + batch_step_size_north_south
      right = left + batch_step_size_east_west

      query_info = '<bbox-query n="%s" s="%s" w="%s" e="%s"/>' % (
          top + batch_margin_size_north_south,
          bottom - batch_margin_size_north_south,
          left - batch_margin_size_east_west,
          right + batch_margin_size_east_west)
      filename = "%s%s-%s_%sx%s.osm.xml" % (
          FLAGS.prefix,
          north_south_bathc_index,
          east_west_batch_index,
          num_batches_north_south,
          num_batches_east_west)
      logging.info(query_info)
      if FLAGS.output_dir is None:
        output_dir = os.getcwd()
      else:
        output_dir = FLAGS.output_dir
      filepath = os.path.join(output_dir, filename)
      try:
        if file_util.f_exists(filepath):
          logging.warning("File already downloaded.")
          file_exists = True
        else:
          file_exists = False
      except file_util.FileIOError:
        logging.warning("Error from file operation.")
        file_exists = False

      if (FLAGS.store_files and
          (not file_exists or FLAGS.force_download) and
          ReadCompressed(query_info, filepath) == -1):
        return
      left += batch_step_size_east_west

    bottom += batch_step_size_north_south
    left = west


if __name__ == "__main__":
  app.run(Get)
