# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

r"""Simple utility for plotting the given languages on world's map.

Example:
--------
  > python3 plot_languages_main.py \
      --training_data_dir /tmp \
      --output_map_file /tmp/world.pdf

Extra Dependencies:
-------------------
To install BaseMap from sources:
  > apt-get install libgeos-dev
  > pip3 install geos
  > pip3 install https://github.com/matplotlib/basemap/archive/master.zip
  > pip3 install seaborn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import app
from absl import flags
from absl import logging

import basic_models
import constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

flags.DEFINE_string(
    "training_data_dir", "",
    "Directory where the training data resides. This data has to be in the "
    "format generated from the original SIGTYP data by the "
    "\"sigtyp_reader_main\" tool.")

flags.DEFINE_string(
    "output_map_file", "",
    "Output PDF containing the world map with languages shown.")

FLAGS = flags.FLAGS

_TOOL_NAME = "plot"


def _process_set(filename, world_map, color, size):
  data = basic_models.load_training_data(
      _TOOL_NAME, FLAGS.training_data_dir, filename)

  # Plot the data.
  mxy = world_map(data["longitude"].tolist(), data["latitude"].tolist())
  world_map.scatter(mxy[0], mxy[1], s=size, c=color, lw=0, alpha=1, zorder=5)


def main(unused_argv):
  if not FLAGS.training_data_dir:
    raise ValueError("Specify --training_data_dir!")
  if not FLAGS.output_map_file:
    raise ValueError("Specify --output_map_file!")

  # ------------------------------------------------
  # Set up the map itself using Mercator projection:
  # ------------------------------------------------
  my_dpi = 96
  plt.figure(figsize=(2600/my_dpi, 1800/my_dpi), dpi=my_dpi)
  world_map = Basemap(projection="merc",
                      llcrnrlat=-60,
                      urcrnrlat=65,
                      llcrnrlon=-180,
                      urcrnrlon=180,
                      lat_ts=0,
                      resolution="c")
  # Dark grey land, black lakes.
  world_map.fillcontinents(color="#191919", lake_color="#17202A")
  world_map.drawmapboundary(fill_color="#17202A")  # Dark background.
  # Thin white line for country borders.
  world_map.drawcountries(linewidth=0.05, color="#5F6A6A")

  # ------------------------------------
  # Load the languages, display and save.
  # ------------------------------------
  set_names = [
      (const.TRAIN_FILENAME, "#1292db", 10),      # Blue.
      (const.DEV_FILENAME, "#84DE02", 20),        # Green.
      (const.TEST_BLIND_FILENAME, "#F4D03F", 20)  # Yellow.
  ]
  for set_name, color, point_size in set_names:
    _process_set(set_name, world_map, color, point_size)

  logging.info("Saving plot to \"%s\" ...", FLAGS.output_map_file)
  plt.savefig(FLAGS.output_map_file)


if __name__ == "__main__":
  app.run(main)
