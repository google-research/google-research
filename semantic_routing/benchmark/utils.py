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

"""Dataset utility functions."""

import csv
from typing import TypedDict

import numpy as np

from semantic_routing.benchmark import config
from semantic_routing.benchmark.query_engines import query_engine


class POIInfoType(TypedDict):
  poi_type_name: str
  poi_type_category_name: str
  poi_type_id: query_engine.POIType
  freq: int


POISpecType = tuple[
    list[POIInfoType],
    dict[str, list[POIInfoType]],
]

PREF_TYPES = ('', 'dislike highways', 'like highways', 'estimate')
BLACKLIST = ['bicycle_parking', 'bench', 'pitch']
SPECIALIZED_POIS = ['healthcare']
ROAD_VALUES = [
    # Major arteries
    'trunk',
    'motorway',
    'primary',
    # Smaller roads
    'secondary',
    'tertiary',
    # Links
    'trunk_link',
    'motorway_link',
    'primary_link',
    'secondary_link',
    'tertiary_link',
    # Tiny roads
    'unclassified',
    'residential',
    'service',
    'living_street',
    'passing_place',
    'traffic_island',
    'unknown',
]
SMALL_ROAD_VALUES = [
    'tertiary',
    'tertiary_link',
    'unclassified',
    'residential',
    'service',
    'living_street',
    'passing_place',
    'traffic_island',
    'unknown',
]


def pad_seq(tokens, max_len, pad_from='right', fill_value=-1):
  """Pad token or embedding sequence."""
  mask = np.zeros(max_len, dtype=np.int32)
  if pad_from == 'right':
    mask[: len(tokens)] = 1
    tokens = tokens[:max_len]
    tokens = np.pad(
        tokens,
        ((0, max_len - len(tokens)), *([(0, 0)] * (len(tokens.shape) - 1))),
        'constant',
        constant_values=(fill_value, fill_value),
    )
  elif pad_from == 'left':
    mask[-len(tokens) :] = 1
    tokens = tokens[-max_len:]
    tokens = np.pad(
        tokens,
        ((max_len - len(tokens), 0), *([(0, 0)] * (len(tokens.shape) - 1))),
        'constant',
        constant_values=(fill_value, fill_value),
    )
  else:
    raise ValueError('pad_from must be "right" or "left".')
  return tokens, mask


def get_poi_specs(
    file_path = config.POI_SPECS_PATH,
):
  """Get point-of-interest specs from POI category list."""
  with open(file_path, 'r') as f:
    reader = csv.DictReader(f)
    poi_csv = {field: [] for field in reader.fieldnames}
    for row in reader:
      for k in poi_csv:
        poi_csv[k].append(row[k])

  num_pois = len(next(iter(poi_csv.values())))
  general_poi_types = []
  specialized_poi_types = {p: [] for p in SPECIALIZED_POIS}
  for i in range(num_pois):
    r = {k: poi_csv[k][i] for k in poi_csv}
    if r['Tag Name'] in BLACKLIST:
      continue
    poi_type_info = {
        'poi_type_name': r['Tag Name'],
        'freq': int(r['Freq']),
        'poi_type_id': int(r['1']),
        'poi_type_category_name': r['OSM Tag'],
    }
    if r['OSM Tag'] in SPECIALIZED_POIS:
      specialized_poi_types[r['OSM Tag']].append(poi_type_info)
    else:
      general_poi_types.append(poi_type_info)
  return (general_poi_types, specialized_poi_types)


def get_modified_cost_fn(pref):
  """Get modified cost function."""

  def modified_cost(data):
    """Cost function."""
    if pref == 'estimate':
      return data['travel_time']
    if not pref:
      return data['current_travel_time']
    factor = 1
    if data['highway'] in ('motorway', 'trunk', 'primary'):
      if pref == 'dislike highways':
        factor *= 4
      elif pref == 'like highways':
        factor *= 0.5
      else:
        raise ValueError()
    return factor * data['current_travel_time']

  return modified_cost
