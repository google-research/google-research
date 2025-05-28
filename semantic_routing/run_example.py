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

"""Run simple example of the benchmark API."""

from absl import app
from semantic_routing.benchmark import api
from semantic_routing.benchmark import config
from semantic_routing.benchmark import utils


def main(_):
  poi_specs = utils.get_poi_specs(config.POI_SPECS_PATH)

  print("Creating edgelist prediction dataset...")
  data_generator = api.EdgeListPredictionDatapointGenerator(
      seed=42,
      query_engine_type="labeled",
      graph_type="simplecity",
      task="routing",
      num_nodes=100,
      receptive_field_size=10,
      poi_receptive_field_size=5,
      poi_prob=0.2,
      use_fresh=True,
      test=False,
      prefix_len=3,
      poi_specs=poi_specs,
  )
  print("Edgelist prediction dataset created.")

  print("Example batch from Tensorflow dataset:")
  #  Alternatively, to get a TF dataset:
  dataset = data_generator.as_dataset(batch_size=2)
  for batch in dataset:
    print(batch)
    break


if __name__ == "__main__":
  app.run(main)
