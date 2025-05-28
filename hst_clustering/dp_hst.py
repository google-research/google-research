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

"""Beam-based pipeline for creating a DP HST."""
import hashlib
import random
from typing import Optional
from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np
from hst_clustering.gaussian_thresholding import get_gaussian_thresholding_params


class CreateVectorFromText(beam.DoFn):
  """Given the input text file, generate a np.array vector.

  The file is assumed to have one point per line. With each line being:
  point_id<Tab>dimension_1<Tab>...<Tab>dimension_d
  """

  def __init__(
      self,
      dimensions,
      min_value_entry,
      max_value_entry,
  ):
    self.dimensions = dimensions
    self.min_value_entry = min_value_entry
    self.max_value_entry = max_value_entry

  def process(self, element):
    # Removing the point id.
    vec = np.array([float(x) for x in element.split()[1:]])
    # Checking the entries are in the correct bounding box.
    max_entry = max(vec)
    min_entry = min(vec)
    if len(vec) != self.dimensions:
      raise ValueError(
          "invalid dimensions in vectors. Set --dimensions to "
          "the correct dimensionality of the input vectors. "
          f"Found: {len(vec)} Expected: {self.dimensions}"
      )
    if max_entry > self.max_value_entry or min_entry < self.min_value_entry:
      raise ValueError(
          "Each dimension should be between min_value_entry and"
          " max_value_entry."
      )
    return (vec,)


def get_diameter_center_one_count_tuple(
    min_dimension_value, max_dimension_value
):
  """Given the vector of the bounding box, output the diameter and center.

  The function takes in input the bounding box of an HST cell defined as two
  d-dimensional vectors of min_dimension_value, max_dimension_value with
  respectively the min and max value for each dimension in order.
  The output is a tuple with diameter and center of the cell and a count of 1.

  Args:
    min_dimension_value: a vector with the lower bound of the cell dimensions.
    max_dimension_value: a vector with the upper bound of the cell dimensions.

  Returns:
    A tuple with (diameter of the cell, center of the cell,
    1 -- representing one point).
  """
  assert min_dimension_value.shape == max_dimension_value.shape
  diameter = 0.0
  center = np.zeros_like(min_dimension_value)
  diameter = np.linalg.norm(max_dimension_value - min_dimension_value, ord=2)
  center = (max_dimension_value - min_dimension_value) / 2 + min_dimension_value
  return (diameter, center, 1.0)


def combine_diameter_center_count(stream):
  """Combiner function for diameter_center_count tuples.

  Args:
    stream: stream of tuples with (diameter, center, count of points).

  Returns:
    A combined tuple with the (diameter, center, total number of points).
  """
  diameter = None
  center = None
  total_count = 0.0
  for value in stream:
    assert len(value) == 3
    diameter, center, c = value
    total_count += c
  return (diameter, center, total_count)


# Given an HST layer, and a dimension, output a deterministic split threshold
# for the HST node in a consistent way.
def layer_dimension_threshold(
    seed, layer, dimension, min_dimension_value, max_dimension_value
):
  # The use of a weak random number generator is acceptable as this affects
  # only the approximation guarantees of the algorithm, not the privacy
  # guarantees.
  layer_dimension_string = f"{seed}|{layer}|{dimension}"
  random.seed(a=hashlib.md5(layer_dimension_string.encode()).digest())
  return (random.random() + 1.0) / 3.0 * (
      max_dimension_value - min_dimension_value
  ) + min_dimension_value


class MapToCellId(beam.DoFn):
  """Maps each input point into the corresponding cells to which it belongs."""

  def cell_id_to_str(self, cell_id):
    return "".join("0" if c else "1" for c in cell_id)

  def __init__(
      self, *, dimensions, min_value_entry, max_value_entry, layers, seed
  ):
    self.dimensions = dimensions
    self.min_value_entry = min_value_entry
    self.max_value_entry = max_value_entry
    self.layers = layers
    self.seed = seed

  def process(self, input_vector):
    # The cell id is a binary vector.
    cell_id = []
    # We start with the bounding box.
    max_dimension_value = np.full((self.dimensions,), self.max_value_entry)
    min_dimension_value = np.full((self.dimensions,), self.min_value_entry)
    # Output this point for the root.
    yield (
        self.cell_id_to_str(cell_id),
        get_diameter_center_one_count_tuple(
            min_dimension_value=min_dimension_value,
            max_dimension_value=max_dimension_value,
        ),
    )
    for layer in range(self.layers):
      for dimension in range(self.dimensions):
        threshold = layer_dimension_threshold(
            seed=self.seed,
            layer=layer,
            dimension=dimension,
            min_dimension_value=min_dimension_value[dimension],
            max_dimension_value=max_dimension_value[dimension],
        )
        if input_vector[dimension] < threshold:
          cell_id.append(False)
          max_dimension_value[dimension] = threshold
        else:
          cell_id.append(True)
          min_dimension_value[dimension] = threshold
        yield (
            self.cell_id_to_str(cell_id),
            get_diameter_center_one_count_tuple(
                min_dimension_value=min_dimension_value,
                max_dimension_value=max_dimension_value,
            ),
        )


class GaussianThresholdMechansimFn(beam.DoFn):
  """Class to add noise and threshold with the Gaussian mechanism."""

  def __init__(self, epsilon, delta, l0_sensitivity):
    self.epsilon = epsilon
    self.delta = delta
    sigma, threshold = get_gaussian_thresholding_params(
        max_bucket_contribution=1,
        max_num_buckets_contributed=l0_sensitivity,
        epsilon=epsilon,
        delta=delta,
    )
    self.sigma = sigma
    self.threshold = threshold

    logging.info(
        "Gaussian thresholding at: %s delta: %s epsilon %s",
        self.threshold, self.delta, self.epsilon
    )

  def process(self, element):
    assert len(element) == 2
    cell_id, t = element
    assert len(t) == 3
    copy_t = (t[0], t[1].copy(), t[2] +
              np.random.normal(loc=0.0, scale=self.sigma))
    if copy_t[2] >= self.threshold:
      yield (cell_id, copy_t)


def output_as_string(element):
  # the output format is: cell_id <comma> left_child <comma> right_child <comma>
  # <comma> diameter <comma> dp_weight,coordinates
  key, value = element
  ret = (
      ",".join([key, key + "0", key + "1", str(value[0]), str(value[2])])
      + ","
      + ",".join([str(x) for x in value[1]])
  )
  return ret


def run_hst_pipeline(
    *,
    input_points,
    output_hst,
    dimensions,
    min_value_entry,
    max_value_entry,
    layers,
    epsilon,
    delta,
    seed,
    num_buckets,
    runner = None,
    pipeline_options = None,
):
  """Running the pipeline."""

  with beam.Pipeline(
      runner=runner, options=pipeline_options
  ) as root:
    # Execute the pipeline.
    logging.info("Beam runner started")
    total_l0_sensitivity = layers * dimensions

    done = (
        root
        | "ReadPoints" >> beam.io.ReadFromText(input_points)
        | "ConvertToVector"
        >> beam.ParDo(
            CreateVectorFromText(
                dimensions=dimensions,
                min_value_entry=min_value_entry,
                max_value_entry=max_value_entry,
            )
        )
        | "Reshuffle" >> beam.Reshuffle(num_buckets=num_buckets)
        | "MapToCellId"
        >> beam.ParDo(
            MapToCellId(
                dimensions=dimensions,
                min_value_entry=min_value_entry,
                max_value_entry=max_value_entry,
                layers=layers,
                seed=seed,
            )
        )
        | "CombinerCounts"
        >> beam.CombinePerKey(fn=combine_diameter_center_count)
        | "GaussianThresholdMechansimFn"
        >> beam.ParDo(
            GaussianThresholdMechansimFn(
                epsilon=epsilon,
                delta=delta,
                l0_sensitivity=total_l0_sensitivity,
            )
        )
        | "OutputAsString" >> beam.Map(fn=output_as_string)
        | "WritePoints" >> beam.io.WriteToText(output_hst)
    )
    logging.info("Pipeline created")
  return done
