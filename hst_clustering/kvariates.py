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

"""Implements a parallel version of the kvariates mechanism."""

import os
import re

from absl import logging
import apache_beam as beam
import numpy as np
import pandas as pd

from hst_clustering import kmedian_plus_plus


beam.coders.registry.register_coder(np.ndarray, beam.coders.PickleCoder)


def parse_row(line):
  r"""Parse row of a data file.

  Args:
    line: A line in the file. We assume features are \s+ separated and the first
      element in the line is an index.

  Returns:
    A numpy array.
  """
  split_line = re.split(r"\s+", line)

  output = [float(x) for i, x in enumerate(split_line) if i != 0]
  return np.array(output)


def merge_arrays(arrays):
  in_memory_arrays = [a for a in arrays]
  return np.vstack(in_memory_arrays)


def select_random_center(arrays):
  in_memory_arrays = list(arrays[1])
  n = len(in_memory_arrays)
  ix = int(np.random.choice(range(n), 1)[0])
  return (arrays[0], in_memory_arrays[ix])


def hash_to_int(hash_vector):
  binary_expression = [x * 2**i for i, x in enumerate(hash_vector)]
  return int(np.sum(binary_expression))


def _simhash_split(array, proj_matrix):
  hash_v = np.sign(np.dot(proj_matrix, array))
  hash_v = (hash_v + 1 / 2).astype(int)
  return hash_to_int(np.reshape(hash_v, -1))


def simhash_split(array, num_keys):
  num_proj = int(np.log2(num_keys))
  d = len(array)
  np.random.seed(12312023)
  proj_mat = np.random.normal(0, 1, (num_proj, d))
  return (_simhash_split(array, proj_mat), array)


def objective(data, mu, agg=np.sum):
  """K medians objective.

  Args:
     data: Dataset. Each row is a datapoint.
     mu: Center to evaluate.
     agg: Aggregation function to combine pointwise objectives.

  Returns:
    agg(||x_i - mu||)
  """
  n, _ = data.shape
  mu_rep = np.repeat(mu.reshape(1, -1), n, 0)
  return _objective_repeated(data, mu_rep, agg)


def _objective_repeated(data, mu_rep, agg):
  return agg(np.linalg.norm(data - mu_rep, axis=1))


def calculate_collection_distance_to_centers(keyed_array, centers):
  key = keyed_array[0]
  arrays = keyed_array[1]
  n_cent = centers.shape[0]
  distances = np.zeros((arrays.shape[0], n_cent))
  for j in range(n_cent):
    c = centers[j, :]
    c_rep = np.repeat(c.reshape(1, -1), arrays.shape[0], 0)
    distances[:, j] = np.linalg.norm(arrays - c_rep, axis=1)
  return (key, np.sum(np.min(distances, 0)))


def add_random_key(data, num_keys):
  ix = int(np.random.choice(range(num_keys), 1)[0])
  return (ix, data)


def calculate_normalizer(keyed_distances):
  normalizer = 0
  for _, distance in keyed_distances[1]:
    normalizer += distance
  return normalizer


def calculate_probabilities(keyed_distance, normalizer):
  return (keyed_distance[0], keyed_distance[1] / normalizer)


def select_center(keyed_centers, probabilities):
  """Select a center from keyed_centers based on probabilities.

  Args:
    keyed_centers: Tuple with a dummy first entry. Second entry is a dictionary
      type structure consisting of a key (for the cluster id) and a center as a
      numpy array.
    probabilities: Optional if passed, a dictionary mapping a key for center to
      the probability of sampling it.

  Returns:
    A single center.
  """

  center_dic = {key: center for key, center in keyed_centers[1]}
  probability_array = []
  indices = []
  n = len(center_dic)
  logging.info("True number of workers %d", n)
  if probabilities is None:
    for k in center_dic.keys():
      indices.append(k)
      probability_array.append(1. / n)
      print(probability_array)
  else:
    for i, p in probabilities:
      probability_array.append(p)
      indices.append(i)
  ix = int(np.random.choice(indices, size=1, p=probability_array)[0])
  return center_dic[ix]


def center_data(array, mean_v):
  return array - mean_v


def get_distance_to_centers(array, centers):
  return objective(centers, array, np.min)


def parse_array_from_line(line):
  line = line.strip("[").strip("]")
  return np.fromstring(line, sep=" ", dtype=float)


class DataRecoveryInfo:
  """Contains information to recover centers in the original data dimension.

  projected_file is the file containing data in the projected space
  raw_data is the file containing data in the original space.
  """

  def __init__(self):
    self.projected_file = ""
    self.raw_data = ""
    self.epsilon = 0.0
    self.delta = 0.0
    self.norm = 0.0


def parse_centers_from_files(center_dir, data_recovery_info=None):
  """Returns centers from a collection of files in center_dir.

  Directory center_dir can only contain files representing the coordinates of
  a center. One center per file and the center is encoded as "[x, x, x,...,]".

  Args:
    center_dir: Directory containing files for the centers. One file per center.
    data_recovery_info: If passed, it contains information needed to recover
      centers in the original dimension. Will recover centers in the original
      dimension by doing one step of kmedians on the raw data.

  Returns:
    Centers as a numpy array.
  """

  centers = []
  for file in os.listdir(center_dir):
    if not os.path.isfile(center_dir + "/" + file):
      logging.warning("File %s does not exist skipping", file)
      print("File %s does not exist skipping" % file)
      continue
    with open(center_dir + "/" + file, "r") as f:
      centers.append(parse_array_from_line(" ".join(f.readlines())))
  centers = np.vstack(centers)
  if data_recovery_info is not None:
    with open(data_recovery_info.projected_file, "r") as f:
      data = pd.read_csv(f, sep=r"\s+", header=None, index_col=0)
    with open(data_recovery_info.raw_data, "rb") as f:
      raw_data = np.load(f)
    assignment = np.argmin(kmedian_plus_plus.find_distances(data, centers), 1)
    centers = kmedian_plus_plus.private_center_recovery(
        raw_data,
        assignment,
        data_recovery_info.norm,
        data_recovery_info.epsilon,
        data_recovery_info.delta,
    )
  return centers


def iteration(
    i,
    keyed_data,
    keyed_data_matrix,
    center_selector,
    previous_centers=None,
):
  """One iteration of kvariates.

  Args:
    i: Number of the iteration.
    keyed_data: A table of machine_number to numpy_array. Each numpy_array
      corresponds to one dataset point.
    keyed_data_matrix: A table of machine number to numpy array. Each array
      corresponds to all datapoints associated with a machine.
      center_selector: A function that takes a collection of points (from the
        output of a group by key operation) and returns a center of these
        points.
      previous_centers: Centers calcualted in previous iterations.

  Returns:
    The center calcualted in this iteration.
  """
  probabilities = None
  # Merge centers into a single array.
  if previous_centers:
    previous_center_collection = (
        previous_centers | "Flatten centers %d" % i >> beam.Flatten()
        | "Add void key to centers %d " % i >> beam.Map(lambda data: (0, data))
        | "Make center array %d" % i >> beam.CombinePerKey(merge_arrays)
        | "Drop void key %d " % i >> beam.Map(lambda x: x[1]))

    distances = (
        keyed_data_matrix
        | "Calculate distances %d" % i >> beam.Map(
            calculate_collection_distance_to_centers,
            beam.pvalue.AsSingleton(previous_center_collection)))

    normalizer = (
        distances
        | "Normalizer/Add void key %d" % i >> beam.Map(lambda data: (0, data))
        | "Normalizer/ Group by key default key %d" % i >> beam.GroupByKey()
        | "Get normalizer %d" % i >> beam.Map(calculate_normalizer))

    probabilities = (
        distances | "Get probabilities %d" % i >> beam.Map(
            calculate_probabilities, beam.pvalue.AsSingleton(normalizer)))
    probabilities = beam.pvalue.AsIter(probabilities)

  center_candidates = (
      keyed_data
      | "Select candidate centers /GBK %d " % i >> beam.GroupByKey()
      |
      "Select candidate centers / Select %d " % i >> beam.Map(center_selector))

  return (
      center_candidates
      |
      "Select center / Add void key %d " % i >> beam.Map(lambda data: (0, data))
      | "Select center /Group all candidates %d" % i >> beam.GroupByKey()
      | "Get final center %d " % i >> beam.Map(select_center, probabilities))


def sum_vectors(vectors, d):
  output = np.zeros(d)
  num_vectors = 0
  for _, v in enumerate(vectors):
    output += v[0]
    num_vectors += v[1]
  return (output, num_vectors)


def sum_pairs(pairs):
  p1 = 0
  p2 = 0
  for p in pairs:
    p1 += p[0]
    p2 += p[1]
  return (p1, p2)


class ReadNumpyToEvents(beam.DoFn):
  """DoFn to parse a numpy file into rows of numpy in a PCollection."""

  def process(self, text):
    """Reads a numpy file into elements of a PCollection.

    Args:
      text: File encoding a numpy array to be read with np.load.

    Yields:
      One element per row in the numpy file.
    """
    data = np.load(open(text, "rb"))
    for i in range(len(data.shape[0])):
      yield data[i, :]


def eval_pipeline(centers_dir,
                  input_file,
                  output_file,
                  data_recovery_info=None):
  """Evaluate kvariates pipeline.

  Args:
    centers_dir: Directory where centers are written to. (See parse_centers_from
      files).
    input_file: Input data to evaluate.
    output_file: Output file to write results.
    data_recovery_info: If passed, it contains information needed to recover
      centers in the original dimension. Will recover centers in the original
      dimension by doing one step of kmedians on the raw data.
  Returns:
    A pipeline to be run on beam.
  """

  centers = parse_centers_from_files(centers_dir, data_recovery_info)

  def pipeline(root):
    if data_recovery_info is not None:
      data = (
          root
          | beam.Create([data_recovery_info.raw_data])
          | "Split into single arrays" >> ReadNumpyToEvents()
      )
    else:
      data = (
          root | "Read input" >> beam.io.ReadFromText(input_file)
          | "Parse rows" >> beam.Map(parse_row))
    objective_value = (
        data
        | "Distance to center" >> beam.Map(get_distance_to_centers, centers)
        | "Sum distances" >> beam.CombineGlobally(sum))
    return objective_value | "Write objective " >> beam.io.WriteToText(
        output_file)

  return pipeline


def create_pipeline(input_file, output_dir, data_splitter, center_selector,
                    num_workers, num_centers, data_dim):
  """Create kvariates pipeline in beam.

  Args:
    input_file: File with input data. Pipeline expects a file with one row per
      example and each example is encoded as a tab separated set of coordinates.
    output_dir: Directory to output results.
    data_splitter: Function to decide how to split the input data.
    center_selector: A rule for selecting centers
    num_workers: Number of workers to split the data into.
    num_centers: Number of centers to return.
    data_dim: The dimension of the data.

  Returns:
    A pipeline to be ran by beam.
  """
  def pipeline(root):
    data = (
        root | "Read file" >> beam.io.ReadFromText(input_file)
        | "Parse rows" >> beam.Map(parse_row))

    mean_vector = (
        data | "Add one to data " >> beam.Map(lambda x: (x, 1))
        | "Add all vectors " >> beam.CombineGlobally(sum_vectors, data_dim)
        | "Mean vector " >> beam.Map(lambda x: x[0] / x[1]))
    negative_mean_vector = mean_vector | "Negation " >> beam.Map(lambda x: -x)
    data = data | "Center data " >> beam.Map(
        center_data, beam.pvalue.AsSingleton(mean_vector))
    split_data = (
        data
        | "Assign to worker" >> beam.Map(data_splitter, num_workers))
    split_data_matrix = (
        split_data | "Join to single array" >> beam.CombinePerKey(merge_arrays))
    previous_centers = []
    for i in range(num_centers):
      last_center = iteration(
          i, split_data, split_data_matrix, center_selector, previous_centers
      )
      os.makedirs(output_dir)
      center_file = output_dir + "/center-%d@1" % i
      # pylint: disable=expression-not-assigned
      (last_center
       | "Shift back %d " % i >> beam.Map(
           center_data, beam.pvalue.AsSingleton(negative_mean_vector))
       | "Write center %d" % i >> beam.io.WriteToText(center_file))
      previous_centers.append(last_center)

  return pipeline
