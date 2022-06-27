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

"""Common utility functions."""

import os

from absl import logging
import tensorflow.compat.v1 as tf


class Writer:
  """Records data in the experiment."""

  def __init__(self, path):
    """Initializer.

    Args:
      path: String, the path to write the data.
    """
    self._handle = tf.io.gfile.GFile(os.path.join(path, 'history.tsv'), 'w')
    self._handle.write('received_time\tfitness\tgene\n')

  def wait(self):
    """Flushes the handle."""
    self._handle.flush()

  def write(self, data_dict):
    """Writes the data to file.

    Args:
      data_dict: Dict with 3 keys: 'received_time', 'fitness', 'gene'.
    """
    self._handle.write(
        '\t'.join([str(data_dict[key])
                   for key in ['received_time', 'fitness', 'gene']]) + '\n')


def create_writer(path):
  """Creates writer.

  Args:
    path: String, the path to write the data.

  Returns:
    Writer
  """
  return Writer(path)


def get_server_name(prefix, xid, wid):
  """Gets unique server name for each work unit in the experiment.

  Args:
    prefix: String, the prefix of the server.
    xid: Integer, the experiment id.
    wid: Integer, the work unit id.

  Returns:
    String.
  """
  return f'{prefix}-{xid}-{wid}'


def start_population_server(prefix, xid, wid, population):
  """Starts population server.

  Args:
    prefix: String, the prefix of the server.
    xid: Integer, the experiment id.
    wid: Integer, the work unit id.
    population: regularized_evolution.Population, the population in the server.
  """
  del prefix, xid, wid, population
  logging.info(
      'Single machine example has no population server to start. Users can '
      'overwrite this function for their own distributed setting.')


def start_population_client(prefix, xid, wid, population):
  """Starts a population client.

  Args:
    prefix: String, the prefix of the server.
    xid: Integer, the experiment id.
    wid: Integer, the work unit id.
    population: regularized_evolution.Population, the population in the server.
      In distributed setting, the client should connect to the server rather
      than use the input argument in this function. Users should remove this
      input argument in distributed setting.

  Returns:
    A client.
  """
  del prefix, xid, wid
  return population


def start_fingerprint_server(prefix, xid, wid):
  """Starts a fingerprint server.

  Args:
    prefix: String, the prefix of the server.
    xid: Integer, the experiment id.
    wid: Integer, the work unit id.
  """
  del prefix, xid, wid
  logging.info(
      'Single machine example has no fingerprint server to start. Users can '
      'overwrite this function for their own distributed setting.')


def start_fingerprint_client(prefix, xid, wid, fingerprint_server):
  """Starts a fingerprint client.

  Args:
    prefix: String, the prefix of the server.
    xid: Integer, the experiment id.
    wid: Integer, the work unit id.
    fingerprint_server: Fingerprint server. In distributed setting, the client
      should connect to the server rather than use the input argument in this
      function. Users should remove this input argument in distributed setting.

  Returns:
    A client.
  """
  del prefix, xid, wid
  return fingerprint_server


def functional_equivalence_checking(query, fingerprints_client):
  """Checks functional equivalence.

  Args:
    query: Integer, the fingerprint to query.
    fingerprints_client: Client to check fingerprints.

  Returns:
    Float, the fitness of the equivalence if found, otherwise None.
  """
  return fingerprints_client.get(query, None)
