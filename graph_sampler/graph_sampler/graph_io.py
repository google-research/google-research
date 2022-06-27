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

"""Utilities for reading and writing files of graphs.

We use graphml format because it's the only one I could find that preserves all
the graph and vertex attributes.

As a (beautiful?) hack, we keep all the output in a single file by dumping a
dictionary of statistics into an igraph object at the end of the file. For
example, to apply rejection sampling to get a uniform sampling of graphs, we
need to know the importance weight of each graph, but also the maximum
importance of any graph.
"""

import os
import tempfile
from typing import Any, Dict, Iterator, TextIO
import igraph

Graph = igraph.Graph


def write_graph(graph, file_obj):
  """Write the graph in graphml format to the file.

  Note that we use graphml format because it's the only one I could find that
  preserves all the graph and vertex attributes.

  You can write multiple graphs to the same file and graph_reader will do the
  work to split things for igraph.

  Args:
    graph: an igraph.Graph.
    file_obj: file object to write to.
  """
  graph.write(file_obj, format='graphml')


def write_stats(stats, file_obj):
  """Writes a dictionary of statistics as an igraph graph."""
  stats_graph = igraph.Graph()
  for k, v in stats.items():
    stats_graph[k] = v
  write_graph(stats_graph, file_obj)


def graph_reader(file_obj):
  """Yields graphs from the given file_obj, skipping the stats graph."""
  prev_graph = None
  for graph in _graph_reader(file_obj):
    if prev_graph is not None:
      yield prev_graph
    prev_graph = graph


def get_stats(graph_fname):
  """Returns the stats dictionary from the end of the given file."""
  with open(graph_fname) as graph_file:
    # Go to near the end of the file.
    file_size = os.path.getsize(graph_fname)
    graph_file.seek(max(0, file_size - 5000))

    # Read until we get to the end of a graph.
    while graph_file.readline() != '</graphml>\n':
      pass

    # Read all remaining graphs. The last one remains in the variable `graph`.
    graph = None
    for graph in _graph_reader(graph_file):
      pass

  if graph is None:
    raise AssertionError(f'Failed to find stats graph for file {graph_fname}')
  if graph.vs:
    raise AssertionError(f'Last graph in file {graph_fname} is not a stats '
                         f'graph. Did the writer crash before writing stats?')

  return {k: graph[k] for k in graph.attributes()}


def _graph_reader(file_obj):
  # Unfortunately, igraph doesn't support file like objects, so we have to
  # write out temporary files and read them back in.
  tmp_file = tempfile.NamedTemporaryFile('w', delete=False)
  num_lines_in_temp_file = 0

  def extract_graph_from_tempfile(reset_tempfile=True):
    """Reads one graphml xml object."""
    nonlocal tmp_file, num_lines_in_temp_file
    assert num_lines_in_temp_file
    tmp_file.close()
    g = igraph.Graph.Read(tmp_file.name, format='graphml')
    os.remove(tmp_file.name)

    if reset_tempfile:
      tmp_file = tempfile.NamedTemporaryFile('w', delete=False)
      num_lines_in_temp_file = 0

    return g

  first_line = True
  for line in file_obj:
    if first_line:
      first_line = False
    elif line.startswith('<?xml'):
      # We've hit the start of the next graph. Emit the current one.
      yield extract_graph_from_tempfile()
    tmp_file.write(line)
    num_lines_in_temp_file += 1

  yield extract_graph_from_tempfile(reset_tempfile=False)
