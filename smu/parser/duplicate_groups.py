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

"""Converts pairwise conformer duplicates to duplicate bond topologies.

We are given pair wise duplicates for individual conformers with different
bond topology ids. We consider two bond topologies equivalent if any conformer
of the two is considered equivalent. We want to consider a whole group
equivalent as long as there are edges connecting them. Yes, this is a graph
connected components problem.

The program outputs a text file of the bond topology ids that are considered
equivalent. Each line represents one connected component and has a list of space
separated bond topology ids.
"""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import scipy.sparse

from tensorflow.io import gfile

from smu.parser import smu_utils_lib

flags.DEFINE_string(
    'input',
    '/namespace/gas/primary/smu/dataset/list.equivalent_isomers.dat',
    'Input file of duplicate isomers')
flags.DEFINE_string(
    'output',
    '/namespace/gas/primary/smu/dataset/equivalent_bond_topologies.txt',
    'Output file path')

FLAGS = flags.FLAGS


def get_components(df_dups):
  """Return the connected components given pairwise duplicates."""
  graphidx_to_btid = pd.concat([df_dups['btid1'], df_dups['btid2']],
                               ignore_index=True).unique()
  btid_to_graphidx = {
      btid: graphidx for graphidx, btid in enumerate(graphidx_to_btid)
  }
  assert len(graphidx_to_btid) == len(btid_to_graphidx)
  logging.info('From %d pairs, have %d unique bond topologies',
               len(df_dups), len(graphidx_to_btid))
  graph = scipy.sparse.csr_matrix(
      (np.ones(len(df_dups)),
       ([btid_to_graphidx[x] for x in df_dups['btid1']],
        [btid_to_graphidx[x] for x in df_dups['btid2']])),
      shape=(len(graphidx_to_btid), len(graphidx_to_btid)))
  n_components, labels = scipy.sparse.csgraph.connected_components(
      csgraph=graph, directed=False, return_labels=True)

  return graphidx_to_btid, n_components, labels


def write_components_to_file(filename, graphidx_to_btid, n_components, labels):
  """Writes the components to a text file."""
  total_written = 0
  with gfile.GFile(filename, 'w') as outfile:
    for comp_idx in range(n_components):
      btids = graphidx_to_btid[np.where(labels == comp_idx)].astype(str)
      total_written += len(btids)
      outfile.write(' '.join(btids))
      outfile.write('\n')
  assert total_written == labels.shape[0]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Reading duplicate file')
  df_dups = smu_utils_lib.parse_duplicates_file(FLAGS.input)
  logging.info('Finding components')
  graphidx_to_btid, n_components, labels = get_components(df_dups)
  logging.info('Writing output')
  write_components_to_file(FLAGS.output, graphidx_to_btid, n_components, labels)

if __name__ == '__main__':
  app.run(main)
