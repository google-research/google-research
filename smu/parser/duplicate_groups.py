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

flags.DEFINE_string(
    'input',
    '/namespace/gas/primary/smu/dataset/list.equivalent_isomers.dat',
    'Input file of duplicate isomers')
flags.DEFINE_string(
    'output',
    '/namespace/gas/primary/smu/dataset/equivalent_bond_topologies.txt',
    'Output file path')

FLAGS = flags.FLAGS


def parse_duplicates_file(filename):
  """Parses duplciate file into a pandas dataframe.

  The duplciate file supplied by our collaborators (called
  list.equivalent_{isomers,conformers.dat) is a two column, space separated
  file of composite names like x07_n4o3h4.091404.073
  which we parse the names into columns
  * nameX: original composiite name from file
  * stoichX: string for the stoichiometry
  * btidX: bond topology id
  * shortconfidX: 3 digit conformer id
  * confidX: full conformer id that we use (btid * 1000 + shortconfid)
  (for X = 1 or 2)

  Args:
    filename: file to read (usually list.equivalent_isomers.dat)

  Returns:
    pd.DataFrame
  """
  with gfile.GFile(filename) as f:
    df_dups = pd.read_csv(
        f, delim_whitespace=True, names=['name1', 'name2'], header=None)

  for idx in ['1', '2']:
    df_dups = pd.concat([
        df_dups,
        df_dups['name' +
                idx].str.extract(r'x07_([\w\d]+)\.(\d+).(\d+)').rename(columns={
                    0: 'stoich' + idx,
                    1: 'btid' + idx,
                    2: 'shortconfid' + idx
                })
    ],
                        axis=1)
    df_dups['btid' + idx] = df_dups['btid' + idx].astype(int)
    df_dups['shortconfid' + idx] = df_dups['shortconfid' + idx].astype(int)
    df_dups['confid' + idx] = (
        df_dups['btid' + idx] * 1000 + df_dups['shortconfid' + idx])

  return df_dups


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
  df_dups = parse_duplicates_file(FLAGS.input)
  logging.info('Finding components')
  graphidx_to_btid, n_components, labels = get_components(df_dups)
  logging.info('Writing output')
  write_components_to_file(FLAGS.output, graphidx_to_btid, n_components, labels)

if __name__ == '__main__':
  app.run(main)
