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

# Lint as: python3
"""Converts data from csv to an scRNA-seq python format."""

from absl import app
from absl import flags

import anndata
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string('input_csv_prefix', None,
                    'Name used during the csv generation.')
flags.DEFINE_enum('format', None, ['anndata'], 'Format to convert the data to.')


def csv_to_h5ad(csv_prefix):
  count = pd.read_csv(f'{csv_prefix}.counts.csv', index_col=0)
  metadata = pd.read_csv(f'{csv_prefix}.metadata.csv', index_col=0)
  featuredata = pd.read_csv(f'{csv_prefix}.featuredata.csv', index_col=0)
  adata = anndata.AnnData(X=count.transpose(), obs=metadata, var=featuredata)
  adata.write(f'{csv_prefix}.h5ad')


def main(unused_argv):
  if FLAGS.format == 'anndata':
    csv_to_h5ad(FLAGS.input_csv_prefix)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_csv_prefix')
  flags.mark_flag_as_required('format')
  app.run(main)
