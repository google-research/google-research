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

import anndata
import numpy as np
import scipy.sparse
import tensorflow as tf

from schptm_benchmark import resize_bins_lib


def generate_dummy_data(binsize=1000):
  test_x = scipy.sparse.csr_matrix(np.eye(10))
  test_bins = [f'chr1:{i*binsize}-{(i+1)*binsize}' for i in range(0, 10)]
  test_adata = anndata.AnnData(test_x)
  test_adata.var_names = test_bins
  return test_adata


class MergeBinsTest(tf.test.TestCase):

  def test_proper_sum(self):
    test_adata = generate_dummy_data(1000)
    resized_adata = resize_bins_lib.merge_bins(test_adata, 2000)

    self.assertAllEqual(
        np.sum(test_adata.X, axis=1), np.sum(resized_adata.X, axis=1))


if __name__ == '__main__':
  tf.test.main()
