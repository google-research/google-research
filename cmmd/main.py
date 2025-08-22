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

"""The main entry point for the CMMD calculation."""

from absl import app
from absl import flags
from cmmd import distance
from cmmd import embedding
from cmmd import io_util
import numpy as np


_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 32, 'Batch size for embedding generation.'
)
_MAX_COUNT = flags.DEFINE_integer(
    'max_count', -1, 'Maximum number of images to read from each directory.'
)


def compute_cmmd(
    ref_dir, eval_dir, batch_size = 32, max_count = -1
):
  """Calculates the CMMD distance between reference and eval image sets.

  Args:
    ref_dir: Path to the directory containing reference images.
    eval_dir: Path to the directory containing images to be evaluated.
    batch_size: Batch size used in the CLIP embedding calculation.
    max_count: Maximum number of images to use from each directory. A
      non-positive value reads all images available except for the images
      dropped due to batching.

  Returns:
    The CMMD value between the image sets.
  """
  embedding_model = embedding.ClipEmbeddingModel()
  ref_embs = io_util.compute_embeddings_for_dir(
      ref_dir, embedding_model, batch_size, max_count
  )
  eval_embs = io_util.compute_embeddings_for_dir(
      eval_dir, embedding_model, batch_size, max_count
  )
  val = distance.mmd(ref_embs, eval_embs)
  return np.asarray(val)


def main(argv):
  if len(argv) != 3:
    raise app.UsageError('Too few/too many command-line arguments.')
  _, dir1, dir2 = argv
  print(
      'The CMMD value is: '
      f' {compute_cmmd(dir1, dir2, _BATCH_SIZE.value, _MAX_COUNT.value):.3f}'
  )


if __name__ == '__main__':
  app.run(main)
