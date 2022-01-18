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

"""2D embedding visualizer."""

from typing import List

from .base import Evaluator
from .base import EvaluatorOutput
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from xirl.models import SelfSupervisedOutput


class EmbeddingVisualizer(Evaluator):
  """Visualize PCA of the embeddings."""

  def __init__(self, num_seqs):
    """Constructor.

    Args:
      num_seqs: How many embedding sequences to visualize.

    Raises:
      ValueError: If the distance metric is invalid.
    """
    super().__init__(inter_class=True)

    self.num_seqs = num_seqs

  def _gen_emb_plot(self, embs):
    """Create a pyplot plot and save to buffer."""
    fig = plt.figure()
    for emb in embs:
      plt.scatter(emb[:, 0], emb[:, 1])
    fig.canvas.draw()
    img_arr = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close()
    return img_arr

  def evaluate(self, outs):
    embs = [o.embs for o in outs]

    # Randomly sample the embedding sequences we'd like to plot.
    seq_idxs = np.random.choice(
        np.arange(len(embs)), size=self.num_seqs, replace=False)
    seq_embs = [embs[idx] for idx in seq_idxs]

    # Subsample embedding sequences to make them the same length.
    seq_lens = [s.shape[0] for s in seq_embs]
    min_len = np.min(seq_lens)
    same_length_embs = []
    for emb in seq_embs:
      emb_len = len(emb)
      stride = emb_len / min_len
      idxs = np.arange(0.0, emb_len, stride).round().astype(int)
      idxs = np.clip(idxs, a_min=0, a_max=emb_len - 1)
      idxs = idxs[:min_len]
      same_length_embs.append(emb[idxs])

    # Flatten embeddings to perform PCA.
    same_length_embs = np.stack(same_length_embs)
    num_seqs, seq_len, emb_dim = same_length_embs.shape
    embs_flat = same_length_embs.reshape(-1, emb_dim)
    embs_2d = PCA(n_components=2, random_state=0).fit_transform(embs_flat)
    embs_2d = embs_2d.reshape(num_seqs, seq_len, 2)

    image = self._gen_emb_plot(embs_2d)
    return EvaluatorOutput(image=image)
