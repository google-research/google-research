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

"""Nearest-neighbor evaluator."""

from typing import List

from .base import Evaluator
from .base import EvaluatorOutput
import numpy as np
from scipy.spatial.distance import cdist
from xirl.models import SelfSupervisedOutput


class NearestNeighbourVisualizer(Evaluator):
  """Nearest-neighbour frame visualizer."""

  def __init__(
      self,
      distance,
      num_videos,
      num_ctx_frames,
  ):
    """Constructor.

    Args:
      distance: The distance metric to use when calculating nearest-neighbours.
      num_videos: The number of video sequences to display.
      num_ctx_frames: The number of context frames stacked together for each
        individual video frame.

    Raises:
      ValueError: If the distance metric is invalid.
    """
    super().__init__(inter_class=True)

    if distance not in ["sqeuclidean", "cosine"]:
      raise ValueError(
          "{} is not a supported distance metric.".format(distance))

    self.distance = distance
    self.num_videos = num_videos
    self.num_ctx_frames = num_ctx_frames

  def evaluate(self, outs):
    """Sample source and target sequences and plot nn frames."""

    def _reshape(frame):
      s, h, w, c = frame.shape
      seq_len = s // self.num_ctx_frames
      return frame.reshape(seq_len, self.num_ctx_frames, h, w, c)

    embs = [o.embs for o in outs]
    frames = [o.frames for o in outs]

    # Randomly sample the video sequences we'd like to plot.
    seq_idxs = np.random.choice(
        np.arange(len(embs)), size=self.num_videos, replace=False)

    # Perform nearest-neighbor lookup in embedding space and retrieve the
    # frames associated with those embeddings.
    cand_frames = [_reshape(frames[seq_idxs[0]])[:, -1]]
    for cand_idx in seq_idxs[1:]:
      dists = cdist(embs[seq_idxs[0]], embs[cand_idx], self.distance)
      nn_ids = np.argmin(dists, axis=1)
      c_frames = _reshape(frames[cand_idx])
      nn_frames = [c_frames[idx, -1] for idx in nn_ids]
      cand_frames.append(np.stack(nn_frames))

    video = np.stack(cand_frames)
    return EvaluatorOutput(video=video)
