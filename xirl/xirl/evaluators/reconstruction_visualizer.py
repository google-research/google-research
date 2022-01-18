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

"""Frame reconstruction visualizer."""

from typing import List

from .base import Evaluator
from .base import EvaluatorOutput
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from xirl.models import SelfSupervisedReconOutput


class ReconstructionVisualizer(Evaluator):
  """Frame reconstruction visualizer."""

  def __init__(self, num_frames, num_ctx_frames):
    """Constructor.

    Args:
      num_frames: The number of reconstructed frames in a sequence to display.
      num_ctx_frames: The number of context frames stacked together for each
        individual video frame.
    """
    super().__init__(inter_class=False)

    self.num_frames = num_frames
    self.num_ctx_frames = num_ctx_frames

  def evaluate(self, outs):
    """Plot a frame along with its reconstruction."""

    def _remove_ctx_frames(frame):
      s, h, w, c = frame.shape
      seq_len = s // self.num_ctx_frames
      frame = frame.reshape(seq_len, self.num_ctx_frames, h, w, c)
      return frame[:, -1]

    frames = [o.frames for o in outs]
    recons = [o.reconstruction for o in outs]

    r_idx = np.random.randint(0, len(frames))
    frame = _remove_ctx_frames(frames[r_idx])
    recon = _remove_ctx_frames(recons[r_idx])

    # Select which frames we want to plot from the sequence.
    frame_idxs = np.random.choice(
        np.arange(frame.shape[0]), size=self.num_frames, replace=False)
    frame = frame[frame_idxs]
    recon = recon[frame_idxs]

    # Downsample the frame.
    _, _, sh, _ = recon.shape
    _, _, h, _ = frame.shape
    scale_factor = sh / h
    frame_ds = F.interpolate(
        torch.from_numpy(frame).permute(0, 3, 1, 2),
        mode='bilinear',
        scale_factor=scale_factor,
        recompute_scale_factor=False,
        align_corners=True).permute(0, 2, 3, 1).numpy()

    # Clip reconstruction between 0 and 1.
    recon = np.clip(recon, 0.0, 1.0)

    imgs = np.concatenate([frame_ds, recon], axis=0)
    img = make_grid(torch.from_numpy(imgs).permute(0, 3, 1, 2), nrow=2)

    return EvaluatorOutput(image=img.permute(1, 2, 0).numpy())
