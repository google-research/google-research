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

"""TCC trainer."""

from typing import Dict, List, Union

from ml_collections import ConfigDict
import torch
from xirl.losses import compute_tcc_loss
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class TCCTrainer(Trainer):
  """A trainer for Temporal Cycle Consistency Learning [1].

  References:
    [1]: arxiv.org/abs/1904.07846
  """

  def __init__(
      self,
      model,
      optimizer,
      device,
      config,
  ):
    super().__init__(model, optimizer, device, config)

    self.normalize_embeddings = config.MODEL.NORMALIZE_EMBEDDINGS
    self.stochastic_matching = config.LOSS.TCC.STOCHASTIC_MATCHING
    self.loss_type = config.LOSS.TCC.LOSS_TYPE
    self.similarity_type = config.LOSS.TCC.SIMILARITY_TYPE
    self.cycle_length = config.LOSS.TCC.CYCLE_LENGTH
    self.temperature = config.LOSS.TCC.SOFTMAX_TEMPERATURE
    self.label_smoothing = config.LOSS.TCC.LABEL_SMOOTHING
    self.variance_lambda = config.LOSS.TCC.VARIANCE_LAMBDA
    self.huber_delta = config.LOSS.TCC.HUBER_DELTA
    self.normalize_indices = config.LOSS.TCC.NORMALIZE_INDICES

  def compute_loss(
      self,
      embs,
      batch,
  ):
    steps = batch["frame_idxs"].to(self._device)
    seq_lens = batch["video_len"].to(self._device)

    # Dynamically determine the number of cycles if using stochastic
    # matching.
    batch_size, num_cc_frames = embs.shape[:2]
    num_cycles = int(batch_size * num_cc_frames)

    return compute_tcc_loss(
        embs=embs,
        idxs=steps,
        seq_lens=seq_lens,
        stochastic_matching=self.stochastic_matching,
        normalize_embeddings=self.normalize_embeddings,
        loss_type=self.loss_type,
        similarity_type=self.similarity_type,
        num_cycles=num_cycles,
        cycle_length=self.cycle_length,
        temperature=self.temperature,
        label_smoothing=self.label_smoothing,
        variance_lambda=self.variance_lambda,
        huber_delta=self.huber_delta,
        normalize_indices=self.normalize_indices,
    )
