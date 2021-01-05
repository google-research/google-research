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

r"""Cycle consistency loss for unsupervised training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from tcc.algos.algorithm import Algorithm
from tcc.config import CONFIG
from tcc.tcc.alignment import compute_alignment_loss

FLAGS = flags.FLAGS


class Alignment(Algorithm):
  """Uses cycle-consistency loss to perform unsupervised training."""

  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels, seq_labels):
    if training:
      batch_size = CONFIG.TRAIN.BATCH_SIZE
      num_steps = CONFIG.TRAIN.NUM_FRAMES
    else:
      batch_size = CONFIG.EVAL.BATCH_SIZE
      num_steps = CONFIG.EVAL.NUM_FRAMES

    loss = compute_alignment_loss(
        embs,
        batch_size,
        steps=steps,
        seq_lens=seq_lens,
        stochastic_matching=CONFIG.ALIGNMENT.STOCHASTIC_MATCHING,
        normalize_embeddings=False,
        loss_type=CONFIG.ALIGNMENT.LOSS_TYPE,
        similarity_type=CONFIG.ALIGNMENT.SIMILARITY_TYPE,
        num_cycles=int(batch_size * num_steps * CONFIG.ALIGNMENT.FRACTION),
        cycle_length=CONFIG.ALIGNMENT.CYCLE_LENGTH,
        temperature=CONFIG.ALIGNMENT.SOFTMAX_TEMPERATURE,
        label_smoothing=CONFIG.ALIGNMENT.LABEL_SMOOTHING,
        variance_lambda=CONFIG.ALIGNMENT.VARIANCE_LAMBDA,
        huber_delta=CONFIG.ALIGNMENT.HUBER_DELTA,
        normalize_indices=CONFIG.ALIGNMENT.NORMALIZE_INDICES)

    return loss
