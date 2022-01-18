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

"""Learner config."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class LearnerConfig:
  """Config for the learner."""
  # Checkpoint save period in seconds.
  save_checkpoint_secs: int = 1800
  # Total iterations to train for.
  total_iterations: int = int(1e6)
  # Batch size for training.
  batch_size: int = 64
  # Whether actors block when enqueueing.
  replay_queue_block: int = 0
  # Batch size for the recurrent inference.
  recurrent_inference_batch_size: int = 32
  # Batch size for initial inference.
  initial_inference_batch_size: int = 4
  # Number of TPUs for training.
  num_training_tpus: int = 1
  # Path to the checkpoint used to initialize the agent.
  init_checkpoint: Optional[str] = None
  # Size of the replay queue.
  replay_buffer_size: int = 1000
  # Size of the replay queue.
  replay_queue_size: int = 100
  # After sampling an episode from the replay buffer, the corresponding priority
  # is set to this value. For a value < 1, no priority update will be done.
  replay_buffer_update_priority_after_sampling_value: float = 1e-6
  # Size of the replay buffer (in number of batches stored).
  flush_learner_log_every_n_s: int = 60
  # If true, logs are written to tensorboard.
  enable_learner_logging: bool = True
  # Log frequency in number of training steps.
  log_frequency: int = 100
  # Exponent used when computing the importance sampling correction. 0 means no
  # importance sampling correction. 1 means full importance sampling correction.
  importance_sampling_exponent: float = 0.0
  # For sampling from priority queue. 0 for uniform. The higher this value the
  # more likely it is to sample an instance for which the model predicts a wrong
  # value.
  priority_sampling_exponent: float = 0.0
  # How many batches the learner skips.
  learner_skip: int = 0
  # Save the agent in ExportAgent format.
  export_agent: bool = False
  # L2 penalty.
  weight_decay: float = 1e-5
  # Scaling for the policy loss term.
  policy_loss_scaling: float = 1.0
  # Scaling for the reward loss term.
  reward_loss_scaling: float = 1.0
  # Entropy loss for the policy loss term.
  policy_loss_entropy_regularizer: float = 0.0
  # Gradient norm clip (0 for no clip).
  gradient_norm_clip: float = 0.0
  # Enables debugging.
  debug: bool = False

  # The fields below are defined in seed_rl/common/common_flags.py

  # TensorFlow log directory.
  logdir: str = '/tmp/agent'
  # Server address.
  server_address: str = 'unix:/tmp/agent_grpc'
