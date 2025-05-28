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

"""Base template config for pre-training and fine-tuning."""

import enum
from typing import Optional

import ml_collections


class ModelArchitecture(enum.Enum):
  """Determines model architecture - in particular, the mixing layer.

  Possible options:
    BERT: BERT architecture using self-attention for mixing.
    F_NET: Fourier Transform mixing.
    LINEAR: Mixing using dense matrix multiplications with learnable weights.
    H_NET: Hartley Transform mixing.
    C_NET: Mixing using parameterized circulant matrix multiplications.
    T_NET: Mixing using parameterized toeplitz matrix multiplications.
  """
  BERT = "bert"
  F_NET = "f_net"
  LINEAR = "linear"
  H_NET = "h_net"
  C_NET = "c_net"
  T_NET = "t_net"


class TrainingMode(str, enum.Enum):
  """Determines type of training.

  Possible options:
    PRETRAINING: Masked Language Modelling and Next Sentence Prediction
      pretraining pmap to distribute training across devices.
    CLASSIFICATION: Fine-tuning on GLUE, SuperGLUE or SQuAD tasks using pmap.
  """
  PRETRAINING = "pretraining"
  CLASSIFICATION = "classification"


class LayerLayout(str, enum.Enum):
  """Specifies where particular sublayers are placed.

  In the descriptions below, a "replacement" layer refers to a sparse MLP or
  an attention layer, while a "regular" layer refers to a dense MLP or standard
  mixing layer.

  Possible options:
    BOTTOM: Replacement layers come first in the network, followed by regular
      layers.
    MIDDLE: Replacement layers are in the middle of the network, sandwiched
      between regular layers.
    MIXED: Replacement layers are interleaved throughout the model; e.g. every
      third layer.
    TOP: Regular layers come first in the network, with Replacement layers
      placed in the final few layers.
  """
  BOTTOM = "bottom"
  MIDDLE = "middle"
  MIXED = "mixed"
  TOP = "top"


class DispatchAlgorithm(enum.Enum):
  """Determines algorithm used to route inputs to experts in MoE layers.

  MASK_* dispatch algorithms use masked matmuls to dispatch and combine expert
  inputs/outputs, whereas SCATTER_* algorithms use scatter and gather to
  dispatch and combine expert inputs/outputs. MASK_* algorithms are generally
  faster on TPU, while SCATTER_* algorithms are faster on GPU and CPU.

  Possible options:
    MASK_TOKENS_CHOOSE: Tokens choose their top-k experts. Items are routed to
      their choice of expert until the expert's capacity is reached. There is no
      guarantee that each token is processed by an expert, or that each expert
      receives at least one token. This common routing algorithm is used
      in Sparsely-Gated Mixture-of-Experts (https://arxiv.org/abs/1701.06538),
      Vision MoE (https://arxiv.org/abs/2106.05974), Switch Transformer
      (https://arxiv.org/abs/2101.03961), Designing Effective Sparse Expert
      Models (https://arxiv.org/abs/2202.08906) and many others.
    MASK_EXPERTS_CHOOSE: Experts choose their top tokens. Each expert selects
      its top EXPERT_CAPACITY tokens. A given token may be processed by multiple
      experts or none at all. "Experts choose" routing should generally not be
      used in decoder blocks because it breaks the autoregressive behavior --
      the model may learn to cheat by relying on future token information to
      improve current token predictions. This router was introduced in
      Mixture-of-Experts with Expert Choice (https://arxiv.org/abs/2202.09368).
    SCATTER_TOKENS_CHOOSE: Same as MASK_TOKENS_CHOOSE, except that dispatch and
      combine logic uses scatters and gathers instead of matmuls.
  """
  MASK_TOKENS_CHOOSE = "mask_tokens_choose"
  MASK_EXPERTS_CHOOSE = "mask_experts_choose"
  SCATTER_TOKENS_CHOOSE = "scatter_tokens_choose"


def get_config():
  """Base config for training models."""
  config = ml_collections.ConfigDict()

  # Determines which model to use.
  # Specific mixing sublayers may be replaced with attention using
  # config.attention_layout and config.num_attention_layers.
  config.model_arch: str = ModelArchitecture.LINEAR.name

  # How often to save the model checkpoint.
  config.save_checkpoints_steps: int = 1000
  # Number of past checkpoint files to keep.
  config.checkpoints_to_keep: int = 2

  # Frequency fo eval during training, e.g. every 1000 steps.
  config.eval_frequency: int = 1000

  # Total batch size for training.
  config.train_batch_size: int = 32
  # Number of mini-steps over which to split train_batch_size and accumulate the
  # gradient. By splitting the training batch size over multiple steps, we can
  # save memory and handle larger training batch sizes. If none or 1, no
  # gradient accumulation is used.  Currently only works with JAX codebase.
  config.gradient_accum_steps: Optional[int] = None
  # Total batch size for eval. The eval batch is NOT split over
  # gradient_accum_steps.
  config.eval_batch_size: int = 8

  # The base learning rate for Adam.
  config.learning_rate: float = 1e-4

  # If set, determines how much to clip the gradient during training.
  config.clipped_grad_norm: Optional[float] = None

  # Initial checkpoint directory or filepath (usually from a pre-trained model).
  config.init_checkpoint_dir: str = ""

  # Model parameters.

  # For pre-training, we only need 2 segment types (for NSP), but we allow up to
  # 4 for GLUE/SuperGLUE fine-tuning.
  config.type_vocab_size: int = 4
  # Embedding dimension for each token.
  config.d_emb: int = 512
  # Hidden dimension of model.
  config.d_model: int = 512
  # Hidden dimension for feed-forward layer.
  config.d_ff: int = 2048
  # The maximum total input sequence length after tokenization. Sequences longer
  # than this will be truncated, and sequences shorter than this will be padded.
  config.max_seq_length: int = 512
  # Number of self-attention heads. Only used for BERT models.
  config.num_heads: int = 8
  # Number of model blocks / layers.
  config.num_layers: int = 14
  # Regular dropout rate, applied throughout model.
  config.dropout_rate: float = 0.1
  # Dropout rate used in mixing module, e.g. self-attention sublayer.
  config.mixing_dropout_rate: float = 0.1

  # Determines whether the FFT is used in lieu of matrix multiplications.
  # Only relevant for the following models:
  # - FNet: If true, favor FFT over matrix multiplications to compute the DFT.
  # - CNet/TNet: If true, compute circulant/toeplitz matrix multiplications
  #              using FFT. Only used in JAX codebase; TF codebase always uses
  #              tf.linalg.LinearOperator(s).
  config.use_fft: bool = False

  # Specific where self-attention sublayers replace mixing sublayers.
  config.attention_layout: LayerLayout = LayerLayout.TOP
  config.num_attention_layers: int = 4

  # Random number generator seed.
  config.seed: int = 0

  # Dummy parameter for repeated runs.
  config.trial: int = 0

  # Mixture of experts architectures.
  #
  # Specifies where sparse MoE sublayers replace dense feed-forward
  # sublayers.
  config.moe_layout: LayerLayout = LayerLayout.MIDDLE
  config.num_moe_layers: int = 4

  # Currently, the number of experts must be a multiple of the number of local
  # devices: num_experts = j * jax.device_count(), with j a positive integer.
  config.num_experts: int = 16
  # The total number of tokens (across the global batch) is subdivided into
  # groups of this size, on each device. Router computations are then performed
  # on a per-group basis.
  config.max_group_size: int = 4096
  # Hidden dimension for individual expert feed-forward layers.
  config.expert_d_ff: int = 2048
  # Dropout rate for individual experts.
  config.expert_dropout_rate: float = 0.1
  # Router dispatch algorithm -- how tokens are distributed amongst experts.
  config.dispatch_algorithm: DispatchAlgorithm = DispatchAlgorithm.MASK_EXPERTS_CHOOSE
  # Scaling factor to increase the expert token capacity during training. This
  # factor plays an analogous, but slightly different, role depending on the
  # dispatch_algorithm:
  # - For *_TOKENS_CHOOSE routing, the capacity_factor only affects the
  #   maximum number of tokens that an expert will process. It does not affect
  #   how many experts a given token is routed to; see num_selected_experts.
  # - For *_EXPERTS_CHOOSE routing, because experts always fill their buffer,
  #   increasing the capacity_factor will increase the number of tokens that
  #   an expert will process AND will indirectly increase the number of experts
  #   that a given token is routed to.
  config.train_capacity_factor: float = 1.0
  # As above, but used during evaluation.
  config.eval_capacity_factor: float = 1.0
  # Minimum token processing capacity for each expert.
  config.min_expert_capacity: int = 4
  # Maximum number of experts to which each token seeks to be routed. Only used
  # by *_TOKENS_CHOOSE ("tokens choose experts") routing. Tokens may be routed
  # to fewer experts if some experts reach capacity.
  config.num_selected_experts: int = 1
  # Whether to use Batch Prioritized Routing (BPR). With BPR, we prioritize
  # routing those top-k tokens with the highest router probability, rather than
  # simply using the left-to-right ordering of tokens in the batch. This
  # prioritization is important because the expert's have limited capacity.
  config.batch_prioritized_routing: bool = True
  # Scaling of router load balancing loss. Only used by *_TOKENS_CHOOSE routing.
  # The load balancing loss penalizes those cases where the routing between
  # experts is unbalanced.
  config.auxiliary_loss_factor: float = 0.01
  # Scaling of router z-loss. Turned off by default. Router z-loss encourages
  # router logits to remain small in an effort to improve stability.
  config.router_z_loss_factor: float = 0.0001
  # Amplitude of jitter noise applied to during token routing.
  config.jitter_noise: float = 0.

  return config
