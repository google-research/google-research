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

r"""Config for unsupervised training on Tetrominoes with 512 train samples."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 42
  config.seed_data = True

  config.batch_size = 64
  config.num_train_steps = 20000
  config.init_checkpoint = ml_collections.ConfigDict()
  config.init_checkpoint.xid = 0  # Disabled by default.
  config.init_checkpoint.wid = 1

  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.optimizer = "adam"

  config.optimizer_configs.grad_clip = ml_collections.ConfigDict()
  config.optimizer_configs.grad_clip.clip_method = "clip_by_global_norm"
  config.optimizer_configs.grad_clip.clip_value = 0.05

  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = "compound"
  config.lr_configs.factors = "constant * cosine_decay * linear_warmup"
  config.lr_configs.warmup_steps = 5000
  config.lr_configs.steps_per_cycle = config.get_ref("num_train_steps")
  # from the original Slot Attention
  config.lr_configs.base_learning_rate = 4e-4

  # TODO(obvis): Implement masked evaluation.
  config.eval_pad_last_batch = False  # True
  config.log_loss_every_steps = 50
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 1000

  config.train_metrics_spec = {
      "loss": "loss",
      "ari": "ari",
      "ari_nobg": "ari_nobg",
  }
  config.eval_metrics_spec = {
      "eval_loss": "loss",
      "eval_ari": "ari",
      "eval_ari_nobg": "ari_nobg",
  }

  config.data = ml_collections.ConfigDict({
      "dataset_name": "tetrominoes",
      "shuffle_buffer_size": config.batch_size * 8,
      "resolution": (35, 35)
  })

  config.max_instances = 4
  config.num_slots = config.max_instances  # Only used for metrics.
  config.logging_min_n_colors = config.max_instances

  config.preproc_train = [
      "tfds_image_to_tfds_video",
      "video_from_tfds",
      "sparse_to_dense_annotation(max_instances=3)"
  ]

  config.preproc_eval = [
      "tfds_image_to_tfds_video",
      "video_from_tfds",
      "sparse_to_dense_annotation(max_instances=3)"
  ]

  config.eval_slice_size = 1
  config.eval_slice_keys = ["video", "segmentations_video"]

  # Dictionary of targets and corresponding channels. Losses need to match.
  targets = {"video": 3}
  config.losses = {"recon": {"targets": list(targets)}}
  config.losses = ml_collections.ConfigDict({
      f"recon_{target}": {"loss_type": "recon", "key": target}
      for target in targets})

  config.model = ml_collections.ConfigDict({
      "module": "invariant_slot_attention.modules.SAVi",

      # Encoder.
      "encoder": ml_collections.ConfigDict({
          "module": "invariant_slot_attention.modules.FrameEncoder",
          "reduction": "spatial_flatten",
          "backbone": ml_collections.ConfigDict({
              "module": "invariant_slot_attention.modules.SimpleCNN",
              "features": [64, 64, 64, 64],
              "kernel_size": [(5, 5), (5, 5), (5, 5), (5, 5)],
              "strides": [(1, 1), (1, 1), (1, 1), (1, 1)]
          }),
          "pos_emb": ml_collections.ConfigDict({
              "module": "invariant_slot_attention.modules.PositionEmbedding",
              "embedding_type": "linear",
              "update_type": "concat"
          }),
      }),

      # Corrector.
      "corrector": ml_collections.ConfigDict({
          "module": "invariant_slot_attention.modules.SlotAttentionTranslEquiv",
          "num_iterations": 3,
          "qkv_size": 64,
          "mlp_size": 128,
          "grid_encoder": ml_collections.ConfigDict({
              "module": "invariant_slot_attention.modules.MLP",
              "hidden_size": 128,
              "layernorm": "pre"
          }),
          "add_rel_pos_to_values": True,  # V3
          "zero_position_init": False,  # Random positions.
      }),

      # Predictor.
      # Removed since we are running a single frame.
      "predictor": ml_collections.ConfigDict({
          "module": "invariant_slot_attention.modules.Identity"
      }),

      # Initializer.
      "initializer": ml_collections.ConfigDict({
          "module":
              "invariant_slot_attention.modules.ParamStateInitRandomPositions",
          "shape":
              (4, 64),  # (num_slots, slot_size)
      }),

      # Decoder.
      "decoder": ml_collections.ConfigDict({
          "module":
              "invariant_slot_attention.modules.SiameseSpatialBroadcastDecoder",
          "resolution": (35, 35),  # Update if data resolution or strides change
          "backbone": ml_collections.ConfigDict({
              "module": "invariant_slot_attention.modules.MLP",
              "hidden_size": 256,
              "output_size": 256,
              "num_hidden_layers": 5,
              "activate_output": True
          }),
          "target_readout": ml_collections.ConfigDict({
              "module": "invariant_slot_attention.modules.Readout",
              "keys": list(targets),
              "readout_modules": [ml_collections.ConfigDict({  # pylint: disable=g-complex-comprehension
                  "module": "invariant_slot_attention.modules.MLP",
                  "num_hidden_layers": 0,
                  "hidden_size": 0,
                  "output_size": targets[k]}) for k in targets],
          }),
          "relative_positions": True,
          "pos_emb": ml_collections.ConfigDict({
              "module":
                  "invariant_slot_attention.modules.RelativePositionEmbedding",
              "embedding_type":
                  "linear",
              "update_type":
                  "project_add",
          }),
      }),
      "decode_corrected": True,
      "decode_predicted": False,
  })

  # Which video-shaped variables to visualize.
  config.debug_var_video_paths = {
      "recon_masks": "decoder/alphas_softmaxed/__call__/0",  # pylint: disable=line-too-long
  }

  # Define which attention matrices to log/visualize.
  config.debug_var_attn_paths = {
      "corrector_attn": "corrector/InvertedDotProductAttention_0/GeneralizedDotProductAttention_0/attn"  # pylint: disable=line-too-long
  }

  # Widths of attention matrices (for reshaping to image grid).
  config.debug_var_attn_widths = {
      "corrector_attn": 35,
  }

  return config


