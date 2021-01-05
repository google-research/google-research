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

"""ConfigDict for T5-base on GLUE."""
import ml_collections


def get_config():
  """ConfigDict for T5-base on GLUE."""
  config = ml_collections.ConfigDict()

  # T5 pretrained checkpoint to use.
  config.restore_t5_checkpoint = (
      'gs://t5-data/pretrained_models/base/model.ckpt-999900')
  # Name of T5 task/mixture to use for finetuning.
  config.mixture_or_task_name = 'glue_v002_proportional'
  # Whether to use T5 preprocessing cache for train task/mixture.
  config.train_use_cached = False
  # Name of T5 task/mixture to use for evaluation.
  config.eval_mixture_or_task_name = 'glue_v002_proportional'
  # Whether to use T5 preprocessing cache for eval task/mixture.
  config.eval_use_cached = False
  # Name of T5 task/mixture split to use for evaluation.
  config.eval_split = 'validation'

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = False
  # Save a checkpoint every Nth epoch.
  config.checkpoint_freq = 5

  # Number of epochs to train for.
  config.num_epochs = 100
  # Number of steps per epoch.
  config.steps_per_epoch = 1000
  # Number of steps to take during evaluation.
  config.num_eval_steps = 20

  # Collect Xprof traces on host 0.
  config.xprof = True
  # Whether to use hardware rng for dropout.
  config.hardware_rng = True
  # Integer for PRNG random seed.
  config.random_seed = 0
  # Use infeed in training loop.
  config.infeed = True

  # Total batch size for training.
  config.batch_size = 128
  # Total batch size for inference on tasks.
  config.eval_batch_size = 128
  # Number of gradient-accumulating microbatches.
  config.microbatches = None
  # Number of SPMD partitions to use.
  config.num_partitions = 1
  # Beam size for inference.
  config.beam_size = 1

  # Learning rate schedule.
  config.schedule = 'constant'
  # Base learning rate.
  config.learning_rate = 0.001
  # Linear learning rate warmup.
  config.warmup_steps = 1000
  # Cross entropy loss label smoothing.
  config.label_smoothing = 0.0
  # Cross entropy auxilliary z-loss coefficient.
  config.z_loss = 0.0001
  # Starting step offset of fine-tuning phase for Adafactor.
  config.step_offset = 999_900

  # Maximum length cutoff for training examples.
  config.max_input_length = 512
  config.max_target_length = 84
  # Maximum length cutoff for eval examples.
  config.max_eval_input_length = 512
  config.max_eval_target_length = 84

  # Vocabulary size if `vocab_path` is not given.
  config.vocab_size = 32128
  # Inputs and targets share embedding.
  config.share_embeddings = True
  # Final logit transform uses embedding matrix transpose.
  config.logits_via_embedding = True
  # Number of transformer layers.
  config.num_layers = 12
  # Size of query/key/value for attention.
  config.qkv_dim = 768
  # Size of embeddings.
  config.emb_dim = 768
  # Size of the MLP.
  config.mlp_dim = 3072
  # Activations in MLP input.
  config.mlp_activations = ('relu',)
  # Number of attention heads.
  config.num_heads = 12
  # Number of relative-attention bins.
  config.relative_attention_num_buckets = 32
  # Number of relative-attention bins.
  config.relative_attention_max_distance = 128
  # Dropout rate.
  config.dropout_rate = 0.1
  # Attention dropout rate.
  config.attention_dropout_rate = 0.1
  # Use bfloat16 mixed precision training instead of float32.
  config.use_bfloat16 = True

  return config
