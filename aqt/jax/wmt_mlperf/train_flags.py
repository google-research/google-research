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

"""Flags shared by train.py and xm_launch.py.

These flags are shared by the traiing executable (train.py)
and the script that launches XMananger experiments (xm_launch.py).
"""

from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    'data_dir', default=None, help='Tensorflow datasets directory.')

flags.DEFINE_integer(
    'stats_serialization_freq', 200,
    'The number of steps in between serializing activation statistics.')

flags.DEFINE_string(
    'vocab_path',
    default=None,
    help='Path to load or store sentencepiece vocab file.')

flags.DEFINE_string(
    'dataset_name',
    default='wmt17_translate/de-en',
    help='Name of TFDS translation dataset to use.')

flags.DEFINE_string(
    'eval_dataset_name',
    default='wmt14_translate/de-en:test',
    help='Optional name of TFDS translation dataset to use for evaluation.')

flags.DEFINE_bool(
    'run_train_eval',
    default=False,
    help='Whether to run eval on training data/ used for eval debuggability.')

# NOTE: if changing this to DEFINE_list, verify if the behavior is as expected.
flags.DEFINE_multi_string(
    'additional_eval_datasets',
    default=[
        'wmt15_translate/de-en:test', 'wmt16_translate/de-en:test',
        'wmt17_translate/de-en:test'
    ],
    help='Names for additional eval datasets.')

flags.DEFINE_bool(
    'reverse_translation',
    default=True,
    help='Reverse the direction of translation.')

flags.DEFINE_integer(
    'eval_batch_size',
    default=256,
    help='Per host eval batch size for training.')

flags.DEFINE_integer(
    'eval_frequency',
    default=1087,
    help='Frequency of eval during training, e.g. every 1000 steps.')

flags.DEFINE_integer(
    'num_eval_steps', default=20, help='Number of evaluation steps.')

flags.DEFINE_integer(
    'max_target_length',
    default=256,
    help='Maximum length of training examples.')

flags.DEFINE_integer(
    'max_eval_target_length',
    default=97,
    help='Maximum length of eval examples.')

flags.DEFINE_integer(
    'max_predict_length',
    default=147,
    help='Maximum length for predicted tokens.')

flags.DEFINE_bool(
    'save_checkpoints',
    default=True,
    help='Whether to save model checkpoints for debugging.')

flags.DEFINE_bool(
    'save_minimum_loss_checkpoint',
    default=True,
    help='Whether to save a model checkpoint when eval loss is miniminal.')

flags.DEFINE_bool(
    'estimate_compute_and_memory_cost',
    default=False,
    help='Whether to estimate compute and memory cost.')

flags.DEFINE_bool(
    'restore_checkpoints',
    default=True,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_string(
    'restore_checkpoint_model_dir',
    default=None,
    help='Directory to load checkpoint from when training first begins. Note '
    'that if a checkpoint already exists in "model_dir", that checkpoint is '
    'loaded instead under the assumption that this indicates the training run '
    'was preempted on Borg and is now being restarted.')

flags.DEFINE_integer(
    'checkpoint_freq',
    default=10000,
    help='Number of training steps between checkpoints.')

flags.DEFINE_bool(
    'use_bfloat16',
    default=True,
    help=('Use bfloat16 mixed precision training instead of float32.'))

flags.DEFINE_bool(
    'compute_train_metrics',
    default=True,
    help='Whether to compute the metrics during training.')

flags.DEFINE_string(
    'output_hlo_filename',
    default='hlo',
    help='Output model HLO to filename in FLAGS.model_dir. Both HLO text '
    '(filename.txt) and protobuf (filename.pb) are emitted. If None, or empty '
    'string, no HLO files will be written.')

flags.DEFINE_string(
    'output_beam_hlo_filename',
    default=None,
    help='Output model HLO, including the decoding beam search, to filename in'
    'FLAGS.model_dir. Both HLO text (filename.txt) and protobuf (filename.pb)'
    ' are emitted.')

flags.DEFINE_bool(
    'visualize_acts_bound',
    default=False,
    help=(
        'Whether to visualize activations bounds for auto-clip in Tensorboard.'
        ' The bounds appear as scalar and will be named as "GetBounds_0/bounds"'
        ' prefixed with the all the parents module name.'))

flags.DEFINE_bool(
    'collect_acts_stats',
    default=False,
    help=(
        'Whether to collect activation statistics and visualize them '
        'distributions in Tensorboard.'))

flags.DEFINE_integer(
    'state_dict_summary_freq',
    default=200,
    help=(
        'Number of training steps between state dict summaries reported to '
        'Tensorboard. Relevant to --visualize_acts_bound and --collect_acts_stats.'
    ))

