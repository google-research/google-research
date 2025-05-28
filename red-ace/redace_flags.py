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

"""Defines common flags for training RED-ACE models."""

from absl import flags

flags.DEFINE_string('train_file', None,
                    'Path to the tfrecord file for training.')
flags.DEFINE_string('eval_file', None,
                    'Path to the tfrecord file for evaluation.')
flags.DEFINE_string(
    'init_checkpoint',
    None,
    ('Path to a pre-trained BERT checkpoint or a to previously trained model'
     ' checkpoint that the current training job will further fine-tune.'),
)
flags.DEFINE_string(
    'model_dir',
    None,
    'Directory where the model weights and summaries are stored.',
)
flags.DEFINE_integer(
    'max_seq_length',
    128,
    ('The maximum total input sequence length after tokenization. '
     'Sequences longer than this will be truncated, and sequences shorter '
     'than this will be padded.'),
)
flags.DEFINE_integer('num_train_examples', 32,
                     'Total size of training dataset.')
flags.DEFINE_integer('num_eval_examples', 32,
                     'Total size of evaluation dataset.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Total batch size for evaluation.')
flags.DEFINE_integer('num_train_epochs', 100,
                     'Total number of training epochs to perform.')
flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for Adam weight decay optimizer.')
flags.DEFINE_integer('log_steps', 1000,
                     'Interval of steps between logging of batch level stats.')
flags.DEFINE_integer('steps_per_loop', 1000, 'Steps per loop.')
flags.DEFINE_integer('keep_checkpoint_max', 3,
                     'How many checkpoints to keep around during training.')
flags.DEFINE_integer(
    'mini_epochs_per_epoch', 1,
    'Only has an effect for values >= 2. This flag enables more frequent '
    'checkpointing + evaluation on the validation set than done by default. '
    'This is achieved by reporting to TF an epoch size that is '
    '"flag value"-times smaller than the true value.')

flags.DEFINE_string('output_file', None, 'Path to the output file.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer(
    'predict_batch_size', 32,
    'Batch size for the prediction of insertion and tagging models.')
flags.DEFINE_bool(
    'split_on_punc',
    True,
    'Whether to split on punctuation characters during tokenization.',
)
flags.DEFINE_string('redace_config', None, 'Path to the RED-ACE config file.')
flags.DEFINE_string(
    'special_glue_string_for_joining_sources',
    ' ',
    ('String that is used to join multiple source strings of a given example'
     ' into one string. Optional.'),
)

# Prediction flags.
flags.DEFINE_string(
    'predict_input_file', None,
    'Path to the input file containing examples for which to'
    'compute predictions.')
flags.DEFINE_string('predict_output_file', None,
                    'Path to the output file for predictions.')

# Training flags.
flags.DEFINE_bool(
    'use_weighted_labels', True,
    'Whether different labels were given different weights. Primarly used to '
    'increase the importance of rare tags.')

flags.DEFINE_string('test_file', None, 'Path to the test file.')

flags.DEFINE_enum(
    'validation_checkpoint_metric',
    None,
    ['bleu', 'exact_match', 'latest', 'tag_accuracy'],
    ('Which metric should be used when choosing the best checkpoint. If'
     ' latest,then all checkpoints are saved.'),
)
