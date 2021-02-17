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

"""Defines common flags for training Felix models."""

from absl import flags

flags.DEFINE_string('train_file', None,
                    'File path to retrieve training data for pre-training.')
flags.DEFINE_string('eval_file', None,
                    'File path to retrieve training data for pre-training.')
flags.DEFINE_string(
    'init_checkpoint', None,
    'Path to a pre-trained BERT checkpoint or a to previously trained model '
    'checkpoint that the current training job will further fine-tune. In the '
    'latter case the name is expected to contain "felix_model.ckpt".')
flags.DEFINE_string(
    'model_dir_insertion', None,
    'The directory where the insertion model weights and training/evaluation '
    'summaries are stored. If not specified, save to /tmp/bert20/.')
flags.DEFINE_string(
    'model_dir_tagging', None,
    'The directory where the tagging model weights and training/evaluation  '
    'summaries are stored. If not specified, save to /tmp/bert20/.')
flags.DEFINE_string('tpu', None, 'TPU address to connect to.')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use (sgd|adam)')

# Model training specific flags.
flags.DEFINE_bool(
    'train_insertion', True,
    'If true the insertion model is trained and not the tagging model.')
flags.DEFINE_bool('use_pointing', True, 'If true a pointing mechanism be used.'
                  'Only True is currently supported.')
flags.DEFINE_float('pointing_weight', 1.0,
                   'How much should the pointing loss be weighed.')
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('num_train_examples', 32,
                     'Total size of training dataset.')
flags.DEFINE_integer('num_eval_examples', 32,
                     'Total size of evaluation dataset.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Total batch size for evaluation.')
flags.DEFINE_integer('num_train_epochs', 3,
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

flags.DEFINE_enum('input_format', None, ['discofuse', 'wikisplit'],
                  'File format for input.')

flags.DEFINE_string(
    'output_file', None,
    'Path to the output file.')

flags.DEFINE_string(
    'label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', None,
                    'Path to the BERT vocabulary file.')

flags.DEFINE_integer(
    'predict_batch_size', 32,
    'Batch size for the prediction of insertion and tagging models.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool(
    'split_on_punc', False,
    'Whether to split on punctuation characters during tokenization. Only False'
    'is currently supported.')
flags.DEFINE_bool(
    'use_open_vocab', True,
    'Is an insertion model used (Felix/FelixInsert). Only True is currently'
    'supported.')
flags.DEFINE_string('bert_config_tagging', None,
                    'Path to the config file for the tagging model.')
flags.DEFINE_string('bert_config_insertion', None,
                    'Path to the config file for the insertion model.')
flags.DEFINE_string('model_tagging_filepath', None,
                    'Path to the tagging TF model.')
flags.DEFINE_string('model_insertion_filepath', None,
                    'Path to the insertion TF model.')
flags.DEFINE_string('tokenizer', None, 'Not currently supported.')
flags.DEFINE_bool(
    'insert_after_token', True,
    'Whether to insert tokens after rather than before the current token. '
    'Currently, only supported with FelixInsert.')
flags.DEFINE_integer(
    'max_mask', 10,
    'The maximum number of MASKs the model can create per input token when '
    '`use_open_vocab == True`.')
flags.DEFINE_string(
    'special_glue_string_for_joining_sources', ' ',
    'String that is used to join multiple source strings of a given example '
    'into one string. Optional.')
flags.DEFINE_integer(
    'num_output_variants', 0,
    'Number of output variants to be considered. By default, the value is set '
    'to 0 and thus, no variants are considered. Warning! This feature only '
    'makes sense if num_output_variants >= 2.')


# Prediction flags.
flags.DEFINE_string('predict_input_file', None,
                    'Path to the input file containing examples for which to'
                    'compute predictions.')
flags.DEFINE_string('predict_output_file', None,
                    'Path to the output file for predictions.')

# Training flags.
flags.DEFINE_bool(
    'use_weighted_labels', True,
    'Whether different labels were given different weights. Primarly used to '
    'increase the importance of rare tags. Only True is currently supported.')
