# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

r"""Run a CFQ experiment.

Example test invocation:
python3 -m cfq.run_experiment \
  --dataset=scan --split=mcd1 \
  --model=transformer --hparams_set=cfq_transformer \
  --train_steps=200 \
  --save_path=/tmp
"""
import os
import subprocess

from absl import app
from absl import flags

from cfq import evaluate as evaluator
from cfq import preprocess as preprocessor

import termcolor

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cfq', 'The dataset to use (cfq or scan).')

flags.DEFINE_string('split', 'mcd1',
                    'The split of the dataset (random, mcd1, mcd2, mcd3).')

flags.DEFINE_string(
    'model', 'lstm_seq2seq_attention',
    'Models from our paper: lstm_seq2seq_attention, transformer, '
    'universal_transformer.'
    'Other models (see subdirectories as well): https://github.com/tensorflow/'
    'tensor2tensor/tree/master/tensor2tensor/models.'
)

flags.DEFINE_integer('train_steps', 35000,
                     'We report experiments with 35,000 steps in our paper.')

flags.DEFINE_string(
    'hparams_set', 'cfq_lstm_attention_multi',
    'Custom hyperparameters are defined in cfq/cfq.py. '
    'You can select tensor2tensor default parameters as well.')

# Optional flags.
# We evaluate the trained model on the dev split of the dataset.
flags.DEFINE_string('questions_path', '${save_path}/dev/dev_encode.txt',
                    'Path to the input questions.')
flags.DEFINE_string('golden_answers_path', '${save_path}/dev/dev_decode.txt',
                    'Path to the expected (golden) answers.')
flags.DEFINE_string('inferred_answers_path',
                    '${save_path}/dev/dev_decode_inferred.txt',
                    'Path to the inferred answers.')

flags.DEFINE_string('eval_results_path', 'evaluation-${model}-${split}.txt',
                    'Path to write evaluation result to.')

# Tensor2tensor results will be written to this path. This includes encode/
# decode files, the vocabulary, and the trained models.
flags.DEFINE_string(
    'save_path',
    't2t_data/${dataset}/${split}/${model}',
    'Path to the directory where to save the files to.')

# The tensor2tensor problem to use. The cfq problem is defined in cfq/cfq.py.
T2T_PROBLEM = 'cfq'


def update_flag_value(flag_value):
  new_flag_value = flag_value
  new_flag_value = new_flag_value.replace('${dataset}', FLAGS.dataset)
  new_flag_value = new_flag_value.replace('${split}', FLAGS.split)
  new_flag_value = new_flag_value.replace('${model}', FLAGS.model)
  new_flag_value = new_flag_value.replace('${save_path}', FLAGS.save_path)
  if flag_value != new_flag_value:
    print('%s -> %s' % (flag_value, new_flag_value))
  return new_flag_value


def print_status(status):
  termcolor.cprint(status, 'yellow')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  FLAGS.save_path = update_flag_value(FLAGS.save_path)
  FLAGS.eval_results_path = update_flag_value(FLAGS.eval_results_path)
  FLAGS.questions_path = update_flag_value(FLAGS.questions_path)
  FLAGS.golden_answers_path = update_flag_value(FLAGS.golden_answers_path)
  FLAGS.inferred_answers_path = update_flag_value(FLAGS.inferred_answers_path)

  if os.path.exists(os.path.join(FLAGS.save_path, 'vocab.cfq.tokens')):
    print_status('Skipping preprocessing')
  else:
    print_status('Running preprocessing')
    dataset = preprocessor.get_dataset_from_tfds(FLAGS.dataset, FLAGS.split)
    preprocessor.write_dataset(dataset, FLAGS.save_path)
    token_vocab = preprocessor.get_token_vocab(FLAGS.save_path)
    preprocessor.write_token_vocab(token_vocab, FLAGS.save_path)

  t2t_usr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cfq')
  output_dir = os.path.join(FLAGS.save_path, 'output')

  print_status('Running t2t-datagen')
  # NOTE: This one skips automatically if the files exist.
  subprocess.run([
      't2t-datagen',
      '--t2t_usr_dir=' + t2t_usr_dir,
      '--data_dir=' + FLAGS.save_path,
      '--problem=' + T2T_PROBLEM,
      '--tmp_dir=/tmp/cfq_tmp',
  ],
                 check=True)

  print_status('Running t2t-trainer')
  subprocess.run([
      't2t-trainer',
      '--t2t_usr_dir=' + t2t_usr_dir,
      '--data_dir=' + FLAGS.save_path,
      '--problem=' + T2T_PROBLEM,
      '--model=' + FLAGS.model,
      '--hparams_set=' + FLAGS.hparams_set,
      '--output_dir=' + output_dir,
      '--train_steps=%s' % FLAGS.train_steps,
  ],
                 check=True)

  print_status('Running t2t-decoder')
  checkpoint_path = os.path.join(output_dir,
                                 'model.ckpt-%s' % FLAGS.train_steps)
  subprocess.run([
      't2t-decoder',
      '--t2t_usr_dir=' + t2t_usr_dir,
      '--data_dir=' + FLAGS.save_path,
      '--problem=' + T2T_PROBLEM,
      '--model=' + FLAGS.model,
      '--hparams_set=' + FLAGS.hparams_set,
      '--checkpoint_path=' + checkpoint_path,
      '--decode_from_file=' + FLAGS.questions_path,
      '--decode_to_file=' + FLAGS.inferred_answers_path,
      '--output_dir=' + output_dir,
  ],
                 check=True)

  print_status('Calculating accuracy')
  accuracy_result = evaluator.get_accuracy_result(FLAGS.questions_path,
                                                  FLAGS.golden_answers_path,
                                                  FLAGS.inferred_answers_path)
  evaluator.write_accuracy_result(
      accuracy_result, FLAGS.eval_results_path, print_output=True)


if __name__ == '__main__':
  app.run(main)
