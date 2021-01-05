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

"""Code for configuring a training run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


class Config(object):
  """Hyperparameters for the model."""

  def __init__(self, model_name, topdir, **kwargs):
    # general
    self.model_name = model_name
    self.debug = False  # debug mode for quickly running things locally
    self.num_trials = 1  # how many train+eval runs to perform

    # writing model outputs to disk
    self.write_test_outputs = True  # whether to write test set outputs
    self.write_distill_outputs = True  # whether to write distillation targets
    self.n_writes_test = 1  # write test set predictions for the first n trials
    self.n_writes_distill = 1  # write distillation targets for the first n
                               # trials

    # model
    self.task_names = ['rte', 'mrpc']  # which tasks to learn
    self.pretrained = True  # whether to use pre-trained weights
    self.pretrained_model_name = 'uncased_L-12_H-768_A-12'
    self.do_lower_case = True
    self.learning_rate = 1e-4
    self.weight_decay_rate = 0.01
    self.lr_decay = 0.9  # if > 0, the learning rate for a particular layer is
                         # learning_rate * lr_decay^(depth - max_depth)
                         # i.e., shallower layers have lower learning rates
    self.num_train_epochs = 6.0
    self.warmup_proportion = 0.1  # how much of training to warm up the LR

    # knowledge distillation;
    self.distill = False  # whether to do knowledge distillation
    self.teachers = {}  # {task: the model providing distill targets}
    self.teacher_annealing = True  # whether to do teacher annealing
    self.distill_weight = 0.5  # if no teacher annealing, how much weight to
                               # put on distill targets vs gold-standard label

    # sizing
    self.max_seq_length = 128
    self.train_batch_size = 128
    self.eval_batch_size = 8
    self.predict_batch_size = 8
    self.double_unordered = True  # for tasks like paraphrase where sentence
                                  # order doesn't matter, train the model on
                                  # on both sentence orderings for each example
    self.task_weight_exponent = 0.75  # exponent for up-weighting small datasets
    self.dataset_multiples = True  # include multiple copies of small datasets

    # training
    self.save_checkpoints_steps = 2000
    self.iterations_per_loop = 100
    self.use_tfrecords_if_existing = True  # don't make tfrecords and write them
                                           # to disk if existing ones are found

    # TPU settings
    self.use_tpu = False
    self.num_tpu_cores = 8
    self.tpu_name = None  # cloud TPU to use for training
    self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
    self.gcp_project = None  # project name for the Cloud TPU-enabled project

    # update with passed-in arguments
    self.update(kwargs)

    # default hyperparameters for single-task models
    self.n_tasks = len(self.task_names)
    if self.n_tasks == 1:
      self.dataset_multiples = False
      self.train_batch_size = 32
      self.num_train_epochs = 3.0
      self.learning_rate = 5e-5

    # debug-mode settings
    if self.debug:
      self.save_checkpoints_steps = 1000000
      self.use_tfrecords_if_existing = False
      self.iterations_per_loop = 1
      self.train_batch_size = 32
      self.num_train_epochs = 3.0
      self.dataset_multiples = False

    # passed-in arguments override the default single-task/debug-mode hparams
    self.update(kwargs)

    # where the raw GLUE data is
    self.raw_data_dir = os.path.join(topdir, 'glue_data/{:}').format

    # where BERT files are
    bert_dir = os.path.join(topdir, 'pretrained_models',
                            self.pretrained_model_name)
    self.bert_config_file = os.path.join(bert_dir, 'bert_config.json')
    self.vocab_file = os.path.join(bert_dir, 'vocab.txt')
    self.init_checkpoint = os.path.join(bert_dir, 'bert_model.ckpt')

    # where to save model checkpoints, results, etc.
    model_dir = os.path.join(topdir, 'models', self.model_name)
    self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    self.results_txt = os.path.join(model_dir, 'results.txt')
    self.results_pkl = os.path.join(model_dir, 'results.pkl')
    self.preprocessed_data_dir = os.path.join(model_dir, 'tfrecords')
    self.test_outputs = os.path.join(
        model_dir, 'outputs', '{:}_{:}_predictions_{:}.pkl').format
    self.distill_outputs = os.path.join(
        model_dir, 'outputs', '{:}_train_predictions_{:}.pkl').format

    # if doing distillation, where to load teacher targets from
    if self.teachers:
      def get_distill_inputs(task):
        return os.path.join(topdir, 'models', '{:}', 'outputs',
                            '{:}_train_predictions_1.pkl').format(
                                self.teachers[task], task)

      self.distill_inputs = get_distill_inputs
    else:
      self.distill_inputs = None

  def update(self, kwargs):
    for k, v in kwargs.items():
      if k not in self.__dict__:
        raise ValueError('Unknown argument', k)
      self.__dict__[k] = v
