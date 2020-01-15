# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Code for serializing raw data into tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import numpy as np
import tensorflow.compat.v1 as tf

from bam.data import feature_spec
from bam.helpers import utils


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  batches could cause silent errors.
  """


class Preprocessor(object):
  """Class for loading, preprocessing, and serializing datasets."""

  def __init__(self, config, tasks):
    self._config = config
    self._tasks = tasks
    self._name_to_task = {task.name: task for task in tasks}

    self._feature_specs = feature_spec.get_shared_feature_specs(config)
    for task in tasks:
      self._feature_specs += task.get_feature_specs()
    self._name_to_feature_config = {
        spec.name: spec.get_parsing_spec()
        for spec in self._feature_specs
    }
    assert len(self._name_to_feature_config) == len(self._feature_specs)

  def prepare_train(self):
    return self._serialize_dataset(self._tasks, True, "train")

  def prepare_eval(self, task):
    return self._serialize_dataset([task], False, "dev")

  def prepare_predict(self, tasks, split):
    return self._serialize_dataset(tasks, False, split)

  def _serialize_dataset(self, tasks, is_training, split):
    """Writes out tfrecord examples for the specified tasks."""
    dataset_name = "_".join(sorted([task.name for task in tasks]))
    dataset_name += "_" + split
    if self._config.distill:
      dataset_name += "_distill"
    dataset_prefix = os.path.join(
        self._config.preprocessed_data_dir, dataset_name)
    tfrecords_path = dataset_prefix + ".tfrecord"
    metadata_path = dataset_prefix + ".metadata"
    batch_size = (self._config.train_batch_size if is_training else
                  self._config.eval_batch_size)

    utils.log("Loading dataset", dataset_name)
    n_examples = None
    sizes = {}
    if (self._config.use_tfrecords_if_existing and
        tf.gfile.Exists(metadata_path)):
      utils.log("Using already-written tfrecords")
      metadata = utils.load_json(metadata_path)
      n_examples = metadata["n_examples"]
      sizes = metadata["sizes"]

    if n_examples is None:
      utils.log("Existing tfrecords not found so creating")
      examples = []
      for task in tasks:
        task_examples = task.get_examples(split)
        sizes[task.name] = len(task_examples)
        examples += task_examples
      while len(examples) % batch_size != 0:
        examples.append(PaddingInputExample())
      if is_training:
        random.shuffle(examples)
      n_examples = len(examples)
      assert n_examples % batch_size == 0
      utils.mkdir(tfrecords_path.rsplit("/", 1)[0])
      self.serialize_examples(examples, is_training, tfrecords_path)
      utils.write_json({"n_examples": n_examples,
                        "sizes": sizes}, metadata_path)

    input_fn = self._input_fn_builder(tfrecords_path, is_training)
    if is_training:
      steps = int(n_examples // batch_size * self._config.num_train_epochs)
    else:
      steps = n_examples // batch_size

    return input_fn, steps, sizes

  def serialize_examples(self, examples, is_training, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    with tf.python_io.TFRecordWriter(output_file) as writer:
      for (ex_index, example) in enumerate(examples):
        if ex_index % 50000 == 0:
          utils.log("Writing example {:} of {:}".format(
              ex_index, len(examples)))
        tf_example = self._example_to_tf_example(example, is_training)
        writer.write(tf_example.SerializeToString())

  def _example_to_tf_example(self, example, is_training):
    if isinstance(example, PaddingInputExample):
      return self._make_tf_example(task_id=self._config.n_tasks)
    else:
      return self._make_tf_example(
          **self._name_to_task[example.task_name].featurize(
              example, is_training))

  def _make_tf_example(self, **kwargs):
    """Construct a tf.train.Example from the provided arguments."""
    for k in kwargs:
      if k not in self._name_to_feature_config:
        raise ValueError("Unknown feature", k)
    features = collections.OrderedDict()
    for spec in self._feature_specs:
      if spec.name in kwargs:
        values = kwargs[spec.name]
      else:
        values = spec.get_default_value()
      if (isinstance(values, int) or isinstance(values, bool) or
          isinstance(values, float) or isinstance(values, np.float32) or
          (isinstance(values, np.ndarray) and values.size == 1)):
        values = [values]
      if spec.is_int_feature:
        feature = tf.train.Feature(int64_list=tf.train.Int64List(
            value=list(values)))
      else:
        feature = tf.train.Feature(float_list=tf.train.FloatList(
            value=list(values)))
      features[spec.name] = feature
    return tf.train.Example(features=tf.train.Features(feature=features))

  def _input_fn_builder(self, input_file, is_training):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
      """The actual input function."""
      d = tf.data.TFRecordDataset(input_file)
      if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
      return d.apply(
          tf.data.experimental.map_and_batch(
              self._decode_tfrecord,
              batch_size=params["batch_size"],
              drop_remainder=True))

    return input_fn

  def _decode_tfrecord(self, record):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, self._name_to_feature_config)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name, tensor in example.items():
      if tensor.dtype == tf.int64:
        example[name] = tf.to_int32(tensor)
      else:
        example[name] = tensor
    return example
