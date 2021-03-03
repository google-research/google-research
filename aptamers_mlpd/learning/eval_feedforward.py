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

# Lint as: python3
"""Loads a model from a checkpoint and runs inference.
"""

import collections
import functools
import itertools
import logging
import os


import numpy as np
import pandas
import scipy.stats
import sklearn.metrics
import tensorflow.compat.v1 as tf
from tensorflow.contrib import labeled_tensor as lt
import xarray


# Google internal
import text_format
import gfile


from ..util import selection
from ..util import selection_pb2
from ..learning import config
from ..learning import data
from ..learning import feedforward as ff
from ..learning import output_layers
from ..preprocess import sequencing_counts


logger = logging.getLogger(__name__)

ACTUAL_OUTPUTS = 'actual_outputs'
ACTUAL_COUNTS = 'actual_counts'
ACTUAL_BINDING = 'actual_binding'
PREDICTED_OUTPUTS = 'predicted_outputs'
PREDICTED_AFFINITY = 'predicted_affinity'
LOSS = 'loss'


class TrainingDivergedException(Exception):
  pass


def pearson_corr(y_true, y_score):
  corr, _ = scipy.stats.pearsonr(y_true, y_score)
  return corr


def spearman_corr(y_true, y_score):
  corr, _ = scipy.stats.spearmanr(y_true, y_score)
  return corr


def roc_auc_score(y_true, y_score):
  try:
    return sklearn.metrics.roc_auc_score(y_true, y_score)
  except ValueError:
    return np.nan


def discounted_cumulative_gain(y_true, y_score, top_n=None, top_frac=None):
  """Returns the Discounted Cumulative Gain of the examples.

  Args:
    y_true: The true 'relevance scores' of the examples.
    y_score: The estimated relevance of the examples, used to order results.
    top_n: If specified, an integer indicating the maximum number of examples
        to use in calculating DCG.
    top_frac: If specified, the fraction of input examples to use in calculating
        DCG.

  Returns:
    The Discounted Cumulative Gain of the examples.

  Raises:
    ValueError: Both top_n and top_frac are specified, or the specified values
        are invalid.
  """
  y_true = np.asarray(y_true)
  y_score = np.asarray(y_score)
  assert len(y_true) == len(y_score)
  if top_n is not None and top_frac is not None:
    raise ValueError('At most one of top_n and top_frac may be specified.')
  elif top_n is not None:
    top_n = min(top_n, len(y_true))
  elif top_frac is not None:
    top_n = min(int(round(top_frac * len(y_true))), len(y_true))
    if top_n <= 0:
      logging.warning(
          'top_frac %s produces no examples, using top single example',
          top_frac)
      top_n = 1
  else:
    top_n = len(y_true)
  true_indices = np.argsort(y_score)[::-1][:top_n]
  ordered_relevance = y_true[true_indices]
  weights = 1.0 / np.log2(1 + np.arange(1, top_n + 1))
  return np.dot(weights, ordered_relevance)


def normalized_dcg(y_true, y_score, top_n=None, top_frac=None):
  """Returns the normalized Discounted Cumulative Gain.

  Args:
    y_true: The true 'relevance scores' of the examples.
    y_score: The estimated relevance of the examples, used to order results.
    top_n: If specified, an integer indicating the maximum number of examples
        to use in calculating DCG.
    top_frac: If specified, the fraction of input examples to use in calculating
        DCG.

  Returns:
    The normalized Discounted Cumulative Gain of the examples.
  """
  dcg = discounted_cumulative_gain(
      y_true, y_score, top_n=top_n, top_frac=top_frac)
  idcg = discounted_cumulative_gain(
      y_true, y_true, top_n=top_n, top_frac=top_frac)
  return dcg / idcg


def scaled_dcg(y_true, y_score, top_n=None, top_frac=None):
  """Returns a scaled normalized discounted cumulative gain.

  The normalized discounted cumulative gain is guaranteed to lie in [0, 1].
  However, when we are using sequence counts as input, the lower bound is
  often substantially higher than 0, and depends on the size of the input.
  To aid interpretability, the scaled normalized discounted cumulative gain
  linearly scales the normalized DCG to its counterpart in [0, 1].

  Args:
    y_true: The true 'relevance scores' of the examples.
    y_score: The estimated relevance of the examples, used to order results.
    top_n: If specified, an integer indicating the maximum number of examples
        to use in calculating DCG.
    top_frac: If specified, the fraction of input examples to use in calculating
        DCG.

  Returns:
    The scaled normalized Discounted Cumulative Gain of the examples.
  """
  dcg = discounted_cumulative_gain(
      y_true, y_score, top_n=top_n, top_frac=top_frac)
  # Ideal is when the ordering is the same as the truth.
  idcg = discounted_cumulative_gain(
      y_true, y_true, top_n=top_n, top_frac=top_frac)
  # Worst-case is when the relevance is directly inverse to the ranking.
  sorted_ytrue = np.sort(y_true)
  wdcg = discounted_cumulative_gain(
      sorted_ytrue,
      np.arange(len(sorted_ytrue), 0, -1),
      top_n=top_n,
      top_frac=top_frac)
  return (dcg - wdcg) / (idcg - wdcg)


def enrichment(y_true, y_pred, minimum_size=10):
  """Calculate the enrichment score.

  Enrichment score is the factor above random that actives are found
  in predicted actives.

  Precisely, enrichment is:

  (true positives / total predicted positive)
  -------------------------------------------
  total positives / total_examples

  Args:
    y_true: 1-dimensional boolean numpy.ndarray with true actives.
    y_pred: 1-dimensional boolean numpy.ndarray with predicted actives.
    minimum_size: optional integer, indicating the minimum number of required
      predicted actives for computing enrichment.

  Returns:
    Enrichment factor as a float, or NaN if there are fewer than `minimum_size`
    predicted positives.
  """
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  pred_positives = np.count_nonzero(y_pred)
  true_positives = np.count_nonzero(y_true & y_pred)
  total_positives = np.count_nonzero(y_true)
  total_examples = len(y_true)
  if pred_positives >= minimum_size:
    return (
        (true_positives / pred_positives) / (total_positives / total_examples))
  else:
    return np.nan


# Note: as much as possible, these metrics only dependent on the ranks of the
# predicteds outputs, not the exact scores (which are not probably scaled).

REGRESSION_METRICS = collections.OrderedDict([
    ('spearman_correlation', spearman_corr),
    ('pearson_correlation', pearson_corr),
    ('scaled_dcg', scaled_dcg),
    ('scaled_dcg_top_1p', functools.partial(scaled_dcg, top_frac=0.01)),
])

REGRESSION_METRICS_VERBOSE = collections.OrderedDict([
    ('spearman_correlation', spearman_corr),
    ('pearson_correlation', pearson_corr),
    ('normalized_dcg', normalized_dcg),
    ('normalized_dcg_top_1p', functools.partial(normalized_dcg, top_frac=0.01)),
    ('scaled_dcg', scaled_dcg),
    ('scaled_dcg_top_1p', functools.partial(scaled_dcg, top_frac=0.01)),
])


def fixed_value_thresholder(threshold):
  """Return a function that indicates scores >= a fixed value."""

  def binarize(scores):
    return scores >= threshold

  return binarize


def top_rank_thresholder(rank):
  """Return a function that indicates top ranked scores."""

  def binarize(scores):
    kth = scores.size - rank
    return np.argpartition(scores, kth) >= kth

  return binarize


def top_fraction_thresholder(fraction):
  """Return a function that indicates top fraction of non-zero scores."""

  def binarize(scores):
    nonzero_scores = scores[scores != 0]
    if nonzero_scores.size:
      threshold = np.percentile(nonzero_scores, 100 * (1 - fraction))
      return scores > threshold
    else:
      return np.zeros(scores.shape, dtype=bool)

  return binarize


THRESHOLDERS = collections.OrderedDict([
    ('top_1p', top_fraction_thresholder(0.01)),
])

THRESHOLDERS_VERBOSE = collections.OrderedDict([
    ('nonzero', fixed_value_thresholder(1)),
    ('top_tenthp', top_fraction_thresholder(0.001)),
    ('top_1p', top_fraction_thresholder(0.01)),
    ('top_10p', top_fraction_thresholder(0.1)),
])


def compute_evaluation_metrics(dataset, verbose=False):
  """Compute all evaluations metrics.

  Args:
    dataset: xarray.Dataset with variables ACTUAL_OUTPUTS and PREDICTED_OUTPUTS.
    verbose: Boolean on whether to report all the evaluation metrics or only
      those used commonly.

  Returns:
    xarray.Dataset with dimensions ('score_threshold', 'true_threshold',
    'output') and data variables for each computed metric.
  """

  def compute_all_metrics(input_ds):  # pylint: disable=missing-docstring
    # NOTE(shoyer): xarray doesn't squeeze out unsorted dimensions in groupby,
    # even though it probably should.
    if 'output' in input_ds.dims:
      input_ds = input_ds.squeeze('output')
    valid_input = input_ds[[ACTUAL_OUTPUTS, PREDICTED_OUTPUTS]].dropna('batch')

    y_true = valid_input[ACTUAL_OUTPUTS].values
    y_score = valid_input[PREDICTED_OUTPUTS].values

    # dashboard has added throttling and will not display models with too
    # many metrics, so we limit the metrics reported by default.
    if verbose:
      thresholders_to_use = THRESHOLDERS_VERBOSE
      regression_to_use = REGRESSION_METRICS_VERBOSE
    else:
      thresholders_to_use = THRESHOLDERS
      regression_to_use = REGRESSION_METRICS

    # TODO(shoyer): compute significance scores for all these metrics
    ds = xarray.Dataset({
        'score_threshold': ['all'] + list(thresholders_to_use),
        'true_threshold': list(thresholders_to_use)
    })

    for metric_name, metric_func in regression_to_use.items():
      values = [metric_func(y_true, y_score)]
      for thresholder in thresholders_to_use.values():
        subset = thresholder(y_score)
        values.append(metric_func(y_true[subset], y_score[subset]))
      ds[metric_name] = ('score_threshold', values)

    auc_values = [
        roc_auc_score(thresholder(y_true), y_score)
        for thresholder in thresholders_to_use.values()
    ]
    ds['auc'] = ('true_threshold', auc_values)

    if verbose:
      enrich_values = []
      for true_thresholder in thresholders_to_use.values():
        values = [np.nan]  # for score_threshold="all"
        for score_thresholder in thresholders_to_use.values():
          values.append(
              enrichment(true_thresholder(y_true), score_thresholder(y_score)))
        enrich_values.append(values)
      ds['enrichment'] = (('true_threshold', 'score_threshold'), enrich_values)

    return ds

  return dataset.groupby('output').apply(compute_all_metrics)


def examples_from_sstable(input_filenames, batch_size, num_epochs=1):
  """Create an examples queue from the given sstables.

  Args:
    input_filenames: list of strings giving paths to sstables.
    batch_size: integer mini-batch size.
    num_epochs: optional integer epoch limit.

  Returns:
    tf.Tensor with dtype=string and shape=[None].
  """
  filename_queue = tf.train.string_input_producer(
      input_filenames, num_epochs=num_epochs)
  reader = tf.SSTableReader()
  _, examples = reader.read_up_to(filename_queue, batch_size)
  return examples


def fetch_across_batches(sess, tensors, max_size=None):
  """Evaluate a queue of tensors batch-wise into a single result.

  Starts queue runners and shuts them down when finished.

  Args:
    sess: tf.Session to use.
    tensors: list of Tensor-like objects (e.g., tf.Tensor or lt.LabeledTensor)
      to evaluate. The first axis of each tensor should correspond to batches,
      and be the same size on all results.
    max_size: integer total number of desired examples. `tensors` will be
      repeatedly evaluated until the total batch size exceeds this number, or
      the input queues are exhausted.

  Returns:
    List of numpy.ndarray objects with evaluated tensors from each batch
    concatenated together.

  Raises:
    ValueError: if any evaluated tensors do not have the same batch size.
  """
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  total_size = 0
  outputs_list = []
  try:
    while not coord.should_stop() and (max_size is None or total_size < max_size
                                      ):
      outputs = sess.run(tensors)
      batch_sizes = set(len(out) for out in outputs)
      if len(batch_sizes) != 1:
        raise ValueError('tensors do not have the same batch size: %r' %
                         batch_sizes)
      batch_size, = batch_sizes
      total_size += batch_size
      outputs_list.append(outputs)
  except tf.errors.OutOfRangeError:
    logger.info('out of range')
  finally:
    coord.request_stop()

  coord.join(threads)

  joined_outputs = [np.concatenate(out) for out in zip(*outputs_list)]
  if max_size is not None:
    joined_outputs = [out[:max_size] for out in joined_outputs]
  return joined_outputs


def _create_dataarray(numpy_array, labeled_tensor_axes):
  """Create an xarray.DataArray from a numpy array and labeled_tensor.Axes."""
  coords = {
      axis.name: list(axis.labels)
      for axis in labeled_tensor_axes.values() if axis.labels is not None
  }
  dims = list(labeled_tensor_axes.keys())
  return xarray.DataArray(numpy_array, coords=coords, dims=dims)


def convert_labeled_tensor_to_xarray(arrays, tensors):
  """Copy axis labels from LabeledTensors to NumPy arrays, using xarray.

  Args:
    arrays: np.ndarray, List[np.ndarray] or Dict[np.ndarray].
    tensors: container of the same type as `arrays` holding LabeledTensor
      objects. Each tensor provides matching labels for the corresponding
      numpy array.

  Returns:
    xarray.DataArray, List[xarray.DataArray] or xarray.Dataset, based on the
    type of the inputs.
  """
  # TODO(shoyer): move something like this into LabeledTensor?
  if isinstance(arrays, dict):
    return xarray.Dataset(
        {k: _create_dataarray(v, tensors[k].axes)
         for k, v in arrays.items()})
  elif isinstance(arrays, list):
    return [
        _create_dataarray(array, fetch.axes)
        for array, fetch in zip(arrays, tensors)
    ]
  else:
    return _create_dataarray(arrays, tensors.axes)


def hparams_from_checkpoint(checkpoint_path):
  """Load saved tf.HParams from a saved checkpoint.

  Args:
    checkpoint_path: string path to model checkpoint.

  Returns:
    tf.HParams saved on the graph specified by the checkpoint.
  """
  with tf.Graph().as_default():
    meta_path = checkpoint_path + '.meta'
    tf.train.import_meta_graph(meta_path)
    hps, = tf.get_collection('hparams')
    return hps


def create_inferer(model_dir, checkpoint_path, affinity_target_map=None):
  """Helper method to create an Inferer object from a model path.

  Args:
    model_dir: string name of the directory with the trained model.
    checkpoint_path: string name of the checkpoint file within model_dir.
    affinity_target_map: string name of affinity target map or None to not use
      one.

  Returns:
    Inferer object, ready to perform inference.

  Raises:
    ValueError: if the target name is not in the affinity map.
  """
  checkpoint_path = os.path.join(model_dir, checkpoint_path)
  with gfile.GFile(os.path.join(model_dir,
                                'wetlab_experiment_train.pbtxt')) as f:
    experiment_proto = text_format.Parse(f.read(), selection_pb2.Experiment())

  return Inferer(
      experiment_proto,
      checkpoint_path,
      affinity_target_map=affinity_target_map)


class Inferer:
  """Run inference on a pretrained FeedForward model."""

  def __init__(self,
               experiment_proto,
               checkpoint_path,
               hparams=None,
               affinity_target_map=None):
    """Initialize an evaluator.

    Args:
      experiment_proto: selection_pb2.Experiment describing the experiment.
      checkpoint_path: path to TensorFlow checkpoint.
      hparams: optional tf.HParams. By default, HParams are loaded from the
        checkpoint metagraph. Beware: the metagraph is written *after* new
        checkpoints appear, so you may not be able to load it immediately
        when a new checkpoint is available.
      affinity_target_map: String name of the affinity target map dictionary.
        The string must be a key in the config.DEFAULT_AFFINITY_TARGET_MAPS.
    """
    self.experiment_proto = experiment_proto
    self.checkpoint_path = checkpoint_path

    if hparams is None:
      hparams = hparams_from_checkpoint(checkpoint_path)

    if affinity_target_map is not None:
      hparams.affinity_target_map = affinity_target_map

    self.hps = hparams
    self.params = list(self.hps.values())

    self.all_count_names = selection.all_count_names(experiment_proto)
    self.binding_array_names = selection.binding_array_names(experiment_proto)

    self.global_step = None
    self._eval_setup_cache = None

  def _create_net_and_output_layer(self):
    """Create FeedForward net and output layer for this restored model.

    Returns:
      net: ff.FeedForward object.
      output_layer: output_layers.AbstractOutputLayer.
    """
    dummy_inputs = data.dummy_inputs(
        self.experiment_proto,
        input_features=self.hps.input_features,
        kmer_k_max=self.hps.kmer_k_max,
        additional_output=self.hps.additional_output.split(','))
    output_layer = output_layers.create_output_layer(
        self.experiment_proto, self.hps)
    net = ff.FeedForward(dummy_inputs, output_layer.logit_axis, self.hps)
    return net, output_layer

  def _create_global_step(self):
    return tf.Variable(0, name='global_step', trainable=False)

  def _split_outputs(self, outputs):
    """Split outputs into counts and binding array LabeledTensors."""
    counts = lt.select(outputs, {'output': self.all_count_names})
    binding = lt.select(outputs, {'output': self.binding_array_names})
    return counts, binding

  def _restore(self, sess):
    """Restore this evaluator's checkpoint into a tf.Session."""
    global_step = self._create_global_step()
    tf.Saver().restore(sess, self.checkpoint_path)

    if self.global_step is None:
      self.global_step = global_step.eval(sess)

    # Uncomment to sanity check and show that random weights don't make good
    # predictions
    # init = tf.initialize_all_variables()
    # sess.run(init)

  def evaluation_tensors(self, examples, keys=None):
    """Build output and evaluation tensors for a feed forward model.

    Args:
      examples: tf.Tensor with dtype=string and shape=[None] holding serialized
        tf.train.Example protos.
      keys: Optional sequence of string tensor names to evaluate. By default,
        uses all known tensors.

    Returns:
      Dict[str, lt.LabeledTensor] giving all possible tensors to run.
    """
    # TODO(shoyer): expose options for injecting/filtering examples?
    inputs, outputs = data.preprocess(
        lt.LabeledTensor(examples, ['batch']),
        self.experiment_proto,
        kmer_k_max=self.hps.kmer_k_max,
        input_features=self.hps.input_features,
        mode=data.PREPROCESS_ALL_COUNTS,
        additional_output=self.hps.additional_output.split(','))

    actual_counts, actual_binding = self._split_outputs(outputs)

    # TODO(shoyer): encapsulate net/output_layer in a single class that makes
    # predictions without building logits as intermediate output.
    # This will be useful for testing other methods.
    net, output_layer = self._create_net_and_output_layer()

    logits = net.fprop(inputs, mode='test')
    predicted_outputs = output_layer.predict_outputs(logits, outputs)

    # only calculate the predicted affinity if it is required.
    predicted_affinity = None
    if not keys or PREDICTED_AFFINITY in keys:
      predicted_affinity = output_layer.predict_affinity(logits)

    loss = output_layer.loss_per_example_and_target(logits,
                                                    outputs,
                                                    self.hps.train_on_array)

    tensors = {
        ACTUAL_OUTPUTS: outputs,
        ACTUAL_COUNTS: actual_counts,
        ACTUAL_BINDING: actual_binding,
        PREDICTED_OUTPUTS: predicted_outputs,
        PREDICTED_AFFINITY: predicted_affinity,
        LOSS: loss
    }
    tensors.update(inputs)
    # remove outputs not provided by the given model
    tensors = {k: v for k, v in tensors.items() if v is not None}
    if keys is not None:
      tensors = {k: tensors[k] for k in keys}
    return tensors

  def _get_evaluation_setup(self):
    """Cache evaluation tensors for run_on_example_protos.

    Returns:
      Dict[str, tf.Tensor] giving tensors to evaluate.
    """
    if self._eval_setup_cache is None:
      with tf.Graph().as_default():
        examples_placeholder = tf.placeholder(tf.string, shape=[None])
        tensors = self.evaluation_tensors(examples_placeholder)
        sess = tf.Session()
        self._restore(sess)
      self._eval_setup_cache = (sess, examples_placeholder, tensors)

    return self._eval_setup_cache

  def run_on_files(self, input_filenames, batch_size, max_size=None, keys=None):
    """Run evaluation on the given files.

    Args:
      input_filenames: list of strings providing paths to input sstables.
      batch_size: integer mini-batch size.
      max_size: optional maximum number of examples to evalute.
      keys: Optional sequence of string tensor names to evaluate. By default,
        uses all known tensors.

    Returns:
      xarray.Dataset with data variables for each of the given keys.
    """
    with tf.Graph().as_default():
      examples = examples_from_sstable(input_filenames, batch_size)
      tensors = self.evaluation_tensors(examples, keys=keys)

      sess = tf.Session()
      # initialize the num_epochs counter
      sess.run(tf.initialize_local_variables())
      self._restore(sess)

      # TODO(shoyer): after cl/133322369 is merged, switch to use
      # tf.contrib.metrics.streaming_concat instead of fetch_across_batches.
      arrays_list = fetch_across_batches(sess, list(tensors.values()), max_size)
      arrays_dict = dict(list(zip(list(tensors.keys()), arrays_list)))
      return convert_labeled_tensor_to_xarray(arrays_dict, tensors)

  def run_on_example_protos(self, examples, keys=None):
    """Runs inference on the input TF Example protos and returns results.

    Args:
      examples: List of string encoded tf.Example protos.
      keys: Optional sequence of string tensor names to evaluate. By default,
        uses all known tensors.

    Returns:
      xarray.Dataset with data variables for each of the given keys.
    """
    sess, examples_placeholder, all_tensors = self._get_evaluation_setup()  # pylint: disable=unpacking-non-sequence
    if keys is None:
      keys = list(all_tensors.keys())
    tensors = {k: all_tensors[k] for k in keys}
    feed_dict = {examples_placeholder: examples}
    arrays = sess.run(tensors, feed_dict=feed_dict)
    return convert_labeled_tensor_to_xarray(arrays, tensors)

  def run_on_sequences(self, sequences, keys=None):
    """Runs inference and returns scores without requiring protos.

    Args:
      sequences: A list of string sequences
      keys: Optional sequence of string tensor names to evaluate. By default,
        uses all known tensors.
    Returns:
      xarray.Dataset with data variables for each of the given keys.
    """
    converted_sequences = [str(seq) for seq in sequences]
    example_protos = [
        sequencing_counts.tensorflow_example({
            'sequence': seq
        }).SerializeToString() for seq in converted_sequences
    ]
    return self.run_on_example_protos(example_protos, keys)

  def get_affinities_for_sequences(self,
                                   sequences,
                                   target_molecule):
    """Convenience function to get affinities only from a list of sequences.

    Args:
      sequences: A list of string sequences
      target_molecule: String name of the target molecule. For example, in
        Aptitude this can be 'target' (i.e. NGAL) or 'serum'.
    Returns:
      numpy.ndarray with a predicted affinity score for each sequence.
    """
    key = PREDICTED_AFFINITY
    ds = self.run_on_sequences(sequences, keys=[key])
    return ds[PREDICTED_AFFINITY].sel(affinity=target_molecule).values


def config_pandas_display(interactive):
  pandas.set_option('display.width', None if interactive else 120)
  pandas.set_option('display.precision', 4)


def summarize_dataset(summary_writer, dataset, global_step):
  """Add each element of a pandas.DataFrame to a tf.SummaryWriter.

  Args:
    summary_writer: tf.SummaryWriter instance to use for writing summaries.
    dataset: xarray.Dataset to summarize.
    global_step: integer current global current value.
  """
  for name, array in dataset.data_vars.items():
    # create a list of 1D coordinate labels (as a Python list of scalars)
    coords_list = [array.coords[d].values.tolist() for d in array.dims]
    # for each combination of coordinate labels, extract values from `array`
    # and summarize
    for keys in itertools.product(*coords_list):
      ids = [name] + ['%s_%s' % (d, k) for d, k in zip(array.dims, keys)]
      tag = '/'.join(ids)
      value = array.loc[keys].item()
      s = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
      summary_writer.add_summary(s, global_step)


def save_netcdf_report(hparams, results_ds, events_dir):
  """Save evaluation results in a NetCDF file.

  Args:
    hparams: tf.HParams instance.
    results_ds: xarray.Dataset with results to save.
    events_dir: path to events directory.
  """
  report = hparams.values()
  merged_ds = results_ds.merge(xarray.Dataset(coords=report))
  merged_ds.attrs['eval_report_version'] = '0.3.0'

  report_path = os.path.join(events_dir, config.experiment_report_name)
  with gfile.GFile(report_path, 'w') as f:
    # xarray has a bug with writing to file-like objects directly:
    # https://github.com/pydata/xarray/issues/1320
    netcdf_bytes = merged_ds.to_netcdf()
    f.write(netcdf_bytes)


class ReportWriter:
  """Class for coordinating writing results to an event directory."""

  def __init__(self, hparams, events_dir):
    self.hparams = hparams
    self.events_dir = events_dir

    self.summary_writer = tf.SummaryWriter(
        self.events_dir, flush_secs=10, max_queue=1000)

  def add_results(self, results_ds, global_step):
    summarize_dataset(self.summary_writer, results_ds, global_step)
    save_netcdf_report(self.hparams, results_ds, self.events_dir)


def flatten_results_dataset(dataset, row_levels=frozenset({'output'})):
  """Given an xarray.Dataset with results, flatten it into a pandas.DataFrame.

  This function individually unstacks arrays, which results in a much more
  compact DataFrame than calling `dataset.to_dataframe()`.

  Args:
    dataset: xarray.Dataset, e.g., as produced by `Inferer.run_on_files`.
    row_levels: optional set giving names of dimensions to keep in the rows.

  Returns:
    pandas.DataFrame with concatenated data variables from `dataset` as columns.
    Dimensions not in `row_levels` are be combined with variable names into
    column names.
  """
  frames = []
  for array in dataset.data_vars.values():
    frame = array.to_dataframe()
    levels = [dim for dim in array.dims if dim not in row_levels]

    if levels:
      frame = frame.unstack(levels)

      # flatten columns, adding level names into the flattened column names
      new_keys = []
      for keys in frame.columns:
        pieces = [keys[0]]
        pieces.extend('%s_%s' % (lev, k) for lev, k in zip(levels, keys[1:]))
        new_keys.append('/'.join(pieces))
      frame.columns = new_keys

    frames.append(frame)

  return pandas.concat(frames, axis=1)


class Evaluator:
  """Class for handling repeated evaluation.
  """

  def __init__(self,
               hps,
               experiment_proto,
               input_filenames,
               events_dir,
               verbose=False):
    """Initialize an Evaluator.

    Args:
      hps: tf.HParams describing the experimental configuration.
      experiment_proto: selection_pb2.Experiment describing the experiment.
      input_filenames: list of strings giving paths to input sstables.
      events_dir: directory in which to save results.
      verbose: boolean for whether to print all the eval metrics vs. only
        the common ones.
    """
    self.hps = hps
    self.experiment_proto = experiment_proto
    self.input_filenames = input_filenames

    self.events_dir = events_dir
    self.report_writer = ReportWriter(hps, self.events_dir)
    self.batch_size = 8 * 1024

    self.verbose = verbose
    self.binding_array_names = selection.binding_array_names(experiment_proto)
    self.target_names = output_layers.get_target_names(experiment_proto)
    self.additional_output = self.hps.additional_output.split(',')

  def run(self, checkpoint_path, max_size=int(1e6)):
    """Run evaluation on a saved checkpoint.

    Args:
      checkpoint_path: Path to saved checkpoint.
      max_size: optional integer desired number of examples to evaluate.

    Returns:
      results_df: pandas.DataFrame with rows for each output name and columns
        for each metric value.
      global_step: integer global step from the checkpoint file.

    Raises:
      TrainingDivergedException: if all predicted outputs are NaN.
    """
    # pull out the trailing directory name, like TensorBoard
    events_name = self.events_dir.rsplit('/', 1)[-1]
    logger.info('Running evaluation for %r', events_name)

    logger.info('Running inference')
    inferer = Inferer(self.experiment_proto, checkpoint_path, self.hps)
    ds = inferer.run_on_files(
        self.input_filenames,
        self.batch_size,
        max_size,
        keys=[LOSS, ACTUAL_OUTPUTS, PREDICTED_OUTPUTS])

    logger.info('Computing evaluation DataFrame')
    ds = ds.dropna('output', how='all', subset=[PREDICTED_OUTPUTS])
    if not ds.sizes['output']:
      raise TrainingDivergedException('all predicted outputs were NaN')

    results_ds = compute_evaluation_metrics(ds, self.verbose)
    results_ds['count'] = ds[ACTUAL_OUTPUTS].notnull().sum('batch')

    # If affinity loss is calculated and we need it, exclude the count loss of
    # novel sequences in the microarray and the affinity loss of sequences not
    # included in the microarray from the average loss calculation.
    if bool(set(np.asarray(ds[LOSS].coords['target'])) &
            set(self.binding_array_names)) and self.hps.train_on_array:
      loss_count = ds[LOSS].sel(target=self.target_names)
      output_count = ds[ACTUAL_OUTPUTS].sel(output=self.target_names)
      loss_count_avg = loss_count[output_count.sum('output') != 0].mean('batch')

      loss_aff = ds[LOSS].sel(target=self.binding_array_names)
      output_aff = ds[ACTUAL_OUTPUTS].sel(output=self.binding_array_names)
      loss_aff_avg = loss_aff[output_aff.sum('output') != 0].mean('batch')

      all_loss_avg = [loss_count_avg, loss_aff_avg]

      if self.hps.additional_output:
        loss_ao = ds[LOSS].sel(target=self.additional_output)
        output_ao = ds[ACTUAL_OUTPUTS].sel(
            output=self.additional_output)
        loss_ao_avg = loss_ao[output_ao.sum('output') != 0].mean('batch')
        all_loss_avg += [loss_ao_avg]

      results_ds['loss'] = xarray.concat(
          all_loss_avg,
          dim='target').rename({'target': 'output'})
    else:
      results_ds['loss'] = ds[LOSS].mean('batch').rename({'target': 'output'})

    # Flatten into a DataFrame for the return value and displaying in the logs.
    # The "threshold" part of these dimension names becomes redundant once
    # flattened.
    results_df = flatten_results_dataset(
        results_ds.rename({
            'true_threshold': 'true',
            'score_threshold': 'score'
        }))
    results_df.loc['mean'] = results_df.mean()

    logger.info('Eval results for %s (%d examples) at global_step=%r:\n%r',
                events_name, results_ds['count'][0], inferer.global_step,
                results_df.T)
    logger.info('Saving summaries')
    self.report_writer.add_results(results_ds, inferer.global_step)

    return results_df, inferer.global_step

  def run_and_report(self,
                     tuner,
                     checkpoint_path,
                     max_size=int(1e6),
                     metrics_targets=None,
                     metrics_measures=None):
    """Run evaluation on a saved checkpoint and report to a tuner.

    Args:
      tuner: An HPTuner, e.g. Vizier.
      checkpoint_path: Path to saved checkpoint.
      max_size: optional integer desired number of examples to evaluate.
      metrics_targets: String list of network targets to report metrics for.
      metrics_measures: Measurements about the performance of the network to
        report, e.g. 'auc/top_1p'.

    Returns:
      results_df: pandas.DataFrame with rows for each output name and columns
        for each metric value.
      global_step: integer global step from the checkpoint file.
    """

    summary_df, global_step = self.run(checkpoint_path, max_size)
    if tuner:
      metrics_map = {}
      if metrics_targets and metrics_measures:
        for target in metrics_targets:
          for measure in metrics_measures:
            key = '%s_%s' % (target, measure)
            metrics_map[key] = summary_df.loc[target, measure]
      logger.info('Metrics map is: %r', metrics_map)

      logger.info('Reporting loss to tuner')
      objective = summary_df.loc[self.hps.tuner_target, self.hps.tuner_loss]
      tuner.report_measure(
          float(objective), global_step=int(global_step), metrics=metrics_map)
    return summary_df, global_step
