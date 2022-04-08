# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Preprocess input and output data for TensorFlow models.
"""

import itertools


import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.contrib import labeled_tensor as lt

# Google internal
import resample

from ..util import dna
from ..util import selection
from ..learning import custom_ops


# names of input_features
SEQUENCE_ONE_HOT = 'SEQUENCE_ONE_HOT'
SEQUENCE_KMER_COUNT = 'SEQUENCE_KMER_COUNT'
STRUCTURE_PARTITION_FUNCTION = 'STRUCTURE_PARTITION_FUNCTION'


# training modes
PREPROCESS_ALL_COUNTS = 'PREPROCESS_ALL_COUNTS'
PREPROCESS_SKIP_ALL_ZERO_COUNTS = 'PREPROCESS_SKIP_ALL_ZERO_COUNTS'
PREPROCESS_INJECT_RANDOM_SEQUENCES = 'PREPROCESS_INJECT_RANDOM_SEQUENCES'


def build_features(experiment_proto):
  """Build FixedLenFeature objects from an Experiment proto.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    Dict[str, lt.FixedLenFeature] describing the features in the experiment.
  """
  has_partition_function = experiment_proto.has_partition_function
  count_names = selection.all_count_names(experiment_proto)
  embedding_dim = experiment_proto.embedding_dim
  embedding_prefix = experiment_proto.embedding_columns_prefix

  features = {'sequence': lt.FixedLenFeature([], tf.string)}
  for count_name in count_names:
    features[count_name] = lt.FixedLenFeature([], tf.int64, 0)

  if has_partition_function:
    features['partition_function'] = lt.FixedLenFeature(
        [], tf.float32, float('nan'))

  return features


def _kmer_labels(k_max):
  kmers = []
  for k in range(1, k_max + 1):
    kmers.extend(''.join(s) for s in itertools.product(dna.DNA_BASES, repeat=k))
  return kmers


def _kmer_mean_and_std(kmer_size, sequence_length):
  # assume a binomial distribution
  n = sequence_length + 1 - kmer_size
  p = 1.0 / 4 ** kmer_size
  return n * p, np.sqrt(n * p * (1 - p))


def _all_kmer_mean_and_std(k_max, sequence_length):
  means = []
  stds = []
  for k in range(1, k_max + 1):
    mean, std = _kmer_mean_and_std(k, sequence_length)
    n_kmers = 4 ** k
    means.extend([mean] * n_kmers)
    stds.extend([std] * n_kmers)
  return np.array(means), np.array(stds)


def create_input_and_outputs(feature_tensors, experiment_proto,
                             input_features=(SEQUENCE_ONE_HOT,),
                             skip_all_zero_counts=True,
                             kmer_k_max=4,
                             additional_output=None):
  """Create inputs and outputs from parsed features.

  Args:
    feature_tensors: Dict[str, tf.Tensor] with parsed featured created by
      `build_features`.
    experiment_proto: selection_pb2.Experiment describing the experiment.
    input_features: optional sequence of feature constants defined in this
      module.
    skip_all_zero_counts: some sequences have no counts, e.g., because they were
      created artificially for validation purposes on the binding array. We want
      to skip these sequences for training.
    kmer_k_max: optional integer giving the maximum kmer length to use if
      SEQUENCE_KMER_COUNT is in `input_features`.
    additional_output: optional list of strings contains additional outputs.

  Returns:
    inputs: LabeledTensor with dtype=float32 and axes
      [batch_axis, input_position_axis, input_channel_axis], of one-hot-encoded
      rasterized sequences for input into machine learning models.
    outputs: LabeledTensor with dtype=float32 and axes [batch_axis, output_axis]
      denoting possible output tensors, including counts and binding array
      measurements.
  """

  sequence_length = experiment_proto.sequence_length
  count_names = selection.all_count_names(experiment_proto)
  array_names = selection.binding_array_names(experiment_proto)

  sequence_tensor = feature_tensors['sequence']
  batch_axis = sequence_tensor.axes['batch']
  position_axis = ('position', list(range(sequence_length)))

  inputs = {}

  if SEQUENCE_ONE_HOT in input_features:
    seq_indices = custom_ops.dna_sequence_to_indices(
        sequence_tensor, sequence_length)
    tensor = tf.one_hot(seq_indices, depth=4, dtype=tf.float32)
    channel_axis = ('channel', list(dna.DNA_BASES))
    axes = [batch_axis, position_axis, channel_axis]
    one_hots = lt.LabeledTensor(tensor, axes)
    inputs[SEQUENCE_ONE_HOT] = one_hots

  if SEQUENCE_KMER_COUNT in input_features:
    raw_counts = custom_ops.count_all_dna_kmers(sequence_tensor, kmer_k_max)
    kmer_axis = lt.Axis('kmer', _kmer_labels(kmer_k_max))
    counts = lt.LabeledTensor(raw_counts, [batch_axis, kmer_axis])
    means, stds = _all_kmer_mean_and_std(kmer_k_max, sequence_length)
    mean_count = lt.constant(means, tf.float32, axes=[kmer_axis])
    std_count = lt.constant(stds, tf.float32, axes=[kmer_axis])
    inputs[SEQUENCE_KMER_COUNT] = ((lt.cast(counts, tf.float32) - mean_count)
                                   / std_count)

  if STRUCTURE_PARTITION_FUNCTION in input_features:
    with tf.name_scope('structure_partition_fn'):
      raw_pf_tensor = lt.expand_dims(
          feature_tensors['partition_function'], ['batch', 'partition_fn_axis'])
      inputs[STRUCTURE_PARTITION_FUNCTION] = lt.log(raw_pf_tensor)

  output_names = count_names + array_names
  outputs = [lt.cast(feature_tensors[k], tf.float32) for k in output_names]

  if additional_output and additional_output[0]:
    outputs += [lt.cast(feature_tensors[k], tf.float32)
                for k in additional_output]
    output_names += additional_output
  outputs = lt.pack(outputs, ('output', output_names), axis_position=1)

  if skip_all_zero_counts:
    with tf.name_scope('counts_filtering'):
      counts = lt.select(outputs, {'output': count_names})
      keep = lt.reduce_any(lt.not_equal(counts, 0.0), 'output')
      inputs = {k: lt.boolean_mask(v, keep) for k, v in inputs.items()}
      outputs = lt.boolean_mask(outputs, keep)

  return inputs, outputs


def preprocess(strs,
               experiment_proto,
               input_features=(SEQUENCE_ONE_HOT,),
               mode=PREPROCESS_SKIP_ALL_ZERO_COUNTS,
               kmer_k_max=4,
               ratio_random_dna=1,
               total_reads_defining_positive=0,
               additional_output=None):
  """Build a small TF graph to preprocess a minibatch of tf.Example protos.

  Args:
    strs: LabeledTensor holding a minibatch of serialized tf.Example protos
    experiment_proto: selection_pb2.Experiment describing the experiment.
    input_features: optional sequence of feature constants defined in this
      module.
    mode: optional preprocess mode defined in this module.
    kmer_k_max: optional integer giving the maximum kmer length to use if
      SEQUENCE_KMER_COUNT is in `input_features`.
    ratio_random_dna: optional ratio of random sequences to inject if mode ==
      PREPROCESS_INJECT_RANDOM_SEQUENCES
    total_reads_defining_positive: optional integer indicating the sum of all
      read counts required to be seen to classify the tensor as a "positive"
      example when balancing input classes.
    additional_output: optional list of strings contains additional outputs.

  Returns:
    inputs: LabeledTensor with dtype=float32 and axes
      [batch_axis, input_position_axis, input_channel_axis], of one-hot-encoded
      rasterized sequences for input into machine learning models.
    outputs: LabeledTensor with dtype=float32 and axes [batch_axis, output_axis]
      denoting possible output tensors, including counts and binding array
      measurements.
  """
  with tf.name_scope('preprocess'):
    features = build_features(experiment_proto)
    parsed_feature_tensors = lt.parse_example(strs, features)
    count_names = selection.all_count_names(experiment_proto)

    if mode == PREPROCESS_SKIP_ALL_ZERO_COUNTS:
      skip_all_zero_counts = True
      feature_tensors = parsed_feature_tensors

    elif mode == PREPROCESS_ALL_COUNTS:
      skip_all_zero_counts = False
      feature_tensors = parsed_feature_tensors

    elif mode == PREPROCESS_INJECT_RANDOM_SEQUENCES:
      skip_all_zero_counts = False

      # replace zero counts with NaN in real data
      for count_name in count_names:
        count = parsed_feature_tensors[count_name]
        parsed_feature_tensors[count_name] = lt.LabeledTensor(
            tf.where(count != 0, tf.cast(count, tf.float32),
                     tf.fill(tf.shape(count), np.float32(np.nan))),
            count.axes)

      # only random sequences will have a count of zero
      input_batch_size = tf.shape(strs.tensor)[list(
          strs.axes.keys()).index('batch')]
      n_randoms = tf.cast(tf.cast(input_batch_size, tf.float32)
                          * ratio_random_dna, tf.int32)
      random_feature_tensors = random_dna_features(
          experiment_proto, n_randoms)
      for count_name in count_names:
        random_feature_tensors[count_name] = lt.cast(
            random_feature_tensors[count_name], tf.float32)

      feature_tensors = {k: lt.concat([random_feature_tensors[k],
                                       parsed_feature_tensors[k]], 'batch')
                         for k in features}

      # shuffle random and non-random inputs because preprocess batches get
      # split across many mini-batches for training
      batch_size = tf.shape(feature_tensors['sequence'].tensor)[0]
      order = tf.random_shuffle(tf.range(batch_size, dtype=tf.int32))
      order.set_shape(feature_tensors['sequence'].tensor.get_shape())
      feature_tensors = {
          k: lt.LabeledTensor(tf.gather(v.tensor, order), v.axes)
          for k, v in feature_tensors.items()}

    else:
      raise ValueError('unknown mode: %r' % mode)  # pylint: disable=g-doc-exception

    feature_tensors = upsample_positives(
        feature_tensors,
        count_names,
        total_reads_defining_positive=total_reads_defining_positive,
        min_fraction_positive=0.1)

    inputs, outputs = create_input_and_outputs(
        feature_tensors, experiment_proto,
        input_features=input_features,
        kmer_k_max=kmer_k_max,
        skip_all_zero_counts=skip_all_zero_counts,
        additional_output=additional_output)

    return inputs, outputs


def upsample_positives(feature_tensors,
                       count_names,
                       total_reads_defining_positive,
                       min_fraction_positive,
                       seed=None):
  """Returns feature tensors with positives upsampled to the desired rate.

  Args:
    feature_tensors: Dict[str, lt.LabeledTensor] with parsed featured created by
      `build_features`.
    count_names: A list of labels that are count names.
    total_reads_defining_positive: The minimum number of reads detected across
      all conditions that defines a sequence as being a positive example.
    min_fraction_positive: The minimum fraction of positive examples to allow
      in the data.
    seed: The random seed to use in upsampling.

  Returns:
    A dictionary mapping from string feature name to lt.LabeledTensor of parsed
    features created by `build_features` and positive examples upsampled to the
    desired rate.

  Raises:
    ValueError: The minimum positive fraction requested is invalid.
  """
  # Goal: Find the fraction of all input feature tensors that should be
  # classified as "positive" based on the total_reads_defining_positive.
  # Upsample those using resample.resample_at_rate() until they are at least
  # min_fraction_positive of the entire set.
  if min_fraction_positive < 0 or min_fraction_positive >= 1:
    raise ValueError('Invalid fraction positive, must be in [0, 1): %s' %
                     min_fraction_positive)

  with tf.name_scope('upsample_positives'):
    # Classify the inputs as positive or negative.
    total_reads_defining_positive = tf.constant(
        total_reads_defining_positive, dtype=tf.float32)
    min_fraction_positive = tf.constant(min_fraction_positive, dtype=tf.float32)
    counts = lt.pack(
        [lt.cast(feature_tensors[k], tf.float32)
         for k in count_names], ('sequence_counts', count_names),
        axis_position=1)
    greater_equal = (lt.reduce_sum(counts, 'sequence_counts') >=
                     total_reads_defining_positive)
    num_pos = lt.reduce_sum(lt.cast(greater_equal, tf.int32))
    less_than = lt.logical_not(greater_equal)
    num_neg = lt.reduce_sum(lt.cast(less_than, tf.int32))

    # With an initial number of positives P and number of negatives N,
    # if we keep the negative sampling rate at 1 (to try to retain negatives),
    # to achieve a total positive input fraction of F, we need a positive
    # sampling rate R that satisfies:
    # P * R / (P * R + N) >= F
    #
    # Solving for R:
    #
    # P * R = F * (P*R + N) = F*P*R + F*N
    # P * R (1 - F) = F * N
    # R = F*N / (P * (1 - F))
    numerator = min_fraction_positive * tf.cast(num_neg, tf.float32)
    denom = tf.cast(num_pos, tf.float32) * (1 - min_fraction_positive)
    denom = tf.cond(
        denom > 0.0,
        lambda: denom,
        # If denom == 0, we can set it to anything we want since the
        # tf.cond below is guaranteed to return the input without
        # resampling.
        lambda: tf.constant(1.0, dtype=tf.float32))
    positive_rate = numerator / denom
    batch_size = tf.shape(greater_equal)[0]
    negative_rates = tf.ones([batch_size], tf.float32)
    positive_rates = tf.fill([batch_size], positive_rate)
    rates = tf.where(greater_equal, positive_rates, negative_rates)

    # Pack the LabeledTensors into normal tensors, keeping relevant information
    # for unpacking back to LabeledTensors available.
    ordered_names = sorted(feature_tensors)
    packed_tensors = []
    tensor_axes = []
    tensor_shapes = []
    for name in ordered_names:
      labeled_tensor = feature_tensors[name]
      packed_tensors.append(labeled_tensor.tensor)
      tensor_axes.append(labeled_tensor.axes)
      tensor_shapes.append(labeled_tensor.get_shape())

    # Perform the resampling.
    resampled_tensors = tf.cond(
        tf.logical_or(tf.equal(num_pos, 0),
                      tf.cast(num_pos, dtype=tf.float32) >=
                      (min_fraction_positive *
                       tf.cast(batch_size, dtype=tf.float32))),
        lambda: packed_tensors,
        lambda: resample.resample_at_rate(packed_tensors, rates, seed=seed))

    # Unpack the tensors into a dictionary of LabeledTensors again.
    # First, change the shape so that the batch axis is unknown.
    tensor_shapes = [[None] + list(shape)[1:] for shape in tensor_shapes]
    for tensor, shape in zip(resampled_tensors, tensor_shapes):
      tensor.set_shape(shape)

    unpacked_feature_tensors = {}
    for i, name in enumerate(ordered_names):
      labeled = lt.LabeledTensor(resampled_tensors[i], tensor_axes[i])
      unpacked_feature_tensors[name] = labeled
    return unpacked_feature_tensors


def dummy_inputs(experiment_proto,
                 input_features=(SEQUENCE_ONE_HOT,),
                 kmer_k_max=4,
                 batch_axis='batch',
                 additional_output=None):
  """Given an experiment proto, construct axes for input data.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    input_features: optional sequence of feature constants defined in this
      module.
    kmer_k_max: optional integer giving the maximum kmer length to use if
      SEQUENCE_KMER_COUNT is in `input_features`.
    batch_axis: optional string name or labeled_tensor.Axis object describing
      the batch axis on the result.
    additional_output: optional list of strings contains additional outputs.

  Returns:
    List[labeled_tensor.Axis] for labeling input.
  """
  example_dummy = lt.placeholder(tf.string, [batch_axis])
  dummy_inputs, _ = preprocess(example_dummy, experiment_proto,
                               input_features=input_features,
                               kmer_k_max=kmer_k_max,
                               additional_output=additional_output)
  return dummy_inputs


def random_dna_sequence(template, shape=()):
  """Generate a random DNA sequence according to a template.

  Args:
    template: string of characters representing a template for making DNA
      sequences. Each occurrence of an 'N' is replaced by a randomly chosen
      base.
    shape: optional tuple indicating the desired shape of the returned Tensor.

  Returns:
    tf.Tensor with dtype=strings.
  """
  n_random = sum(base == 'N' for base in template)
  all_bases = tf.constant([x.encode('utf8') for x in dna.DNA_BASES])
  random_index = tf.random_uniform((n_random,) + shape, maxval=3,
                                   dtype=tf.int32)
  random_bases = tf.gather(all_bases, random_index)
  iter_bases = iter(random_bases[i] for i in range(n_random))
  bases = [next(iter_bases) if base == 'N' else tf.fill(shape, base)
           for base in template]
  return tf.reduce_join(bases, 0)


def random_dna_features(experiment_proto, size):
  """Create a dict of feature tensors for random DNA sequences.

  All features other than 'sequence' should use the default value.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    size: scalar integer tf.Tensor giving the number of desired sequences.

  Returns:
    Dict[Any, labeled_tensor.LabeledTensor] providing generated features.
  """
  with tf.name_scope('preprocess_random_input'):
    template = selection.get_template_sequence(experiment_proto)
    features = build_features(experiment_proto)
    feature_tensors = {}
    for k, v in features.items():
      if k != 'sequence':
        default_value = lt.constant(v.default_value, v.dtype, v.axes)
        expanded_default = lt.expand_dims(default_value, ['batch'] + v.axes)
        tiled_default = lt.tile(expanded_default, {'batch': size})
        feature_tensors[k] = tiled_default
    feature_tensors['sequence'] = lt.LabeledTensor(
        random_dna_sequence(template, (size,)), ['batch'])
    return feature_tensors


def input_pipeline(filenames,
                   experiment_proto,
                   final_mbsz,
                   hps,
                   num_epochs=None,
                   num_threads=1):
  """Using Queues create an infinite stream of training minibatches.

  Args:
    filenames: list of paths to sstables tf.Example protos containing training
      data.
    experiment_proto: selection_pb2.Experiment describing the experiment.
    final_mbsz: minibatch size for returned tensors
    hps: optional tf.HParams with hyper-parameters to pass on to preprocess.
    num_epochs: optional number of epochs to pass over the data.
    num_threads: optional number of threads to use for batching output.

  Returns:
    A dequeue_many node that produces input/output pairs.
  """
  prepro_mbsz = 8 * 8 * 1024
  with tf.name_scope('input_pipeline'):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs)
    reader = tf.SSTableReader()
    _, raw_strs = reader.read_up_to(filename_queue, prepro_mbsz)
    strs = lt.LabeledTensor(raw_strs, ['batch'])
    input_features = getattr(hps, 'input_features', ())
    inputs, outputs = preprocess(
        strs,
        experiment_proto,
        input_features=input_features,
        kmer_k_max=hps.kmer_k_max,
        ratio_random_dna=hps.ratio_random_dna,
        mode=hps.preprocess_mode,
        total_reads_defining_positive=hps.total_reads_defining_positive,
        additional_output=hps.additional_output.split(','))
    args = lt.batch(
        list(inputs.values()) + [outputs],
        batch_size=final_mbsz,
        enqueue_many=True,
        capacity=4 * final_mbsz,
        num_threads=num_threads,
        allow_smaller_final_batch=(num_epochs is not None))
    inputs = dict(list(zip(list(inputs.keys()), args[:-1])))
    outputs = args[-1]
    return inputs, outputs
