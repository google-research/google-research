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

"""A scikit-lean like interface around tensorflow graphs."""
import abc
import time
import numpy as np
import six
import tensorflow as tf

from meta_reward_learning.semantic_parsing.nsm import data_utils
from meta_reward_learning.semantic_parsing.nsm import tf_utils


@six.add_metaclass(abc.ABCMeta)
class SeqModel(object):
  """Abstract class for a sequence model."""

  @abc.abstractmethod
  def step(self, inputs, state, context, parameters):
    raise NotImplementedError

  @abc.abstractmethod
  def train(self, inputs, targets, context, parameters):
    raise NotImplementedError


class RNNSeqModel(SeqModel):
  """A scikit-learn like interface to a RNN sequence model.

  The model handles the batching and padding for sequence data.

  B is the batch size, T is the time steps or sequence length.
  "..." means scalar, arrays, or tuples.

  Conceptually, the input should contain 4 parts:
    1) inputs that is different at every timestep. shape: (B, T, ...)
    2) initial states that is different for each example. shape: (B, ...)
    3) context that is different at every example, but the same at different
       timestep. shape: (B, ...), may be different for training
       (input sequence for backprop to encoder) and inference
       (encoded sequence).
    4) parameters that is the same for each example. For example,
       the dropout rate. Usually a scalar. shape: (...)

  The output usually contains 2 parts:
    1) outputs at each step, shape: (B, T, ...).
    2) final states. shape: (B, ...)

  In terms of implementation, we use list to represent
  variable length inputs.

  Assume:
  Atom = np.array or float or integer or tuple

  For normal inputs (handled by data_utils.BatchGenerator):
  inputs = [[Atom1, Atom2, ...]]
  size is (B, ...)

  For sequence inputs (handled by data_utils.SeqBatchGenerator):
  inputs = [[Atom_11, Atom_12, ...],
            [Atom_21, Atom_22, ...], ...]
  size is (B, T, ...)
  """

  def __init__(self, graph, batch_size=32, en_maxlen=None, maxlen=None):
    """Creates a RNN sequence model for a given Graph instance."""
    self.graph = graph
    self.session = graph.session
    self.saver = graph.saver
    self.batch_size = batch_size
    self._outputs = graph.outputs
    self._final_state = graph.final_state
    self._n_examples = graph.n_examples
    self._predictions = graph.predictions
    self._probs = graph.prediction_probs
    self._samples = graph.samples
    self.en_maxlen = en_maxlen
    self.maxlen = maxlen
    self.meta_learn = graph.meta_learn

    self._loss = 'loss'
    self._train = 'train'
    self._meta_train = 'meta_train'
    self._count = 'n'
    self._policy_ent = 'ent_reg'

    self._step_bc = data_utils.BatchConverter(
        tuple_keys=['initial_state'], seq_keys=['inputs', 'encoded_context'])
    self._step_ba = data_utils.BatchAggregator(
        tuple_keys=[self._final_state], seq_keys=[self._outputs])

    self._train_bc = data_utils.BatchConverter(['initial_state'],
                                               'inputs targets context'.split())
    self._train_ba = data_utils.BatchAggregator(
        num_keys=[self._loss, self._policy_ent, self._count])

  def set_lr(self, new_lr):
    """Set the learning rate to a new value."""
    self.graph.run(['update_lr'], feed_dict=dict(new_lr=new_lr))

  def get_global_step(self):
    global_step = self.graph.run(['global_step'], {})['global_step']
    return global_step

  def run_epoch(self,
                fetch_list,
                feed_dict,
                batch_converter,
                batch_aggregator,
                shuffle=False,
                parameters=None,
                val_feed_dict=None,
                verbose=1,
                writer=None):
    """Run the TF graph for one pass through the data in feed_dict.

    Args:
      fetch_list: A list of the names of the nodes to be fetched.
      feed_dict: A dictionary with names of the nodes to be feed to as keys.
        Contains the fixed length data.
      prepare_feed_dict_fn: A function to prepare a batch of examples to a feed
        dict for TF graph.
      reduce_result_dict_fn: A reducer to collect results from each iteration.
      shuffle: whether to shuffle the data.
      parameters: A dictionary of parameters.
      writer: A TF Filewriter to write summaries.

    Returns:
      epoch_result_dict: A dictionary with keys from fetch_list and
        the outputs collected through the epoch.
    """
    batch_iterator = data_utils.BatchIterator(
        feed_dict, shuffle=shuffle, batch_size=self.batch_size)
    batch_aggregator.reset()
    if val_feed_dict is not None:
      val_feed_dict = batch_converter.convert(val_feed_dict)
    for batch_data in batch_iterator:
      batch_feed_dict = batch_converter.convert(batch_data)
      if val_feed_dict is not None:
        batch_feed_dict.update(val_feed_dict)
      if parameters is not None:
        batch_feed_dict.update(parameters)
      result_dict = self.graph.run(fetch_list, batch_feed_dict, writer=writer)
      batch_aggregator.merge(result_dict)
    return batch_aggregator.result

  def step(self, inputs, state=None, parameters=None, context=None):
    """Step the RNN with the given inputs and state."""
    feed_dict = dict(
        initial_state=state, inputs=inputs, encoded_context=context)
    fetch_list = [self._outputs, self._final_state]
    result_dict = self.run_epoch(
        fetch_list,
        feed_dict,
        self._step_bc,
        self._step_ba,
        parameters=parameters)
    outputs = result_dict[self._outputs]
    final_state = result_dict[self._final_state]
    return outputs, final_state

  def train(self,
            inputs,
            targets,
            weights=None,
            baselines=None,
            context=None,
            initial_state=None,
            shuffle=True,
            update=True,
            n_epochs=1,
            parameters=None,
            writer=None,
            val_feed_dict=None,
            sim_features=None,
            env_indices=None):
    # TODO(rishabhagarwal): Create a MetaLearnRNNSeq2Seq model to pass the meta
    # learning specific arguments like sim_features and env_indices.
    if weights is None:
      weights = [1.0] * len(inputs)
    if baselines is None:
      baselines = [0.0] * len(inputs)

    feed_dict = dict(
        initial_state=initial_state,
        inputs=inputs,
        targets=targets,
        weights=weights,
        context=context,
        baselines=baselines)
    if env_indices is not None:
      feed_dict.update(env_indices=env_indices)
    if sim_features is not None:
      feed_dict.update(sim_features=sim_features)
    if val_feed_dict is not None:
      # Assumes that val_feed_dict is a python dict containing val_inputs,
      # val_targets, val_weights and val_context
      val_feed_dict['val_batch_size'] = len(val_feed_dict['val_inputs'])

    for _ in xrange(n_epochs):
      t1 = time.time()
      fetch_list = [self._loss, self._count, self._policy_ent]
      if update:
        if self.meta_learn:
          fetch_list += [self._meta_train, self._train]
        else:
          fetch_list += [self._train]
      result_dict = self.run_epoch(
          fetch_list,
          feed_dict,
          self._train_bc,
          self._train_ba,
          shuffle=shuffle,
          val_feed_dict=val_feed_dict,
          parameters=parameters,
          writer=writer)
      t2 = time.time()
      tf.logging.info('{} sec used in one epoch'.format(t2 - t1))
      total_loss = result_dict[self._loss]
      total_n = result_dict[self._count]
      avg_loss = total_loss / total_n
      wps = total_n / (t2 - t1)
    result = dict(loss=avg_loss, wps=wps)
    result['policy_entropy'] = -result_dict[self._policy_ent] / total_n
    return result

  def compute_probs(self,
                    inputs,
                    targets,
                    context=None,
                    initial_state=None,
                    parameters=None):
    feed_dict = dict(
        initial_state=initial_state,
        inputs=inputs,
        targets=targets,
        context=context)
    ba = data_utils.BatchAggregator(tuple_keys=['sequence_probs'])
    fetch_list = ['sequence_probs']
    result_dict = self.run_epoch(
        fetch_list, feed_dict, self._train_bc, ba, parameters=parameters)
    probs = [l[0] for l in result_dict.get('sequence_probs', [])]
    return probs

  def compute_simple_scores(self, sim_features, env_indices=None):
    feed_dict = {}
    if sim_features is not None:
      feed_dict.update(sim_features=sim_features)
    if env_indices is not None:
      feed_dict.update(env_indices=env_indices)
    result_dict = self.graph.run(['scores'], feed_dict=feed_dict)
    return result_dict['scores']

  def compute_scores(self,
                     inputs,
                     targets,
                     context=None,
                     initial_state=None,
                     parameters=None):
    """Computes the scores for the attn based score function."""
    feed_dict = dict(
        initial_state=initial_state,
        inputs=inputs,
        targets=targets,
        context=context)
    ba = data_utils.BatchAggregator(keep_keys=['scores'])
    fetch_list = ['scores']
    result_dict = self.run_epoch(
        fetch_list, feed_dict, self._train_bc, ba, parameters=parameters)
    scores = result_dict.get('scores', [])
    return scores

  def compute_step_logprobs(self,
                            inputs,
                            targets,
                            context=None,
                            initial_state=None,
                            parameters=None):
    feed_dict = dict(
        initial_state=initial_state,
        inputs=inputs,
        targets=targets,
        context=context)
    ba = data_utils.BatchAggregator(seq_keys=['step_logprobs'])
    fetch_list = ['step_logprobs']
    result_dict = self.run_epoch(
        fetch_list, feed_dict, self._train_bc, ba, parameters=parameters)
    logprobs = result_dict.get('step_logprobs', [])
    return logprobs

  def evaluate(self,
               inputs,
               targets,
               weights=None,
               context=None,
               initial_state=None,
               writer=None):
    return self.train(
        inputs,
        targets,
        weights=weights,
        context=context,
        initial_state=initial_state,
        shuffle=False,
        update=False,
        n_epochs=1,
        writer=writer)

  def _predict(self, cell_outputs, predictions_node, temperature=1.0):
    fetch_list = [predictions_node]
    feed_dict = {self._outputs: cell_outputs}

    bc = data_utils.BatchConverter(seq_keys=[self._outputs], maxlen=self.maxlen)
    ba = data_utils.BatchAggregator(seq_keys=[predictions_node])

    result_dict = self.run_epoch(
        fetch_list, feed_dict, bc, ba, parameters=dict(temperature=temperature))
    outputs = result_dict[predictions_node]
    return outputs

  def predict(self, cell_outputs):
    outputs = self._predict(cell_outputs, predictions_node=self._predictions)
    return outputs

  def predict_prob(self, cell_outputs, temperature=1.0):
    return self._predict(
        cell_outputs, predictions_node=self._probs, temperature=temperature)

  def sampling(self, cell_outputs, temperature=1.0):
    return self._predict(
        cell_outputs, predictions_node=self._samples, temperature=temperature)


class RNNSeq2seqModel(RNNSeqModel):
  """Basic seq2seq model."""

  def __init__(self, graph, batch_size=32, en_maxlen=None, maxlen=None):
    """Creates a RNN seq2seq model for a given Graph object."""
    super(RNNSeq2seqModel, self).__init__(
        graph, batch_size=batch_size, en_maxlen=en_maxlen, maxlen=maxlen)
    self._en_outputs = graph.en_outputs
    self._initial_state = graph.initial_state
    self._en_initial_state = graph.en_initial_state
    self._encode_bc = data_utils.BatchConverter(
        tuple_keys=[self._en_initial_state], seq_keys=['context'])
    self._encode_ba = data_utils.BatchAggregator(
        tuple_keys=[self._initial_state], seq_keys=[self._en_outputs])

  def encode(self, en_inputs, en_initial_state=None, parameters=None):
    # The returned outputs and states can be directly used
    # in step as en_outputs (for attention) and initial
    # state (the attention context vector is already concatenated).
    feed_dict = {self._en_initial_state: en_initial_state, 'context': en_inputs}
    fetch_list = [self._en_outputs, self._initial_state]
    result_dict = self.run_epoch(
        fetch_list,
        feed_dict,
        self._encode_bc,
        self._encode_ba,
        parameters=parameters)
    outputs = result_dict[self._en_outputs]
    final_state = result_dict[self._initial_state]
    return outputs, final_state


class MemorySeq2seqModel(RNNSeq2seqModel):
  """Seq2seq model with augmented with key-variable memory."""

  def __init__(self, graph, batch_size=32, en_maxlen=75, maxlen=25):
    super(MemorySeq2seqModel, self).__init__(
        graph, batch_size=batch_size, en_maxlen=en_maxlen, maxlen=maxlen)
    self.max_n_valid_indices = graph.config['core_config'][
        'max_n_valid_indices']
    self.n_mem = graph.config['core_config']['n_mem']
    self.hidden_size = graph.config['core_config']['hidden_size']
    self.value_embedding_size = graph.config['core_config'][
        'value_embedding_size']
    preprocess_fn = lambda x: self._preprocess(x, maxlen=self.maxlen)
    self._encode_bc = data_utils.BatchConverter(
        seq_keys=['en_inputs', 'en_input_features'],
        tuple_keys=[
            'en_initial_state', 'n_constants', 'constant_spans',
            'constant_value_embeddings'
        ],
        preprocess_fn=preprocess_fn,
        maxlen=self.en_maxlen)
    self._step_bc = data_utils.BatchConverter(
        tuple_keys=['initial_state'],
        seq_keys=['encoded_context'],
        preprocess_fn=preprocess_fn,
        maxlen=self.en_maxlen)

    tuple_keys = ['n_constants', 'constant_spans', 'constant_value_embeddings']
    seq_keys = ['targets', 'en_inputs', 'en_input_features']
    out_keys = ['targets']
    if self.meta_learn:
      tuple_keys += ['val_{}'.format(k) for k in tuple_keys]
      seq_keys += ['val_{}'.format(k) for k in seq_keys]
      # Use decoder maxlen for padding the placeholder inputs for these keys
      out_keys += ['val_targets']
    self._train_bc = data_utils.BatchConverter(
        tuple_keys=tuple_keys,
        seq_keys=seq_keys,
        out_keys=out_keys,
        preprocess_fn=preprocess_fn,
        maxlen=self.en_maxlen,
        out_maxlen=self.maxlen)

  def init_pretrained_embeddings(self, pretrained_embeddings):
    self.graph.run(
        ['en_pretrained_embeddings_init'],
        feed_dict={'en_pretrained_embeddings': pretrained_embeddings})

  def init_score_fn(self, ctxt_scores):
    self.graph.run(['scores_init'],
                   feed_dict={'scores_placeholder': ctxt_scores})

  def init_num_trajs(self, num_trajs):
    # TODO(rishabhagarwal): Create a common init_fn for scores and num_trajs.
    self.graph.run(['num_trajs_init'],
                   feed_dict={'num_trajs_placeholder': num_trajs})

  def _preprocess(self, batch_dict, maxlen=None):
    prefixes = ['', 'val_'] if self.meta_learn else ['']
    for prefix in prefixes:
      context_key = prefix + 'context'
      if context_key in batch_dict:
        packed_context = batch_dict[context_key]
        del batch_dict[context_key]
        batch_dict[prefix + 'en_inputs'] = [x[0] for x in packed_context]
        constant_value_embeddings = [x[2] for x in packed_context]
        constant_value_embeddings = [
            _pad_list(cs, np.zeros(self.value_embedding_size), self.n_mem)
            for cs in constant_value_embeddings
        ]
        batch_dict[prefix + 'constant_value_embeddings'] = [
            np.array([x]) for x in constant_value_embeddings
        ]
        batch_dict[prefix + 'n_constants'] = [len(x[1]) for x in packed_context]
        constant_spans = [
            _pad_list(x[1], [-1, -1], self.n_mem) for x in packed_context
        ]
        batch_dict[prefix + 'constant_spans'] = [
            np.array([x]) for x in constant_spans
        ]
        batch_dict[prefix + 'en_input_features'] = [
            np.array(x[3]) for x in packed_context
        ]
      input_key = prefix + 'inputs'
      if input_key in batch_dict:
        processed_step_inputs = self._process_step_inputs(
            batch_dict[input_key], maxlen=maxlen)
        batch_dict[input_key] = processed_step_inputs[0]
        batch_dict[prefix + 'output_features'] = processed_step_inputs[1]

  def _process_step_inputs(self, inputs, maxlen=None):
    """Turn a list of MemoryInputTuple into one MemoryInputTuple.

    Args:
      inputs: a list of MemoryInputTuple, like [MemTuple(1, 2, [1,2,3]),
        MemTuple(1, 2, [1,2,3])...].
      maxlen: Maximum length of a program.

    Returns:
      processed_inputs: a MemoryInputTuple like
        MemTuple(np.array([1, 1, ...]), np.array([2, 2, ...]),
                 np.array([[1, 2, 3, -1, ...], [1, 2, 3, -1,...]))).
    """
    read_ind = np.array([[x[0].read_ind for x in seq] for seq in inputs])
    write_ind = np.array([[x[0].write_ind for x in seq] for seq in inputs])
    valid_indices = np.array([[
        _pad_list(x[0].valid_indices, -1, self.max_n_valid_indices) for x in seq
    ] for seq in inputs])
    output_features = np.array(
        [[_pad_list(x[1], [0], self.max_n_valid_indices)
          for x in seq]
         for seq in inputs])

    read_ind_batch, sequence_length = data_utils.convert_seqs_to_batch(
        read_ind, maxlen)
    output_feature_batch, _ = data_utils.convert_seqs_to_batch(
        output_features, maxlen)
    write_ind_batch, _ = data_utils.convert_seqs_to_batch(write_ind, maxlen)
    valid_indices_batch, _ = data_utils.convert_seqs_to_batch(
        valid_indices, maxlen)
    processed_inputs = tf_utils.MemoryInputTuple(
        read_ind_batch, write_ind_batch, valid_indices_batch)
    return (processed_inputs, sequence_length), (output_feature_batch,
                                                 sequence_length)


def _pad_list(lst, pad, length):
  return np.array(lst + (length - len(lst)) * [pad])
