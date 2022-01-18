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

"""Specifications for some datasets."""

import json
import os
from typing import Mapping, List, Optional, Sequence, Tuple, Type

import gin
import numpy as np
import tensorflow as tf

from dedal import multi_task
from dedal import vocabulary
from dedal.data import align_transforms
from dedal.data import builder
from dedal.data import loaders
from dedal.data import serialization
from dedal.data import transforms


TAPE_SPECS = {
    'proteinnet': {
        'primary': tf.string,
        'evolutionary': tf.float32,
        'tertiary': tf.float32,
        'valid_mask': tf.int64,
    },
    'secondary_structure': {
        'primary': tf.string,
        'asa_max': tf.float32,
        'disorder': tf.int64,
        'evolutionary': tf.float32,
        'interface': tf.int64,
        'phi': tf.float32,
        'psi': tf.float32,
        'rsa': tf.float32,
        'ss3': tf.int64,
        'ss8': tf.int64,
        'valid_mask': tf.float32,
    },
    'remote_homology': {
        'primary': tf.string,
        'class_label': tf.io.FixedLenFeature([], tf.int64),
        'fold_label': tf.io.FixedLenFeature([], tf.int64),
        'superfamily_label': tf.io.FixedLenFeature([], tf.int64),
        'family_label': tf.io.FixedLenFeature([], tf.int64),
        'secondary_structure': tf.float32,
        'solvent_accessibility': tf.float32,
        'evolutionary': tf.float32,
    },
    'fluorescence': {
        'primary': tf.string,
        'num_mutations': tf.io.FixedLenFeature([], tf.int64),
        'log_fluorescence': tf.float32,
    },
    'stability': {
        'primary': tf.string,
        'topology': tf.string,
        'parent': tf.string,
        'stability_score': tf.float32,
    }
}

TAPE_NUM_OUTPUTS = {
    'secondary_structure': {
        'ss3': 3,
        'ss8': 8,
        'phi': 2,
        'psi': 2,
    },
    'remote_homology': {
        'fold_label': 1195,
        'superfamily_label': 1462,
        'family_label': 2251,
    },
    'fluorescence': {},
    'stability': {},
}

TAPE_SEQ2SEQ_TASKS = ('ss3', 'ss8', 'phi', 'psi', 'asa_max', 'disorder', 'rsa')
TAPE_MULTI_CL_TASKS = ('ss3', 'ss8', 'fold_label', 'superfamily_label',
                       'family_label')
TAPE_BACKBONE_ANGLE_TASKS = ('phi', 'psi')
TAPE_PROT_ENGINEERING_TASKS = ('log_fluorescence', 'stability_score')


@gin.configurable
def make_pfam34_loader(
    root_dir,
    extra_keys = ('fam_key',),
    task = 'iid'):
  """Creates a loader for Pfam-A seed 34.0 data."""
  has_context = task.endswith('with_ctx')

  folder = os.path.join(root_dir, 'pfam34')
  with tf.io.gfile.GFile(os.path.join(folder, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

  all_specs = {
      'seq': tf.io.VarLenFeature(tf.int64),
      'seq_key': tf.io.FixedLenFeature([], tf.int64),
      'fam_key': tf.io.FixedLenFeature([], tf.int64),
      'cla_key': tf.io.FixedLenFeature([], tf.int64),
      'seq_len': tf.io.FixedLenFeature([], tf.int64),
      'id': tf.io.FixedLenFeature([], tf.string),
      'ac': tf.io.FixedLenFeature([], tf.string),
      'start': tf.io.FixedLenFeature([], tf.int64),
      'end': tf.io.FixedLenFeature([], tf.int64),
      'ss': tf.io.FixedLenFeature([], tf.string),
  }
  for pid_th in metadata['pid_ths']:
    all_specs[f'ci_{pid_th}'] = tf.io.FixedLenFeature([], tf.int64)
  if has_context:
    all_specs['full_seq'] = tf.io.VarLenFeature(tf.int64)

  # Silently ignores misspecified keys.
  input_sequence_key = (('seq', 'full_seq') if has_context
                        else ('seq',))
  output_sequence_key = (('sequence', 'full_sequence') if has_context
                         else ('sequence',))
  extra_keys = (extra_keys,) if isinstance(extra_keys, str) else extra_keys
  specs = {
      k: all_specs[k]
      for k in input_sequence_key + tuple(extra_keys)
      if k in all_specs
  }
  return loaders.TFRecordsLoader(
      folder=os.path.join(folder, task),
      coder=serialization.FlatCoder(specs=specs),
      input_sequence_key=input_sequence_key,
      output_sequence_key=output_sequence_key)


def get_sequence_coder(specs):
  """Returns a SequenceCoder with the proper specs."""

  def not_sequence(v):
    return v == tf.string or isinstance(v, tf.io.FixedLenFeature)

  sequence_keys = [k for k, v in specs.items() if not not_sequence(v)]
  return serialization.SequenceCoder(specs=specs, sequence_keys=sequence_keys)


def make_sequence_coded_loader(specs, root_dir, folder):
  """Returns a Loader for Proteinnet data."""
  folder = os.path.join(root_dir, folder)
  coder = get_sequence_coder(specs)
  return loaders.TFRecordsLoader(
      folder, coder=coder, input_sequence_key='primary', split_folder=False)


@gin.configurable
def make_tape_loader(root_dir, task = 'proteinnet'):
  specs = TAPE_SPECS.get(task, None)
  if specs is None:
    raise ValueError(f'Unknown TAPE task {task}')
  return make_sequence_coded_loader(specs, root_dir, task)


@gin.configurable
class PfamLoader(loaders.CSVLoader):
  """Loader for Pfam-A seed 32.0 data."""

  def __init__(self,
               root_dir,
               extra_keys = ('fam_key',),
               task = 'random_split',
               output_sequence_key = 'sequence'):
    self._output_sequence_key = output_sequence_key
    fields = {
        'seq_key': tf.int32,
        'ci_100': tf.int32,
        'fam_key': tf.int32,
        'cla_key': tf.int32,
        'seq_len': tf.int32,
        self._output_sequence_key: tf.string,
    }

    super().__init__(
        folder=os.path.join(root_dir, 'pfam', task),
        fields=fields,
        fields_to_use=tuple(extra_keys) + (self._output_sequence_key,),
    )

  def load(self, split):
    """Creates CSVDataset for split, encoding the sequence."""
    ds = super().load(split)
    return ds.map(
        transforms.Encode(on=self._output_sequence_key),
        num_parallel_calls=tf.data.AUTOTUNE)


@gin.configurable
class ScopLoader(loaders.CSVLoader):
  """Loader for ASTRAL SCOPe v2.07 data."""

  def __init__(self,
               root_dir,
               extra_keys = ('sf_key',),
               pid = 40,
               output_sequence_key = 'sequence'):
    self._output_sequence_key = output_sequence_key
    fields = {
        'seq_key': tf.int32,
        'seq_len': tf.int32,
        'id': tf.string,
        'cl_key': tf.int32,
        'cf_key': tf.int32,
        'sf_key': tf.int32,
        'fa_key': tf.int32,
        self._output_sequence_key: tf.string,
    }

    super().__init__(
        folder=os.path.join(root_dir, f'astral{pid}'),
        fields=fields,
        fields_to_use=tuple(extra_keys) + (self._output_sequence_key,),
    )

  def load(self, split):
    """Creates CSVDataset for split, encoding the sequence."""
    ds = super().load(split)
    return ds.map(
        transforms.Encode(on=self._output_sequence_key),
        num_parallel_calls=tf.data.AUTOTUNE)


@gin.configurable
def make_pfam_pairs_loader(
    root_dir,
    extra_keys = ('fam_key',),
    task = 'pfam34_pairs/iid_ood_clans',
    branch_key = 'ci_100',
    suffixes = ('1', '2')):
  """Creates a Loader for pre-paired Pfam-A seed data."""
  has_context = task.endswith('with_ctx')

  input_sequence_key = []
  for key in ('seq', 'full_seq') if has_context else ('seq',):
    for suffix in suffixes:
      input_sequence_key.append(f'{key}_{suffix}')
  input_sequence_key = tuple(input_sequence_key)

  output_sequence_key = []
  for key in ('sequence', 'full_sequence') if has_context else ('sequence',):
    for suffix in suffixes:
      output_sequence_key.append(f'{key}_{suffix}')
  output_sequence_key = tuple(output_sequence_key)

  specs = {}
  for key in input_sequence_key:
    specs[key] = tf.io.VarLenFeature(tf.int64)
  for suffix in suffixes:
    for key in extra_keys:
      specs[f'{key}_{suffix}'] = tf.io.FixedLenFeature([], tf.int64)

  return loaders.TFRecordsLoader(
      folder=os.path.join(root_dir, task, branch_key),
      coder=serialization.FlatCoder(specs=specs),
      input_sequence_key=input_sequence_key,
      output_sequence_key=output_sequence_key)


@gin.configurable
def make_pair_builder(
    max_len = 512,
    index_keys = ('fam_key', 'ci_100'),
    process_negatives = True,
    gap_token = '-',
    sequence_key = 'sequence',
    context_sequence_key = 'full_sequence',
    loader_cls=make_pfam_pairs_loader,
    pairing_cls = None,
    lm_cls = None,
    has_context = False,
    append_eos = True,
    append_eos_context = True,
    add_random_tails = False,
    **kwargs):
  """Creates a dataset for pairs of sequences."""
  # Convenience function to index key pairs.
  paired_keys = lambda k: tuple(f'{k}_{i}' for i in (1, 2))

  def stack_and_pop(on):
    stack = transforms.Stack(on=paired_keys(on), out=on)
    pop = transforms.Pop(on=paired_keys(on))
    return [stack, pop]

  # Defines fields to be read from the TFRecords.
  metadata_keys = ['cla_key', 'seq_key'] + list(index_keys)
  extra_keys = metadata_keys.copy()
  # Pre-paired datasets already been filtered by length, seq_len only needed
  # when pairing sequences on-the-fly.
  if pairing_cls is not None:
    extra_keys.append('seq_len')
  # Optionally, adds fields needed by the `AddAlignmentContext` `Transform`.
  if has_context:
    extra_keys.extend(['start', 'end'])
  add_alignment_context_extra_args = (paired_keys(context_sequence_key) +
                                      paired_keys('start') + paired_keys('end'))
  # Accounts for EOS token if necessary.
  max_len_eos = max_len - 1 if append_eos else max_len

  ### Sets up the `DatasetTransform`s.

  ds_transformations = []
  if pairing_cls is not None:
    filter_by_length = transforms.FilterByLength(max_len=max_len_eos)
    # NOTE(fllinares): pairing on-the-fly is memory intensive on TPU for some
    # reason not yet understood...
    pair_sequences = pairing_cls(index_keys=index_keys)
    ds_transformations.extend([filter_by_length,
                               pair_sequences])

  ### Sets up the `Transform`s applied *before* batching.

  project_msa_rows = align_transforms.ProjectMSARows(
      on=paired_keys(sequence_key),
      token=gap_token)
  append_eos_to_context = transforms.EOS(
      on=paired_keys(context_sequence_key))
  add_alignment_context = align_transforms.AddAlignmentContext(
      on=paired_keys(sequence_key) + add_alignment_context_extra_args,
      out=paired_keys(sequence_key),
      max_len=max_len_eos,
      gap_token=gap_token)
  trim_alignment = align_transforms.TrimAlignment(
      on=paired_keys(sequence_key),
      gap_token=gap_token)
  pop_add_alignment_context_extra_args = transforms.Pop(
      on=add_alignment_context_extra_args)
  add_random_prefix_and_suffix = align_transforms.AddRandomTails(
      on=paired_keys(sequence_key),
      max_len=max_len_eos)
  create_alignment_targets = align_transforms.CreateAlignmentTargets(
      on=paired_keys(sequence_key),
      out='alignment/targets',
      gap_token=gap_token,
      n_prepend_tokens=0)
  pid1 = align_transforms.PID(
      on=paired_keys(sequence_key),
      out='alignment/pid1',
      definition=1,
      token=gap_token)
  pid3 = align_transforms.PID(
      on=paired_keys(sequence_key),
      out='alignment/pid3',
      definition=3,
      token=gap_token)
  remove_gaps = transforms.RemoveTokens(
      on=paired_keys(sequence_key),
      tokens=gap_token)
  append_eos_to_sequence = transforms.EOS(
      on=paired_keys(sequence_key))
  pad_sequences = transforms.CropOrPad(
      on=paired_keys(sequence_key),
      size=max_len)
  pad_alignment_targets = transforms.CropOrPadND(
      on='alignment/targets',
      size=2 * max_len)

  transformations = [project_msa_rows]
  if has_context:
    if append_eos_context:
      transformations.append(append_eos_to_context)
    transformations.extend([add_alignment_context,
                            trim_alignment,
                            pop_add_alignment_context_extra_args])
  if add_random_tails:
    transformations.append(add_random_prefix_and_suffix)
  transformations.append(create_alignment_targets)

  transformations.extend([pid1,
                          pid3,
                          remove_gaps])
  if append_eos:
    transformations.append(append_eos_to_sequence)
  transformations.extend([pad_sequences,
                          pad_alignment_targets])
  for key in [sequence_key] + metadata_keys:
    transformations.extend(stack_and_pop(key))

  ### Sets up the `Transform`s applied *after* batching.

  flatten_sequence_pairs = transforms.Reshape(
      on=sequence_key,
      shape=[-1, max_len])
  flatten_metadata_pairs = transforms.Reshape(
      on=metadata_keys,
      shape=[-1])
  create_homology_targets = align_transforms.CreateHomologyTargets(
      on='fam_key',
      out='homology/targets',
      process_negatives=process_negatives)
  create_alignment_weights = align_transforms.CreateBatchedWeights(
      on='alignment/targets',
      out='alignment/weights')
  add_neg_alignment_targets_and_weights = align_transforms.PadNegativePairs(
      on=('alignment/targets', 'alignment/weights'))
  pad_neg_pid = align_transforms.PadNegativePairs(
      on=('alignment/pid1', 'alignment/pid3'),
      value=-1.0)

  batched_transformations = [flatten_sequence_pairs,
                             flatten_metadata_pairs,
                             create_homology_targets]
  if process_negatives:
    batched_transformations.extend([create_alignment_weights,
                                    add_neg_alignment_targets_and_weights,
                                    pad_neg_pid])
  if lm_cls is not None:
    create_lm_targets = lm_cls(
        on=sequence_key,
        out=(sequence_key, 'masked_lm/targets', 'masked_lm/weights'))
    batched_transformations.append(create_lm_targets)

  ### Sets up the remainder of the `DatasetBuilder` configuration.

  masked_lm_labels = ('masked_lm/targets', 'masked_lm/weights')
  alignment_labels = ('alignment/targets' if not process_negatives else
                      ('alignment/targets', 'alignment/weights'))
  homology_labels = 'homology/targets'
  embeddings = () if lm_cls is None else (masked_lm_labels,)
  alignments = (alignment_labels, homology_labels)

  return builder.DatasetBuilder(
      data_loader=loader_cls(extra_keys),
      ds_transformations=ds_transformations,
      transformations=transformations,
      batched_transformations=batched_transformations,
      labels=multi_task.Backbone(embeddings=embeddings, alignments=alignments),
      metadata=('seq_key', 'alignment/pid1', 'alignment/pid3'),
      sequence_key=sequence_key,
      **kwargs)


def split_key(key):
  keys = tf.random.experimental.stateless_split(key)
  return keys[0], keys[1]


@gin.configurable
def make_tape_builder(root_dir,
                      task,
                      target,
                      weights = None,
                      metadata = (),
                      max_len = 1024,
                      input_sequence_key = 'primary',
                      output_sequence_key = 'sequence'):
  """Creates a DatasetBuilder for TAPE's benchmark."""
  supported_tasks = list(TAPE_NUM_OUTPUTS)
  if task not in supported_tasks:
    raise ValueError(f'Task {task} not recognized.'
                     f'Supported tasks: {", ".join(supported_tasks)}.')
  num_outputs = TAPE_NUM_OUTPUTS[task].get(target, 1)

  used_keys = [input_sequence_key, target]
  if weights is not None:
    used_keys.append(weights)
  if metadata:
    used_keys.extend(metadata)
  unused_keys = [k for k in TAPE_SPECS[task] if k not in used_keys]

  ds_transformations = []
  if max_len is not None:
    ds_transformations.append(
        transforms.FilterByLength(
            on=output_sequence_key, precomputed=False, max_len=max_len - 1))

  transformations = [
      transforms.Pop(on=unused_keys),
      transforms.Reshape(on=output_sequence_key, shape=[]),
      transforms.Encode(on=output_sequence_key),
      transforms.EOS(on=output_sequence_key),
      transforms.CropOrPad(on=output_sequence_key, size=max_len),
  ]

  if target in TAPE_MULTI_CL_TASKS:
    transformations.append(transforms.OneHot(on=target, depth=num_outputs))
  elif target in TAPE_BACKBONE_ANGLE_TASKS:
    transformations.append(transforms.BackboneAngleTransform(on=target))
  elif target in TAPE_PROT_ENGINEERING_TASKS:
    transformations.append(transforms.Reshape(on=target, shape=[-1]))

  if target in TAPE_SEQ2SEQ_TASKS:
    transformations.extend([
        transforms.Reshape(on=target, shape=[-1, num_outputs]),
        transforms.CropOrPadND(on=target, size=max_len, axis=0),
    ])

  if weights is not None:  # Note: no seq-level TAPE task has weights.
    transformations.extend([
        transforms.Reshape(on=weights, shape=[-1]),
        transforms.CropOrPadND(on=weights, size=max_len),
    ])

  embeddings_labels = [target] if weights is None else [(target, weights)]
  return builder.DatasetBuilder(
      data_loader=make_tape_loader(root_dir=root_dir, task=task),
      ds_transformations=ds_transformations,
      transformations=transformations,
      labels=multi_task.Backbone(embeddings=embeddings_labels),
      metadata=metadata,
      sequence_key=output_sequence_key)


@gin.configurable
class FakePairsLoader:
  """Generates synthetic sequence pairs, sampled from simplified PairHMM."""
  START = 0
  MATCH = 1
  GAP_IN_X = 2
  GAP_IN_Y = 3
  END = 4
  INIT_TRANS = 0

  def __init__(self,
               max_len = 512,
               tau = 0.01,
               alpha = 0.05,
               eta = 0.7,
               vocab = None):
    self._max_len = max_len
    vocab = vocabulary.get_default() if vocab is None else vocab
    self._sampler = vocabulary.Sampler(vocab=vocab)
    self._eos = vocab.get(vocab.specials[-1])
    self._pad = vocab.padding_code

    # Transition look-up table (excluding special initial transition).
    look_up = {
        (self.MATCH, self.MATCH): 1,
        (self.GAP_IN_X, self.MATCH): 2,
        (self.GAP_IN_Y, self.MATCH): 3,
        (self.MATCH, self.GAP_IN_X): 4,
        (self.GAP_IN_X, self.GAP_IN_X): 5,
        (self.GAP_IN_Y, self.GAP_IN_X): 9,  # "forbidden" transition.
        (self.MATCH, self.GAP_IN_Y): 6,
        (self.GAP_IN_X, self.GAP_IN_Y): 7,
        (self.GAP_IN_Y, self.GAP_IN_Y): 8,
    }
    # Builds data structures for efficiently encoding transitions.
    self._hash_fn = lambda d0, d1: 3 * (d1 + 1) + (d0 + 1)
    hashes = [self._hash_fn(d0, d1) for (d0, d1) in look_up]
    trans_encoder = tf.scatter_nd(
        indices=[[x] for x in hashes],
        updates=list(look_up.values()),
        shape=[max(hashes) + 1])
    self._trans_encoder = tf.cast(trans_encoder, tf.int32)
    self._init_trans = tf.convert_to_tensor([self.INIT_TRANS], dtype=tf.int32)

    cond_probs = tf.convert_to_tensor(
        [[0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0 - 2.0 * alpha - tau, alpha, alpha, tau],
         [0.0, eta, 1.0 - eta - alpha, alpha, 0.0],
         [0.0, eta, 0.0, 1.0 - eta, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
        tf.float32)
    self._logits = tf.where(cond_probs > 0.0, tf.math.log(cond_probs), -np.inf)

    self._delta_len_x = tf.convert_to_tensor([0, 1, 0, 1, 0])
    self._delta_len_y = tf.convert_to_tensor([0, 1, 1, 0, 0])

  def sample_state(self, key, state):
    """Samples next PairHMM state given current."""
    new_state = tf.random.stateless_categorical(
        self._logits[state][None, :], 1, key, dtype=tf.int32)
    return tf.reshape(new_state, ())

  def sample_states(self, key):
    """Autoregressively samples PairHMM hidden state sequence."""
    states = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    key, subkey = split_key(key)
    state = self.sample_state(subkey, 0)
    len_x, len_y = 0, 0
    i = 0
    while state != self.END and tf.maximum(len_x, len_y) < self._max_len - 1:
      states = states.write(i, state)
      len_x += self._delta_len_x[state]
      len_y += self._delta_len_y[state]
      i += 1
      key, subkey = split_key(key)
      state = self.sample_state(subkey, state)
    return states.stack()

  def extend_tails(self, key,
                   seq):
    """Samples flanking sequences."""
    key_s, key_e = split_key(key)
    spare = tf.maximum(0, self._max_len - 1 - tf.shape(seq)[0])
    start = tf.convert_to_tensor(0, tf.int32)
    end = tf.convert_to_tensor(0, tf.int32)
    if spare > 0:
      start = tf.random.stateless_uniform((),
                                          key_s,
                                          maxval=spare,
                                          dtype=tf.int32)
      spare -= start
    if spare > 0:
      end = tf.random.stateless_uniform((), key_e, maxval=spare, dtype=tf.int32)
      spare -= end
    seq = tf.concat([
        self._sampler.sample([start]), seq,
        self._sampler.sample([end]), [self._eos],
        tf.fill([spare], self._pad)
    ], 0)
    return seq, start

  def generate_pair(self, key):
    """Samples sequence pair from (simplified) pair HMM given seed."""
    states = self.sample_states(key)
    aln_len = tf.shape(states)[0]
    tokens = self._sampler.sample([aln_len])

    em_x = tf.cast(
        tf.logical_or(states == self.MATCH, states == self.GAP_IN_Y), tf.int32)
    em_y = tf.cast(
        tf.logical_or(states == self.MATCH, states == self.GAP_IN_X), tf.int32)
    seq_x = tf.gather(tokens, tf.squeeze(tf.where(em_x), axis=1))
    seq_y = tf.gather(tokens, tf.squeeze(tf.where(em_y), axis=1))
    key_x, key_y = split_key(key)
    seq_x, start_x = self.extend_tails(key_x, seq_x)
    seq_y, start_y = self.extend_tails(key_y, seq_y)

    pos_x = start_x + tf.cumsum(em_x)
    pos_y = start_y + tf.cumsum(em_y)
    enc_trans = tf.gather(self._trans_encoder,
                          self._hash_fn(states[:-1], states[1:]))
    enc_trans = tf.concat([self._init_trans, enc_trans], 0)
    alignment_targets = tf.stack([pos_x, pos_y, enc_trans])

    sequence = tf.stack([seq_x, seq_y])
    aln_to_pad = tf.maximum(0, 2 * self._max_len - aln_len)
    alignment_targets = tf.pad(alignment_targets, [[0, 0], [0, aln_to_pad]])

    fam_key = tf.random.stateless_uniform(
        [], key, maxval=tf.int32.max, dtype=tf.int32)
    fam_key = tf.repeat(fam_key, 2)
    sequence.set_shape([2, self._max_len])
    alignment_targets.set_shape([3, 2 * self._max_len])
    fam_key.set_shape([2])
    pid1_key, pid3_key = split_key(key)
    pid1 = tf.random.stateless_uniform([], pid1_key)
    pid3 = tf.random.stateless_uniform([], pid3_key)

    return {
        'sequence': sequence,
        'alignment/targets': alignment_targets,
        'fam_key': fam_key,
        'alignment/pid1': pid1,
        'alignment/pid3': pid3,
    }

  def load(self, _):
    ds = tf.data.experimental.RandomDataset().batch(2)
    ds = ds.map(self.generate_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


@gin.configurable
def make_fake_builder(max_len = 512):
  return builder.DatasetBuilder(
      data_loader=FakePairsLoader(max_len=max_len),
      labels=multi_task.Backbone(alignments=[
          ('alignment/targets', 'alignment/weights'),
          'homology/targets']),
      batched_transformations=[
          transforms.Reshape(shape=(-1, max_len), on='sequence'),
          transforms.Reshape(shape=(-1,), on='fam_key'),
          align_transforms.CreateBatchedWeights(
              on='alignment/targets', out='alignment/weights'),
          align_transforms.PadNegativePairs(
              on=['alignment/targets', 'alignment/weights']),
          align_transforms.PadNegativePairs(
              value=-1.0, on=['alignment/pid1', 'alignment/pid3']),
          align_transforms.CreateHomologyTargets(
              process_negatives=True, on='fam_key', out='homology/targets'),
      ],
      metadata=('alignment/pid1', 'alignment/pid3'),
      split='train')
