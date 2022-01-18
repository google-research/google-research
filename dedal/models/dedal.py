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

"""Models for alignment tasks."""

from typing import Optional, Sequence, Type, Union
import gin
import tensorflow as tf

from dedal import multi_task
from dedal import pairs as pairs_lib


Head = Union[tf.keras.layers.Layer, tf.keras.Model]
LayerFactory = Type[tf.keras.layers.Layer]


@gin.configurable
class Selector(tf.keras.layers.Layer):
  """A layer that just returns whatever is passed to it, or a subset of it."""

  def __init__(self, indices=None, **kwargs):
    super().__init__(**kwargs)
    self._indices = indices

  def get_config(self):
    return dict(indices=self._indices)

  def call(self, inputs, mask=None, training=True):
    return (inputs if self._indices is None else
            tuple(inputs[i] for i in self._indices))


def merge(*args):
  """Merges one or several tensor or (nested) list of paired tensors.

  Used to merge potentially complex nested outputs for positives and negatives
  examples that are processed one after the other and needs to be merged.

  One should think about the outputs of a given head for positives and then
  negatives examples. Those two outputs have the same structure even if not the
  same number of examples (batch size). The goal of the merge function is to
  turn it into a single structure where positives results and negatives results
  are merged accordingly.

  Args:
    *args: the sequence of things to be merged. The structures of all the args
    should match.

  Returns:
    The same nested structure of tensors than each of the individual arguments,
    but merged across the different arguments.
  """
  if not args:
    return tf.constant([])

  if len(args) < 2:
    return args[0]

  if all(arg is None for arg in args):
    return None

  if all(isinstance(arg, tf.Tensor) for arg in args):
    return tf.concat(args, axis=0)

  return [merge(*x) for x in zip(*args)]


@gin.configurable
class Dedal(tf.keras.Model):
  """Main architecture for the aligner. This is shared among all our models."""

  def __init__(self,
               encoder_cls = gin.REQUIRED,
               aligner_cls = gin.REQUIRED,
               heads_cls = gin.REQUIRED,
               process_negatives = True,
               switch = None,
               backprop = None,
               **kwargs):
    """Initializes.

    Args:
      encoder_cls: a keras layer (or model) that turns a batch of sequences
        into a batch of sequence embeddings.
      aligner_cls: a layer that turns a pairs of sequence embeddings into
        scores (and optionally paths).
      heads_cls: for a multi-task setup all the layers to be plugged either on
        the embeddings or on the alignments.
      process_negatives: should the network consider the negatives or not in a
        batch. Aligning might be expensive, so switching off negative
        supervision saves half of the alignment cost.
      switch: optional mapping of output heads to inputs, required only when
        calling the model in multi-input mode. See `call` for additional
        details.
      backprop: for a multi-task setup, whether to backprop each loss from the
        output head to the encoder (i.e. finetune the encoder on the loss) or
        train the output head params only.
      **kwargs: optional keyword arguments to be passed to `tf.keras.Model`.
    """
    super().__init__(**kwargs)
    self.encoder = encoder_cls()
    self.aligner = aligner_cls() if aligner_cls else None
    self.heads = multi_task.Backbone(
        embeddings=[head_cls() if head_cls is not None else None
                    for head_cls in heads_cls.embeddings],
        alignments=[head_cls() if head_cls is not None else None
                    for head_cls in heads_cls.alignments])
    self.process_negatives = process_negatives
    self.switch = switch
    if self.switch is None:
      self.switch = multi_task.SwitchBackbone.constant_like(self.heads)
    self.backprop = backprop
    if self.backprop is None:
      self.backprop = self.heads.constant_copy(True)
    # For TF to keep track of variables
    self._flat_heads = self.heads.flatten()

  def head_output(self,
                  head,
                  on,
                  backprop,
                  inputs,
                  mask = None,
                  training = True):
    """Returns the output of the given head."""
    if not on:
      return tf.constant([])
    elif head is None:
      return inputs
    else:
      if not backprop:
        inputs = tf.nest.map_structure(tf.stop_gradient, inputs)
      return head(inputs, mask=mask, training=training)

  def forward(self,
              inputs,
              selector = None,
              training = True):
    """Run the models on a single input and potentially selects some heads only.

    Args:
      inputs: a Tensor<int32>[batch, seq_len] representing protein sequences.
      selector: If set a multi_task.Backbone[bool] to specify which head to
        apply. For non selected heads, a None will replace the output. If not
        set, all the heads will be output.
      training: whether to run in training mode or eval mode.

    Returns:
      A multi_task.Backbone of tensor corresponding to the output of the
      different heads of the model.
    """
    selector = self.heads.constant_copy(True) if selector is None else selector

    embeddings = self.encoder(inputs, training=training)
    masks = self.encoder.compute_mask(inputs)

    result = multi_task.Backbone()
    for head, on, backprop, in zip(self.heads.embeddings,
                                   selector.embeddings,
                                   self.backprop.embeddings):
      head_output = self.head_output(
          head, on, backprop, embeddings, mask=masks, training=training)
      result.embeddings.append(head_output)

    if not self.heads.alignments or not any(selector.alignments):
      # Ensures structure of result matches self.heads even when method skips
      # alignment phase due to selector.
      for _ in selector.alignments:
        result.alignments.append(tf.constant([]))
      return result

    # For each head, we compute the output of positive pairs and negative ones,
    # then concatenate to obtain an output batch where the first half is
    # positive and the second half is negative.
    outputs = []
    pos_indices = pairs_lib.consecutive_indices(inputs)
    neg_indices = (pairs_lib.roll_indices(pos_indices)
                   if self.process_negatives else None)
    num_alignment_calls = 1 + int(self.process_negatives)
    for indices in (pos_indices, neg_indices)[:num_alignment_calls]:
      curr = []
      embedding_pairs, mask_pairs = pairs_lib.build(indices, embeddings, masks)
      alignments = self.aligner(
          embedding_pairs, mask=mask_pairs, training=training)
      for head, on, backprop, in zip(self.heads.alignments,
                                     selector.alignments,
                                     self.backprop.alignments):
        head_output = self.head_output(
            head, on, backprop, alignments, mask=mask_pairs, training=training)
        curr.append(head_output)
      outputs.append(curr)

    for output in merge(*outputs):
      result.alignments.append(output)
    return result

  def call(self,
           inputs,
           selector = None,
           training = True):
    """Run the models and potentially selects some heads only.

    Currently supports two modes of operation:
    + Single-input mode: encodes a single batch of protein sequences, which is
        then fed to *all* embedding and alignment output heads.
    + Multi-input mode: encodes N batches of protein sequences, with each batch
        being fed to *a fixed disjoint subset* of embedding and alignment output
        heads as configured by the `switch` argument of the model's constructor.
    Arbitrary (e.g. overlapping) input -> output head assignments in multi-input
    mode not yet supported.

    Args:
      inputs: a Tensor<int32>[batch, seq_len] list of Tensor<int32>[batch_i,
        seq_len_i] representing batches of protein sequences. Multi-input mode
        requires the argument `switch` having been provided to the model's
        constructor, and the number of inputs must match `switch.n` as defined
        by `multi_task.SwitchBackbone`.
      selector: If set a multi_task.Backbone[bool] to specify which head to
        apply. For non selected heads, a None will replace the output. If not
        set, all the heads will be output.
      training: whether to run in training mode or eval mode.

    Returns:
      A multi_task.Backbone of tensor corresponding to the output of the
      different heads of the model.
    """
    inputs_list = tf.nest.flatten(inputs)
    selector = self.heads.constant_copy(True) if selector is None else selector
    outputs_list = []
    for i, inputs in enumerate(inputs_list):
      selector_i = self.heads.pack(
          [f1 and f2 for f1, f2 in zip(selector, self.switch.get_selector(i))])
      outputs_i = self.forward(inputs, selector=selector_i, training=training)
      # Removes "dummy" entries in `outputs_i` corresponding to positions where
      # `self._switch.get_selector(i)` is False, while keeping "dummy" entries
      # due to the externaly provided `selector` argument.
      outputs_i = self.switch.filter(outputs_i, i)
      outputs_list.append(outputs_i)
    return self.switch.merge(outputs_list)


@gin.configurable
class DedalLight(tf.keras.Model):
  """A light-weight model to be easily exported with tf.saved_model."""

  def __init__(self, encoder, aligner, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.aligner = aligner

  @tf.function
  def call(self, inputs, training = False, embeddings_only = False):
    embeddings = self.encoder(inputs, training=training)
    if embeddings_only:
      return embeddings
    masks = self.encoder.compute_mask(inputs)
    indices = pairs_lib.consecutive_indices(inputs)
    embedding_pairs, mask_pairs = pairs_lib.build(indices, embeddings, masks)
    alignments = self.aligner(
        embedding_pairs, mask=mask_pairs, training=training)
    return alignments

