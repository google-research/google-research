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

"""Library for BERT sequence tagger."""
import collections
import logging
from typing import Any, Callable, Optional, Union

import ml_collections
import tensorflow as tf
import tensorflow_hub as hub

from official.nlp.bert import bert_models
from official.nlp.bert import configs as bert_configs

_ActivationFn = Union[str, Callable[[tf.Tensor], tf.Tensor]]
_Initializer = Union[str, Any]


class BertSequenceTagger(tf.keras.Model):
  """Sequence Tagger model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertSequenceTagger allows a user to pass in a transformer encoder, and
  instantiates a sequence tagging network based on a single dense layer.

  *Note* that the model is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).
  """

  def __init__(self,
               network,
               initializer = "glorot_uniform",
               output = "predictions",
               **kwargs):
    """Initializes a BERT sequence tagger.

    Args:
      network: A transformer network. This network should output a sequence
        output.
      initializer: The initializer (if any) to use in the span labeling network.
        Defaults to a Glorot uniform initializer.
      output: The output style for this network. Can be either `logits`' or
        `predictions`.
      **kwargs: Other key word arguments.
    """

    # We want to use the inputs of the passed network as the inputs to this
    # Model. To do this, we need to keep a handle to the network inputs for use
    # when we construct the Model object at the end of init.
    inputs = network.inputs

    # Because we have a copy of inputs to create this Model object, we can
    # invoke the Network object with its own input tensors to start the Model.
    outputs = network(inputs)
    if isinstance(outputs, list):
      sequence_output = outputs[0]
    else:
      sequence_output = outputs["sequence_output"]

    # The input network (typically a transformer model) may get outputs from all
    # layers. When this case happens, we retrieve the last layer output.
    if isinstance(sequence_output, list):
      sequence_output = sequence_output[-1]

    output_logits = tf.keras.layers.Dense(
        1,  # This layer predicts token level binary label.
        kernel_initializer=initializer,
        name="predictions/transform/logits")(
            sequence_output)

    predictions = tf.keras.layers.Activation(tf.nn.sigmoid)(output_logits)

    if output == "logits":
      output_tensors = output_logits
    elif output == "predictions":
      output_tensors = predictions
    else:
      raise ValueError(
          f'Unknown `output` value {output!r}. `output` can be either "logits" '
          'or "predictions"'
      )

    # Use identity layers wrapped in lambdas to explicitly name the output
    # tensors. This allows us to use string-keyed dicts in Keras fit/predict/
    # evaluate calls.
    output_tensors = tf.keras.layers.Lambda(
        tf.identity, name="output_tensors")(
            output_tensors)

    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super(BertSequenceTagger, self).__init__(
        inputs=inputs, outputs=output_tensors, **kwargs)
    self._network = network
    config_dict = {
        "network": network,
        "initializer": initializer,
        "output": output,
    }
    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple("Config", config_dict.keys())
    self._config = config_cls(**config_dict)

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


def build_model(config):
  """Returns BERT sequence tageer model along with core BERT model."""
  bert_config = bert_configs.BertConfig.from_json_file(
      config.bert.bert_config_file)
  initializer = tf.keras.initializers.TruncatedNormal(
      stddev=bert_config.initializer_range)
  if not config.bert.bert_hub_module_url:
    logging.info("BERT model constructed using provided `bert_config`:\n %r",
                 bert_config)
    bert_encoder = bert_models.get_transformer_encoder(bert_config)
    model = BertSequenceTagger(network=bert_encoder, initializer=initializer)
  else:
    bert_inputs = {
        "input_word_ids":
            tf.keras.layers.Input(
                shape=(config.model.seq_length,),
                dtype=tf.int32,
                name="input_word_ids"),
        "input_mask":
            tf.keras.layers.Input(
                shape=(config.model.seq_length,),
                dtype=tf.int32,
                name="input_mask"),
        "input_type_ids":
            tf.keras.layers.Input(
                shape=(config.model.seq_length,),
                dtype=tf.int32,
                name="input_type_ids"),
    }
    core_model = hub.KerasLayer(
        config.bert.bert_hub_module_url, trainable=config.bert.bert_trainable)
    bert_outputs = core_model(bert_inputs)
    bert_encoder = tf.keras.Model(
        inputs=bert_inputs, outputs=bert_outputs, name="core_model")
    model = BertSequenceTagger(network=bert_encoder, initializer=initializer)

  if config.bert.initial_checkpoint:
    # Not using tf.train.Checkpoint.read, since it loads Keras step counter.
    model.load_weights(config.bert.initial_checkpoint)
    logging.info("BERT tagger initialized from initial checkpoint: %s",
                 config.bert.initial_checkpoint)
  elif config.bert.pretrain_checkpoint:
    checkpoint = tf.train.Checkpoint(model=bert_encoder, encoder=bert_encoder)
    checkpoint.read(
        config.bert.pretrain_checkpoint).assert_existing_objects_matched()
    logging.info("BERT backbone initialized from pretrained checkpoint: %s",
                 config.bert.pretrain_checkpoint)

  return model
