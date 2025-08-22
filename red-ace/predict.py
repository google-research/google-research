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

"""For running inference with a RED-ACE model.

Throughout this file we use the terms predict and realize. Prediction involves
making a decision (what tag to apply).
Realization takes this decision and applies it to the input sentence to produce
the output sentence.
"""
import dataclasses
from typing import Mapping, Sequence, cast

from absl import flags
import example_builder
import numpy as np
from official.nlp import optimization
import redace_flags  # pylint: disable=unused-import
import redace_models
import tensorflow as tf
import tokenization

FLAGS = flags.FLAGS


@dataclasses.dataclass
class DetailedInferenceResult:
  """A detailed result of RED-ACE inference.

  Attributes:
    input_tokens: The input tokens to the tagging model.
    tag_probabilities: The probabilities of the selected tags.
    tags: The sequences of tags selected.
  """

  input_tokens: Sequence[str]
  tag_probabilities: Sequence[float]
  tags: Sequence[str]

  def __init__(self):
    pass


class RedAcePredictor:
  """Class for computing and realizing predictions with RED-ACE.

  Attributes:
    tagging_model: The TF model responsible for tagging.
  """

  def __init__(
      self,
      redace_config,
      model_filepath,
      sequence_length=128,
      batch_size=None,
  ):
    """Initializes an instance of RedAcePredictor.

    Args:
        redace_config: The config file for the tagging model.
        model_filepath: The file location of the tagging model.  If not provided
          in a randomly initialized model is used (not recommended).
        sequence_length: Maximum length of a sequence.
        batch_size: Batch size is needed for joint model, can bet set to None
          for all other models.
    """
    self._redace_config = redace_config
    self._model_filepath = model_filepath
    self.tagging_model = None
    self._sequence_length = sequence_length
    self._batch_size = batch_size

    self._builder = example_builder.RedAceExampleBuilder(
        tokenization.FullTokenizer(FLAGS.vocab_file), sequence_length)

    self._inverse_label_map = {
        tag_id: tag for tag, tag_id in self._builder.label_map.items()
    }

  def predict_end_to_end_batch(
      self,
      batch,
      confidence_scores_batch,
  ):
    """Takes in a batch of source sentences and runs RED-ACE on them.

    Args:
      batch: Source inputs, where each input is composed of multiple source
        utterances.
      confidence_scores_batch: Confidence scores.

    Returns:
      taggings_outputs: Intermediate realized output of the tagging model.
      detailed_inference_results: A DetailedInferenceResult.
    """

    detailed_inference_results = [
        DetailedInferenceResult() for i in range(len(batch))
    ]
    taggings_outputs = self._predict_and_realize_batch(
        batch,
        confidence_scores_batch,
        detailed_inference_results=detailed_inference_results,
    )
    return taggings_outputs, detailed_inference_results

  def load_model(self):
    """Loads either an insertion or tagging model for inference."""

    def _get_fake_loss_fn():
      """A fake loss function for inference."""

      def _fake_loss_fn(unused_labels, unused_losses, **unused_args):
        return 0.0

      return _fake_loss_fn

    model, _ = redace_models.get_model(
        self._redace_config,
        self._sequence_length,
        is_training=False,
        batch_size=self._batch_size,
    )
    if self._model_filepath:
      checkpoint = tf.train.Checkpoint(model=model)
      latest_checkpoint_file = tf.train.latest_checkpoint(self._model_filepath)
      checkpoint.restore(latest_checkpoint_file).expect_partial()
    fake_optimizer = optimization.create_optimizer(0, 0, 0)
    model.compile(loss=_get_fake_loss_fn(), optimizer=fake_optimizer)
    self.tagging_model = model

  def _predict_and_realize_batch(
      self,
      source_sentences,
      confidence_scores_batch,
      detailed_inference_results=None,
  ):
    """Run tagging inference on a batch and return the realizations."""
    batch_dictionaries, batch_list = self._convert_source_sentences_into_batch(
        source_sentences,
        confidence_scores_batch,
        detailed_inference_results=detailed_inference_results,
    )
    predictions = self._predict_batch(batch_list)
    return self._realize_tagging_batch(
        batch_dictionaries,
        predictions,
        detailed_inference_results=detailed_inference_results,
    )

  def _convert_source_sentences_into_batch(
      self,
      source_sentences,
      confidence_scores_batch,
      detailed_inference_results=None,
  ):
    """Converts source sentence into a batch."""
    batch_dictionaries = []
    assert len(source_sentences) == len(confidence_scores_batch), (
        f'3Source batch {str(len(source_sentences))} different from confidence '
        f'scores batch {str(len(confidence_scores_batch))}')
    for num_sent, source_sentence in enumerate(source_sentences):
      example = self._builder.build_redace_example(
          source_sentence,
          confidence_scores=confidence_scores_batch[num_sent],
          target='',
      )
      assert (example
              is not None), f'Tagging could not convert {source_sentence}.'

      dict_element = {
          'input_word_ids':
              np.array(example.features['input_ids']),
          'input_mask':
              np.array(example.features['input_mask']),
          'input_type_ids':
              np.array(example.features['segment_ids']),
          'input_confidence_scores':
              np.array(example.features['bucketed_confidence_scores']),
      }
      if detailed_inference_results is not None:
        detailed_inference_results[
            num_sent].input_tokens = example.debug_features['input_tokens']
      batch_dictionaries.append(dict_element)
    # Convert from a list of dictionaries to dictionary of lists.
    batch_list = ({
        k: np.array([dic[k] for dic in batch_dictionaries
                    ]) for k in batch_dictionaries[0]
    })
    return batch_dictionaries, batch_list

  def _predict_batch(self, source_batch):
    """Produce output from tensorflow model."""
    if self.tagging_model is None:
      self.load_model()
    prediction = self.tagging_model(source_batch, training=False)
    return prediction['redace_tagger'].numpy()

  def _realize_tagging_batch(self,
                             source_batch,
                             prediction_batch,
                             detailed_inference_results=None):
    """Produces the realized predictions for a batch from the tagging model."""
    realizations = []
    for i, (source,
            prediction) in enumerate(zip(source_batch, prediction_batch)):
      end_index = sum(source['input_mask']) - 1

      tag_logits = prediction
      tag_logits = np.exp(tag_logits)
      tag_probabilities = (np.max(tag_logits, axis=-1)) / (
          np.sum(tag_logits, axis=-1))
      self._inverse_label_map = cast(Mapping[int, str], self._inverse_label_map)
      tags = [
          self._inverse_label_map[tag_id]
          for tag_id in np.argmax(tag_logits, axis=-1)
      ]
      if detailed_inference_results is not None:
        detailed_inference_results[i].tag_probabilities = tag_probabilities
        detailed_inference_results[i].tags = tags

      realization = self._builder.tokenizer.convert_ids_to_tokens(
          source['input_word_ids'][:end_index + 1])
      realization = ' '.join(realization)
      realizations.append(realization)
    return realizations
