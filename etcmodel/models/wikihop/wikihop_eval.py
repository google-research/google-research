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

"""Binary for WikiHop eval."""

import json
from typing import Any, Dict, List, Text, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

from etcmodel.models import tokenization
from etcmodel.models.wikihop import data_utils

tf.compat.v1.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

FLAGS = flags.FLAGS

# Populate these constants appropriately at the time of submisssion.
MODEL_PATH = "105/"
SPM_MODEL_VOCAB = "vocab_gpt.model"


class WikiHopInference(object):
  """WikiHop for inference / prediction using SavedModel."""

  def __init__(self, model_dir_path: Text, session_target: Text):
    """Loads the WikiHop from an exported `tf.SavedModel`.

    Args:
      model_dir_path: Path to the exported directory of the model.
      session_target: The session target.
    """
    self.sess = tf.Session(graph=tf.Graph(), target=session_target)

    # Loads the saved model (graph + variables) to the given session.
    graph_def = tf.saved_model.load(
        self.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        export_dir=model_dir_path)

    signature = graph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.input_tensor_name = signature.inputs["serialized_tf_example"].name
    self.logits_tensor_name = signature.outputs["logits"].name

  def predict(self,
              serialized_tf_examples: List[Text]) -> List[List[List[float]]]:
    """Retrieves logits for the given list of serialized tf examples.

    Args:
      serialized_tf_examples: Batched input serialized_tf_examples.

    Returns:
      A List[List[float]] representing the logits. Each entry i in the list
      corresponds to the result i-th serialized_tf_example.
    """
    feed_dict = {self.input_tensor_name: serialized_tf_examples}

    logits = self.sess.run([self.logits_tensor_name], feed_dict=feed_dict)
    return logits


def get_serialized_tf_example(wikihop_example: data_utils.WikiHopExample,
                              tokenizer: tokenization.FullTokenizer,
                              long_seq_len: int = 4096,
                              global_seq_len: int = 430,
                              max_num_sentences: int = 200) -> Text:
  """Returns serialized TF example from the given json example."""
  converter = data_utils.WikiHopTFExampleConverter(
      tokenizer=tokenizer,
      long_seq_len=long_seq_len,
      global_seq_len=global_seq_len,
      max_num_sentences=max_num_sentences)
  tf_example = converter.convert_single_example(example=wikihop_example)
  return tf_example.SerializeToString()


def get_predicted_answer(wikihop_example: data_utils.WikiHopExample,
                         logits: List[float],
                         global_seq_len: int = 430) -> Text:
  """Returns prediçted answer for the given example and its logits."""

  assert len(logits) == global_seq_len, (
      "Mismatch in logits len. Expected: {}, found: {}, logits are: {} ".format(
          global_seq_len, len(logits), logits))
  logits = logits[0:len(wikihop_example.candidate_answers)]
  max_label_index = np.argmax(logits)

  assert max_label_index >= 0 and (max_label_index < len(
      wikihop_example.candidate_answers))
  answer = wikihop_example.candidate_answers[max_label_index]
  answer = answer.lower().strip()
  return answer


def get_output_single_example(
    tokenizer: tokenization.FullTokenizer,
    wikihop_inference: WikiHopInference,
    json_obj: Dict[Text, Any],
    long_seq_len: int = 4096,
    global_seq_len: int = 430,
    max_num_sentences: int = 200) -> Tuple[Text, Text]:
  """Generates output for a single example."""
  wikihop_example = data_utils.WikiHopExample.from_json(single_example=json_obj)
  serialized_tf_example = get_serialized_tf_example(
      wikihop_example=wikihop_example,
      tokenizer=tokenizer,
      long_seq_len=long_seq_len,
      global_seq_len=global_seq_len,
      max_num_sentences=max_num_sentences)
  logits = wikihop_inference.predict([serialized_tf_example])[0][0]
  assert len(logits) == global_seq_len, (
      "Mismatch in0 logits len. Expected: {}, found: {} for example_id: {}. "
      "Actual logits are: {}".format(global_seq_len, len(logits),
                                     wikihop_example.example_id, logits))
  answer = get_predicted_answer(
      wikihop_example=wikihop_example,
      logits=logits,
      global_seq_len=global_seq_len)
  return (wikihop_example.example_id, answer)


def generate_eval_output_bulk(json_examples: List[Dict[Text, Any]],
                              model_dir_path: Text,
                              tokenizer: tokenization.FullTokenizer,
                              long_seq_len: int = 4096,
                              global_seq_len: int = 430,
                              max_num_sentences: int = 200,
                              batch_size: int = 4,
                              session_target: Text = "") -> Dict[Text, Any]:
  """Bulk mode inference."""
  serialized_tf_examples = []
  wikihop_examples = []
  output = {}
  for json_obj in json_examples:
    wikihop_example = data_utils.WikiHopExample.from_json(
        single_example=json_obj)
    wikihop_examples.append(wikihop_example)
    serialize_tf_example = get_serialized_tf_example(
        wikihop_example=wikihop_example,
        tokenizer=tokenizer,
        long_seq_len=long_seq_len,
        global_seq_len=global_seq_len,
        max_num_sentences=max_num_sentences)
    serialized_tf_examples.append(serialize_tf_example)

  wikihop_inference = WikiHopInference(
      model_dir_path=model_dir_path, session_target=session_target)

  index = 0
  num_examples = len(serialized_tf_examples)
  # Note that we getting "all" the serialized examples and then "batching"
  # only for prediction. The bottleneck is almost always going to be the
  # GPU anyway (for both memory and compute).
  while index < num_examples:
    predict_batch = serialized_tf_examples[index:min(index +
                                                     batch_size, num_examples)]
    batch_logits = wikihop_inference.predict(predict_batch)[0]
    for (offset, logits) in enumerate(batch_logits):
      answer = get_predicted_answer(
          wikihop_example=wikihop_examples[index + offset],
          logits=logits,
          global_seq_len=global_seq_len)
      output[wikihop_examples[index + offset].example_id] = answer
    index += batch_size

  return output


def generate_eval_output(json_examples: List[Dict[Text, Any]],
                         tokenizer: tokenization.FullTokenizer,
                         model_dir_path: Text,
                         long_seq_len: int = 4096,
                         global_seq_len: int = 430,
                         max_num_sentences: int = 200,
                         batch_inference: bool = False,
                         batch_size: int = 4,
                         session_target: Text = "") -> Dict[Text, Any]:
  """Generates output for the input json.

  Returns the dict output key'ed by the example_id, with the value being the
  answer string.

  Args:
    json_examples: List of examples loaded from json input file.
    tokenizer: The BERT or ALBERT tokenizer.
    model_dir_path: The path to the directory containing the SavedModel.
    long_seq_len: The long input.
    global_seq_len: The global input.
    max_num_sentences: The max num sentences to be used per example.
    batch_inference: If True, we batch together all the examples at once for
      faster inference. Given that there are only 1K test examples, we might be
      able to fit everything in memeroy (500K per example * 1K).
    batch_size: Number of examples to be batched in one to predict. Applicable
      only when `batch_inference` is set to True.
    session_target: The TF session target.

  Returns:
    Dict[Text, Text] key'ed by the example_id to the corresponding prediction
    answer.
  """
  output = {}

  if batch_inference:
    return generate_eval_output_bulk(
        json_examples=json_examples,
        model_dir_path=model_dir_path,
        tokenizer=tokenizer,
        long_seq_len=long_seq_len,
        global_seq_len=global_seq_len,
        max_num_sentences=max_num_sentences,
        batch_size=batch_size,
        session_target=session_target)

  wikihop_inference = WikiHopInference(
      model_dir_path=model_dir_path, session_target=session_target)

  for json_obj in json_examples:
    (example_id, label) = get_output_single_example(
        tokenizer=tokenizer,
        wikihop_inference=wikihop_inference,
        json_obj=json_obj,
        long_seq_len=long_seq_len,
        global_seq_len=global_seq_len,
        max_num_sentences=max_num_sentences)
    output[example_id] = label

  return output


def main(argv):
  if len(argv) != 3:
    raise tf.app.UsageError("Exactly two arguments expected.")
  input_json_filepath = argv[1].strip()
  output_json_filepath = argv[2].strip()
  tokenizer = tokenization.FullTokenizer(
      vocab_file=None, do_lower_case=None, spm_model_file=SPM_MODEL_VOCAB)

  with tf.gfile.Open(input_json_filepath, "r") as test_data:
    json_examples = json.load(test_data)

  predictions = generate_eval_output(
      tokenizer=tokenizer,
      json_examples=json_examples,
      model_dir_path=MODEL_PATH)
  with tf.gfile.GFile(output_json_filepath, "w") as output_writer:
    json.dump(predictions, output_writer)


if __name__ == "__main__":
  tf.app.run()
