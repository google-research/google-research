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
"""T5 CBQA preprocessors."""
import tensorflow.compat.v1 as tf


def natural_questions_nocontext(
    dataset,
    prefix='nq question: ',
    drop_yes_no=False,
    max_tokens=None,
    max_answers=None,
    ):
  """Convert Natural Questions TFDS to open domain with multiple answers.

  Examples with no short or yes/no answers are filtered. All short and yes/no
  answers (even across annotations) are emitted, so the targets produced by this
  preprocessor are invalid in the case of multiple annotations. However, these
  should not occur in the train set.

  The function takes the natural_questions TFDS dataset an emits examples of the
  form:
  {
    'inputs': 'nq question: what are the names of the olsen twins'
    'targets': 'answer: Mary-Kate answer: Ashley'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    drop_yes_no: bool, whether to drop yes/no answers, keeping only short
      answers.
    max_tokens: (Optional) int, the maximum number of tokens (as specified by
      NQ) beyond which a short answer is dropped. None are dropped if set to
      `None`.
    max_answers: (Optional) int, the maximum number of answers to include in the
      targets. Will be selected deterministically from the beginning of the
      list. All answers are included if set to `None`.

  Returns:
    a tf.data.Dataset
  """
  def nq_map(ex):
    """Map Natural Questions example to text-to-text example."""
    inputs = prefix + ex['question']['text']

    annotations = ex['annotations']

    yes_no_labels = annotations['yes_no_answer']
    if drop_yes_no:
      yes_no_labels = -1 * tf.ones_like(yes_no_labels)
    yes_no_answers = tf.boolean_mask(yes_no_labels, yes_no_labels > -1)
    yes_no_answers = tf.where_v2(tf.equal(yes_no_answers, 1), 'yes', 'no')

    short_answers = annotations['short_answers']['text'].flat_values
    short_answer_starts = annotations['short_answers']['text'].row_starts()
    if max_tokens:
      start_tokens = annotations['short_answers']['start_token']
      end_tokens = annotations['short_answers']['end_token']
      dropped_answers = end_tokens - start_tokens > max_tokens
      short_answers = tf.boolean_mask(
          short_answers, tf.math.logical_not(dropped_answers.values))
      # Subtract dropped answers from row starts.
      row_drop_count = tf.math.reduce_sum(
          tf.cast(dropped_answers, tf.int64), axis=1)
      short_answer_starts -= tf.concat(
          [[0], tf.math.cumsum(row_drop_count[:-1])], axis=0)

    answers = tf.concat([yes_no_answers, short_answers], axis=0)
    if max_answers:
      answers = answers[:max_answers]
    targets = tf.strings.reduce_join('answer: ' + answers, separator=' ')

    return {
        'inputs': inputs,
        'targets': targets,
        'short_answers/values': short_answers,
        'short_answers/row_starts': short_answer_starts,
        'yes_no_answers': yes_no_labels
    }

  dataset = dataset.map(
      nq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset.filter(lambda ex: tf.strings.length(ex['targets']) > 0)


def natural_questions_open(
    dataset,
    prefix='nq question: '
    ):
  """Convert Natural Questions Open TFDS to examples.

  If there are multiple answers in the input, selects the first one as the
  target.

  The function takes the natural_question_open TFDS dataset and emits examples
  of the form:
  {
    'inputs': 'nq question: What are the names of the Olsen Twins?'
    'targets': 'Mary-Kate and Ashley',
    'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.

  Returns:
    a tf.data.Dataset
  """

  def nq_map(ex):
    """Map Natural Questions example to text-to-text example."""
    return {
        'inputs': prefix + ex['question'],
        'targets': ex['answer'][0],
        'answers': ex['answer'],
    }
  return dataset.map(nq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def trivia_qa_open(
    dataset,
    prefix='trivia_qa question: '
    ):
  """Convert TriviaQA dataset to open domain qa examples.

  The function takes the trivia_qa TFDS dataset and emits examples of the
  form:
  {
    'inputs': 'trivia_qa question: What are the names of the Olsen Twins?'
    'targets': 'Mary-Kate and Ashley',
    'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.

  Returns:
    a tf.data.Dataset
  """
  def tqa_map(ex):
    """Map TriviaQA example to text-to-text example."""
    return {
        'inputs': prefix + ex['question'],
        'targets': ex['answer']['value'],
        'answers': ex['answer']['aliases'],
    }

  return dataset.map(tqa_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def web_questions_open(
    dataset,
    prefix='wq question: '
    ):
  """Convert WebQuestions TFDS to open domain examples.

  If there are multiple answers in the input, selects the first one as the
  target.

  The function takes the web_questions TFDS dataset and emits examples of the
  form:
  {
    'inputs': 'wq question: What are the names of the Olsen Twins?'
    'targets': 'Mary-Kate and Ashley',
    'answers': ['Mary-Kate and Ashley', 'Ashley and Mary-Kate']
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.

  Returns:
    a tf.data.Dataset
  """

  def wq_map(ex):
    """Map WebQuestions example to text-to-text example."""
    return {
        'inputs': prefix + ex['question'],
        'targets': ex['answers'][0],
        'answers': ex['answers'],
    }
  return dataset.map(wq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def sample_answer(
    dataset,
    ):
  """Replaces target with sampled answer."""

  def samp_map(ex):
    answers = tf.random.shuffle(ex['answers'])
    return {
        'inputs': ex['inputs'],
        'targets': answers[0],
        'answers': answers,
    }
  return dataset.map(samp_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def mask_salient_spans(
    dataset,
    prefix='nem',
    mask='_X_'):
  """Mask each salient span in the provided dataset.

  The function takes sentences with indices of salient spanss and for each span,
  outputs an example with it masked.

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    mask: str, the value to put in place of a salient span in the inputs.

  Returns:
    a tf.data.Dataset
  """
  def tile_sentence(ex):
    return {
        'sentence': tf.tile([ex['text']], tf.shape(ex['spans']['start'])),
        'span_start': tf.cast(ex['spans']['start'], tf.int32),
        'span_limit': tf.cast(ex['spans']['limit'], tf.int32),
    }

  def ssm_map(ex):
    return {
        'inputs':
            tf.strings.join([
                prefix, ': ',
                tf.strings.substr(ex['sentence'], 0, ex['span_start']),
                mask,
                tf.strings.substr(ex['sentence'], ex['span_limit'], -1)
            ]),
        'targets':
            tf.strings.substr(
                ex['sentence'], ex['span_start'],
                ex['span_limit'] - ex['span_start'])
    }
  dataset = dataset.filter(lambda ex: tf.size(ex['spans']['start']) > 0)
  dataset = dataset.map(
      tile_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
  return dataset.map(ssm_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
