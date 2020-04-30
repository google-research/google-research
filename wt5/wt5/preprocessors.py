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

"""WT5 preprocessors."""
import tensorflow.compat.v1 as tf


def _explanation_targets(answer, explanations, prefix='explanation:'):
  # Add prefix before each explanation.
  return tf.strings.reduce_join(
      tf.concat([[answer], explanations], axis=0),
      separator=' %s ' % prefix)


def cos_e(
    dataset,
    prefix='explain cos_e',
    question_prefix='question:',
    choice_prefix='choice:',
    drop_explanations=False,
):
  """Convert the CoS-E dataset to a text-to-text dataset.

  The CoS-E dataset contains an example of the following format:

  {
    'question': 'Question with some context?',
    'choices': ['First', 'Second', 'Third'],
    'abstractive_explanation': 'Abstractive explanation.',
    'extractive_explanation': 'Not currently used.',
    'answer': 'First',
  }

  Without dropping explanations, this will transform to:
  {
      'inputs': 'explain cos_e question: Question with some context? '
                'choice: First choice: Second choice: Third',
      'targets': 'First explanation: Abstractive explanation'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    question_prefix: str, prefix for the cos_e question. Can be overridden to
        look like other tasks, i.e. for zero-shot evaluation.
    choice_prefix: str, prefix for each cos_e choice. Can be overridden to look
        like other tasks, i.e. for zero-shot evaluation.
    drop_explanations: bool, whether to drop the explanations from the target.

  Returns:
    a tf.data.Dataset
  """
  def my_fn(x):
    """Helper function to transform CoS-E dataset to inputs/targets."""

    choices_text = tf.strings.reduce_join(
        choice_prefix + ' ' + x['choices'], separator=' ')

    inputs = tf.strings.join(
        [prefix, question_prefix, x['question'], choices_text],
        separator=' ')
    targets = (
        x['answer'] if drop_explanations else
        _explanation_targets(x['answer'], [x['abstractive_explanation']]))

    return {'inputs': inputs, 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def esnli(
    dataset,
    prefix='explain nli',
    drop_explanations=False,
    add_choices=False,
):
  """Convert the e-SNLI dataset to a text-to-text dataset.

  The e-SNLI dataset contains example of the following format.

  {
    'premise': 'A person walks in the park.',
    'hypothesis': 'A person is present in the park.',
    'label': 0,
    'explanation_1': 'A person is walking in the park means they are in a park.'
  }

  If `drop_explanations` is False, this will be transformed to:
  {
    'inputs': 'explain nli hypothesis: A person is present in the park
               premise: A person walks in the park'
    'targets': 'entailment explanation: A person is walking in the park means
                they are in a park.'
  }

  The test and validation sets contain multiple explanations per example. The
  additional examples are added to the end of the targets separated by a semi
  colon. E.g.

  {
    'premise': 'A person walks in the park.',
    'hypothesis': 'A person is present in the park.',
    'label': 0,
    'explanation_1': 'A person is walking in the park means they are in a
                      park.',
    'explanation_2': 'Walking implies being present in the park.',
    'explanation_3': 'You have to be present to walk in a park.'
  }

  If `drop_explanations` is False, this will be transformed to:
  {
    'inputs': 'explain nli hypothesis: A person is present in the park
               premise: A person walks in the park'
    'targets': 'entailment explanation: A person is walking in the park means
                they are in a park. explanation: Walking implies being present
                in the park. explanation: You have to be present to walk in a
                park.'
  }

  If `drop_explanations` is True, the above example will be transformed to leave
  out the explanation. You will likely also want to change the `prefix` to
  be 'nli'. E.g.:
  {
    'inputs': 'nli hypothesis: A person is present in the park
               premise: A person walks in the park'
    'targets': 'entailment'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    drop_explanations: bool, whether to drop the explanations from the target.
    add_choices: bool, whether to add choices for each label, like cos_e.
  Returns:
    a tf.data.Dataset

  """
  def my_fn(x):
    """Helper function to transform e-Snli dataset to inputs/targets."""
    labels = ['entailment', 'neutral', 'contradiction']
    inputs = tf.strings.join(
        [prefix, 'hypothesis:', x['hypothesis'], 'premise:', x['premise']],
        separator=' ')
    if add_choices:
      inputs = tf.strings.join([
          inputs, 'choice: entailment choice: neutral choice: contradiction'],
                               separator=' ')

    class_label = tf.gather(labels, x['label'])

    if drop_explanations:
      targets = class_label
    else:
      explanations = [x.get('explanation_%d' % i, '') for i in range(1, 4)]
      explanations = tf.boolean_mask(
          explanations, tf.not_equal(explanations, ''))
      targets = _explanation_targets(class_label, explanations)

    return {'inputs': inputs, 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def imdb_reviews(
    dataset,
    prefix='sentiment',
    output_classes=('negative', 'positive')
    ):
  """Preprocessor to handle imdb movie reviews.

  Preprocessor converts an example from the IMDB movie review dataset to the
  text to text format. The following is an example from IMDB dataset.
  {
    'text': 'This is a bad movie. Skip it.'
    'label': 0,
  }

  The example will be transformed to the following format by the preprocessor:
  {
    'inputs': 'sentiment review: This is a bad movie. Skip it.'
    'targets': 'negative'
  }

  Examples with no label (-1) will have '<unk>' as their target.

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    output_classes: list of output classes in the input dataset. Defaults to
      ['negative', 'positive'] for the movie reviews dataset.

  Returns:
    a tf.data.Dataset

  """
  def my_fn(x):
    """Helper function to transform a rationale dataset to inputs/targets."""
    inputs = tf.strings.join([prefix + ':', x['text']], separator=' ')

    class_label = tf.cond(
        x['label'] > -1,
        lambda: tf.gather(output_classes, x['label']),
        lambda: '<unk>')

    return {'inputs': inputs, 'targets': class_label}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def amazon_reviews(
    dataset,
    prefix='sentiment',
    binary_output=True,
    ):
  """Preprocessor for Amazon reviews dataset.

  The Amazon reviews dataset contains the star rating, review headline, and
  review body amongst other features. This preprocessor converts these features
  into the text-to-text format.

  If binary_output is set to True, reviews are binarized based on
  https://arxiv.org/abs/1509.01626 which distributed the amazon polarity
  dataset. Reviews with 1,2 stars are negative and reviews with 4,5 stars are
  posative. Reviews with 3 stars are ignored.
  {
    data/review_headline: 'Great toy!',
    data/review_body: 'My daughter loved this toy.'
    data/star_rating: 5
  }

  This will be converted to:

  {
    input: 'explain sentiment review: Great toy! My daugher loved this toy.'
    output: 'positive'
  }

  If binary_output is set to False, the above example will be converted to:

  {
    input: 'explain sentiment review: Great toy! My daugher loved this toy.'
    output: '5'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    binary_output: boolean, whether the output will be a star rating or a
      binary rating.

  Returns:
    a tf.data.Dataset

  """
  def my_fn(x):
    """Helper function to transform a rationale dataset to inputs/targets."""
    input_label = tf.strings.join(['review', ':'], separator='')
    inputs = tf.strings.join([
        prefix, input_label, x['data']['review_headline'],
        x['data']['review_body']
    ],
                             separator=' ')

    star_rating = x['data']['star_rating']
    if binary_output:
      targets = tf.cond(tf.math.less(star_rating, tf.constant([3])),
                        lambda: 'negative', lambda: 'positive')
    else:
      targets = tf.strings.as_string(star_rating)

    return {'inputs': inputs, 'targets': targets}

  if binary_output:
    dataset = dataset.filter(
        lambda ex: tf.math.not_equal(ex['data']['star_rating'], 3))
  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def extractive_explanations(
    dataset,
    prefix='explain sentiment',
    input_feature='review',
    output_classes=('negative', 'positive'),
    drop_explanations=False
    ):
  """Preprocessor to handle extractive rationale prediction datasets.

  The preprocessor expects a dataset with the provided 'input_feature', a label,
  and a list of evidences. E.g. the movie rationale dataset consists of the
  following features.

  {
    review: 'This is a bad movie. Skip it.'
    label: 0,
    evidences: ['bad movie', 'Skip it']
  }

  The example will be transformed to the following format by the preprocessor:
  {
    inputs: 'explain sentiment review: This is a bad movie. Skip it.'
    targets: 'NEG because bad movie explanation: Skip it'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    input_feature: str, feature name in input dataset.
    output_classes: list of output classes in the input dataset. Defaults to
      ['negative', 'positive'] for the movie reviews dataset.
    drop_explanations: bool, whether or not to drop explanations.

  Returns:
    a tf.data.Dataset

  """

  if output_classes is None:
    output_classes = ['negative', 'positive']

  def my_fn(x):
    """Helper function to transform a rationale dataset to inputs/targets."""
    input_label = tf.strings.join([input_feature, ':'], separator='')
    inputs = tf.strings.join(
        [prefix, input_label, x[input_feature]], separator=' ')

    class_label = tf.gather(output_classes, x['label'])
    if drop_explanations:
      targets = class_label
    else:
      targets = _explanation_targets(class_label, x['evidences'])

    return {'inputs': inputs, 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def eraser_multi_rc(dataset,
                    prefix='explain multirc',
                    input_features=None,
                    explanation_separator='explanation:',
                    drop_explanations=False,):
  """Preprocessor to handle ERASER MultiRC dataset.

  The preprocessor expects a dataset with the provided `input_features`, a
  label, and a list of evidences. The eraser_multi_rc dataset consists of the
  following features.

  {
    'passage': 'This is a passage. It has sentences',
    'query_and_answer': 'Is this a passage? || yes',
    'label': 'True',
    'evidences': ['This is a passage', 'It has sentences']
  }

  The example will be transformed to the following format by the preprocessor:
  {
    'inputs': 'explain multirc passage: This is a passage. It has sentences '
    'query': Is this a passage? answer: yes'
    'targets': 'True explanation: This is a passage. explanation: It has
                sentences'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, label to prepend to the inputs.
    input_features: list, feature name in input dataset.
    explanation_separator: str, separator string for the list of rationales.
    drop_explanations: bool, whether or not to drop explanations.

  Returns:
    a tf.data.Dataset

  """
  if not input_features:
    input_features = ['passage', 'query', 'answer']

  def my_fn(x):
    """Helper function to transform a eraser_multirc dataset to inputs/targets."""

    # Separate out query and answer components
    split_query_answer = tf.strings.split(x['query_and_answer'], '||').values
    x['query'] = tf.strings.strip(tf.slice(split_query_answer, [0], [1]))
    x['answer'] = tf.strings.strip(tf.slice(split_query_answer, [1], [1]))

    # Creating inputs
    inputs = prefix
    for input_feature in input_features:
      ip_feat_str = tf.strings.join([input_feature+':', x[input_feature]],
                                    separator=' ')
      inputs = tf.strings.join([inputs, ip_feat_str], ' ')

    # Creating targets
    class_label = tf.gather(['False', 'True'], x['label'])
    if drop_explanations:
      targets = class_label
    else:
      targets = _explanation_targets(
          class_label,
          x['evidences'],
          prefix=explanation_separator)

    return {'inputs': inputs, 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def hypothesis_prediction(
    dataset,
    label_names,
    prefix='hyp',
    ):
  """Convert any NLI dataset to predict hypothesis given premise and label.

  The function takes a NLI dataset with the following features:

  {
    'premise': 'A person walks in the park.',
    'hypothesis': 'A person is present in the park.',
    'label': 0,
  }

  Assuming the dataset has label_names = ['entailment', 'not_entailment'],
  it will be transformed to:

  {
    'inputs': 'hyp premise: A person walks in the park label: entailment'
    'targets': 'A person is present in the park.'
  }

  Args:
    dataset: a tf.data.Dataset to process.
    label_names: a list of label names corresponding to class index.
    prefix: str, prefix to prepend to the inputs.

  Returns:
    a tf.data.Dataset

  """
  def my_fn(x):
    inputs = tf.strings.join(
        [prefix, 'premise:', x['premise'], 'label:',
         tf.gather(label_names, x['label'])], separator=' ')

    targets = x['hypothesis']

    return {'inputs': inputs, 'targets': targets}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def mask_named_entities(
    dataset,
    prefix='nem',
    mask='_X_'):
  """Mask each named entity in the provided dataset.

  The function takes sentences with indices of named entities and for each named
  entity, outputs an example with it masked.

  Args:
    dataset: a tf.data.Dataset to process.
    prefix: str, prefix to prepend to the inputs.
    mask: str, the value to put in place of a masked named entity in the inputs.

  Returns:
    a tf.data.Dataset
  """
  def tile_sentence(ex):
    return {
        'sentence': tf.tile([ex['sentence']], tf.shape(ex['entities']['pos'])),
        'entity_pos': tf.cast(ex['entities']['pos'], tf.int32),
        'entity_len': tf.cast(ex['entities']['len'], tf.int32),
    }

  def nem_map(ex):
    return {
        'inputs':
            tf.strings.join([
                prefix, ': ',
                tf.strings.substr(ex['sentence'], 0, ex['entity_pos']),
                mask,
                tf.strings.substr(
                    ex['sentence'], ex['entity_pos'] + ex['entity_len'], -1)
            ]),
        'targets':
            tf.strings.substr(
                ex['sentence'], ex['entity_pos'], ex['entity_len'])
    }
  dataset = dataset.filter(lambda ex: tf.size(ex['entities']['pos']) > 0)
  dataset = dataset.map(
      tile_sentence, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
  return dataset.map(nem_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
