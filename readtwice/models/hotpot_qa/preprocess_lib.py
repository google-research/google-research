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

"""Preprocessing for HotpotQA data."""
import json
import os
import re
import string
from typing import (Any, Iterator, List, MutableSequence, Optional, Set, Text,
                    Tuple, Union)

from absl import logging
import apache_beam as beam
from apache_beam import metrics
import dataclasses
import nltk
import tensorflow.compat.v1 as tf

from readtwice.data_utils import beam_utils
from readtwice.data_utils import data_utils
from readtwice.data_utils import tokenization
from readtwice.models.hotpot_qa import evaluation

METRICS_NAMESPACE = 'read_it_twice.hotpot_qa'
SAMPLE_NO_ANSWER_QUESTIONS = 100


@dataclasses.dataclass(frozen=True)
class Question(object):
  id: int
  question_id: Text
  value: Text


@dataclasses.dataclass(frozen=True)
class Evidence(object):
  name: Text
  text: Text
  is_supporting_fact: bool


@dataclasses.dataclass(frozen=True)
class Answer(object):
  """Class represents answer for the question."""
  values: List[Text]

  def _alias_answer(self, answer, include=None):
    alias = answer.replace('_', ' ').lower()
    exclude = set(string.punctuation + ''.join(['‘', '’', '´', '`']))
    include = include or []
    alias = ''.join(
        c if c not in exclude or c in include else ' ' for c in alias)
    return ' '.join(alias.split()).strip()

  def make_answer_set(self, other_answers):
    """Apply less aggressive normalization to the answer aliases."""
    answers = []
    for alias in self.values + other_answers:
      answers.append(self._alias_answer(alias))
      answers.append(self._alias_answer(alias, [',', '.']))
      answers.append(self._alias_answer(alias, ['-']))
      answers.append(self._alias_answer(alias, [',', '.', '-']))
      answers.append(self._alias_answer(alias, string.punctuation))
    answers = set(answers)
    # Filter out empty or all-whitespace strings
    answers = {answer for answer in answers if answer.strip()}
    return answers


class EnhancedJSONEncoder(json.JSONEncoder):

  def default(self, o):
    assert dataclasses.is_dataclass(o)
    return dataclasses.asdict(o)


@dataclasses.dataclass(frozen=True)
class QuestionAnswerEvidence(object):
  question: Question
  evidence: List[Evidence]
  answer: Optional[Answer] = None

  def to_json(self):
    return json.dumps(self, cls=EnhancedJSONEncoder)


@dataclasses.dataclass
class FilteredAnnotation(object):
  question: Question
  answer: Answer
  annotation: Text
  sentence: Text

  def __str__(self):
    return '%s\t%s\t%s\t%s' % (self.question.question_id, ','.join(
        self.answer.values), self.annotation,
                               self.sentence.replace(
                                   tokenization.SPIECE_UNDERLINE, ' '))


def get_feature(feature_name,
                example):
  """Gets Tensorflow feature by name.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The Tensorflow feature with the given feature name in the example.

  Raises:
    ValueError: If the given feature name is not in the Tensorflow example.
  """
  if feature_name in example.features.feature:
    return example.features.feature[feature_name]
  else:
    raise ValueError('Feature name {} is not in the example {}'.format(
        feature_name, example))


def get_repeated_values(
    feature_name,
    example):
  """Gets the underlying repeated values of a feature by feature name.

  The return type depends on which oneof `kind` is populated for the feature.
  Whichever one is populated is returned.

  Args:
    feature_name: The name of the feature.
    example: A Tensorflow example.

  Returns:
    The underlying repeated values for the given feature name in the example.
    Modifying these repeated values will modify the example.

  Raises:
    ValueError: If the given feature name is not in the Tensorflow example or
      none of the oneof `kind` fields is populated.
  """
  feature = get_feature(feature_name, example)
  which_oneof = feature.WhichOneof('kind')
  if which_oneof is None:
    raise ValueError(
        'No field populated in oneof `kind` for feature name {} in example '
        '{}'.format(feature_name, example))
  return getattr(feature, which_oneof).value


class MakeExampleOutput(object):
  SUCCESS = None
  SUCCESS_FILTERED_ANNOTATIONS = 'success_filtered_annotations'
  NO_ANSWER = 'no_answer'
  NO_ANSWER_TOKENIZED = 'no_answer_tokenized'
  NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS = 'no_answer_tokenized_filtered_annotations'
  TOO_MANY_ANSWERS = 'too_many_answers'


def read_question_answer_json(csv_path):
  """Read a CVS file into a list of QuestionAnswer objects."""
  question_answers = []
  with tf.io.gfile.GFile(csv_path) as f:
    data = json.load(f)
    for datum in data:
      # Note that document IDs start from 1.
      # We keep 0 as an ID of an empty document
      question = Question(
          id=len(question_answers) + 1,
          question_id=datum['_id'],
          value=datum['question'])
      answer = Answer(values=[datum['answer']])
      evidence = []
      supporting_fact_titles = {x[0] for x in datum['supporting_facts']}
      assert len(supporting_fact_titles) == 2

      num_suppoering_facts = 0
      for context in datum['context']:
        is_supporting_fact = context[0] in supporting_fact_titles
        num_suppoering_facts += int(is_supporting_fact)
        text = ''.join(context[1])
        evidence.append(
            Evidence(
                name=context[0],
                text=text,
                is_supporting_fact=is_supporting_fact))
      assert num_suppoering_facts == 2
      question_answers.append(
          QuestionAnswerEvidence(question, evidence, answer))

  logging.info('Read %d questions from %s', len(question_answers), csv_path)
  return question_answers


# TODO(urikz): Potentially, we should filter out all intersecting
# annotations and try to pick only, for example, the largest ones
def find_answer_annotations(
    text, answer_set):
  """Find answer annotations."""
  annotations = []
  for answer in answer_set:
    # We use regex matching to search for the answer for two reasons:
    # (1) We want to ignore case (so `flags=re.IGNORECASE`)
    # (2) We want to the space and the hyphen to be treated as the same token.
    # Sometimes the answer is "TSR 2", but the actual text contains only "TSR-2"
    #
    # Note that we have to espace -- `re.escape(answer)` -- because the answer
    # can contain parentheses, etc.
    # Finally, to accommodate (2) we replace spaces ('\\ ' due to escaping)
    # with a group '[ -]'.
    answer_regex = re.compile(
        re.escape(answer).replace('\\ ', '[ -]'), flags=re.IGNORECASE)
    for match in re.finditer(answer_regex, text):
      if not answer.strip() or match.end() == 0:
        raise ValueError('Invalid answer string "%s" from answer set %s' %
                         (answer, str(answer_set)))
      annotations.append(
          data_utils.Annotation(
              begin=match.start(), end=match.end() - 1, text=match.group(0)))
  return sorted(annotations)


class MakeExamples(beam.DoFn):
  """Function to make tf.train.Examples."""

  def __init__(self, spm_model_path, num_blocks_per_example,
               block_overlap_length, block_length,
               max_num_annotations_per_block, padding_token_id,
               cls_token_id, sep_token_id, generate_answers,
               min_rouge_l_oracle_score, nltk_data_path):
    self.spm_model_path = spm_model_path
    self.num_blocks_per_example = num_blocks_per_example
    self.block_overlap_length = block_overlap_length
    self.block_length = block_length
    self.max_num_annotations_per_block = max_num_annotations_per_block
    self.padding_token_id = padding_token_id
    self.cls_token_id = cls_token_id
    self.sep_token_id = sep_token_id
    self.generate_answers = generate_answers
    self.nltk_data_path = nltk_data_path
    nltk.data.path.append(self.nltk_data_path)

  def setup(self):
    nltk.data.path.append(self.nltk_data_path)
    self.tokenizer = tokenization.FullTokenizer(self.spm_model_path)
    self.nltk_tokenizer = nltk.TreebankWordTokenizer()
    self.nltk_pos_types = {'PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'}

  def process(
      self, question_answer_evidence):
    metrics.Metrics.counter(METRICS_NAMESPACE, 'num_questions').inc()

    if self.generate_answers:
      oracle_answers = []
      answer_set = question_answer_evidence.answer.make_answer_set(
          oracle_answers)
      normalized_answer_set = {
          evaluation.normalize_answer(answer) for answer in answer_set
      }

    tokenized_question = self._tokenize_text(
        question_answer_evidence.question.value)

    metrics.Metrics.distribution(METRICS_NAMESPACE, 'question_length').update(
        len(tokenized_question))

    filtered_annotations = []
    tf_examples = []
    num_answer_annotations = 0
    num_answer_annotations_tokenized = 0
    num_entity_annotations = 0
    num_entity_annotations_tokenized = 0

    no_answer, yes_answer, yes_no_answer = False, False, False
    if question_answer_evidence.answer.values[0] == 'yes':
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_answer_type.yes').inc()
      yes_no_answer = True
      yes_answer = True
    if question_answer_evidence.answer.values[0] == 'no':
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_answer_type.no').inc()
      yes_no_answer = True
      no_answer = True
    if yes_no_answer:
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_answer_type.yes_no').inc()
    else:
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_answer_type.span').inc()

    for evidence in question_answer_evidence.evidence:
      sentence = self._split_into_sentences(evidence)
      sentence_obj = self._annotate_entities(sentence)
      metrics.Metrics.counter(METRICS_NAMESPACE, 'nltk_entities').inc(
          sentence_obj.num_annotations(1))

      if self.generate_answers and not yes_no_answer:
        annotations = find_answer_annotations(sentence_obj.text, answer_set)
        sentence_obj.annotations.extend(annotations)

      document = data_utils.BertDocument(
          sentences=[sentence_obj],
          document_id=question_answer_evidence.question.id)

      num_entity_annotations += document.num_annotations(1)
      num_answer_annotations += document.num_annotations(0)

      tokenized_document = data_utils.tokenize_document_for_bert(
          document, self.tokenizer)

      metrics.Metrics.distribution(METRICS_NAMESPACE,
                                   'tokenized_doc_length_per_paragraph').update(
                                       tokenized_document.num_tokens())

      if self.generate_answers and not yes_no_answer:
        assert len(tokenized_document.sentences) == 1
        (should_update, annotations,
         current_filtered_annotations) = self._verify_annotations(
             tokenized_document.sentences[0].annotations, normalized_answer_set)
        if should_update:
          tokenized_document.sentences[0].annotations = annotations
          # pylint: disable=g-complex-comprehension
          filtered_annotations.extend([
              FilteredAnnotation(
                  question=question_answer_evidence.question,
                  answer=question_answer_evidence.answer,
                  annotation=annotation,
                  sentence=''.join(tokenized_document.sentences[0].tokens))
              for annotation in current_filtered_annotations
          ])
          metrics.Metrics.counter(METRICS_NAMESPACE,
                                  'num_filtered_annotations').inc(
                                      len(current_filtered_annotations))

      num_entity_annotations_tokenized += tokenized_document.num_annotations(1)
      num_answer_annotations_tokenized += tokenized_document.num_annotations(0)

      tf_example = tokenized_document.to_tf_strided_large_example(
          overlap_length=self.block_overlap_length,
          block_length=self.block_length,
          padding_token_id=self.padding_token_id,
          prefix_token_ids=tokenized_question,
          max_num_annotations=self.max_num_annotations_per_block)

      if yes_answer:
        assert yes_no_answer
        assert not no_answer
        tf_example.features.feature['answer_type'].int64_list.value[:] = [1]
      elif no_answer:
        assert yes_no_answer
        assert not yes_answer
        tf_example.features.feature['answer_type'].int64_list.value[:] = [2]
      else:
        assert not yes_no_answer
        tf_example.features.feature['answer_type'].int64_list.value[:] = [0]

      if evidence.is_supporting_fact:
        tf_example.features.feature[
            'is_supporting_fact'].int64_list.value[:] = [1]
      else:
        tf_example.features.feature[
            'is_supporting_fact'].int64_list.value[:] = [0]

      tf_examples.append(tf_example)

    metrics.Metrics.distribution(METRICS_NAMESPACE,
                                 'num_paragraphs_per_question').update(
                                     len(tf_examples))
    metrics.Metrics.distribution(
        METRICS_NAMESPACE,
        'num_answer_annotations_per_question').update(num_answer_annotations)
    metrics.Metrics.distribution(
        METRICS_NAMESPACE,
        'num_entity_annotations_per_question').update(num_entity_annotations)

    if (self.generate_answers and not yes_no_answer and
        num_answer_annotations == 0):
      metrics.Metrics.counter(METRICS_NAMESPACE,
                              'make_example_status.no_answer').inc()
      yield beam.pvalue.TaggedOutput(MakeExampleOutput.NO_ANSWER,
                                     question_answer_evidence.to_json())
      return

    metrics.Metrics.distribution(
        METRICS_NAMESPACE,
        'num_answer_tokenize_annotations_per_question').update(
            num_answer_annotations_tokenized)
    metrics.Metrics.distribution(
        METRICS_NAMESPACE,
        'num_entity_tokenize_annotations_per_question').update(
            num_entity_annotations_tokenized)
    metrics.Metrics.distribution(METRICS_NAMESPACE,
                                 'num_filtered_annotations').update(
                                     len(filtered_annotations))

    if (self.generate_answers and not yes_no_answer and
        num_answer_annotations_tokenized == 0):
      metrics.Metrics.counter(
          METRICS_NAMESPACE,
          'make_example_status.no_answer_tokenized_annotations').inc()
      yield beam.pvalue.TaggedOutput(
          MakeExampleOutput.NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS,
          filtered_annotations)
      return

    yield beam.pvalue.TaggedOutput(
        MakeExampleOutput.SUCCESS_FILTERED_ANNOTATIONS, filtered_annotations)

    if len(tf_examples) != 10:
      metrics.Metrics.counter(METRICS_NAMESPACE,
                              'num_not_10_paragraphs_per_question').inc()

    tf_example = tf_examples[0]
    for i in range(1, len(tf_examples)):
      for name in tf_example.features.feature:
        repeated_values = get_repeated_values(name, tf_example)
        extension_values = list(get_repeated_values(name, tf_examples[i]))
        repeated_values.extend(extension_values)
    metrics.Metrics.counter(METRICS_NAMESPACE,
                            'make_example_status.success').inc()
    yield tf_example

  def _split_into_sentences(self, evidence):
    re_combine_whitespace = re.compile(r'\s+')
    return re_combine_whitespace.sub(' ', evidence.text).strip()

  def _annotate_entities(self, text):
    spans = list(self.nltk_tokenizer.span_tokenize(text))
    tokens = [text[b:e] for (b, e) in spans]
    annotations = []
    trees = nltk.ne_chunk(nltk.pos_tag(tokens))
    start_index = 0
    for tree in trees:
      if hasattr(tree, 'label'):
        children = [text for text, pos in tree]
        end_index = start_index + len(children)
        if tree.label() in self.nltk_pos_types:
          begin, _ = spans[start_index]
          _, end = spans[end_index - 1]
          surface_form = ' '.join(children)
          # There are edge cases when these are not equal.
          # For example, Diminish'd != Diminish 'd
          # assert text[begin:end] == surface_form, text
          surface_form = text[begin:end]
          annotations.append(
              data_utils.Annotation(
                  begin=begin, end=end - 1, text=surface_form, label=1, type=1))
        start_index = end_index
      else:
        start_index += 1
    annotations.sort(key=lambda a: (a.begin, a.end))
    sentence = data_utils.Sentence(text=text, annotations=annotations)
    sentence.strip_whitespaces()
    return sentence

  def _verify_annotations(
      self, annotations, answer_set
  ):
    should_update = False
    new_annotations = []
    filtered_annotations = set()
    for annotation in annotations:
      if (annotation.type == 0 and
          evaluation.normalize_answer(annotation.text) not in answer_set):
        filtered_annotations.add(annotation.text)
        should_update = True
      else:
        new_annotations.append(annotation)
    return should_update, new_annotations, filtered_annotations

  def _get_max_tokens_per_raw_doc(self, question_len):
    """Computes the maximal number of tokens per single document."""
    # The document will be split into several overlapping blocks --
    # see TokenizedBertDocument.to_tf_strided_large_example for details.
    # The first block will contain (`block_length` - `question_len`) tokens
    # Other blocks will contain fewer tokens because of the overlap --
    # (`block_length` - `question_len` - `block_overlap_length`) tokens.
    # Finally, `num_blocks_per_example` blocks will in total
    # have the following number of tokens:
    # (`block_length` - `question_len`) + (`num_blocks_per_example` - 1) *
    # (`block_length` - `question_len` - `block_overlap_length`) tokens =
    # = `num_blocks_per_example` * (`block_length` - `question_len`)
    # - (`num_blocks_per_example` - 1) * `block_overlap_length`
    return self.num_blocks_per_example * (self.block_length - question_len) - (
        self.num_blocks_per_example - 1) * self.block_overlap_length

  def _tokenize_text(self, question):
    tokens = self.tokenizer.tokenize(question)
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
    token_ids = token_ids[:self.block_length]
    return token_ids


def write_to_file_fn(output_prefix, filename):
  return beam.io.WriteToText(
      os.path.join(output_prefix + '.' + filename),
      append_trailing_newlines=True,
      shard_name_template='',  # To force unsharded output.
  )


def get_pipeline(input_file, spm_model_path,
                 num_blocks_per_example, block_overlap_length,
                 block_length, max_num_annotations_per_block,
                 padding_token_id, cls_token_id, sep_token_id,
                 generate_answers, min_rouge_l_oracle_score,
                 nltk_data_path, output_prefix,
                 output_num_shards):
  """Makes a Beam pipeline."""

  def pipeline(root):
    question_answers = read_question_answer_json(input_file)
    question_answers = (
        root | 'CreateQuestionAnswers' >> beam.Create(question_answers)
        | 'ShuffleAfterCreatingQA' >> beam.Reshuffle())

    outputs = (
        question_answers
        | 'ShuffleBeforeMakeExamples' >> beam.Reshuffle()
        | 'MakeExamples' >> beam.ParDo(
            MakeExamples(
                spm_model_path=spm_model_path,
                num_blocks_per_example=num_blocks_per_example,
                block_overlap_length=block_overlap_length,
                block_length=block_length,
                max_num_annotations_per_block=max_num_annotations_per_block,
                padding_token_id=padding_token_id,
                cls_token_id=cls_token_id,
                sep_token_id=sep_token_id,
                generate_answers=generate_answers,
                min_rouge_l_oracle_score=min_rouge_l_oracle_score,
                nltk_data_path=nltk_data_path)).with_outputs())

    #       if FLAGS.generate_answers:
    #         # Write failure cases, when no answer was found
    #         _ = (
    #             outputs[MakeExampleOutput.NO_ANSWER]
    #             | 'ShuffleNoAnswer' >> beam.Reshuffle()
    #             | 'SampleNoAnswer' >>
    #             beam.combiners.Sample.FixedSizeGlobally(
    #                 SAMPLE_NO_ANSWER_QUESTIONS)
    #             | 'WriteNoAnswer' >> write_to_file_fn('no_answer.jsonl'))

    #         _ = (
    #             outputs[MakeExampleOutput.NO_ANSWER_TOKENIZED]
    #             | 'ShuffleNoAnswerTokenized' >> beam.Reshuffle()
    #             | 'SampleNoAnswerTokenized' >>
    #             beam.combiners.Sample.FixedSizeGlobally(
    #                 SAMPLE_NO_ANSWER_QUESTIONS)
    #             | 'WriteNoAnswerTokenized' >>
    #             write_to_file_fn('no_answer_tokenized.jsonl'))

    #         # Write annotations that have been filtered out after tokenization
    #         _ = (
    #             outputs[MakeExampleOutput.SUCCESS_FILTERED_ANNOTATIONS]
    #             | 'ShuffleSuccessFilteredAnnotations' >> beam.Reshuffle()
    #             | ('FlattenSuccessFilteredAnnotations' >>
    #                 beam.FlatMap(lambda x: x))
    #             | 'WriteSuccessFilteredAnnotations' >>
    #             write_to_file_fn('success.filtered_annotations.txt'))

    #         _ = (
    #             outputs[MakeExampleOutput.NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS]
    #             | 'ShuffleNoAnswerTokenizedFilteredAnnotations' >>
    #             beam.Reshuffle()
    #             | 'FlattenNoAnswerTokenizedFilteredAnnotations' >>
    #             beam.FlatMap(lambda x: x)
    #             | 'WriteNoAnswerTokenizedFilteredAnnotations' >>
    #             write_to_file_fn('no_answer_tokenized.filtered_annotations.txt'))

    #         # Write cases where the too many answer spans were found
    #         _ = (
    #             outputs[MakeExampleOutput.TOO_MANY_ANSWERS]
    #             | 'ShuffleTooManyAnswers' >> beam.Reshuffle()
    #             | 'WriteTooManyAnswers' >>
    #             write_to_file_fn('too_many_answers.jsonl'))

    max_tokens = num_blocks_per_example * block_length
    max_num_annotations = num_blocks_per_example * max_num_annotations_per_block
    max_lengths = dict(
        token_ids=max_tokens,
        is_continuation=max_tokens,
        block_ids=num_blocks_per_example,
        answer_annotation_begins=max_num_annotations,
        answer_annotation_ends=max_num_annotations,
        answer_annotation_labels=max_num_annotations,
        entity_annotation_begins=max_num_annotations,
        entity_annotation_ends=max_num_annotations,
        entity_annotation_labels=max_num_annotations,
        prefix_length=num_blocks_per_example,
        answer_type=num_blocks_per_example,
        is_supporting_fact=num_blocks_per_example)

    example_packer = beam_utils.PriorityExamplePacker(
        priority_feature='token_ids',
        max_lengths=max_lengths,
        breakpoint_features=dict(),
        cumulative_features=[],
        min_packing_fraction=1.0,
        max_cache_len=num_blocks_per_example)
    _ = (
        outputs[MakeExampleOutput.SUCCESS]
        | 'ShuffleBeforePacking' >> beam.Reshuffle()
        | 'PackExamples' >> beam_utils.PackExamples(example_packer)
        | 'ShuffleAfterPacking' >> beam.Reshuffle()
        | 'WriteTfExamples' >> beam.io.WriteToTFRecord(
            os.path.join(output_prefix + '.tfrecord'),
            coder=beam.coders.ProtoCoder(tf.train.Example),
            num_shards=output_num_shards))

  return pipeline
