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

"""Preprocessing for TriviaQA data."""
import json
import os
import re
import string
from typing import Any, Iterator, List, Optional, Set, Text, Tuple

import apache_beam as beam
from apache_beam import metrics
import dataclasses
import nltk
import tensorflow.compat.v1 as tf

from readtwice.data_utils import beam_utils
from readtwice.data_utils import data_utils
from readtwice.data_utils import tokenization
from readtwice.models.trivia_qa import evaluation


METRICS_NAMESPACE = 'read_it_twice.trivia_qa'


@dataclasses.dataclass(frozen=True)
class Question(object):
  id: int
  question_id: Text
  value: Text


@dataclasses.dataclass(frozen=True)
class EvidenceInfo(object):
  id: Text
  source: Text
  title: Text


@dataclasses.dataclass(frozen=True)
class Evidence(object):
  info: EvidenceInfo
  text: Text


@dataclasses.dataclass(frozen=True)
class Answer(object):
  """Class represents answer for the question."""
  value: Text
  aliases: List[Text]
  normalized_aliases: List[Text]

  def _alias_answer(self, answer, include=None):
    alias = answer.replace('_', ' ').lower()
    exclude = set(string.punctuation + ''.join(['‘', '’', '´', '`']))
    include = include or []
    alias = ''.join(
        c if c not in exclude or c in include else ' ' for c in alias)
    return ' '.join(alias.split()).strip()

  def make_answer_set(self):
    """Apply less aggressive normalization to the answer aliases."""
    answers = []
    for alias in [self.value] + self.aliases:
      answers.append(self._alias_answer(alias))
      answers.append(self._alias_answer(alias, [',', '.']))
      answers.append(self._alias_answer(alias, ['-']))
      answers.append(self._alias_answer(alias, [',', '.', '-']))
      answers.append(self._alias_answer(alias, string.punctuation))
    answers = set(answers + self.normalized_aliases)
    # Filter out empty or all-whitespace strings
    answers = {answer for answer in answers if answer.strip()}
    return answers


@dataclasses.dataclass(frozen=True)
class QuestionAnswer(object):
  """Single record in TriviaQA dataset."""
  question: Question
  evidence_info: List[EvidenceInfo]
  answer: Optional[Answer] = None

  @classmethod
  def from_dict(cls, idx, datum):
    """Create `QuestionAnswer` object from a dictionary."""
    question = Question(
        id=idx, question_id=datum['QuestionId'], value=datum['Question'])
    if 'Answer' in datum:
      answer = Answer(
          value=datum['Answer']['Value'],
          aliases=datum['Answer']['Aliases'],
          normalized_aliases=datum['Answer']['NormalizedAliases'])
    else:
      answer = None
    evidence_info = []
    for key in ['EntityPages', 'SearchResults']:
      for document in datum.get(key, []):
        evidence_info.append(
            EvidenceInfo(
                id=document['Filename'], title=document['Title'], source=key))
    return cls(question=question, evidence_info=evidence_info, answer=answer)


class EnhancedJSONEncoder(json.JSONEncoder):

  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    return super().default(o)


@dataclasses.dataclass
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
    return '%s\t%s\t%s\t%s' % (
        self.question.question_id, self.answer.value, self.annotation,
        self.sentence.replace(tokenization.SPIECE_UNDERLINE, ' '))


class MakeExampleOutput(object):
  SUCCESS = None
  SUCCESS_FILTERED_ANNOTATIONS = 'success_filtered_annotations'
  NO_ANSWER = 'no_answer'
  NO_ANSWER_TOKENIZED = 'no_answer_tokenized'
  NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS = 'no_answer_tokenized_filtered_annotations'
  TOO_MANY_ANSWERS = 'too_many_answers'


def read_question_answer_json(json_path):
  with tf.io.gfile.GFile(json_path) as f:
    data = json.load(f)['Data']
  # Note that document IDs start from 1. We keep 0 as an ID of an empty document
  return [
      QuestionAnswer.from_dict(idx + 1, datum) for idx, datum in enumerate(data)
  ]


class ReadEvidence(beam.DoFn):
  """Read evidence from Wikipedia and/or Web files."""

  def __init__(self, wikipedia_dir, web_dir):
    self.wikipedia_dir = wikipedia_dir
    self.web_dir = web_dir

  def process(
      self,
      question_answer):
    evidence = []
    for info in question_answer.evidence_info:
      if info.source == 'EntityPages':
        evidence_path = os.path.join(self.wikipedia_dir, info.id)
      elif info.source == 'SearchResult':
        evidence_path = os.path.join(self.web_dir, info.id)
      else:
        raise ValueError(f'Unknown evidence source: {info.source}.')
      with tf.io.gfile.GFile(evidence_path, 'rb') as f:
        text = f.read().decode('utf-8')
      evidence.append(Evidence(info=info, text=text))
    if not evidence:
      raise ValueError('Question %s does not have evidence.' %
                       str(question_answer))

    metrics.Metrics.counter(METRICS_NAMESPACE, 'ReadEvidence.questions').inc()
    metrics.Metrics.distribution(METRICS_NAMESPACE,
                                 'ReadEvidence.num_evidence').update(
                                     len(evidence))
    yield QuestionAnswerEvidence(
        question=question_answer.question,
        evidence=evidence,
        answer=question_answer.answer)


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
               nltk_data_path):
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
    self.tokenizer = tokenization.FullTokenizer(
        spm_model_file=self.spm_model_path)
    self.nltk_tokenizer = nltk.TreebankWordTokenizer()
    self.nltk_pos_types = {'PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'}

  def process(
      self, question_answer_evidence):
    metrics.Metrics.counter(METRICS_NAMESPACE, 'num_questions').inc()

    if self.generate_answers:
      answer_set = question_answer_evidence.answer.make_answer_set()

    sentences = []
    for sentence in self._split_into_sentences(
        question_answer_evidence.evidence):
      sentence_obj = self._annotate_entities(sentence)
      metrics.Metrics.counter(METRICS_NAMESPACE, 'nltk_entities').inc(
          sentence_obj.num_annotations(1))
      if self.generate_answers:
        annotations = find_answer_annotations(sentence_obj.text, answer_set)
        sentence_obj.annotations.extend(annotations)

      sentences.append(sentence_obj)

    big_document = data_utils.BertDocument(
        sentences=sentences, document_id=question_answer_evidence.question.id)
    metrics.Metrics.distribution(METRICS_NAMESPACE,
                                 'doc_length_per_question').update(
                                     big_document.num_characters())

    if self.generate_answers:
      num_annotations = big_document.num_annotations(0)
      metrics.Metrics.distribution(
          METRICS_NAMESPACE,
          'num_annotations_per_question').update(num_annotations)
      if num_annotations == 0:
        metrics.Metrics.counter(
            METRICS_NAMESPACE,
            'make_example_status.answer_span_not_found').inc()
        yield beam.pvalue.TaggedOutput(MakeExampleOutput.NO_ANSWER,
                                       question_answer_evidence.to_json())
        return

    tokenized_big_document = data_utils.tokenize_document_for_bert(
        big_document, self.tokenizer)

    metrics.Metrics.distribution(METRICS_NAMESPACE,
                                 'tokenized_doc_length_per_question').update(
                                     tokenized_big_document.num_tokens())

    tokenized_question = self._tokenize_question(
        question_answer_evidence.question.value)

    metrics.Metrics.distribution(METRICS_NAMESPACE, 'question_length').update(
        len(tokenized_question))

    filtered_annotations = []
    if self.generate_answers:
      for i, sentence in enumerate(tokenized_big_document.sentences):
        (should_update, annotations,
         current_filtered_annotations) = self._verify_annotations(
             sentence.annotations, answer_set)
        if should_update:
          tokenized_big_document.sentences[i].annotations = annotations

          # pylint: disable=g-complex-comprehension
          filtered_annotations.extend([
              FilteredAnnotation(
                  question=question_answer_evidence.question,
                  answer=question_answer_evidence.answer,
                  annotation=annotation,
                  sentence=''.join(sentence.tokens))
              for annotation in current_filtered_annotations
          ])
          metrics.Metrics.counter(METRICS_NAMESPACE,
                                  'num_filtered_annotations').inc(
                                      len(current_filtered_annotations))

      num_annotations = tokenized_big_document.num_annotations(0)
      metrics.Metrics.distribution(
          METRICS_NAMESPACE,
          'num_annotations_tokenized_per_question').update(num_annotations)
      if num_annotations == 0:
        metrics.Metrics.counter(
            METRICS_NAMESPACE,
            'make_example_status.answer_not_found_tokenized').inc()
        yield beam.pvalue.TaggedOutput(MakeExampleOutput.NO_ANSWER_TOKENIZED,
                                       question_answer_evidence.to_json())
        yield beam.pvalue.TaggedOutput(
            MakeExampleOutput.NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS,
            filtered_annotations)
        return
      else:
        approx_num_blocks = (
            tokenized_big_document.num_tokens() /
            (self.block_length - self.block_overlap_length -
             len(tokenized_question)))
        if num_annotations > self.max_num_annotations_per_block * approx_num_blocks:
          metrics.Metrics.counter(METRICS_NAMESPACE,
                                  'num_questions_with_too_many_answers').inc()
          yield beam.pvalue.TaggedOutput(MakeExampleOutput.TOO_MANY_ANSWERS,
                                         question_answer_evidence.to_json())

        yield beam.pvalue.TaggedOutput(
            MakeExampleOutput.SUCCESS_FILTERED_ANNOTATIONS,
            filtered_annotations)

    tokenized_documents = data_utils.split_tokenized_documents(
        tokenized_big_document,
        max_tokens=self._get_max_tokens_per_raw_doc(len(tokenized_question)),
        max_sentences=None)

    metrics.Metrics.distribution(METRICS_NAMESPACE,
                                 'num_examples_per_question').update(
                                     len(tokenized_documents))
    if len(tokenized_documents) > 1:
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_too_large_evidence').inc()

    for tokenized_document in tokenized_documents:
      if self.generate_answers and tokenized_document.num_annotations(0) == 0:
        metrics.Metrics.counter(
            METRICS_NAMESPACE,
            'make_example_status.answer_not_found_splitted').inc()
        continue
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_examples').inc()
      yield tokenized_document.to_tf_strided_large_example(
          overlap_length=self.block_overlap_length,
          block_length=self.block_length,
          padding_token_id=self.padding_token_id,
          prefix_token_ids=tokenized_question,
          max_num_annotations=self.max_num_annotations_per_block)

    metrics.Metrics.counter(METRICS_NAMESPACE,
                            'make_example_status.success').inc()

  def _split_into_sentences(self, evidences):
    for evidence in evidences:
      for line in evidence.text.strip().split('\n'):
        line_stripped = line.strip()
        if line_stripped:
          yield line_stripped

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

  def _tokenize_question(self, question):
    tokens = self.tokenizer.tokenize(question)
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    return [self.cls_token_id] + token_ids + [self.sep_token_id]


def write_to_file_fn(output_prefix, filename):
  return beam.io.WriteToText(
      os.path.join(output_prefix + '.' + filename),
      append_trailing_newlines=True,
      shard_name_template='',  # To force unsharded output.
  )


def get_pipeline(input_file, wikipedia_dir, web_dir,
                 spm_model_path, num_blocks_per_example,
                 block_overlap_length, block_length,
                 max_num_annotations_per_block, padding_token_id,
                 cls_token_id, sep_token_id, generate_answers,
                 nltk_data_path, output_prefix,
                 output_num_shards):
  """Makes a Beam pipeline."""

  def pipeline(root):
    question_answers = read_question_answer_json(input_file)
    question_answers = (
        root | 'CreateQuestionAnswers' >> beam.Create(question_answers))

    outputs = (
        question_answers
        | 'ReadEvidence' >> beam.ParDo(
            ReadEvidence(wikipedia_dir=wikipedia_dir, web_dir=web_dir))
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
                nltk_data_path=nltk_data_path)).with_outputs())

    if generate_answers:

      # Write failure cases, when no answer was found
      _ = (
          outputs[MakeExampleOutput.NO_ANSWER]
          |
          'WriteNoAnswer' >> write_to_file_fn(output_prefix, 'no_answer.jsonl'))

      _ = (
          outputs[MakeExampleOutput.NO_ANSWER_TOKENIZED]
          | 'WriteNoAnswerTokenized' >> write_to_file_fn(
              output_prefix, 'no_answer_tokenized.jsonl'))

      # Write annotations that have been filtered out after tokenization
      _ = (
          outputs[MakeExampleOutput.SUCCESS_FILTERED_ANNOTATIONS]
          | 'FlattenSuccessFilteredAnnotations' >> beam.FlatMap(lambda x: x)
          | 'WriteSuccessFilteredAnnotations' >> write_to_file_fn(
              output_prefix, 'success.filtered_annotations.txt'))

      _ = (
          outputs[MakeExampleOutput.NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS]
          | 'FlattenNoAnswerTokenizedFilteredAnnotations' >>
          beam.FlatMap(lambda x: x)
          | 'WriteNoAnswerTokenizedFilteredAnnotations' >> write_to_file_fn(
              output_prefix, 'no_answer_tokenized.filtered_annotations.txt'))

      # Write cases where the too many answer spans were found
      _ = (
          outputs[MakeExampleOutput.TOO_MANY_ANSWERS]
          | 'WriteTooManyAnswers' >> write_to_file_fn(output_prefix,
                                                      'too_many_answers.jsonl'))

    max_tokens = num_blocks_per_example * block_length
    max_num_annotations = num_blocks_per_example * max_num_annotations_per_block
    example_packer = beam_utils.PriorityExamplePacker(
        priority_feature='token_ids',
        max_lengths=dict(
            token_ids=max_tokens,
            is_continuation=max_tokens,
            block_ids=num_blocks_per_example,
            answer_annotation_begins=max_num_annotations,
            answer_annotation_ends=max_num_annotations,
            answer_annotation_labels=max_num_annotations,
            entity_annotation_begins=max_num_annotations,
            entity_annotation_ends=max_num_annotations,
            entity_annotation_labels=max_num_annotations,
            prefix_length=num_blocks_per_example),
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
