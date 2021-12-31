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

"""Preprocessing for NarrativeQA data."""
import csv
import json
import math
import os
import re
import string
from typing import Any, Iterator, List, Optional, Set, Text, Tuple

from absl import logging
import apache_beam as beam
from apache_beam import metrics
import dataclasses
import nltk
import tensorflow.compat.v1 as tf

from readtwice.data_utils import beam_utils
from readtwice.data_utils import data_utils
from readtwice.data_utils import tokenization
from readtwice.models.narrative_qa import extractive_oracle


METRICS_NAMESPACE = 'read_it_twice.narrative_qa'
SAMPLE_NO_ANSWER_QUESTIONS = 100


@dataclasses.dataclass(frozen=True)
class Question(object):
  id: int
  question_id: Text
  value: Text
  tokenized: Text


@dataclasses.dataclass(frozen=True)
class EvidenceInfo(object):
  id: Text
  source: Text
  url: Text


@dataclasses.dataclass(frozen=True)
class Evidence(object):
  info: EvidenceInfo
  text: Text
  summary: Optional[Text]


@dataclasses.dataclass(frozen=True)
class Answer(object):
  """Class represents answer for the question."""
  values: List[Text]
  tokenized: List[Text]

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
    for alias in self.values + self.tokenized + other_answers:
      answers.append(self._alias_answer(alias))
      answers.append(self._alias_answer(alias, [',', '.']))
      answers.append(self._alias_answer(alias, ['-']))
      answers.append(self._alias_answer(alias, [',', '.', '-']))
      answers.append(self._alias_answer(alias, string.punctuation))
    answers = set(answers)
    # Filter out empty or all-whitespace strings
    answers = {answer for answer in answers if answer.strip()}
    return answers


@dataclasses.dataclass(frozen=True)
class QuestionAnswer(object):
  """Single record in TriviaQA dataset."""
  question: Question
  answer: Answer
  evidence_info: EvidenceInfo


class EnhancedJSONEncoder(json.JSONEncoder):

  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    return super().default(o)


@dataclasses.dataclass
class QuestionAnswerEvidence(object):
  question: Question
  evidence: Evidence
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


class ReadEvidenceOutput(object):
  SUCCESS = None
  NO_STAR_END_CONTENT = 'no_star_end_content'
  TOO_SHORT_CONTENT = 'too_short_content'


class MakeExampleOutput(object):
  SUCCESS = None
  SUCCESS_FILTERED_ANNOTATIONS = 'success_filtered_annotations'
  NO_ANSWER = 'no_answer'
  NO_ANSWER_TOKENIZED = 'no_answer_tokenized'
  NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS = 'no_answer_tokenized_filtered_annotations'
  TOO_MANY_ANSWERS = 'too_many_answers'


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace.

  This normalization is the same as for the TriviaQA dataset. However,
  it is NOT used during evaluation in case of NarrativeQA -- only for training
  purposes.

  Args:
    s: Text

  Returns:
    normalized text
  """

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def handle_punc(text):
    exclude = set(string.punctuation + ''.join([u'‘', u'’', u'´', u'`']))
    return ''.join(ch if ch not in exclude else ' ' for ch in text)

  def lower(text):
    return text.lower()

  def replace_underscore(text):
    return text.replace('_', ' ')

  return white_space_fix(
      remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def read_question_answer_csv(csv_path, documents_path,
                             data_split):
  """Read a CVS file into a list of QuestionAnswer objects."""
  id_to_evidence_info = {}
  with tf.io.gfile.GFile(documents_path) as f:
    reader = csv.DictReader(f)
    for datum in reader:
      if datum['set'] != data_split:
        continue
      assert datum['kind'] in ['movie', 'gutenberg'], datum['kind']
      document_id = datum['document_id']
      id_to_evidence_info[document_id] = EvidenceInfo(
          id=document_id, source=datum['kind'], url=datum['story_url'])
  logging.info('Read %d evidence info from file %s', len(id_to_evidence_info),
               documents_path)

  question_answers = []
  with tf.io.gfile.GFile(csv_path) as f:
    reader = csv.DictReader(f)
    for datum in reader:
      if datum['set'] != data_split:
        continue
      # Note that document IDs start from 1.
      # We keep 0 as an ID of an empty document
      question = Question(
          id=len(question_answers) + 1,
          question_id=datum['question'],
          value=datum['question'],
          tokenized=datum['question_tokenized'])
      answer = Answer(
          values=[datum['answer1'], datum['answer2']],
          tokenized=[datum['answer1_tokenized'], datum['answer2_tokenized']])

      question_answers.append(
          QuestionAnswer(question, answer,
                         id_to_evidence_info[datum['document_id']]))

  logging.info('Read %d questions for the data spit "%s" from file %s',
               len(question_answers), data_split, csv_path)
  return question_answers


def _gutenberg_simple_parse(raw_content):
  """Clean a project Gunteberg file content."""
  content = raw_content
  # *** START OF THIS PROJECT GUTENBERG EBOOK THE RED BADGE OF COURAGE ***
  starts = [
      '*** START OF THIS PROJECT GUTENBERG EBOOK',
      '***START OF THE PROJECT GUTENBERG EBOOK',
      '*** START OF THE PROJECT GUTENBERG EBOOK',
      '*END*THE SMALL PRINT! FOR PUBLIC DOMAIN',
      '*END THE SMALL PRINT! FOR PUBLIC DOMAIN',
      'This etext was prepared by',
      'This Etext was prepared by',
      'This etext was provided by',
      'This Etext prepared by ',
      '***START OF THIS PROJECT GUTENBERG EBOOK',
  ]
  # *** END OF THIS PROJECT GUTENBERG EBOOK THE RED BADGE OF COURAGE ***
  ends = [
      '*** END OF THIS PROJECT GUTENBERG EBOOK',
      '***END OF THE PROJECT GUTENBERG EBOOK',
      '*** END OF THE PROJECT GUTENBERG EBOOK',
      'End of Project Gutenberg Etext',
      'End of this Project Gutenberg Etext',
      'End of the Project Gutenberg Etext',
      'End of The Project Gutenberg Etext',
      'End of the Project Gutenberg etext',
      'End of Project Gutenberg\'s Etext of ',
      'END OF PROJECT GUTENBERG ETEXT OF ',
      '***END OF THIS PROJECT GUTENBERG EBOOK',
  ]

  has_start = any([s in content for s in starts])
  has_end = any([e in content for e in ends])

  if not has_start or not has_end:
    return None

  start_index = max([content.rfind(s) for s in starts])
  end_index = min([content.find(e) % len(content) for e in ends])

  # Strip the prefix: '*** START OF THIS PROJECT GUTENBERG EBOOK ***'
  _, content = content[start_index:end_index].split('\n', 1)
  return content


def _movie_simple_parse(raw_content):
  """Clean a movie script file content."""
  content = raw_content
  starts = [
      '<pre>',
  ]
  ends = [
      '</pre>',
  ]

  has_start = any([s in content for s in starts])
  has_end = any([e in content for e in ends])

  if not has_start or not has_end:
    return None

  start_index = max([content.find(s) for s in starts])
  end_index = min([content.rfind(e) for e in ends])
  content = content[start_index:end_index]
  # _, content = content[start_index:end_index].split('>', 1)

  content = re.sub('<[^>]+?>', '', content)  # remove tags
  return content


def load_story(path):
  """Load and decode a story from the file and."""
  with tf.io.gfile.GFile(path, 'rb') as f:
    raw_content = f.read()

  # file encoding
  charset_search = re.search(b'Character set encoding: ([-0-9a-zA-Z()]+)',
                             raw_content)
  if charset_search is None:
    charset_search = re.search(b'charset=([-0-9a-zA-Z()]+)', raw_content)
  charset = None
  if raw_content[:3] == b'\xef\xbb\xbf':
    raw_content = raw_content[3:]
    charset = 'utf-8'
  elif raw_content[:2] == b'\xfe\xff':
    raw_content = raw_content[3:]
    charset = 'utf-16'
  elif charset_search is None:
    charset = 'utf-8'
  else:
    charset = charset_search.groups()[0]
    assert charset is not None, path
    charset = charset.decode('utf-8')
  charset = charset.lower()
  if charset == 'utf':
    charset = 'utf-8'
  if charset not in ['utf-8', 'iso-8859-1', 'utf-16']:
    logging.warn('Uncommon charset "%s" for the file %s', charset, path)
  try:
    raw_content = raw_content.decode(charset)
  # pylint: disable=broad-except
  except Exception as e:
    logging.warn('Failed to decode file %s with charset "%s". Error: %s', path,
                 charset, e)
    raw_content = raw_content.decode(charset, errors='replace')
  # pylint: enable=broad-except
  return raw_content


class ReadEvidence(beam.DoFn):
  """Read evidence from directory."""

  def __init__(self, stories_dir, summaries_path):
    assert tf.io.gfile.isdir(stories_dir)
    self.stories_dir = stories_dir
    self.summaries_path = summaries_path
    if self.summaries_path is not None:
      assert tf.io.gfile.exists(summaries_path)

  def setup(self):
    self.summary = None
    if self.summaries_path is not None:
      self.summary = {}
      with tf.io.gfile.GFile(self.summaries_path) as f:
        reader = csv.DictReader(f)
        for datum in reader:
          self.summary[datum['document_id']] = datum['summary'].replace(
              '\n', ' ')

  def process(
      self,
      question_answer):
    path = os.path.join(self.stories_dir,
                        question_answer.evidence_info.id + '.content')
    raw_content = load_story(path)
    if question_answer.evidence_info.source == 'gutenberg':
      content = _gutenberg_simple_parse(raw_content)
    elif question_answer.evidence_info.source == 'movie':
      content = _movie_simple_parse(raw_content)
    else:
      raise ValueError(
          f'Unknown evidence source: {question_answer.evidence_info.source}.')

    if content is None:
      metrics.Metrics.counter(
          METRICS_NAMESPACE, 'read_evidence_status.no_start_end_content').inc()
      yield beam.pvalue.TaggedOutput(ReadEvidenceOutput.NO_STAR_END_CONTENT,
                                     path)
      return

    num_words = len(re.sub(r'\s+', ' ', content).split(' '))
    if num_words < 100:
      logging.error('Content is missing (less than 100 words) in file %s', path)
      metrics.Metrics.counter(METRICS_NAMESPACE,
                              'read_evidence_status.too_short_content').inc()
      yield beam.pvalue.TaggedOutput(ReadEvidenceOutput.TOO_SHORT_CONTENT, path)
      return

    metrics.Metrics.counter(METRICS_NAMESPACE,
                            'read_evidence_status.success').inc()
    if self.summary is not None:
      summary = self.summary[question_answer.evidence_info.id]
    else:
      summary = None
    yield QuestionAnswerEvidence(
        question=question_answer.question,
        evidence=Evidence(
            info=question_answer.evidence_info, text=content, summary=summary),
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

  def __init__(self, spm_model_path,
               num_blocks_per_example, block_overlap_length,
               block_length, max_num_annotations_per_block,
               padding_token_id, cls_token_id, sep_token_id,
               generate_answers, generate_summaries,
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
    self.generate_summaries = generate_summaries
    self.extractive_oracle = extractive_oracle.ExtractiveOracle(
        min_roughe_l_score=min_rouge_l_oracle_score,
        top_percentile=0.9,
        top_k=100)
    self.nltk_data_path = nltk_data_path
    nltk.data.path.append(self.nltk_data_path)

  def setup(self):
    nltk.data.path.append(self.nltk_data_path)
    self.tokenizer = tokenization.FullTokenizer(
        spm_model_file=self.spm_model_path)
    self.nltk_tokenizer = nltk.TreebankWordTokenizer()
    self.nltk_pos_types = {'PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'}
    # self.spacy_annotator = spacy.load('en_core_web_sm')

  def process(
      self, question_answer_evidence):
    metrics.Metrics.counter(METRICS_NAMESPACE, 'num_questions').inc()

    if self.generate_answers:
      oracle_answers = []
      for answer in question_answer_evidence.answer.values:
        oracle_answers.extend(
            self.extractive_oracle.find_approximate_answers(
                question_answer_evidence.evidence.text,
                answer,
                remove_all_stopwords_answers=True))
      metrics.Metrics.distribution(METRICS_NAMESPACE,
                                   'oracle_answers_per_question').update(
                                       len(oracle_answers))
      answer_set = question_answer_evidence.answer.make_answer_set(
          oracle_answers)
      normalized_answer_set = {
          normalize_answer(answer) for answer in answer_set
      }

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
                                 'num_sentences_per_question').update(
                                     len(sentences))
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

    tokenized_question = self._tokenize_text(
        question_answer_evidence.question.value)

    metrics.Metrics.distribution(METRICS_NAMESPACE, 'question_length').update(
        len(tokenized_question))

    filtered_annotations = []
    if self.generate_answers:
      for i, sentence in enumerate(tokenized_big_document.sentences):
        should_update, annotations, current_filtered_annotations = self._verify_annotations(
            sentence.annotations, normalized_answer_set)
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

    tokenized_big_document = data_utils.split_tokenized_sentences(
        tokenized_big_document,
        max_tokens=self._get_max_tokens_per_raw_doc(len(tokenized_question)),
        min_tokens_for_graceful_split=math.ceil(
            self._get_max_tokens_per_raw_doc(len(tokenized_question)) * 0.5))

    if self.generate_answers:
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
        if (num_annotations >
            self.max_num_annotations_per_block * approx_num_blocks):
          metrics.Metrics.counter(METRICS_NAMESPACE,
                                  'num_questions_with_too_many_answers').inc()
          yield beam.pvalue.TaggedOutput(MakeExampleOutput.TOO_MANY_ANSWERS,
                                         question_answer_evidence.to_json())

        yield beam.pvalue.TaggedOutput(
            MakeExampleOutput.SUCCESS_FILTERED_ANNOTATIONS,
            filtered_annotations)

    # message = question_answer_evidence.evidence.info.id
    tokenized_documents = data_utils.split_tokenized_documents(
        tokenized_big_document,
        max_tokens=self._get_max_tokens_per_raw_doc(len(tokenized_question)),
        max_sentences=None)

    metrics.Metrics.distribution(METRICS_NAMESPACE,
                                 'num_examples_per_question').update(
                                     len(tokenized_documents))
    if len(tokenized_documents) > 1:
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_too_large_evidence').inc()

    if self.generate_summaries:
      tokenized_summary = self._tokenize_text(
          question_answer_evidence.evidence.summary)
      if len(tokenized_summary) < self.block_length:
        tokenized_summary.extend([self.padding_token_id] *
                                 (self.block_length - len(tokenized_summary)))

    for tokenized_document in tokenized_documents:
      if self.generate_answers and tokenized_document.num_annotations(0) == 0:
        metrics.Metrics.counter(
            METRICS_NAMESPACE,
            'make_example_status.answer_not_found_splitted').inc()
        continue
      metrics.Metrics.counter(METRICS_NAMESPACE, 'num_examples').inc()
      tf_example = tokenized_document.to_tf_strided_large_example(
          overlap_length=self.block_overlap_length,
          block_length=self.block_length,
          padding_token_id=self.padding_token_id,
          prefix_token_ids=tokenized_question,
          max_num_annotations=self.max_num_annotations_per_block)
      if self.generate_summaries:
        num_blocks = len(
            tf_example.features.feature['block_ids'].int64_list.value)
        tf_example.features.feature[
            'summary_token_ids'].int64_list.value.extend(tokenized_summary *
                                                         num_blocks)
      yield tf_example

    metrics.Metrics.counter(METRICS_NAMESPACE,
                            'make_example_status.success').inc()

  def _split_into_sentences(self, evidence):
    current_line = ''
    re_combine_whitespace = re.compile(r'\s+')

    for line in evidence.text.strip().split('\n'):
      line_stripped = re_combine_whitespace.sub(' ', line).strip()
      if line_stripped:
        if current_line:
          current_line = current_line + ' ' + line_stripped
        else:
          current_line = line_stripped
      else:
        if current_line:
          yield current_line
        current_line = ''
    if current_line:
      yield current_line

  # TODO(urikz): Use spacy
  # def _annotate_entities(self, text: Text):
  #   annotations = []
  #   for entity in self.spacy_annotator(text):
  #     begin = entity.start_char
  #     end = entity.end_char - 1
  #     assert end >= begin, text
  #     assert text[begin:end + 1] == entity.text, text
  #     annotations.append(
  #         data_utils.Annotation(
  #             begin=begin, end=end, text=entity.text, label=None, type=1))
  #   annotations.sort(key=lambda a: (a.begin, a.end))
  #   sentence = data_utils.Sentence(text=text, annotations=annotations)
  #   sentence.strip_whitespaces()
  #   return sentence

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
          normalize_answer(annotation.text) not in answer_set):
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


def get_pipeline(input_qaps, input_documents, data_split,
                 stories_dir, summaries_path, spm_model_path,
                 num_blocks_per_example, block_overlap_length,
                 block_length, max_num_annotations_per_block,
                 padding_token_id, cls_token_id, sep_token_id,
                 generate_answers, generate_summaries,
                 min_rouge_l_oracle_score, nltk_data_path,
                 output_prefix, output_num_shards):
  """Makes a Beam pipeline."""

  def pipeline(root):

    question_answers = read_question_answer_csv(input_qaps, input_documents,
                                                data_split)
    question_answers = (
        root | 'CreateQuestionAnswers' >> beam.Create(question_answers)
        | 'ShuffleAfterCreatingQA' >> beam.Reshuffle())

    read_outputs = (
        question_answers
        | 'ReadEvidence' >> beam.ParDo(
            ReadEvidence(
                stories_dir=stories_dir,
                summaries_path=summaries_path)).with_outputs())

    _ = (
        read_outputs[ReadEvidenceOutput.NO_STAR_END_CONTENT]
        | 'ShuffleNoStarEndContent' >> beam.Reshuffle()
        | 'WriteNoStarEndContent' >> write_to_file_fn(
            output_prefix, 'no_star_end_content.txt'))

    _ = (
        read_outputs[ReadEvidenceOutput.TOO_SHORT_CONTENT]
        | 'ShuffleTooShortContent' >> beam.Reshuffle()
        | 'WriteTooShortContent' >> write_to_file_fn(output_prefix,
                                                     'too_short_content.txt'))

    outputs = (
        read_outputs[ReadEvidenceOutput.SUCCESS]
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
                generate_summaries=generate_summaries,
                min_rouge_l_oracle_score=min_rouge_l_oracle_score,
                nltk_data_path=nltk_data_path)).with_outputs())

    #  if generate_answers:
    #     # Write failure cases, when no answer was found
    #     _ = (
    #         outputs[MakeExampleOutput.NO_ANSWER]
    #         | 'ShuffleNoAnswer' >> beam.Reshuffle()
    #         | 'SampleNoAnswer' >>
    #         beam.combiners.Sample.FixedSizeGlobally(SAMPLE_NO_ANSWER_QUESTIONS)
    #         | 'WriteNoAnswer' >> write_to_file_fn('no_answer.jsonl'))

    #     _ = (
    #         outputs[MakeExampleOutput.NO_ANSWER_TOKENIZED]
    #         | 'ShuffleNoAnswerTokenized' >> beam.Reshuffle()
    #         | 'SampleNoAnswerTokenized' >>
    #         beam.combiners.Sample.FixedSizeGlobally(SAMPLE_NO_ANSWER_QUESTIONS)
    #         | 'WriteNoAnswerTokenized' >>
    #         write_to_file_fn('no_answer_tokenized.jsonl'))

    #     # Write annotations that have been filtered out after tokenization
    #     _ = (
    #         outputs[MakeExampleOutput.SUCCESS_FILTERED_ANNOTATIONS]
    #         | 'ShuffleSuccessFilteredAnnotations' >> beam.Reshuffle()
    #         | 'FlattenSuccessFilteredAnnotations' >> beam.FlatMap(lambda x: x)
    #         | 'WriteSuccessFilteredAnnotations' >>
    #         write_to_file_fn('success.filtered_annotations.txt'))

    # _ = (
    #     outputs[
    #         MakeExampleOutput.NO_ANSWER_TOKENIZED_FILTERED_ANNOTATIONS]
    #     |
    #     'ShuffleNoAnswerTokenizedFilteredAnnotations' >> beam.Reshuffle()
    #     | 'FlattenNoAnswerTokenizedFilteredAnnotations' >>
    #     beam.FlatMap(lambda x: x)
    #     | 'WriteNoAnswerTokenizedFilteredAnnotations' >>
    #     write_to_file_fn('no_answer_tokenized.filtered_annotations.txt'))

    #     # Write cases where the too many answer spans were found
    #     _ = (
    #         outputs[MakeExampleOutput.TOO_MANY_ANSWERS]
    #         | 'ShuffleTooManyAnswers' >> beam.Reshuffle()
    #         | ('WriteTooManyAnswers' >>
    #             write_to_file_fn('too_many_answers.jsonl')))

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
        prefix_length=num_blocks_per_example)

    if generate_summaries:
      max_lengths['summary_token_ids'] = max_tokens

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
