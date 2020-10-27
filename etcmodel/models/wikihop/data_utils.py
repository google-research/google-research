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

"""Utils for data processing for WikiHop (QA model)."""

import collections
import heapq
import json
import random
from typing import Any, Dict, Iterable, List, Text, Tuple
import unicodedata

import attr
import nltk
import tensorflow.compat.v1 as tf

from etcmodel.models import tokenization

SENTENCE_GLOBAL_TOKEN_ID = 1
QUESTION_GLOBAL_TOKEN_ID = 3
CANDIDATE_GLOBAL_TOKEN_ID = 4
PARAGRAPH_GLOBAL_TOKEN_ID = 5

GLOBAL_TOKEN_TYPE_ID = 0
SENTENCE_TOKEN_TYPE_ID = 1
QUESTION_TOKEN_TYPE_ID = 3
CANDIDATE_TOKEN_TYPE_ID = 4
QUESTION_GLOBAL_TOKEN_TYPE_ID = 5
CANDIDATE_GLOBAL_TOKEN_TYPE_ID = 6
PARAGRAPH_GLOBAL_TOKEN_TYPE_ID = 7


def create_int_feature(values: Iterable[int]) -> tf.train.Feature:
  """Creates TensorFlow int features.

  Args:
    values: A sequence of integers.

  Returns:
    An entry of int tf.train.Feature.
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def create_float_feature(values: Iterable[float]) -> tf.train.Feature:
  """Creates TensorFlow float features.

  Args:
    values: A sequence of floats.

  Returns:
    An entry of float tf.train.Feature.
  """
  return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def create_string_feature(values: Iterable[Text]) -> tf.train.Feature:
  """Creates TensorFlow string features.

  Args:
    values: A sequence of unicode strings.

  Returns:
    An entry of int tf.train.Feature.
  """
  # Converts to `str` (in Python 2) and `bytes` (in Python 3) as
  # `tf.train.Feature` only takes bytes.
  values = [value.encode("utf-8") for value in values]

  feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
  return feature


class MaxHeap(object):
  """A simple custom MaxHeap wrapper class.

  By default, heapq library constructs only MinHeaps and doesn't take a
  comparator function. This class adds those two missing pieces.
  """

  # The elements of the heap are tuples with first entry being the index,
  # second one representing the actual value of the entry. We invert the values
  # to negative to enable MaxHeap. This is a bit weird, but is a workaround
  # for having to deal with some shortcomings of the heapq lib.
  def __init__(self, cmp=lambda x: -x[1]):
    self.cmp = cmp
    # We keep track of the index to resolve collisions when the cmp is same
    # for two or more entries.
    self.index = 0
    self.items = []

  def push(self, item: Tuple[int, int]):
    heapq.heappush(self.items, (self.cmp(item), self.index, item))
    self.index += 1

  def pop(self) -> Tuple[int, int]:
    return heapq.heappop(self.items)[2]

  def is_empty(self) -> bool:
    return not self.items


def _build_max_heap(num_entries_per_doc: List[int]) -> MaxHeap:
  max_heap = MaxHeap()
  for (index, num_sentences) in enumerate(num_entries_per_doc):
    max_heap.push((index, num_sentences))
  return max_heap


def truncate_doc_sentences(docs: List[Text], tokenizer: Any,
                           max_num_sentences: int,
                           max_num_tokens) -> List[List[Text]]:
  """Truncates sentences from the given docs until constraints are satisfied.

  This method truncates sentences across the docs so that the total number of
  sentences across the docs and the total number of tokens across the docs
  satisfy the given constraints. However, we prefer removing from the docs
  with highest number of sentences. The intuition is that the sentences in the
  docs with a small number of sentences are potentially more important than the
  ones with a lot of sentences. We do so in the following two steps:
    1) We try to remove a total of (num_total_sentences - max_num_sentences)
        iteratively by reducing one sentence at a time from the "current" doc
        with the highest number of sentences. We do this with the help of a
        MaxHeap.
    2) We also try to remove sentences until the total number of tokens
       (WordPieces) across the docs is no more than `long_seq_len`.

  Note:
  This method uses nltk tokenizer to get sentences from a document, uses
  the provided `tokenizer` (e.g., one of the BERT tokenizers) to tokenize
  the sentences. The provided `tokenizer` should support a tokenize(..) method
  that returns a list of tokens.

  Args:
    docs: The list of input docs. Each entry in the list represents the doc
      content.
    tokenizer: The tokenizer (e.g., BERT tokenizer) to be used to get the words
      (e.g., WordPieces) of a sentence. This tokenizer should support a
      tokenize(..) method that should return a list of tokens.
    max_num_sentences: The max number of sentences we'd like to keep across all
      the docs.
    max_num_tokens: The max number of tokens we'd like to keep across all the
      docs as determined by the given `tokenizer`.

  Returns:
    A List[List[]] representing sentences per doc. The outer list corresponds
    to docs, and the inner list corresponds to the sentences per doc.
  """

  # List[[]] for keeping track of sentences per doc.
  sentences_per_doc = []

  # List to track the number of sentences per doc. This will be useful in
  # identifying the number of sentences to be truncated (if required).
  num_sentences_per_doc = []

  def _get_num_tokens_and_sentences() -> Tuple[int, int]:
    """Returns number of total tokens (WordPieces) in the example."""
    num_total_tokens = 0
    num_total_sentences = 0

    # Count tokens in the docs
    for doc in docs:
      sentences = nltk.sent_tokenize(doc)
      sentences_per_doc.append(sentences)
      num_sentences_per_doc.append(len(sentences))
      num_total_sentences += len(sentences)
      for single_sentence in sentences:
        num_total_tokens += len(tokenizer.tokenize(single_sentence))

    return (num_total_tokens, num_total_sentences)

  # Also track the number of tokens (WordPieces) and sentences in the example.
  (num_total_tokens, num_total_sentences) = _get_num_tokens_and_sentences()

  if (num_total_sentences > max_num_sentences or
      num_total_tokens > max_num_tokens):
    # There are two steps we'd like to do here:
    # 1) We need to truncate a few sentences here. We try to remove
    #    a total of (num_total_sentences - max_num_sentences). We iteratively
    #    reduce one sentence at a time from the "current" doc with the highest
    #    number of sentences. We do that with the help of a MaxHeap.
    # 2) We also try to remove sentences until the total number of tokens
    #    (WordPieces) across the docs is no more than `long_seq_len`.
    max_heap = _build_max_heap(num_sentences_per_doc)
    num_sentences_to_truncate = num_total_sentences - max_num_sentences
    num_tokens_to_truncate = num_total_tokens - max_num_tokens

    while num_sentences_to_truncate > 0 or num_tokens_to_truncate > 0:
      (index, num_sentences) = max_heap.pop()

      if num_sentences > 0:
        # The sentence to be truncated would be the one at the end of the
        # sentences of the doc.
        truncated_sentence = sentences_per_doc[index][num_sentences - 1]
        num_tokens_to_truncate -= len(tokenizer.tokenize(truncated_sentence))
        num_sentences -= 1
        num_sentences_to_truncate -= 1
      max_heap.push((index, num_sentences if num_sentences > 0 else 0))

    # Update number of sentences to keep per doc.
    while not max_heap.is_empty():
      (index, keep_num_sentences) = max_heap.pop()
      sentences_per_doc[index] = sentences_per_doc[index][0:keep_num_sentences]

  return sentences_per_doc


def _is_whitespace(char):
  """Checks whether `char` is a whitespace character."""
  # \t, \n, and \r are technically control characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def find_candidate_mentions(
    input_text: Text,
    candidate: Text,
    tokenizer: tokenization.FullTokenizer,
    offset=0) -> Tuple[List[Text], List[Tuple[int, int]]]:
  """Finds the candidate string mentions in the sentence post tokenization.

  Args:
    input_text: The input for searching the candidate.
    candidate: The candidate to be searched for mentions in the input.
    tokenizer: The tokenizer to be used. For BERT tokenzier, we assume an
      uncased vocab.
    offset: Offset to be added to all the span values.

  Returns:
    A tuple of (input_tokens_list, list_of_candidate_spans_in_the_list)

  Example:
  input = "Thisss is Saaan Franciscooo"
  candidate = "saan franciscooo"

  Let's say we are using the ALBERT tokenizer. Tokenizing the input would give:
  ['▁This', 'ss', '▁is', '▁Sa', 'aan', '▁Franc', 'isc', 'ooo']

  We return the tokens of the sentence.
  We also return [(3, 7)] representing the only span where the candidate
  occurs in the tokenized sentence. Note that the span is  inclusive.
  """

  assert isinstance(tokenizer, tokenization.FullTokenizer)

  input_lower = input_text.lower()
  candidate_lower = candidate.lower()
  tokens = tokenizer.tokenize(input_text)
  if (not candidate_lower or not input_lower or
      candidate_lower not in input_lower):
    return (tokens, [])

  if isinstance(tokenizer, tokenization.FullTokenizer):
    # We assume a tokenizer with lower cased vocab here. We do a simple
    # substring match of the candidate tokens to the input text tokens.
    if (tokenizer.sp_model is None and
        not tokenizer.basic_tokenizer.do_lower_case):
      raise ValueError("BERT tokenizer should be lower cased.")
    candidate_tokens = tokenizer.tokenize(candidate.lower())
    candidate_len = len(candidate_tokens)
    candidate_spans = []
    for i in range(0, len(tokens)):
      if i + candidate_len <= len(tokens):
        if tokens[i:i + candidate_len] == candidate_tokens:
          candidate_spans.append((offset + i, offset + i + candidate_len - 1))
    return (tokens, candidate_spans)

  # Now that we know the candidate is present in the input_text, we do a
  # best effort matching.
  spiece_underline = tokenization.SPIECE_UNDERLINE.decode("utf-8")
  char_index_to_token_index = collections.OrderedDict()
  i = 0
  for (j, token) in enumerate(tokens):
    k = 0
    if token.startswith(spiece_underline):
      k += 1
    for c in token[k:len(token)]:
      c = c.lower()
      # Most chars, in general, other than the special_token in tokens have a
      # corresponding mapping in the input_text.
      while i < len(input_lower) and _is_whitespace(input_lower[i]):
        # To handle cases like a space etc in the input_text. Spaces, tabs etc.
        # generally don't appear in the tokens.
        char_index_to_token_index[i] = j
        i += 1
      if _is_whitespace(c):
        # This shouldn't generally happen - ALBERT tokenizer collapses
        # whitespaces.
        continue
      if c != input_lower[i]:
        # Tokenizer probably has extra characters for this token.
        continue
      if i < len(input_lower):
        assert c == input_lower[i]
        char_index_to_token_index[i] = j
        i += 1

  if i != len(input_text):
    # Our best effort matching chars to tokens failed. As a fallback, we will
    # just match the given candidate with the entire input_text and return.
    # Because we know that the candidate is already present in the input_text,
    # it's better to assign the candidate to the entire input_text (which in
    # our case is a sentence), rather than dropping it altogether.
    return (tokens, [(offset, offset + len(tokens) - 1)])

  # We now have matched every char in the input to its corresponding token
  # index successfully.
  candidate_spans = []
  cand_len = len(candidate_lower)
  # Using re.finditer for substring match seems to be throwing a weird python
  # error -- "nothing to repeat at position 0", in some cases. So doing a
  # brute force substring match.
  for start in range(0, len(input_lower)):
    if (start + cand_len <= len(input_lower) and
        input_lower[start:start + cand_len] == candidate_lower):
      end = start + cand_len - 1
      assert start in char_index_to_token_index, (
          "no mapping found for index %d for candidate %s ", start, candidate)
      assert end in char_index_to_token_index, (
          "no mapping found for index %d for candidate %s ", end, candidate)
      token_span_start = char_index_to_token_index[start]
      token_span_end = char_index_to_token_index[end]
      candidate_spans.append(
          (offset + token_span_start, offset + token_span_end))

  return (tokens, candidate_spans)


@attr.s
class InputFeatures(object):
  """The feautres for ETC QA model."""
  # Context features
  # Long token ids with format question paragraph1 paragraph2 ...
  long_token_ids = attr.ib(factory=list, type=List[int])
  # The sentence ids for the long tokens. Each id `i` corresponds to a sentence
  # global token, `global_token_ids[i]`. Each question token has a
  # unique sentence id.
  long_sentence_ids = attr.ib(factory=list, type=List[int])
  # The paragraph ids for the long tokens. Each id `i` corresponds to a
  # paragraph global token, `global_token_ids[i]`. Question tokens don't
  # correspond to any paragraph global tokens.
  long_paragraph_ids = attr.ib(factory=list, type=List[int])
  # Ending breakpoints separating question and paragraphs long tokens into
  # different segments, which are not attended by local attentions.
  long_paragraph_breakpoints = attr.ib(factory=list, type=List[int])
  # The token type ids for long tokens. With default values in `InputConfig`.
  long_token_type_ids = attr.ib(factory=list, type=List[int])
  # The global token ids. With default values in `InputConfig`.
  global_token_ids = attr.ib(factory=list, type=List[int])
  # Ending breakpoints separating question and paragraphs global
  # tokens into different segments.
  global_paragraph_breakpoints = attr.ib(factory=list, type=List[int])
  # The token type ids for global tokens. With default values in `InputConfig`.
  global_token_type_ids = attr.ib(factory=list, type=List[int])
  # Flag to indicate whether this is a real / padding example. Padding examples
  # can be useful to pad up to a multiple of batch_size examples. This can
  # be helpful in cases where TPUs require fixed batch_size (esp. for eval /
  # predict). For training, it can help us so that we don't drop remainder
  # (num_examples % batch_size) examples.
  is_real_example = attr.ib(default=True)  # type: bool


@attr.s
class WikiHopInputFeatures(object):
  """A simple wrapper around InputFeatures for WikiHop.

  Includes candidate start and end positions in the global input.
  """
  input_features = attr.ib(type=InputFeatures, default=None)
  l2g_linked_ids = attr.ib(factory=list, type=List[int])


@attr.s
class WikiHopExample(object):
  """A single training/test example for the WikiHop dataset."""

  # Unique id of the example.
  example_id = attr.ib(default=None)  # type: Text

  # Query text of the example.
  query = attr.ib(default=None)  # type: Text

  # Represents the set of docs for the example. Each entry in the list
  # corresponds to the content of the one doc. For this dataset, the order
  # of the docs isn't necessarily important.
  docs = attr.ib(default=None)  # type: List[Text]

  # The list of candidate answers in the example. Exactly one of them is the
  # ground_truth_answer.
  candidate_answers = attr.ib(default=None)  # type: List[Text]

  # The ground truth entity answer for the given (query, list_of_docs).
  ground_truth_answer = attr.ib(default=None)  # type: Text

  # The index of ground truth entity answer in the `candidate_answers`.
  ground_truth_answer_index = attr.ib(default=None)  # type: int

  @classmethod
  def from_json(cls,
                single_example: Dict[Text, Any],
                shuffle_docs_within_example: bool = False) -> "WikiHopExample":
    """Returns a single one `WikiHopExample` from the given json example."""
    example_id = single_example["id"]
    query = single_example["query"]
    docs = single_example["supports"]
    if shuffle_docs_within_example:
      random.shuffle(docs)
    candidate_answers = single_example["candidates"]
    candidate_answers = [candidate.strip() for candidate in candidate_answers]
    candidate_answers = [candidate.lower() for candidate in candidate_answers]
    ground_truth_answer = single_example["answer"].strip()
    ground_truth_answer = single_example["answer"].lower()
    ground_truth_answer_index = candidate_answers.index(ground_truth_answer)
    return cls(
        example_id=example_id,
        query=query,
        docs=docs,
        candidate_answers=candidate_answers,
        ground_truth_answer=ground_truth_answer,
        ground_truth_answer_index=ground_truth_answer_index)

  @classmethod
  def parse_examples(
      cls,
      json_str: Text,
      shuffle_docs_within_example: bool = False) -> List["WikiHopExample"]:
    """Parses multiple WikiHopExamples from the given json string."""
    examples = []
    for single_example in json.loads(json_str):
      examples.append(
          cls.from_json(
              single_example=single_example,
              shuffle_docs_within_example=shuffle_docs_within_example))
    return examples


@attr.s
class PaddingInputExample(WikiHopExample):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """
  query = attr.ib(default="padding_example_query")  # type: Text
  example_id = attr.ib(default="padding_example_0")  # type: Text
  ground_truth_answer_index = attr.ib(default=0)  # type: int


class WikiHopTFExampleConverter(object):
  """Converts a single `WikiHopExample` to `WikiHopInputFeatures`, TFExample."""

  def __init__(self, tokenizer: tokenization.FullTokenizer, long_seq_len: int,
               global_seq_len: int, max_num_sentences: int):
    self.tokenizer = tokenizer
    self.long_seq_len = long_seq_len
    self.global_seq_len = global_seq_len
    self.max_num_sentences = max_num_sentences

    # Long features
    self.long_token_ids = []
    self.long_sentence_ids = []
    self.long_paragraph_ids = []
    self.long_paragraph_breakpoints = []
    self.long_token_type_ids = []
    self.long_token_type_ids = []

    # Global features
    self.global_paragraph_breakpoints = []
    self.global_token_ids = []
    self.global_token_type_ids = []

    # For enabling arbitrary linking between global and long tokens. We use
    # these to link candidate global tokens to every mention of the candidate
    # in the long context.
    self.l2g_linked_ids = [-1] * long_seq_len

    # A dict for keeping track of the list of spans in the long input of their
    # mentions. The span begin / end indices correspond to of the span of
    # indices in the long input.
    self.cand_to_span_positions = {}  # type: Dict[int, List[Tuple[int, int]]]

  def convert_single_example(self, example: WikiHopExample) -> tf.train.Example:
    """Converts a single `WikiHopExample` to TF Example."""

    features = self.convert_single_example_to_features(example=example)
    # Long features
    features_dict = collections.OrderedDict()
    features_dict["long_token_ids"] = create_int_feature(
        features.input_features.long_token_ids)
    features_dict["long_token_type_ids"] = create_int_feature(
        features.input_features.long_token_type_ids)
    features_dict["long_sentence_ids"] = create_int_feature(
        features.input_features.long_sentence_ids)
    features_dict["long_paragraph_ids"] = create_int_feature(
        features.input_features.long_paragraph_ids)
    features_dict["long_paragraph_breakpoints"] = create_int_feature(
        features.input_features.long_paragraph_breakpoints)
    features_dict["l2g_linked_ids"] = create_int_feature(
        features.l2g_linked_ids)

    # Global features
    features_dict["global_paragraph_breakpoints"] = create_int_feature(
        features.input_features.global_paragraph_breakpoints)

    features_dict["global_token_ids"] = create_int_feature(
        features.input_features.global_token_ids)

    features_dict["global_token_type_ids"] = create_int_feature(
        features.input_features.global_token_type_ids)

    # Other features
    features_dict["is_real_example"] = create_int_feature(
        [int(features.input_features.is_real_example)])

    features_dict["label_id"] = create_float_feature(
        [float(example.ground_truth_answer_index)])

    # Debug features
    if not isinstance(example, PaddingInputExample):
      features_dict["example_id"] = create_string_feature([example.example_id])
      features_dict["query"] = create_string_feature([example.query])
      features_dict["candidates"] = create_string_feature(
          example.candidate_answers)
      features_dict["ground_truth_answer"] = create_string_feature(
          [example.ground_truth_answer])
    return tf.train.Example(features=tf.train.Features(feature=features_dict))

  def convert_single_example_to_features(
      self, example: WikiHopExample) -> WikiHopInputFeatures:
    """Converts one `WikiHopExample` to `WikiHopInputFeatures`."""

    if isinstance(example, PaddingInputExample):
      return WikiHopInputFeatures(
          input_features=InputFeatures(
              long_token_ids=[0] * self.long_seq_len,
              long_token_type_ids=[0] * self.long_seq_len,
              long_sentence_ids=[-1] * self.long_seq_len,
              long_paragraph_ids=[-1] * self.long_seq_len,
              long_paragraph_breakpoints=[0] * self.long_seq_len,
              global_paragraph_breakpoints=[0] * self.global_seq_len,
              global_token_ids=[0] * self.global_seq_len,
              global_token_type_ids=[0] * self.global_seq_len,
              is_real_example=False),
          l2g_linked_ids=[-1] * self.long_seq_len)

    # We do the following three steps here.
    # 1) Truncate to ensure that the total number of sentences across all the
    #    docs is no more than `max_num_sentences`.
    #
    # 2) Truncate tokens further to ensure that the total number of WordPieces
    # across all of the docs is no more than `long_seq_len`.
    #
    # 3) Convert WikiHopExample to InputFeatures. The ETC features
    # are structured as follows:
    #
    # Global Input Structure:
    # [1 token per candidate][1 token per query
    # WordPiece][1 doc level token + 1 token per sentence in the doc]
    # [1 doc level token + 1 token per sentence of another doc]......[Padding]
    #
    # Long Input Structure:
    # [Candidate WordPieces][Query WordPieces][Doc tokens][Another doc tokens]...
    # [Padding]
    #
    # self.long_sentence_ids assignment:
    # 1) every candidate would be assigned a different sentence_id
    # 2) every query "WordPiece" would be assigned a different sentence_id
    # 3) every sentence in every doc would be assigned a different sentence_id
    # 4) sentence_ids are padded using -1s (and not 0s as is the case generally)
    #
    # self.long_paragraph_breakpoints:
    # 1) at the end of every candidate
    # 2) at the end of the query
    # 3) at the end of every doc
    #
    # self.long_paragraph_ids assignment:
    # 1) global input has a doc level token for every doc
    # 2) the goal of these ids is to match the doc tokens in the long input
    #   to the corresponding global level tokens
    # 3) every candidate / query token gets a -1
    # 4) all tokens of a doc gets the same paragraph_id (the ids should be such
    #    that global token at index i should map to doc with paragraph_id = i)
    # 5) paragraph_ids are padded using -1s (and not 0s as is the case generally)

    docs = example.docs
    num_query_tokens = len(self.tokenizer.tokenize(example.query))
    num_candidate_tokens = 0
    # Count candidate tokens
    candidates = example.candidate_answers
    for candidate in candidates:
      num_candidate_tokens += len(self.tokenizer.tokenize(candidate))

    max_allowed_doc_tokens = (
        self.long_seq_len - num_query_tokens - num_candidate_tokens)

    # List[List[]] to store list of sentences per doc.
    sentences_per_doc = truncate_doc_sentences(
        docs=docs,
        tokenizer=self.tokenizer,
        max_num_sentences=self.max_num_sentences,
        max_num_tokens=max_allowed_doc_tokens)

    begin_sentence_id = 0
    next_sentence_id = self._add_candidate_tokens(
        example=example, begin_sentence_id=begin_sentence_id)
    next_sentence_id = self._add_query_tokens(
        example=example, begin_sentence_id=next_sentence_id)
    self._add_doc_tokens(
        example=example,
        sentences_per_doc=sentences_per_doc,
        begin_sentence_id=next_sentence_id,
        max_allowed_doc_tokens=max_allowed_doc_tokens)
    self._link_long_global_tokens(example=example)

    self._pad_features()

    return WikiHopInputFeatures(
        input_features=InputFeatures(
            long_token_ids=self.long_token_ids,
            long_token_type_ids=self.long_token_type_ids,
            long_sentence_ids=self.long_sentence_ids,
            long_paragraph_ids=self.long_paragraph_ids,
            long_paragraph_breakpoints=self.long_paragraph_breakpoints,
            global_paragraph_breakpoints=self.global_paragraph_breakpoints,
            global_token_ids=self.global_token_ids,
            global_token_type_ids=self.global_token_type_ids),
        l2g_linked_ids=self.l2g_linked_ids)

  def _add_candidate_tokens(self, example: WikiHopExample,
                            begin_sentence_id: int) -> int:
    """Adds candidate tokens and returns the end_sentence_id.

    Every candidate is treated as a separate sentence.
    Args:
      example: The `WikiHopExample` to add the canddiate tokens to.
      begin_sentence_id: Begin sentence id to assign to candidates.

    Returns:
      end_sentence_id = begin_sentence_id + num_candidates
    """
    sentence_id = begin_sentence_id
    candidates = example.candidate_answers
    for (i, candidate) in enumerate(candidates):
      self.global_paragraph_breakpoints.append(1)
      self.global_token_ids.append(CANDIDATE_GLOBAL_TOKEN_ID)
      self.global_token_type_ids.append(CANDIDATE_GLOBAL_TOKEN_TYPE_ID)

      candidate = tokenization.convert_to_unicode(candidate)
      candidate = self.tokenizer.tokenize(candidate)
      candidate_token_ids = self.tokenizer.convert_tokens_to_ids(candidate)
      if i not in self.cand_to_span_positions:
        self.cand_to_span_positions[i] = []
      # Trivial span addition. Every candidate is present by default in the
      # long input.
      self.cand_to_span_positions[i].append(
          (len(self.long_token_ids),
           len(self.long_token_ids) + len(candidate_token_ids) - 1))

      for token_id in candidate_token_ids:
        self.long_token_ids.append(token_id)
        self.long_token_type_ids.append(CANDIDATE_TOKEN_TYPE_ID)
        self.long_sentence_ids.append(sentence_id)
        self.long_paragraph_ids.append(-1)
        self.long_paragraph_breakpoints.append(0)

      self.long_paragraph_breakpoints[-1] = 1
      sentence_id += 1

    return sentence_id

  def _add_query_tokens(self, example: WikiHopExample,
                        begin_sentence_id: int) -> int:
    """Adds query tokens to long / global input.

    We mirror query tokens in global as well, i.e, we will have one global
    token per query WordPiece. Every WordPiece of the query is treated as a
    separate sentence.

    Args:
      example: The `WikiHopExample` to add the query tokens to.
      begin_sentence_id: The begin sentence id to be used to start assiging
        sentence ids to query tokens.

    Returns:
      end_sentence_id = begin_sentence_id + num_query_word_pieces
    """
    sentence_id = begin_sentence_id
    query = example.query
    query = tokenization.convert_to_unicode(query)
    query_tokens = self.tokenizer.tokenize(query)
    query_token_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)

    for token_id in query_token_ids:
      self.long_token_ids.append(token_id)
      self.global_token_ids.append(QUESTION_GLOBAL_TOKEN_ID)

      self.long_token_type_ids.append(QUESTION_TOKEN_TYPE_ID)
      self.global_token_type_ids.append(QUESTION_GLOBAL_TOKEN_TYPE_ID)

      self.long_sentence_ids.append(sentence_id)
      self.long_paragraph_ids.append(-1)
      self.long_paragraph_breakpoints.append(0)
      self.global_paragraph_breakpoints.append(0)
      sentence_id += 1

    self.long_paragraph_breakpoints[-1] = 1
    self.global_paragraph_breakpoints[-1] = 1
    return sentence_id

  def _add_doc_tokens(self, example: WikiHopExample,
                      sentences_per_doc: List[List[str]],
                      begin_sentence_id: int, max_allowed_doc_tokens: int):
    """Adds doc tokens to global / long input."""
    sentence_id = begin_sentence_id
    num_total_tokens = 0
    for doc_sentences in sentences_per_doc:
      paragraph_id = sentence_id
      self.global_token_ids.append(PARAGRAPH_GLOBAL_TOKEN_ID)
      self.global_token_type_ids.append(PARAGRAPH_GLOBAL_TOKEN_TYPE_ID)
      self.global_paragraph_breakpoints.append(0)
      sentence_id += 1

      for sentence in doc_sentences:
        tokens = self.tokenizer.tokenize(sentence)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        num_total_tokens += len(tokens)
        # Find all mentions of every candidate in the sentence.
        for (i, candidate) in enumerate(example.candidate_answers):
          (_, candidate_spans) = find_candidate_mentions(
              input_text=sentence,
              candidate=candidate,
              tokenizer=self.tokenizer,
              offset=len(self.long_token_ids))
          # Every candidate should have already an entry in the span dict as
          # we would've added it in _add_candidate_tokens(..)
          assert i in self.cand_to_span_positions
          if candidate_spans:
            self.cand_to_span_positions[i].extend(candidate_spans)

        for token_id in token_ids:
          self.long_token_ids.append(token_id)
          self.long_token_type_ids.append(SENTENCE_TOKEN_TYPE_ID)
          self.long_sentence_ids.append(sentence_id)
          self.long_paragraph_ids.append(paragraph_id)
          self.long_paragraph_breakpoints.append(0)

        sentence_id += 1
        self.global_token_ids.append(SENTENCE_GLOBAL_TOKEN_ID)
        self.global_token_type_ids.append(GLOBAL_TOKEN_TYPE_ID)
        self.global_paragraph_breakpoints.append(0)
      self.long_paragraph_breakpoints[-1] = 1
      self.global_paragraph_breakpoints[-1] = 1
    assert num_total_tokens <= max_allowed_doc_tokens, (
        "num_total_tokens %d more than max_allowed_doc_tokens %d for example %s",
        num_total_tokens, max_allowed_doc_tokens, example.example_id)

  def _link_long_global_tokens(self, example: WikiHopExample):
    """Links candidate mentions in global to that in long."""
    # For every candidate, search across the context for all the occurrences
    # of the candidate mentions and link it. Note that this will also trivially
    # end up linking the candidates that have been mirrored at the beginning
    # of the long input also.
    for (i, _) in enumerate(example.candidate_answers):
      cand_spans = self.cand_to_span_positions[i]
      for span in cand_spans:
        assert span[1] >= span[0]
        assert span[1] < len(self.long_token_ids)
        span_len = span[1] - span[0] + 1
        self.l2g_linked_ids[span[0]:span[1] + 1] = [i] * span_len

  def _pad_features(self):
    """Pads all the features to appropriate lengths."""
    # Pad the long input
    assert len(self.long_token_ids) <= self.long_seq_len, (
        "len of long_token_ids is: %d ", len(self.long_token_ids))
    for _ in range(len(self.long_token_ids), self.long_seq_len):
      self.long_token_ids.append(0)
      self.long_token_type_ids.append(0)
      self.long_paragraph_breakpoints.append(0)
      self.long_sentence_ids.append(-1)
      self.long_paragraph_ids.append(-1)

    # Pad the global input
    assert len(self.global_token_ids) <= self.global_seq_len, (
        "len of global_token_ids is: %d ", len(self.global_token_ids))
    for _ in range(len(self.global_token_ids), self.global_seq_len):
      self.global_paragraph_breakpoints.append(0)
      self.global_token_ids.append(0)
      self.global_token_type_ids.append(0)

    assert len(self.long_token_ids) == self.long_seq_len
    assert len(self.long_token_type_ids) == self.long_seq_len
    assert len(self.long_sentence_ids) == self.long_seq_len
    assert len(self.long_paragraph_ids) == self.long_seq_len
    assert len(self.long_paragraph_breakpoints) == self.long_seq_len

    # Sentence / paragraph ids in the long input should map to global token
    # indices.
    assert max(self.long_sentence_ids) < len(self.global_token_ids)
    assert max(self.long_paragraph_ids) < len(self.global_token_ids)

    assert len(self.global_token_ids) == self.global_seq_len
    assert len(self.global_token_type_ids) == self.global_seq_len
    assert len(self.global_paragraph_breakpoints) == self.global_seq_len
