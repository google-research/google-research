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

"""Library for generating ETC HotpotQA tf examples."""
import collections
import functools
import json
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import apache_beam as beam
import attr
import tensorflow.compat.v1 as tf

from etcmodel.models import input_utils
from etcmodel.models import tokenization
from etcmodel.models.hotpotqa import data_utils

_LONG_FEATURE_NAMES_TO_PADDINGS = {
    "long_token_ids": 0,
    "long_sentence_ids": -1,
    "long_paragraph_ids": -1,
    "long_paragraph_breakpoints": 0,
    "long_token_type_ids": 0,
    "long_tokens_to_unigrams": -1,
}

_GLOBAL_FEATURE_NAMES_TO_PADDINGS = {
    "global_token_ids": 0,
    "global_sentence_ids": -1,
    "global_paragraph_ids": -1,
    "global_paragraph_breakpoints": 0,
    "global_token_type_ids": 0,
    "supporting_facts": 0,
}

WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS = dict(
    CLS_TOKEN_ID=101,  # "[CLS]"
    SENTENCE_GLOBAL_TOKEN_ID=1,  # "[unused0]", used in pretrianing.
    CPC_MASK_GLOBAL_TOKEN_ID=2,  # "[unused1]", used in pretrianing.
    QUESTION_GLOBAL_TOKEN_ID=3,  # "[unused2]"
    CANDIDATE_GLOBAL_TOKEN_ID=4,  # "[unused3]"
    PARAGRAPH_GLOBAL_TOKEN_ID=5,  # "[unused4]"
)

SENTENCEPIECE_DEFAULT_GLOBAL_TOKEN_IDS = dict(
    CLS_TOKEN_ID=40,  # "<unused_39>"
    SENTENCE_GLOBAL_TOKEN_ID=1,  # "</s>", used in pretrianing.
    CPC_MASK_GLOBAL_TOKEN_ID=2,  # "<s>", used in pretrianing.
    QUESTION_GLOBAL_TOKEN_ID=41,  # "<unused_40>"
    CANDIDATE_GLOBAL_TOKEN_ID=42,  # "<unused_41>"
    PARAGRAPH_GLOBAL_TOKEN_ID=43,  # "<unused_42>"
)

GLOBAL_TOKEN_TYPE_ID = 0
SENTENCE_TOKEN_TYPE_ID = 1
TITLE_TOKEN_TYPE_ID = 2
QUESTION_TOKEN_TYPE_ID = 3
CANDIDATE_TOKEN_TYPE_ID = 4
QUESTION_GLOBAL_TOKEN_TYPE_ID = 5
CANDIDATE_GLOBAL_TOKEN_TYPE_ID = 6
PARAGRAPH_GLOBAL_TOKEN_TYPE_ID = 7


@attr.s(auto_attribs=True)
class HotpotQAInputConfig:
  """Config options for ETC HotpotQA model input."""
  # Sequence length of the global input.
  global_seq_length: int = 2048
  # Sequence length of the long input.
  long_seq_length: int = 256
  # Whether in training mode.
  is_training: bool = True
  # Whether the debug mode is on.
  debug: bool = True
  # CLS token id.
  cls_token_id: int = WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS["CLS_TOKEN_ID"]

  # Global token ids. Global and long tokens share a same vocab.
  # Sentence global token id. Using 1 for transfer learning.
  sentence_global_token_id: int = WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS[
      "SENTENCE_GLOBAL_TOKEN_ID"]
  # CPC mask global token id. Using 2 for transfer learning.
  cpc_mask_global_token_id: int = WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS[
      "CPC_MASK_GLOBAL_TOKEN_ID"]
  # Question global token id.
  question_global_token_id: int = WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS[
      "QUESTION_GLOBAL_TOKEN_ID"]
  # Candidate global token id.
  candidate_global_token_id: int = WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS[
      "CANDIDATE_GLOBAL_TOKEN_ID"]
  # Paragraph global token id.
  paragraph_global_token_id: int = WORDPIECE_DEFAULT_GLOBAL_TOKEN_IDS[
      "PARAGRAPH_GLOBAL_TOKEN_ID"]

  # Token type ids. Global and long token types share a same vocab of size 16.
  # Global token type id. Using 0 for transfer learning.
  global_token_type_id: int = GLOBAL_TOKEN_TYPE_ID
  # Sentence token type id.
  sentence_token_type_id: int = SENTENCE_TOKEN_TYPE_ID
  # Title token type id.
  title_token_type_id: int = TITLE_TOKEN_TYPE_ID
  # question token type id.
  question_token_type_id: int = QUESTION_TOKEN_TYPE_ID
  # Candidate token type id.
  candidate_token_type_id: int = CANDIDATE_TOKEN_TYPE_ID

  # Answer text to answer type id map.
  answer_types_map: Dict[str,
                         int] = attr.Factory(lambda: dict(span=0, yes=1, no=2))

  # Answer encoding method.
  answer_encoding_method: str = "span"

  def use_sentencepiece_default_global_token_ids(self):
    self.__dict__.update({
        k.lower(): v for k, v in SENTENCEPIECE_DEFAULT_GLOBAL_TOKEN_IDS.items()
    })


@attr.s(auto_attribs=True)
class HotpotQAParagraph:
  """A raw HotpotQA paragraph."""
  # The title for the paragraph.
  title: str = ""
  # The sentences in the paragraph.
  sentences: List[str] = attr.Factory(list)

  @classmethod
  def from_json(cls, json_objs: Sequence[Any]) -> List["HotpotQAParagraph"]:
    """Loads a list of HotpotQAParagraph from json objs."""
    return [cls(title=x[0], sentences=x[1]) for x in json_objs]


@attr.s(auto_attribs=True)
class HotpotQAExample:
  """A raw HotpotQA data point."""
  # A unique id for this question-answer data point, useful for evaluation.
  id: str
  # The question for the document.
  question: str
  # Each entry is a paragraph, which is represented as a list with two elements
  # [title, sentences].
  context: List[HotpotQAParagraph]
  # The answer for the question.
  answer: str = ""
  # Keys are titles of the paragraphs. Values are lists of sentence ids
  # (0-based) in the paragraphs.
  supporting_facts: Dict[str, List[int]] = attr.Factory(
      lambda: collections.defaultdict(list))
  # The question type. Not used for inference.
  type: str = ""
  # The difficulity level of the  example.  Not used for inference.
  level: str = ""

  @classmethod
  def from_json(cls, json_str: str) -> List["HotpotQAExample"]:
    """Loads a list of HotpotQAExamples from json str."""
    examples = []
    for json_obj in json.loads(json_str):
      example = cls(
          id=json_obj["_id"],
          question=json_obj["question"],
          context=HotpotQAParagraph.from_json(json_obj["context"]))
      example.answer = json_obj.get("answer", "")
      example.type = json_obj.get("type", "")
      example.level = json_obj.get("level", "")
      for title, sent_id in json_obj.get("supporting_facts", []):
        example.supporting_facts[title].append(sent_id)
      examples.append(example)
    return examples


@attr.s(auto_attribs=True)
class HotPotQAInputFeatures:
  """The feautres in the tf example for HotpotQA."""
  # The config options for ETC HotpotQA model input. The default is None, but
  # the value must be set before adding tokens to the features.
  input_config: Optional[HotpotQAInputConfig] = None

  # Context features
  # Long token ids with format question paragraph1 paragraph2 ...
  long_token_ids: List[int] = attr.Factory(list)
  # The sentence ids for the long tokens. Each id `i` corresponds to a sentence
  # global token, `global_token_ids[i]`. Each question token has a
  # unique sentence id.
  long_sentence_ids: List[int] = attr.Factory(list)
  # The paragraph ids for the long tokens. Each id `i` corresponds to a
  # paragraph global token, `global_token_ids[i]`. Question tokens don't
  # correspond to any paragraph global tokens.
  long_paragraph_ids: List[int] = attr.Factory(list)
  # Ending breakpoints separating question and paragraphs long tokens into
  # different segments, which are not attended by local attentions.
  long_paragraph_breakpoints: List[int] = attr.Factory(list)
  # The token type ids for long tokens. With default values in `InputConfig`.
  long_token_type_ids: List[int] = attr.Factory(list)
  # The long token indices to unigrams indices mapping. The unigram indices are
  # within each title or sentence without offset.
  long_tokens_to_unigrams: List[int] = attr.Factory(list)

  # The global token ids. With default values in `InputConfig`.
  global_token_ids: List[int] = attr.Factory(list)
  # Ending breakpoints separating question and paragraphs global
  # tokens into different segments.
  global_paragraph_breakpoints: List[int] = attr.Factory(list)
  # The token type ids for global tokens. With default values in `InputConfig`.
  global_token_type_ids: List[int] = attr.Factory(list)
  # The sentence ids (0-based) for global tokens. Each id `i` corresponds to
  # the `i`'th sentence in the paragraph. [CLS], Question, paragraph, title or
  # padding global tokens don't correspond to any sentence.
  global_sentence_ids: List[int] = attr.Factory(list)
  # The paragraph ids (0-based) for global tokens. Each id `i` corresponds to
  # the `i`'th paragraph in the json example. [CLS], Question or padding global
  # tokens don't correspond to any paragraph.
  global_paragraph_ids: List[int] = attr.Factory(list)

  # The following fields are labels for training.
  # 1 if a global token corresponds to a supporting facts sentence, or
  # corresponds to a supporting facts title, or corresponds to a supporting
  # fact paragraph. O otherwise.
  supporting_facts: List[int] = attr.Factory(list)
  # The Wordpiece token level span (inclusive) for the answer text in tokens.
  # If multiple answer occurrences only the first occurrence will be selected.
  # The default value of (0, 0) means the question is a boolean question.
  answer_span: Tuple[int, int] = (0, 0)
  # The answer BIO (begin, inside, outside) encoding ids. 0 for begin, 1 for
  # inside, 2 for outside.
  answer_bio_ids: List[int] = attr.Factory(list)
  # The  answer type, 0 for span answer, 1 for yes, and 2 for no.
  answer_type: int = 0

  # The following fields are for debugging purpose.
  # The long tokens.
  long_tokens: List[str] = attr.Factory(list)
  # The global tokens.
  global_tokens: List[str] = attr.Factory(list)

  # The current long sentence id.
  _curr_sentence_id: int = 0
  # The current long paragraph id.
  _curr_paragraph_id: int = 0
  # The current actual paragraph id (0-based), which corresponds to the
  # paragraph id in the json example..
  _curr_actual_paragraph_id: int = 0
  # The current actual sentence id (0-based), which corresponds to the sentence
  # id in the paragraph.
  _curr_actual_sentence_id: int = 0

  def _can_add_long_tokens(self, num_new_tokens) -> bool:
    if self.input_config.long_seq_length is None:
      return True
    return self.num_long_tokens + num_new_tokens <= self.input_config.long_seq_length

  def _can_add_global_tokens(self) -> bool:
    if self.input_config.global_seq_length is None:
      return True
    return self.num_global_tokens + 1 <= self.input_config.global_seq_length

  def add_global_token(self,
                       global_token_id: int,
                       global_token_type_id: int,
                       is_title: bool = False,
                       is_supporting_fact: bool = False) -> None:
    """Adds a global token. No breakpoints."""
    if not self._can_add_global_tokens():
      return
    self.global_token_ids.append(global_token_id)
    self.global_paragraph_breakpoints.append(0)
    self.global_token_type_ids.append(global_token_type_id)
    self.supporting_facts.append(int(is_supporting_fact))
    if (global_token_id == self.input_config.sentence_token_type_id and
        not is_title):
      self.global_sentence_ids.append(self._curr_actual_sentence_id)
      self._curr_actual_sentence_id += 1
    else:
      self.global_sentence_ids.append(-1)
    if global_token_id in [
        self.input_config.paragraph_global_token_id,
        self.input_config.sentence_global_token_id
    ]:
      self.global_paragraph_ids.append(self._curr_actual_paragraph_id)
    else:
      self.global_paragraph_ids.append(-1)
    # The current sentence id is updated to the next global token index.
    self._curr_sentence_id = self.num_global_tokens

  def add_long_tokens(self,
                      long_token_ids: Sequence[int],
                      long_token_type_id: int,
                      long_tokens_to_unigrams: Sequence[int],
                      has_sentence: bool = True,
                      has_paragraph: bool = True) -> None:
    """Adds a sequence of long tokens. No breakpoints."""
    num_new_tokens = len(long_token_ids)
    if not self._can_add_long_tokens(num_new_tokens):
      return
    self.long_token_ids.extend(long_token_ids)
    sentence_id = self._curr_sentence_id if has_sentence else -1
    self.long_sentence_ids.extend([sentence_id] * num_new_tokens)
    paragraph_id = self._curr_paragraph_id if has_paragraph else -1
    self.long_paragraph_ids.extend([paragraph_id] * num_new_tokens)
    self.long_paragraph_breakpoints.extend([0] * num_new_tokens)
    self.long_token_type_ids.extend([long_token_type_id] * num_new_tokens)
    self.long_tokens_to_unigrams.extend(long_tokens_to_unigrams)

  def set_paragraph_breakpoint(self, is_actual_paragraph: bool = False) -> None:
    """Sets the long and global last tokens to be paragraph breakpoints."""
    if self.long_paragraph_breakpoints:
      self.long_paragraph_breakpoints[-1] = 1
    if self.global_paragraph_breakpoints:
      self.global_paragraph_breakpoints[-1] = 1
    # The current paragraph id is updated to the next global token index.
    self._curr_paragraph_id = self.num_global_tokens
    if is_actual_paragraph:
      self._curr_actual_paragraph_id += 1
      self._curr_actual_sentence_id = 0

  def add_sentence(self,
                   long_token_ids: Sequence[int],
                   long_token_type_id: int,
                   long_tokens_to_unigrams: Sequence[int],
                   global_token_id: int,
                   global_token_type_id: int,
                   has_sentence: bool = True,
                   has_paragraph: bool = True,
                   is_title: bool = False,
                   is_supporting_fact: bool = False) -> None:
    """Adds long and global tokens for a sentence. No breakpoints."""
    if (not self._can_add_long_tokens(len(long_token_ids)) or
        not self._can_add_global_tokens()):
      return
    self.add_long_tokens(
        long_token_ids,
        long_token_type_id,
        long_tokens_to_unigrams,
        has_sentence=has_sentence,
        has_paragraph=has_paragraph)
    self.add_global_token(
        global_token_id,
        global_token_type_id,
        is_title=is_title,
        is_supporting_fact=is_supporting_fact)

  def pad(self) -> None:
    """Pads features to `long_seq_length` and `global_seq_length`."""
    if self.input_config.long_seq_length is not None:
      num_long_paddings = (
          self.input_config.long_seq_length - self.num_long_tokens)
      assert num_long_paddings >= 0, "Too many long tokens added."
      for field_name, padding_id in _LONG_FEATURE_NAMES_TO_PADDINGS.items():
        self.__dict__[field_name].extend([padding_id] * num_long_paddings)
    if self.input_config.global_seq_length is not None:
      num_global_paddings = (
          self.input_config.global_seq_length - self.num_global_tokens)
      assert num_global_paddings >= 0, "Too many global tokens added."
      for field_name, padding_id in _GLOBAL_FEATURE_NAMES_TO_PADDINGS.items():
        self.__dict__[field_name].extend([padding_id] * num_global_paddings)

  def add_answer_bio_ids(self, answer_spans: Sequence[Tuple[int, int]]) -> None:
    """Adds answer BIO encoding ids. 0 for B, 1 for I, 2 for O."""
    long_seq_length = self.input_config.long_seq_length
    self.answer_bio_ids = [2] * long_seq_length
    for begin, end in answer_spans:
      if end >= long_seq_length:
        break
      # It is assumed the answer spans have no overlap.
      self.answer_bio_ids[begin:end + 1] = [0, *([1] * (end - begin))]

  @property
  def num_long_tokens(self) -> int:
    return len(self.long_token_ids)

  @property
  def num_global_tokens(self) -> int:
    return len(self.global_token_ids)


class HotpotQATFExampleConverter(object):
  """HoppotQA example to tf.Example converter."""

  def __init__(self,
               config: HotpotQAInputConfig,
               spm_model_file: Optional[str] = None,
               vocab_file: Optional[str] = None) -> None:
    self._config = config
    if spm_model_file:
      self._use_wordpiece = False
      self._tokenizer = tokenization.FullTokenizer(None, None, spm_model_file)
      self._get_tokenized_text = functools.partial(
          data_utils.get_sentencepiece_tokenized_text,
          tokenizer=self._tokenizer)
      self._find_answer_spans = data_utils.find_answer_spans_sentencepiece
    elif vocab_file:
      self._use_wordpiece = True
      self._tokenizer = tokenization.FullTokenizer(vocab_file)
      self._get_tokenized_text = functools.partial(
          data_utils.get_wordpiece_tokenized_text, tokenizer=self._tokenizer)
      self._find_answer_spans = functools.partial(
          data_utils.find_answer_spans_wordpiece, tokenizer=self._tokenizer)
    else:
      raise ValueError(
          "Either a 'sp_model' or a 'vocab_file' need to specified to create a"
          "tokenizer.")

  def _find_answer_spans_with_offset(self, tokenized_context, answer: str,
                                     offset: int) -> List[Tuple[int, int]]:
    answer_spans = self._find_answer_spans(tokenized_context, answer)
    return [(b + offset, e + offset) for b, e in answer_spans]

  def convert(self, example: HotpotQAExample) -> Optional[tf.train.Example]:
    features = self._create_input_features(example)
    if features is None:
      return None
    return self._to_tf_example(features, example)

  def _create_input_features(
      self, example: HotpotQAExample) -> Optional[HotPotQAInputFeatures]:
    """Creates HotPotQAInputFeatures give a HotpotQAExample."""
    # All answer occurrence token level spans, inclusive.
    answer_spans = []

    features = HotPotQAInputFeatures(input_config=self._config)

    # A [CLS] token is appended to the front of global tokens.
    features.add_global_token(self._config.cls_token_id,
                              self._config.global_token_type_id)
    features.set_paragraph_breakpoint()
    # A [CLS] token is appended to the front of long tokens. For yes/no answers
    # the begin/end indices will occupy this position.
    features.add_long_tokens([self._config.cls_token_id],
                             self._config.sentence_token_type_id, [-1],
                             has_sentence=False,
                             has_paragraph=False)
    features.long_paragraph_breakpoints[-1] = 1

    # Each question token corresponds to one sentence global token.
    tokenized_question = self._get_tokenized_text(example.question)
    for question_token_id in tokenized_question.token_ids:
      features.add_sentence([question_token_id],
                            self._config.question_token_type_id, [-1],
                            self._config.question_global_token_id,
                            self._config.global_token_type_id,
                            has_paragraph=False)
    features.set_paragraph_breakpoint()

    for paragraph in example.context:
      sp_sent_ids = example.supporting_facts.get(paragraph.title, [])
      # One paragraph global token before sentence global tokens.
      features.add_global_token(
          self._config.paragraph_global_token_id,
          self._config.global_token_type_id,
          is_supporting_fact=bool(sp_sent_ids))

      tokenized_title = self._get_tokenized_text(paragraph.title)
      # Checks if answer is in title.
      if sp_sent_ids:
        answer_spans.extend(
            self._find_answer_spans_with_offset(tokenized_title, example.answer,
                                                features.num_long_tokens))

      # Title is added as a sentence.
      features.add_sentence(
          tokenized_title.token_ids,
          self._config.title_token_type_id,
          tokenized_title.tokens_to_unigrams,
          self._config.sentence_global_token_id,
          self._config.global_token_type_id,
          is_title=True,
          is_supporting_fact=bool(sp_sent_ids))

      for sent_id, sentence in enumerate(paragraph.sentences):
        tokenized_sentence = self._get_tokenized_text(sentence)
        # Checks if answer is in a supporting fact sentence.
        if sent_id in sp_sent_ids:
          answer_spans.extend(
              self._find_answer_spans_with_offset(tokenized_sentence,
                                                  example.answer,
                                                  features.num_long_tokens))

        features.add_sentence(
            tokenized_sentence.token_ids,
            self._config.sentence_token_type_id,
            tokenized_sentence.tokens_to_unigrams,
            self._config.sentence_global_token_id,
            self._config.global_token_type_id,
            is_supporting_fact=sent_id in sp_sent_ids)
      features.set_paragraph_breakpoint(is_actual_paragraph=True)

    if self._config.debug:
      features.long_tokens = self._tokenizer.convert_ids_to_tokens(
          features.long_token_ids)
      features.global_tokens = self._tokenizer.convert_ids_to_tokens(
          features.global_token_ids)

    features.pad()

    if self._config.is_training:
      if not any(features.supporting_facts):
        return None

      answer_spans = sorted(answer_spans, key=lambda x: x[1])
      if example.answer in ["yes", "no"]:
        features.answer_type = self._config.answer_types_map[example.answer]
      elif (answer_spans and answer_spans[0][1] < self._config.long_seq_length):
        features.answer_type = self._config.answer_types_map["span"]
      else:
        return None

      if self._config.answer_encoding_method == "span":
        features.answer_span = answer_spans[0] if answer_spans else (0, 0)
      else:
        features.add_answer_bio_ids(answer_spans)

    return features

  def _to_tf_example(self, features: HotPotQAInputFeatures,
                     example: HotpotQAExample) -> tf.train.Example:
    """Converts a HotPotQAInputFeatures to a tf.Example."""
    features_dict = collections.OrderedDict()
    features_dict["unique_ids"] = input_utils.create_bytes_feature([example.id])
    features_dict["type"] = input_utils.create_bytes_feature([example.type])
    features_dict["level"] = input_utils.create_bytes_feature([example.level])
    features_dict["long_token_ids"] = input_utils.create_int_feature(
        features.long_token_ids)
    features_dict["long_sentence_ids"] = input_utils.create_int_feature(
        features.long_sentence_ids)
    features_dict["long_paragraph_ids"] = input_utils.create_int_feature(
        features.long_paragraph_ids)
    features_dict[
        "long_paragraph_breakpoints"] = input_utils.create_int_feature(
            features.long_paragraph_breakpoints)
    features_dict["long_token_type_ids"] = input_utils.create_int_feature(
        features.long_token_type_ids)
    features_dict["global_token_ids"] = input_utils.create_int_feature(
        features.global_token_ids)
    features_dict[
        "global_paragraph_breakpoints"] = input_utils.create_int_feature(
            features.global_paragraph_breakpoints)
    features_dict["global_token_type_ids"] = input_utils.create_int_feature(
        features.global_token_type_ids)
    if self._config.is_training:
      features_dict["supporting_facts"] = input_utils.create_int_feature(
          features.supporting_facts)
      features_dict["answer_types"] = input_utils.create_int_feature(
          [features.answer_type])
      if self._config.answer_encoding_method == "span":
        features_dict["answer_begins"] = input_utils.create_int_feature(
            [features.answer_span[0]])
        features_dict["answer_ends"] = input_utils.create_int_feature(
            [features.answer_span[1]])
      else:
        features_dict["answer_bio_ids"] = input_utils.create_int_feature(
            features.answer_bio_ids)
    if not self._config.is_training or self._config.debug:
      if self._use_wordpiece:
        features_dict[
            "long_tokens_to_unigrams"] = input_utils.create_int_feature(
                features.long_tokens_to_unigrams)
      features_dict["global_paragraph_ids"] = input_utils.create_int_feature(
          features.global_paragraph_ids)
      features_dict["global_sentence_ids"] = input_utils.create_int_feature(
          features.global_sentence_ids)
    if self._config.debug:
      features_dict["long_tokens"] = input_utils.create_bytes_feature(
          features.long_tokens)
      features_dict["global_tokens"] = input_utils.create_bytes_feature(
          features.global_tokens)
    return tf.train.Example(features=tf.train.Features(feature=features_dict))


class HotpotQAExampleToTfExamplesFn(beam.DoFn):
  """DoFn for converting HotpotQAExample to tf.Example."""

  def __init__(self, global_seq_length, long_seq_length, is_training,
               answer_encoding_method, spm_model_file, vocab_file, debug):
    self._global_seq_length = global_seq_length
    self._long_seq_length = long_seq_length
    self._is_training = is_training
    self._answer_encoding_method = answer_encoding_method
    self._spm_model_file = spm_model_file
    self._vocab_file = vocab_file
    self._debug = debug

  def setup(self):
    super().setup()
    self._config = HotpotQAInputConfig(
        global_seq_length=self._global_seq_length,
        long_seq_length=self._long_seq_length,
        is_training=self._is_training,
        answer_encoding_method=self._answer_encoding_method,
        debug=self._debug)
    if self._spm_model_file:
      self._config.use_sentencepiece_default_global_token_ids()
    self._converter = HotpotQATFExampleConverter(self._config,
                                                 self._spm_model_file,
                                                 self._vocab_file)

  def process(self, example: HotpotQAExample, *args,
              **kwargs) -> Iterator[tf.train.Example]:
    tf_example = self._converter.convert(example)
    if tf_example is not None:
      yield tf_example
