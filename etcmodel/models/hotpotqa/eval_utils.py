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

"""Library for making predictions for HotpotQA."""
import collections
from typing import Sequence, Mapping, Tuple, List

import numpy as np

from etcmodel.models import tokenization
from etcmodel.models.hotpotqa import data_utils
from etcmodel.models.hotpotqa import generate_tf_examples_lib as lib

_TITLE_AND_SENTENCE_TYPE_IDS = [
    lib.SENTENCE_TOKEN_TYPE_ID,
    lib.TITLE_TOKEN_TYPE_ID,
]

_SpanType = Tuple[int, int]
_RawPredictionType = Mapping[str, np.ndarray]


def get_final_text(token_text: str,
                   unigram_text: str,
                   do_lower_case: bool = True) -> str:
  """Projects the token-concated text back to the unigram-concated text.

  This function is branched from the original BERT `run_squad.py`.

  When we created the data, we kept track of the alignment between original
  (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  now `unigram_text` contains the span of our original text corresponding to the
  span that we predicted.

  However, `unigram_text` may contain extra characters that we don't want in
  our prediction.

  For example, let's say:
    token_text = steve smith
    unigram_text = Steve Smith's

  We don't want to return `unigram_text` because it contains the extra "'s".

  We don't want to return `token_text` because it's already been normalized
  (the SQuAD eval script also does punctuation stripping/lower casing but
  our tokenizer does additional normalization like stripping accent
  characters).

  What we really want to return is "Steve Smith".

  Therefore, we have to apply a semi-complicated alignment heruistic between
  `token_text` and `unigram_text` to get a character-to-charcter alignment. This
  can fail in certain cases in which case we just return `unigram_text`.

  Args:
    token_text: The text obtained by concatenating wordpiece tokens and removing
      '##' and ' ##' symbols.
    unigram_text: The text obtained by concatenating unigrams.
    do_lower_case: Whether the tokenizer is doing lower case.

  Returns:
    The text corresponding to `token_text` in `unigram_text`. If unable to find
    such correspondence, `unigram_text` is returned directly.
  """

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `unigram_text`, strip whitespace from the result
  # and `token_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(unigram_text))

  start_position = tok_text.find(token_text)
  if start_position == -1:
    return unigram_text
  end_position = start_position + len(token_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(unigram_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    return unigram_text

  # We then project the characters in `token_text` back to `unigram_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for i, tok_index in tok_ns_to_s_map.items():
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    return unigram_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    return unigram_text

  output_text = unigram_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_sentence_text(sentence_id: int, raw_prediction: _RawPredictionType,
                       data_point) -> str:
  """Gets the sentence (or title) text in the json data point."""
  actual_paragraph_id = raw_prediction["global_paragraph_ids"][sentence_id]
  if actual_paragraph_id == -1:
    return ""
  actual_sentence_id = raw_prediction["global_sentence_ids"][sentence_id]
  title, sentences = data_point["context"][actual_paragraph_id]
  if actual_sentence_id == -1:
    return title
  return sentences[actual_sentence_id]


def _get_answer_unigram_text(token_span: _SpanType,
                             raw_prediction: _RawPredictionType,
                             data_point) -> str:
  """Gets the original answer unigram text corresponding to the token span."""
  unigram_span = tuple(
      raw_prediction["long_tokens_to_unigrams"][idx] for idx in token_span)
  sentence_id = raw_prediction["long_sentence_ids"][token_span[0]]
  sentence_text = _get_sentence_text(sentence_id, raw_prediction, data_point)
  sentence_unigrams, _, _ = data_utils.whitespace_split_with_indices(
      sentence_text)
  answer_unigrams = sentence_unigrams[unigram_span[0]:unigram_span[1] + 1]
  return " ".join(answer_unigrams)


def _get_wordpiece_detokenized_text(
    token_span: _SpanType, raw_prediction: _RawPredictionType,
    tokenizer: tokenization.FullTokenizer) -> str:
  """Gets the normalized answer token text given the token span."""
  answer_tokens = tokenizer.convert_ids_to_tokens(
      raw_prediction["long_token_ids"][token_span[0]:token_span[1] + 1])
  return data_utils.wordpiece_tokens_to_normalized_text(answer_tokens)


def _get_wordpiece_final_text(token_span: _SpanType,
                              raw_prediction: _RawPredictionType, data_point,
                              tokenizer: tokenization.FullTokenizer):
  """Gets final text using WordPiece tokens."""
  answer_unigram_text = _get_answer_unigram_text(token_span, raw_prediction,
                                                 data_point)
  answer_token_text = _get_wordpiece_detokenized_text(token_span,
                                                      raw_prediction, tokenizer)
  return get_final_text(answer_token_text, answer_unigram_text, True)


def _get_sentencepiece_detokenized_text(token_span: _SpanType,
                                        raw_prediction: _RawPredictionType,
                                        tokenizer: tokenization.FullTokenizer):
  """Gets final text using SentencePiece tokens."""
  long_token_ids = raw_prediction["long_token_ids"]
  answer_tokens = tokenizer.convert_ids_to_tokens(
      long_token_ids[token_span[0]:token_span[1] + 1].tolist())
  return data_utils.sentencepiece_detokenize(answer_tokens)


def get_spans_from_bio_encoding(
    raw_prediction: _RawPredictionType, max_answer_length: int,
    supporting_facts: Sequence[bool]) -> List[Tuple[float, _SpanType]]:
  """Gets top-1 answer span from BIO encoding."""
  answer_bio_probs = raw_prediction["answer_bio_probs"]
  answer_bio_ids = raw_prediction["answer_bio_ids"]
  long_token_type_ids = raw_prediction["long_token_type_ids"]
  long_sentence_ids = raw_prediction["long_sentence_ids"]
  answer_spans = []
  for begin in np.where(answer_bio_ids == 0)[0]:
    if long_token_type_ids[begin] not in _TITLE_AND_SENTENCE_TYPE_IDS:
      continue
    end = begin
    while end + 1 < len(answer_bio_ids) and answer_bio_ids[end + 1] == 1:
      end += 1
    if long_token_type_ids[end] not in _TITLE_AND_SENTENCE_TYPE_IDS:
      continue
    # Begin and end must belong to a same sentence.
    begin_sentence_id = long_sentence_ids[begin]
    end_sentence_id = long_sentence_ids[end]
    if begin_sentence_id != end_sentence_id:
      continue
    # The sentence containing begin and end must be a supporting facts.
    if not supporting_facts[begin_sentence_id]:
      continue
    if end - begin + 1 > max_answer_length:
      continue
    answer_spans.append((answer_bio_probs[begin], (begin, end)))
  return answer_spans


def get_spans_from_bio_encoding_v2(
    raw_prediction: _RawPredictionType, max_answer_length: int,
    supporting_facts: Sequence[bool]) -> List[Tuple[float, _SpanType]]:
  """Gets top-1 answer span from BIO encoding."""
  answer_bio_probs = raw_prediction["answer_bio_probs"]
  answer_bio_ids = raw_prediction["answer_bio_ids"]
  long_token_type_ids = raw_prediction["long_token_type_ids"]
  long_sentence_ids = raw_prediction["long_sentence_ids"]
  span_candidates = []
  curr_begin = None
  for index, bio_id in enumerate(answer_bio_ids):
    if bio_id == 0:
      if curr_begin is not None:
        span_candidates.append((curr_begin, index - 1))
      curr_begin = index
    elif bio_id == 1:
      # Even a span do not start with "B", still consider as a candidate span.
      if curr_begin is None:
        curr_begin = index
    elif curr_begin is not None:
      span_candidates.append((curr_begin, index - 1))
      curr_begin = None

  answer_spans = []
  for begin, end in span_candidates:
    # Begin and end must be of title and sentence type.
    if (long_token_type_ids[begin] not in _TITLE_AND_SENTENCE_TYPE_IDS or
        long_token_type_ids[end] not in _TITLE_AND_SENTENCE_TYPE_IDS):
      continue
    # Begin and end must belong to a same sentence.
    begin_sentence_id = long_sentence_ids[begin]
    end_sentence_id = long_sentence_ids[end]
    if begin_sentence_id != end_sentence_id:
      continue
    # The sentence containing begin and end must be a supporting facts.
    if not supporting_facts[begin_sentence_id]:
      continue
    if end - begin + 1 > max_answer_length:
      continue
    score = sum(answer_bio_probs[begin:end + 1]) / (end - begin + 1)
    answer_spans.append((score, (begin, end)))

  return answer_spans


def get_spans_from_span_encoding(
    raw_prediction: _RawPredictionType, max_answer_length: int,
    supporting_facts: Sequence[bool]) -> List[Tuple[float, _SpanType]]:
  """Gets top-1 answer span from SPAN encoding."""
  begin_probs = raw_prediction["answer_begin_top_probs"]
  begin_indices = raw_prediction["answer_begin_top_indices"]
  end_probs = raw_prediction["answer_end_top_probs"]
  end_indices = raw_prediction["answer_end_top_indices"]
  long_token_type_ids = raw_prediction["long_token_type_ids"]
  long_sentence_ids = raw_prediction["long_sentence_ids"]
  answer_spans = []
  for begin_prob, begin in zip(begin_probs, begin_indices):
    if long_token_type_ids[begin] not in _TITLE_AND_SENTENCE_TYPE_IDS:
      continue
    for end_prob, end in zip(end_probs, end_indices):
      if long_token_type_ids[end] not in _TITLE_AND_SENTENCE_TYPE_IDS:
        continue
      # Begin and end must belong to a same sentence.
      begin_sentence_id = long_sentence_ids[begin]
      end_sentence_id = long_sentence_ids[end]
      if begin_sentence_id != end_sentence_id:
        continue
      # The sentence containing begin and end must be a supporting facts.
      if not supporting_facts[begin_sentence_id]:
        continue
      if begin > end or end - begin + 1 > max_answer_length:
        continue
      answer_spans.append((begin_prob * end_prob, (begin, end)))
  return answer_spans


def get_top1_answer(raw_prediction: _RawPredictionType, data_point,
                    max_answer_length: int, supporting_facts: Sequence[bool],
                    tokenizer: tokenization.FullTokenizer, use_wordpiece: bool,
                    answer_encoding_method: str) -> str:
  """Gets top-1 answer text."""
  if answer_encoding_method == "span":
    answer_spans = get_spans_from_span_encoding(raw_prediction,
                                                max_answer_length,
                                                supporting_facts)
  elif answer_encoding_method == "bio":
    answer_spans = get_spans_from_bio_encoding_v2(raw_prediction,
                                                  max_answer_length,
                                                  supporting_facts)
  else:
    raise ValueError(f"Invalid answer encoding method {answer_encoding_method}")
  if not answer_spans:
    return ""
  token_span = sorted(answer_spans)[-1][1]
  if use_wordpiece:
    return _get_wordpiece_final_text(token_span, raw_prediction, data_point,
                                     tokenizer)
  return _get_sentencepiece_detokenized_text(token_span, raw_prediction,
                                             tokenizer)


def generate_prediction_json(raw_predictions: Sequence[_RawPredictionType],
                             gold_json_data,
                             tokenizer: tokenization.FullTokenizer,
                             sp_threshold: float = 0.5,
                             max_answer_length: int = 30,
                             use_wordpiece: bool = False,
                             answer_encoding_method: str = "span"):
  """Generates HotpotQA official format prediction json object.

  Args:
    raw_predictions: Raw model predict outputs.
    gold_json_data: Gold json eval data.
    tokenizer: The BERT tokenizer.
    sp_threshold: Probability threshold for prediction supporting facts.
    max_answer_length: Max number of wordpiece tokens allowed for answer.
    use_wordpiece: Whether WordPirce tokenizer is used.
    answer_encoding_method: The answer encoding method.

  Returns:
    The official json format of predictions.
  """
  ids_to_raw_predictions = {}
  for raw_prediction in raw_predictions:
    unique_id = raw_prediction["unique_ids"]
    if isinstance(unique_id, bytes):
      unique_id = unique_id.decode("utf-8")
    ids_to_raw_predictions[unique_id] = raw_prediction

  answers = {}
  sps = {}
  for data_point in gold_json_data:
    unique_id = data_point["_id"]
    answers[unique_id] = ""
    sps[unique_id] = []
    raw_prediction = ids_to_raw_predictions.get(unique_id, None)
    if raw_prediction is None:
      continue

    # Predicts supporting facts.
    supporting_facts = raw_prediction["supporting_facts_probs"] >= sp_threshold
    for sp, para_id, sent_id in zip(supporting_facts,
                                    raw_prediction["global_paragraph_ids"],
                                    raw_prediction["global_sentence_ids"]):
      if para_id != -1 and sent_id != -1 and sp:
        title = data_point["context"][para_id][0]
        sps[unique_id].append([title, int(sent_id)])

    # Predicts answer text.
    answer_type = raw_prediction["answer_types"]
    if answer_type == 0:
      answers[unique_id] = get_top1_answer(raw_prediction, data_point,
                                           max_answer_length, supporting_facts,
                                           tokenizer, use_wordpiece,
                                           answer_encoding_method)
    elif answer_type == 1:
      answers[unique_id] = "yes"
    else:
      answers[unique_id] = "no"

  return {"answer": answers, "sp": sps}
