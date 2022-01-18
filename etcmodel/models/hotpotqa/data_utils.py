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

"""Data processing utility functions."""
from typing import Tuple, List, Sequence

import attr
import six

from etcmodel.models import tokenization

_WHITESPACE_DELIMITER = u" \t\r\n\u202f"  # \u202f corresponds to ""


@attr.s(auto_attribs=True)
class TokenizedText:
  """Tokenized text with indices mappings."""
  # The original text.
  text: str = ""
  # The Wordpiece tokenized text.
  tokens: List[str] = attr.Factory(list)
  # The Wordpiece token ids.
  token_ids: List[int] = attr.Factory(list)
  # The whitespace tokenized text.
  unigrams: List[str] = attr.Factory(list)
  # The indices mapping from chars to unigrams. The char at index `i` belongs to
  # the unigram at index `chars_to_unigrams[i]`. A whitespace belongs to the
  # previous unigram. Only used with WordPiece tokenizer.
  chars_to_unigrams: List[int] = attr.Factory(list)
  # The indices mapping from unigrams to tokens. The unigram at index `i` starts
  # at the Wordpiece token at index `unigrams_to_tokens[i]`. Only used with
  # WordPiece tokenizer.
  unigrams_to_tokens: List[int] = attr.Factory(list)
  # The indices mapping from tokens to unigrams. The token at index `i` belongs
  # to the unigram at index `tokens_to_unigrams[i]`. Only used with WordPiece
  # tokenizer.
  tokens_to_unigrams: List[int] = attr.Factory(list)
  # The indices mapping from chars to tokens. The char at index `i` belongs to
  # the token at index `char_to_token_index[i]`. A whitespace belongs to the
  # later token. Note that the `text` stored in this class is obtained from
  # first SentencePiece tokenize the input text, then detokenize the tokens.
  # Only used with SentencePiece tokenizer.
  chars_to_tokens: List[int] = attr.Factory(list)


def whitespace_split_with_indices(
    text: str) -> Tuple[List[str], List[int], List[int]]:
  """Whitespace splits a text into unigrams and returns indices mapping."""
  if not isinstance(text, str):
    raise ValueError("The input text is not of unicode format.")
  unigrams = []
  unigram_to_char_map = []
  char_to_unigram_map = []
  prev_is_separator = True
  for i, c in enumerate(text):
    if c in _WHITESPACE_DELIMITER:
      prev_is_separator = True
    else:
      if prev_is_separator:
        unigrams.append(c)
        unigram_to_char_map.append(i)
      else:
        unigrams[-1] += c
      prev_is_separator = False
    char_to_unigram_map.append(len(unigrams) - 1)
  return unigrams, unigram_to_char_map, char_to_unigram_map


def wordpiece_tokenize_with_indices(
    doc_unigrams: Sequence[str], tokenizer: tokenization.FullTokenizer
) -> Tuple[List[str], List[int], List[int]]:
  """Wordpiece tokenizes unigrams to tokens and returns indices mapping."""
  token_to_unigram_map = []
  unigram_to_token_map = []
  doc_tokens = []
  for (i, token) in enumerate(doc_unigrams):
    unigram_to_token_map.append(len(doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    token_to_unigram_map.extend([i] * len(sub_tokens))
    doc_tokens.extend(sub_tokens)
  return doc_tokens, unigram_to_token_map, token_to_unigram_map


def get_wordpiece_tokenized_text(
    text: str, tokenizer: tokenization.FullTokenizer) -> TokenizedText:
  """Gets WordPiece TokenizedText for a text with indices mapping."""
  unigrams, _, chars_to_unigrams = whitespace_split_with_indices(text)
  tokens, unigrams_to_tokens, tokens_to_unigrams = (
      wordpiece_tokenize_with_indices(unigrams, tokenizer))
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  tokenized_text = TokenizedText()
  tokenized_text.text = text
  tokenized_text.tokens = tokens
  tokenized_text.token_ids = token_ids
  tokenized_text.unigrams = unigrams
  tokenized_text.chars_to_unigrams = chars_to_unigrams
  tokenized_text.unigrams_to_tokens = unigrams_to_tokens
  tokenized_text.tokens_to_unigrams = tokens_to_unigrams
  return tokenized_text


def sentencepiece_detokenize(tokens: Sequence[str]) -> str:
  """Recovers SenencePiece token to original text, with whitespace removal."""
  spiece_token = tokenization.SPIECE_UNDERLINE.decode("utf-8")
  tokens = list(tokens)
  if tokens and tokens[0].startswith(spiece_token):
    tokens[0] = tokens[0][1:]
  return "".join(tokens).replace(spiece_token, " ")


def get_sentencepiece_tokenized_text(
    text: str, tokenizer: tokenization.FullTokenizer) -> TokenizedText:
  """Gets SentencePiece TokenizedText for a text with indices mapping."""
  tokens = [six.ensure_text(tk, "utf-8") for tk in tokenizer.tokenize(text)]
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  chars_to_tokens = []
  for i, token in enumerate(tokens):
    num_chars = len(token)
    if i == 0:
      num_chars -= 1
    chars_to_tokens.extend([i] * num_chars)
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  tokenized_text = TokenizedText()
  tokenized_text.text = sentencepiece_detokenize(tokens)
  tokenized_text.tokens = tokens
  tokenized_text.token_ids = token_ids
  tokenized_text.chars_to_tokens = chars_to_tokens
  return tokenized_text


def find_char_spans(text: str, substring: str) -> List[Tuple[int, int]]:
  """Finds all substring occurrence char level spans (inclusive)."""
  if not substring:
    return []
  char_spans = []
  char_begin = text.find(substring)
  while char_begin != -1:
    char_end = char_begin + len(substring) - 1
    char_spans.append((char_begin, char_end))
    char_begin = text.find(substring, char_end + 1)
  return char_spans


def _improve_answer_span(
    doc_tokens: Sequence[str],
    unimproved_span: Tuple[int, int],
    orig_answer_text: str,
    tokenizer: tokenization.FullTokenizer,
):
  """Returns answer token spans that better match the annotated answer.

  This function is branched from the original BERT `run_squad.py` code

  Usually question answer span annotations are character based. We first project
  them to whitespace-tokenized words (unigrams). But then after WordPiece
  tokenization, we can often find a "better match". For example:

    Question: What year was John Smith born?
    Context: The leader was John Smith (1895-1943).
    Answer: 1895

  The original whitespace-tokenized answer will be "(1895-1943).". However
  after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  the exact answer, 1895. The purpose of this function is to find such "better
  match".

  However, this is not always possible. Consider the following:

    Question: What country is the top exporter of electornics?
    Context: The Japanese electronics industry is the lagest in the world.
    Answer: Japan

  In this case, the annotator chose "Japan" as a character sub-span of
  the word "Japanese". Since our WordPiece tokenizer does not split
  "Japanese", we just use "Japanese" as the annotation. This is expected to be
  fairly rare.

  Args:
    doc_tokens: Sequence of Text, the wordpiece tokenized tokens of the doc.
    unimproved_span: Tuple of two ints, the unimproved answer token span. In the
      first example, it is the token span for "(" and ")".
    orig_answer_text: Text, the original answer text. In the first example, it
      is "1895".
    tokenizer: FullTokenizer, wordpiece tokenizer to tokenize the original
      answer text.

  Returns:
    Tuple of two ints, the improved answer token span. In the first example, it
    corresponds to the answer token span for "1895".
  """
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
  for new_begin in range(unimproved_span[0], unimproved_span[1] + 1):
    for new_end in range(unimproved_span[1], new_begin - 1, -1):
      text_span = " ".join(doc_tokens[new_begin:(new_end + 1)])
      if text_span == tok_answer_text:
        return new_begin, new_end

  return unimproved_span


def _convert_answer_spans(answer_unigram_spans: Sequence[Tuple[int, int]],
                          unigram_to_token_map: Sequence[int],
                          num_tokens: int) -> List[Tuple[int, int]]:
  """Converts answer unigram spans to token spans."""
  answer_token_spans = []
  for unigram_begin, unigram_end in answer_unigram_spans:
    token_begin = unigram_to_token_map[unigram_begin]
    if unigram_end + 1 < len(unigram_to_token_map):
      token_end = unigram_to_token_map[unigram_end + 1] - 1
    else:
      token_end = num_tokens - 1
    answer_token_spans.append((token_begin, token_end))
  return answer_token_spans


def find_answer_spans_wordpiece(
    tokenized_context: TokenizedText, answer: str,
    tokenizer: tokenization.FullTokenizer) -> List[Tuple[int, int]]:
  """Finds all answer occurrence WordPiece token spans (inclusive).

  Args:
    tokenized_context: WordPiece tokenized context with indices mapping.
    answer: Answer string.
    tokenizer: A WordPiece tokenizer.

  Returns:
    A list of (begin, end) WordPiece token level indices (inclusive) of all the
    answer occurrences in the context. If the answer is empty or there is no
    answer occurrence in the context, return empty list.
  """
  # The answer occurrence always corresponds to char level occurrence.
  # This is to avoid the following case,
  #     context: "..Italian composer who wrote 39 operas.."
  #     answer:  "opera"
  # Since both "operas" and "opera" are in the vocab, simply searching token
  # level spans will miss such kind of occurrence.
  token_spans = []
  for char_begin, char_end in find_char_spans(tokenized_context.text, answer):
    unigram_span = (tokenized_context.chars_to_unigrams[char_begin],
                    tokenized_context.chars_to_unigrams[char_end])
    unimproved_token_span = _convert_answer_spans(
        [unigram_span], tokenized_context.unigrams_to_tokens,
        len(tokenized_context.tokens))[0]
    token_spans.append(
        _improve_answer_span(tokenized_context.tokens, unimproved_token_span,
                             answer, tokenizer))
  return token_spans


def find_answer_spans_sentencepiece(tokenized_context: TokenizedText,
                                    answer: str) -> List[Tuple[int, int]]:
  """Finds all answer occurrence SentencePiece token spans (inclusive).

  Args:
    tokenized_context: SentencePiece tokenized context with indices mapping.
    answer: Answer string.

  Returns:
    A list of (begin, end) WordPiece token level indices (inclusive) of all the
    answer occurrences in the context. If the answer is empty or there is no
    answer occurrence in the context, return empty list.
  """
  # The answer occurrence always corresponds to char level occurrence.
  # This is to avoid the following case,
  #     context: "..Italian composer who wrote 39 operas.."
  #     answer:  "opera"
  # Since both "operas" and "opera" are in the vocab, simply searching token
  # level spans will miss such kind of occurrence.
  token_spans = []
  for char_begin, char_end in find_char_spans(tokenized_context.text, answer):
    token_spans.append((tokenized_context.chars_to_tokens[char_begin],
                        tokenized_context.chars_to_tokens[char_end]))
  return token_spans


def wordpiece_tokens_to_normalized_text(wordpiece_tokens: Sequence[str]) -> str:
  """Concatenates wordpiece tokens to a normalized text and cleans up.

  The wordpiece tokens are results from BERT tokenization. They may contain
  symbols of '##' or ' ##' and some extra whitespaces. The function first
  concatenate the tokens and then removes those extrac symbols and whitespaces.

  Args:
    wordpiece_tokens: A sequence of wordpiece tokens from BERT tokenization.

  Returns:
    The text by concatenating the wordpiece tokens and cleaning up.
  """
  text = " ".join(wordpiece_tokens)

  # De-tokenize WordPieces that have been split off.
  text = text.replace(" ##", "")
  text = text.replace("##", "")

  # Clean whitespace
  text = text.strip()
  text = " ".join(text.split())
  return text
