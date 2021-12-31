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

"""This modules demonstrates how to convert code to subtokenized sentences."""
import itertools
from typing import List, Sequence, Tuple


from absl import logging
from tensor2tensor.data_generators import text_encoder


from cubert import cubert_tokenizer
from cubert import unified_tokenizer


def wordpiece_ids_from_wordpiece_tokens(
    wordpiece_subtokens,
    subword_tokenizer):
  return tuple(
      subword_tokenizer._subtoken_string_to_id[w]  # pylint: disable=protected-access
      for w in wordpiece_subtokens)


def next_whole_token(
    wordpiece_subtokens,
    initial_tokenizer,
    subword_tokenizer):
  """Greedily reconstitutes a whole token from a WordPiece list.

  This function assumes that the wordpiece subtokens were constructed correctly
  from a correctly subtokenized CuBERT tokenizer, but the sequence may be
  truncated and thus incomplete.

  The implementation is done in two stages: recognizing the first whole token
  and then finding the correspondence of that first whole token to a prefix of
  the subtoken sequence.

  The implementation assumes that untokenization can do the best job on the full
  context. So, it first untokenizes the whole sequence, and chooses the first
  whole token.

  To figure out the subtoken prefix that corresponds to that whole token, the
  implementation greedily untokenizes longer and longer subtoken prefixes, until
  the whole token is recognized in the output.

  The reason for this somewhat expensive implementation is that the logic for
  merging subtokens (for WordPiece and then for CuBERT) is intricate, and does
  not export how many initial subtokens were consumed for each output token of
  the next higher abstraction. What's more, a subtoken may align itself with
  the previous or the next whole token, when the subtoken sequence is
  incomplete.

  Args:
    wordpiece_subtokens: The subtokens to scan through.
    initial_tokenizer: A CuBERT tokenizer.
    subword_tokenizer: A SubwordTextEncoder.

  Returns:
    The first whole token matched, and the end index of the first subtoken index
    after the first whole token. wordpiece_subtokens[0:end_index] should be
    the subtokens corresponding to the whole token returned.

  Raises:
    ValueError if no whole token can be parsed.
  """

  wordpiece_ids = wordpiece_ids_from_wordpiece_tokens(wordpiece_subtokens,
                                                      subword_tokenizer)
  full_cubert_subtokens: List[str] = (
      subword_tokenizer._subtoken_ids_to_tokens(  # pylint: disable=protected-access
          wordpiece_ids))

  full_cubert_subtokens.append(
      unified_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS.name))

  full_whole_tokens = initial_tokenizer.untokenize_agnostic(
      full_cubert_subtokens)

  if len(full_whole_tokens) < 2:
    # It all came out a jumble. Reject it.
    raise ValueError(f'Whole tokens {full_whole_tokens} ended up '
                     f'undifferentiable in {wordpiece_subtokens}.')

  whole_token = full_whole_tokens[0]

  for end_index in range(1, len(wordpiece_ids) + 1):
    prefix_list = wordpiece_ids[:end_index]
    partial_cubert_subtokens: List[str] = (
        subword_tokenizer._subtoken_ids_to_tokens(  # pylint: disable=protected-access
            prefix_list))

    # We strip EOS in `code_to_cubert_sentences`, so we have to add it back
    # here.
    partial_cubert_subtokens.append(
        unified_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS.name))

    partial_whole_tokens = initial_tokenizer.untokenize_agnostic(
        partial_cubert_subtokens)
    if len(partial_whole_tokens) > 1:
      if partial_whole_tokens[0] == whole_token:
        return whole_token, end_index

  # We got here because we couldn't match the whole token we found from the
  # full sequence
  raise ValueError('Could not find a whole token in %r' %
                   (wordpiece_subtokens,))


def wordpiece_subtokens_to_code(
    wordpiece_subtokens,
    initial_tokenizer,
    subword_tokenizer):
  """Reverses the Wordpiece-to-CuBERT Subtoken-to-whole token conversion."""
  # We have to map WordPiece subtoken strings back to WordPiece vocabulary IDs.
  wordpiece_ids = wordpiece_ids_from_wordpiece_tokens(wordpiece_subtokens,
                                                      subword_tokenizer)

  return wordpiece_ids_to_code(wordpiece_ids, initial_tokenizer,
                               subword_tokenizer)


def wordpiece_ids_to_code(
    wordpiece_ids,
    initial_tokenizer,
    subword_tokenizer):
  """Reverses the Wordpiece-to-CuBERT Subtoken-to-whole token conversion."""
  cubert_subtokens: List[str] = (
      subword_tokenizer._subtoken_ids_to_tokens(  # pylint: disable=protected-access
          wordpiece_ids))

  # We strip EOS in `code_to_cubert_sentences`, so we have to add it back here.
  cubert_subtokens.append(
      unified_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS.name))

  code = initial_tokenizer.untokenize(cubert_subtokens)
  return code


def code_to_cubert_sentences(
    code,
    initial_tokenizer,
    subword_tokenizer,
):
  """Tokenizes code into a list of CuBERT sentences.

  Args:
    code: The source code to tokenize. This must be a parseable unit of code,
      meaning it represents an AST (or a complete subtree of an AST). For
      example, there should be no unmatched parentheses, and `if` and other
      blocks of code must have bodies.
    initial_tokenizer: The first tokenizer that creates sentences, probably a
      cubert_tokenizer.CuBertTokenizer.
    subword_tokenizer: A second tokenizer that splits tokens of the
      `initial_tokenizer` into subtokens.

  Returns:
    A list of sentences.
  """
  tokens: Sequence[str] = initial_tokenizer.tokenize(code)[:-1]
  logging.vlog(5, 'Code >>>%s<<< is tokenized into >>>%s<<<.', code, tokens)

  # This will split the list into sublists of non-NEWLINE tokens (key is
  # False) and NEWLINE tokens (key is True).
  groups_by_endtoken = itertools.groupby(
      tokens, key=lambda x: x == unified_tokenizer.NEWLINE)
  # This will produce raw sentences from the groups. For groups that were made
  # of non-NEWLINE tokens, the entire group makes up a single raw sentence.
  # For groups of NEWLINE tokens (e.g., generated from consecutive empty lines),
  # we ignore the first NEWLINE (since it terminated a preceding sentence) and
  # create empty sentences for each of the rest.
  #
  # For example, [a, b, NEWLINE, c, d, NEWLINE] will yield the raw sentences
  # [a, b], and [c, d].
  # Instead, [a, b, NEWLINE, NEWLINE, NEWLINE, c, d, NEWLINE] will yield the
  # raw sentences [a, b], [], [], [c, d].
  #
  # We call these raw_sentences, because they're not terminated.
  raw_sentences: List[List[str]] = []
  for i, (k, v) in enumerate(groups_by_endtoken):
    tokens = list(v)
    if k:
      # True `k` means this was a NEWLINE group.
      if i == 0:
        # This was the first group, and a NEWLINE group. There was no group of
        # non-NEWLINE tokens before it, therefore we turn all NEWLINE tokens
        # into empty sentences.
        raw_sentences.extend([[]] * len(tokens))
      else:
        # This wasn't the first group. The non-NEWLINE group that preceded this
        # one consumes one of the NEWLINEs in this group, as its terminator.
        # The rest of the NEWLINEs, if any remain, turn into empty sentences.
        if len(tokens) == 1:
          # This is just a terminator of the previous sentence. Just throw it
          # out.
          continue
        elif len(tokens) > 1:
          # This is a chain of NEWLINE tokens. We throw out the first, but we
          # add empty sentences for the rest.
          raw_sentences.extend([[]] * (len(tokens) - 1))
        else:
          # We shouldn't have a group of empty tokens. This should not happen.
          raise AssertionError('itertools.groupby seems to have returned an '
                               'empty group: %r' % tokens)
    else:
      # False means this was a non-NEWLINE group. Create a single raw sentence
      # out of it.
      raw_sentences.append(tokens)

  # Now we append a NEWLINE token after all sentences. Note that our tokenizer
  # drops any trailing \n's before tokenizing, but for the purpose of forming
  # properly terminated sentences, we always end sentences in a NEWLINE token.
  sentences = [s + [unified_tokenizer.NEWLINE] for s in raw_sentences
              ]  # type: List[List[str]]
  logging.vlog(5, 'Tokens are split into sentences: >>>%s<<<.',
               sentences)

  # Now we have to encode tokens using the subword text encoder, expanding the
  # sentences.
  subtokenized_sentences = []  # type: List[List[str]]
  for sentence in sentences:
    encoded_tokens = [subword_tokenizer.encode_without_tokenizing(t)
                      for t in sentence]  # type: List[List[int]]
    logging.vlog(5, 'Sentence encoded into >>>%s<<<.', encoded_tokens)
    flattened_encodings = sum(encoded_tokens, [])  # type: List[int]
    logging.vlog(5, 'Flattened into >>>%s<<<.', flattened_encodings)
    decoded_tokens = subword_tokenizer.decode_list(
        flattened_encodings)  # type: List[str]
    logging.vlog(5, 'Sentence re-decoded into >>>%s<<<.', decoded_tokens)

    subtokenized_sentences.append(decoded_tokens)
  logging.vlog(5, 'Sentences are further subtokenized: >>>%s<<<.',
               subtokenized_sentences)

  return subtokenized_sentences
