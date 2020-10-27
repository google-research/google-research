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

r"""A program to convert NQ data to TF examples for XLNet style processing."""

import collections
import json
import math
import random

import apache_beam as beam
import numpy as np
import tensorflow.compat.v1 as tf

from etcmodel.models import tokenization


class NQPreprocessor(object):
  """Class for NQ example preprocessing.

  The generated TFExamples have the following features:
  - token_ids: sequence of word-piece token ids in the ETC long input.
  - long_breakpoints: indicates the end of the ETC long input with a 1 in the
      last non-padding position.
  - candidate_ids: the ID of the sentence in the original document corresponding
      to each token in the long input.
  - token_pos: index of the corresponding word in the original document for each
      token in the long input.
  - sentence_ids: used for block-based ETC attention, index of which token from
      the global input corresponds to each of the tokens in the long input.

  - global_token_ids: global input for ETC.
  - global_type_id: the "type" of teach of the tokens in the global input (this
      can be CLS / question / segment).
  - global_candidate_ids: the ID of the sentence in the original document
      corresponding to each token in the global input.

  - global_breakpoints: indicates the end of the ETC global input with a 1 in
      the last non-padding position.
  - global_token_pos_end: index of the first word in the original document
      sentence corresponding to each of the tokens in the global input.
  - global_token_pos_start: index of the last word in the original document
      sentence corresponding to each of the tokens in the global input.

  - answer_type: 0: no answer, 1: "yes", 2: "no", 3: long, 4: short
  - sa_start / sa_end: ground truth for short answer prediction. Index of the
      long input token where the short answer starts or ends. If there is no
      short answer or if the start / end is outside of the window of this
      TFExample, then the index is 0 (pointing to the CLS token)
  - la_start / la_end: same as sa_start / sa_end but for the long answer.
  - la_global: ground truth for long answer, indicating which token from the
      global input corresponds to the ground truth long answer sentence. This
      is not used by the current code.

  - unique_ids: the ID of the NQ example from where this TFExample was
      generated.
  - question_ids: word-piece token ids of the question.
  """

  def __init__(self, stride, seq_len, global_seq_len, question_len, vocab_file,
               do_lower_case, predict_la_when_no_sa, include_unknown_rate,
               include_unknown_rate_for_unanswerable,
               include_html_tokens, global_token_types,
               spm_model_path, tokenizer_type, is_train, fixed_blocks=False,
               fixed_block_size=27, global_size_counter=None,
               long_size_counter=None,
               global_size_threshold_counters=None,
               global_sentence_counter=None, long_sentence_tokens_counter=None):
    if tokenizer_type == "BERT":
      # Use BERT tokenization:
      self.tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=do_lower_case)
      self.sos_id = self.tokenizer.vocab["[unused102]"]
      self.eos_id = self.tokenizer.vocab["[unused103]"]
      self.pad_id = self.tokenizer.vocab["[PAD]"]
      self.cls_id = self.tokenizer.vocab["[CLS]"]
      self.sep_id = self.tokenizer.vocab["[SEP]"]
    elif tokenizer_type == "ALBERT":
      # Use ALBERT SentencePiece tokenization:
      # Notice that 'vocab_file' and 'do_lower_case' are ignored when
      # 'spm_model_path' is not None
      self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case,
                                                  spm_model_path)
      self.sos_id = self.tokenizer.vocab["<unused_35>"]
      self.eos_id = self.tokenizer.vocab["<unused_36>"]
      self.pad_id = self.tokenizer.vocab["<pad>"]
      self.cls_id = self.tokenizer.vocab["<unused_63>"]
      self.sep_id = self.tokenizer.vocab["<unused_2>"]
    else:
      raise ValueError("Only 'BERT' and 'ALBERT' are supported: %s" %
                       (tokenizer_type))

    self.answer_type_enum = {
        "NULL": 0,
        "YES": 1,
        "NO": 2,
        "LONG": 3,
        "SHORT": 4
    }
    self.seq_len = seq_len
    self.question_len = question_len
    self.stride = stride
    self.predict_la_when_no_sa = predict_la_when_no_sa
    self.include_unknown_rate = include_unknown_rate
    if include_unknown_rate_for_unanswerable is None:
      self.include_unknown_rate_for_unanswerable = include_unknown_rate*4
    else:
      self.include_unknown_rate_for_unanswerable = (
          include_unknown_rate_for_unanswerable)
    self.include_html_tokens = include_html_tokens
    self.global_seq_len = global_seq_len
    self.gt_type_sentence = global_token_types[0]
    self.gt_type_cls = global_token_types[1]
    self.gt_type_question = global_token_types[2]
    self.is_train = is_train
    self.fixed_blocks = fixed_blocks
    self.fixed_block_size = fixed_block_size
    self.question_ids_in_long = True
    self.cls_in_long = True
    self.global_cls_id = self.cls_id
    # 35 corresponds to "unused34" both in BERT (uncased) and ALBERT vocabs,
    # it will be "unused35" in BERT cased.
    self.global_question_id = 35
    self.global_sentence_id = 1
    self._global_size_counter = global_size_counter
    self._long_size_counter = long_size_counter
    if global_size_threshold_counters is None:
      self._global_size_threshold_counters = []
    else:
      self._global_size_threshold_counters = global_size_threshold_counters
    self.global_sentence_counter = global_sentence_counter
    self.long_sentence_tokens_counter = long_sentence_tokens_counter

  def to_tf_example(self, nq_example):
    """Converts an NQ example to a tf record."""
    question_ids = self.tokenizer.convert_tokens_to_ids(
        self.tokenizer.tokenize(nq_example["question_text"]))
    token_ids = []
    candidate_ids = []
    token_pos = []
    for c_idx, c in enumerate(nq_example["long_answer_candidates"]):
      if not bool(c["top_level"]):
        continue
      for pos in range(c["start_token"], c["end_token"]):
        t = nq_example["document_tokens"][pos]
        if bool(t["html_token"]):
          if self.include_html_tokens:
            if t["token"].startswith("</"):
              new_ids = [self.eos_id]
            else:
              new_ids = [self.sos_id]
          else:
            new_ids = []
        else:
          new_ids = self.tokenizer.convert_tokens_to_ids(
              self.tokenizer.tokenize(t["token"]))
        token_ids.extend(new_ids)
        candidate_ids.extend([c_idx] * len(new_ids))
        token_pos.extend([pos] * len(new_ids))
    la_start = []
    la_end = []
    sa_start = []
    sa_end = []
    answer_type = []

    def _has_short_answer(a):
      return bool(a["short_answers"])

    def _has_long_answer(a):
      return (a["long_answer"]["start_token"] >= 0 and
              a["long_answer"]["end_token"] >= 0)

    def _is_yes_no_answer(a):
      return a["yes_no_answer"] in ("YES", "NO")

    for a in nq_example["annotations"][:5]:
      la_start.append(a["long_answer"]["start_token"])
      la_end.append(a["long_answer"]["end_token"])
      if _has_short_answer(a):
        # Note that this merges short answers with multiple spans.
        sa_start.append(a["short_answers"][0]["start_token"])
        sa_end.append(a["short_answers"][-1]["end_token"])
      elif _has_long_answer(a) and self.predict_la_when_no_sa:
        sa_start.append(a["long_answer"]["start_token"])
        sa_end.append(a["long_answer"]["end_token"])
      else:
        sa_start.append(-1)
        sa_end.append(-1)

      # Sets answer type.
      if _is_yes_no_answer(a):
        answer_type.append(self.answer_type_enum[a["yes_no_answer"]])
      elif _has_short_answer(a):
        answer_type.append(self.answer_type_enum["SHORT"])
      elif _has_long_answer(a):
        answer_type.append(self.answer_type_enum["LONG"])
      else:
        answer_type.append(self.answer_type_enum["NULL"])

    # Populate dummy labels if the example had no annotations.
    for list_ in (la_start, la_end, sa_start, sa_end, answer_type):
      if not list_:
        list_.append(-1)

    def make_int64_feature(v):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

    y = collections.OrderedDict()
    y["unique_ids"] = make_int64_feature([nq_example["example_id"]])
    y["question_ids"] = make_int64_feature(question_ids)
    y["token_ids"] = make_int64_feature(token_ids)
    y["candidate_ids"] = make_int64_feature(candidate_ids)
    y["token_pos"] = make_int64_feature(token_pos)
    y["la_start"] = make_int64_feature(la_start)
    y["la_end"] = make_int64_feature(la_end)
    y["sa_start"] = make_int64_feature(sa_start)
    y["sa_end"] = make_int64_feature(sa_end)
    y["answer_type"] = make_int64_feature(answer_type)
    return tf.train.Example(features=tf.train.Features(feature=y))

  def split_example(self, tf_example):
    """Yields fixed size instances from a TF example."""
    features = tf_example.features.feature
    question_ids = features["question_ids"].int64_list.value[:self.question_len]
    token_ids = features["token_ids"].int64_list.value
    candidate_ids = features["candidate_ids"].int64_list.value
    candidate_ids_np = np.array(candidate_ids)
    token_pos = features["token_pos"].int64_list.value

    # Note that the ends here become inclusive.
    orig_sa = (features["sa_start"].int64_list.value[0],
               features["sa_end"].int64_list.value[0] - 1)
    orig_la = (features["la_start"].int64_list.value[0],
               features["la_end"].int64_list.value[0] - 1)

    if self.question_ids_in_long:
      # The "-2" comes from the 2 SEPs
      doc_len = self.seq_len - len(question_ids) - 2
      if self.cls_in_long:
        doc_len -= 1
    else:
      # The "-1" comes from the final SEP
      doc_len = self.seq_len - 1
      if self.cls_in_long:
        doc_len -= 1

    def make_int64_feature(v):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

    for start in range(0,
                       max(len(token_ids) - doc_len, 0) + self.stride,
                       self.stride):
      y = collections.OrderedDict()

      # If self.question_ids_in_long and self.cls_in_long, long input is this:
      #   [CLS], question, [SEP], document chunk, [SEP].
      ids = token_ids[start:start + doc_len]
      pad_len = doc_len - len(ids)
      if self.question_ids_in_long:
        if self.cls_in_long:
          long_preamble_ids = [self.cls_id] + question_ids + [self.sep_id]
        else:
          long_preamble_ids = question_ids + [self.sep_id]
      else:
        if self.cls_in_long:
          long_preamble_ids = [self.cls_id]
        else:
          long_preamble_ids = []

      y["token_ids"] = make_int64_feature(long_preamble_ids + ids +
                                          [self.sep_id] +
                                          [self.pad_id] * pad_len)
      # This contains the candidate id for every position in the long input.
      y["candidate_ids"] = make_int64_feature([-1] * len(long_preamble_ids) +
                                              candidate_ids[start:start +
                                                            doc_len] + [-1] *
                                              (pad_len + 1))

      # This maps every position in the window to the original token index.
      # Note that `pos` needs to be a numpy array so we can use argmax below.
      pos = np.array([-1] * len(long_preamble_ids) +
                     token_pos[start:start + doc_len] + [-1] * (pad_len + 1))
      y["token_pos"] = make_int64_feature(pos)

      # This should have a 1 corresponding to the last valid token id.
      y["long_breakpoints"] = make_int64_feature([0] *
                                                 (self.seq_len - 1 - pad_len) +
                                                 [1] + [0] * pad_len)

      # Check that lengths are correct.
      assert len(y["token_ids"].int64_list.value) == self.seq_len
      assert len(y["long_breakpoints"].int64_list.value) == self.seq_len
      assert len(y["candidate_ids"].int64_list.value) == self.seq_len
      assert len(y["token_pos"].int64_list.value) == self.seq_len

      # The following functions converts start and end positions for long
      # and short answer from word indices in the original document to word
      # piece indices in the current instance. Note that when a position is not
      # found both find functions return the index 0.

      # This returns the word piece position of the start of the matching word.
      def forward_find(p, pos=pos):
        return np.argmax(pos == p)

      # This returns the word piece position of the end of the matching word.
      def reverse_find(p, pos=pos):
        i = np.argmax(pos == p)
        while i < self.seq_len - 1 and pos[i + 1] == p:
          i += 1
        return i

      # This returns the word piece position of the start of the matching
      # sentence in the original document.
      def forward_find_sentence(p, candidate_ids_np=candidate_ids_np):
        i = int(np.argmax(candidate_ids_np == p))
        return token_pos[i]

      # This returns the word piece position of the end of the matching
      # sentence in the original document.
      def reverse_find_sentence(p, candidate_ids_np=candidate_ids_np):
        i = int(np.argmax(candidate_ids_np == p))
        while i < len(candidate_ids) - 1 and candidate_ids_np[i + 1] == p:
          i += 1
        return token_pos[i]

      y["sa_start"] = make_int64_feature([forward_find(orig_sa[0])])
      y["sa_end"] = make_int64_feature([reverse_find(orig_sa[1])])
      la_start_idx = forward_find(orig_la[0])
      y["la_start"] = make_int64_feature([la_start_idx])
      y["la_end"] = make_int64_feature([reverse_find(orig_la[1])])

      # If the answer is outside of this chunk, then overwrite the answer type
      # to be NULL.
      answer_type = features["answer_type"].int64_list.value[0]
      if (answer_type == self.answer_type_enum["SHORT"] and
          y["sa_start"].int64_list.value[0] == 0 and
          y["sa_end"].int64_list.value[0] == 0):
        y["answer_type"] = make_int64_feature([0])
      elif (answer_type == self.answer_type_enum["LONG"] and
            y["la_start"].int64_list.value[0] == 0 and
            y["la_end"].int64_list.value[0] == 0):
        y["answer_type"] = make_int64_feature([0])
      else:
        y["answer_type"] = make_int64_feature([answer_type])

      # Global token ids contain the query followed by a 1 token for every
      # candidate context. The sentence ids map each input token the
      # corresponding 1 in the global token ids.
      if self.question_ids_in_long:
        if self.cls_in_long:
          sentence_ids = list(range(len(question_ids) + 2))
        else:
          sentence_ids = list(range(len(question_ids) + 1))

        global_token_ids = [self.global_cls_id] + ([self.global_question_id] *
                                                   (len(question_ids) + 1))
        global_type_ids = [self.gt_type_cls] + ([self.gt_type_question] *
                                                (len(question_ids) + 1))
      else:
        sentence_ids = []
        if self.cls_in_long:
          sentence_ids += [0]
        # a 1 for the CLS token, and then the quesion word pieces
        global_token_ids = [self.global_cls_id] + question_ids
        global_type_ids = [self.gt_type_cls] + ([self.gt_type_question] *
                                                len(question_ids))

      # These features store the start/end word indexes of the sentences
      # represented by each of the global tokens (with (0,0) for the CLS and
      # question tokens):
      global_token_pos_start = [0]*len(global_token_ids)
      global_token_pos_end = [0]*len(global_token_ids)
      global_candidate_ids = [-1]*len(global_token_ids)

      n_global_sentences = 0

      for idx in range(len(sentence_ids), self.seq_len):
        if idx > 0:
          prev_candidate_id = y["candidate_ids"].int64_list.value[idx - 1]
        else:
          prev_candidate_id = -1
        candidate_id = y["candidate_ids"].int64_list.value[idx]
        if (candidate_id != prev_candidate_id and
            len(global_token_ids) < self.global_seq_len):
          if candidate_id >= 0:
            n_global_sentences += 1
            global_token_ids += [self.global_sentence_id]
            global_type_ids += [self.gt_type_sentence]
            global_token_pos_start += [forward_find_sentence(candidate_id)]
            global_token_pos_end += [reverse_find_sentence(candidate_id)]
          else:
            global_token_ids += [0]
            global_type_ids += [0]
            global_token_pos_start += [0]
            global_token_pos_end += [0]
          global_candidate_ids += [candidate_id]
        sentence_ids.append(len(global_token_ids) - 1)

      if self.fixed_blocks:
        # Overwrite 'sentence_ids' and 'global_token_ids' to represent a flat
        # "1 global token per "fixed_block_size" long tokens.
        # ('global_breakpoints' is modified accordingly already automatically
        # after this).
        n_long_tokens = self.seq_len - pad_len
        n_global_tokens = min(self.global_seq_len,
                              int(math.ceil(n_long_tokens /
                                            float(self.fixed_block_size))))
        global_token_ids = [self.global_sentence_id] * n_global_tokens
        sentence_ids = []
        for i in range(self.seq_len):
          sentence_id = i // self.fixed_block_size
          if i > n_long_tokens:
            sentence_ids.append(0)  # pad
          else:
            sentence_ids.append(sentence_id)

      if self.global_sentence_counter is not None:
        self.global_sentence_counter.update(n_global_sentences)
      if self.long_sentence_tokens_counter is not None:
        self.long_sentence_tokens_counter.update(len(ids))

      y["sentence_ids"] = make_int64_feature(sentence_ids)
      y["global_token_ids"] = make_int64_feature(
          global_token_ids + [0] *
          (self.global_seq_len - len(global_token_ids)))
      y["global_breakpoints"] = make_int64_feature(
          [0] * (len(global_token_ids) - 1) + [1] + [0] *
          (self.global_seq_len - len(global_token_ids)))

      if not self.fixed_blocks:
        # global_token_pos_start, global_token_pos_end, and
        # global_candidate_ids only make sense if self.fixed_blocks == False.
        y["global_token_pos_start"] = make_int64_feature(
            global_token_pos_start + [0] *
            (self.global_seq_len - len(global_token_pos_start)))
        y["global_token_pos_end"] = make_int64_feature(
            global_token_pos_end + [0] *
            (self.global_seq_len - len(global_token_pos_end)))
        y["global_candidate_ids"] = make_int64_feature(
            global_candidate_ids + [-1] *
            (self.global_seq_len - len(global_candidate_ids)))

      if (y["la_start"].int64_list.value[0] != 0 or
          y["la_end"].int64_list.value[0] != 0) and la_start_idx >= 0:
        y["la_global"] = make_int64_feature([sentence_ids[la_start_idx]])
      else:
        # if there is no long answer, point the label to the CLS token:
        y["la_global"] = make_int64_feature([0])

      # collect statistics of global memory usage:
      if self._global_size_counter is not None:
        self._global_size_counter.update(len(global_token_ids))
      if self._long_size_counter is not None:
        self._long_size_counter.update(self.seq_len - pad_len)
      for threshold, counter in self._global_size_threshold_counters:
        if len(global_token_ids) >= threshold:
          counter.inc()

      y["global_type_id"] = make_int64_feature(
          global_type_ids + [0] *
          (self.global_seq_len - len(global_token_ids)))

      assert len(y["sentence_ids"].int64_list.value) == self.seq_len
      assert len(y["global_token_ids"].int64_list.value) == self.global_seq_len
      assert len(
          y["global_breakpoints"].int64_list.value) == self.global_seq_len

      # Pass through all remaining features.
      for name in features.keys():
        if name not in y:
          y[name] = make_int64_feature(features[name].int64_list.value)

      if y["answer_type"].int64_list.value[0] == 0:
        if self.is_train:
          if answer_type == self.answer_type_enum["NULL"]:
            # For instances that actually do not have an answer at all, we
            # increase the sampling rate for windows without an answer:
            rate = self.include_unknown_rate_for_unanswerable
          else:
            rate = self.include_unknown_rate
        else:
          # we only want to downsample the training set:
          rate = 1.0
        # Downsample null instances if rate < 1.0.
        if random.random() < rate:
          yield tf.train.Example(features=tf.train.Features(feature=y))
      else:
        # Always output non-null instances.
        yield tf.train.Example(features=tf.train.Features(feature=y))


class NQPreprocessFn(beam.DoFn):
  """Converts NQ examples to tf records."""

  def __init__(self, stride, seq_len, global_seq_len, question_len, vocab_file,
               do_lower_case, global_token_types, spm_model_path,
               tokenizer_type, is_train, predict_la_when_no_sa,
               include_unknown_rate, include_unknown_rate_for_unanswerable,
               fixed_blocks, fixed_block_size):
    self.stride = stride
    self.seq_len = seq_len
    self.global_seq_len = global_seq_len
    self.question_len = question_len
    self.vocab_file = vocab_file
    self.do_lower_case = do_lower_case
    self.global_token_types = global_token_types
    self.spm_model_path = spm_model_path
    self.tokenizer_type = tokenizer_type
    self.is_train = is_train
    self.predict_la_when_no_sa = predict_la_when_no_sa
    self.include_unknown_rate = include_unknown_rate
    self.include_unknown_rate_for_unanswerable = (
        include_unknown_rate_for_unanswerable)
    self.fixed_blocks = fixed_blocks
    self.fixed_block_size = fixed_block_size

  def start_bundle(self):
    self.preprocessor = NQPreprocessor(
        stride=self.stride,
        seq_len=self.seq_len,
        global_seq_len=self.global_seq_len,
        question_len=self.question_len,
        vocab_file=self.vocab_file,
        do_lower_case=self.do_lower_case,
        predict_la_when_no_sa=self.predict_la_when_no_sa,
        include_unknown_rate=self.include_unknown_rate,
        include_unknown_rate_for_unanswerable=(
            self.include_unknown_rate_for_unanswerable),
        include_html_tokens=True,
        global_token_types=self.global_token_types,
        spm_model_path=self.spm_model_path,
        tokenizer_type=self.tokenizer_type,
        is_train=self.is_train,
        fixed_blocks=self.fixed_blocks,
        fixed_block_size=self.fixed_block_size)

  def process(self, line):
    example = json.loads(line)
    yield self.preprocessor.to_tf_example(example).SerializeToString()


class NQSplitFn(beam.DoFn):
  """Yields fixed size instances from TF examples."""

  def __init__(self, stride, seq_len, global_seq_len, question_len, vocab_file,
               do_lower_case, global_token_types, spm_model_path,
               tokenizer_type, is_train, predict_la_when_no_sa,
               include_unknown_rate, include_unknown_rate_for_unanswerable,
               fixed_blocks, fixed_block_size):
    self.stride = stride
    self.seq_len = seq_len
    self.global_seq_len = global_seq_len
    self.question_len = question_len
    self.vocab_file = vocab_file
    self.do_lower_case = do_lower_case
    self.global_token_types = global_token_types
    self.spm_model_path = spm_model_path
    self.tokenizer_type = tokenizer_type
    self.is_train = is_train
    self.predict_la_when_no_sa = predict_la_when_no_sa
    self.include_unknown_rate = include_unknown_rate
    self.include_unknown_rate_for_unanswerable = (
        include_unknown_rate_for_unanswerable)
    self.fixed_blocks = fixed_blocks
    self.fixed_block_size = fixed_block_size

  def start_bundle(self):
    self.preprocessor = NQPreprocessor(
        stride=self.stride,
        seq_len=self.seq_len,
        global_seq_len=self.global_seq_len,
        question_len=self.question_len,
        vocab_file=self.vocab_file,
        do_lower_case=self.do_lower_case,
        predict_la_when_no_sa=self.predict_la_when_no_sa,
        include_unknown_rate=self.include_unknown_rate,
        include_unknown_rate_for_unanswerable=(
            self.include_unknown_rate_for_unanswerable),
        include_html_tokens=True,
        global_token_types=self.global_token_types,
        spm_model_path=self.spm_model_path,
        tokenizer_type=self.tokenizer_type,
        is_train=self.is_train,
        fixed_blocks=self.fixed_blocks,
        fixed_block_size=self.fixed_block_size,
        global_size_counter=beam.metrics.Metrics.distribution("global", "size"),
        long_size_counter=beam.metrics.Metrics.distribution("long", "size"),
        global_size_threshold_counters=[
            (x, beam.metrics.Metrics.counter("global-threshold", ">=" + str(x)))
            for x in [100, 200, 300, 400, 460]
        ],
        global_sentence_counter=(beam.metrics.Metrics.distribution(
            "global", "sentences")),
        long_sentence_tokens_counter=(beam.metrics.Metrics.distribution(
            "long", "sentence_tokens")))

  def process(self, serialized_example):
    example = tf.train.Example.FromString(serialized_example)
    for instance in self.preprocessor.split_example(example):
      yield instance.SerializeToString()
