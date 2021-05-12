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

"""Extractive oracle for NarrativeQA dataset."""

import hashlib
import string
from typing import List, Text, Tuple, Union

import intervaltree
import numpy as np


class ExtractiveOracle(object):
  """Extractive oracle based on ROUGE-L metric.

    See third_party/py/nlgeval/pycocoevalcap/rouge/rouge.py for details.
  """

  ARTICLES = {'a', 'an', 'the'}
  PREPOSITIONS_SHORT_LIST = {
      'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid',
      'among', 'anti', 'around', 'as', 'at', 'before', 'behind', 'below',
      'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by',
      'concerning', 'considering', 'despite', 'down', 'during', 'except',
      'excepting', 'excluding', 'following', 'for', 'from', 'in', 'inside',
      'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite',
      'outside', 'over', 'past', 'per', 'plus', 'regarding', 'round', 'save',
      'since', 'than', 'through', 'to', 'toward', 'towards', 'under',
      'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with',
      'within', 'without'
  }
  PRONOUNS_SHORT_LIST = {
      'all', 'another', 'any', 'anybody', 'anyone', 'anything', 'as', 'aught',
      'both', 'each', 'either', 'enough', 'everybody', 'everyone', 'everything',
      'few', 'he', 'her', 'hers', 'herself', 'him', 'himself', 'his', 'I',
      'idem', 'it', 'its', 'itself', 'many', 'me', 'mine', 'most', 'my',
      'myself', 'naught', 'neither', 'nobody', 'none', 'nothing', 'nought',
      'one', 'other', 'others', 'ought', 'our', 'ours', 'ourself', 'ourselves',
      'several', 'she', 'some', 'somebody', 'someone', 'something', 'somewhat',
      'such', 'suchlike', 'that', 'thee', 'their', 'theirs', 'theirself',
      'theirselves', 'them', 'themself', 'themselves', 'there', 'these', 'they',
      'thine', 'this', 'those', 'thou', 'thy', 'thyself', 'us', 'we', 'what',
      'whatever', 'whatnot', 'whatsoever', 'whence', 'where', 'whereby',
      'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wherever',
      'wheresoever', 'whereto', 'whereunto', 'wherewith', 'wherewithal',
      'whether', 'which', 'whichever', 'whichsoever', 'who', 'whoever', 'whom',
      'whomever', 'whomso', 'whomsoever', 'whose', 'whosever', 'whosesoever',
      'whoso', 'whosoever', 'ye', 'yon', 'yonder', 'you', 'your', 'yours',
      'yourself', 'yourselves'
  }
  SERVICE_VERBS = {
      'be', 'is', 'will', 'shall', 'should', 'won\'t', 'shouldn\'t', 'are',
      'been', 'was', 'were', 'being', 'isn\'t', 'aren\'t', 'wasn\'t',
      'weren\'t', 'have', 'haven\'t', 'has', 'hasn\'t', 'had', 'hadn\'t',
      'having', 'do', 'don\'t', 'does', 'doesn\'t', 'did', 'didn\'t', 'done',
      'doing', 'can', 'cannot', 'could', 'couldn\'t'
  }
  CONJUNCTIONS_LIST = {
      'for', 'and', 'nor', 'but', 'or', 'not', 'no', 'n\'t', 'yet', 'so',
      'only', 'once', 'although', 'after', 'as', 'while', 'when', 'whereas',
      'whenever', 'wherever', 'whether', 'how', 'if', 'though', 'because',
      'before', 'until', 'unless', 'since', 'so', 'as', 'both', 'either',
      'whether', 'neither', 'not', 'such', 'scarcely', 'when', 'rather', 'than',
      'accordingly', 'after', 'also', 'before', 'besides', 'consequently',
      'conversely', 'finally', 'furthermore', 'hence', 'however', 'indeed',
      'instead', 'likewise', 'meanwhile', 'moreover', 'nevertheless', 'next',
      'nonetheless', 'otherwise', 'similarly', 'still', 'subsequently', 'then',
      'therefore', 'thus'
  }

  STOPWORDS = {'s'}
  STOPWORDS.update(string.punctuation)
  STOPWORDS.update(PREPOSITIONS_SHORT_LIST)
  STOPWORDS.update(ARTICLES)
  STOPWORDS.update(PRONOUNS_SHORT_LIST)
  STOPWORDS.update(SERVICE_VERBS)
  STOPWORDS.update(CONJUNCTIONS_LIST)

  DUMMY_VALUE = 1

  def __init__(self, min_roughe_l_score, top_percentile,
               top_k):
    self.beta = 1.2
    self.beta2 = self.beta * self.beta
    self.top_percentile = top_percentile
    assert self.top_percentile >= 0 and self.top_percentile <= 1
    self.min_roughe_l_score = min_roughe_l_score
    assert self.min_roughe_l_score >= 0 and self.min_roughe_l_score <= 1
    self.top_k = top_k
    assert self.top_k > 0

  def _estimate_max_answer(self, actual_answer_length):
    return 2 * actual_answer_length + 5

  def _hash_text(self, text):
    return [
        int(hashlib.md5(word.encode('utf8')).hexdigest()[:8], 16)
        for word in text
    ]

  def is_all_stopwords(self, words_list):
    for word in words_list:
      if word.lower() not in self.STOPWORDS:
        return False
    return True

  def find_approximate_answers(
      self,
      text,
      answer,
      return_score = False,
      remove_all_stopwords_answers = False,
  ):
    """Localates an approximate answers with the highest ROUGE-L score."""
    document = text.lower().split()
    document_hashed = self._hash_text(document)
    answer_words = answer.lower().split()
    answer_words_set = set(answer_words)
    answer_words_hashed = self._hash_text(answer_words)

    max_hypo_length = self._estimate_max_answer(len(answer_words))
    dp = np.zeros((max_hypo_length + 1, len(answer_words) + 1))
    counter = 0
    candidates = []
    for start in range(len(document)):
      if document[start] not in answer_words_set:
        continue
      counter += 1
      current_max_hypo_length = min(max_hypo_length, len(document) - start)
      dp[:, 0] = 0
      dp[0, :] = 0
      for i in range(1, current_max_hypo_length + 1):
        for j in range(1, len(answer_words) + 1):
          if document_hashed[start + i - 1] == answer_words_hashed[j - 1]:
            if document[start + i - 1] == answer_words[j - 1]:
              dp[i, j] = 1 + dp[i - 1, j - 1]
            else:
              dp[i, j] = max(dp[i, j - 1], dp[i - 1, j])
          else:
            dp[i, j] = max(dp[i, j - 1], dp[i - 1, j])

        # score = dp[i, len(answer_words)]
        # See nlgeval/pycocoevalcap/rouge/rouge.py
        # for the ROUGE-L computations.
        precision = dp[i, len(answer_words)] / i
        recall = dp[i, len(answer_words)] / len(answer_words)
        if precision != 0 and recall != 0:
          score = ((1 + self.beta2) * precision * recall) / (
              recall + self.beta2 * precision)
        else:
          score = 0.0
        candidates.append((score, start, start + i))
    candidates.sort(reverse=True)

    if not candidates:
      return []

    best_score = candidates[0][0]
    min_roughe_l_score = max(self.min_roughe_l_score,
                             best_score * self.top_percentile)
    result = []
    tree = intervaltree.IntervalTree()
    predicted_answers = set()
    for candidate in candidates:
      score = candidate[0]
      if score < min_roughe_l_score:
        break
      begin = candidate[1]
      end = candidate[2]
      if (remove_all_stopwords_answers and
          self.is_all_stopwords(document[begin:end])):
        continue
      if tree.overlap(begin, end):
        continue
      predicted_answer = ' '.join(document[begin:end])
      if predicted_answer in predicted_answers:
        continue
      tree.addi(begin, end, self.DUMMY_VALUE)
      if return_score:
        result.append((predicted_answer, score))
      else:
        result.append(predicted_answer)
      predicted_answers.add(predicted_answer)

      if len(result) >= self.top_k:
        break
    return result
