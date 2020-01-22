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

r"""Implementation of Splitter embedding using Gensim.

===============================

This is part of the implementation accompanying the WWW 2019 paper, [_Is a
Single Embedding Enough? Learning Node Representations that Capture Multiple
Social Contexts_](https://ai.google/research/pubs/pub46238).

The code in this file allows one to create persona embeddings.

Known issues:  The inner loop (train_batch_sg_constraints) is written in pure
python, and its speed could be greatly improved by porting to an optimized
C implementation.

Citing
------
If you find _Persona Embedding_ useful in your research, we ask that you cite
the following paper:
> Epasto, A., Perozzi, B., (2019).
> Is a Single Embedding Enough? Learning Node Representations that Capture
Multiple Social Contexts.
> In _The Web Conference_.
"""
#pylint: skip-file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
from types import GeneratorType

from gensim import utils, matutils
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec
from gensim.models.word2vec import logger, train_sg_pair, train_cbow_pair
import numpy
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL, double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack
from six import iteritems, itervalues, string_types
from six.moves import xrange
from timeit import default_timer
from random import shuffle
try:
  from queue import Queue, Empty
except ImportError:
  from Queue import Queue, Empty

try:
  from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
  from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
  from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
  logger.debug("Fast version of {0} is being used".format(__name__))
except ImportError:
  # failed... fall back to plain numpy (20-80x slower training than the above)
  logger.warning("Slow version of {0} is being used".format(__name__))
  FAST_VERSION = -1
  MAX_WORDS_IN_BATCH = 10000


def train_batch_sg_constraints(model, constraints, alpha, work=None):
  """This function adds an additional constraint to the representation."""
  result = 0
  for constraint in constraints:
    word = model.vocab[constraint[0]]
    word2 = model.vocab[constraint[1]]

    # the representation of word2.index is used to predict model.index2word[word.index]
    train_sg_pair(model, model.index2word[word.index], word2.index, alpha)
    result += 1
  return result


class Persona2Vec(Word2Vec):

  def __init__(self,
               sentences=None,
               size=100,
               alpha=0.025,
               window=5,
               min_count=5,
               max_vocab_size=None,
               sample=1e-3,
               seed=1,
               workers=3,
               min_alpha=0.0001,
               sg=0,
               hs=0,
               negative=5,
               cbow_mean=1,
               hashfxn=hash,
               iter=5,
               null_word=0,
               trim_rule=None,
               sorted_vocab=1,
               batch_words=MAX_WORDS_IN_BATCH,
               constraint_learning_rate_scaling_factor=0.1,
               initial_weight_map={},
               extra_constraint_map={},
               initial_syn1_map={}):

    self.beta = constraint_learning_rate_scaling_factor
    self.initial_weights = initial_weight_map
    self.constraints = extra_constraint_map

    self.pairwise_constraints = []
    self.constraint_ids = set()
    for key, value in self.constraints.items():
      self.constraint_ids.add(key)
      for v in value:
        self.pairwise_constraints.append([key, v])

    batch_words = len(self.pairwise_constraints)
    shuffle(self.pairwise_constraints)

    print("constraints added: ", batch_words)

    # sanity check
    assert sg == 1  # only sg supported for now

    # initialize the rest of the variables & start training
    super(Persona2Vec, self).__init__(
        sentences=sentences,
        size=size,
        alpha=alpha,
        window=window,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
        sample=sample,
        seed=seed,
        workers=workers,
        min_alpha=min_alpha,
        sg=sg,
        hs=hs,
        negative=negative,
        cbow_mean=cbow_mean,
        hashfxn=hashfxn,
        iter=iter,
        null_word=null_word,
        trim_rule=trim_rule,
        sorted_vocab=sorted_vocab,
        batch_words=batch_words)


  def build_vocab(self,
                  sentences,
                  keep_raw_vocab=False,
                  trim_rule=None,
                  progress_per=10000):
    """
        Build vocabulary from a sequence of sentences (can be a once-only
        generator stream).
        Each sentence must be a list of unicode strings.
        """
    self.scan_vocab(
        sentences, progress_per=progress_per,
        trim_rule=trim_rule)  # initial survey

    # add constraints
    for key, value in self.constraints.items():
      for v in value:
        self.raw_vocab[key] += 1
        self.raw_vocab[v] += 1

    self.scale_vocab(
        keep_raw_vocab=keep_raw_vocab,
        trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
    self.finalize_vocab()  # build tables & arrays

  def reset_weights(self):
    if self.initial_weights:
      """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
      print("using weights passed in")

      initial_weights_used_cnt = 0
      random_inits_cnt = 0

      self.syn0 = empty((len(self.vocab), self.vector_size), dtype=REAL)
      for i in xrange(len(self.vocab)):
        # use initial_weights to initialize
        word = self.index2word[i]
        if word in self.initial_weights:
          self.syn0[i] = self.initial_weights[word]
          initial_weights_used_cnt += 1
        else:
          # constraints may not be given initial weights ...
          self.syn0[i] = self.seeded_vector(self.index2word[i] + str(self.seed))
          random_inits_cnt += 1
      if self.hs:
        self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
      if self.negative:
        self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)

      print("initial weights used: ", initial_weights_used_cnt)
      print("random weights used: ", random_inits_cnt)

      self.syn0norm = None
      self.syn0_lockf = ones(
          len(self.vocab), dtype=REAL)  # zeros suppress learning

      # suppress learning for constraints
      for v in self.constraint_ids:
        self.syn0_lockf[self.vocab[v].index] = 0

    else:
      print("initializing with random weights")
      super(Persona2Vec, self).reset_weights()

  def train(self,
            sentences,
            total_words=None,
            word_count=0,
            total_examples=None,
            queue_factor=2,
            report_delay=1.0):
    """ Update the model's neural weights from a sequence of sentences (can be a

        once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings.
        (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha,
        either total_examples
        (count of sentences) or total_words (count of raw words in sentences)
        should be provided, unless the
        sentences are the same as those that were used to initially build the
        vocabulary.
    """
    logger.info("Starting training.")

    self.neg_labels = []
    if self.negative > 0:
      # precompute negative labels optimization for pure-python training
      self.neg_labels = zeros(self.negative + 1)
      self.neg_labels[0] = 1.

    if FAST_VERSION < 0:
      import warnings
      warnings.warn(
          "C extension not loaded for Word2Vec, training will be slow. "
          "Install a C compiler and reinstall gensim for fast training.")
      self.neg_labels = []
      if self.negative > 0:
        # precompute negative labels optimization for pure-python training
        self.neg_labels = zeros(self.negative + 1)
        self.neg_labels[0] = 1.

    logger.info(
        "training model with %i workers on %i vocabulary and %i features, "
        "using sg=%s hs=%s sample=%s negative=%s window=%s", self.workers,
        len(self.vocab), self.layer1_size, self.sg, self.hs, self.sample,
        self.negative, self.window)

    if not self.vocab:
      raise RuntimeError(
          "you must first build vocabulary before training the model")
    if not hasattr(self, "syn0"):
      raise RuntimeError(
          "you must first finalize vocabulary before training the model")

    if total_words is None and total_examples is None:
      if self.corpus_count:
        total_examples = self.corpus_count
        logger.info(
            "expecting %i sentences, matching count from corpus used for vocabulary survey",
            total_examples)
      else:
        raise ValueError(
            "you must provide either total_words or total_examples, to enable alpha and progress calculations"
        )

    job_tally = 0

    if self.iter > 1:
      sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
      total_words = total_words and total_words * self.iter
      total_examples = total_examples and total_examples * self.iter

    def worker_loop():
      """Train the model, lifting lists of sentences from the job_queue."""
      work = matutils.zeros_aligned(
          self.layer1_size, dtype=REAL)  # per-thread private work memory
      neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
      jobs_processed = 0
      while True:
        job = job_queue.get()
        if job is None:
          progress_queue.put(None)
          break  # no more jobs => quit this worker
        sentences, pairwise, alpha = job
        tally, raw_tally = self._do_train_job(sentences, pairwise, alpha,
                                              (work, neu1))
        progress_queue.put(
            (len(sentences), tally, raw_tally))  # report back progress
        jobs_processed += 1
      logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def job_producer():
      """Fill jobs queue using the input `sentences` iterator."""
      job_batch, batch_size = [], 0
      pushed_words, pushed_examples = 0, 0
      next_alpha = self.alpha
      if next_alpha > self.min_alpha_yet_reached:
        logger.warn("Effective 'alpha' higher than previous training cycles")
      self.min_alpha_yet_reached = next_alpha
      job_no = 0

      for sent_idx, sentence in enumerate(sentences):
        sentence_length = self._raw_word_count([sentence])

        # can we fit this sentence into the existing job batch?
        if batch_size + sentence_length <= self.batch_words:
          # yes => add it to the current job
          job_batch.append(sentence)
          batch_size += sentence_length
        else:
          # no => submit the existing job
          pair_idx = list(
              numpy.random.choice(
                  range(len(self.pairwise_constraints)), int(batch_size * 0.2)))
          pairwise_samples = [self.pairwise_constraints[x] for x in pair_idx]
          logger.debug(
              "queueing job #%i (%i words, %i sentences, %i constraints) at alpha %.05f",
              job_no, batch_size, len(job_batch), len(pairwise_samples),
              next_alpha)
          job_no += 1
          job_queue.put((job_batch, pairwise_samples, next_alpha))

          # update the learning rate for the next job
          if self.min_alpha < next_alpha:
            if total_examples:
              # examples-based decay
              pushed_examples += len(job_batch)
              progress = 1.0 * pushed_examples / total_examples
            else:
              # words-based decay
              pushed_words += self._raw_word_count(job_batch)
              progress = 1.0 * pushed_words / total_words
            next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
            next_alpha = max(self.min_alpha, next_alpha)

          # add the sentence that didn't fit as the first item of a new job
          job_batch, batch_size = [sentence], sentence_length

      # add the last job too (may be significantly smaller than batch_words)
      if job_batch:
        logger.debug(
            "queueing job #%i (%i words, %i sentences, %i constraints) at alpha %.05f",
            job_no, batch_size, len(job_batch), len(self.pairwise_constraints),
            next_alpha)
        job_no += 1
        job_queue.put((job_batch, self.pairwise_constraints, next_alpha))

      if job_no == 0 and self.train_count == 0:
        logger.warning(
            "train() called with an empty iterator (if not intended, "
            "be sure to provide a corpus that offers restartable "
            "iteration = an iterable).")

      # give the workers heads up that they can finish -- no more work!
      for _ in xrange(self.workers):
        job_queue.put(None)
      logger.debug("job loop exiting, total %i jobs", job_no)

    # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
    job_queue = Queue(maxsize=queue_factor * self.workers)
    progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

    workers = [
        threading.Thread(target=worker_loop) for _ in xrange(self.workers)
    ]
    unfinished_worker_count = len(workers)
    workers.append(threading.Thread(target=job_producer))

    for thread in workers:
      thread.daemon = True  # make interrupting the process with ctrl+c easier
      thread.start()

    example_count, trained_word_count, raw_word_count = 0, 0, word_count
    start, next_report = default_timer() - 0.00001, 1.0

    while unfinished_worker_count > 0:
      report = progress_queue.get()  # blocks if workers too slow
      if report is None:  # a thread reporting that it finished
        unfinished_worker_count -= 1
        logger.info(
            "worker thread finished; awaiting finish of %i more threads",
            unfinished_worker_count)
        continue
      examples, trained_words, raw_words = report
      job_tally += 1

      # update progress stats
      example_count += examples
      trained_word_count += trained_words  # only words in vocab & sampled
      raw_word_count += raw_words

      # log progress once every report_delay seconds
      elapsed = default_timer() - start
      if elapsed >= next_report:
        if total_examples:
          # examples-based progress %
          logger.info(
              "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
              100.0 * example_count / total_examples,
              trained_word_count / elapsed, utils.qsize(job_queue),
              utils.qsize(progress_queue))
        else:
          # words-based progress %
          logger.info(
              "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
              100.0 * raw_word_count / total_words,
              trained_word_count / elapsed, utils.qsize(job_queue),
              utils.qsize(progress_queue))
        next_report = elapsed + report_delay

    # all done; report the final stats
    elapsed = default_timer() - start
    logger.info(
        "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
        raw_word_count, trained_word_count, elapsed,
        trained_word_count / elapsed)
    if job_tally < 10 * self.workers:
      logger.warn(
          "under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay"
      )

    # check that the input corpus hasn't changed during iteration
    if total_examples and total_examples != example_count:
      logger.warn(
          "supplied example count (%i) did not equal expected count (%i)",
          example_count, total_examples)
    if total_words and total_words != raw_word_count:
      logger.warn(
          "supplied raw word count (%i) did not equal expected count (%i)",
          raw_word_count, total_words)

    self.train_count += 1  # number of times train() has been called
    self.total_train_time += elapsed
    self.clear_sims()
    return trained_word_count

  def _do_train_job(self, sentences, pairwise, alpha, inits):
    """
        Train a single batch of sentences. Return 2-tuple `(effective word count
        after
        ignoring unknown words and sentence length trimming, total word count)`.
    """
    work, neu1 = inits
    tally = 0
    if self.sg:
      tally += train_batch_sg(self, sentences, alpha, work)
      tally += train_batch_sg_constraints(self, pairwise, alpha * self.beta,
                                          work)
    else:
      assert 0  # cbow not supported
      # tally += train_batch_cbow(self, sentences, alpha, work, neu1)
    return tally, self._raw_word_count(sentences)
