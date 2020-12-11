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

"""Tensorflow 2 Retriever using a (poss. fine-tuned) BERT query encoder.
"""
import logging
import os
from typing import Any, Dict, Union  # pylint: disable=unused-import

from absl import flags
import bert_utils
import dataclasses
import h5py
import numpy as np
import scann_utils
import tensorflow as tf
import tensorflow_hub as hub
import tf_utils
import utils


FLAGS = flags.FLAGS
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ScannConfig:

  num_neighbors: int
  training_sample_size: int
  num_leaves: int
  num_leaves_to_search: int
  reordering_num_neighbors: int


class BERTScaNNRetriever:
  """Class used for BERT based retrievers such as REALM and DPR.

  Parameters:
    self.query_encoder: Model instance to encode the queries.
    self.tokenizer: Tokenizer for the query model.
    self.vocab_lookup_table: Vocabulary index for the query and key embedders.
    self.scann_config: Configuration dataclass for the ScaNN builder.
    self.block_emb: Contains the dense vectors over which MIPS is done.
    self.scann_searcher: ScaNN MIPS index instance.
    self.cls_token_id: Id of the CLS token for the query and the embedder
    self.sep_token_id: Id of the SEP token for the query and the embedder
      modules.
    self.blocks: Object from which the raw text is obtained with indices.
  """

  def __init__(self, retriever_module_path,
               block_records_path, num_block_records,
               mode, scann_config):
    """Constructor for BERTScaNNRetriever.

    Arguments:
      retriever_module_path: Path of the BERT tf-hub checkpoint.
      block_records_path: Path of the textual form of the retrieval dataset in
                          the TFRecord format.
      num_block_records: Number of samples in the retrieval dataset.
      mode: tf.estimator.ModeKeys for the model, currently only eval is
            supported.

      scann_config: Configuration dataclass used to initialize the ScaNN MIPS
                    searcher object.

    """

    # Two and a half min. on CPU
    with utils.log_duration(LOGGER, "BERTScaNNRetriever.__init__",
                            "hub load query enc"):
      self.query_encoder = hub.load(retriever_module_path, tags={"train"} if
                                    mode == tf.estimator.ModeKeys.TRAIN else {})

    # Instantaneous
    with utils.log_duration(LOGGER, "BERTScaNNRetriever.__init__",
                            "build own tok info"):
      # Building our own tokenization info saves us 5 min where we would load
      # the BERT model again in bert_utils.get_tf_tokenizer
      # Getting the vocab path from the tf2 hub object (from tf.load) seems
      # broken
      vocab_file = os.path.join(retriever_module_path, "assets", "vocab.txt")
      utils.check_exists(vocab_file)
      do_lower_case = self.query_encoder.signatures["tokenization_info"
                                                    ]()["do_lower_case"]
      tokenization_info = dict(vocab_file=vocab_file,
                               do_lower_case=do_lower_case)

    # Instantaneous (for something that happens once) if tokenization_info
    # is passed (our addition) a few minutes otherwise, on CPU
    # (not passing tokenization_info makes it have to load BERT).
    with utils.log_duration(LOGGER, "BERTScaNNRetriever.__init__",
                            "get_tf_tokenizer"):

      self.tokenizer, self.vocab_lookup_table = bert_utils.get_tf_tokenizer(
          retriever_module_path, tokenization_info)

    # 9 min on CPU if not in dev mode. Longuest part of the setup phase.
    # We are using a lot of default values in the load_scann_searcher call
    # that it would probably be helpful to finetune
    with utils.log_duration(LOGGER, "BERTScaNNRetriever.__init__",
                            "load_scann_searcher"):
      checkpoint_path = os.path.join(retriever_module_path, "encoded",
                                     "encoded.ckpt")
      self.scann_config = scann_config
      self.block_emb, self.scann_searcher = scann_utils.load_scann_searcher(
          var_name="block_emb", checkpoint_path=checkpoint_path,
          **vars(scann_config))

    # Instantaneous for something that happens once
    with utils.log_duration(LOGGER, "BERTScaNNRetriever", "CLS and SEP tokens"):
      self.cls_token_id = tf.cast(self.vocab_lookup_table.lookup(
          tf.constant("[CLS]")), tf.int32)
      self.sep_token_id = tf.cast(self.vocab_lookup_table.lookup(
          tf.constant("[SEP]")), tf.int32)

      # 3 min on CPU whwn nor in dev mode
    with utils.log_duration(LOGGER, "BERTScaNNRetriever",
                            "Load the textual dataset"):
      # Extract the appropriate text
      # The buffer_size is taken from the original ORQA code.
      blocks_dataset = tf.data.TFRecordDataset(block_records_path,
                                               # Value taken from the REALM
                                               # code.
                                               buffer_size=512 * 1024 * 1024)
      # Get a single batch with all elements (?)
      blocks_dataset = blocks_dataset.batch(num_block_records,
                                            drop_remainder=True)
      # Create a thing that gets single elements over the dataset
      self.blocks = tf.data.experimental.get_single_element(blocks_dataset)

  @tf.function
  def retrieve(self, query_text):
    """Retrieves over the retrieval dataset, from a batch of text queries.

    First generates the query vector from the text, then queries the
    approximate maximum inner-product search engine.
    Args:
      query_text: Batch of text queries. In string form.

    Returns:
      Returns the text of the approximate nearest neighbors, as well as their
      inner product similarity with their query's vector representation.
    """

    # Tokenize the input tokens
    utils.check_equal(len(query_text), FLAGS.batch_size)
    question_token_ids = self.tokenizer.batch_encode_plus(
        query_text)["input_ids"]
    question_token_ids = tf.cast(
        question_token_ids.merge_dims(1, 2).to_tensor(), tf.int32)

    # Add a CLS token at the start of the input, and a SEP token at the end
    cls_ids = tf.fill((question_token_ids.shape[0], 1), self.cls_token_id)
    sep_ids = tf.fill((question_token_ids.shape[0], 1), self.sep_token_id)
    question_token_ids = tf.concat((cls_ids, question_token_ids, sep_ids), 1)
    utils.check_equal(question_token_ids.shape[0], FLAGS.batch_size)

    with utils.log_duration(LOGGER, "retrieve_multi", "Encode the query"):
      question_emb = self.query_encoder.signatures["projected"](
          input_ids=question_token_ids,
          input_mask=tf.ones_like(question_token_ids),
          segment_ids=tf.zeros_like(question_token_ids))["default"]
    LOGGER.debug("question_emb.shape: %s", question_emb.shape)
    utils.check_equal(question_emb.shape[0], FLAGS.batch_size)

    with utils.log_duration(LOGGER, "retrieve_multi", "search with ScaNN"):
      retrieved_block_ids, _ = self.scann_searcher.search_batched(question_emb)
      utils.check_equal(retrieved_block_ids.shape, (
          FLAGS.batch_size, self.scann_config.num_neighbors))

    # Gather the embeddings
    # [batch_size, retriever_beam_size, projection_size]
    retrieved_block_ids = retrieved_block_ids.astype(np.int64)
    retrieved_block_emb = tf.gather(self.block_emb, retrieved_block_ids)
    utils.check_equal(retrieved_block_emb.shape[:2], (
        FLAGS.batch_size, self.scann_config.num_neighbors))

    # Actually retrieve the text
    retrieved_blocks = tf.gather(self.blocks, retrieved_block_ids)
    utils.check_equal(retrieved_blocks.shape, (
        FLAGS.batch_size, self.scann_config.num_neighbors
    ))
    return retrieved_blocks


class FullyCachedRetriever:
  def __init__(
      self, db_path, block_records_path, num_block_records
  ):
    """Uses a file where all the retrievals have been made in advance.

    Uses the exact retrievals from query_cacher.py, which have been made in
    advance, as the questions don't change. The retrievals are made by fetching
    the pre-made retrievals by using the question-id in a lookup table.
    The inner products are also present in the file; they are used to sample
    from the pre-made retrievals to teach the model to adapt to having a wider
    variety of retrievals each epoch.

    Args:
      db_path: Path to the hdf5 file that was generated with `query_cacher.py`,
        that contains the pre-made retrievals for all questions.
      block_records_path: Path to the file with the reference (often wikipedia)
        text, that gets retrieved.
      num_block_records: Number of entries in the reference db.
    """
    # Load the db

    input_file = h5py.File(tf.io.gfile.GFile(db_path, "rb"), "r")
    self._keys = ["train", "eval", "test"]

    LOGGER.debug("Building the hash table")

    self._indices_by_ids = {}
    for split in self._keys:
      self._indices_by_ids[split] = (
          tf.lookup.StaticHashTable(
              tf.lookup.KeyValueTensorInitializer(
                  input_file[split]["sample_ids"],
                  tf.range(input_file[split]["retrieval"]["indices"].shape[0])
              ), 1))

    LOGGER.debug("Building the self._distances_by_h5_index")
    self._distances_by_h5_index = {
        split: tf.constant(input_file[split]["retrieval"]["distances"][:])
        for split in self._keys
    }

    LOGGER.debug("Building the self._db_entry_by_h5_index")
    self._db_entry_by_h5_index = {
        split: tf.constant(input_file[split]["retrieval"]["indices"][:])
        for split in self._keys
    }

    with utils.log_duration(
        LOGGER, "FullyCachedRetriever.__init__", "Load the textual dataset"
    ):
      # Extract the appropriate text
      # The buffer_size is taken from the original ORQA code.
      blocks_dataset = tf.data.TFRecordDataset(
          block_records_path, buffer_size=512 * 1024 * 1024
      )
      blocks_dataset = blocks_dataset.batch(
          num_block_records, drop_remainder=True
      )
      self._blocks = tf.data.experimental.get_single_element(blocks_dataset)

  @tf.function
  def retrieve(
      self, ds_split, question_ids, temperature, k
  ):
    """Does the retrieving.

    Args:
      ds_split:
        The h5 files are split per dataset split "train", "eval", "test". This
        argument tells us which one to use.
      question_ids: Id of the question. To be used to get the cached retrievals.
      temperature: Temperature to be used when sampling from the neighbors.
      k: Number of neighbors to use.

    Returns:
      A dict with the logits and the retrieved reference text blocks.
    """

    indices = self._indices_by_ids[ds_split].lookup(question_ids)
    distances = tf.gather(self._distances_by_h5_index[ds_split], indices)
    db_indices = tf.gather(self._db_entry_by_h5_index[ds_split], indices)

    # pick block ids
    logits = distances / temperature
    selections = tf_utils.sample_without_replacement(logits, k)
    final_indices = tf.gather(db_indices, selections, batch_dims=-1)
    # final_logits = tf.gather(logits, selections, batch_dims=-1)

    retrieved_blocks = tf.gather(self._blocks, final_indices)
    # utils.check_equal(final_logits.shape, final_indices.shape)
    return retrieved_blocks


RetrieverType = Union[BERTScaNNRetriever, FullyCachedRetriever]
