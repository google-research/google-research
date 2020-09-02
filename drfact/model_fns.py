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

# Lint as: python3
"""Collection of model functions implementing different multihop variants."""

from language.labs.drkit import model_fns as model_utils
from language.labs.drkit import search_utils
import tensorflow.compat.v1 as tf

from tensorflow.contrib import layers as contrib_layers

DEFAULT_VALUE = -10000.


def follow_mention(batch_entities,
                   relation_st_qry,
                   relation_en_qry,
                   entity_word_ids,
                   entity_word_masks,
                   ent2ment_ind,
                   ent2ment_val,
                   ment2ent_map,
                   word_emb_table,
                   word_weights,
                   mips_search_fn,
                   tf_db,
                   hidden_size,
                   mips_config,
                   qa_config,
                   is_training,
                   ensure_index=None):
  """Sparse implementation of the relation follow operation.

  Args:
    batch_entities: [batch_size, num_entities] SparseTensor of incoming entities
      and their scores.
    relation_st_qry: [batch_size, dim] Tensor representating start query vectors
      for dense retrieval.
    relation_en_qry: [batch_size, dim] Tensor representating end query vectors
      for dense retrieval.
    entity_word_ids: [num_entities, max_entity_len] Tensor holding word ids of
      each entity.
    entity_word_masks: [num_entities, max_entity_len] Tensor with masks into
      word ids above.
    ent2ment_ind: [num_entities, num_mentions] RaggedTensor mapping entities to
      mention indices which co-occur with them.
    ent2ment_val: [num_entities, num_mentions] RaggedTensor mapping entities to
      mention scores which co-occur with them.
    ment2ent_map: [num_mentions] Tensor mapping mentions to their entities.
    word_emb_table: [vocab_size, dim] Tensor of word embedddings.  (?)
    word_weights: [vocab_size, 1] Tensor of word weights.  (?)
    mips_search_fn: Function which accepts a dense query vector and returns the
      top-k indices closest to it (from the tf_db).
    tf_db: [num_mentions, 2 * dim] Tensor of mention representations.
    hidden_size: Scalar dimension of word embeddings.
    mips_config: MIPSConfig object.
    qa_config: QAConfig object.
    is_training: Boolean.
    ensure_index: [batch_size] Tensor of mention ids. Only needed if
      `is_training` is True.  (? each example only one ensure entity?)

  Returns:
    ret_mentions_ids: [batch_size, k] Tensor of retrieved mention ids.
    ret_mentions_scs: [batch_size, k] Tensor of retrieved mention scores.
    ret_entities_ids: [batch_size, k] Tensor of retrieved entities ids.
  """
  if qa_config.entity_score_threshold is not None:
    # Remove the entities which have scores lower than the threshold.
    mask = tf.greater(batch_entities.values, qa_config.entity_score_threshold)
    batch_entities = tf.sparse.retain(batch_entities, mask)
  batch_size = batch_entities.dense_shape[0]  # number of the batches
  batch_ind = batch_entities.indices[:, 0]  # the list of the batch ids
  entity_ind = batch_entities.indices[:, 1]  # the list of the entity ids
  entity_scs = batch_entities.values  # the list of the scores of each entity

  # Obtain BOW embeddings for the given set of entities.
  # [NNZ, dim]  NNZ (number of non-zero entries) = len(entity_ind)
  batch_entity_emb = model_utils.entity_emb(entity_ind, entity_word_ids,
                                            entity_word_masks, word_emb_table,
                                            word_weights)
  batch_entity_emb = batch_entity_emb * tf.expand_dims(entity_scs, axis=1)
  # [batch_size, dim]
  uniq_batch_ind, uniq_idx = tf.unique(batch_ind)
  agg_emb = tf.unsorted_segment_sum(batch_entity_emb, uniq_idx,
                                    tf.shape(uniq_batch_ind)[0])
  batch_bow_emb = tf.scatter_nd(
      tf.expand_dims(uniq_batch_ind, 1), agg_emb,
      tf.stack([batch_size, hidden_size], axis=0))
  batch_bow_emb.set_shape([None, hidden_size])
  if qa_config.projection_dim is not None:
    with tf.variable_scope("projection"):
      batch_bow_emb = contrib_layers.fully_connected(
          batch_bow_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          reuse=tf.AUTO_REUSE,
          scope="bow_projection")
  # Each instance in a batch has onely one vector as embedding.

  # Ragged sparse search.
  # (num_batch x num_entities) * (num_entities x num_mentions)
  # [batch_size x num_mentions] sparse
  sp_mention_vec = model_utils.sparse_ragged_mul(
      batch_entities,
      ent2ment_ind,
      ent2ment_val,
      batch_size,
      mips_config.num_mentions,
      qa_config.sparse_reduce_fn,  # max or sum
      threshold=qa_config.entity_score_threshold,
      fix_values_to_one=qa_config.fix_sparse_to_one)
  if is_training and qa_config.ensure_answer_sparse:
    ensure_indices = tf.stack([tf.range(batch_size), ensure_index], axis=-1)
    sp_ensure_vec = tf.SparseTensor(
        tf.cast(ensure_indices, tf.int64),
        tf.ones([batch_size]),
        dense_shape=[batch_size, mips_config.num_mentions])
    sp_mention_vec = tf.sparse.add(sp_mention_vec, sp_ensure_vec)
    sp_mention_vec = tf.SparseTensor(
        indices=sp_mention_vec.indices,
        values=tf.minimum(1., sp_mention_vec.values),
        dense_shape=sp_mention_vec.dense_shape)

  # Dense scam search.
  # [batch_size, 2 * dim]
  # Constuct query embeddings (dual encoder: [subject; relation]).
  scam_qrys = tf.concat(
      [batch_bow_emb + relation_st_qry, batch_bow_emb + relation_en_qry],
      axis=1)
  with tf.device("/cpu:0"):
    # [batch_size, num_neighbors]
    _, ret_mention_ids = mips_search_fn(scam_qrys)
    if is_training and qa_config.ensure_answer_dense:
      ret_mention_ids = model_utils.ensure_values_in_mat(
          ret_mention_ids, ensure_index, tf.int32)
    # [batch_size, num_neighbors, 2 * dim]
    ret_mention_emb = tf.gather(tf_db, ret_mention_ids)

  if qa_config.l2_normalize_db:
    ret_mention_emb = tf.nn.l2_normalize(ret_mention_emb, axis=2)
  # [batch_size, 1, num_neighbors]
  ret_mention_scs = tf.matmul(
      tf.expand_dims(scam_qrys, 1), ret_mention_emb, transpose_b=True)
  # [batch_size, num_neighbors]
  ret_mention_scs = tf.squeeze(ret_mention_scs, 1)
  # [batch_size, num_mentions] sparse
  dense_mention_vec = model_utils.convert_search_to_vector(
      ret_mention_scs, ret_mention_ids, tf.cast(batch_size, tf.int32),
      mips_config.num_neighbors, mips_config.num_mentions)

  # Combine sparse and dense search.
  if (is_training and qa_config.train_with_sparse) or (
      (not is_training) and qa_config.predict_with_sparse):
    # [batch_size, num_mentions] sparse
    if qa_config.sparse_strategy == "dense_first":
      ret_mention_vec = model_utils.sp_sp_matmul(dense_mention_vec,
                                                 sp_mention_vec)
    elif qa_config.sparse_strategy == "sparse_first":
      with tf.device("/cpu:0"):
        ret_mention_vec = model_utils.rescore_sparse(sp_mention_vec, tf_db,
                                                     scam_qrys)
    else:
      raise ValueError("Unrecognized sparse_strategy %s" %
                       qa_config.sparse_strategy)
  else:
    # [batch_size, num_mentions] sparse
    ret_mention_vec = dense_mention_vec

  # Get entity scores and ids.
  # [batch_size, num_entities] sparse
  entity_indices = tf.cast(
      tf.gather(ment2ent_map, ret_mention_vec.indices[:, 1]), tf.int64)
  ret_entity_vec = tf.SparseTensor(
      indices=tf.concat(
          [ret_mention_vec.indices[:, 0:1],
           tf.expand_dims(entity_indices, 1)],
          axis=1),
      values=ret_mention_vec.values,
      dense_shape=[batch_size, qa_config.num_entities])

  return ret_entity_vec, ret_mention_vec, dense_mention_vec, sp_mention_vec


def maxscale_spare_tensor(sp_tensor):
  """Scales the sparse tensor with its maximum per row."""
  sp_tensor_maxmiums = tf.sparse.reduce_max(sp_tensor, 1)  # batch_size
  gather_sp_tensor_maxmiums = tf.gather(sp_tensor_maxmiums,
                                        sp_tensor.indices[:, 0:1])
  gather_sp_tensor_maxmiums = tf.reshape(gather_sp_tensor_maxmiums,
                                         tf.shape(sp_tensor.values))
  scaled_val = sp_tensor.values / gather_sp_tensor_maxmiums
  scaled_sp_tensor = tf.SparseTensor(sp_tensor.indices, scaled_val,
                                     sp_tensor.dense_shape)
  return scaled_sp_tensor


def follow_fact(
    batch_facts,
    relation_st_qry,
    relation_en_qry,
    fact2fact_ind,
    fact2fact_val,
    fact2ent_ind,
    fact2ent_val,
    fact_mips_search_fn,
    tf_fact_db,
    fact_mips_config,
    qa_config,
    is_training,
    hop_id=0,
    is_printing=True,
):
  """Sparse implementation of the relation follow operation.

  Args:
    batch_facts: [batch_size, num_facts] SparseTensor of incoming facts and
      their scores.
    relation_st_qry: [batch_size, dim] Tensor representating start query vectors
      for dense retrieval.
    relation_en_qry: [batch_size, dim] Tensor representating end query vectors
      for dense retrieval.
    fact2fact_ind: [num_facts, num_facts] RaggedTensor mapping facts to entity
      indices which co-occur with them.
    fact2fact_val: [num_facts, num_facts] RaggedTensor mapping facts to entity
      scores which co-occur with them.
    fact2ent_ind: [num_facts, num_entities] RaggedTensor mapping facts to entity
      indices which co-occur with them.
    fact2ent_val: [num_facts, num_entities] RaggedTensor mapping facts to entity
      scores which co-occur with them.
    fact_mips_search_fn: Function which accepts a dense query vector and returns
      the top-k indices closest to it (from the tf_fact_db).
    tf_fact_db: [num_facts, 2 * dim] Tensor of fact representations.
    fact_mips_config: MIPS Config object.
    qa_config: QAConfig object.
    is_training: Boolean.
    hop_id: int, the current hop id.
    is_printing: if print results for debugging.

  Returns:
    ret_entities: [batch_size, num_entities] Tensor of retrieved entities.
    ret_facts: [batch_size, num_facts] Tensor of retrieved facts.
    dense_fact_vec: [batch_size, num_facts] Tensor of retrieved facts (dense).
    sp_fact_vec: [batch_size, num_facts] Tensor of retrieved facts (sparse).
  """
  num_facts = fact_mips_config.num_facts
  batch_size = batch_facts.dense_shape[0]  # number of examples in a batch
  example_ind = batch_facts.indices[:, 0]  # the list of the example ids
  fact_ind = batch_facts.indices[:, 1]  # the list of the fact ids
  fact_scs = batch_facts.values  # the list of the scores of each fact
  uniq_original_example_ind, uniq_local_example_idx = tf.unique(example_ind)
  # uniq_original_example_ind: local to original example id
  # uniq_local_example_idx: a list of local example id
  # tf.shape(uniq_original_example_ind)[0] = num_examples
  if qa_config.fact_score_threshold is not None:
    # Remove the facts which have scores lower than the threshold.
    mask = tf.greater(batch_facts.values, qa_config.fact_score_threshold)
    batch_facts = tf.sparse.retain(batch_facts, mask)
  # Sparse: Ragged sparse search from the current facts to the next facts.
  # (num_batch x num_facts) X (num_facts x num_facts)
  # [batch_size x num_facts] sparse
  if hop_id > 0:
    sp_fact_vec = model_utils.sparse_ragged_mul(
        batch_facts,
        fact2fact_ind,
        fact2fact_val,
        batch_size,
        num_facts,
        "sum",  # Note: check this.
        threshold=None,
        fix_values_to_one=True)
    # Note: find a better way for this.
    mask = tf.greater(sp_fact_vec.values, 3)  # 1/0.2 = 5
    sp_fact_vec = tf.sparse.retain(sp_fact_vec, mask)
  else:
    # For the first hop, then we use the init fact itself.
    # Because the sparse retieval is already done from the question.
    sp_fact_vec = batch_facts

  # Note: Remove the previous hop's facts
  # Note: Limit the number of fact followers.

  # Dense: Aggregate the facts in each batch as a single fact embedding vector.
  fact_embs = tf.gather(tf_fact_db, fact_ind)  # len(fact_ind) X 2dim
  # Note: check, does mean make sense?
  # sum if it was softmaxed
  # mean..
  del fact_scs  # Not used for now.
  # fact_embs = fact_embs * tf.expand_dims(fact_scs, axis=1)  #batch_fact.values
  ### Start of debugging w/ tf.Print ###
  if is_printing:
    fact_embs = tf.compat.v1.Print(
        input_=fact_embs,
        data=[tf.shape(batch_facts.indices)[0], batch_facts.indices],
        message="\n\n###\n batch_facts.indices and total #facts at hop %d \n" %
        hop_id,
        first_n=10,
        summarize=50)
    fact_embs = tf.compat.v1.Print(
        input_=fact_embs,
        data=[
            batch_facts.values,
        ],
        message="batch_facts.values at hop %d \n" % hop_id,
        first_n=10,
        summarize=25)
    fact_embs = tf.compat.v1.Print(
        input_=fact_embs,
        data=[tf.shape(sp_fact_vec.indices)[0], sp_fact_vec.indices],
        message="\n Sparse Fact Results @ hop %d \n" % hop_id +
        " sp_fact_vec.indices at hop %d \n" % hop_id,
        first_n=10,
        summarize=50)
    fact_embs = tf.compat.v1.Print(
        input_=fact_embs,
        data=[
            sp_fact_vec.values,
        ],
        message="sp_fact_vec.values at hop %d \n" % hop_id,
        first_n=10,
        summarize=25)
  ### End of debugging w/ tf.Print ###

  agg_emb = tf.math.unsorted_segment_mean(
      fact_embs, uniq_local_example_idx,
      tf.shape(uniq_original_example_ind)[0])
  batch_fact_emb = tf.scatter_nd(
      tf.expand_dims(uniq_original_example_ind, 1), agg_emb,
      tf.stack([batch_size, 2 * qa_config.projection_dim], axis=0))
  # Each instance in a batch has onely one vector as the overall fact emb.
  batch_fact_emb.set_shape([None, 2 * qa_config.projection_dim])

  # Note: Normalize the embeddings if they are not from SoftMax.
  # batch_fact_emb = tf.nn.l2_normalize(batch_fact_emb, axis=1)

  # Dense scam search.
  # [batch_size, 2 * dim]
  # Note: reform query embeddings.
  scam_qrys = batch_fact_emb + tf.concat([relation_st_qry, relation_en_qry],
                                         axis=1)
  with tf.device("/cpu:0"):
    # [batch_size, num_neighbors]
    _, ret_fact_ids = fact_mips_search_fn(scam_qrys)
    # [batch_size, num_neighbors, 2 * dim]
    ret_fact_emb = tf.gather(tf_fact_db, ret_fact_ids)

  if qa_config.l2_normalize_db:
    ret_fact_emb = tf.nn.l2_normalize(ret_fact_emb, axis=2)
  # [batch_size, 1, num_neighbors]
  # The score of a fact is its innder product with qry.
  ret_fact_scs = tf.matmul(
      tf.expand_dims(scam_qrys, 1), ret_fact_emb, transpose_b=True)
  # [batch_size, num_neighbors]
  ret_fact_scs = tf.squeeze(ret_fact_scs, 1)
  # [batch_size, num_facts] sparse
  dense_fact_vec = model_utils.convert_search_to_vector(
      ret_fact_scs, ret_fact_ids, tf.cast(batch_size, tf.int32),
      fact_mips_config.num_neighbors, fact_mips_config.num_facts)

  # Combine sparse and dense search.
  if (is_training and qa_config.train_with_sparse) or (
      (not is_training) and qa_config.predict_with_sparse):
    # [batch_size, num_mentions] sparse
    if qa_config.sparse_strategy == "dense_first":
      ret_fact_vec = model_utils.sp_sp_matmul(dense_fact_vec, sp_fact_vec)
    elif qa_config.sparse_strategy == "sparse_first":
      with tf.device("/cpu:0"):
        ret_fact_vec = model_utils.rescore_sparse(sp_fact_vec, tf_fact_db,
                                                  scam_qrys)
    else:
      raise ValueError("Unrecognized sparse_strategy %s" %
                       qa_config.sparse_strategy)
  else:
    # [batch_size, num_facts] sparse
    ret_fact_vec = dense_fact_vec

  # # Scaling facts with SoftMax.
  ret_fact_vec = tf.sparse.reorder(ret_fact_vec)
  # max_ip_scores = tf.reduce_max(ret_fact_vec.values)
  # min_ip_scores = tf.reduce_min(ret_fact_vec.values)
  # range_ip_scores = max_ip_scores - min_ip_scores
  # scaled_values = (ret_fact_vec.values - min_ip_scores) / range_ip_scores
  scaled_facts = tf.SparseTensor(
      indices=ret_fact_vec.indices,
      values=ret_fact_vec.values / tf.reduce_max(ret_fact_vec.values),
      dense_shape=ret_fact_vec.dense_shape)
  # ret_fact_vec_sf = tf.sparse.softmax(scaled_facts)
  ret_fact_vec_sf = scaled_facts

  # Remove the facts which have scores lower than the threshold.
  mask = tf.greater(ret_fact_vec_sf.values, 0.5)  # Must larger than max/5
  ret_fact_vec_sf_fitered = tf.sparse.retain(ret_fact_vec_sf, mask)

  # Note: add a soft way to score (all) the entities based on the facts.
  # Note: maybe use the pre-computed (tf-idf) similarity score here. e2e
  # Retrieve entities before Fact-SoftMaxing
  ret_entities_nosc = model_utils.sparse_ragged_mul(
      ret_fact_vec_sf,  # Use the non-filtered scores of the retrieved facts.
      fact2ent_ind,
      fact2ent_val,
      batch_size,
      qa_config.num_entities,
      "sum",
      threshold=qa_config.fact_score_threshold,
      fix_values_to_one=True)

  ret_entities = tf.SparseTensor(
      indices=ret_entities_nosc.indices,
      values=ret_entities_nosc.values / tf.reduce_max(ret_entities_nosc.values),
      dense_shape=ret_entities_nosc.dense_shape)

  ### Start of debugging w/ tf.Print ###
  if is_printing:
    tmp_vals = ret_entities.values

    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[tf.shape(ret_fact_vec.indices)[0], ret_fact_vec.indices],
        message="\n\n-rescored- ret_fact_vec.indices at hop %d \n" % hop_id,
        first_n=10,
        summarize=51)
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            ret_fact_vec.values,
        ],
        message="-rescored- ret_fact_vec.values at hop %d \n" % hop_id,
        first_n=10,
        summarize=25)
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            ret_fact_vec_sf.values,
        ],
        message="ret_fact_vec_sf.values at hop %d \n" % hop_id,
        first_n=10,
        summarize=25)
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            tf.shape(ret_fact_vec_sf_fitered.values),
            ret_fact_vec_sf_fitered.values,
        ],
        message="ret_fact_vec_sf_fitered.values at hop %d \n" % hop_id,
        first_n=10,
        summarize=25)
    ret_entities = tf.SparseTensor(
        indices=ret_entities.indices,
        values=tmp_vals,
        dense_shape=ret_entities.dense_shape)
  ### End of debugging w/ tf.Print ###

  return ret_entities, ret_fact_vec_sf_fitered, None, None


def multi_hop_fact(qry_input_ids,
                   qry_input_mask,
                   qry_entity_ids,
                   entity_ids,
                   entity_mask,
                   ent2fact_ind,
                   ent2fact_val,
                   fact2ent_ind,
                   fact2ent_val,
                   fact2fact_ind,
                   fact2fact_val,
                   is_training,
                   use_one_hot_embeddings,
                   bert_config,
                   qa_config,
                   fact_mips_config,
                   num_hops,
                   exclude_set=None,
                   is_printing=True):
  """Multi-hops of propagation from input to output facts.

  Args:
    qry_input_ids:
    qry_input_mask:
    qry_entity_ids:
    entity_ids: (entity_word_ids) [num_entities, max_entity_len] Tensor holding
      word ids of each entity.
    entity_mask: (entity_word_masks) [num_entities, max_entity_len] Tensor with
      masks into word ids above.
    ent2fact_ind:
    ent2fact_val:
    fact2ent_ind:
    fact2ent_val:
    fact2fact_ind:
    fact2fact_val:
    is_training:
    use_one_hot_embeddings:
    bert_config:
    qa_config:
    fact_mips_config:
    num_hops:
    exclude_set:
    is_printing:

  Returns:
    layer_entities:
    layer_facts:
    layer_dense:
    layer_sp:
    batch_entities_nosc:
    qry_seq_emb:
  """
  del entity_ids, entity_mask, exclude_set  # Not used for now.
  # MIPS search for facts.  Build fact feature Database
  with tf.device("/cpu:0"):
    tf_fact_db, fact_mips_search_fn = search_utils.create_mips_searcher(
        fact_mips_config.ckpt_var_name,
        # [fact_mips_config.num_facts, fact_mips_config.emb_size],
        fact_mips_config.ckpt_path,
        fact_mips_config.num_neighbors,
        local_var_name="scam_init_barrier_fact")

  # for question BOW embedding
  with tf.variable_scope("qry/bow"):
    # trainable word weights over the BERT vocab for all query embeddings.
    word_weights = tf.get_variable(
        "word_weights", [bert_config.vocab_size, 1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
  qry_seq_emb, word_emb_table = model_utils.shared_qry_encoder_v2(
      qry_input_ids, qry_input_mask, is_training, use_one_hot_embeddings,
      bert_config, qa_config)

  del word_weights, word_emb_table  # Not used for now.

  batch_size = tf.shape(qry_input_ids)[0]
  # Get question entities w/o scores.
  batch_qry_entities = tf.SparseTensor(
      indices=tf.concat([
          qry_entity_ids.indices[:, 0:1],
          tf.cast(tf.expand_dims(qry_entity_ids.values, 1), tf.int64)
      ],
                        axis=1),
      values=tf.ones_like(qry_entity_ids.values, dtype=tf.float32),
      dense_shape=[batch_size, qa_config.num_entities])
  # Prepare initial facts.
  initial_facts = model_utils.sparse_ragged_mul(
      batch_qry_entities,
      ent2fact_ind,
      ent2fact_val,
      batch_size,
      fact_mips_config.num_facts,
      "sum",  # max or sum
      threshold=None,
      fix_values_to_one=True)

  # Note: set a hyper parameter in qa.config
  # Note: can we do top k here for sparse tensor?
  # Limit the number of init facts such that we won't have too many facts.

  # mask = tf.greater(initial_facts.values, 1)  # >= 2 qry concepts
  # initial_facts = tf.sparse.retain(initial_facts, mask)

  scaled_initial_facts = maxscale_spare_tensor(initial_facts)
  mask_thresold = tf.greater(scaled_initial_facts.values, 0.25)
  final_initial_facts = tf.sparse.retain(scaled_initial_facts, mask_thresold)

  if is_printing:
    tmp_vals = final_initial_facts.values
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            tf.shape(initial_facts.indices),
            initial_facts.values,
        ],
        message="-" * 100 + "\n\n ## Initial Facts (at hop 0):\n"
        "shape(initial_facts), initial_facts.values,",
        first_n=10,
        summarize=52)
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            tf.shape(scaled_initial_facts.indices),
            scaled_initial_facts.values,
        ],
        message="shape(scaled_initial_facts), scaled_initial_facts.values,",
        first_n=10,
        summarize=52)
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            tf.shape(final_initial_facts.indices),
            final_initial_facts.values,
        ],
        message="shape(final_initial_facts), final_initial_facts.values,",
        first_n=10,
        summarize=52)

    final_initial_facts = tf.SparseTensor(final_initial_facts.indices, tmp_vals,
                                          final_initial_facts.dense_shape)
  layer_facts, layer_entities = [], []
  layer_dense, layer_sp = [], []
  batch_facts = final_initial_facts
  for hop in range(num_hops):
    with tf.name_scope("hop_%d" % hop):
      # The question start/end embeddings for each hop.
      qry_start_emb, qry_end_emb = model_utils.layer_qry_encoder(
          qry_seq_emb,
          qry_input_ids,
          qry_input_mask,
          is_training,
          bert_config,
          qa_config,
          suffix="_%d" % hop,
          project_dim=qa_config.projection_dim)  # project=True
      ret_entities, ret_facts, _, _ = follow_fact(
          batch_facts, qry_start_emb, qry_end_emb, fact2fact_ind, fact2fact_val,
          fact2ent_ind, fact2ent_val, fact_mips_search_fn, tf_fact_db,
          fact_mips_config, qa_config, is_training, hop, is_printing)
      batch_facts = ret_facts  # Update to next hop.
      # Update results.
      layer_facts.append(ret_facts)
      layer_entities.append(ret_entities)

  tf.logging.info("len layer_facts: %d", len(layer_facts))
  tf.logging.info("len layer_entities: %d", len(layer_entities))
  return (layer_entities, layer_facts, layer_dense, layer_sp,
          batch_qry_entities, initial_facts, qry_seq_emb)


def multi_hop_mention(qry_input_ids,
                      qry_input_mask,
                      qry_entity_ids,
                      entity_ids,
                      entity_mask,
                      ent2ment_ind,
                      ent2ment_val,
                      ment2ent_map,
                      is_training,
                      use_one_hot_embeddings,
                      bert_config,
                      qa_config,
                      mips_config,
                      num_hops,
                      exclude_set=None,
                      bridge_mentions=None,
                      answer_mentions=None):  # answer mentions?
  """Multi-hops of propagation from input to output entities.

  Args:
    qry_input_ids:
    qry_input_mask:
    qry_entity_ids:
    entity_ids: (entity_word_ids) [num_entities, max_entity_len] Tensor holding
      word ids of each entity.
    entity_mask: (entity_word_masks) [num_entities, max_entity_len] Tensor with
      masks into word ids above.
    ent2ment_ind:
    ent2ment_val:
    ment2ent_map:
    is_training:
    use_one_hot_embeddings:
    bert_config:
    qa_config:
    mips_config:
    num_hops:
    exclude_set:
    bridge_mentions:
    answer_mentions:

  Returns:
    layer_entities:
    layer_mentions:
    layer_dense:
    layer_sp:
    batch_entities_nosc:
    qry_seq_emb:
  """
  # for question BOW embedding
  with tf.variable_scope("qry/bow"):
    # Note: trainable word weights over the BERT vocab for query
    word_weights = tf.get_variable(
        "word_weights", [bert_config.vocab_size, 1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
  # Note: we can use the [CLS] token here?
  qry_seq_emb, word_emb_table = model_utils.shared_qry_encoder_v2(
      qry_input_ids, qry_input_mask, is_training, use_one_hot_embeddings,
      bert_config, qa_config)

  batch_size = tf.shape(qry_input_ids)[0]
  # Multiple entities per question. We need to re-score.
  with tf.name_scope("entity_linking"):
    batch_entity_emb = model_utils.entity_emb(
        tf.cast(qry_entity_ids.values, tf.int64), entity_ids, entity_mask,
        word_emb_table, word_weights)  # question entity embeddings.
    # Embed query into start and end vectors for dense retrieval for a hop.
    qry_el_emb, _ = model_utils.layer_qry_encoder(  # question embeddings
        qry_seq_emb,
        qry_input_ids,
        qry_input_mask,
        is_training,
        bert_config,
        qa_config,
        suffix="_el",
        project=False)
    batch_qry_el_emb = tf.gather(qry_el_emb, qry_entity_ids.indices[:, 0])
    batch_entity_el_scs = tf.reduce_sum(batch_qry_el_emb * batch_entity_emb, -1)
    batch_entities_nosc = tf.SparseTensor(
        # Note: double check this.
        indices=tf.concat([
            qry_entity_ids.indices[:, 0:1],
            tf.cast(tf.expand_dims(qry_entity_ids.values, 1), tf.int64)
        ],
                          axis=1),
        values=batch_entity_el_scs,
        dense_shape=[batch_size, qa_config.num_entities])
    batch_entities = tf.sparse.softmax(tf.sparse.reorder(batch_entities_nosc))

  ensure_mentions = bridge_mentions  # Note: check "supporoting facts"

  with tf.device("/cpu:0"):
    # MIPS search for mentions. Mention Feature Database
    tf_db, mips_search_fn = search_utils.create_mips_searcher(
        mips_config.ckpt_var_name,
        # [mips_config.num_mentions, mips_config.emb_size],
        mips_config.ckpt_path,
        mips_config.num_neighbors,
        local_var_name="scam_init_barrier")
  layer_mentions, layer_entities = [], []
  layer_dense, layer_sp = [], []
  for hop in range(num_hops):
    with tf.name_scope("hop_%d" % hop):
      # Note: the question start/end embeddings for each hop?
      qry_start_emb, qry_end_emb = model_utils.layer_qry_encoder(
          qry_seq_emb,
          qry_input_ids,
          qry_input_mask,
          is_training,
          bert_config,
          qa_config,
          suffix="_%d" % hop)  # project=True

      (ret_entities, ret_mentions,
       dense_mention_vec, sp_mention_vec) = follow_mention(
           batch_entities, qry_start_emb, qry_end_emb, entity_ids, entity_mask,
           ent2ment_ind, ent2ment_val, ment2ent_map, word_emb_table,
           word_weights, mips_search_fn, tf_db, bert_config.hidden_size,
           mips_config, qa_config, is_training, ensure_mentions)
      # Note:  check this. Shouldn't for wrong choices.
      if exclude_set:
        # batch_ind = tf.expand_dims(tf.range(batch_size), 1)
        exclude_indices = tf.concat([
            tf.cast(exclude_set.indices[:, 0:1], tf.int64),
            tf.cast(tf.expand_dims(exclude_set.values, 1), tf.int64)
        ],
                                    axis=1)
        ret_entities = model_utils.remove_from_sparse(ret_entities,
                                                      exclude_indices)
      ret_entities = tf.sparse.reorder(ret_entities)
      scaled_entities = tf.SparseTensor(
          indices=ret_entities.indices,
          values=ret_entities.values / qa_config.softmax_temperature,
          dense_shape=ret_entities.dense_shape)
      batch_entities = tf.sparse.softmax(scaled_entities)  # entities updated.

      ### Start of debugging w/ tf.Print ###
      tmp_vals = batch_entities.values
      tmp_vals = tf.compat.v1.Print(
          input_=tmp_vals,
          data=[
              ret_entities.indices,
          ],
          message="ret_entities.indices at hop %d \n" % hop,
          first_n=10,
          summarize=50)
      tmp_vals = tf.compat.v1.Print(
          input_=tmp_vals,
          data=[
              ret_entities.values,
          ],
          message="ret_entities.values at hop %d \n" % hop,
          first_n=10,
          summarize=25)
      tmp_vals = tf.compat.v1.Print(
          input_=tmp_vals,
          data=[
              batch_entities.indices,
          ],
          message="scaled_entities.indices at hop %d \n" % hop,
          first_n=10,
          summarize=50)
      tmp_vals = tf.compat.v1.Print(
          input_=tmp_vals,
          data=[
              batch_entities.values,
          ],
          message="scaled_entities.values at hop %d \n" % hop,
          first_n=10,
          summarize=25)
      batch_entities = tf.SparseTensor(
          indices=batch_entities.indices,
          values=tmp_vals,
          dense_shape=batch_entities.dense_shape)
      ### End of debugging w/ tf.Print ###

      ensure_mentions = answer_mentions  # Note: seems not helpful now?
      layer_mentions.append(ret_mentions)
      layer_entities.append(ret_entities)  # Note that this is not sfed.
      layer_dense.append(dense_mention_vec)
      layer_sp.append(sp_mention_vec)

  return (layer_entities, layer_mentions, layer_dense, layer_sp,
          batch_entities_nosc, qry_seq_emb)


def create_drfact_model(bert_config,
                        qa_config,
                        fact_mips_config,
                        is_training,
                        features,
                        ent2fact_ind,
                        ent2fact_val,
                        fact2ent_ind,
                        fact2ent_val,
                        fact2fact_ind,
                        fact2fact_val,
                        entity_ids,
                        entity_mask,
                        use_one_hot_embeddings,
                        summary_obj,
                        num_hops=2,
                        num_preds=100):
  """Creates a classification model wrapper of the DrFact model."""
  qas_ids = features["qas_ids"]  # question ids
  qry_input_ids = features["qry_input_ids"]  # question text token ids
  qry_input_mask = features["qry_input_mask"]  # question text masks (for bert)
  batch_size = tf.shape(qry_input_ids)[0]
  qry_entity_ids = features["qry_entity_id"]  # VarLenFeature
  tf.logging.info("type(qry_entity_ids): %s", type(qry_entity_ids))

  answer_entities = None
  exclude_set_ids = None
  if is_training:
    answer_entities = features["answer_entities"]
    tf.logging.info("type(answer_entities): %s", type(answer_entities))
    tf.logging.info("type(answer_entities.indices): %s",
                    type(answer_entities.indices))
    tf.logging.info("answer_entities.indices.shpae: %s",
                    answer_entities.indices.shape)
    answer_index = tf.SparseTensor(
        indices=tf.concat([
            answer_entities.indices[:, 0:1],
            tf.cast(tf.expand_dims(answer_entities.values, 1), tf.int64)
        ],
                          axis=1),
        values=tf.ones_like(answer_entities.values, dtype=tf.float32),
        dense_shape=[batch_size, qa_config.num_entities])
    # Make sparse version of exclude concepts.
    num_ents = qa_config.num_entities

  (layer_entities, layer_facts, _, _, qry_ents, _,
   qry_seq_emb) = multi_hop_fact(
       qry_input_ids,
       qry_input_mask,
       qry_entity_ids,
       entity_ids,
       entity_mask,
       ent2fact_ind,
       ent2fact_val,
       fact2ent_ind,
       fact2ent_val,
       fact2fact_ind,
       fact2fact_val,
       is_training,
       use_one_hot_embeddings,
       bert_config,
       qa_config,
       fact_mips_config,
       num_hops=num_hops,
       exclude_set=exclude_set_ids,
   )

  # Compute weights for each layer.
  with tf.name_scope("classifier"):
    qry_emb, _ = model_utils.layer_qry_encoder(
        qry_seq_emb,
        qry_input_ids,
        qry_input_mask,
        is_training,
        bert_config,
        qa_config,
        suffix="_cl",
        project_dim=qa_config.projection_dim)
    # Ideally, higher weights on k-th layer for a k-hop question
    output_weights = tf.get_variable(
        "cl_weights", [qa_config.projection_dim,
                       len(layer_entities)],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "cl_bias", [len(layer_entities)], initializer=tf.zeros_initializer())
    logits = tf.matmul(qry_emb, output_weights)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)

    # if is_training:
    #   nrows = qa_config.train_batch_size
    # else:
    #   nrows = qa_config.predict_batch_size

    # def _to_ragged(sp_tensor):
    #   r_ind = tf.RaggedTensor.from_value_rowids(
    #       value_rowids=sp_tensor.indices[:, 0],
    #       values=sp_tensor.indices[:, 1],
    #       nrows=nrows)
    #   r_val = tf.RaggedTensor.from_value_rowids(
    #       value_rowids=sp_tensor.indices[:, 0],
    #       values=sp_tensor.values,
    #       nrows=nrows)
    #   return r_ind, r_val

  def _layer_softmax(entities, hop_id=0):
    uniq_entity_ids, uniq_entity_scs = model_utils.aggregate_sparse_indices(
        entities.indices, entities.values, entities.dense_shape,
        qa_config.entity_score_aggregation_fn)
    # uniq_entity_scs /= 2.0  # Note: softmax_temperature
    logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                             entities.dense_shape)
    logits_sf = tf.sparse.softmax(tf.sparse.reorder(logits))
    ### Start Debugging w/ Print ###
    tmp_vals = logits_sf.values
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            tf.shape(logits.indices)[0],
            logits.indices,
        ],
        message="\n # Layer Entity SoftMax %d \n logits.indices" % hop_id,
        first_n=10,
        summarize=27)
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            tf.shape(logits.values)[0],
            logits.values,
        ],
        message="\n logits.values",
        first_n=10,
        summarize=25)
    tmp_vals = tf.compat.v1.Print(
        input_=tmp_vals,
        data=[
            tf.shape(logits_sf.values)[0],
            logits_sf.values,
        ],
        message="\n logits_sf.values # End of Entity SoftMax #\n",
        first_n=10,
        summarize=25)
    logits_sf = tf.SparseTensor(logits_sf.indices, tmp_vals,
                                logits_sf.dense_shape)
    ### End Debugging w/ Print ###
    return logits_sf

  predictions = {"qas_ids": qas_ids}

  layer_entities_weighted = []
  for i, layer_entity in enumerate(layer_entities):
    # ent_ind, ent_val = _to_ragged(layer_entity)
    # probabilities is the predicted weights of the layer
    layer_entity_sf = _layer_softmax(layer_entity, hop_id=i)
    layer_entities_weighted.append(
        model_utils.batch_multiply(layer_entity_sf, probabilities[:, i]))
    layer_entity_sf_dense = tf.sparse.to_dense(
        layer_entity_sf, default_value=DEFAULT_VALUE, validate_indices=False)
    layer_entity_sf_val, layer_entity_sf_ind = tf.nn.top_k(
        layer_entity_sf_dense, k=100, sorted=True)
    predictions.update({
        "layer_%d_ent" % i: layer_entity_sf_ind,
        "layer_%d_scs" % i: layer_entity_sf_val,
    })

  probs = layer_entities_weighted[0]
  tf.logging.info("layer_entities_weighted: %d", len(layer_entities_weighted))
  for i in range(1, len(layer_entities_weighted)):
    probs = tf.sparse.add(probs, layer_entities_weighted[i])
  probs_dense = tf.sparse.to_dense(
      probs, default_value=DEFAULT_VALUE, validate_indices=False)
  answer_preds = tf.argmax(probs_dense, axis=1)
  top_ent_vals, top_ent_idx = tf.nn.top_k(probs_dense, k=num_preds, sorted=True)

  for hop_id, current_facts in enumerate(layer_facts):
    current_facts_dense = tf.sparse.to_dense(
        current_facts, default_value=DEFAULT_VALUE, validate_indices=False)
    current_fact_vals, current_facts_idx = tf.nn.top_k(
        current_facts_dense, k=100, sorted=True)
    predictions.update({
        "layer_%d_fact_ids" % hop_id: current_facts_idx,
        "layer_%d_fact_scs" % hop_id: current_fact_vals
    })

  qry_ents_dense = tf.sparse.to_dense(
      qry_ents, default_value=DEFAULT_VALUE, validate_indices=False)
  qry_ent_vals, qry_ent_idx = tf.nn.top_k(qry_ents_dense, k=100, sorted=True)
  predictions.update({"qry_ents": qry_ent_idx, "qry_ent_scores": qry_ent_vals})

  total_loss = None
  if is_training:
    # Note: check if this loss function is suitable for multiple answers
    sp_loss = model_utils.compute_loss_from_sptensors(probs, answer_index)
    total_loss = tf.reduce_sum(sp_loss.values) / tf.cast(batch_size, tf.float32)

    # Note: convert probs&ans_index to dense and compute_loss()
    # dense_answer_index = tf.sparse.to_dense(
    #   answer_index, default_value=DEFAULT_VALUE, validate_indices=False)
    # dense_loss = compute_loss(probs_dense, dense_answer_index)

    if summary_obj is not None:  # Note: Where is this?
      num_answers_ret = tf.shape(sp_loss.values)[0]
      for i in range(len(layer_entities)):
        num_ents = tf.cast(tf.shape(layer_entities[i].indices)[0],
                           tf.float32) / tf.cast(batch_size, tf.float32)
        summary_obj.scalar("train/layer_weight_%d" % i,
                           tf.reduce_mean(probabilities[:, i], keepdims=True))
        summary_obj.scalar("train/num_entities_%d" % i,
                           tf.expand_dims(num_ents, 0))
      summary_obj.scalar("train/total_loss", tf.expand_dims(total_loss, 0))
      summary_obj.scalar("train/ans_in_ret", tf.expand_dims(num_answers_ret, 0))
      summary_obj.scalar("train/total_prob_mass",
                         tf.reduce_sum(probs.values, keepdims=True))

  # Update the entity-related prediction information.
  predictions.update({
      "layer_probs": probabilities,
      "top_vals": top_ent_vals,
      "top_idx": top_ent_idx,
      "predictions": answer_preds,
  })

  return total_loss, predictions


def create_drkit_model(bert_config,
                       qa_config,
                       mips_config,
                       is_training,
                       features,
                       ent2ment_ind,
                       ent2ment_val,
                       ment2ent_map,
                       entity_ids,
                       entity_mask,
                       use_one_hot_embeddings,
                       summary_obj,
                       num_hops=2,
                       num_preds=100,
                       is_excluding=False):
  """Creates a classification model."""
  qas_ids = features["qas_ids"]
  qry_input_ids = features["qry_input_ids"]
  qry_input_mask = features["qry_input_mask"]
  batch_size = tf.shape(qry_input_ids)[0]
  qry_entity_ids = features["qry_entity_id"]
  tf.logging.info("type(qry_entity_ids): %s", type(qry_entity_ids))

  answer_entities = None
  exclude_set_ids = None
  if is_training:
    answer_entities = features["answer_entities"]
    answer_index = tf.SparseTensor(
        indices=tf.concat([
            answer_entities.indices[:, 0:1],
            tf.cast(tf.expand_dims(answer_entities.values, 1), tf.int64)
        ],
                          axis=1),
        values=tf.ones_like(answer_entities.values, dtype=tf.float32),
        dense_shape=[batch_size, qa_config.num_entities])
    # Make sparse version of exclude concepts.
    num_ents = qa_config.num_entities
    # Only when it is training.
    if is_excluding:
      exclude_set_ids = features["exclude_set"]
      tf.logging.info("type(exclude_set_ids): %s", type(exclude_set_ids))

  layer_entities, layer_mentions, _, _, el, qry_seq_emb = multi_hop_mention(
      qry_input_ids,
      qry_input_mask,
      qry_entity_ids,
      entity_ids,
      entity_mask,
      ent2ment_ind,
      ent2ment_val,
      ment2ent_map,
      is_training,
      use_one_hot_embeddings,
      bert_config,
      qa_config,
      mips_config,
      num_hops=num_hops,
      exclude_set=exclude_set_ids)
  # The first layer is the query concepts.
  layer_entities = [el] + layer_entities

  # Compute weights for each layer.
  with tf.name_scope("classifier"):
    qry_emb, _ = model_utils.layer_qry_encoder(
        qry_seq_emb,
        qry_input_ids,
        qry_input_mask,
        is_training,
        bert_config,
        qa_config,
        suffix="_cl")
    # Ideally, higher weights on k-th layer for a k-hop question
    # Note: can we make answer-aware hop weighting?
    output_weights = tf.get_variable(
        "cl_weights", [qa_config.projection_dim,
                       len(layer_entities)],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "cl_bias", [len(layer_entities)], initializer=tf.zeros_initializer())
    logits = tf.matmul(qry_emb, output_weights)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)

  if is_training:
    nrows = qa_config.train_batch_size
  else:
    nrows = qa_config.predict_batch_size

  def _to_ragged(sp_tensor):
    r_ind = tf.RaggedTensor.from_value_rowids(
        value_rowids=sp_tensor.indices[:, 0],
        values=sp_tensor.indices[:, 1],
        nrows=nrows)
    r_val = tf.RaggedTensor.from_value_rowids(
        value_rowids=sp_tensor.indices[:, 0],
        values=sp_tensor.values,
        nrows=nrows)
    return r_ind, r_val

  def _layer_softmax(entities):
    uniq_entity_ids, uniq_entity_scs = model_utils.aggregate_sparse_indices(
        entities.indices, entities.values, entities.dense_shape,
        qa_config.entity_score_aggregation_fn)
    uniq_entity_scs /= qa_config.softmax_temperature
    logits = tf.SparseTensor(uniq_entity_ids, uniq_entity_scs,
                             entities.dense_shape)
    return tf.sparse.softmax(tf.sparse.reorder(logits))

  predictions = {"qas_ids": qas_ids}

  layer_preds = []
  for i, layer_mention in enumerate(layer_mentions):
    layer_preds.append(
        tf.argmax(
            tf.sparse.to_dense(
                layer_mention,
                default_value=DEFAULT_VALUE,
                validate_indices=False),
            axis=1))
    men_ind, men_val = _to_ragged(layer_mention)
    predictions.update({
        "layer_%d_men" % i: men_ind.to_tensor(default_value=-1),
        "layer_%d_mscs" % i: men_val.to_tensor(default_value=-1),
    })

  layer_entities_weighted = []
  for i, layer_entity in enumerate(layer_entities):
    ent_ind, ent_val = _to_ragged(layer_entity)
    predictions.update({
        "layer_%d_ent" % i: ent_ind.to_tensor(default_value=-1),
        "layer_%d_scs" % i: ent_val.to_tensor(default_value=-1),
    })
    layer_entities_weighted.append(
        model_utils.batch_multiply(
            _layer_softmax(layer_entity), probabilities[:, i]))

  probs = tf.sparse.add(layer_entities_weighted[0], layer_entities_weighted[1])
  for i in range(2, len(layer_entities_weighted)):
    probs = tf.sparse.add(probs, layer_entities_weighted[i])

  probs_dense = tf.sparse.to_dense(
      probs, default_value=DEFAULT_VALUE, validate_indices=False)
  answer_preds = tf.argmax(probs_dense, axis=1)
  top_vals, top_idx = tf.nn.top_k(probs_dense, k=num_preds, sorted=True)

  total_loss = None
  if is_training:
    # Note: check if this loss function is suitable for multiple answers
    # Note: convert probs&ans_index to dense and compute_loss()
    sp_loss = model_utils.compute_loss_from_sptensors(probs, answer_index)
    total_loss = tf.reduce_sum(sp_loss.values) / tf.cast(batch_size, tf.float32)
    num_answers_ret = tf.shape(sp_loss.values)[0]
    if summary_obj is not None:
      for i in range(len(layer_entities)):
        num_ents = tf.cast(tf.shape(layer_entities[i].indices)[0],
                           tf.float32) / tf.cast(batch_size, tf.float32)
        summary_obj.scalar("train/layer_weight_%d" % i,
                           tf.reduce_mean(probabilities[:, i], keepdims=True))
        summary_obj.scalar("train/num_entities_%d" % i,
                           tf.expand_dims(num_ents, 0))
      summary_obj.scalar("train/total_loss", tf.expand_dims(total_loss, 0))
      summary_obj.scalar("train/ans_in_ret", tf.expand_dims(num_answers_ret, 0))
      summary_obj.scalar("train/total_prob_mass",
                         tf.reduce_sum(probs.values, keepdims=True))

  predictions.update({
      "layer_probs": probabilities,
      "top_vals": top_vals,
      "top_idx": top_idx,
      "predictions": answer_preds,
      "layer_predictions": tf.stack(layer_preds, axis=1),
  })

  return total_loss, predictions
