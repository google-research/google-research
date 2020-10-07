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

"""Run trainer."""

import json
import os
import time

from absl import flags
import tensorflow.compat.v1 as tf

from multi_resolution_rec import model as model_lib
from multi_resolution_rec import util

# Dataset
flags.DEFINE_string('dataset', '', 'Path to dataset')
flags.DEFINE_string('train_dir', '', 'Training dir.')
flags.DEFINE_string('query_map_path', '', 'Path to the query map object.')

# Model variants
flags.DEFINE_string(
    'use_last_query', 'no_query',
    'Currently we have two model variants: no_query, query_bow')
flags.DEFINE_string(
    'query_item_combine', 'concat',
    """Strategy to combine history embedding with query embedding.
    Options are: only_query, concat, sum""")
flags.DEFINE_string(
    'query_item_attention', 'none',
    """Strategy to apply query to item history attention. Available variants:
    none: Attention is not applied in any form.
    multihead: Applies multihead attention (depends on num_query_attn_heads).
    memory: Applies memory attention (depends on num_query_attn_layers).
    multihead_position: Applies multihead attention with position prior.
    multihead_time: Applies multihead attention with time prior.""")

# Model hyper-parameters
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer('maxseqlen', 30, 'Max len.')
flags.DEFINE_integer('maxquerylen', 30, 'Max len of tokens per query.')
flags.DEFINE_integer('hidden_units', 60, 'Number of hidden units.')
flags.DEFINE_integer('num_self_attn_layers', 2,
                     'Number of self-attention layers.')
flags.DEFINE_integer('num_epochs', 2001, 'Number of epochs.')
flags.DEFINE_integer('num_self_attn_heads', 1,
                     'Number of self-attention heads.')
flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate.')
flags.DEFINE_float('l2_emb', 0.0, 'L2 regularization.')

# Query-specific hyper-parameters
flags.DEFINE_float('token_drop_prob', 0.0,
                   'Probability of dropping a token from a query.')
flags.DEFINE_bool(
    'query_layer_norm', True,
    'If true, layers will be normalized after query embedding is combined.')
flags.DEFINE_bool(
    'query_residual', False,
    'If true, the final feed-forward layer has residual connections.')
flags.DEFINE_integer('num_final_layers', 1,
                     'Defines the depth of the final feed-forward layer.')
flags.DEFINE_integer('num_query_attn_heads', 1,
                     'Number of query to history attention heads.')
flags.DEFINE_integer(
    'num_query_attn_layers', 1,
    """Number of query to history attention layers. This parameter is only used
    for memory attention. For all other settings, it is set to 1.""")

# Hyper-parameters for multihead_time attention variant.
flags.DEFINE_float('time_exp_base', 3., 'Base for exponential time intervals.')
flags.DEFINE_bool(
    'overlapping_chunks', False,
    """If true, chunks will overlap: say we have times t1<t2<..<tn. Then, the
    1st chunk covers [0, t1], 2nd chunk covers [0, t2] and so on. If False,
    chunks will respectively cover [0, t1], (t1, t2] and so on.""")

# Evaluation
flags.DEFINE_integer('neg_sample_size_eval', 100,
                     'Number of negatives for evaluation.')
flags.DEFINE_string(
    'sampling_strategy', 'popularity',
    'Strategy to sample evaluation samples: popularity, random.')

# Logging and summary
flags.DEFINE_bool(
    'save_train_eval', True,
    'If true, model saves summary of evaluation on training set.')
flags.DEFINE_float('eval_frequency', 20,
                   'Number of epochs between two evaluations.')

FLAGS = flags.FLAGS
NUM_PRESAMPLED_LISTS_OF_POPULARITY_NEGATIVES = 1000


def main(_):

  # Load raw data and organize training directory.
  with tf.gfile.Open(FLAGS.query_map_path, 'r') as f:
    query_map = json.load(f)
  tf.logging.info('Query map is loaded.')
  dataset = util.data_partition(FLAGS.dataset)
  tf.logging.info('Data is loaded')
  [user_train, _, _, usernum, _, itemnum, user_query_seed,
   item_popularity] = dataset
  num_batch = int(len(user_train) / FLAGS.batch_size)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  # Create summary/log files and respective variables.
  train_summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.train_dir, 'train'))
  valid_summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.train_dir, 'validation'))
  test_summary_writer = tf.summary.FileWriter(
      os.path.join(FLAGS.train_dir, 'test'))
  value_loss = tf.placeholder(tf.float32, [])
  value_ndcg = tf.placeholder(tf.float32, [])
  value_hit = tf.placeholder(tf.float32, [])
  summary_loss = tf.summary.scalar('Loss', value_loss)
  summary_ndcg = tf.summary.scalar('NDCG@10', value_ndcg)
  summary_hit = tf.summary.scalar('HIT@10', value_hit)
  log_filename = os.path.join(FLAGS.train_dir, 'log.txt')
  f = tf.gfile.Open(log_filename, 'w')
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess = tf.Session(config=config)

  # Fetch dataset
  tf_dataset, vocab, query_word_ids = util.create_tf_dataset(
      user_train,
      FLAGS.batch_size,
      itemnum=itemnum,
      query_map=query_map,
      maxquerylen=FLAGS.maxquerylen,
      maxseqlen=FLAGS.maxseqlen,
      token_drop_prob=FLAGS.token_drop_prob,
      user_query_seed=user_query_seed,
      randomize_input=True,
      random_seed=0)

  # Create model
  model = model_lib.Model(
      usernum,
      itemnum,
      len(vocab),
      use_last_query=FLAGS.use_last_query,
      maxseqlen=FLAGS.maxseqlen,
      maxquerylen=FLAGS.maxquerylen,
      hidden_units=FLAGS.hidden_units,
      l2_emb=FLAGS.l2_emb,
      dropout_rate=FLAGS.dropout_rate,
      lr=FLAGS.lr,
      num_self_attn_heads=FLAGS.num_self_attn_heads,
      num_query_attn_heads=FLAGS.num_query_attn_heads,
      num_self_attn_layers=FLAGS.num_self_attn_layers,
      num_query_attn_layers=FLAGS.num_query_attn_layers,
      num_final_layers=FLAGS.num_final_layers,
      query_item_attention=FLAGS.query_item_attention,
      query_item_combine=FLAGS.query_item_combine,
      query_layer_norm=FLAGS.query_layer_norm,
      query_residual=FLAGS.query_residual,
      time_exp_base=FLAGS.time_exp_base,
      overlapping_chunks=FLAGS.overlapping_chunks)
  tf.logging.info('Model is created.')
  sess.run(tf.global_variables_initializer())
  raw_time = 0.0
  t0 = time.time()

  iterator = tf_dataset.make_one_shot_iterator()
  batch_data = iterator.get_next()
  user_id = batch_data['user_ids']
  item_seq = batch_data['items']
  query_seq = batch_data['queries']
  query_words_seq = batch_data['query_words']
  time_seq = batch_data['times']
  label_seq = batch_data['labels']
  random_neg = batch_data['random_neg']

  # For popularity based negative sampling, we priorly sample a large set of
  # lists (each consisting FLAGS.neg_sample_size_eval many negative samples);
  # and later randomly select one of the pre-sampled lists while evaluating each
  # user. Since sampling a list of elements with a given probability
  # distributuion is a rather slow operation, this is a much faster approach
  # compared to re-sampling for each user every time we perform evaluation.
  presampled_negatives = []
  if FLAGS.sampling_strategy == 'popularity':
    tf.logging.info('Presampling negatives for popularity based strategy.')
    presampled_negatives.extend(
        util.presample_popularity_negatives(
            1,
            itemnum + 1,
            FLAGS.neg_sample_size_eval,
            item_popularity,
            NUM_PRESAMPLED_LISTS_OF_POPULARITY_NEGATIVES,
        ),
    )

  # Start training.
  for epoch in range(1, FLAGS.num_epochs + 1):
    tf.logging.info('Epoch %d' % epoch)
    epoch_loss = 0
    for _ in range(num_batch):
      u, x, q, q_w, t, y, ny = sess.run([
          user_id, item_seq, query_seq, query_words_seq, time_seq, label_seq,
          random_neg
      ])
      loss, _ = sess.run(
          [model.loss, model.train_op], {
              model.u: u,
              model.item_seq: x,
              model.query_seq: q,
              model.query_words_seq: q_w,
              model.time_seq: t,
              model.pos: y,
              model.neg: ny,
              model.is_training: True
          })
      epoch_loss += loss

    # Adding average epoch train loss summary.
    train_summary_writer.add_summary(
        sess.run(
            summary_loss,
            feed_dict={value_loss: float(epoch_loss / num_batch)}), epoch)

    # Evaluate.
    if epoch % FLAGS.eval_frequency == 0:
      t1 = time.time() - t0
      raw_time += t1
      tf.logging.info('Evaluating')
      tf.logging.info('Sampling strategy is: {}'.format(
          FLAGS.sampling_strategy))
      t_test = util.evaluate(
          model,
          dataset,
          query_word_ids,
          FLAGS.maxseqlen,
          FLAGS.maxquerylen,
          sess,
          FLAGS.token_drop_prob,
          neg_sample_size=FLAGS.neg_sample_size_eval,
          presampled_negatives=presampled_negatives,
          eval_on='test')
      t_valid = util.evaluate(
          model,
          dataset,
          query_word_ids,
          FLAGS.maxseqlen,
          FLAGS.maxquerylen,
          sess,
          FLAGS.token_drop_prob,
          neg_sample_size=FLAGS.neg_sample_size_eval,
          presampled_negatives=presampled_negatives,
          eval_on='valid')
      eval_str = ('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f),'
                  ' test (NDCG@10: %.4f, HR@10: %.4f)') % (
                      epoch, raw_time, t_valid[0], t_valid[1], t_test[0],
                      t_test[1])
      tf.logging.info(eval_str)
      f.write(eval_str + '\n')
      f.flush()
      t0 = time.time()
      valid_summary_writer.add_summary(
          sess.run(summary_ndcg, feed_dict={value_ndcg: t_valid[0]}), epoch)
      valid_summary_writer.add_summary(
          sess.run(summary_hit, feed_dict={value_hit: t_valid[1]}), epoch)
      test_summary_writer.add_summary(
          sess.run(summary_ndcg, feed_dict={value_ndcg: t_test[0]}), epoch)
      test_summary_writer.add_summary(
          sess.run(summary_hit, feed_dict={value_hit: t_test[1]}), epoch)

      # Evaluate on train split.
      if FLAGS.save_train_eval:
        t_train = util.evaluate(
            model,
            dataset,
            query_word_ids,
            FLAGS.maxseqlen,
            FLAGS.maxquerylen,
            sess,
            FLAGS.token_drop_prob,
            neg_sample_size=FLAGS.neg_sample_size_eval,
            presampled_negatives=presampled_negatives,
            eval_on='train')
        train_str = ('epoch:%d, time: %f(s), train (NDCG@10: %.4f, HR@10: %.4f)'
                    ) % (epoch, raw_time, t_train[0], t_train[1])
        tf.logging.info(train_str)
        train_summary_writer.add_summary(
            sess.run(summary_ndcg, feed_dict={value_ndcg: t_train[0]}), epoch)
        train_summary_writer.add_summary(
            sess.run(summary_hit, feed_dict={value_hit: t_train[1]}), epoch)
  tf.logging.info('Done. Log written to %s' % log_filename)
  sess.close()
  f.close()

if __name__ == '__main__':
  tf.app.run()
