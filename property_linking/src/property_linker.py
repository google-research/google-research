# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Main file for property_linking project.
"""
import tensorflow as tf
import tensorflow_hub as hub

from property_linking.src import kb
from property_linking.src import model
from property_linking.src import trainer
from property_linking.src import util
import property_linking.src.bert_util as bert

flags = tf.flags
FLAGS = flags.FLAGS

# Data params
flags.DEFINE_string("root_dir", "",
                    "Root directory. Contains *_{kb, cats, name}.tsv files")
flags.DEFINE_string("bert_dir",
                    "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1",
                    "Path to BERT tfhub module")
flags.DEFINE_string("sitelinks", None, "Path to wikidata sitelinks file")
flags.DEFINE_string("ckpt_dir", None, "Directory to save model checkpoints")
flags.DEFINE_string("restore_ckpt_dir", None, "Location to restore from")
flags.DEFINE_string("train", "", "Suffix of train file")
flags.DEFINE_string("test", "_dev", "Suffix of test file")
flags.DEFINE_integer("max_training_size", None, "Limit size of training set")
# zoo is a small default domain good for debugging
flags.DEFINE_string("domain", "zoo",
                    "Domain/partition of KB, fills the * in data_dir")
flags.DEFINE_string("test_domain", None,
                    "Domain/partition of KB, fills the * in data_dir")


# KB Params
flags.DEFINE_float("kb_dropout", 0.0, "Delete fraction of edges from kb")
flags.DEFINE_integer("max_relations", 200, "No. of relations we care about")
flags.DEFINE_integer("max_core_size", 45000, "Max core entities in KG")
flags.DEFINE_integer("max_noncore_size", 60000, "Max noncore entities in KG")
flags.DEFINE_integer("min_noncore_cutoff", 0,
                     "Keep values ranked less frequent than this")
flags.DEFINE_integer("max_constants_cutoff", 3000,
                     "Keep numeric values ranked more frequent than this")

# BERT preprocessing params

flags.DEFINE_integer("max_query_length", 20, "Maximum length to be encoded")
flags.DEFINE_integer("bert_batch_size", 7000, "Batch size with BERT")
flags.DEFINE_list("layers", [0, 11], "Layers of BERT to use")

# Model architecture params
flags.DEFINE_float("dropout", 0.05, "Dropout currently used for prior only.")
flags.DEFINE_string("logits", "mixed",
                    "Strategy for mixing logits [prior, sim, mixed]")
flags.DEFINE_string("loss_type", "mixed",
                    "Strategy for mixing loss [distant, direct, mixed]")
flags.DEFINE_bool("enforce_type", False,
                  "Whether to force first (and only first) property to be isa")
flags.DEFINE_bool("weight_examples", False,
                  "Whether to weight examples by predicted set size")
flags.DEFINE_float("weight_regularizer", 3,
                   "Coefficient for regularization term")

flags.DEFINE_integer("layer_size", 768, "Size of input embeddings")
flags.DEFINE_integer("num_layers", 3, "Number of layers in model")
flags.DEFINE_integer("max_properties", 2,
                     "Maximum number of property distributions to decode")

# Training params
flags.DEFINE_float("smoothing_param", 1e-2,
                   "Smooth output distributions to discourage underflow")
flags.DEFINE_float("learning_rate", 1e-2, "Learning rate")
flags.DEFINE_float("time_reg", 0, "distance regularization per timestep")
flags.DEFINE_integer("num_epochs", 10000, "Number of iterations through data")
flags.DEFINE_integer("batch_size", 128, "Batch size in training")
flags.DEFINE_integer("log_frequency", 100, "Log after this many batches")


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  root_dir = FLAGS.root_dir
  domain = FLAGS.domain
  kb_file = "{}/{}_{}.tsv".format(root_dir, domain, "kb")
  cats_file = "{}/{}_{}{}.tsv".format(root_dir, domain, "cats", FLAGS.train)
  cats_test_file = "{}/{}_{}{}{}.tsv".format(root_dir, domain, "cats",
                                             FLAGS.train, FLAGS.test)
  names_file = "{}/{}_{}.tsv".format(root_dir, domain, "names")
  if FLAGS.ckpt_dir is not None:
    tf.gfile.MakeDirs(FLAGS.ckpt_dir)
  restore_ckpt_dir = FLAGS.restore_ckpt_dir

  train_builder = kb.Builder(kb_file=kb_file,
                             cats_file=cats_file,
                             names_file=names_file,
                             kb_dropout=FLAGS.kb_dropout,
                             max_relations=FLAGS.max_relations,
                             max_core_size=FLAGS.max_core_size,
                             max_noncore_size=FLAGS.max_noncore_size,
                             min_noncore_cutoff=FLAGS.min_noncore_cutoff,
                             max_constants_cutoff=FLAGS.max_constants_cutoff)

  train_context = train_builder.build_context()

  bert_session = tf.Session()
  bert_module = hub.Module(FLAGS.bert_dir, trainable=False)
  bert_session.run(tf.global_variables_initializer())
  bh = bert.BertHelper(bert_session,
                       FLAGS.bert_dir,
                       FLAGS.max_query_length,
                       FLAGS.bert_batch_size,
                       bert_module)

  train_value_encodings = util.create_node_encodings(train_builder,
                                                     train_context,
                                                     bh,
                                                     "val_g")
  train_relation_encodings = util.create_node_encodings(train_builder,
                                                        train_context,
                                                        bh,
                                                        "rel_g")
  prune_empty = (FLAGS.loss_type == "mixed" or FLAGS.loss_type == "distant")
  train_examples = util.create_examples(train_builder,
                                        train_context,
                                        cats_file,
                                        bh,
                                        prune_empty=prune_empty)

  if FLAGS.max_training_size:
    train_examples = train_examples[:FLAGS.max_training_size]

  test_examples = util.create_examples(train_builder,
                                       train_context,
                                       cats_test_file,
                                       bh)
  tf.logging.info("{} training examples, {} dev/test examples".format(
      len(train_examples),
      len(test_examples)))
  tf.logging.info("Example: {}".format(train_examples[0]))

  # Build test stuff before model, use training domain if nothing specified
  if FLAGS.test_domain is not None:
    test_kb_file = "{}/{}_{}.tsv".format(root_dir, FLAGS.test_domain, "kb")
    test_cats_file = "{}/{}_{}.tsv".format(root_dir, FLAGS.test_domain, "cats")
    test_cats_test_file = "{}/{}_{}{}.tsv".format(root_dir,
                                                  FLAGS.test_domain,
                                                  "cats",
                                                  FLAGS.test)
    test_names_file = "{}/{}_{}.tsv".format(root_dir,
                                            FLAGS.test_domain,
                                            "names")

    test_builder = kb.Builder(kb_file=test_kb_file,
                              cats_file=test_cats_file,
                              names_file=test_names_file,
                              max_constants_cutoff=FLAGS.max_constants_cutoff)

    test_context = test_builder.build_context()

    test_value_encodings = util.create_node_encodings(test_builder,
                                                      test_context,
                                                      bh,
                                                      "val_g")
    test_relation_encodings = util.create_node_encodings(test_builder,
                                                         test_context,
                                                         bh,
                                                         "rel_g")
    true_test_examples = util.create_examples(test_builder,
                                              test_context,
                                              test_cats_test_file,
                                              bh)

  tf.reset_default_graph()  # Clear out anything residual from BERT

  with tf.variable_scope("model_wrapper"):
    train_model = model.Model(train_context,
                              train_value_encodings,
                              train_relation_encodings,
                              num_gpus=1)
  if FLAGS.test_domain is not None:
    with tf.variable_scope("model_wrapper", reuse=True):
      test_model = model.Model(test_context,
                               test_value_encodings,
                               test_relation_encodings,
                               num_gpus=1,
                               encoder=train_model.encoder)

  pl_trainer = trainer.Trainer(train_builder,
                               train_context,
                               train_model,
                               train_examples,
                               test_examples,
                               loss_type=FLAGS.loss_type,
                               save_ckpt_dir=FLAGS.ckpt_dir,
                               restore_ckpt_dir=FLAGS.restore_ckpt_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  sess.run(tf.global_variables_initializer())
  pl_trainer.restore_model(sess)
  pl_trainer.train(sess)
  tf.logging.info("Training Done.")
  pl_trainer.evaluate(sess)
  tf.logging.info("Evaluation on train kb done.")

  if FLAGS.ckpt_dir is not None:
    restore_ckpt_dir = FLAGS.ckpt_dir
    tf.logging.info("Reset; already saved at {}".format(restore_ckpt_dir))
  elif FLAGS.restore_ckpt_dir is not None:
    tf.logging.info("Continuing; restoring from {}".format(restore_ckpt_dir))
  else:
    tf.logging.info("No restore_ckpt_dir, continuing anyway")

  if FLAGS.test_domain is not None:
    pl_tester = trainer.Trainer(test_builder,
                                test_context,
                                test_model,
                                None,
                                true_test_examples,
                                loss_type=FLAGS.loss_type,
                                restore_ckpt_dir=restore_ckpt_dir)
    pl_trainer.restore_model(sess)
    pl_tester.evaluate(sess)
    tf.logging.info("Evaluation on test kb done.")
  else:
    tf.logging.info("Skipping: Evaluation on test kb")

if __name__ == "__main__":
  tf.compat.v1.app.run()
