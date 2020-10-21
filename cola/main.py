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

"""Self-supervised Contrastive Learning for Audio."""

from absl import app
from absl import flags
import tensorflow as tf

from cola import constants
from cola import contrastive
from cola import supervised

FLAGS = flags.FLAGS

flags.DEFINE_string("tpu_address", None, "TPU Address.")

flags.DEFINE_string("experiment_id", None,
                    "Unique id to use for model checkpointing.")

flags.DEFINE_string("strategy", "tpu",
                    "TF distribute strategy either of `tpu` or `gpu`.")

flags.DEFINE_enum_class("training_mode", constants.TrainingMode.SSL,
                        constants.TrainingMode, "Mode of model training.")

flags.DEFINE_string("model_dir", None,
                    "Path to directory where to store models.")

flags.DEFINE_enum_class(
    "ssl_dataset", constants.Dataset.AS, constants.Dataset,
    "Name of the dataset to use for self-supervised pre-training.")

flags.DEFINE_enum_class("ds_dataset", constants.Dataset.MUSAN,
                        constants.Dataset,
                        "Name of the downstream task dataset.")

flags.DEFINE_string("ssl_checkpoint_id", None,
                    "Self-supervised model checkpoint id.")

flags.DEFINE_integer("batch_size", 64,
                     "Batch size to use for training the network.")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

flags.DEFINE_integer("epochs", 100, "Number of training epochs.")

flags.DEFINE_integer("embedding_dim", 512,
                     "Embedding size of contrastive model.")

flags.DEFINE_float("temperature", 0.2,
                   "Temperature for normalizing similarities.")

flags.DEFINE_string("pooling_type", "max", "Global pooling type.")

flags.DEFINE_float("noise", 0.001, "Noise rate to use for postive samples.")

flags.DEFINE_enum_class("similarity_type", constants.SimilarityMeasure.BILINEAR,
                        constants.SimilarityMeasure,
                        "Similarity measure for the contrastive model.")

flags.DEFINE_bool("freeze_encoder", True,
                  "Whether to freeze encoder or fine tune entire model.")


def main(_):
  if FLAGS.strategy == "tpu":
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu_address)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
  elif FLAGS.strategy == "gpu":
    strategy = tf.distribute.MirroredStrategy()
  else:
    raise ValueError("Unknown distribution strategy.")

  if FLAGS.training_mode == constants.TrainingMode.SSL:
    model = contrastive.ContrastiveModel(
        strategy=strategy,
        ssl_dataset_name=FLAGS.ssl_dataset,
        ds_dataset_name=FLAGS.ds_dataset,
        model_path=FLAGS.model_dir,
        experiment_id=FLAGS.experiment_id,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        learning_rate=FLAGS.learning_rate,
        temperature=FLAGS.temperature,
        embedding_dim=FLAGS.embedding_dim,
        similarity_type=FLAGS.similarity_type,
        pooling_type=FLAGS.pooling_type,
        noise=FLAGS.noise)
    model.train()
  elif FLAGS.training_mode == constants.TrainingMode.RND:
    model = supervised.SupervisedModule(
        ssl_dataset_name=FLAGS.ssl_dataset,
        ds_dataset_name=FLAGS.ds_dataset,
        model_path=FLAGS.model_dir,
        experiment_id=FLAGS.experiment_id,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        learning_rate=FLAGS.learning_rate)
    model.train_eval(
        load_pretrained=False, contrastive_pooling_type=FLAGS.pooling_type)
  elif FLAGS.training_mode == constants.TrainingMode.SUP:
    model = supervised.SupervisedModule(
        ssl_dataset_name=FLAGS.ssl_dataset,
        ds_dataset_name=FLAGS.ds_dataset,
        model_path=FLAGS.model_dir,
        experiment_id=FLAGS.experiment_id,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        learning_rate=FLAGS.learning_rate)
    model.train_eval(
        load_pretrained=False,
        freeze_encoder=False,
        contrastive_pooling_type=FLAGS.pooling_type)
  elif FLAGS.training_mode == constants.TrainingMode.DS:
    model = supervised.SupervisedModule(
        ssl_dataset_name=FLAGS.ssl_dataset,
        ds_dataset_name=FLAGS.ds_dataset,
        model_path=FLAGS.model_dir,
        experiment_id=FLAGS.experiment_id,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        learning_rate=FLAGS.learning_rate)
    model.train_eval(
        freeze_encoder=FLAGS.freeze_encoder,
        ssl_model_ckpt_id=FLAGS.ssl_checkpoint_id,
        contrastive_embedding_dim=FLAGS.embedding_dim,
        contrastive_temperature=FLAGS.temperature,
        contrastive_pooling_type=FLAGS.pooling_type,
        contrastive_similarity_type=FLAGS.similarity_type)
  else:
    raise ValueError("Unknown training mode.")


if __name__ == "__main__":
  app.run(main)
