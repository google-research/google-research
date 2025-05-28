# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# pylint: skip-file
import os

from absl import app
from absl import flags

import ml_collections
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
import itertools
import pickle
from typing import Any, Sequence
from scipy.stats import mode
# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

import flax.optim
from flax.training import checkpoints

from robustness.tools.breeds_helpers import BreedsDatasetGenerator
from representation_clustering import resnet_v1
from representation_clustering.configs.default_breeds import get_config
from representation_clustering.input_pipeline import predicate, RescaleValues, ResizeSmall, CentralCrop
from clu import preprocess_spec

flags.DEFINE_string("exp_dir", None,
                    "Experiment directory where checkpoints are saved.")


BREEDS_INFO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "breeds")


FLAGS = flags.FLAGS


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  batch_stats: Any


def create_train_state(config, rng,
                       input_shape, num_classes):
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.
    num_classes: Number of classes in the output layer.

  Returns:
    The initialized TrainState with the optimizer.
  """
  if config.model_name == "resnet18":
    model_cls = resnet_v1.ResNet18
  elif config.model_name == "resnet50":
    model_cls = resnet_v1.ResNet50
  else:
    raise ValueError(f"Model {config.model_name} not supported.")
  model = model_cls(num_classes=num_classes)
  variables = model.init(rng, jnp.ones(input_shape), train=False)
  params = variables["params"]
  batch_stats = variables["batch_stats"]
  optimizer = flax.optim.Momentum(beta=config.sgd_momentum).create(params)
  return model, TrainState(step=0, optimizer=optimizer, batch_stats=batch_stats)


def compute_purity(clusters, classes):
  """Compute purity of the cluster."""
  n_cluster_points = 0
  for cluster_idx in set(clusters):
    instance_idx = np.where(clusters == cluster_idx)[0]
    subclass_labels = classes[instance_idx]
    mode_stats = mode(subclass_labels)
    n_cluster_points += mode_stats[1][0]
  purity = n_cluster_points / len(clusters)
  return purity


def cluster_each_checkpoint(checkpoint_path,
                            num_classes,
                            train_subclasses,
                            eval_ds,
                            overcluster_factor=5):
  """Cluster (within each superclass) the representations of a model loaded from a checkpoint."""
  config = get_config()
  model, state = create_train_state(
      config,
      jax.random.PRNGKey(0),
      input_shape=(8, 224, 224, 3),
      num_classes=num_classes)
  state = checkpoints.restore_checkpoint(checkpoint_path, state)

  all_intermediates = []
  all_subclass_labels = []
  all_filenames = []
  all_images = []
  for step, batch in enumerate(eval_ds):
    if step % 20 == 0:
      print(step)
    intermediates = predict(model, state, batch)
    labels = batch["label"].numpy()
    bs = labels.shape[0]
    all_subclass_labels.append(labels)
    all_images.append(batch["image"].numpy())
    if "file_name" in batch:
      all_filenames.append(batch["file_name"].numpy())
    all_intermediates.append(
        np.mean(intermediates["stage4"]["__call__"][0],
                axis=(1, 2)).reshape(bs, -1))

  all_intermediates = np.vstack(all_intermediates)
  all_subclass_labels = np.hstack(all_subclass_labels)
  all_images = np.vstack(all_images)

  all_clfs = []
  for subclasses in train_subclasses:
    subclass_idx = np.array([
        i for i in range(len(all_subclass_labels))
        if all_subclass_labels[i] in subclasses
    ])
    hier_clustering = AgglomerativeClustering(
        n_clusters=len(subclasses) * overcluster_factor,
        linkage="ward").fit(all_intermediates[subclass_idx])
    all_clfs.append(hier_clustering)

  purity_list = []
  for i, clf in enumerate(all_clfs):
    subclasses = train_subclasses[i]
    subclass_idx = np.array([
        i for i in range(len(all_subclass_labels))
        if all_subclass_labels[i] in subclasses
    ])
    subclass_labels = all_subclass_labels[subclass_idx]
    purity = compute_purity(clf.labels_, subclass_labels)
    purity_list.append(purity)
  return purity_list


def load_eval_ds(train_subclasses):
  """Load validation dataset."""
  all_subclasses = list(itertools.chain(*train_subclasses))
  new_label_map = {}
  for subclass_idx, sub in enumerate(all_subclasses):
    new_label_map.update({sub: subclass_idx})

  dataset_builder = tfds.builder("imagenet2012", try_gcs=True)
  eval_preprocess = preprocess_spec.PreprocessFn([
      RescaleValues(),
      ResizeSmall(256),
      CentralCrop(224),
  ],
                                                 only_jax_types=True)
  dataset_options = tf.data.Options()
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.experimental_threading.private_threadpool_size = 48
  dataset_options.experimental_threading.max_intra_op_parallelism = 1

  read_config = tfds.ReadConfig(shuffle_seed=None, options=dataset_options)
  eval_ds = dataset_builder.as_dataset(
      split=tfds.Split.VALIDATION,
      shuffle_files=False,
      read_config=read_config,
      decoders=None)

  batch_size = 128
  eval_ds = eval_ds.filter(
      functools.partial(predicate, all_subclasses=all_subclasses))
  eval_ds = eval_ds.cache()
  eval_ds = eval_ds.map(
      eval_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  eval_ds = eval_ds.batch(batch_size, drop_remainder=False)
  eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)
  return eval_ds


def predict(model, state, batch):
  """Use model to make predictions on a batch."""
  variables = {
      "params": state.optimizer.target,
      "batch_stats": state.batch_stats
  }
  _, state = model.apply(
      variables,
      batch["image"],
      capture_intermediates=True,
      mutable=["intermediates"],
      train=False)
  intermediates = state["intermediates"]
  return intermediates


def main(argv):
  del argv
  try:
    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
      assert device.device_type != "GPU"
  except:
    pass

  DG = BreedsDatasetGenerator(BREEDS_INFO_DIR)
  # (TODO) customize level and Nsubclasses based on exp_dir
  ret = DG.get_superclasses(
      level=3, Nsubclasses=20, split=None, ancestor=None, balanced=True)
  _, subclass_split, _ = ret
  train_subclasses = subclass_split[0]
  num_classes = len(train_subclasses)

  eval_ds = load_eval_ds(train_subclasses)
  for ckpt_number in range(1, 161, 20):
    checkpoint_path = os.path.join(FLAGS.exp_dir,
                                   f"/checkpoints-0/ckpt-{ckpt_number}.flax")
    purity_list = cluster_each_checkpoint(checkpoint_path, num_classes,
                                          train_subclasses, eval_ds)
    out_file = os.path.join(FLAGS.exp_dir, f"class_purity_ckpt_{ckpt_number}.pkl")
    with tf.io.gfile.GFile(out_file, 'wb') as f:
      pickle.dump(cluster_filenames_dict, f)


if __name__ == "__main__":
  app.run(main)
