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

"""Transfer Experiment common functions."""

import dataclasses

import numpy as np
import tensorflow as tf
import tqdm

from stable_transfer.transferability import config_transfer_experiment as config
from stable_transfer.transferability import datasets
from stable_transfer.transferability import datasets_info
from stable_transfer.transferability import features_targets_utils
from stable_transfer.transferability import networks
from stable_transfer.transferability import utils


@dataclasses.dataclass
class TransferExperiment():
  """Transfer Experiment Functions."""
  config: config.TransferExperimentConfig

  @property
  def target_classes(self):
    """Returns (sorted) list of selected classes."""
    if self.config.source.network_architecture == 'HRNet':
      return None  # Not required for semantic segmentation experiments.

    selection_method = self.config.target.class_selection.method
    selection_experiment = self.config.target.class_selection.experiment_number
    selection_seed = self.config.target.class_selection.seed

    assert selection_method in ['all', 'random', 'fixed']

    if selection_method == 'all':
      return None

    np.random.seed(selection_seed + selection_experiment)
    if selection_method == 'random':
      num_class = np.random.randint(2, self.num_classes_in_target_dataset)
    else:
      num_class = int(self.config.target.class_selection.fixed.percentage *
                      self.num_classes_in_target_dataset)

    target_classes = np.random.permutation(
        self.num_classes_in_target_dataset)[:num_class]
    target_classes.sort()
    return target_classes

  @property
  def num_classes_in_target_dataset(self):
    if self.config.target.dataset.name in datasets_info.DATASET_NUM_CLASSES:
      return datasets_info.DATASET_NUM_CLASSES[self.config.target.dataset.name]
    else:
      return None

  @property
  def num_target_classes(self):
    # This is important when multiple source datasets are used.
    if self.target_classes is None:
      return self.num_classes_in_target_dataset
    return len(self.target_classes)

  @property
  def network_architecture(self):
    if self.config.source.network_architecture == 'HRNet':
      na = networks.HRNET48_TRAINED_MODELS[self.config.source.dataset.name]
    else:
      na = networks.NETWORK_ARCHITECTURES[
          self.config.source.network_architecture]
      na.set_weights(self.config.source.dataset.name)
    return na

  @property
  def target_train_dataset(self):
    """Get Target Train Dataset."""

    if 'num_examples' in self.config.target.dataset:
      num_examples = self.config.target.dataset.num_examples
    else:
      num_examples = -1   # Use the entire dataset
    ds_train = datasets.load_dataset(
        self.config.target.dataset.name,
        num_examples,
        split='train',
        with_info=False)

    batch_size = 64
    do_train_shuffle = False
    if self.config.experiment.metric == 'accuracy':
      do_train_shuffle = True
    if 'batch_size' in self.config.experiment[self.config.experiment.metric]:
      batch_size = self.config.experiment[
          self.config.experiment.metric].batch_size

    return datasets.get_experiment_dataset(
        ds_train,
        self.config.target.dataset.name,
        self.network_architecture,
        self.target_classes,
        do_shuffle=do_train_shuffle,
        as_supervised=self.config.experiment.dataset_as_supervised,
        batch_size=batch_size)

  @property
  def target_test_dataset(self):
    ds_test = datasets.load_dataset(self.config.target.dataset.name,
                                    split='test', with_info=False)
    batch_size = 64
    if self.config.experiment.metric == 'accuracy':
      batch_size = self.config.experiment.accuracy.batch_size

    return datasets.get_experiment_dataset(
        ds_test,
        self.config.target.dataset.name,
        self.network_architecture,
        self.target_classes,
        do_shuffle=False,
        as_supervised=self.config.experiment.dataset_as_supervised,
        batch_size=batch_size)

  def source_model(self, model_type='features'):
    if model_type == 'features':
      return self.network_architecture.get_feature_model()
    if model_type == 'source_predictions':
      return self.network_architecture.get_prediction_model()
    if model_type == 'target_predictions':
      return self.network_architecture.get_target_model(
          num_classes=self.num_target_classes,
          base_trainable=self.config.experiment.accuracy.base_trainable)
    raise ValueError(f'Model type {model_type} not recognized.')

  def _model_ouput_segmentation(self, model):
    """Get model output for semantic segmentation."""

    sampling_seed = 0
    data_config = self.config.target.dataset
    if data_config.class_balanced_sampling and data_config.pixels_per_image:
      class_counts = features_targets_utils.compute_class_frequencies(
          self.target_train_dataset)
    else:
      class_counts = None

    labels = []
    outputs = []
    for image, label in tqdm.tqdm(self.target_train_dataset):
      output = model.predict(image)
      if output.shape[1:3] != label.shape[1:3]:
        # Downsample labels to match the size of the features
        # HRNet based models produce features with resolution [H/4, W/4]
        # Note: this should be done only when evaluating transferability metrics
        label = tf.image.resize(label, output.shape[1:3], method='nearest')
      if data_config.pixels_per_image:
        num_samples = data_config.pixels_per_image * image.shape[0]
      else:
        num_samples = None  # It will use all the pixels without sampling
      output, label = features_targets_utils.sample_from_image(
          output, label, sampling_seed, class_counts, num_samples)
      sampling_seed += 1  # For better sampling change seed over batches
      outputs.extend(output)
      labels.extend(label)

    return outputs, labels

  def _model_ouput_classification(self, model):
    """Get model output for classification."""
    labels = []
    outputs = []
    for image, label in tqdm.tqdm(self.target_train_dataset):
      output = model.predict_on_batch(image)
      outputs.extend(output)
      labels.extend(label)
    return outputs, labels

  def model_output_on_target_train_dataset(self, model_type):
    """Return the models embeddings or predictions, and corresponding labels."""
    model = self.source_model(model_type)
    if self.config.source.network_architecture == 'HRNet':  # Segmentation
      outputs, labels = self._model_ouput_segmentation(model)
    else:  # Classification
      outputs, labels = self._model_ouput_classification(model)

    outputs = tf.stack(outputs, axis=0)
    labels = tf.stack(labels, axis=0)
    labels = features_targets_utils.shift_target_labels(labels)
    if 'pca_reduction' in self.config.experiment[self.config.experiment.metric]:
      pca_components = self.config.experiment[
          self.config.experiment.metric].pca_reduction
      assert pca_components > 0
      outputs = features_targets_utils.pca_reduction(outputs, pca_components)

    return outputs, labels

  @property
  def source_name(self):
    return f'{self.config.source.network_architecture}_{self.config.source.dataset.name}'

  @property
  def target_name(self):
    """Provide unique name for each class selection setting."""
    if self.config.source.network_architecture == 'HRNet':
      num_examples = self.config.target.dataset.num_examples
      cbs = self.config.target.dataset.class_balanced_sampling
      sampling_desc = 'cb_sampling' if cbs else 'rnd_sampling'
      pixels_per_image = self.config.target.dataset.pixels_per_image
      if pixels_per_image:
        sampling_desc += f'_{pixels_per_image}_pixels'
      else:  # All pixels, no sampling
        sampling_desc = 'no_sampling'
      class_selection_name = f'{sampling_desc}_on_{num_examples}_imgs'
    else:
      class_selection_name = self.config.target.class_selection.method
      if self.config.target.class_selection.method == 'fixed':
        percentage = self.config.target.class_selection.fixed.percentage
        class_selection_name += f'_{int(percentage*100):02d}P'
      if self.config.target.class_selection.method in ['fixed', 'random']:
        class_selection_name += (
            f'_{self.config.target.class_selection.seed:04d}S'
            f'_{self.config.target.class_selection.experiment_number:02d}E')
    return f'{self.config.target.dataset.name}/{class_selection_name}'

  @property
  def metric_name(self):
    """Provide unique name for the current metric / experiment."""
    metric = self.config.experiment.metric
    if metric == 'accuracy':
      optim = self.config.experiment.accuracy.optimizer
      optim_args = self.config.experiment.accuracy[optim]
      optim_name = f'{optim.upper()}'
      if self.config.experiment.accuracy.base_trainable:
        base_model = 'ft'
        epochs = self.config.experiment.accuracy.base_trainable_epochs
      else:
        base_model = 'only_head'
        epochs = self.config.experiment.accuracy.base_frozen_epochs
      optim_name += f'_E{epochs:02d}'
      optim_name += f'_B{self.config.experiment.accuracy.batch_size:03d}'
      optim_name += f"_LR{optim_args['learning_rate']:5.0e}"
      if optim_args == 'sgd':
        sgd_name = ''
        if not optim_args['nesterov']:
          sgd_name += '_NoNesterov'
        if optim_args['momentum'] != 0.9:
          sgd_name += f"_M{optim_args['momentum']}".replace('.', '_')
        optim_name += sgd_name
      return f'accuracy_{base_model}_{optim_name}'
    options = ''
    if 'pca_reduction' in self.config.experiment[metric]:
      pca_reduction = self.config.experiment[metric].pca_reduction
      options += f'_PCA{pca_reduction!s}'.replace('.', '_')  # Can be 0.*
    if 'gaussian_type' in self.config.experiment[metric]:
      options += f'_{self.config.experiment[metric].gaussian_type}'
    return metric + options

  @property
  def experiment_name(self):
    return f'{self.target_name}/{self.source_name}/{self.metric_name}'

  @property
  def results_file(self):
    return f'{self.config.results.basedir}/{self.experiment_name}.{self.config.results.extension}'


def load_or_compute(func):
  """Load the values from disk, or compute these using the function."""

  def wrapper(
      experiment,
      **kwargs):
    results_file = experiment.results_file
    if tf.io.gfile.exists(results_file):
      return utils.load_dict(results_file)
    elif experiment.config.only_load:
      return None

    results_dict = func(experiment, **kwargs)
    utils.save_dict(results_dict, results_file)
    return results_dict

  return wrapper
