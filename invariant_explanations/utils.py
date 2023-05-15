# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Utilities used for approxNN project."""
import time
import gc
import itertools
import os
import pickle
import sys

from absl import flags
from absl import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.stats
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_kernels
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
import tensorflow_datasets as tfds
from tqdm import tqdm

import config
import explanation_utils
import other
import plotting

logging.set_verbosity(logging.INFO)

from debug import ipsh

FLAGS = flags.FLAGS


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


def create_experimental_folders():
  """Create timestamped experimental folder and subfolders.

  /../_experiments/
  |--    <timestamp>_<setup_name>
  |         `-- _models/
  |         `-- _plots/
  """
  if not gfile.exists(config.cfg.EXP_DIR_PATH):
    gfile.makedirs(config.cfg.EXP_DIR_PATH)
  if not gfile.exists(config.cfg.PLOTS_DIR_PATH):
    gfile.makedirs(config.cfg.PLOTS_DIR_PATH)
  if not gfile.exists(config.cfg.MODELS_DIR_PATH):
    gfile.makedirs(config.cfg.MODELS_DIR_PATH)


def file_handler(dir_path, file_name, stream_flags):
  """Central method to handle saving and loading a file by filename.

  Args:
    dir_path: the directory path in/from which the file is to be saved/loaded.
    file_name: the name of the file that is being saved/loaded.
    stream_flags: flag to whether to read or write {wb, rb}.

  Returns:
    a function handle used to stream the file bits.
  """
  file_opener = gfile.GFile
  return file_opener(os.path.join(dir_path, file_name), stream_flags)


def load_processed_data_files(data_file_dict, explanation_type, chkpt=86):
  """Load the corresponding data files from the appropriate folder.

  Args:
    data_file_dict: a dictionary of {'data file': Boolean}
                    pairs specifying the data file to load.
    explanation_type: name of explanation method for which to load data.
    chkpt: the checkpoint at which to load data files; default=86.
  """

  start_time = time.time()
  if config.cfg.RUN_ON_PRECOMPUTED_GCP_DATA and not data_file_dict['metrics']:
    raise ValueError('Metrics must always be loaded so to filter min accuracy.')

  for key in data_file_dict.keys():
    if key not in config.ALLOWABLE_DATA_FILES:
      raise ValueError('Unrecognized data file of type `%s`.' % key)

  # Determine read path from which to load files.
  f_suffix = get_file_suffix(chkpt, explanation_type)
  if config.cfg.RUN_ON_PRECOMPUTED_GCP_DATA:
    read_path = (
        f'{config.MERGED_DATA_PATH}/{config.cfg.MEDIATION_TYPE}/{config.cfg.DATASET}'
    )
  else:
    read_path = config.cfg.EXP_DIR_PATH

  # Determine the data files to be loaded.
  keys_for_which_to_load_data = [
      key for key in data_file_dict.keys() if data_file_dict[key] is not False
  ]
  all_loaded_data_files = dict.fromkeys(keys_for_which_to_load_data, None)

  # For each data file marked above, load the file, check it is of correct
  # instance type, and mark is as the value of the appropriate key in the
  # output dictionary.
  for key in all_loaded_data_files.keys():

    with file_handler(read_path, f'{key}{f_suffix}', 'rb') as f:
      loaded_data_file = pickle.load(f)

    if key in {'hparams', 'metrics'}:
      assert isinstance(loaded_data_file, pd.core.frame.DataFrame)
    else:
      assert isinstance(loaded_data_file, np.ndarray)

    all_loaded_data_files[key] = loaded_data_file

  # Confirm that all loaded data files share the same length (.shape[0]).
  all_data_file_lens = np.array([
      data_file.shape[0] for data_file in all_loaded_data_files.values()
  ])
  assert np.all(all_data_file_lens == all_data_file_lens[0]), 'One or more loaded file sizes not consistent.'

  # # When loading merged batch data from GCP (which was run using min acc > 0),
  # # make sure to filter to the right stratum here before further analysis.
  # if config.cfg.RUN_ON_PRECOMPUTED_GCP_DATA:
  #   filtered_indices = np.where(
  #       (
  #           all_loaded_data_files['metrics']['test_accuracy'] >
  #           config.cfg.MIN_BASE_MODEL_ACCURACY
  #       ) &
  #       (
  #           all_loaded_data_files['metrics']['test_accuracy'] <
  #           config.cfg.MAX_BASE_MODEL_ACCURACY
  #       )
  #   )[0]
  #   for key, value in all_loaded_data_files.items():
  #     loaded_data_file = value
  #     if isinstance(loaded_data_file, np.ndarray):
  #       all_loaded_data_files[key] = loaded_data_file[filtered_indices]
  #     elif isinstance(loaded_data_file, pd.core.frame.DataFrame):
  #       all_loaded_data_files[key] = loaded_data_file.iloc[filtered_indices]

  type_of_data = 'preprocessed' if config.cfg.RUN_ON_PRECOMPUTED_GCP_DATA else 'generated'
  logging.info(
      '%sElapsed time (loading `%s` data): `%f`.%s',
      Bcolors.HEADER,
      type_of_data,
      time.time() - start_time,
      Bcolors.ENDC,
  )
  return all_loaded_data_files


# Inspired by: https://stackoverflow.com/a/287944
class Bcolors:
  HEADER = '\033[95m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'


def get_file_suffix(chkpt, explanation_type):
  """Defining a consistent file suffix for saving temporary files."""
  return f'_@_epoch_{chkpt}_{explanation_type}.npy'


def print_memory_usage():
  """A debugging tool for clearning unused vairables."""
  mem_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
  logging.debug(
      '\t%smem_usage: %.4f MB%s', Bcolors.HEADER, mem_usage_mb, Bcolors.ENDC
  )


def reset_model_using_weights(model_wireframe, weights):
  """A tool to load flattened weights into model wireframe."""
  all_boundaries = {
      0: [(0, 16), (16, 160)],
      1: [(160, 176), (176, 2480)],
      2: [(2480, 2496), (2496, 4800)],
      3: [],  # GlobalAvgPool has no params
      4: [(4800, 4810), (4810, 4970)],  # FC
  }
  for layer_idx, layer_obj in enumerate(model_wireframe.layers):
    if not layer_obj.get_weights(): continue  # Skip GlobalAvgPool.
    layer_bias_start_idx = all_boundaries[layer_idx][0][0]
    layer_bias_stop_idx = all_boundaries[layer_idx][0][1]
    layer_weights_start_idx = all_boundaries[layer_idx][1][0]
    layer_weights_stop_idx = all_boundaries[layer_idx][1][1]
    layer_bias = np.reshape(
        weights[layer_bias_start_idx:layer_bias_stop_idx],
        layer_obj.get_weights()[1].shape,
    )  # b
    layer_weights = np.reshape(
        weights[layer_weights_start_idx:layer_weights_stop_idx],
        layer_obj.get_weights()[0].shape,
    )
    layer_obj.set_weights([layer_weights, layer_bias])
  return model_wireframe


def rounder(values, markers, use_log_rounding=False):
  """Round values in a list to the closest marker from a list markers.

  Args:
    values: list of values to be rounded.
    markers: list of markers to which values are rounded.
    use_log_rounding: boolean flag where or not to apply log-rounding.

  Returns:
    List of values, rounded to the nearest values as set out in config.py.
  """
  # Inspired by: https://stackoverflow.com/a/2566508
  if use_log_rounding:
    values = np.log10(values)
    markers = np.log10(markers)
  def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
  tmp = np.array([find_nearest(markers, val) for val in values])
  if use_log_rounding:
    return 10 ** tmp
  return tmp


def process_hparams(hparams, round_num, cat_to_code):
  """Convert columns of the hparams dataframe to appropriate datatypes.

  Args:
    hparams: a dataframe of hyperparameters.
    round_num: flag to round numerical values.
    cat_to_code: flag to convert categorical features to numeric codes.

  Returns:
    A numpy array or dataframe of hparams whereby values in each column are
    processed according to the input arguemnts, e.g., (log-)rounded to the
    nearest market values as set out in config.py.
  """

  start_time = time.time()
  assert isinstance(hparams, pd.core.frame.DataFrame)

  # Convert numerical columns to numerical values (s/dtype=Object/dtype=float32)
  for col in config.NUM_HPARAMS:
    hparams[col] = hparams[col].astype('float32')

  if cat_to_code:
    # Convert categorical columns to numerical values for training meta_model.
    for hparam in config.CAT_HPARAMS:
      hparams[hparam] = pd.Categorical(hparams[hparam]).codes
  else:
    for col in config.CAT_HPARAMS:
      logging.info(('Unique values for `%s`:\t', col), hparams[col].unique())

  if round_num:
    # Because num_hparams are sampled (log-)uniformly from some range, we first
    # bin the values by rounding to carefully selected hparam values, and then
    # use the (few) selected bins are markers for drawing out ITE plots.
    for col in config.NUM_HPARAMS:
      values = hparams[col].values
      markers = config.ALL_HPARAM_RANGES[col]
      use_log_rounding = False
      if col in {'config.l2reg', 'config.init_std', 'config.learning_rate'}:
        use_log_rounding = True

      rounded_values = rounder(values, markers, use_log_rounding)
      hparams[col] = rounded_values.astype('float32')
  else:
    hparams = hparams.to_numpy().astype('float32')

  logging.info(
      '%sElapsed time (processing hparams): `%f`.%s',
      Bcolors.HEADER,
      time.time() - start_time,
      Bcolors.ENDC,
  )
  return hparams


def load_base_model_weights_and_metrics():
  """Load base weights and metrics from CNN collections."""

  logging.info('Loading CNN Zoo weights and metrics...')
  with file_handler(config.cfg.DATA_DIR_PATH, 'weights.npy', 'rb') as f:
    # A numpy array of weights of the trained models.
    base_model_weights = np.load(f)

  with file_handler(config.cfg.DATA_DIR_PATH, 'metrics.csv', 'r') as f:
    # A pandas DataFrame with metrics of the trained models.
    base_model_metrics = pd.read_csv(f, sep=',')

  assert base_model_weights.shape[0] == base_model_metrics.values.shape[0]
  logging.info('Done.')
  return base_model_weights, base_model_metrics


def analyze_accuracies_of_base_models():
  """Plot & compare train/test accuracies of base models in CNN collection."""

  logging.info('Analyzing base model accuracies...')

  _, base_model_metrics = load_base_model_weights_and_metrics()

  accuracy_tracker = pd.DataFrame({
      'chkpt': [],
      'accuracy': [],
      'accuracy_type': [],
  })

  for chkpt in [0, 1, 2, 3, 20, 40, 60, 80, 86]:

    indices = base_model_metrics.index[
        base_model_metrics['step'] == chkpt
    ].tolist()

    for accuracy_type in ['train', 'test']:
      chkpt_list = [chkpt] * len(indices)
      accuracy_type_list = [accuracy_type] * len(indices)
      accuracy_list = base_model_metrics.iloc[
          indices
      ][f'{accuracy_type}_accuracy'].to_numpy()

      # Add to accuracy_tracker.
      accuracy_tracker = accuracy_tracker.append(
          pd.DataFrame({
              'chkpt': chkpt_list,
              'accuracy': accuracy_list,
              'accuracy_type': accuracy_type_list,
          }),
          ignore_index=True,
      )

  with file_handler(config.cfg.EXP_DIR_PATH, 'accuracy_tracker.npy', 'wb') as f:
    pickle.dump(accuracy_tracker, f, protocol=4)

  catplot = sns.catplot(
      x='chkpt',
      y='accuracy',
      hue='accuracy_type',
      data=accuracy_tracker,
      kind='violin',
  )
  fig = catplot.fig
  fig.set_size_inches(18, 6)
  fig.savefig(
      gfile.GFile(
          os.path.join(
              config.cfg.PLOTS_DIR_PATH,
              'base_model_accuracies.png',
          ),
          'wb',
      ),
      dpi=400,
  )


def extract_new_covariates_and_targets(random_seed, model, dataset_info,
                                       covariates_setting, base_model_weights,
                                       base_model_metrics):
  """Extract new dataset from the weights and metrics of the CNN collection.

  The new dataset is used to train a meta-model, on covariates and targets
  corresponding to X,H-->Y and X,W@epoch-->Y.

  Args:
    random_seed: the random seed used for reproducibility of results.
    model: the wireframe of the CNN model in the zoo.
    dataset_info: a tfds dataset info with dimnesionality and number of samples.
    covariates_setting: a dictionary specifying the checkpoint at which to
                        extract data from the saved weights/metrics in the zoo.
    base_model_weights: the weights of the base models in the CNN zoo.
    base_model_metrics: the metrics of the base models in the CNN zoo.

  Returns:
    samples: samples used to train each base model; instance of np.ndarray.
    y_preds: the predicted target of samples on each base-model;
             instance of np.ndarray.
    y_trues: the true target of samples; instance of np.ndarray.
    explans: the model explanations for each sample; instance of np.ndarray.
    hparams: the hparams used to train the base model; instance of pd.DataFrame.
    weights_chkpt: flattened weights of each base model at chkpt epoch;
                   instance of np.ndarray.
    weights_final: flattened weights of each base model at final epoch;
                   instance of np.ndarray
    metrics: the metrics (train and test accuracy, etc.) of each base model;
             instance of pd.DataFrame
  """

  random_state = np.random.RandomState(random_seed)

  assert base_model_weights.shape[0] == base_model_metrics.shape[0]
  if not config.cfg.RUN_ON_TEST_DATA:
    assert (
        base_model_weights.shape[0] ==
        config.NUM_MODELS_PER_DATASET[config.cfg.DATASET]
    )

  logging.info('\tConstructing new dataset...')

  ############################################################################
  # Weights contains 270,000 rows (30K hparam settings @ 9 checkpoints).
  # Filter to relevant rows; then do a 50/50 tr/te split (see page 5, col 1)
  # from this paper: https://arxiv.org/pdf/2002.11448.pdf .
  ############################################################################

  # Filter to the appropriate rows for the weights at checkpoint `chkpt`.
  weights_chkpt_indices = base_model_metrics.index[
      base_model_metrics['step'] == covariates_setting['chkpt']
  ].tolist()
  # Also keep track of the rows for the weights at the final checkpoint, 86.
  weights_final_indices = base_model_metrics.index[
      base_model_metrics['step'] == 86
  ].tolist()
  # Rows in metrics file repeat for 9 rows (to match 9 checkpoint epoch).
  # Therefore, sample every other 9th row; IMPORTANT: any sequence would yield
  # the same hparams, but make sure you get the correct rows for tr/te acc!!!
  metrics_indices = weights_final_indices  # NOT weights_chkpt_indices, b/c we
                                           # filter (to keep) based on the final
                                           # performance

  # It's more reasonable to predict the targets for the final epoch, NOT at
  # chkpt epoch. Right? Yes. Because we aim for the meta-model to take in
  # X,H or X,W@epoch and give Y_pred at end of training, w/o needing to train.

  assert (
      len(metrics_indices) ==
      len(weights_chkpt_indices) ==
      len(weights_final_indices)
  )

  # IMPORTANT: indices is used for metrics, weights_chkpt, and weights_final;
  #            shuffling the order should be done consistently on all 3 arrays
  permuted_indices = random_state.permutation(range(len(metrics_indices)))
  metrics_indices = np.array(metrics_indices)[permuted_indices]
  weights_chkpt_indices = np.array(weights_chkpt_indices)[permuted_indices]
  weights_final_indices = np.array(weights_final_indices)[permuted_indices]

  base_model_metrics = base_model_metrics.iloc[metrics_indices]
  base_model_weights_chkpt = base_model_weights[weights_chkpt_indices, :]
  base_model_weights_final = base_model_weights[weights_final_indices, :]

  # Further filter the weights to those that yield good accuracy
  # and limit selection to only NUM_BASE_MODELS models.
  filtered_indices = np.where(
      (
          base_model_metrics['test_accuracy'] >
          config.cfg.MIN_BASE_MODEL_ACCURACY
      ) &
      (
          base_model_metrics['test_accuracy'] <
          config.cfg.MAX_BASE_MODEL_ACCURACY
      )
  )[0][:config.cfg.NUM_BASE_MODELS]
  # For faster processing, select a batch of models to process.
  filtered_indices = np.array_split(
      filtered_indices,
      config.cfg.MODEL_BATCH_COUNT,
  )[config.cfg.MODEL_BATCH_IDX]
  if not list(filtered_indices):
    raise ValueError('No base-models identifed after filtering.')
  base_model_weights_chkpt = base_model_weights_chkpt[filtered_indices]
  base_model_weights_final = base_model_weights_final[filtered_indices]
  base_model_metrics = base_model_metrics.iloc[filtered_indices]
  local_num_base_models = base_model_metrics.shape[0]  # update this value

  ############################################################################
  # Construct and fill arrays of appropriate size for covariates and targets.
  ############################################################################

  size_x = dataset_info['data_shape'][0] * dataset_info['data_shape'][1]
  size_y = dataset_info['num_classes']

  num_new_samples = (
      config.cfg.NUM_SAMPLES_PER_BASE_MODEL * local_num_base_models
  )
  samples = np.zeros((num_new_samples, size_x))
  y_preds = np.zeros((num_new_samples, size_y))
  y_trues = np.zeros((num_new_samples, size_y))
  if config.cfg.EXPLANATION_TYPE == 'identity':
    explans = np.zeros((num_new_samples, size_y))  # E.shape = Y.shape
  else:
    explans = np.zeros((num_new_samples, size_x))  # E.shape = X.shape

  tmp = (
      f'[INFO] For each base model, construct network, load base_model_weights,'
      f'then get preds for {config.cfg.NUM_SAMPLES_PER_BASE_MODEL} samples.'
  )
  if config.cfg.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS:
    tmp += '(on the same samples).'
  else:
    tmp += '(on different samples... to cover a wider distribution).'
  logging.info(tmp)

  # BUG IN COMMENTED CODE BELOW, ORDERING IS DIFFERENT FOR SAMPLES/TRUE LABELS
  # all_img_samples = np.concatenate([x for x, y in data_tr], axis=0)
  # all_img_y_trues = np.concatenate([y for x, y in data_tr], axis=0)
  all_img_samples, all_img_y_trues = tfds.as_numpy(
      tfds.load(
          config.cfg.DATASET,
          split='train',
          batch_size=-1,
          as_supervised=True,
      )
  )
  all_img_y_trues = tf.keras.utils.to_categorical(
      all_img_y_trues,
      num_classes=dataset_info['num_classes'],
  )

  # IMPORTANT: w/o the processing below, the meta-model learning drops by >%50.
  all_img_samples = all_img_samples.astype(np.float32)
  all_img_samples /= 255
  min_out = -1.0
  max_out = 1.0
  all_img_samples = min_out + all_img_samples * (max_out - min_out)
  all_img_samples = tf.math.reduce_mean(all_img_samples, axis=-1, keepdims=True)
  all_img_samples = all_img_samples.numpy()

  num_train_samples = all_img_samples.shape[0]

  if config.cfg.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS:
    # Select identical samples in such a way that they are class-balanced.
    num_classes = other.get_dataset_info(config.cfg.DATASET)['num_classes']
    num_samples_per_class = [len(x) for x in np.array_split(
        np.arange(config.cfg.NUM_SAMPLES_PER_BASE_MODEL),
        num_classes,
    )]
    all_rand_indices = np.array([], dtype='int64')  # int64 b/c it stores idx
    for class_idx in range(num_classes):
      class_specific_indices = np.argwhere(
          np.argmax(all_img_y_trues, axis=1) == class_idx
      ).flatten()
      rand_indices = random_state.choice(
          class_specific_indices,
          size=num_samples_per_class[class_idx],
          replace=False,
      )
      all_rand_indices = np.hstack((all_rand_indices, rand_indices))
    batch_img_samples = all_img_samples[all_rand_indices, :, :, :]
    batch_img_y_trues = all_img_y_trues[all_rand_indices, :]

  for idx in tqdm(range(local_num_base_models)):

    model = reset_model_using_weights(model, base_model_weights_final[idx, :])

    if not config.cfg.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS:
      rand_indices = random_state.choice(
          num_train_samples,
          size=config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
          replace=False,
      )
      batch_img_samples = all_img_samples[rand_indices, :, :, :]
      batch_img_y_trues = all_img_y_trues[rand_indices, :]

    # We use `predict_on_batch` instead of `predict` to avoid a memory leak; see
    # github.com/keras-team/keras/issues/13118#issuecomment-541688220
    predictions = model.predict_on_batch(batch_img_samples)
    # explanation_utils.plot_and_save_various_explanations(
    #     model,
    #     batch_img_samples,
    #     predictions,
    #     f'explanations_for_model_{idx}.png'
    # )

    # TODO(amirhkarimi): this shuffling should either happen here, or in
    #                    explanation_utils(), depending on def'n of unmediation.
    # if config.cfg.MEDIATION_TYPE == 'unmediated':
    #   y_preds = shuffle_logits(y_preds)

    explanations = explanation_utils.get_model_explanations_for_instances(
        model,
        batch_img_samples,
        predictions,
        config.cfg.EXPLANATION_TYPE,
    )

    # TODO(amirhkarimi): this shuffling should either happen here, or in
    #                    explanation_utils(), depending on def'n of unmediation.
    # if config.cfg.MEDIATION_TYPE == 'unmediated':
    #   y_preds = shuffle_logits(y_preds)

    new_instances_range = range(
        idx * config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
        (idx + 1) * config.cfg.NUM_SAMPLES_PER_BASE_MODEL
    )

    # TODO(amirhkarimi): bring this commented block back; commented out becuase
    #                    it was very expensive to compute locally -> sigkill 9.
    # samples[new_instances_range, :] = np.reshape(
    #     batch_img_samples,
    #     (batch_img_samples.shape[0], -1),
    # )  # collapse all image dims
    y_preds[new_instances_range, :] = predictions
    y_trues[new_instances_range, :] = batch_img_y_trues
    explans[new_instances_range, :] = np.reshape(
        explanations,
        (explanations.shape[0], -1),
    )  # collapse all image dims

  # weights_chkpt, weights_final, hparams, and metrics are global properties of
  # a model shared for all instances predicted on each model; apply np.repeat
  # outside of loop for efficiency.
  # TODO(amirhkarimi): bring this commented block back; commented out becuase
  #                    it was very expensive to compute locally -> sigkill 9.
  # weights_chkpt = np.repeat(
  #     base_model_weights_chkpt,
  #     config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
  #     axis=0,
  # )
  # weights_final = np.repeat(
  #     base_model_weights_final,
  #     config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
  #     axis=0,
  # )
  weights_chkpt = np.zeros(samples.shape)
  weights_final = np.zeros(samples.shape)


  hparams = pd.DataFrame(
      np.repeat(
          base_model_metrics[config.ALL_HPARAMS].values,
          config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
          axis=0,
      )
  )
  hparams.columns = base_model_metrics[config.ALL_HPARAMS].columns

  metrics = pd.DataFrame(
      np.repeat(
          base_model_metrics[config.ALL_METRICS].values,
          config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
          axis=0,
      )
  )
  metrics.columns = base_model_metrics[config.ALL_METRICS].columns

  logging.info('Done.')

  assert isinstance(samples, np.ndarray)
  assert isinstance(y_preds, np.ndarray)
  assert isinstance(y_trues, np.ndarray)
  assert isinstance(explans, np.ndarray)
  assert isinstance(hparams, pd.core.frame.DataFrame)
  assert isinstance(weights_chkpt, np.ndarray)
  assert isinstance(weights_final, np.ndarray)
  assert isinstance(metrics, pd.core.frame.DataFrame)
  assert (
      samples.shape[0] ==
      y_preds.shape[0] ==
      y_trues.shape[0] ==
      explans.shape[0] ==
      hparams.shape[0] ==
      weights_chkpt.shape[0] ==
      weights_final.shape[0] ==
      metrics.shape[0]
  )
  return (
      samples, y_preds, y_trues, explans, hparams,
      weights_chkpt, weights_final, metrics,
  )


def process_and_resave_cnn_zoo_data(random_seed, model_wireframe,
                                    covariates_settings):
  """Load weights and matrices from CNN zoo dataset to process for new training.

  Upon loading the data from the CNN zoo, this method feeds the corresponding
  weights and matrices for each epoch (designated in config.py) into the
  `extract_new_covariates_and_targets` method to process the data according for
  the training of the meta-model. The resulting covaritates, targets, and meta
  information is then saved (to be loaded later for meta-model training).

  Args:
    random_seed: the random seed used for reproducibility of results.
    model_wireframe: the tf model graph whose weights are then populated from
                     the save weights in the CNN zoo.
    covariates_settings: a dictionary specifying the checkpoints at which to
                         extract data from the saved weights/metrics in the zoo.
  """

  if config.cfg.RUN_ON_PRECOMPUTED_GCP_DATA:
    logging.info(
        'Data files already processed on GCP and merged locally; '
        'do not reprocess.'
    )
    return

  base_model_weights, base_model_metrics = load_base_model_weights_and_metrics()

  for explanation_type in config.ALLOWABLE_EXPLANATION_METHODS:
    config.cfg.EXPLANATION_TYPE = explanation_type

    for covariates_setting in covariates_settings:

      chkpt = int(covariates_setting['chkpt'])
      f_suffix = get_file_suffix(chkpt, explanation_type)
      logging.info(
          'Extracting new covariates and targets for '
          'chkpt %s @ test acc in [%.2f, %.2f]',
          covariates_setting['chkpt'],
          config.cfg.MIN_BASE_MODEL_ACCURACY,
          config.cfg.MAX_BASE_MODEL_ACCURACY,
      )

      samples, y_preds, y_trues, explans, hparams, w_chkpt, w_final, metrics = extract_new_covariates_and_targets(
          random_seed,
          model_wireframe,
          other.get_dataset_info(config.cfg.DATASET),
          covariates_setting,
          base_model_weights,
          base_model_metrics,
      )

      # with file_handler(config.cfg.EXP_DIR_PATH, f'samples{f_suffix}', 'wb') as f:
      #   pickle.dump(samples, f, protocol=4)
      with file_handler(config.cfg.EXP_DIR_PATH, f'y_preds{f_suffix}', 'wb') as f:
        pickle.dump(y_preds, f, protocol=4)
      with file_handler(config.cfg.EXP_DIR_PATH, f'y_trues{f_suffix}', 'wb') as f:
        pickle.dump(y_trues, f, protocol=4)
      with file_handler(config.cfg.EXP_DIR_PATH, f'explans{f_suffix}', 'wb') as f:
        pickle.dump(explans, f, protocol=4)
      with file_handler(config.cfg.EXP_DIR_PATH, f'hparams{f_suffix}', 'wb') as f:
        pickle.dump(hparams, f, protocol=4)
      # with file_handler(config.cfg.EXP_DIR_PATH, f'w_chkpt{f_suffix}', 'wb') as f:
      #   pickle.dump(w_chkpt, f, protocol=4)
      # with file_handler(config.cfg.EXP_DIR_PATH, f'w_final{f_suffix}', 'wb') as f:
      #   pickle.dump(w_final, f, protocol=4)
      with file_handler(config.cfg.EXP_DIR_PATH, f'metrics{f_suffix}', 'wb') as f:
        pickle.dump(metrics, f, protocol=4)

      del samples, y_preds, y_trues, explans, hparams, w_chkpt, w_final, metrics
      logging.info('\tdone.')


def process_per_class_explanations(random_seed):
  """Process saved explanations in lower dimensions using SPCA.

  Args:
    random_seed: the random seed used for reproducibility of results.
  """

  np.random.RandomState(random_seed)

  all_loaded_data_files = load_processed_data_files({
      'samples': True,
      'y_preds': True,
      'y_trues': False,
      'w_chkpt': False,
      'w_final': False,
      'explans': True,
      'hparams': True,
      'metrics': True,
  }, explanation_type=config.cfg.EXPLANATION_TYPE)
  samples = all_loaded_data_files['samples']
  y_preds = all_loaded_data_files['y_preds']
  hparams = all_loaded_data_files['hparams']

  # Reorder columns for easier readability when debugging.
  hparams = hparams[[*config.CAT_HPARAMS, *config.NUM_HPARAMS]]
  hparams = process_hparams(hparams, round_num=False, cat_to_code=True)

  _, samples_proj, samples_targets = project_using_spca(
      samples=samples,
      targets=y_preds,
      n_components=2,
  )

  _, y_preds_proj, y_preds_targets = project_using_spca(
      samples=y_preds,
      targets=y_preds,
      n_components=2,
  )

  _, explans_proj, explans_targets = project_using_spca(
      samples=explans,
      targets=y_preds,
      n_components=2,
  )

  _, hparams_proj, hparams_targets = project_using_spca(
      samples=hparams,
      targets=y_preds,
      n_components=2,
  )

  # Show projections in 2D.
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
  num_classes = other.get_dataset_info(config.cfg.DATASET)['num_classes']

  for class_idx in range(num_classes):

    ax1.scatter(
        samples_proj[np.argmax(samples_targets, axis=1) == class_idx, 0],
        samples_proj[np.argmax(samples_targets, axis=1) == class_idx, 1],
        label=class_idx, alpha=0.4,
    )

    ax2.scatter(
        y_preds_proj[np.argmax(y_preds_targets, axis=1) == class_idx, 0],
        y_preds_proj[np.argmax(y_preds_targets, axis=1) == class_idx, 1],
        label=class_idx, alpha=0.4,
    )

    ax3.scatter(
        explans_proj[np.argmax(explans_targets, axis=1) == class_idx, 0],
        explans_proj[np.argmax(explans_targets, axis=1) == class_idx, 1],
        label=class_idx, alpha=0.4,
    )

    ax4.scatter(
        hparams_proj[np.argmax(hparams_targets, axis=1) == class_idx, 0],
        hparams_proj[np.argmax(hparams_targets, axis=1) == class_idx, 1],
        label=class_idx, alpha=0.4,
    )

  ax1.set_title('Projected samples (784-D --> 2-D)')
  ax2.set_title('Projected y_preds (10-D --> 2-D)')
  ax3.set_title('Projected explans (784-D --> 2-D)')
  ax4.set_title('Projected hparams (9-D --> 2-D)')
  ax1.grid(True)
  ax2.grid(True)
  ax3.grid(True)
  ax4.grid(True)

  # Share handle between subplots in same fig.
  handles, labels = ax4.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left', title='Class')
  fig.tight_layout()

  # Save to file.
  fig.savefig(
      gfile.GFile(
          os.path.join(
              config.cfg.PLOTS_DIR_PATH,
              'spca_projections.png',
          ),
          'wb',
      ),
      dpi=400,
  )


def measure_prediction_explanation_variance(random_seed):
  for explanation_type in config.ALLOWABLE_EXPLANATION_METHODS:
    config.cfg.EXPLANATION_TYPE = explanation_type

    all_loaded_data_files = load_processed_data_files({
        'samples': False,
        'y_preds': True,
        'y_trues': True,
        'w_chkpt': False,
        'w_final': False,
        'explans': True,
        'hparams': True,
        'metrics': True,
    }, explanation_type=config.cfg.EXPLANATION_TYPE)

    filtered_loaded_data_files = dict.fromkeys(
        all_loaded_data_files.keys(), None)

    accuracy_ranges = config.RANGE_ACCURACIES[config.cfg.DATASET]
    for (min_accuracy, max_accuracy) in accuracy_ranges:
      # Overwrite values of config.cfg which were set by FLAGS.
      config.cfg.MIN_BASE_MODEL_ACCURACY = min_accuracy
      config.cfg.MAX_BASE_MODEL_ACCURACY = max_accuracy

      filtered_indices = np.where(
          (
              all_loaded_data_files['metrics']['test_accuracy'] >
              config.cfg.MIN_BASE_MODEL_ACCURACY
          ) &
          (
              all_loaded_data_files['metrics']['test_accuracy'] <
              config.cfg.MAX_BASE_MODEL_ACCURACY
          )
      )[0]
      for key, value in all_loaded_data_files.items():
        loaded_data_file = value
        if isinstance(loaded_data_file, np.ndarray):
          filtered_loaded_data_files[key] = loaded_data_file[filtered_indices]
        elif isinstance(loaded_data_file, pd.core.frame.DataFrame):
          filtered_loaded_data_files[key] = loaded_data_file.iloc[
              filtered_indices]

      for treatment_kernel in config.ALLOWABLE_TREATMENT_KERNELS:
        config.cfg.TREATMENT_KERNEL = treatment_kernel
        if (  # Should only use non-RBF kernels when not bucketing.
              # == if bucketing, should only use the RBF kernel.
            not (
                np.isclose(float(config.cfg.MIN_BASE_MODEL_ACCURACY), 0) and
                np.isclose(float(config.cfg.MAX_BASE_MODEL_ACCURACY), 1)
            ) and config.cfg.TREATMENT_KERNEL != 'rbf'
        ):
          continue

        _measure_prediction_explanation_variance(
            random_seed, filtered_loaded_data_files)

    del all_loaded_data_files
    gc.collect()


def _measure_prediction_explanation_variance(
    random_seed, filtered_loaded_data_files):
  """Measure and compare the change in predictions with that of explanations.

  We aim to understand the relative effect that changing hparams has on both the
  y_preds and the explans for a trained model. Therefore, we compute pairwise
  similarity between y_preds and explans under certain values of the hparams,
  and then plot and compare the similarities in a scatter plot. Ultimately, we
  would like to study the ITE := y_h_treatment(x) - y_h_control(x), but because
  there does not exist a canonical definition of treatment or control, we resort
  to comparing all pairwise relations under some similarity function.

  For every instance x...
  |--  For every hparam type...
       |--  For every pair (h1, h2) of unique hparam values of this type...
            |--  Compute and plot the average dissimilarity between the y_preds
            |--  (explans) resulting from models trained under h1 in contrast to
            |--  models trained under h2. Finally, scatter the dissimilarity in
            |--  y_preds against that of explans. Essentially, this plots
            |--  d(y_{h_{ij}}(x), y_{h_{ik}}(x)) vs.
            |--  d(e_{h_{ij}}(x), e_{h_{ik}}(x))
            |--  forall j != k in |unique(h_i)|
            |--  forall i in |hparams|

  Args:
    random_seed: the random seed used for reproducibility of results.
    filtered_loaded_data_files: filtered based on accuracy bucket.
  """

  if not config.cfg.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS:
    raise ValueError('Expected use of identical samples for base models.')

  random_state = np.random.RandomState(random_seed)

  # all_loaded_data_files = load_processed_data_files({
  #     'samples': False,
  #     'y_preds': True,
  #     'y_trues': True,
  #     'w_chkpt': False,
  #     'w_final': False,
  #     'explans': True,
  #     'hparams': True,
  #     'metrics': True,
  # }, explanation_type=config.cfg.EXPLANATION_TYPE)
  # # samples = all_loaded_data_files['samples']
  # y_preds = all_loaded_data_files['y_preds']
  # y_trues = all_loaded_data_files['y_trues']
  # explans = all_loaded_data_files['explans']
  # hparams = all_loaded_data_files['hparams']
  # metrics = all_loaded_data_files['metrics']
  y_preds = filtered_loaded_data_files['y_preds']
  y_trues = filtered_loaded_data_files['y_trues']
  explans = filtered_loaded_data_files['explans']
  hparams = filtered_loaded_data_files['hparams']
  metrics = filtered_loaded_data_files['metrics']

  # Reorder columns for easier readability when debugging.
  hparams = hparams[[*config.CAT_HPARAMS, *config.NUM_HPARAMS]]
  hparams = process_hparams(hparams, round_num=True, cat_to_code=False)

  num_base_models_times_samples = y_preds.shape[0]

  # # Compare all performance brackets with ref_* bracket
  # config.cfg.MIN_BASE_MODEL_ACCURACY = config.cfg.MIN_BASE_MODEL_ACCURACY
  # config.cfg.MAX_BASE_MODEL_ACCURACY = config.cfg.MAX_BASE_MODEL_ACCURACY

  # # config.cfg.MIN_BASE_MODEL_ACCURACY = 0.056
  # # config.cfg.MAX_BASE_MODEL_ACCURACY = 0.154

  # # config.cfg.MIN_BASE_MODEL_ACCURACY = 0.385
  # # config.cfg.MAX_BASE_MODEL_ACCURACY = 0.574

  # all_loaded_data_files = load_processed_data_files({
  #     'samples': False,
  #     'y_preds': True,
  #     'y_trues': True,
  #     'w_chkpt': False,
  #     'w_final': False,
  #     'explans': True,
  #     'hparams': True,
  #     'metrics': True,
  # }, explanation_type=config.cfg.EXPLANATION_TYPE)
  # # ref_samples = all_loaded_data_files['samples']
  # ref_y_preds = all_loaded_data_files['y_preds']
  # ref_y_trues = all_loaded_data_files['y_trues']
  # ref_explans = all_loaded_data_files['explans']
  # ref_hparams = all_loaded_data_files['hparams']
  # ref_metrics = all_loaded_data_files['metrics']

  # # Reorder columns for easier readability when debugging.
  # ref_hparams = ref_hparams[[*config.CAT_HPARAMS, *config.NUM_HPARAMS]]
  # ref_hparams = process_hparams(ref_hparams, round_num=True, cat_to_code=False)

  # ref_num_base_models_times_samples = ref_y_preds.shape[0]

  treatment_effect_tracker = pd.DataFrame({
      'x_offset_idx': [],
      'col_type': [],
      'h1_h2_str': [],
      'y_pred_ite': [],
      'explan_ite': [],
      'y_pred_ite_var': [],
      'explan_ite_var': [],
      'pred_correctness': [],
  })

  treatment_effect_tracker_all = pd.DataFrame({
      'x_offset_idx': [],
      'col_type': [],
      'h1_h2_str': [],
      'd_y_preds': [],
      'd_explans': [],
  })

  # This only holds for the 0-1 accuracy bracket
  # assert np.isclose(
  #     num_base_models_times_samples,
  #     config.cfg.NUM_BASE_MODELS * config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
  # )

  # Iterate over different samples, X.
  for row_idx, x_offset_idx in enumerate(
      range(config.cfg.NUM_SAMPLES_TO_PLOT_TE_FOR)):

    start_time = time.time()
    logging.info(
        '%sProcessing instance w/ index `%d`...%s',
        Bcolors.HEADER,
        x_offset_idx,
        Bcolors.ENDC,
    )

    # x_* prefix is used for variables that correspond to instance x.
    x_indices = range(
        x_offset_idx,
        num_base_models_times_samples,
        config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
    )
    # Within each performance bucket, when computing the effect of h=h1 vs h=h2,
    # we need to look at the Y (or E) samples from some subset of X instances on
    # some someset of models (say 100 identical X samples for the portion of 30K
    # models that fall within that bucket). If one were to look at ALL X samples
    # on ALL models with h=h1 models vs ALL models with h=h2 models, this would
    # amount to comparing (i.e., computing the kernel matrix) between two large
    # matrices, which would yield a very expensive processing. This is wasteful
    # especially since we could do with only say 100 random samples from each of
    # the matrices Therefore, only limit samples to a random subset of samples.
    x_indices = random_state.permutation(
        x_indices)[:config.cfg.NUM_BASE_MODELS_FOR_KERNEL]
    # x_samples = samples[x_indices, :]
    x_y_preds = y_preds[x_indices, :]
    x_y_trues = y_trues[x_indices, :]
    x_explans = explans[x_indices, :]
    x_hparams = hparams.iloc[x_indices]

    # Normalize x_explans accordingly (faster compute when running on subsample)
    start_time = time.time()
    image_shape = other.get_dataset_info(config.cfg.DATASET)['data_shape'][:2]
    x_explans = explanation_utils.normalize_explans(x_explans, image_shape)
    logging.info(
        '%sElapsed time (normalizing x_explans): `%f`.%s',
        Bcolors.HEADER,
        time.time() - start_time,
        Bcolors.ENDC,
    )

    # # ref_x_* prefix is used for variables that correspond to instance x.
    # ref_x_indices = range(
    #     x_offset_idx,
    #     ref_num_base_models_times_samples,
    #     config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
    # )
    # Within each performance bucket, when computing the effect of h=h1 vs h=h2,
    # we need to look at the Y (or E) samples from some subset of X instances on
    # some someset of models (say 100 identical X samples for the portion of 30K
    # models that fall within that bucket). If one were to look at ALL X samples
    # on ALL models with h=h1 models vs ALL models with h=h2 models, this would
    # amount to comparing (i.e., computing the kernel matrix) between two large
    # matrices, which would yield a very expensive processing. This is wasteful
    # especially since we could do with only say 100 random samples from each of
    # the matrices Therefore, only limit samples to a random subset of samples.
    # ref_x_indices = random_state.permutation(
    #     ref_x_indices)[:config.cfg.NUM_BASE_MODELS_FOR_KERNEL]
    # # ref_x_samples = ref_samples[ref_x_indices, :]
    # ref_x_y_preds = ref_y_preds[ref_x_indices, :]
    # ref_x_y_trues = ref_y_trues[ref_x_indices, :]
    # ref_x_explans = ref_explans[ref_x_indices, :]
    # ref_x_hparams = ref_hparams.iloc[ref_x_indices]

    # # Normalize x_explans accordingly (faster compute if running on subsample)
    # start_time = time.time()
    # image_shape = other.get_dataset_info(config.cfg.DATASET)['data_shape'][:2]
    # ref_x_explans = explanation_utils.normalize_explans(
    #    ref_x_explans, image_shape)
    # logging.info(
    #     '%sElapsed time (processing ref_x_explans): `%f`.%s',
    #     Bcolors.HEADER,
    #     time.time() - start_time,
    #     Bcolors.ENDC,
    # )

    # For now, compare treatment/control within same group.
    # To save memory, comment out the ref_* code above.
    ref_x_y_preds = x_y_preds
    ref_x_y_trues = x_y_trues
    ref_x_explans = x_explans
    ref_x_hparams = x_hparams

    # Sanity check: irrespective of the base model,
    # X_i is shared and so should share y_true value.
    assert np.all(
        np.argmax(x_y_trues, axis=1) ==
        np.argmax(x_y_trues, axis=1)[0]
    )
    assert np.all(
        np.argmax(ref_x_y_trues, axis=1) ==
        np.argmax(ref_x_y_trues, axis=1)[0]
    )
    assert np.argmax(x_y_trues, axis=1)[0] == np.argmax(
        ref_x_y_trues, axis=1)[0]

    def get_y_preds_and_y_trues_and_explans_for_hparams_of_type(x_y_preds,
                                                                x_y_trues,
                                                                x_explans,
                                                                x_hparams,
                                                                col, hi, eq):
      """Get y_preds and explans when hparam of type col is hi (i.e., h_i).

      Args:
        x_y_preds: the y_preds of instance `x` over multiple base models.
        x_y_trues: the y_trues of instance `x` over multiple base models.
        x_explans: the explans of instance `x` over multiple base models.
        x_hparams: the hparams of instance `x` over multiple base models.
        col: the particular hparam column which is the be filtered over.
        hi: the unique value of the hparam to filter over.
        eq: flag stating whether or not the hparams should be equal to hi.

      Returns:
        x_hi_y_preds: y_preds for when hparam of type col is hi.
        x_hi_y_trues: y_trues for when hparam of type col is hi.
        x_hi_explans: explans for when hparam of type col is hi.
      """

      # Get list of indices where hparam of type col is hi,
      # then filter the y_preds and explans matrices accordingly.
      if eq:
        x_hi_indices = x_hparams.index[x_hparams[col] == hi].to_list()
      else:
        x_hi_indices = x_hparams.index[x_hparams[col] != hi].to_list()
      x_hi_y_preds = x_y_preds[x_hi_indices, :]
      x_hi_y_trues = x_y_trues[x_hi_indices, :]
      x_hi_explans = x_explans[x_hi_indices, :]

      # Some explanations may unfortunately have nan values;
      # remove these and the corresponding (non-nan) prediction values.
      keep_idx = []
      for idx in range(x_hi_explans.shape[0]):
        if np.any(np.isnan(x_hi_explans[idx])):
          continue
        keep_idx.append(idx)
      x_hi_y_preds = x_hi_y_preds[keep_idx]
      x_hi_y_trues = x_hi_y_trues[keep_idx]
      x_hi_explans = x_hi_explans[keep_idx]
      assert (
          x_hi_y_preds.shape[0] ==
          x_hi_explans.shape[0] ==
          x_hi_explans.shape[0]
      )

      return x_hi_y_preds, x_hi_y_trues, x_hi_explans

    # Reset index of hparams df as they will be used to filter np arrays.
    x_hparams = x_hparams.reset_index()
    ref_x_hparams = ref_x_hparams.reset_index()

    # Iterate over different hparams, H.
    for col_idx, col in enumerate(config.ALL_HPARAMS):

      # logging.info('Processing hparam `%s`...', col)

      scatter_tracker = pd.DataFrame({
          'd_y_preds': [],
          'd_explans': [],
          'h1_h2_str': [],
          'pred_correctness': [],
      })

      # Iterate over all pairs (h1, h2) of unique values of this hparam type...
      # for h1, h2 in itertools.permutations(x_hparams[col].unique(), 2):
      for hi in x_hparams[col].unique():

        x_h1_y_preds, x_h1_y_trues, x_h1_explans = get_y_preds_and_y_trues_and_explans_for_hparams_of_type(
            x_y_preds,
            x_y_trues,
            x_explans,
            x_hparams,
            col,
            hi,
            True,
        )
        x_h2_y_preds, x_h2_y_trues, x_h2_explans = get_y_preds_and_y_trues_and_explans_for_hparams_of_type(
            ref_x_y_preds,
            ref_x_y_trues,
            ref_x_explans,
            ref_x_hparams,
            col,
            hi,
            False,
        )

        if (
            len(x_h1_y_preds) == 0 or
            len(x_h2_y_preds) == 0 or
            len(x_h1_explans) == 0 or
            len(x_h2_explans) == 0
        ):
          continue  # Skip to next combination of h1 h2 when kernels empty.

        def get_dissimilarity(X, Y):
          k = config.cfg.TREATMENT_KERNEL
          k_XX = pairwise_kernels(X, X, metric=k, n_jobs=-1)
          k_XY = pairwise_kernels(X, Y, metric=k, n_jobs=-1)
          k_YY = pairwise_kernels(Y, Y, metric=k, n_jobs=-1)

          return (
              (- 2 * k_XY + np.expand_dims(np.diag(k_XX), axis=1)).T +
              np.expand_dims(np.diag(k_YY), axis=1)
          ).T

          # kernel_X_Y = np.zeros((X.shape[0], Y.shape[0]))
          # # ITE: || \phi(x) - \phi(y) ||^2 = k(x, x) - 2 k(x, y) + k(y, y)
          # for idx_1 in range(len(X)):
          #   for idx_2 in range(len(Y)):
          #     kernel_X_Y[idx_1, idx_2] = (
          #         k_XX[idx_1, idx_1] -
          #         2 * k_XY[idx_1, idx_2] +
          #         k_YY[idx_2, idx_2]
          #     )
          # return kernel_X_Y

        x_h1_h2_kernel_y_preds = get_dissimilarity(x_h1_y_preds, x_h2_y_preds)
        x_h1_h2_kernel_explans = get_dissimilarity(x_h1_explans, x_h2_explans)

        assert x_h1_h2_kernel_y_preds.shape == x_h1_h2_kernel_explans.shape

        if False:
          num_samples = 10
          fig, axes = plt.subplots(1, 3)
          color_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
          im = axes[0].imshow(
              x_h1_h2_kernel_y_preds[:num_samples,:num_samples],
              norm=color_norm)
          axes[0].axis('off')
          im = axes[1].imshow(x_h1_y_preds[:num_samples], norm=color_norm)
          axes[1].set_title(r'$Y_{h_i} \in R^{10}$')
          axes[1].axis('off')
          im = axes[2].imshow(x_h2_y_preds[:num_samples], norm=color_norm)
          axes[2].set_title(r'$Y_{h_{not-i}} \in R^{10}$')
          axes[2].axis('off')
          tmp_col = config.HPARAM_NAME_CONVERTER[col]
          if '$' in tmp_col:
            tmp_col = tmp_col[1:-1]
          tmp_range = config.RANGE_ACCURACY_CONVERTER[
              '%s_%s' %
              (
                  config.cfg.MIN_BASE_MODEL_ACCURACY,
                  config.cfg.MAX_BASE_MODEL_ACCURACY,
              )
          ]
          tmp_hi = hi if isinstance(hi, str) else f'{float(hi):.4f}'

          def highlight_cell(x, y, ax=None, **kwargs):
            rect = matplotlib.patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)
            ax.add_patch(rect)
            return rect

          assert np.all(
              np.argmax(x_h2_y_trues[:num_samples], axis=1) ==
              np.argmax(x_h2_y_trues[:num_samples], axis=1)[0]
          )
          true_label = np.argmax(x_h2_y_trues[:num_samples], axis=1)[0]
          for idx in range(len(x_h1_y_preds[:num_samples])):
            highlight_cell(
                true_label, idx, ax=axes[1], color="red", linewidth=2)
          for idx in range(len(x_h2_y_preds[:num_samples])):
            highlight_cell(
                true_label, idx, ax=axes[2], color="red", linewidth=2)

          axes[0].set_title((
            r'kernel ($\mu$: %.3f)' %
            np.mean(x_h1_h2_kernel_y_preds[:num_samples,:num_samples])
          ), fontsize=8)
          axes[1].set_title((
              r'$Y_{h_'
              '{'
              f'{tmp_col}'
              r'} ='
              f' {tmp_hi}'
              r'} \in R^{10}$'
          ), fontsize=8)
          axes[2].set_title((
              r'$Y_{h_'
              '{'
              f'{tmp_col}'
              r'} \neq'
              f' {tmp_hi}'
              r'} \in R^{10}$'
          ), fontsize=8)

          # fig.suptitle('%s - %s (%s)' % (tmp_col, hi, tmp_range))
          fig.colorbar(im, ax=axes.ravel().tolist(), location='bottom')
          plt.tight_layout(rect=[0, 0.95, 1, 0.95])
          fig.savefig(
              gfile.GFile(
                  os.path.join(
                      config.cfg.PLOTS_DIR_PATH,
                      '%s - %s - %s.png' % (x_offset_idx, col, hi),
                  ),
                  'wb',
              ),
              dpi=400,
          )

        # Depending on the type of kernel function used, the kernel matrix
        # may or may not be symmetric (consider, e.g., k(x,y) = x - y is not
        # symmetric). Therefore, we DO NOT only take the lower-/upper-triangular
        # portions of the kernel matrix but take all values.
        d_y_preds = x_h1_h2_kernel_y_preds.flatten()
        d_explans = x_h1_h2_kernel_explans.flatten()

        # Under h1 and h2, zero, one, or both of the resulting y_preds may be
        # incorrect. Distinguish between three cases of T-T, T-F/F-T, and F-F
        # and plot accordingly.
        h1_pred_correctness = (
            np.argmax(x_h1_y_preds, axis=1) == np.argmax(x_h1_y_trues, axis=1)
        ).astype(int)
        h2_pred_correctness = (
            np.argmax(x_h2_y_preds, axis=1) == np.argmax(x_h2_y_trues, axis=1)
        ).astype(int)
        pred_correctness = np.add.outer(
            h1_pred_correctness,
            h2_pred_correctness,
        ).flatten().astype(int)

        scatter_tracker = scatter_tracker.append(
            pd.DataFrame({
                'd_y_preds': d_y_preds,
                'd_explans': d_explans,
                'h1_h2_str': ['%s: %s vs not' % (col[7:], hi)] * len(d_explans),
                'pred_correctness': pred_correctness,
            }),
            ignore_index=True,
        )

      # Compute averages (this is the ITE value).
      for h1_h2_str in scatter_tracker['h1_h2_str'].unique():
        for pred_correctness in scatter_tracker['pred_correctness'].unique():
          filtered = scatter_tracker.where(
              (scatter_tracker['h1_h2_str'] == h1_h2_str) &
              (scatter_tracker['pred_correctness'] == pred_correctness)
          ).dropna()
          y_pred_ite = np.mean(filtered['d_y_preds'])
          explan_ite = np.mean(filtered['d_explans'])
          y_pred_ite_var = np.var(filtered['d_y_preds'])
          explan_ite_var = np.var(filtered['d_explans'])
          treatment_effect_tracker = treatment_effect_tracker.append(
              pd.DataFrame({
                  'x_offset_idx': [f'x_{x_offset_idx}'],
                  'col_type': [col],
                  'h1_h2_str': [h1_h2_str],
                  'y_pred_ite': [y_pred_ite],
                  'explan_ite': [explan_ite],
                  'y_pred_ite_var': [y_pred_ite_var],
                  'explan_ite_var': [explan_ite_var],
                  'pred_correctness': [pred_correctness],
              }),
              ignore_index=True,
          )
          treatment_effect_tracker_all = treatment_effect_tracker_all.append(
              pd.DataFrame({
                  'x_offset_idx': [f'x_{x_offset_idx}'] * len(filtered),
                  'col_type': [col] * len(filtered),
                  'h1_h2_str': [h1_h2_str] * len(filtered),
                  'd_y_preds': filtered['d_y_preds'],
                  'd_explans': filtered['d_explans'],
              }),
              ignore_index=True,
          )
    logging.info(
        '%sElapsed time (processing ITE of all hparams): `%f`.%s',
        Bcolors.HEADER,
        time.time() - start_time,
        Bcolors.ENDC,
    )

  suffix = (
      f'{config.cfg.TREATMENT_KERNEL}_'
      f'{config.cfg.EXPLANATION_TYPE}_'
      f'{config.cfg.MIN_BASE_MODEL_ACCURACY}_'
      f'{config.cfg.MAX_BASE_MODEL_ACCURACY}'
  )

  with file_handler(
      config.cfg.EXP_DIR_PATH,
      f'treatment_effect_tracker_{suffix}.npy',
      'wb',
  ) as f:
    pickle.dump(treatment_effect_tracker, f, protocol=4)

  with file_handler(
      config.cfg.EXP_DIR_PATH,
      f'treatment_effect_tracker_all_{suffix}.npy',
      'wb',
  ) as f:
    pickle.dump(treatment_effect_tracker_all, f, protocol=4)

  del y_preds, y_trues, explans, hparams, metrics
  gc.collect()
