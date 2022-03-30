# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Auxiliary functions to run the UK Biobank.

References:
https://github.com/Google-Health/genomics-research/tree/main/ml-based-vcdr
https://keras.io/examples/keras_recipes/creating_tfrecords/
"""
import concurrent.futures
import copy
import math
import os
import pathlib
from typing import Dict, List, Optional, Tuple
import ml_collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import gfile

from icetea import ukb_utils


def build_datasets(
    dataset_config,
    outcomes,
    cache = False):
  """Returns train and evaluation datasets."""
  train_ds = ukb_utils.build_train_dataset(dataset_config, outcomes,
                                           cache=cache)
  pred_ds = ukb_utils.build_predict_dataset(dataset_config, outcomes,
                                            cache=cache)
  return train_ds, pred_ds


def extract_hx(path_train,
               path_output = None,
               epochs = 50,
               predict_name = 'ukb_image_examples-587x587.tfrecord',
               train_name = 'eval.tfrecord'):
  """Extracts the last-layer of an image model.

  Note that here we train the base model in a smaller set (eval.tfrecords) and
  save the predictions from the full set.
  Args:
    path_train: folder for TFRecords to train the base model.
    path_output: path to save csv file with h(x).
    epochs: how many epochs the base model will use.
    predict_name: name of the tfrecords used to extract features.
    train_name: name of the tfrecords used to train base model
  Returns:
    - Nothing. It saves on the path_output the extracted features in a csv file.
  """

  # Start Code
  config = ml_collections.ConfigDict()
  config.dataset = ml_collections.ConfigDict({
      'path': path_train,
      'predict': predict_name,
      'train': train_name,
      'num_train_examples': 79355,
      'batch_size': 16,
      'image_size': (587, 587),
      'random_horizontal_flip': True,
      'random_vertical_flip': True,
      'random_brightness_max_delta': 0.1147528,
      'random_saturation_lower': 0.5597273,
      'random_saturation_upper': 1.2748845,
      'random_hue_max_delta': 0.0251488,
      'random_contrast_lower': 0.9996807,
      'random_contrast_upper': 1.7704824,
      'use_cache': False,
  })

  config.train = ml_collections.ConfigDict({
      'use_mixed_precision': True,
      'use_distributed_training': False,
      'max_num_steps': 250000,
      'log_step_freq': 15,
      'fit_verbose': 1,
      'initial_epoch': 0,
  })

  config.outcomes = [
      ml_collections.ConfigDict({
          'name': 'vertical_cup_to_disc',
          'type': 'regression',
          'num_classes': 1,
          'loss': 'mse',
          'loss_weight': 1.0,
      }),
      ml_collections.ConfigDict({
          'name': 'vertical_cd_visibility',
          'type': 'classification',
          'num_classes': 3,
          'loss': 'ce',
          'loss_weight': 1.0,
      }),
      ml_collections.ConfigDict({
          'name': 'glaucoma_suspect_risk',
          'type': 'classification',
          'num_classes': 4,
          'loss': 'ce',
          'loss_weight': 1.0,
      }),
      ml_collections.ConfigDict({
          'name': 'glaucoma_gradability',
          'type': 'classification',
          'num_classes': 3,
          'loss': 'ce',
          'loss_weight': 1.0,
      }),
  ]

  config.model = ml_collections.ConfigDict({
      'backbone': 'inceptionv3',
      'backbone_drop_rate': 0.2,
      'input_shape': (587, 587, 3),
      'weights': 'imagenet',
      'weight_decay': 0.00004,
  })

  # optimizer
  config.opt = ml_collections.ConfigDict({
      'optimizer': 'adam',
      'learning_rate': 0.001,
      'beta_1': 0.9,
      'beta_2': 0.999,
      'epsilon': 0.1,
      'amsgrad': False,
      'use_model_averaging': True,
      'update_model_averaging_weights': False,
  })
  config.outcomes_sim = [
      ml_collections.ConfigDict({
          'name': 'sim_0_mu0',
          'type': 'regression',
          'num_classes': 1,
          'loss': 'mse',
          'loss_weight': 1.0,
      }),
      ml_collections.ConfigDict({
          'name': 'sim_0_mu1',
          'type': 'regression',
          'num_classes': 1,
          'loss': 'mse',
          'loss_weight': 1.0,
      })
  ]

  train_ds, eval_ds = ukb_utils.build_datasets(config.dataset, config.outcomes,
                                               cache=False,)

  verbose = config.train.get('fit_verbose', 0) if 'train' in config else 0

  ukb_model_utils = None  # replace with import from source in documentation.
  model = ukb_model_utils.compile_model(config)

  model.fit(
      train_ds,
      epochs=epochs,
      initial_epoch=config.train.get('initial_epoch', 0),
      steps_per_epoch=config.train.log_step_freq,
      verbose=verbose)

  extract = tf.keras.Model(model.inputs, model.layers[-5].output)

  path_out = os.path.join(path_output, 'hx.csv')
  _extraction(eval_ds, path_out, extract)


def _extraction(data, path, model):
  """Make predictions and extract last layer.

  Args:
    data: built dataset.
    path: folder to save csv file.
    model: fitted model.
  """
  progbar = tf.keras.utils.Progbar(
      None,
      width=30,
      verbose=1,
      interval=0.05,
      stateful_metrics=None,
      unit_name='step')

  features = []
  image_id = []
  # Save the first 5000 batches.
  for i, (inputs_batch, _, _) in enumerate(data):
    predict_batch = model.predict_on_batch(inputs_batch)
    image_id.append(inputs_batch['id'])
    features.append(predict_batch)
    progbar.add(1)
    if i > 5000:
      break

  features = np.array(features)
  s = features.shape
  features = features.reshape(s[0] * s[1], s[2])

  columns = [f'f{i}' for i in range(features.shape[1])]
  features = pd.DataFrame(data=features,
                          columns=columns)

  flat_id = np.concatenate(image_id).ravel()
  features['image_id'] = flat_id
  with gfile.GFile(path, 'wt') as table_names:
    table_names.write(features.to_csv(index=False))


def simulating_y_from_clinical(
    clinical_sim,
    var = 'creatinine',
    simulate_y = False,
    rep = 30):
  """Simulate the outcome Y from clinical or mu(x,t).

  Args:
    clinical_sim: pd.DataFrame with simulated pi() and mu(). Colnames: 'eid',
    'image_id', 'sim_0_pi','sim_0_mu0','sim_0_mu1',
       ...
    'sim_b_pi','sim_b_mu0','sim_b_mu1', clinical variables (optional)
    var: which variable from clinical data will be used as outcome.
    simulate_y: use mu(x,t) instead of var.
    rep: number of repetitions.
  Returns:
    clinical_sim: pd.DataFrame, keys: 'eid', 'image_id',
      'sim_0_pi','sim_0_mu0','sim_0_mu1',
       ...
      'sim_b_pi','sim_b_mu0','sim_b_mu1', clinical variables (optional),
      'sim_0_y', ..., 'sim_b_pi' (new columns)
    true_treatment_effect: np.array, length b (one value per repetition b).
  """
  scaler = MinMaxScaler((0, 100))
  true_treatment_effect = []

  for b in range(rep):
    y_name = f'sim_{b}_y'
    t_name = f'sim_{b}_pi'
    mu1_name = f'sim_{b}_mu1'
    mu0_name = f'sim_{b}_mu0'
    t = clinical_sim[t_name]
    t = t == 1

    if simulate_y:
      mu0 = copy.deepcopy(clinical_sim[mu0_name].values.reshape(-1, 1))
      mu1 = copy.deepcopy(clinical_sim[mu1_name].values.reshape(-1, 1))
      y = clinical_sim['sim_' + str(b) + '_mu0'].values
      y[t] = clinical_sim['sim_' + str(b) + '_mu1'].values[t]
      y = scaler.fit_transform(y.reshape(-1, 1))
      mu0 = scaler.transform(mu0)
      mu1 = scaler.transform(mu1)
      dif = mu0 - mu1
      true_treatment_effect.append(dif.mean())
      y = y.ravel()
    else:
      clinical_sim = clinical_sim.dropna(subset=[var])
      clinical_sim.reset_index(inplace=True, drop=True)
      y_ = scaler.fit_transform(clinical_sim[var].values.reshape(-1, 1))
      np.random.seed(b)
      tau = np.random.uniform(0, 5, 1)[0]
      y = [
          y_[i][0] + tau if t[i] else y_[i][0]
          for i in range(clinical_sim.shape[0])
      ]
      true_treatment_effect.append(tau)
    clinical_sim[y_name] = y
  return clinical_sim, true_treatment_effect


def join_tfrecord_csv(path_simulations,
                      input_prefix,
                      path_input,
                      path_output,
                      output_prefix = 'train_features'):
  """Join TFRecord files and a csv file with a common id.

  Args:
    path_simulations: path for csv file
    input_prefix: prefix of TFRecords
    path_input: path to TFRecords
    path_output: path to save TFRecors (should be differet to avoid overwrite)
    output_prefix: prefix of joined TFRecords
  """

  def _to_basename(path):
    path = path.split('/')[-1]
    return path.split('_')[0]

  def _get_value_key(field_name):
    """Returns an output-formatted TFRecord value key for `field_name`."""
    return os.path.join('image', field_name, 'value')

  def _get_weight_key(field_name):
    """Returns an output-formatted TFRecord weight key for `field_name`."""
    image_prefix = 'image/'
    weight_suffix = '/value'
    return f'{image_prefix}{field_name}{weight_suffix}'

  def _build_ukb_tfrecord_features():
    """Returns a feature dictionary used to parse TFRecord examples.

    We assume that the UKB TFRecords are defined using the following schema:

      1. An encoded image with key `IMAGE_ENCODED_TFR_KEY` that can be decoded
         using `tf.image.decode_png`.
      2. A unique identifier for each image with key `IMAGE_ID_TFR_KEY`.

    The `tf.io.parse_single_example` function uses the resulting feature
    dictionary to parse each TFRecord.

    Returns:
      A feature dictionary for parsing TFRecords.
    """
    image_prefix = 'image/'
    image_id_tfr_key = f'{image_prefix}id'
    image_encoded_tfr_key = f'{image_prefix}encoded'
    features = {}
    features[image_encoded_tfr_key] = tf.io.FixedLenFeature([], tf.string)
    features[image_id_tfr_key] = tf.io.FixedLenFeature([1], tf.string)
    return features

  def _get_parse_example_fn(features):
    """Returns a function that parses a TFRecord example using `features`."""

    def _parse_example(example):
      return tf.io.parse_single_example(example, features)

    return _parse_example

  def _field_to_keys(
      field,
      delimiter = ':'):
    """Returns a label column field's base field name, value key, and weight key.

    For example, given `glaucoma_gradability:GRADABLE`, this function returns a
    triple containing `glaucoma_gradability`, image/glaucoma_gradability/value`,
    and `image/glaucoma_gradability/weight`.

    Note: We handle the following set of special fields differently. This subset
    of fields may not have an associated value or weight key:
      - eid: field='eid', value_key='eid', weight_key=None
      - image_id: field='image_id', value_key='image/id', weight_key=None

    Args:
      field: A prediction column field.
      delimiter: The delimiter used to split the field into the base field name
        and the field value.

    Returns:
      The field's base field name, value key, and weight key.
    """
    image_id_pred_key = 'image_id'
    eid = 'eid'
    special_cases = {image_id_pred_key, eid}
    if field in special_cases:
      if field == eid:
        return field, None
      return image_id_pred_key, None

    if delimiter not in field:
      ValueError(f'Unexpected field format: {field}')
    try:
      field_name, _ = field.split(delimiter)
    except ValueError:
      field_name = field.split(delimiter)
    field_name = field_name[0]
    return _get_value_key(field_name), _get_weight_key(field_name)

  def _map_id_to_encoded_image(ds):
    """Convert a `tf.data.Dataset` containing images and ids to a tensor dict."""
    ids_to_encoded = {}
    image_prefix = 'image/'
    image_id_tfr_key = f'{image_prefix}id'
    image_encoded_tfr_key = f'{image_prefix}encoded'
    for example in ds:
      image_id = example[image_id_tfr_key].numpy()[0].decode('utf-8')
      image_id = _to_basename(image_id)
      image_encoded = example[image_encoded_tfr_key]
      ids_to_encoded[image_id] = image_encoded
    return ids_to_encoded

  def _map_ukb_id_to_encoded_images(
      ukb_tfrecord_path):
    """Returns a dictionary of encoded image tensors keyed on image id."""
    # Load and parse the TFRecords located at `ukb_tfrecord_path`.
    outcome_features = _build_ukb_tfrecord_features()
    tfrecord_ds = tf.data.TFRecordDataset(filenames=ukb_tfrecord_path)
    parsed_tfrecord_ds = tfrecord_ds.map(
        _get_parse_example_fn(outcome_features),
        num_parallel_calls=tf.data.AUTOTUNE)
    # Build a map of image ids to encoded image tensors.
    id_to_encoded_images = _map_id_to_encoded_image(parsed_tfrecord_ds)
    return id_to_encoded_images

  def _record_to_tensor_dict(label_record):
    """Converts a CSV label record to the expected tensor dictionary format."""

    filtered_record = label_record.keys()
    # Initialize empty arrays of the expected field size so that we can set the
    # value of each head's prediction at the corresponding index.
    # Since we have predictions for all record fields, we set the sample
    # weight to `1.0` for all records.
    tf_dict = {}
    for field in filtered_record:
      value_key, weight_key = _field_to_keys(field)
      tf_dict[value_key] = [None]
      if weight_key:
        tf_dict[weight_key] = [1.0]

    # Populate each key at the given index.
    for field, value in label_record.items():
      value_key, weight_key = _field_to_keys(field)
      tf_dict[value_key][0] = value

    # Assert that we have populated all expected field indices.
    for field, value in tf_dict.items():
      assert None not in value
    return tf_dict

  def _build_label_tensor_dicts(id_to_encoded_images, label_records):
    """Converts label records w/ids in `id_to_encoded_images` to tensor dicts."""
    tensor_dicts = []
    for first_name in id_to_encoded_images:
      if first_name not in label_records:
        continue
      record = label_records[first_name]
      tensor_dicts.append(_record_to_tensor_dict(record))

    return tensor_dicts

  def _tf_dict_to_example(tf_dict, encoded_image):
    """Merges a tensor dict and the encoded image into a `tf.train.Example`."""
    example = tf.train.Example()
    image_prefix = 'image/'
    image_id_tfr_key = f'{image_prefix}id'
    image_encoded_tfr_key = f'{image_prefix}encoded'
    eid = 'eid'
    for key, value in tf_dict.items():
      if key in {image_id_tfr_key, eid}:
        example.features.feature[key].bytes_list.value.append(
            str(value[0]).encode('utf-8'))
      else:
        example.features.feature[key].float_list.value.extend(value)
    example.features.feature[image_encoded_tfr_key].bytes_list.value.append(
        encoded_image.numpy())
    return example

  def _convert_tensor_dicts_to_examples(
      records,
      id_to_encoded_images):
    """Converts a list of tensor dict records to `tf.train.Example`s."""
    examples = []
    image_prefix = 'image/'
    image_id_tfr_key = f'{image_prefix}id'
    for record in records:
      image_basename = _to_basename(record[image_id_tfr_key][0])
      encoded_image = id_to_encoded_images[image_basename]
      examples.append(_tf_dict_to_example(record, encoded_image))
    return examples

  def _write_tf_examples(tf_examples, output_path):
    """Writes a list of `tf.train.Example`s as TFRecords the `output_path`."""
    with tf.io.TFRecordWriter(str(output_path)) as writer:
      for example in tf_examples:
        writer.write(example.SerializeToString())

  def _print_status(status,
                    tfrecord_path,
                    output_path,
                    error = None):
    """Prints the update status for the given paths."""
    lines = [
        f'\nTFRecord update {status}:'
        f'\n\ttfrecord_path="{tfrecord_path}"',
        f'\n\toutput_path="{output_path}"',
    ]
    if error:
      lines.append(f'\n\terror=\n{error}')

  def _add_labels_to_tfrecords(tfrecord_path, output_path,
                               label_records, overwrite=True):
    """Runs the pipeline, constructing new labeled UKB TFRecords."""
    # Only regenerate existing TFRecords if `overwrite==True`.
    if os.path.exists(str(output_path)) and not overwrite:
      _print_status('SKIPPED (`output_path` already exists)', tfrecord_path,
                    output_path)
      return
    try:
      id_to_encoded_images = _map_ukb_id_to_encoded_images(tfrecord_path)
      label_tfrecords = _build_label_tensor_dicts(id_to_encoded_images,
                                                  label_records)
      tf_examples = _convert_tensor_dicts_to_examples(label_tfrecords,
                                                      id_to_encoded_images)
      _write_tf_examples(tf_examples, output_path)
      _print_status('completed successfully', tfrecord_path, output_path)
    except ValueError:
      _print_status('FAILED', tfrecord_path, output_path)
      return tfrecord_path

  with gfile.GFile(path_simulations) as f:
    features = pd.read_csv(f)

  if path_input == path_output:
    raise ValueError('Input and Output path should be different!')

  # Global variable inside this function
  label_records = {
      _to_basename(record['image_id']): record
      for record in features.to_dict('records')
  }

  # Fetch the set of UKB input TFRecord shards.
  try:
    ukb_tfrecord_input_filenames = [
        filename for filename in os.listdir(str(path_input))
        if filename.startswith(input_prefix)
    ]
  except FileNotFoundError:
    ukb_tfrecord_input_filenames = [
        filename for filename in gfile.listdir(str(path_input))
        if filename.startswith(input_prefix)
    ]

  ukb_tfrecord_input_filepaths = [
      path_input + '/' + filename for filename in ukb_tfrecord_input_filenames
  ]
  ukb_tfrecord_output_filenames: List[str] = [
      filename.replace(input_prefix, output_prefix)
      for filename in ukb_tfrecord_input_filenames
  ]
  ukb_tfrecord_output_filepaths: List[pathlib.Path] = [
      path_output +'/'+ filename
      for filename in ukb_tfrecord_output_filenames
  ]
  # Generate arguments for converting each shard.
  add_labels_to_tfrecords_args1 = []
  add_labels_to_tfrecords_args2 = []

  for tfrecord_path, output_path in zip(ukb_tfrecord_input_filepaths,
                                        ukb_tfrecord_output_filepaths):
    add_labels_to_tfrecords_args1.append(tfrecord_path)
    add_labels_to_tfrecords_args2.append(output_path)

  # Process each shard in parallel, pairing images and labels.
  with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
    failed_paths_with_none = list(
        executor.map(_add_labels_to_tfrecords,
                     add_labels_to_tfrecords_args1,
                     add_labels_to_tfrecords_args2,
                     label_records))

  # Print the filepaths of any failed runs for reprocessing.
  failed_paths = [str(path) for path in failed_paths_with_none if path]
  if failed_paths:
    failed_path_str = '\n\t'.join([''] + failed_paths)
    print(f'The following TFRecord updates failed:{failed_path_str}')
  print('DONE with JOIN features')


def generate_simulations(path,
                         b = 30,
                         hx_name = 'hx.csv',
                         output_name = 'sim_seeds.csv',
                         extract_id = True,
                         load_features = False,
                         features = None):
  """Generate the p(x) and mu(x,y) from h(x).

  Args:
    path: path for the hx.csv file.
    b: repetitions
    hx_name: features extracted name.
    output_name: new file name.
    extract_id: for the UK Biobank we need to extract ID from url.
    load_features: read features from path.
    features: pass features pd.DataFrame as argument.
  """
  def _eid_from_image_id(image_id):
    """Parses an `image_id` and returns the corresponding eid."""
    eid = image_id.split('/')[-1]
    eid = eid.split('_')[0]
    try:
      eid = int(eid)
    except ValueError:
      raise ValueError('Image ID did not match our transformation')
    return str(eid)

  def _generation_weights(n_cols = 2048,
                          seed = 0):
    np.random.seed(seed)
    gam = np.random.uniform(-1, 1, n_cols)
    eta1 = np.random.uniform(2, 3, n_cols)
    eta0 = np.random.uniform(1, 3, n_cols)
    return gam, eta1, eta0

  def _pi_x_function(features, gam):
    def sigmoid(x):
      return 1 / (1 + math.exp(-x))
    pi = np.matmul(features, gam)
    pi = pi.reshape(-1, 1)
    scaler = MinMaxScaler((-2, 2))
    pi = scaler.fit_transform(pi)
    pi = pi.ravel()
    pi = [sigmoid(item) for item in pi]
    np.random.seed(0)
    t = [np.random.binomial(1, item) for item in pi]
    return t

  def _mu_x_function(features, eta1, eta0):
    mu1 = np.array(np.matmul(features, eta1))
    mu0 = np.array(np.matmul(features, eta0))
    full = np.array(np.concatenate([mu1, mu0]))
    scaler = MinMaxScaler()
    scaler.fit(full.reshape(-1, 1))
    mu1 = scaler.transform(mu1.reshape(-1, 1))
    mu0 = scaler.transform(mu0.reshape(-1, 1))
    return mu1, mu0

  if load_features:
    with gfile.GFile(os.path.join(path, hx_name), mode='rt') as f:
      features = pd.read_csv(f)

  features_only = features.drop(['image_id'], axis=1)
  output = pd.DataFrame(features['image_id'])

  if extract_id:
    eid = [_eid_from_image_id(item) for item in features['image_id']]
    output['eid'] = eid
  for i in range(b):
    gam, eta1, eta0 = _generation_weights(features_only.shape[1], i)
    pi = _pi_x_function(features_only.values, gam)
    mu1, mu0 = _mu_x_function(features_only.values, eta1, eta0)
    output['sim_' + str(i) + '_pi'] = pi
    output['sim_' + str(i) + '_mu1'] = mu1
    output['sim_' + str(i) + '_mu0'] = mu0

  with gfile.GFile(os.path.join(path, output_name), 'wt') as out:
    out.write(output.to_csv(index=False))
