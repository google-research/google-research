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

"""Dataset utilities for MNIST and Omniglot datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


_ALL_SPLITS = ['train', 'validation', 'test']


def get_all_datapoints(dataset):
  """Returns all datapoints in a dataset.

  Args:
    dataset: (tf.data.Dataset) dataset containing the images to hash.

  Returns:
    A list of datapoints returned from the dataset.
  """
  session = tf.Session(graph=tf.get_default_graph())
  iterator = dataset.make_one_shot_iterator().get_next()

  data = []

  try:
    while True:
      data.append(session.run(iterator))
  except tf.errors.OutOfRangeError:
    pass

  return data


def convert_list_to_dataset(data):
  """Shuffles a list of datapoints and converts it into tf.data.Dataset.

  Args:
    data: (list of dicts) list of datapoints, each being a dict containing
      keys 'image' and 'label'.

  Returns:
    A tf.data.Dataset containing the datapoints from `data` in random order.
  """
  np.random.shuffle(data)

  images = np.array([datapoint['image'] for datapoint in data])
  labels = np.array([datapoint['label'] for datapoint in data])

  # Convert into a tf.data.Dataset.
  data = tf.data.Dataset.from_tensor_slices((images, labels))

  # Convert the datapoints from pairs back to dicts.
  data = data.map(lambda image, label: {'image': image, 'label': label})

  return data.cache()


def group_by_label(datapoints, num_labels):
  """Groups a list of datapoints by the classification label.

  Args:
    datapoints: (list of dicts) list of datapoints, each being a dict containing
      keys 'image' and 'label'.
    num_labels: (int) number of labels.

  Returns:
    A list of lists, containing all elements of `datapoints` grouped by
    the 'label' key.
  """
  data_grouped_by_label = [[] for _ in range(num_labels)]

  for datapoint in datapoints:
    data_grouped_by_label[datapoint['label']].append(datapoint)

  return data_grouped_by_label


def get_mnist():
  """Loads the MNIST dataset.

  Returns:
    A pair of:
      - a dictionary with keys 'train' and 'test', containing `tf.data.Dataset`s
        for train and test, respectively.
      - an integer denoting the number of classes in the dataset
  """
  dataset = tfds.load('mnist')

  train_dataset = dataset['train'].map(normalize).cache()
  test_dataset = dataset['test'].map(normalize).cache()

  return {
      'train': train_dataset.shuffle(buffer_size=60000),
      'test': test_dataset
  }, 10


def get_mnist_in_cifar_format():
  """Loads the MNIST dataset, converts the inputs to 32x32 RGB images.

  Returns:
    A pair of:
      - a dictionary with keys 'train' and 'test', containing `tf.data.Dataset`s
        for train and test, respectively.
      - an integer denoting the number of classes in the dataset
  """
  task_data, num_classes = get_mnist()
  return convert_format_mnist_to_cifar(task_data), num_classes


def get_rotated_mnist():
  """Loads the MNIST dataset with each input image rotated by 90 degrees.

  Returns:
    A pair of:
      - a dictionary with keys 'train' and 'test', containing `tf.data.Dataset`s
        for train and test, respectively.
      - an integer denoting the number of classes in the dataset
  """
  dataset = tfds.load('mnist')

  train_dataset = dataset['train'].map(normalize).map(rotate90).cache()
  test_dataset = dataset['test'].map(normalize).map(rotate90).cache()

  return {
      'train': train_dataset.shuffle(buffer_size=60000),
      'test': test_dataset
  }, 10


def get_fashion_mnist():
  """Loads the Fashion-MNIST dataset.

  Returns:
    A pair of:
      - a dictionary with keys 'train' and 'test', containing `tf.data.Dataset`s
        for train and test, respectively.
      - an integer denoting the number of classes in the dataset
  """
  dataset = tfds.load('fashion_mnist')

  train_dataset = dataset['train'].map(normalize).cache()
  test_dataset = dataset['test'].map(normalize).cache()

  return {
      'train': train_dataset.shuffle(buffer_size=60000),
      'test': test_dataset
  }, 10


def get_fashion_mnist_in_cifar_format():
  """Loads the Fashion-MNIST dataset, converts the inputs to 32x32 RGB images.

  Returns:
    A pair of:
      - a dictionary with keys 'train' and 'test', containing `tf.data.Dataset`s
        for train and test, respectively.
      - an integer denoting the number of classes in the dataset
  """
  task_data, num_classes = get_fashion_mnist()
  return convert_format_mnist_to_cifar(task_data), num_classes


def get_leave_one_out_classification(task_data, num_classes, leave_out_class):
  """Creates a task of telling apart all classes besides one.

  Args:
    task_data: (dict) dictionary containing `tf.data.Dataset`s, for example
      as returned from `get_mnist`.
    num_classes: (int) number of classification classes in the original task.
    leave_out_class: (int) id of the class that should be left out in
      the returned task.

  Returns:
    A pair of:
      - a dictionary containing `tf.data.Dataset`s for the new task.
      - an integer denoting the number of classes in the new task.
  """
  task_data = task_data.copy()

  def convert_label(data):
    data['label'] -= tf.cast(
        tf.math.greater(data['label'], leave_out_class), dtype=tf.int64)

    return data

  def is_good_class(data):
    if tf.math.equal(data['label'], leave_out_class):
      return False
    else:
      return True

  for split in task_data:
    task_data[split] = task_data[split].filter(is_good_class)
    task_data[split] = task_data[split].map(convert_label)
    task_data[split] = task_data[split].cache()

  return task_data, num_classes - 1


def convert_format_mnist_to_cifar(task_data):
  """Converts a dataset of MNIST-like grayscale images to 32x32 rgb images.

  Args:
    task_data: (dict) dictionary containing `tf.data.Dataset`s, for example
      as returned from `get_mnist`.

  Returns:
    The `task_data` dict after conversion.
  """
  task_data = task_data.copy()

  for split in task_data:
    task_data[split] = task_data[split].map(resize((32, 32)))
    task_data[split] = task_data[split].map(convert_to_rgb)
    task_data[split] = task_data[split].cache()

  return task_data


def get_cifar100(coarse_label_id):
  """Loads one of the CIFAR-100 coarse label tasks.

  Args:
    coarse_label_id: (int) coarse label id, must be between 0 and 19 inclusive.

  Returns:
    A pair of:
      - a dictionary with keys: 'train', 'validation' and 'test'. Values for
        these keys are `tf.data.Dataset`s for train, validation and test,
        respectively.
      - an integer denoting the number of classes in the dataset
  """
  assert 0 <= coarse_label_id < 20

  def pred(datapoint):
    return tf.math.equal(datapoint['coarse_label'], coarse_label_id)

  dataset = tfds.load(
      name='cifar100', as_dataset_kwargs={'shuffle_files': False})

  def preprocess(dataset_split):
    """Preprocess the input dataset."""
    dataset_split = dataset_split.filter(pred)
    dataset_split = dataset_split.map(normalize)

    all_datapoints = get_all_datapoints(dataset_split)

    fine_labels = set([datapoint['label'] for datapoint in all_datapoints])
    fine_labels = sorted(list(fine_labels))

    assert len(fine_labels) == 5

    split_size = len(all_datapoints)

    formatter = get_cifar100_formatter(fine_labels)

    dataset_split = dataset_split.map(formatter)
    dataset_split = dataset_split.cache()

    return dataset_split, split_size

  train_dataset, train_size = preprocess(dataset['train'])
  test_dataset, _ = preprocess(dataset['test'])

  raw_train_data = []
  raw_valid_data = []

  for data_group in group_by_label(get_all_datapoints(train_dataset), 5):
    group_size = len(data_group)

    # Make sure that the datapoints can be divided evenly.
    assert group_size % 5 == 0

    np.random.shuffle(data_group)

    train_size = int(0.8 * group_size)

    raw_train_data += data_group[:train_size]
    raw_valid_data += data_group[train_size:]

  train_dataset = convert_list_to_dataset(raw_train_data)
  valid_dataset = convert_list_to_dataset(raw_valid_data)

  return {
      'train': train_dataset.shuffle(buffer_size=len(raw_train_data)),
      'validation': valid_dataset.shuffle(buffer_size=len(raw_valid_data)),
      'test': test_dataset
  }, 5


def get_omniglot_order():
  """Returns the omniglot alphabet names, in the order used in previous works.

  Returns:
    Alphabet names of the 50 Omniglot tasks, in the same order as used by
    multiple previous works, such as "Diversity and Depth in Per-Example
    Routing Models" (https://openreview.net/pdf?id=BkxWJnC9tX).
  """
  return [
      'Gujarati', 'Sylheti', 'Arcadian', 'Tibetan',
      'Old_Church_Slavonic_(Cyrillic)', 'Angelic', 'Malay_(Jawi_-_Arabic)',
      'Sanskrit', 'Cyrillic', 'Anglo-Saxon_Futhorc', 'Syriac_(Estrangelo)',
      'Ge_ez', 'Japanese_(katakana)', 'Keble', 'Manipuri',
      'Alphabet_of_the_Magi', 'Gurmukhi', 'Korean', 'Early_Aramaic',
      'Atemayar_Qelisayer', 'Tagalog', 'Mkhedruli_(Georgian)',
      'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Tengwar', 'Hebrew', 'N_Ko',
      'Grantha', 'Latin', 'Syriac_(Serto)', 'Tifinagh', 'Balinese', 'Mongolian',
      'ULOG', 'Futurama', 'Malayalam', 'Oriya',
      'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Avesta', 'Kannada', 'Bengali',
      'Japanese_(hiragana)', 'Armenian', 'Aurek-Besh', 'Glagolitic',
      'Asomtavruli_(Georgian)', 'Greek', 'Braille', 'Burmese_(Myanmar)',
      'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Atlantean'
  ]


def map_omniglot_alphabet_names_to_ids(alphabet_names):
  """Maps a list of Omniglot alphabet names into their ids.

  Args:
    alphabet_names: (list of strings) names of Omniglot alphabets to be mapped.

  Returns:
    A list of ints, corresponding to alphabet ids for the given names. All ids
    are between 0 and 49 inclusive.
  """
  _, info = tfds.load(name='omniglot', split=tfds.Split.ALL, with_info=True)

  alphabet_ids = [
      info.features['alphabet'].str2int(alphabet_name)
      for alphabet_name in alphabet_names
  ]

  return alphabet_ids


def get_omniglot(alphabet_id, size=None):
  """Loads one of the Omniglot alphabets.

  Args:
    alphabet_id: (int) alphabet id, must be between 0 and 49 inclusive.
    size: either None or a pair of ints. If set to None, the images will not
      be resized, and will retain their original size of 105x105. If set to
      a pair of ints, then the images will be resized to this size.

  Returns:
    A pair of:
      - a dictionary with keys: 'train', 'validation' and 'test'. Values for
        these keys are `tf.data.Dataset`s for train, validation and test,
        respectively.
      - an integer denoting the number of classes in the dataset
  """
  assert 0 <= alphabet_id < 50
  np.random.seed(seed=alphabet_id)

  pred = lambda datapoint: tf.math.equal(datapoint['alphabet'], alphabet_id)

  # The `as_dataset_kwargs` argument makes this function deterministic.
  dataset = tfds.load(
      name='omniglot',
      split=tfds.Split.ALL,
      as_dataset_kwargs={'shuffle_files': False})

  dataset = dataset.filter(pred)
  dataset = dataset.map(format_omniglot)
  dataset = dataset.map(normalize)
  dataset = dataset.map(convert_to_grayscale)

  # Flip to make the background consist of 0's and characters consist of 1's
  # (instead of the other way around).
  dataset = dataset.map(make_negative)

  if size:
    dataset = dataset.map(resize(size))

  all_datapoints = get_all_datapoints(dataset)
  num_classes = max([datapoint['label'] for datapoint in all_datapoints]) + 1

  data = {data_split: [] for data_split in _ALL_SPLITS}

  for data_group in group_by_label(all_datapoints, num_classes):
    group_size = len(data_group)

    # Make sure that the datapoints can be divided evenly.
    assert group_size % 10 == 0

    np.random.shuffle(data_group)

    train_size = int(0.5 * group_size)
    validation_size = int(0.2 * group_size)

    data['train'] += data_group[:train_size]
    data['validation'] += data_group[train_size:train_size+validation_size]
    data['test'] += data_group[train_size+validation_size:]

  train_size = len(data['train'])

  for split in _ALL_SPLITS:
    data[split] = convert_list_to_dataset(data[split])

  # Ensure that the order of training data is different in every epoch.
  data['train'] = data['train'].shuffle(buffer_size=train_size)

  return data, num_classes


def get_data_for_multitask_omniglot_setup(num_alphabets):
  """Loads a given number of Omniglot datasets for multitask learning.

  Args:
    num_alphabets: (int) number of alphabets to use, must be between 1 and 50
      inclusive.

  Returns:
    A pair of two lists (`task_data`, `num_classes_for_tasks`), each
    containing one element per alphabet. These lists respectively contain
    task input data and number of classification classes, as returned from
    `get_omniglot`.
  """
  alphabet_names = get_omniglot_order()[:num_alphabets]
  alphabet_ids = map_omniglot_alphabet_names_to_ids(alphabet_names)

  alphabets = [get_omniglot(alphabet_id) for alphabet_id in alphabet_ids]

  # Convert a list of pairs into a pair of lists and return
  return [list(tup) for tup in zip(*alphabets)]


def get_cifar100_formatter(fine_labels):
  """Formats a CIFAR-100 input into a standard format."""

  def format_cifar100(data):
    """Formats a CIFAR-100 input into a standard format.

    The formatted sample will have two keys: 'image', containing the input
    image, and 'label' containing the label.

    Args:
      data: dict, a sample from the CIFAR-100 dataset. Contains keys: 'image',
        'coarse_label' and 'label'.

    Returns:
      Formatted `data` dict.
    """
    del data['coarse_label']

    label = data['label']
    data['label'] = -1

    for i in range(len(fine_labels)):
      if tf.math.equal(label, fine_labels[i]):
        data['label'] = i

    return data

  return format_cifar100


def augment_with_random_crop(data, size=32):
  """Applies the "resize and crop" image augmentation.

  Args:
    data: dict, a sample from the dataset of size [32, 32, 3].
      Contains keys: 'image' and 'label'.
    size: (int) image size.

  Returns:
    The same dict, but after applying the image augmentation.
  """
  x = data['image']
  x = tf.image.resize_with_crop_or_pad(x, size + 8, size + 8)
  x = tf.image.random_crop(x, [size, size, 3])

  data['image'] = x
  return data


def format_omniglot(data):
  """Formats an Omniglot input into a standard format.

  The formatted sample will have two keys: 'image', containing the input image,
  and 'label' containing the label.

  Args:
    data: dict, a sample from the Omniglot dataset. Contains keys: 'image',
      'alphabet' and 'alphabet_char_id'.

  Returns:
    Formatted `data` dict.
  """
  data['label'] = data['alphabet_char_id']

  del data['alphabet_char_id']
  del data['alphabet']

  return data


def normalize(data):
  data['image'] = tf.to_float(data['image']) / 255.
  return data


def make_negative(data):
  data['image'] = 1. - data['image']
  return data


def rotate90(data):
  data['image'] = tf.image.rot90(data['image'])
  return data


def resize(size):
  def resize_fn(data):
    data['image'] = tf.image.resize_images(data['image'], size)
    return data

  return resize_fn


def convert_to_grayscale(data):
  data['image'] = tf.image.rgb_to_grayscale(data['image'])
  return data


def convert_to_rgb(data):
  data['image'] = tf.image.grayscale_to_rgb(data['image'])
  return data


def batch_all(dataset, batch_size):
  """Batches all splits in a dataset into batches of size `batch_size`."""
  return {
      key: dataset[key].batch(batch_size)
      for key in dataset.keys()
  }
