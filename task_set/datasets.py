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

# Lint as: python3
"""Helper module for dealing with datasets loaded from TFDS."""

import copy
import enum
from typing import List, Dict, Optional, Text, Any, Tuple, Callable
import attr
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def tfds_load_dataset(dataset_name, *args, **kwargs):
  """Helper function used to bridge internal google, and the external world."""
  data_dir = kwargs.pop("data_dir", None)
  return tfds.load(
      dataset_name, *args, data_dir=data_dir, download=True, **kwargs)


class Split(enum.Enum):
  """Enum representing different splits of data for cross validation.

  Two validation sets are needed for meta-learning optimizers.
  """

  TRAIN = "TRAIN"
  VALID_INNER = "VALID_INNER"
  VALID_OUTER = "VALID_OUTER"
  TEST = "TEST"


def split_dataset(
    dataset,
    num_per_split,
    num_splits = 3,
):
  """Helper to split a dataset for cross validaton.

  The first num_splits-1 datasets contain num_per_split examples.

  The last dataset contains the remaining number of examples.
  This often used to split a training set into more validation sets:
  e.g. train_old --> [valid_inner, valid_outer, train]

  Args:
    dataset: name of tfds dataset
    num_per_split: number of examples to have for each of the split off dataset.
    num_splits: number of splits to create.

  Returns:
    A list of the requested datasets.
  """
  new_datasets = []

  # make the first n_split-1 splits containing num_per_split examples
  for i in range(num_splits - 1):
    new_datasets.append(dataset.skip(num_per_split * i).take(num_per_split))
  # The remainder of the dataset
  new_datasets.append(dataset.skip(num_per_split * (num_splits - 1)))

  return new_datasets


def _add_onehot_label_to_dict(d,
                              num_label):
  """Returns a new dictionary with a label_onehot key."""
  d = copy.copy(d)
  d["label_onehot"] = tf.one_hot(d["label"], num_label)
  return d


def _process_image_in_dict(d):
  """Returns a new dict with a uint8 image converted to 0-1 scaled image."""
  d = copy.copy(d)
  image = d["image"]
  if image.dtype != tf.uint8:
    raise ValueError("Only supports uint8 images")
  d["image"] = tf.cast(image, tf.float32) / 255.
  return d


@attr.s
class Datasets(object):
  train = attr.ib(Any)
  valid_inner = attr.ib(Any)
  valid_outer = attr.ib(Any)
  test = attr.ib(Any)


def get_image_datasets(
    dataset_name,
    batch_size,
    num_per_valid = 3000,
    num_train = None,
    cache_dataset = True,
    shuffle_buffer = None,
    data_dir = None,
    augmentation_fn = None,
):
  """Get an image `Datasets` instance that is ready to train with.

  This includes caching for speed, repeating, shuffling, preprocessing, and
  batching for each of the 4 splits.

  Args:
    dataset_name: Name of tfds dataset.
    batch_size: Batch size to use.
    num_per_valid: Number of validation images.
    num_train: Number of training examples to use. If None, use all.
    cache_dataset: Optionally cache the dataset for speed.
    shuffle_buffer: Size of shuffle buffer. If none, use the full train set
      size.
    data_dir: Location of tfds data_dir.
    augmentation_fn: Function to apply before batching for augmentation.

  Returns:
    `Datasets` ready to train with.
  """
  # TODO(lmetz) pin all versions of datasets so they are consistent in time.

  splits, info = tfds_load_dataset(
      dataset_name, with_info=True, data_dir=data_dir)
  num_classes = info.features["label"].num_classes

  # Some datasets have different splits defined. For meta-learning we need 4
  # splits. The following takes the splits that are defined, and tries to use
  # them when possible. For missing splits, examples are taken off of the train
  # dataset.

  if set(splits.keys()) == set(["train", "validation", "test"]):
    train = splits["train"]
    test = splits["test"]
    valid_outer = splits["validation"]

    # pylint: disable=unbalanced-tuple-unpacking
    valid_inner, train = split_dataset(
        train, num_per_split=num_per_valid, num_splits=2)
    num_test = info.splits["test"].num_examples
    total_num_train = info.splits["train"].num_examples
    num_valid = info.splits["validation"].num_examples

  elif (set(splits.keys()) == set(["train", "test"]) or
        set(splits.keys()) == set(["train", "validation"])):

    train = splits["train"]
    # pylint: disable=unbalanced-tuple-unpacking
    valid_inner, valid_outer, train = split_dataset(
        train, num_per_split=num_per_valid, num_splits=3)

    if "test" in info.splits:
      heldout_split = info.splits["test"]
    else:
      heldout_split = info.splits["validation"]
    num_test = heldout_split.num_examples

    test = splits["test"] if "test" in splits else splits["validation"]
    total_num_train = info.splits["train"].num_examples - num_per_valid * 2
    num_valid = num_per_valid

  elif set(splits.keys()) == set(["train"]):
    train = splits["train"]
    # pylint: disable=unbalanced-tuple-unpacking
    valid_inner, valid_outer, test, train = split_dataset(
        train, num_per_split=num_per_valid, num_splits=4)

    total_num_train = info.splits["train"].num_examples - num_per_valid * 3
    num_test = num_per_valid
    num_valid = num_per_valid
  else:
    raise ValueError("Unsure how to manage the following splits: %s" %
                     str(list(splits.keys())))

  if num_train:
    train = train.take(num_train)
  else:
    num_train = total_num_train

  datasets = Datasets(
      train=train, valid_inner=valid_inner, valid_outer=valid_outer, test=test)

  if cache_dataset:
    datasets = tf.nest.map_structure(lambda ds: ds.cache(), datasets)

  datasets = tf.nest.map_structure(lambda ds: ds.repeat(), datasets)

  train_shuffle = shuffle_buffer if shuffle_buffer else num_train
  valid_shuffle = shuffle_buffer if shuffle_buffer else num_valid
  test_shuffle = shuffle_buffer if shuffle_buffer else num_test

  datasets = Datasets(
      train=datasets.train.shuffle(train_shuffle),
      valid_inner=datasets.valid_inner.shuffle(valid_shuffle),
      valid_outer=datasets.valid_outer.shuffle(valid_shuffle),
      test=datasets.test.shuffle(test_shuffle))

  def pre_process(example):
    example = _add_onehot_label_to_dict(example, num_classes)
    return _process_image_in_dict(example)

  datasets = tf.nest.map_structure(lambda ds: ds.map(pre_process), datasets)

  if augmentation_fn:
    datasets = tf.nest.map_structure(lambda ds: ds.map(augmentation_fn),
                                     datasets)

  return tf.nest.map_structure(
      lambda ds: ds.batch(batch_size, drop_remainder=True), datasets)


def _random_slice(example,
                  length):
  """Extract a random slice or pad to make all sequences a fixed length.

  For example -- if one passes in [1,2,3,4] with length=2, this would return
  one of the following: [1,2], [2,3], [3,4].

  If the input is [1, 2] with length=4, this would return [1, 2, 0, 0].

  Args:
    example: Dictionary containing a single example with the "text" key. This
      "text" key should be a vector with an integer type.
    length: Length of the slice.

  Returns:
    An example containing only a fixed slice of text.
  """
  input_length = tf.shape(example["text"])[0]
  max_idx = input_length - length
  # pylint: disable=g-long-lambda
  start_idx = tf.cond(
      tf.greater(max_idx, 0), lambda: tf.random_uniform(
          [], tf.to_int32(0), tf.cast(max_idx, tf.int32), dtype=tf.int32),
      lambda: 0)
  # pylint: enable=g-long-lambda

  to_pad = tf.maximum(length - input_length, 0)
  pad_input = tf.pad(example["text"], [[0, to_pad]])
  # copy to prevent a mutation of inputs.
  example = copy.copy(example)
  example["text"] = pad_input[start_idx:start_idx + length]
  example["text"].set_shape([length])

  pad_mask = tf.pad(tf.ones([input_length]), [[0, to_pad]])
  example["mask"] = pad_mask[start_idx:start_idx + length]
  example["mask"].set_shape([length])

  return example


def random_slice_text_data(
    dataset_name,
    batch_size,
    num_train = None,
    patch_length = 128,
    num_per_valid = 3000,
    cache_dataset = False,
    shuffle_buffer = None,
):
  """Gets a text dataset ready to train on.

  This splits the dataset into 4 cross validation splits, takes a random slice
  to make all entries the same length, and batches the examples.

  Args:
    dataset_name: tensorflow_dataset's dataset name.
    batch_size: batch size.
    num_train: number of training examples. If None use all examples.
    patch_length: length of patch to extract.
    num_per_valid: number of images for each validation set.
    cache_dataset: Cache the dataset or not.
    shuffle_buffer: Shuffle buffer size. If None, use dataset size.

  Returns:
    Datasets object containing tf.Dataset.
  """

  train, info = tfds_load_dataset(
      dataset_name, split="train", with_info=True, shuffle_files=True)
  total_num_train = info.splits["train"].num_examples
  num_test = info.splits["test"].num_examples

  # pylint: disable=unbalanced-tuple-unpacking
  valid_inner, valid_outer, train = split_dataset(
      train, num_per_split=num_per_valid)
  # pylint: enable=unbalanced-tuple-unpacking
  if num_train:
    train = train.take(num_train)

  test = tfds_load_dataset(dataset_name, split="test", shuffle_files=True)

  datasets = Datasets(
      train=train, valid_inner=valid_inner, valid_outer=valid_outer, test=test)

  if cache_dataset:
    datasets = tf.nest.map_structure(lambda ds: ds.cache(), datasets)

  datasets = tf.nest.map_structure(lambda ds: ds.repeat(), datasets)

  train_shuffle = shuffle_buffer if shuffle_buffer else total_num_train - num_per_valid * 2
  valid_shuffle = shuffle_buffer if shuffle_buffer else num_per_valid
  test_shuffle = shuffle_buffer if shuffle_buffer else num_test

  datasets = Datasets(
      train=datasets.train.shuffle(train_shuffle),
      valid_inner=datasets.valid_inner.shuffle(valid_shuffle),
      valid_outer=datasets.valid_outer.shuffle(valid_shuffle),
      test=datasets.test.shuffle(test_shuffle))

  def pre_process(example):
    """Preprocess example by adding onehot label, and taking a random slice."""
    if "label" in info.features:
      num_classes = info.features["label"].num_classes
      example = _add_onehot_label_to_dict(example, num_classes)
    return _random_slice(example, patch_length)

  datasets = tf.nest.map_structure(lambda ds: ds.map(pre_process), datasets)
  return tf.nest.map_structure(
      lambda ds: ds.batch(batch_size, drop_remainder=True), datasets)


class ResizedDataset(tfds.core.GeneratorBasedBuilder):
  """Base class for a resized image tensorflow dataset."""

  def __init__(self, parent_builder,
               size, *args, **kwargs):
    """Initialize the resized image dataset builder.

    Args:
      parent_builder: The builder to build the resized image dataset from.
      size: size to resize each example to.
      *args: args passed super class.
      **kwargs: kwargs passed super class.
    """

    parent_builder.download_and_prepare()
    self._builder = parent_builder
    self._size = size
    super(ResizedDataset, self).__init__(*args, **kwargs)

  def _info(self):
    info = self._builder.info
    description = "\n This dataset has been resized to %dx%d!" % (self._size[0],
                                                                  self._size[1])

    new_feature_dict = {k: v for k, v in info.features.items()}
    new_feature_dict["image"] = tfds.features.Image(
        shape=list(self._size) + [3])

    return tfds.core.DatasetInfo(
        builder=self,
        description=info.description + description,
        homepage=info.homepage,
        features=tfds.features.FeaturesDict(new_feature_dict),
        supervised_keys=info.supervised_keys,
        citation=info.citation)

  def _split_generators(self, dl_manager):
    return [
        tfds.core.SplitGenerator(
            name=split, gen_kwargs=dict(split=split))
        for split in self._builder.info.splits.keys()
    ]

  def _generate_examples(self, split):
    for exi, ex in enumerate(
        tfds.as_numpy(self._builder.as_dataset(split=split))):
      ex = self._process_example(ex)
      yield exi, ex

  def _process_example(self, example):
    # As of now, this simply converts the image to the passed in size.
    # TODO(lmetz) It might also make sense to resize then crop out the center.
    example["image"] = cv2.resize(
        example["image"], dsize=self._size, interpolation=cv2.INTER_CUBIC)
    return example


class Food101_32x32(ResizedDataset):  # pylint: disable=invalid-name
  """The Food101 dataset resized to be 32x32."""

  VERSION = "1.0.0"

  def __init__(self, *args, **kwargs):
    parent_builder = tfds.builder("food101", version="1.0.0")
    super(Food101_32x32, self).__init__(
        *args, parent_builder=parent_builder, size=(32, 32), **kwargs)


class Food101_64x64(ResizedDataset):  # pylint: disable=invalid-name
  """The Food101 dataset resized to be 64x64."""

  VERSION = "1.0.0"

  def __init__(self, *args, **kwargs):
    parent_builder = tfds.builder("food101", version="1.0.0")
    super(Food101_64x64, self).__init__(
        *args, parent_builder=parent_builder, size=(64, 64), **kwargs)


class Coil100_32x32(ResizedDataset):  # pylint: disable=invalid-name
  """The coil100 dataset resized to be 32x32."""

  VERSION = "1.0.0"

  def __init__(self, *args, **kwargs):
    parent_builder = tfds.builder("coil100", version="1.0.0")
    super(Coil100_32x32, self).__init__(
        *args, parent_builder=parent_builder, size=(32, 32), **kwargs)


class ColorectalHistology_32x32(ResizedDataset):  # pylint: disable=invalid-name
  """The colorectal_histology dataset resized to be 32x32."""

  VERSION = "1.0.0"

  def __init__(self, *args, **kwargs):
    parent_builder = tfds.builder("colorectal_histology", version="2.*.*")
    super(ColorectalHistology_32x32, self).__init__(
        *args, parent_builder=parent_builder, size=(32, 32), **kwargs)


class DeepWeeds_32x32(ResizedDataset):  # pylint: disable=invalid-name
  """The deep_weeds dataset resized to be 32x32."""

  VERSION = "1.0.0"

  def __init__(self, *args, **kwargs):
    parent_builder = tfds.builder("deep_weeds", version="1.0.0")
    super(DeepWeeds_32x32, self).__init__(
        *args, parent_builder=parent_builder, size=(32, 32), **kwargs)


class Sun397_32x32(ResizedDataset):  # pylint: disable=invalid-name
  """The sun397/tfds dataset resized to be 32x32."""

  VERSION = "1.0.0"

  def __init__(self, *args, **kwargs):
    parent_builder = tfds.builder("sun397/tfds", version="4.0.0")
    super(Sun397_32x32, self).__init__(
        *args, parent_builder=parent_builder, size=(32, 32), **kwargs)


class TokenizedConfig(tfds.core.BuilderConfig):
  """BuilderConfig for tokenized text datasets."""

  def __init__(self, version=None, text_encoder_config=None, **kwargs):
    """BuilderConfig for tokenized text datasets.

    Args:
      version (string): version as string.
      text_encoder_config: `tfds.deprecated.text.TextEncoderConfig`, configuration
        for the `tfds.deprecated.text.TextEncoder` used for the `"text"` feature.
      **kwargs: keyword arguments forwarded to super.
    """
    super(TokenizedConfig, self).__init__(
        version=tfds.core.Version(version), **kwargs)
    self.text_encoder_config = (
        text_encoder_config or tfds.deprecated.text.TextEncoderConfig())


# This is an arbitrarily chosen subset of languages.
WIKIPEDIA_PREFIX = [
    "20190301.zh", "20190301.ru", "20190301.ja", "20190301.hsb", "20190301.en"
]


def _get_builder_configs(base_configs):
  """Get the builder configs for tokenized datasets."""
  configs = []
  for prefix in base_configs:
    configs.append(
        TokenizedConfig(
            name="%s_bytes" % prefix,
            version="0.0.1",
            description=("Uses byte-level text encoding with "
                         "`tfds.deprecated.text.ByteTextEncoder`"),
            text_encoder_config=tfds.deprecated.text.TextEncoderConfig(
                encoder=tfds.deprecated.text.ByteTextEncoder()),
        ))
    configs.append(
        TokenizedConfig(
            name="%s_subwords8k" % prefix,
            version="0.0.1",
            description=("Uses `tfds.deprecated.text.SubwordTextEncoder` with 8k "
                         "vocab size"),
            text_encoder_config=tfds.deprecated.text.TextEncoderConfig(
                encoder_cls=tfds.deprecated.text.SubwordTextEncoder,
                vocab_size=8192),
        ))
  return configs


class TokenizedWikipedia(tfds.core.GeneratorBasedBuilder):
  """Builder which tokenizes the tfds wikipedia datasets.

  This dataset returns 1 paragraph (split via new line) per example
  extracted from the articles. We additionally filter examples to have more than
  5 bytes. Encoding is either bytes, or subwords. The vocab is constructed out
  of the first 200k examples. While this is likely not perfect this should be
  sufficient for meta-learning optimizers.

  Additionally, we make a train and test split by hashing the article seed.

  Finally, for computational reasons we only use 1 millon articles. For the size
  of the models we are training here this should be plenty.
  """
  BUILDER_CONFIGS = _get_builder_configs(WIKIPEDIA_PREFIX)

  def __init__(self, config=None, **kwargs):
    """Initialize the resized image dataset builder.

    Args:
      config: str Config string specified to build dataset with.
      **kwargs: kwargs passed super class.
    """

    # extract the base dataset.
    base, _ = config.split("_")
    self._builder = tfds.builder("wikipedia/%s" % base)
    super(TokenizedWikipedia, self).__init__(config=config, **kwargs)

    self._perc_train = 0.7
    self._max_num_articles = 1000000
    # Number of examples used to build the tokenizer.
    self._examples_for_tokenizer = 200000

  def _info(self):
    info = self._builder.info
    description = "\n This dataset has been tokenized!"
    return tfds.core.DatasetInfo(
        builder=self,
        description=info.description + description,
        features=tfds.features.FeaturesDict({
            "title":
                tfds.features.Text(),
            "text":
                tfds.features.Text(
                    encoder_config=self.builder_config.text_encoder_config),
        }),
        supervised_keys=("text", "text"),
        homepage=info.homepage,
        citation=info.citation)

  def _split_generators(self, dl_manager):
    self.info.features["text"].maybe_build_from_corpus(self._vocab_text_gen())

    return [
        tfds.core.SplitGenerator(
            name=split, gen_kwargs=dict(split=split))
        for split in ["train", "test"]
    ]

  def _split_article(self, ex):
    for i, split in enumerate(ex["text"].split("\n")):
      if len(split.strip()) > 5:
        yield i, {"title": ex["title"], "text": split}

  def _generate_examples(self, split):
    hasher = tfds.core.hashing.Hasher("token_wikipedia_salt")
    for exi, example in enumerate(
        tfds.as_numpy(self._builder.as_dataset(split="train"))):

      if exi > self._max_num_articles:
        return

      # To make a train test split we first hash the key and convert it to a
      # floating point value between 0-1. Depending on this value we either
      # yield the example or not depending on the split.
      p = hasher.hash_key(exi) % 100000 / 100000.

      if split == "train" and p < self._perc_train:
        for i, sub_example in self._split_article(example):
          key = (exi, i)
          yield key, sub_example

      elif split == "test" and p >= self._perc_train:
        for i, sub_example in self._split_article(example):
          key = (exi, i)
          yield key, sub_example

  def _vocab_text_gen(self):
    for i, (_, ex) in enumerate(self._generate_examples("train")):
      # Only yield a subset of the data used for tokenization for
      # performance reasons.
      if self._examples_for_tokenizer > i:
        yield ex["text"]
      else:
        return


# Arbitrary subset of datasets.
AMAZON_PRODUCTS = ["Books_v1_02", "Camera_v1_00", "Home_v1_00", "Video_v1_00"]


class TokenizedAmazonReviews(tfds.core.GeneratorBasedBuilder):
  """Builder which tokenizes the tfds amazon reviews datasets.

  For compute reasons we only tokenize with 200000 examples.

  We make a train and test split by hashing the example index.
  """
  BUILDER_CONFIGS = _get_builder_configs(AMAZON_PRODUCTS)

  def __init__(self, config=None, **kwargs):
    """Initialize the resized image dataset builder.

    Args:
      config: str Config string specified to build dataset with.
      **kwargs: kwargs passed super class.
    """

    # extract the base dataset.
    base = "_".join(config.split("_")[0:-1])
    self._builder = tfds.builder("amazon_us_reviews/%s" % base)

    super(TokenizedAmazonReviews, self).__init__(config=config, **kwargs)

    self._perc_train = 0.7
    self._examples_for_tokenizer = 200000

  def _info(self):
    info = self._builder.info
    description = "\n This dataset has been tokenized!"
    return tfds.core.DatasetInfo(
        builder=self,
        description=info.description + description,
        features=tfds.features.FeaturesDict({
            # 1-5 stars are the labels.
            "label":
                tfds.features.ClassLabel(num_classes=5),
            "text":
                tfds.features.Text(
                    encoder_config=self.builder_config.text_encoder_config),
        }),
        supervised_keys=("text", "label"),
        homepage=info.homepage,
        citation=info.citation)

  def _split_generators(self, dl_manager):
    self.info.features["text"].maybe_build_from_corpus(self._vocab_text_gen())

    return [
        tfds.core.SplitGenerator(
            name=split, gen_kwargs=dict(split=split))
        for split in ["train", "test"]
    ]

  def _generate_examples(self, split):
    hasher = tfds.core.hashing.Hasher("token_wikipedia_salt")
    for exi, example in enumerate(
        tfds.as_numpy(self._builder.as_dataset(split="train"))):

      p = hasher.hash_key(exi) % 1000 / 1000.

      example = {
          "text": example["data"]["review_body"],
          # subtract one to zero index.
          "label": example["data"]["star_rating"] - 1
      }
      if split == "train" and p < self._perc_train:
        yield exi, example

      elif split == "test" and p > self._perc_train:
        yield exi, example

  def _vocab_text_gen(self):
    for i, (_, ex) in enumerate(self._generate_examples("train")):
      if self._examples_for_tokenizer > i:
        yield ex["text"]
      else:
        return


def _single_associative_retrieval(batch_size=128, num_pairs=5, num_tokens=10):
  """See associative_retrieval."""

  def _onehot_pack(inp, out, loss_mask):
    inp_seq, outputs, loss_mask = (tf.one_hot(inp, num_tokens + 2),
                                   tf.one_hot(out, num_tokens + 2), loss_mask)
    return {"input": inp_seq, "output": outputs, "loss_mask": loss_mask}

  def _py_make_example():
    """Iterator that makes single examples in python."""
    while True:
      keys = np.random.choice(num_tokens, size=num_pairs, replace=False)
      values = np.random.choice(num_tokens, size=num_pairs, replace=True)
      empty_token_idx = num_tokens
      query_token_idx = num_tokens + 1
      input_seq = []
      output_seq = []
      for k, v in zip(keys, values):
        input_seq.extend([k, v])
        output_seq.extend([empty_token_idx, empty_token_idx])

      input_seq.append(query_token_idx)
      output_seq.append(empty_token_idx)

      query_key = np.random.randint(0, num_pairs)
      input_seq.append(keys[query_key])
      output_seq.append(values[query_key])
      loss_mask = np.zeros(2 * num_pairs + 2, dtype=np.float32)
      loss_mask[-1] = 1.
      input_seq = np.asarray(input_seq, dtype=np.int32)
      output_seq = np.asarray(output_seq, dtype=np.int32)
      yield input_seq, output_seq, loss_mask

  # per pair, there is a key and a value. Extra 2 account for query indicator
  # and query key.
  seq_len = 2 * num_pairs + 2
  dataset = tf.data.Dataset.from_generator(_py_make_example,
                                           (tf.int32, tf.int32, tf.float32),
                                           ([seq_len], [seq_len], [seq_len]))
  dataset = dataset.map(_onehot_pack)
  return dataset.batch(batch_size, drop_remainder=True)


def associative_sequence(batch_size=128, num_pairs=5, num_tokens=10):
  """Associative Retrieval datasets.

  The inputs consist of pairs of key and value sequentially followed by an
  indicator token and then a retrieval token.

  Output consists of the value associated with the retrieval key in the final
  step of the sequence, preceded by empty tokens.

  The problem can be perfectly solved, as in the 'key' tokens will be unique.
  There can be duplicate values, however, for different keys.

  Example (using characters instead of the onehot representations):

  input:     A1B2C3D4?A
  output:    _________1
  loss_mask: 0000000001

  The outputs are represented using a one-hot encoding.

  The problem is based off of the one used in
  https://arxiv.org/pdf/1610.06258.pdf.

  Args:
    batch_size: int
    num_pairs: int, number of pairs to put into memory.
    num_tokens: int, number of possible tokens to choose from.

  Returns:
    datasets: Datasets object with each split containing the same data
      generating process.
  """
  fn = lambda: _single_associative_retrieval(batch_size, num_pairs, num_tokens)
  return Datasets(train=fn(), valid_inner=fn(), valid_outer=fn(), test=fn())


def _single_copy_sequence(batch_size=128,
                          sequence_length=5,
                          num_separator=1,
                          num_tokens=10):
  """See copy_sequence for docs."""

  def _build_batch(_):
    """Construct a batch.

    Args:
      _: tf.Tensor Needed to construct a tf.data.Dataset that iteratively calls
        this function. This is a dummy value that never changes.

    Returns:
      batch: SequencePrediction, containing a batch of sequences.
    """
    inp = tf.random_uniform([batch_size, sequence_length],
                            0,
                            num_tokens,
                            dtype=tf.int32)
    sep = tf.ones([batch_size, num_separator], dtype=tf.int32) * num_tokens
    emit = tf.ones([batch_size, sequence_length], dtype=tf.int32) * (
        num_tokens + 1)
    inp_seq_pre_onehot = tf.concat([inp, sep, emit], axis=1)
    inp_seq = tf.one_hot(inp_seq_pre_onehot, num_tokens + 2)

    loss_mask = tf.concat([
        tf.zeros([batch_size, sequence_length + num_separator]),
        tf.ones([batch_size, sequence_length])
    ],
                          axis=1)

    outputs_pre_onehot = tf.concat(
        [tf.zeros_like(inp), tf.zeros_like(sep), inp], axis=1)
    outputs = tf.one_hot(outputs_pre_onehot, num_tokens + 2)

    return {"input": inp_seq, "output": outputs, "loss_mask": loss_mask}

  return tf.data.Dataset.from_tensor_slices([0]).repeat().map(_build_batch)


def copy_sequence(batch_size=128,
                  sequence_length=5,
                  num_separator=1,
                  num_tokens=10):
  """A simple input copy to output task.

  Input consists of `seq_len` tokens drawn from a vocab size of `num_tokens`
  followed by `n_sep` separation tokens, followed by 3 empty tokens.

  The output consists of `seq_len + n_sep` empty tokens followed by the same
  input tokens from the input.

  All token outputs are onehot.

  A sample input output pair for seq_len=3, num_tokens=3, n_sep=1

  input::        <tokenA><tokenB><tokenC><sep>  <empty> <empty> <empty>
  output::       <empty> <empty> <empty> <empty><tokenA><tokenB><tokenC>
  loss_mask::  0.       0.     0.      0.     1.      1.      1.

  Args:
    batch_size: int
    sequence_length: int, length of sequence to copy
    num_separator: int, number of empty tokens separating input from output
    num_tokens: int, number of tokens to build input from

  Returns:
    dataset: tf.Data.Dataset
  """

  def fn():
    return _single_copy_sequence(batch_size, sequence_length, num_separator,
                                 num_tokens)

  return Datasets(train=fn(), valid_inner=fn(), valid_outer=fn(), test=fn())
