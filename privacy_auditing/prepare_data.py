# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

# NOTE: Set ENRON_TRAIN_PATH and ENRON_TEST_PATH environment variables.
"""Load and format the Enron dataset."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import os
import threading
from typing import Any, TypeAlias

from absl import logging
import grain
import grain.python as pygrain
import numpy as np
from privacy_auditing import filesystem
from privacy_auditing import poisson_dataset
import tunix


def get_bytes_from_np_array(arr):
  """Returns bytes from a numpy array.

  Args:
    arr: A numpy array. Must contain at most one element.

  Returns:
    The bytes contained in the array, or b'' if the array is empty.

  Raises:
    ValueError: If the array contains more than one element.
  """
  return b'' if arr.size == 0 else arr.item()


def format_enron_message_to_string(
    subject, body, it_formatting = False
):
  """Formats subject and body into a string for Enron messages."""
  target_bytes = b'Subject:\n' + subject + b'\nBody:\n' + body
  if it_formatting:
    target_bytes = target_bytes + b'<end_of_turn>'
  return target_bytes.decode('utf-8')


@dataclasses.dataclass(frozen=True)
class PoissonConfig:
  """Configuration for Poisson sampling.

  Attributes:
    training_steps: The number of batches formed by Poisson sampling, one used
      per training step.
    expected_batch_size: The expected batch size. The maximum batch size is
      specified the same way as the batch size for training without Poisson
      sampling.
    seed: An option fixed seed to use for Poisson sampling for reproducibility.
  """

  training_steps: int
  expected_batch_size: float
  seed: int | None = None


def get_input_prompt(it_formatting):
  """Returns the input prompt, optionally with IT formatting."""
  inputs = 'Write an email thread with 1 message.'
  if it_formatting:
    inputs = (
        f'<start_of_turn>user\n{inputs}<end_of_turn>\n<start_of_turn>model\n'
    )
  return inputs


class _FormatTokenize(grain.transforms.Map):
  """Format and tokenize a dataset."""

  def __init__(
      self,
      *,
      it_formatting = False,
      tokenizer_initializer,
  ):
    self._it_formatting = it_formatting
    self._tokenizer_initializer = tokenizer_initializer
    self._tokenizer = None
    self._initializer_lock = threading.Lock()

  def _parse_example(self, example):
    return pygrain.fast_proto.parse_tf_example_experimental(example)

  def _format_example(self, subject, body):
    return {
        'inputs': get_input_prompt(self._it_formatting),
        'targets': format_enron_message_to_string(
            subject, body, self._it_formatting
        ),
    }

  def _tokenize_inputs(
      self, features
  ):
    with self._initializer_lock:
      # Only load the tokenizer once even if the transform is being
      # used multi-threaded.
      if self._tokenizer is None:
        logging.info('Initializing tokenizer')
        self._tokenizer = self._tokenizer_initializer()
    src_tokens = np.array(
        [self._tokenizer.bos_id()] + self._tokenizer.encode(features['inputs'])
    )
    dst_tokens = np.array(
        self._tokenizer.encode(features['targets']) + [self._tokenizer.eos_id()]
    )
    return src_tokens, dst_tokens

  # Skip the tokenizer and the threading lock when pickling the transform,
  # they are not picklable. Instead we'll require they were initialized on
  # construction in __init__.
  def __getstate__(self):
    state = self.__dict__.copy()
    del state['_tokenizer']
    del state['_initializer_lock']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._tokenizer = None
    self._initializer_lock = threading.Lock()


class _FormatTokenizeCanaries(_FormatTokenize):
  """Format and tokenize the canary dataset."""

  def __init__(
      self,
      tokenizer_initializer,
      max_seq_len,
      it_formatting = False,
  ):
    super().__init__(
        tokenizer_initializer=tokenizer_initializer, it_formatting=it_formatting
    )
    self._max_seq_len = max_seq_len

  def _format_canaries(self, features):
    subject = b'0'
    body = get_bytes_from_np_array(features['BODY_NORMALIZED_LIST'])
    return self._format_example(subject, body)

  def map(self, element):
    element = self._parse_example(element)
    element = self._format_canaries(element)
    src_tokens, dst_tokens = self._tokenize_inputs(element)
    if len(src_tokens) >= self._max_seq_len:
      raise ValueError(
          f'Source tokens are too long: {len(src_tokens)=} >='
          f' {self._max_seq_len=}. Choose a larger max_seq_len.'
      )
    dst_tokens = dst_tokens[: self._max_seq_len - len(src_tokens)]
    return src_tokens, dst_tokens


class _FormatTokenizeEnron(_FormatTokenize):
  """Format and tokenize the Enron dataset."""

  def _format_enron(self, features):
    subject = get_bytes_from_np_array(features['subject'])
    body = get_bytes_from_np_array(features['body'])
    return self._format_example(subject, body)

  def map(self, element):
    element = self._format_enron(element)
    src_tokens, dst_tokens = self._tokenize_inputs(element)
    return src_tokens, dst_tokens


class _FormatTokenizeSynthetic(_FormatTokenize):
  """Format and tokenize the synthetic dataset."""

  def _format_synthetic(self, body):
    subject = b'0'
    return self._format_example(subject, body)

  def map(self, element):
    element = self._format_synthetic(element)
    src_tokens, dst_tokens = self._tokenize_inputs(element)
    return src_tokens, dst_tokens


class _BuildTrainInput(grain.transforms.Map):
  """Build a TrainingInput from a tuple of source and destination tokens."""

  def __init__(self, max_seq_len, pad_value):
    self._max_seq_len = max_seq_len
    self._pad_value = pad_value

  def map(
      self, tokens
  ):
    src_tokens, dst_tokens = tokens
    src_tokens = np.array(src_tokens)
    dst_tokens = np.array(dst_tokens)

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = np.concat([src_tokens, dst_tokens], axis=0)

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    q_mask = np.zeros_like(src_tokens, dtype=np.bool)
    a_mask = np.ones_like(dst_tokens, dtype=np.bool)
    mask = np.concat([q_mask, a_mask], axis=0)

    # If the input tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._truncate_or_pad_up_to_max_len(tokens, self._pad_value)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._truncate_or_pad_up_to_max_len(mask, 0)

    return tunix.sft.peft_trainer.TrainingInput(
        input_tokens=tokens, input_mask=mask
    )

  def _truncate_or_pad_up_to_max_len(
      self, input_tensor, pad_value
  ):
    """Truncate or pad the given tensor to the max sequence length."""
    input_tensor = input_tensor[: self._max_seq_len]
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self._max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode='constant',
        constant_values=pad_value,
    )


def load_canaries(
    canary_path,
    seen,
):
  """Load the canaries from the given base path.

  Args:
    canary_path: The base path to the canary dataset, without the
      `.seen.arrayrecord*` / `.unseen.arrayrecord*` suffix.
    seen: Whether to load the seen or unseen split.

  Returns:
    An `ArrayRecordDataSource` for the requested canary split.

  Raises:
    FileNotFoundError: If no canary files are found for the given path.
  """
  suffix = 'seen.arrayrecord' if seen else 'unseen.arrayrecord'
  canary_base_path = f'{canary_path}.{suffix}*'
  logging.info('canary_base_path: %s', canary_base_path)
  canary_paths = sorted(filesystem.glob(canary_base_path))
  logging.info('canary_paths: %s', canary_paths)
  if not canary_paths:
    raise FileNotFoundError(f'No canary files matched: {canary_base_path}')
  return pygrain.ArrayRecordDataSource(canary_paths)


_SourceDataset: TypeAlias = (
    pygrain.BagDataSource
    | pygrain.ArrayRecordDataSource
    | pygrain.TfExampleBagDataSource
)


def _format_dataset(
    source_ds,
    format_fn,
    shuffle = False,
    seed = None,
):
  """Formats the dataset for training."""
  ds = grain.MapDataset.source(source_ds)
  if seed is not None:
    ds = ds.seed(seed=seed)
  if shuffle:
    ds = ds.shuffle()
  return ds.map(format_fn)


def get_canary_ds(
    canary_path,
    tokenizer_initializer,
    max_seq_len,
    num_unique_canaries,
    seen,
    it_formatting = False,
):
  """Get the canary dataset."""
  canary_ds = load_canaries(canary_path, seen=seen)
  format_fn = _FormatTokenizeCanaries(
      tokenizer_initializer, max_seq_len, it_formatting
  )
  canary_ds = _format_dataset(canary_ds, format_fn, seed=43)
  canary_ds = canary_ds[:num_unique_canaries]
  return canary_ds


def get_pooled_canary_ds(
    canary_path,
    tokenizer_initializer,
    max_seq_len,
    num_unique_canaries,
    it_formatting = False,
):
  """Creates the pooled (seen + unseen) canary dataset.

  Args:
    canary_path: The base path to the canary dataset, without the
      `.seen.arrayrecord*` / `.unseen.arrayrecord*` suffix.
    tokenizer_initializer: A no-arg callable that returns a tokenizer.
    max_seq_len: The maximum sequence length for the tokenized examples.
    num_unique_canaries: The number of unique canaries to use.
    it_formatting: Whether to use IT formatting.

  Returns:
    A `grain.MapDataset` containing both seen and unseen canaries.
  """
  seen_ds = get_canary_ds(
      canary_path,
      tokenizer_initializer,
      max_seq_len,
      num_unique_canaries,
      seen=True,
      it_formatting=it_formatting,
  )
  unseen_ds = get_canary_ds(
      canary_path,
      tokenizer_initializer,
      max_seq_len,
      num_unique_canaries,
      seen=False,
      it_formatting=it_formatting,
  )
  return grain.MapDataset.concatenate([seen_ds, unseen_ds])


class SubsetDataSource:
  """A MapDataset source that selects elements by an explicit index list.

  Attributes:
    ds: The underlying `grain.MapDataset` to sample from.
    indices: A list of indices to select from the underlying dataset.
  """

  def __init__(self, ds, indices):
    self.ds = ds
    self.indices = list(indices)

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    return self.ds[int(self.indices[idx])]


def combine_canaries_with_train(
    canary_ds,
    train_ds,
    canary_reps,
    seed,
):
  """Combines the canary dataset with the training dataset.

  The training dataset is truncated to make space for the canaries. The combined
  dataset is then shuffled.

  Args:
    canary_ds: The dataset containing the canaries.
    train_ds: The original training dataset.
    canary_reps: The number of times to repeat each canary.
    seed: The random seed used for shuffling the combined dataset.

  Returns:
    A `grain.MapDataset` containing the training data with canaries inserted.

  Raises:
    ValueError: If the size of the canary dataset exceeds 25% of the training
      dataset size.
  """
  canary_ds = canary_ds.repeat(canary_reps)

  # Ensure that the canary dataset is not too large.
  if len(canary_ds) / len(train_ds) > 0.25:
    raise ValueError(
        'Canary dataset cannot exceed 25% of the training dataset.'
        f' Currently {len(canary_ds)=}/{len(train_ds)=} = '
        f'{len(canary_ds) / len(train_ds)}. Consider reducing'
        ' canary_reps.'
    )

  # Truncate the training dataset to account for canary replacement.
  train_ds = train_ds[: -len(canary_ds)]

  # Concatenate the training and canary datasets and shuffle.
  combined_ds = grain.MapDataset.concatenate([train_ds, canary_ds])
  combined_ds = combined_ds.shuffle(seed=seed)
  return combined_ds


def _build_data_loader(
    data_source,
    num_records,
    num_epochs,
    tokenizer,
    max_seq_len,
    batch_size,
    grain_worker_count,
    grain_worker_buffer_size,
):
  """Creates a `grain.DataLoader` for the given dataset."""
  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.samplers.IndexSampler(
          num_records=num_records,
          num_epochs=num_epochs,
          shard_options=grain.sharding.NoSharding(),
      ),
      operations=[
          _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
          grain.transforms.Batch(batch_size=batch_size, drop_remainder=True),
      ],
      worker_count=grain_worker_count,
      worker_buffer_size=grain_worker_buffer_size,
      read_options=pygrain.ReadOptions(num_threads=4),
  )


def create_datasets(
    train_batch_size,
    validation_batch_size,
    num_epochs,
    tokenizer_initializer,
    max_seq_len,
    num_unique_canaries,
    canary_path = None,
    canary_reps = 1,
    poisson_config = None,
    it_formatting = False,
    *,
    data_shuffle_seed = 44,
    grain_worker_count = 4,
):
  """Creates Grain DataLoaders for the Enron dataset.

  Args:
    train_batch_size: The batch size for the training DataLoader.
    validation_batch_size: The batch size for the validation DataLoader. If <=
      0, no validation DataLoader is created.
    num_epochs: The number of epochs to repeat the dataset. If None, the dataset
      will repeat indefinitely. Ignored if Poisson sampling is enabled.
    tokenizer_initializer: A no-arg callable that returns a tokenizer.
    max_seq_len: The maximum sequence length for the tokenized examples.
    num_unique_canaries: The number of unique canaries to use.
    canary_path: The base path to the canary dataset. If None, no canaries are
      used.
    canary_reps: The number of canary repetitions to use.
    poisson_config: If not None, Poisson sampling is enabled and the options
      will be used to configure Poisson sampling.
    it_formatting: Whether to use IT formatting.
    data_shuffle_seed: The random seed for shuffling the data.
    grain_worker_count: The number of processesto use for reading the dataset.

  Returns:
    A tuple containing the training DataLoader and the validation DataLoader.
  """
  grain_worker_buffer_size = 128 if poisson_config is not None else 16

  train_ds = pygrain.TfExampleBagDataSource(
      os.environ.get('ENRON_TRAIN_PATH', '<PATH_TO_ENRON_TRAIN_BAGZ>'),
      # Dataset is ~407MB.
      memory_cache_size=600 * 1024 * 1024,
  )
  format_fn = _FormatTokenizeEnron(
      it_formatting=it_formatting,
      tokenizer_initializer=tokenizer_initializer,
  )
  # Note: We shuffle the training dataset with a fixed seed before Enron is
  # mixed with the canary dataset. This is to ensure that the canary
  # replacement is deterministic across multiple runs.
  train_ds = _format_dataset(train_ds, format_fn, shuffle=True, seed=42)

  if canary_path is not None:
    canary_ds = get_canary_ds(
        canary_path,
        tokenizer_initializer,
        max_seq_len,
        num_unique_canaries,
        seen=True,
        it_formatting=it_formatting,
    )
    train_ds = combine_canaries_with_train(
        canary_ds, train_ds, canary_reps, data_shuffle_seed
    )

  tokenizer = tokenizer_initializer()
  if poisson_config is not None:
    # PoissonMapDataset assumes that the last example of the parent dataset is a
    # zero-gradient example. We use a tuple of empty arrays of tokens.
    padding_example = (
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )
    padding_ds = grain.MapDataset.source([padding_example])
    train_ds = grain.MapDataset.concatenate([train_ds, padding_ds])

    train_ds = poisson_dataset.PoissonMapDataset(
        parent=train_ds,
        train_batch_size=train_batch_size,
        expected_batch_size=poisson_config.expected_batch_size,
        training_steps=poisson_config.training_steps,
        seed=poisson_config.seed,
    )
    train_loader = _build_data_loader(
        train_ds,
        num_records=poisson_config.training_steps * train_batch_size,
        num_epochs=1,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=train_batch_size,
        grain_worker_count=grain_worker_count,
        grain_worker_buffer_size=grain_worker_buffer_size,
    )
  else:
    train_loader = _build_data_loader(
        train_ds,
        num_records=len(train_ds),
        num_epochs=num_epochs,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=train_batch_size,
        grain_worker_count=grain_worker_count,
        grain_worker_buffer_size=grain_worker_buffer_size,
    )

  if validation_batch_size > 0:
    validation_ds = pygrain.TfExampleBagDataSource(
        os.environ.get('ENRON_TEST_PATH', '<PATH_TO_ENRON_TEST_BAGZ>'),
        # Dataset is ~46MB.
        memory_cache_size=100 * 1024 * 1024,
    )
    validation_ds = _format_dataset(validation_ds, format_fn)
    validation_loader = _build_data_loader(
        validation_ds,
        num_records=len(validation_ds),
        num_epochs=1,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=validation_batch_size,
        grain_worker_count=grain_worker_count,
        grain_worker_buffer_size=grain_worker_buffer_size,
    )
  else:
    validation_loader = None

  return train_loader, validation_loader


def create_synthetic_dataset(
    file_path,
    train_batch_size,
    num_epochs,
    tokenizer_initializer,
    max_seq_len,
    it_formatting = False,
    *,
    grain_worker_count = 4,
):
  """Creates Grain DataLoaders for the Enron dataset.

  Args:
    file_path: The path to the synthetic dataset.
    train_batch_size: The batch size for the training DataLoader.
    num_epochs: The number of epochs to repeat the dataset.
    tokenizer_initializer: A no-arg callable that returns a tokenizer.
    max_seq_len: The maximum sequence length for the tokenized examples.
    it_formatting: Whether to use IT formatting.
    grain_worker_count: The number of processesto use for reading the dataset.

  Returns:
    A tuple containing the training DataLoader and the total number of steps.
  """
  grain_worker_buffer_size = 128

  train_ds = pygrain.BagDataSource(file_path)
  format_fn = _FormatTokenizeSynthetic(
      it_formatting=it_formatting,
      tokenizer_initializer=tokenizer_initializer,
  )
  train_ds = _format_dataset(train_ds, format_fn, shuffle=True, seed=42)

  tokenizer = tokenizer_initializer()

  train_loader = _build_data_loader(
      train_ds,
      num_records=len(train_ds),
      num_epochs=num_epochs,
      tokenizer=tokenizer,
      max_seq_len=max_seq_len,
      batch_size=train_batch_size,
      grain_worker_count=grain_worker_count,
      grain_worker_buffer_size=grain_worker_buffer_size,
  )

  num_steps = (len(train_ds) // train_batch_size) * num_epochs
  return train_loader, num_steps


def create_audit_datasets(
    tokenizer_initializer,
    canary_path,
    max_seq_len,
    num_unique_canaries,
    it_formatting = False,
):
  """Creates Grain DataLoaders for the Enron canary dataset."""
  seen_ds = get_canary_ds(
      canary_path,
      tokenizer_initializer,
      max_seq_len,
      num_unique_canaries,
      seen=True,
      it_formatting=it_formatting,
  )
  logging.info('Read %d seen canaries', len(seen_ds))
  unseen_ds = get_canary_ds(
      canary_path,
      tokenizer_initializer,
      max_seq_len,
      num_unique_canaries,
      seen=False,
      it_formatting=it_formatting,
  )
  logging.info('Read %d unseen canaries', len(unseen_ds))
  seen_loader = grain.DataLoader(
      data_source=seen_ds,
      sampler=grain.samplers.IndexSampler(
          num_records=len(seen_ds),
          num_epochs=1,
          shard_options=grain.sharding.NoSharding(),
      ),
  )
  unseen_loader = grain.DataLoader(
      data_source=unseen_ds,
      sampler=grain.samplers.IndexSampler(
          num_records=len(unseen_ds),
          num_epochs=1,
          shard_options=grain.sharding.NoSharding(),
      ),
  )
  return seen_loader, unseen_loader
