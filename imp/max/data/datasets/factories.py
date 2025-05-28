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

"""All custom factory modules."""

import abc
import copy
import functools
import os
from typing import Any, Iterable

from absl import logging
from dmvr import builders
from dmvr import video_dataset as dmvr_base
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds

from imp.max.core import constants
from imp.max.data import config as data_config
from imp.max.data import loading
from imp.max.data import processing
from imp.max.data import tokenizers
from imp.max.data.datasets import seqio_tasks  # pylint: disable=unused-import
from imp.max.data.datasets import seqio_utils


DataFeatureName = constants.DataFeatureName
Modality = constants.Modality
ParsingFeatureName = constants.ParsingFeatureName


# ----------------------------------------------------------------------
# --------------------------- Base factories ---------------------------
# ----------------------------------------------------------------------

RAWFORMATS = {
    'tf_example': builders.ExampleParserBuilder,
    'tf_sequence_example': builders.SequenceExampleParserBuilder,
}


class MultimodalDataset(dmvr_base.BaseVideoDatasetFactory):
  """Interface for adding modalities with common processing design."""

  _modalities: tuple[str, Ellipsis] = ()
  tokenizer: tokenizers.TextTokenizer | None = None

  def _all_builders(self):
    return {
        'parser_builder': self.parser_builder,
        'sampler_builder': self.sampler_builder,
        'decoder_builder': self.decoder_builder,
        'preprocessor_builder': self.preprocessor_builder,
        'postprocessor_builder': self.postprocessor_builder,
        'filter_builder': self.filter_builder,
    }

  def modalities_provided(self):
    """Returns the modalities provided by this dataset."""
    return self._modalities

  def _build(self, config, **kwargs):
    """Default build for this dataset.

    Args:
      config: Dataset configuration.
      **kwargs: Optional kwargs.
    """
    modality_selection = config.modalities
    audio_config = modality_selection.audio
    vision_config = modality_selection.vision
    text_config = modality_selection.text
    if audio_config:
      if config.is_training is not None:
        audio_config.is_training = config.is_training
      if isinstance(audio_config, data_config.CustomModality):
        self._add_custom_modality(audio_config)
      else:
        self._add_audio(audio_config)
    if vision_config:
      if config.is_training is not None:
        vision_config.is_training = config.is_training
      if isinstance(vision_config, data_config.CustomModality):
        self._add_custom_modality(vision_config)
      else:
        self._add_vision(vision_config)
    if text_config:
      if config.is_training is not None:
        text_config.is_training = config.is_training
      if isinstance(text_config, data_config.CustomModality):
        self._add_custom_modality(text_config)
      else:
        self._add_text(text_config)

    self.postprocessor_builder.add_fn(  # type: ignore
        fn=processing.unflatten_features_dict,
        fn_name='unflatten_features_dict')

  def _add_audio(
      self,
      config,
  ):
    """Adds audio modality."""
    if isinstance(config, data_config.Waveform):
      processing_func = processing.add_waveform
      modality = constants.Modality.WAVEFORM
    elif isinstance(config, data_config.Spectrogram):
      processing_func = processing.add_spectrogram
      modality = constants.Modality.SPECTROGRAM
    else:
      raise ValueError(f'Audio config not valid: {config}')
    self._modalities += processing_func(
        **self._all_builders(), **config.as_dict())
    if config.annotation:
      self._add_annotation(config.annotation, modality)

      # If spectrogram and waveform features co-exist, replicate the labels for
      # both modalities
      if (isinstance(config, data_config.Spectrogram)
          and config.keep_waveform_features
          and config.annotation and config.annotation.label):
        spectrogram_label_feature_name = processing.get_flattened_key(
            ftype=config.annotation.label.data_collection_type,
            froute=config.annotation.label.data_collection_route,
            fname=DataFeatureName.LABEL,
            modality=Modality.SPECTROGRAM)
        waveform_label_feature_name = processing.get_flattened_key(
            ftype=config.annotation.label.data_collection_type,
            froute=config.annotation.label.data_collection_route,
            fname=DataFeatureName.LABEL,
            modality=Modality.WAVEFORM)
        self.preprocessor_builder.add_fn(
            fn=lambda x: processing.copy_feature(  # pylint: disable=g-long-lambda
                x, spectrogram_label_feature_name, waveform_label_feature_name),
            fn_name=(f'{spectrogram_label_feature_name}_to_'
                     f'{waveform_label_feature_name}'))

  def _add_vision(self, config):
    """Adds vision modality."""
    self._modalities += processing.add_vision(**self._all_builders(),
                                              **config.as_dict())

    if config.annotation:
      self._add_annotation(config.annotation, constants.Modality.VISION)

  def _get_tokenizer(self, config):
    """Returns tokenizer."""
    if self.tokenizer is not None:
      raise NotImplementedError('Multiple tokenizer instances currently not '
                                'supported, i.e. you cannot use text and '
                                'label modality at the same time.')
    self.tokenizer = tokenizers.get_tokenizer(config.name)
    self.tokenizer.initialize()

    return self.tokenizer

  def _add_text(
      self,
      config,
  ):
    """Adds text modality."""
    tokenizer = self._get_tokenizer(config.tokenizer_specs)
    if isinstance(config, data_config.Text):
      processing_func = processing.add_text
    elif isinstance(config, data_config.TextFromLabel):
      processing_func = processing.add_text_from_label
    else:
      raise ValueError(f'Text config not valid: {config}')
    self._modalities += processing_func(
        tokenizer=tokenizer, **self._all_builders(), **config.as_dict())

    if config.annotation:
      self._add_annotation(config.annotation, constants.Modality.TEXT)

  def _add_annotation(self,
                      config,
                      modality):
    """Adds annotation."""

    if config.label:
      processing.add_label(
          modality=modality, **self._all_builders(), **config.label.as_dict())

  def _add_custom_modality(
      self, config):
    """Adds custom modality with customizable configuration."""
    def _apply_builder_bundles(builder, bundle_sequence):
      """Applies a (sequence of) bundle(s) to the builder."""
      bundle_sequence = (
          bundle_sequence
          if isinstance(bundle_sequence, tuple)
          else (bundle_sequence,)
      )
      for bundle in bundle_sequence:
        bundle_config = bundle.__arguments__
        bundle_fn = bundle.__fn_or_cls__
        bundle_fn(builder, **bundle_config)
      return

    def _apply_function_bundles(builder, bundle_sequence):
      """Adds a (sequence of) function(s) to the builder."""
      bundle_sequence = (
          bundle_sequence
          if isinstance(bundle_sequence, tuple)
          else (bundle_sequence,)
      )
      for bundle in bundle_sequence:
        bundle_config = bundle.__arguments__
        bundle_fn = bundle.__fn_or_cls__
        builder.add_fn(fn=functools.partial(bundle_fn, **bundle_config))
      return

    if config.parsing is not None:
      _apply_builder_bundles(self.parser_builder, config.parsing)

    if config.sampling is not None:
      _apply_builder_bundles(self.sampler_builder, config.sampling)

    if config.decoding is not None:
      _apply_builder_bundles(self.decoder_builder, config.decoding)

    if config.preprocessing is not None:
      _apply_function_bundles(self.preprocessor_builder, config.preprocessing)

    if config.postprocessing is not None:
      _apply_function_bundles(self.postprocessor_builder, config.postprocessing)




class ThirdPartyDatasetsBase(MultimodalDataset, abc.ABC):
  """Factory for datasets from a filesystem directly."""

  def __init__(self,
               base_dir,
               table = None,
               num_examples = None,
               source = constants.SSTABLE,
               raw_data_format = constants.TF_SEQUENCE_EXAMPLE,
               prop_data = 1.0,
               prop_seed = None,
               num_shards = None,
               shard_index = None,
               tables = None):
    """Initializes the `ThirdPartyDatasetsBase`.

    Args:
      base_dir: The path to the base directory of the dataset, where the
        SSTables can be found.
      table: The SSTable to be read. Available tables must be provided via
        `tables`.
      num_examples: The number of examples for each table in dict format.
      source: The method which the tables are stored.
      raw_data_format: Format of serialized raw data. See `builders.RawFormat`.
      prop_data: Proportion of the data to be consumed. Only that proportion of
        the shards is returned.
      prop_seed: Whether to shuffle the shards (with the given seed) before
        choosing the data used (given the proportion).
      num_shards: If specified will reshard the data according to `num_shards`.
        A `shard_index` should be specified if using `num_shards`.
      shard_index: Index of the shard to return after resharding. `num_shards`
        should be specified if using `shard_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
      tables: All known SSTables belonging to the dataset. Has to be a superset
        of table.

    Raises:
      ValueError: Table name does not exist.
    """
    self._tables = tables
    tables_dict = self.tables()
    if table not in tables_dict.keys():
      raise ValueError(f'Invalid table \'{table}\'. '
                       f'The available tables are: {tables_dict.keys()}.')
    table_relative_path = tables_dict[table]

    base_dir = loading.get_latest_dir(base_dir)
    if isinstance(table_relative_path, list):
      shards = [os.path.join(base_dir, x) for x in table_relative_path]
    else:
      table_path = os.path.join(base_dir, table_relative_path)
      shards = loading.get_sharded_files(
          table_path=table_path,
          prop_data=prop_data,
          prop_seed=prop_seed,
          num_shards=num_shards,
          shard_index=shard_index)
    logging.info('Using %d table shards for %s:%s',
                 len(shards), base_dir, table_relative_path)

    self.source = loading.get_source(source)
    if num_examples:
      self._num_examples = num_examples
    parser_builder_class = RAWFORMATS[raw_data_format]
    super().__init__(
        shards=shards,
        parser_builder_class=parser_builder_class,
        source=self.source())

  # ----------------------------------------------------------------------
  # ---------- Methods that must be implemented by child class. ----------
  # ----------------------------------------------------------------------

  def lookup(self, key_prefix):
    """A copy.deepcopy of the existing factory producing key limited datasets.

    Notes:
     - this approach uses key_prefix which might be slower than specifying
    start/stop, this might be a bottleneck for some datasets.
     - key_prefix means more than one entry might be returned

    Args:
      key_prefix: A key in the dataset

    Returns:
      A copy of the dataset limited to the key

    """
    ret = copy.deepcopy(self)
    ret._source = self.source(key_prefix=key_prefix)
    return ret

  def tables(self):
    """Returns a dictionary from table name to relative path."""
    if self._tables is not None:
      return self._tables
    else:
      raise NotImplementedError(
          'tables() only defined for datasets that initialize _tables.')

  def get_num_examples(self, table):
    if self._num_examples is not None:
      return self._num_examples[table]
    else:
      raise NotImplementedError(
          'get_num_exsamples() only defined for datasets that initialize '
          '_num_examples')


class TfdsBase(ThirdPartyDatasetsBase):
  """Factory for TFDS datasets."""

  def __init__(self,
               dataset_name,
               table,
               prop_data = 1.0,
               prop_seed = None,
               num_shards = None,
               shard_index = None):
    """Initializes `TfdsBase`.

    Args:
      dataset_name: the name of the dataset in TFDS. Supports versioning.
      table: the TFDS split to use.
      prop_data: Proportion of the data to be consumed. Only that proportion of
        the shards is returned.
      prop_seed: Whether to shuffle the shards (with the given seed) before
        choosing the data used (given the proportion).
      num_shards: If specified will reshard the data according to `num_shards`.
        A `shard_index` should be specified if using `num_shards`.
      shard_index: Index of the shard to return after resharding. `num_shards`
        should be specified if using `shard_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.

    Raises:
      ValueError: if the table name does not exist.
    """
    builder = tfds.builder(dataset_name)
    file_format = builder.info.file_format.value  # pylint: disable=attribute-error

    if file_format != 'tfrecord':
      raise NotImplementedError(f'Unsupported file format {file_format}')

    base_dir = builder.data_dir
    splits = builder.info.splits

    if table not in splits:
      raise ValueError(f'{table=} not present in {splits=}')

    num_examples = {name: split.num_examples for name, split in splits.items()}
    shards = splits[table].num_shards

    tables = {}
    for split in splits:
      tables[split] = f'{builder.info.name}-{split}.{file_format}@{shards}'

    super().__init__(
        base_dir=base_dir,
        table=table,
        num_examples=num_examples,
        source=file_format,
        raw_data_format=constants.TF_EXAMPLE,
        prop_data=prop_data,
        prop_seed=prop_seed,
        num_shards=num_shards,
        shard_index=shard_index,
        tables=tables)


class SeqioDataset:
  """Factory for SeqIO datasets."""

  def __init__(self,
               task_name,
               table,
               prop_seed = None,
               num_shards = None,
               shard_index = None,
               **kwargs):
    """Initializes a SeqioDataset.

    Args:
      task_name: the name of the SeqIO task in the TaskRegistry.
      table: the name of the split to use.
      prop_seed: Whether to shuffle the shards (with the given seed) before
        choosing the data used (given the proportion).
      num_shards: If specified will reshard the data according to `num_shards`.
        A `shard_index` should be specified if using `num_shards`.
      shard_index: Index of the shard to return after resharding. `num_shards`
        should be specified if using `shard_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
      **kwargs: additional args
    """
    del kwargs
    self.task_name = task_name
    self.split = table
    self.prop_seed = prop_seed
    self.num_shards = num_shards
    self.shard_index = shard_index
    self.tokenizer = None
    self.is_training = None
    self.sequence_length = None
    self.seed = None
    self.configured = False

  def tune(self, seed = None, **unused_kwargs):
    del unused_kwargs
    self.seed = seed

  def configure(self, config):
    self.is_training = config.is_training
    if config.modalities.text is None:
      raise ValueError('Text config should not be None.')
    elif not isinstance(config.modalities.text, data_config.Text):
      raise ValueError('Config should be an instance of `Text`.')
    elif config.modalities.text.max_num_tokens is None:
      raise ValueError('Max num tokens should not be None.')
    self.sequence_length = config.modalities.text.max_num_tokens
    self.configured = True
    return self

  def make_dataset(self,
                   shuffle = False,
                   num_epochs = -1,
                   batch_size = 1,
                   padded_batch = False,
                   drop_remainder = True,
                   keep_key = False,
                   override_preprocess_fn = None,
                   ignore_processing_errors = False):
    """Function to construct the data graph and return tf.data.Dataset."""
    del padded_batch
    del keep_key
    del override_preprocess_fn

    if not self.configured:
      raise ValueError('Dataset object is not configured.')

    shard_info = None
    if self.shard_index and self.num_shards:
      shard_info = seqio.ShardInfo(
          index=self.shard_index, num_shards=self.num_shards)

    dataset = seqio.get_dataset(
        mixture_or_task_name=self.task_name,
        task_feature_lengths={
            'inputs': self.sequence_length,
            'targets': self.sequence_length,
        },
        feature_converter=seqio.PassThroughFeatureConverter(),
        dataset_split=self.split,
        use_cached=False,
        shuffle=shuffle,
        num_epochs=num_epochs,
        shard_info=shard_info,
        seed=self.seed,
    )
    dataset = seqio_utils.seqio_to_dmvr_format(dataset)

    if ignore_processing_errors:
      dataset = dataset.apply(  # pylint: disable=attribute-error
          tf.data.experimental.ignore_errors(log_warning=True))

    dataset = dataset.batch(
        batch_size,
        drop_remainder=drop_remainder)

    return dataset


# ----------------------------------------------------------------------
# -------------------------- Custom factories --------------------------
# ----------------------------------------------------------------------




# pytype: enable=attribute-error
ALL_FACTORIES = {
    # Generic
    constants.DataFactory.SEQIO: SeqioDataset,
    constants.DataFactory.THIRDPARTY: ThirdPartyDatasetsBase,
    constants.DataFactory.TFDS: TfdsBase,
}

FactoryT = type(ThirdPartyDatasetsBase |
                TfdsBase |
                SeqioDataset)



def get_data_factory(factory_name):
  """Fetches the data factory class given their name."""
  return ALL_FACTORIES[factory_name]
