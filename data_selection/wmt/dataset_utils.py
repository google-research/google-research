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

"""Util to build datasets to experiments."""

import pickle

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


PARACRAWL_DEFAULT_SIZE = 4500000
WMT_BASE_DATASET_NAME = 'wmt_translate'
NEWS_COMMENTARY = 'newscommentary'
NEWS_COMMENTARY_FT = 'newscommentary'
PARACRAWL = 'paracrawl'
NEWSTEST = 'newstest'
RANDOM_SAMPLE_SEED = 42
enru_paracrawl = 'paracrawl-en-ru.txt.gz'
enru_newscomm = 'news-commentary-v15.en-ru-shuffled.tsv.gz'
wmt_commoncrawl = 'commoncrawl/train.tsv-0000%d-of-00001'
wmt_euro = 'europarl-origde/train.tsv-0000%d-of-00004'
wmt_newscomment = 'news_commentary_v15/train.tsv-0000%d-of-00004'
wmt_paracrawl = 'paracrawl/train.tsv-000%02d-of-00079'
wmt_train = 'train.tsv'
wmt_train_small = 'train_small.tsv'
wmt_test = 'test.tsv'
wmt_test_large = 'test_large.tsv'


class WmtDatasetBuilder():
  """Util class for building WMT datasets for MT experiments."""

  def __init__(self, shard_idx=0, shard_count=1, data_dir=None,
               shuffle_train_files=True, pseudo_path=None):

    self.paracrawl_size = 0
    self.newscommentary_size = 75000
    self.data_dir = data_dir
    self.shard_idx = shard_idx
    self.shard_count = shard_count
    self.default_builder_obj = None
    self.shuffle_train_files = shuffle_train_files
    self.pseudo_path = pseudo_path
    self.newscomment_sample_ratio = 1.0
    self.configs = {
        NEWS_COMMENTARY:
            tfds.translate.wmt.WmtConfig(
                version='1.0.0',
                language_pair=('de', 'en'),
                subsets={
                    tfds.Split.TRAIN: ['newscommentary_v13'],
                    tfds.Split.VALIDATION: ['newstest2013'],
                },
                name='newscommentary'),
        NEWS_COMMENTARY_FT:
            tfds.translate.wmt.WmtConfig(
                version='1.0.0',
                language_pair=('de', 'en'),
                subsets={
                    tfds.Split.TRAIN: ['newscommentary_v13'],
                    tfds.Split.VALIDATION: ['newscommentary_v13'],
                },
                name='newscommentary'),
        PARACRAWL:
            tfds.translate.wmt.WmtConfig(
                version='1.0.0',
                language_pair=('de', 'en'),
                subsets={
                    tfds.Split.TRAIN: ['paracrawl_v1'],
                },
                name='paracrawl'),
        NEWSTEST:
            tfds.translate.wmt.WmtConfig(
                version='1.0.0',
                language_pair=('de', 'en'),
                subsets={
                    tfds.Split.TRAIN: ['newstest2011', 'newstest2012'],
                    tfds.Split.VALIDATION: ['newstest2013'],
                },
                name='newstest_finetune')
    }

    self.custom_dataset = {
        'newscommentary_only': self.build_newscomment_only,
        'newscommentary_paracrawl': self.build_newscomment_paracrawl,
        'nc_para_var': self.build_newscomment_paracrawl_var,
        'newstest_finetune': self.build_newstest_finetune,
        'paracrawl_only': self.build_paracrawl_only,
        'pseudo_ref': self.build_pseudo_ref,
        'newscommentary_ft': self.build_newscomment_ft,
        'newscommentary_ft_1k': self.build_newscomment_ft_1k,
        'paracrawl_eval_nc': self.build_paracrawl_eval_nc,
        'paracrawl_new_eval_nc': self.build_paracrawl_new_eval_nc,
        'newscommentary_ft_alt': self.build_newscomment_ft_alt,
        'newscommentary_ft_full': self.build_newscomment_ft_full,
        'newscommentary_ft_dont_use': self.build_newscomment_dont_use,
        'newscommentary_ft_large': self.build_newscomment_ft_large,
        'newscommentary_ft_train_var': self.build_newscomment_train_var,
        'newscomment_eval_ft': self.build_newscomment_eval_ft,
        'newscomment_eval_alt1': self.build_newscomment_eval_alt1,
        'newscomment_eval_alt2': self.build_newscomment_eval_alt2,
        'newscomment_eval_alt3': self.build_newscomment_eval_alt3,
        'newscomment_eval_alt4': self.build_newscomment_eval_alt4,
        'newscomment_ft_var': self.build_newscomment_var,
        'newscomment_ft_var_unseen': self.build_newscomment_var_unseen,
        'enru_custom': self.build_enru_custom,
        'enru_custom_ft': self.build_enru_custom_ft,
        'enru_custom_test': self.build_enru_custom_test,
        'newscommentary_test': self.build_newscommentary_test,
        'wmt_filtered': self.build_wmt_filtered,
        'wmt_filtered_half': self.build_wmt_filtered_half,
        'wmt_ft': self.build_wmt_ft,
        'wmt_ft_half': self.build_wmt_ft_half,
        'newscomment_eval_train': self.build_newscomment_eval_train
    }

  def build_shard_spec(self, max_size=100, percent=True, start=0):
    spec_type = '%' if percent else ''
    shard_spec = (
        f'[{int(max_size * self.shard_idx / self.shard_count) + start}'
        f'{spec_type}:{int(max_size * (self.shard_idx + 1) / self.shard_count)}'
        f'{spec_type}]')
    return shard_spec

  def retrieve_builder(self):
    return self.default_builder_obj

  def build_wmt_ft_half(self):
    """Create en-ru paracrawl / newscommentary dataset."""
    train_files = [self.data_dir + '/' + wmt_train_small]
    eval_files = [self.data_dir + '/' + wmt_test_large]

    train_data = tf.data.experimental.CsvDataset(
        train_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)
    eval_data = tf.data.experimental.CsvDataset(
        eval_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)

    eval_data = eval_data.cache()
    train_data = train_data.cache()  # only read once

    def to_features_dict(eng, rus):
      return {'inputs': eng, 'targets': rus}

    train_data = train_data.map(to_features_dict)
    eval_data = eval_data.map(to_features_dict)

    self.default_builder_obj = None
    return train_data, eval_data

  def build_wmt_ft(self):
    """Create en-ru paracrawl / newscommentary dataset."""
    train_files = [self.data_dir + '/' + wmt_train]
    eval_files = [self.data_dir + '/' + wmt_test]

    train_data = tf.data.experimental.CsvDataset(
        train_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)
    eval_data = tf.data.experimental.CsvDataset(
        eval_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)

    eval_data = eval_data.cache()
    train_data = train_data.cache()  # only read once

    def to_features_dict(eng, rus):
      return {'inputs': eng, 'targets': rus}

    train_data = train_data.map(to_features_dict)
    eval_data = eval_data.map(to_features_dict)

    self.default_builder_obj = None
    return train_data, eval_data

  def build_wmt_filtered(self):
    return self._build_wmt_filtered()

  def build_wmt_filtered_half(self):
    return self._build_wmt_filtered(half=True)

  def _build_wmt_filtered(self, half=False):
    """Create en-ru paracrawl / newscommentary dataset."""
    paracrawl_files = [
        self.data_dir + '/' + wmt_paracrawl % i for i in range(40)
    ]
    europarl_files = [
        self.data_dir + '/' + wmt_euro % i for i in range(4)
    ]
    newscomment_files = [
        self.data_dir + '/' + wmt_newscomment % i for i in range(4)
    ]
    commoncrawl_files = [
        self.data_dir + '/' + wmt_commoncrawl % i for i in range(1)
    ]

    pc_data = tf.data.experimental.CsvDataset(
        paracrawl_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)
    euro_data = tf.data.experimental.CsvDataset(
        europarl_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)
    nc_data = tf.data.experimental.CsvDataset(
        newscomment_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)
    cc_data = tf.data.experimental.CsvDataset(
        commoncrawl_files,
        record_defaults=[tf.string, tf.string],
        field_delim='\t',
        use_quote_delim=False)

    pc_eval_data = pc_data.skip(10000).take(5000)
    euro_eval_data = euro_data.skip(10000).take(5000)
    nc_eval_data = nc_data.skip(10000).take(5000)
    cc_eval_data = cc_data.skip(10000).take(5000)

    pc_train_data = pc_data.skip(15000)
    euro_train_data = euro_data.skip(15000)
    nc_train_data = nc_data.skip(15000)
    cc_train_data = cc_data.skip(15000)

    if half:
      pc_train_data = pc_train_data.take(14_125_429)
      euro_train_data = euro_train_data.take(89_725)
      nc_train_data = nc_train_data.take(125_726)
      cc_train_data = cc_train_data.take(747_389)

    # Save these examples for testing
    # this is not intended to be uncommented. It just shows
    # pseudo-code for which examples are saved for testing.
    # pc_test_data = pc_data.take(10000)
    # euro_test_data = euro_data.take(10000)
    # nc_test_data = nc_data.take(10000)
    # cc_test_data = cc_data.take(10000)

    eval_data = tf.data.experimental.sample_from_datasets(
        [pc_eval_data, euro_eval_data, nc_eval_data, cc_eval_data], seed=42)
    eval_data = eval_data.cache()

    train_data = tf.data.experimental.sample_from_datasets(
        [pc_train_data, euro_train_data, nc_train_data, cc_train_data],
        weights=[0.9375, 0.0054, 0.00785, 0.0491], seed=42)
    train_data = train_data.cache()  # only read once

    def to_features_dict(eng, rus):
      return {'inputs': eng, 'targets': rus}

    train_data = train_data.map(to_features_dict)
    eval_data = eval_data.map(to_features_dict)

    self.default_builder_obj = None

    return train_data, eval_data

  def build_enru_custom_ft(self):
    """Create en-ru paracrawl / newscommentary dataset."""
    eval_data_file = self.data_dir + '/' + enru_newscomm
    eval_data = tf.data.experimental.CsvDataset(
        [eval_data_file],
        record_defaults=[tf.string, tf.string],
        compression_type='GZIP',
        field_delim='\t',
        use_quote_delim=False)
    train_data = eval_data.skip(3000).take(6000)
    eval_data = eval_data.take(3000)

    eval_data = eval_data.cache()
    train_data = train_data.cache()
    def to_features_dict(eng, rus):
      return {'inputs': eng, 'targets': rus}

    train_data = train_data.map(to_features_dict)
    eval_data = eval_data.map(to_features_dict)

    self.default_builder_obj = None

    return train_data, eval_data

  def build_enru_custom_test(self):
    """Create en-ru paracrawl / newscommentary dataset."""
    train_data_file = self.data_dir + '/' + enru_paracrawl
    eval_data_file = self.data_dir + '/' + enru_newscomm
    train_data = tf.data.experimental.CsvDataset(
        [train_data_file],
        record_defaults=[tf.string, tf.string],
        compression_type='GZIP',
        field_delim='\t',
        use_quote_delim=False)
    train_data = train_data.cache()  # only read once
    eval_data = tf.data.experimental.CsvDataset(
        [eval_data_file],
        record_defaults=[tf.string, tf.string],
        compression_type='GZIP',
        field_delim='\t',
        use_quote_delim=False)

    eval_data = eval_data.skip(9000).take(10000)
    eval_data = eval_data.cache()
    def to_features_dict(eng, rus):
      return {'inputs': eng, 'targets': rus}

    train_data = train_data.map(to_features_dict)
    eval_data = eval_data.map(to_features_dict)

    self.default_builder_obj = None

    return train_data, eval_data

  def build_enru_custom(self):
    """Create en-ru paracrawl / newscommentary dataset."""
    train_data_file = self.data_dir + '/' + enru_paracrawl
    eval_data_file = self.data_dir + '/' + enru_newscomm
    train_data = tf.data.experimental.CsvDataset(
        [train_data_file],
        record_defaults=[tf.string, tf.string],
        compression_type='GZIP',
        field_delim='\t',
        use_quote_delim=False)
    train_data = train_data.cache()  # only read once
    eval_data = tf.data.experimental.CsvDataset(
        [eval_data_file],
        record_defaults=[tf.string, tf.string],
        compression_type='GZIP',
        field_delim='\t',
        use_quote_delim=False)

    eval_data = eval_data.take(3000)
    eval_data = eval_data.cache()
    def to_features_dict(eng, rus):
      return {'inputs': eng, 'targets': rus}

    train_data = train_data.map(to_features_dict)
    eval_data = eval_data.map(to_features_dict)

    self.default_builder_obj = None

    return train_data, eval_data

  def build_train_and_eval_datasets(self,
                                    dataset_name,
                                    eval_dataset_name,
                                    paracrawl_size=PARACRAWL_DEFAULT_SIZE,
                                    newscommentary_size=None,
                                    newscomment_sample_ratio=1.0):
    """Build train and eval datasets."""
    self.paracrawl_size = paracrawl_size
    if newscommentary_size:
      self.newscommentary_size = newscommentary_size
    self.newscomment_sample_ratio = newscomment_sample_ratio
    if dataset_name in self.custom_dataset.keys():
      logging.info('Building custom datatset: %s', dataset_name)
      return self.custom_dataset[dataset_name]()
    else:
      logging.info('Building DEFAULT datatset: %s', dataset_name)
      return self.default_builder(dataset_name, eval_dataset_name)

  def default_builder(self, dataset_name, eval_dataset_name):
    """Default data builder from flax/examples/wmt."""
    builder = tfds.builder(dataset_name, data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec()
    logging.info('Training on TFDS dataset %s with split %s',
                 dataset_name, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=self.shuffle_train_files)

    if eval_dataset_name is None:
      logging.info('Evaluating on TFDS dataset %s with split %s',
                   dataset_name, 'validation' + shard_spec)
      eval_data = self.default_eval_builder(builder, shard_spec)
    else:
      eval_dataset, *eval_split = eval_dataset_name.split(':')
      if not eval_split:
        eval_split = 'validation'
      else:
        eval_split = eval_split[0]
      logging.info('Evaluating on TFDS dataset %s with split %s',
                   eval_dataset, eval_split + shard_spec)
      eval_builder = tfds.builder(eval_dataset, data_dir=self.data_dir)
      eval_data = eval_builder.as_dataset(split=eval_split + shard_spec,
                                          shuffle_files=False)
    return train_data, eval_data

  def default_eval_builder(self, builder, shard_spec):
    logging.info('Default eval dataset using provided builder')
    eval_data = builder.as_dataset(split='validation' + shard_spec,
                                   shuffle_files=False)
    return eval_data

  def build_newscomment_only(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec()
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=self.shuffle_train_files)
    eval_data = self.default_eval_builder(builder, shard_spec)
    return train_data, eval_data

  def build_newscomment_limited(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(start=84000, percent=False,
                                       max_size=85000)  # 284246 full
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    return train_data, None

  def build_newscomment_var(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    max_size = 9000 + self.newscommentary_size
    shard_spec = self.build_shard_spec(start=9000, percent=False,
                                       max_size=max_size)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    _, nc_eval_data = self.build_newscomment_ft()
    return train_data, nc_eval_data

  def build_newscomment_var_unseen(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    max_size = 159000 + self.newscommentary_size
    shard_spec = self.build_shard_spec(start=159000, percent=False,
                                       max_size=max_size)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    _, nc_eval_data = self.build_newscomment_ft()
    return train_data, nc_eval_data

  def build_newscomment_train_var(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY])
    builder = tfds.builder(
        WMT_BASE_DATASET_NAME,
        config=self.configs[NEWS_COMMENTARY],
        data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(
        start=84000, percent=False, max_size=84000 + self.newscommentary_size)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(
        split='train' + shard_spec, shuffle_files=False)

    valid_shard_spec = self.build_shard_spec(
        max_size=9000, percent=False, start=6000)
    eval_data = builder.as_dataset(
        split='train' + valid_shard_spec, shuffle_files=False)
    return train_data, eval_data

  def build_newscomment_large(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(start=9000, percent=False,
                                       max_size=9000+self.newscommentary_size)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    return train_data, None

  def build_newscomment_ft(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(max_size=6000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    valid_shard_spec = self.build_shard_spec(max_size=9000, percent=False,
                                             start=6000)
    eval_data = builder.as_dataset(split='train' + valid_shard_spec,
                                   shuffle_files=False)
    return train_data, eval_data

  def build_newscommentary_test(self):
    """Build dataset of testing 10k from News Commentary V13."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(max_size=1, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    valid_shard_spec = self.build_shard_spec(max_size=19000, percent=False,
                                             start=9000)
    eval_data = builder.as_dataset(split='train' + valid_shard_spec,
                                   shuffle_files=False)
    return train_data, eval_data

  def build_newscomment_ft_1k(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(max_size=1000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    valid_shard_spec = self.build_shard_spec(max_size=9000, percent=False,
                                             start=6000)
    eval_data = builder.as_dataset(split='train' + valid_shard_spec,
                                   shuffle_files=False)
    return train_data, eval_data

  def build_newscomment_eval_ft(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    new_train_data, _ = self.build_newscomment_ft()
    return new_train_data, new_train_data

  def build_newscomment_eval_train(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    new_train_data, _ = self.build_newscomment_var()
    return new_train_data, new_train_data

  def build_newscomment_eval_alt1(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(
        WMT_BASE_DATASET_NAME,
        config=self.configs[NEWS_COMMENTARY_FT],
        data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(
        start=100000, max_size=110000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(
        split='train' + shard_spec, shuffle_files=False)
    return train_data, train_data

  def build_newscomment_eval_alt2(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(
        WMT_BASE_DATASET_NAME,
        config=self.configs[NEWS_COMMENTARY_FT],
        data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(
        start=110000, max_size=120000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(
        split='train' + shard_spec, shuffle_files=False)
    return train_data, train_data

  def build_newscomment_eval_alt3(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(
        WMT_BASE_DATASET_NAME,
        config=self.configs[NEWS_COMMENTARY_FT],
        data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(
        start=120000, max_size=130000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(
        split='train' + shard_spec, shuffle_files=False)
    return train_data, train_data

  def build_newscomment_eval_alt4(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(
        WMT_BASE_DATASET_NAME,
        config=self.configs[NEWS_COMMENTARY_FT],
        data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(
        start=130000, max_size=140000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(
        split='train' + shard_spec, shuffle_files=False)
    return train_data, train_data

  def build_newscomment_dont_use(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(max_size=6000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    valid_shard_spec = self.build_shard_spec(max_size=9000, percent=False,
                                             start=6000)
    eval_data = builder.as_dataset(split='train' + valid_shard_spec,
                                   shuffle_files=False)
    return eval_data, eval_data

  def build_newscomment_ft_full(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset for ft.')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)

    valid_shard_spec = self.build_shard_spec(max_size=9000, percent=False,
                                             start=6000)
    eval_data = builder.as_dataset(split='train' + valid_shard_spec,
                                   shuffle_files=False)

    train_data, _ = self.build_newscomment_limited()
    return train_data, eval_data

  def build_newscomment_ft_large(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset for ft.')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)

    valid_shard_spec = self.build_shard_spec(max_size=9000, percent=False,
                                             start=6000)
    eval_data = builder.as_dataset(split='train' + valid_shard_spec,
                                   shuffle_files=False)

    train_data, _ = self.build_newscomment_large()
    return train_data, eval_data

  def build_newscomment_ft_alt(self):
    """Build dataset of news_commentary_v13 only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[NEWS_COMMENTARY_FT])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWS_COMMENTARY_FT],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(start=9000,
                                       max_size=15000, percent=False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=False)
    valid_shard_spec = self.build_shard_spec(max_size=9000, percent=False,
                                             start=6000)
    eval_data = builder.as_dataset(split='train' + valid_shard_spec,
                                   shuffle_files=False)
    return train_data, eval_data

  def build_paracrawl_only(self):
    """Build dataset of paracrawl only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[PARACRAWL])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[PARACRAWL],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(self.paracrawl_size, False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=self.shuffle_train_files)
    # _, eval_data = self.build_newscomment_only()
    _, eval_data = self.build_newscomment_ft()
    return train_data, eval_data

  def build_paracrawl_eval_nc(self):
    """Build dataset of paracrawl only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[PARACRAWL])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[PARACRAWL],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(self.paracrawl_size, False)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=self.shuffle_train_files)
    _, eval_data = self.build_newscomment_ft()
    return train_data, eval_data

  def build_paracrawl_new_eval_nc(self):
    """Build dataset of paracrawl only, including validation."""
    logging.info('Building news commentary only dataset')
    logging.info(self.configs[PARACRAWL])
    builder = tfds.builder(
        WMT_BASE_DATASET_NAME,
        config=self.configs[PARACRAWL],
        data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec(
        PARACRAWL_DEFAULT_SIZE + self.paracrawl_size, False,
        PARACRAWL_DEFAULT_SIZE)
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(
        split='train' + shard_spec, shuffle_files=self.shuffle_train_files)
    _, eval_data = self.build_newscomment_ft()
    return train_data, eval_data

  def build_newstest_finetune(self):
    """Build dataset of newstest_2011 and 2012, including validation."""
    # Note that this function is purposefully similar to build_newscomment_only
    # The two datasets have very similar structure and it would just be more
    # confusing to refactor code, creating multiple overlapping paths.
    logging.info('Building newstest finetune dataset')
    logging.info(self.configs[NEWSTEST])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWSTEST],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec()
    logging.info('Training on TFDS dataset %s with split %s',
                 WMT_BASE_DATASET_NAME, 'train' + shard_spec)
    train_data = builder.as_dataset(split='train' + shard_spec,
                                    shuffle_files=self.shuffle_train_files)
    eval_data = self.default_eval_builder(builder, shard_spec)
    return train_data, eval_data

  def build_newscomment_paracrawl(self):
    """Combine newscommentary dataset with paracrawl."""
    # Note: build_newscomment_only sets a default_builder_obj
    # if removed, set explicitly
    nc_train_data, _ = self.build_newscomment_limited()

    nc_data_size = nc_train_data.cardinality().numpy()  # Should be 284246
    logging.info('News commentary size is... %d', nc_data_size)
    paracrawl_builder = tfds.builder(WMT_BASE_DATASET_NAME,
                                     config=self.configs[PARACRAWL],
                                     data_dir=self.data_dir)
    paracrawl_shard_spec = self.build_shard_spec(self.paracrawl_size,
                                                 False)
    para_train_data = paracrawl_builder.as_dataset(
        split='train' + paracrawl_shard_spec,
        shuffle_files=self.shuffle_train_files)
    logging.info('Paracrawl size is... %d',
                 para_train_data.cardinality().numpy())

    total_dataset_size = float(nc_data_size + self.paracrawl_size)
    nc_prop = float(nc_data_size) / total_dataset_size
    pc_prop = float(self.paracrawl_size) / total_dataset_size
    logging.info('Sampling proportion is %f, %f', nc_prop, pc_prop)

    train_data = tf.data.experimental.sample_from_datasets(
        [nc_train_data, para_train_data],
        weights=[nc_prop, pc_prop],
        seed=RANDOM_SAMPLE_SEED)

    _, nc_eval_data = self.build_newscomment_ft()

    return train_data, nc_eval_data

  def build_newscomment_paracrawl_var(self):
    """Combine newscommentary dataset with paracrawl."""
    # Note: build_newscomment_only sets a default_builder_obj
    # if removed, set explicitly
    nc_train_data, _ = self.build_newscomment_var()

    nc_data_size = nc_train_data.cardinality().numpy()  # Should be 284246
    assert abs(nc_data_size - self.newscommentary_size) < 10_000
    logging.info('News commentary size is... %d', nc_data_size)
    paracrawl_builder = tfds.builder(WMT_BASE_DATASET_NAME,
                                     config=self.configs[PARACRAWL],
                                     data_dir=self.data_dir)
    paracrawl_shard_spec = self.build_shard_spec(self.paracrawl_size,
                                                 False)
    para_train_data = paracrawl_builder.as_dataset(
        split='train' + paracrawl_shard_spec,
        shuffle_files=self.shuffle_train_files)
    logging.info('Paracrawl size is... %d',
                 para_train_data.cardinality().numpy())

    nc_data_size *= self.newscomment_sample_ratio
    if self.newscomment_sample_ratio != 1:
      nc_train_data = nc_train_data.repeat(int(self.newscomment_sample_ratio))
    total_dataset_size = float(nc_data_size + self.paracrawl_size)
    nc_prop = float(nc_data_size) / total_dataset_size
    pc_prop = float(self.paracrawl_size) / total_dataset_size
    logging.info('Sampling proportion is %f, %f', nc_prop, pc_prop)

    train_data = tf.data.experimental.sample_from_datasets(
        [nc_train_data, para_train_data],
        weights=[nc_prop, pc_prop],
        seed=RANDOM_SAMPLE_SEED)

    _, nc_eval_data = self.build_newscomment_ft()

    return train_data, nc_eval_data

  def build_pseudo_ref(self):
    """Build pseudo ref dataset from pickle."""
    logging.info('Building pseudo finetune dataset')
    logging.info(self.configs[NEWSTEST])
    builder = tfds.builder(WMT_BASE_DATASET_NAME,
                           config=self.configs[NEWSTEST],
                           data_dir=self.data_dir)
    self.default_builder_obj = builder
    shard_spec = self.build_shard_spec()
    eval_data = self.default_eval_builder(builder, shard_spec)

    new_data = pickle.load(tf.io.gfile.GFile(self.pseudo_path, 'rb'))
    # Create tensorflow dataset
    tf_pre_dataset = {'inputs': [], 'targets': []}
    for data in new_data:
      inp = data[-2]
      targ = data[-1]  # [1:]  # Targets have dummy first variable
      tf_pre_dataset['inputs'].append(inp)
      tf_pre_dataset['targets'].append(targ)

    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_pre_dataset)
    return tf_dataset, eval_data
