# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

r"""TFDS databuilder for fmow_v1.0."""

import datetime as dt

import pytz

from gift.data.builders import fmow_builder


class GradualFmow(fmow_builder.Fmow):
  """TFDS builder for fmow_v1.

  The Functional Map of the World land use / building classification dataset.
    This is a processed version of the Functional Map of the World dataset
    originally sourced from https://github.com/fMoW/dataset.

  This dataset is part of the WILDS benchmark.
  """

  _SPLITS = [
      'train', 'val_id', 'val_ood_1', 'val_ood_2', 'val_ood_3', 'test_id',
      'test_ood'
  ]

  # pylint: disable=g-tzinfo-datetime
  # pylint: disable=g-long-lambda
  _DOMAIN_FILTERS = {
      'train':
          lambda date: filter(date, dt.datetime(2002, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2013, 1, 1, tzinfo=pytz.UTC)),
      'val_id':
          lambda date: filter(date, dt.datetime(2002, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2013, 1, 1, tzinfo=pytz.UTC)),
      'val_ood_1':
          lambda date: filter(date, dt.datetime(2013, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2014, 1, 1, tzinfo=pytz.UTC)),
      'val_ood_2':
          lambda date: filter(date, dt.datetime(2014, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2015, 1, 1, tzinfo=pytz.UTC)),
      'val_ood_3':
          lambda date: filter(date, dt.datetime(2015, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2016, 1, 1, tzinfo=pytz.UTC)),
      'test_id':
          lambda date: filter(date, dt.datetime(2002, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2013, 1, 1, tzinfo=pytz.UTC)),
      'test_ood':
          lambda date: filter(date, dt.datetime(2016, 1, 1, tzinfo=pytz.UTC),
                              dt.datetime(2018, 1, 1, tzinfo=pytz.UTC)),
  }
  # pylint: enable=g-long-lambda
  # pylint: enable=g-tzinfo-datetime
