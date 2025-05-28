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

"""Factory for dataset."""

# pylint: disable=g-bad-import-order
from q_match.datasets.example import ExampleDataset

from q_match.datasets.covtype import CoverType10PDataset
from q_match.datasets.covtype import CoverTypeDataset
from q_match.datasets.covtype import CoverTypeImixDataset
from q_match.datasets.covtype import CoverTypeNew1PDataset

from q_match.datasets.higgs import Higgs100k10pDataset
from q_match.datasets.higgs import Higgs100k1pDataset
from q_match.datasets.higgs import Higgs100k20pDataset
from q_match.datasets.higgs import Higgs100kDataset
from q_match.datasets.higgs import Higgs10kDataset
from q_match.datasets.higgs import Higgs1MDataset
from q_match.datasets.higgs import Higgs50kDataset
from q_match.datasets.higgs import HiggsDataset
from q_match.datasets.higgs import HiggsPre10kDataset
from q_match.datasets.higgs import HiggsPre40kDataset
from q_match.datasets.higgs import HiggsPre160kDataset
from q_match.datasets.higgs import HiggsPre640kDataset
from q_match.datasets.higgs import HiggsPre2560kDataset

from q_match.datasets.mnist import MNIST10PDataset
from q_match.datasets.mnist import MNIST1PDataset

from q_match.datasets.adult import Adult1PDataset

from q_match.datasets.mnist import MNISTPre100Dataset
from q_match.datasets.mnist import MNISTPre10Dataset

from q_match.datasets.higgs import HiggsPre640kSup100Dataset
from q_match.datasets.higgs import HiggsPre640kSup500Dataset
from q_match.datasets.higgs import HiggsPre640kSup1kDataset

DATASET_NAME_TO_CLASS = {'example': ExampleDataset,
                         'higgs': HiggsDataset,
                         'higgspre10k': HiggsPre10kDataset,
                         'higgspre40k': HiggsPre40kDataset,
                         'higgspre160k': HiggsPre160kDataset,
                         'higgspre640k': HiggsPre640kDataset,
                         'higgspre2560k': HiggsPre2560kDataset,
                         'higgs100k': Higgs100kDataset,
                         'higgs100k1p': Higgs100k1pDataset,
                         'higgs100k20p': Higgs100k20pDataset,
                         'higgs100k10p': Higgs100k10pDataset,
                         'higgs1M': Higgs1MDataset,
                         'higgs50k': Higgs50kDataset,
                         'higgs10k': Higgs10kDataset,
                         'covtype': CoverTypeDataset,
                         'covtype_imix': CoverTypeImixDataset,
                         'covtype_new_1p': CoverTypeNew1PDataset,
                         'covtype_10p': CoverType10PDataset,
                         'mnist_1p': MNIST1PDataset,
                         'mnist_10p': MNIST10PDataset,
                         'adult_1p': Adult1PDataset,
                         'mnistpre100p': MNISTPre100Dataset,
                         'mnistpre10p': MNISTPre10Dataset,
                         'higgspre640ksup100': HiggsPre640kSup100Dataset,
                         'higgspre640ksup500': HiggsPre640kSup500Dataset,
                         'higgspre640ksup1k': HiggsPre640kSup1kDataset,
                         }


def get_dataset_class(dataset_name):
  """Returns dataset."""
  if dataset_name not in DATASET_NAME_TO_CLASS:
    raise ValueError('%s not supported yet.' % dataset_name)
  return DATASET_NAME_TO_CLASS[dataset_name]
