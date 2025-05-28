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

"""Libary containing all datasets."""

import dataclasses
import os
from typing import Mapping

import immutabledict

_DATA_BASE_DIR = 'mave/datasets/splits/PRODUCT'


@dataclasses.dataclass(frozen=True)
class Dataset:
  """Dataset class."""
  train_jsonl: str = ''
  train_tf_records: str = ''
  train_size: int = 0
  eval_jsonl: str = ''
  eval_tf_records: str = ''
  eval_size: int = 0
  test_jsonl: str = ''
  test_tf_records: str = ''
  test_size: int = 0


DATASETS: Mapping[str, Dataset] = immutabledict.immutabledict({
    '00_All_bert':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/00_All/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR, 'train/00_All/mave_*_bert_tfrecord-*'),
            train_size=(1425936 + 2388842),
            eval_jsonl=os.path.join(_DATA_BASE_DIR, 'eval/00_All/mave_*.jsonl'),
            eval_tf_records=os.path.join(_DATA_BASE_DIR,
                                         'eval/00_All/mave_*_bert_tfrecord-*'),
            eval_size=(176978 + 298696),
            test_jsonl=os.path.join(_DATA_BASE_DIR, 'test/00_All/mave_*.jsonl'),
            test_tf_records=os.path.join(_DATA_BASE_DIR,
                                         'test/00_All/mave_*_bert_tfrecord-*'),
            test_size=(177514 + 299613),
        ),
    '00_All_etc':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/00_All/mave_*.jsonl'),
            train_tf_records=os.path.join(_DATA_BASE_DIR,
                                          'train/00_All/mave_*_etc_tfrecord-*'),
            train_size=(1425936 + 2388842),
            eval_jsonl=os.path.join(_DATA_BASE_DIR, 'eval/00_All/mave_*.jsonl'),
            eval_tf_records=os.path.join(_DATA_BASE_DIR,
                                         'eval/00_All/mave_*_etc_tfrecord-*'),
            eval_size=(176978 + 298696),
            test_jsonl=os.path.join(_DATA_BASE_DIR, 'test/00_All/mave_*.jsonl'),
            test_tf_records=os.path.join(_DATA_BASE_DIR,
                                         'test/00_All/mave_*_etc_tfrecord-*'),
            test_size=(177514 + 299613),
        ),
    '02_Type_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/02_Type/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR, 'train/02_Type/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(86661 + 805401),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/02_Type/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR, 'eval/02_Type/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(10738 + 100763),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/02_Type/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR, 'test/02_Type/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(10726 + 101147),
        ),
    '03_Style_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/03_Style/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR, 'train/03_Style/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(47899 + 138124),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/03_Style/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR, 'eval/03_Style/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(6151 + 17173),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/03_Style/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR, 'test/03_Style/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(6004 + 17231),
        ),
    '04_Material_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/04_Material/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/04_Material/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(30204 + 93701),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/04_Material/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/04_Material/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(3762 + 11794),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/04_Material/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/04_Material/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(3737 + 11640),
        ),
    '05_Size_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/05_Size/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR, 'train/05_Size/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(13323 + 21906),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/05_Size/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR, 'eval/05_Size/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(1690 + 2836),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/05_Size/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR, 'test/05_Size/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(1689 + 2811),
        ),
    '06_Capacity_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/06_Capacity/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/06_Capacity/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(9149 + 15088),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/06_Capacity/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/06_Capacity/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(1161 + 1881),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/06_Capacity/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/06_Capacity/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(1116 + 1794),
        ),
    '07_Black_Tea_Variety_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/07_Black_Tea_Variety/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/07_Black_Tea_Variety/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(3434 + 124),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/07_Black_Tea_Variety/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/07_Black_Tea_Variety/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(439 + 15),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/07_Black_Tea_Variety/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/07_Black_Tea_Variety/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(473 + 16),
        ),
    '08_Staple_Type_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/08_Staple_Type/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/08_Staple_Type/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(249 + 135),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/08_Staple_Type/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/08_Staple_Type/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(33 + 22),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/08_Staple_Type/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/08_Staple_Type/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(28 + 10),
        ),
    '09_Web_Pattern_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/09_Web_Pattern/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/09_Web_Pattern/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(1110 + 163),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/09_Web_Pattern/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/09_Web_Pattern/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(144 + 19),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/09_Web_Pattern/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/09_Web_Pattern/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(148 + 13),
        ),
    '10_Cabinet_Configuration_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(
                _DATA_BASE_DIR, 'train/10_Cabinet_Configuration/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/10_Cabinet_Configuration/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(112 + 158),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR, 'eval/10_Cabinet_Configuration/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/10_Cabinet_Configuration/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(15 + 24),
            test_jsonl=os.path.join(
                _DATA_BASE_DIR, 'test/10_Cabinet_Configuration/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/10_Cabinet_Configuration/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(8 + 15),
        ),
    '11_Power_Consumption_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'train/11_Power_Consumption/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/11_Power_Consumption/mave_*_bilstm_crf_tfrecord-*'),
            train_size=(140 + 132),
            eval_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'eval/11_Power_Consumption/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/11_Power_Consumption/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(27 + 16),
            test_jsonl=os.path.join(_DATA_BASE_DIR,
                                    'test/11_Power_Consumption/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/11_Power_Consumption/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(17 + 16),
        ),
    '12_Front_Camera_Resolution_bilstm_crf':
        Dataset(
            train_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'train/12_Front_Camera_Resolution/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'train/12_Front_Camera_Resolution/mave_*_bilstm_crf_tfrecord-*'
            ),
            train_size=(140 + 132),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR, 'eval/12_Front_Camera_Resolution/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'eval/12_Front_Camera_Resolution/mave_*_bilstm_crf_tfrecord-*'),
            eval_size=(27 + 16),
            test_jsonl=os.path.join(
                _DATA_BASE_DIR, 'test/12_Front_Camera_Resolution/mave_*.jsonl'),
            test_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'test/12_Front_Camera_Resolution/mave_*_bilstm_crf_tfrecord-*'),
            test_size=(17 + 16),
        ),
    'holdout_01_Remain_bert':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'holdout/01_Remain/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR, 'holdout/01_Remain/mave_*_bert_tfrecord-*'),
            train_size=(1481054 + 2625039),
            eval_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR, 'holdout/0[^01]_*/mave_*.jsonl'),
                os.path.join(_DATA_BASE_DIR, 'holdout/1?_*/mave_*.jsonl')
            ]),
            eval_tf_records=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/0[^01]_*/mave_*_bert_tfrecord-*'),
                os.path.join(_DATA_BASE_DIR,
                             'holdout/1?_*/mave_*_bert_tfrecord-*')
            ]),
            eval_size=(121860 + 62499),
        ),
    'holdout_01_Remain_etc':
        Dataset(
            train_jsonl=os.path.join(_DATA_BASE_DIR,
                                     'holdout/01_Remain/mave_*.jsonl'),
            train_tf_records=os.path.join(
                _DATA_BASE_DIR, 'holdout/01_Remain/mave_*_etc_tfrecord-*'),
            train_size=(1481054 + 2625039),
            eval_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR, 'holdout/0[^01]_*/mave_*.jsonl'),
                os.path.join(_DATA_BASE_DIR, 'holdout/1?_*/mave_*.jsonl')
            ]),
            eval_tf_records=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/0[^01]_*/mave_*_etc_tfrecord-*'),
                os.path.join(_DATA_BASE_DIR,
                             'holdout/1?_*/mave_*_etc_tfrecord-*')
            ]),
            eval_size=(121860 + 62499),
        ),
    'fewshot_001_bert':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp001/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_bert_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp001/mave_*_bert_tfrecord-*'
                )
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_bert_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_002_bert':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp002/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_bert_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp002/mave_*_bert_tfrecord-*'
                )
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_bert_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_003_bert':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp003/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_bert_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp003/mave_*_bert_tfrecord-*'
                )
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_bert_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_005_bert':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp005/mave_*.jsonl')
            ]),
            train_size=(20000 + 20000),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_bert_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp005/mave_*_bert_tfrecord-*'
                )
            ]),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_bert_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_010_bert':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp010/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_bert_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp010/mave_*_bert_tfrecord-*'
                )
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_bert_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_050_bert':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp050/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_bert_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp050/mave_*_bert_tfrecord-*'
                )
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_bert_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_100_bert':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp100/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_bert_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp100/mave_*_bert_tfrecord-*'
                )
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_bert_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_001_etc':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp001/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_etc_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp001/mave_*_etc_tfrecord-*')
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_etc_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_002_etc':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp002/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_etc_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp002/mave_*_etc_tfrecord-*')
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_etc_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_003_etc':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp003/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_etc_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp003/mave_*_etc_tfrecord-*')
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_etc_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_005_etc':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp005/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_etc_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp005/mave_*_etc_tfrecord-*')
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_etc_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_010_etc':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp010/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_etc_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp010/mave_*_etc_tfrecord-*')
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_etc_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_050_etc':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp050/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_etc_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp050/mave_*_etc_tfrecord-*')
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_etc_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
    'fewshot_100_etc':
        Dataset(
            train_jsonl=','.join([
                os.path.join(_DATA_BASE_DIR,
                             'holdout/01_Remain/sample/sppca010/mave_*.jsonl'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp100/mave_*.jsonl')
            ]),
            train_tf_records=','.join([
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/01_Remain/sample/sppca010/mave_*_etc_tfrecord-*'),
                os.path.join(
                    _DATA_BASE_DIR,
                    'holdout/?[234567890]_*/sample/sp100/mave_*_etc_tfrecord-*')
            ]),
            train_size=(20000 + 20000),
            eval_jsonl=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*.jsonl'),
            eval_tf_records=os.path.join(
                _DATA_BASE_DIR,
                'holdout/?[234567890]_*/sample/remain/mave_*_etc_tfrecord-*'),
            eval_size=(4063 + 105808 + 4402 + 5028 + 36 + 273 + 297 + 346 +
                       707 + 16210 + 16042 + 12057 + 9409 + 6728 + 409 + 334 +
                       249 + 161),
        ),
})
