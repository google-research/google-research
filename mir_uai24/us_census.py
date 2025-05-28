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

"""Utilities for the US Census dataset."""

import pandas as pd

from mir_uai24 import enum_utils


TRAIN_BAGS_DATA_PATH = 'mir_uai24/datasets/us_census/processed/bags_train.ftr'
TRAIN_INSTANCE_DATA_PATH = 'mir_uai24/datasets/us_census/processed/instance_train.ftr'
VAL_DATA_PATH = 'mir_uai24/datasets/us_census/processed/val.ftr'
TEST_DATA_PATH = 'mir_uai24/datasets/us_census/processed/test.ftr'


BAG_SIZE = 16
# pylint: disable=bad-whitespace
FEATURES = [
    'AGE',
    'SEX_1', 'SEX_2',
    'MARST_1', 'MARST_2', 'MARST_3', 'MARST_4', 'MARST_5', 'MARST_6',
    'CHBORN_0',  'CHBORN_1',  'CHBORN_2',  'CHBORN_3',  'CHBORN_4',
    'CHBORN_5',  'CHBORN_6',  'CHBORN_7',  'CHBORN_8',  'CHBORN_9',
    'CHBORN_10', 'CHBORN_11', 'CHBORN_12', 'CHBORN_13', 'CHBORN_14',
    'CHBORN_15', 'CHBORN_16', 'CHBORN_17', 'CHBORN_18', 'CHBORN_19',
    'CHBORN_20', 'CHBORN_21', 'CHBORN_99',
    'SCHOOL_1', 'SCHOOL_2',
    'EMPSTAT_0', 'EMPSTAT_1', 'EMPSTAT_2', 'EMPSTAT_3',
    'OCC_1',   'OCC_2',   'OCC_3',   'OCC_4',   'OCC_5',   'OCC_6',
    'OCC_7',   'OCC_8',   'OCC_9',   'OCC_10',  'OCC_11',  'OCC_12',
    'OCC_13',  'OCC_14',  'OCC_15',  'OCC_16',  'OCC_17',  'OCC_18',
    'OCC_19',  'OCC_20',  'OCC_21',  'OCC_22',  'OCC_23',  'OCC_24',
    'OCC_25',  'OCC_26',  'OCC_27',  'OCC_28',  'OCC_29',  'OCC_30',
    'OCC_31',  'OCC_32',  'OCC_33',  'OCC_34',  'OCC_35',  'OCC_36',
    'OCC_37',  'OCC_38',  'OCC_39',  'OCC_40',  'OCC_41',  'OCC_42',
    'OCC_43',  'OCC_44',  'OCC_45',  'OCC_98',  'OCC_99',  'OCC_100',
    'OCC_102', 'OCC_104', 'OCC_106', 'OCC_108', 'OCC_110', 'OCC_112',
    'OCC_114', 'OCC_116', 'OCC_118', 'OCC_120', 'OCC_122', 'OCC_124',
    'OCC_126', 'OCC_128', 'OCC_130', 'OCC_132', 'OCC_134', 'OCC_136',
    'OCC_156', 'OCC_200', 'OCC_210', 'OCC_220', 'OCC_222', 'OCC_224',
    'OCC_226', 'OCC_236', 'OCC_240', 'OCC_242', 'OCC_244', 'OCC_246',
    'OCC_248', 'OCC_250', 'OCC_252', 'OCC_254', 'OCC_256', 'OCC_258',
    'OCC_260', 'OCC_266', 'OCC_270', 'OCC_272', 'OCC_274', 'OCC_276',
    'OCC_278', 'OCC_280', 'OCC_282', 'OCC_284', 'OCC_286', 'OCC_298',
    'OCC_300', 'OCC_302', 'OCC_304', 'OCC_306', 'OCC_308', 'OCC_310',
    'OCC_312', 'OCC_314', 'OCC_316', 'OCC_318', 'OCC_320', 'OCC_322',
    'OCC_324', 'OCC_326', 'OCC_327', 'OCC_328', 'OCC_330', 'OCC_332',
    'OCC_334', 'OCC_336', 'OCC_338', 'OCC_340', 'OCC_342', 'OCC_344',
    'OCC_346', 'OCC_348', 'OCC_350', 'OCC_352', 'OCC_354', 'OCC_356',
    'OCC_358', 'OCC_362', 'OCC_364', 'OCC_366', 'OCC_368', 'OCC_370',
    'OCC_372', 'OCC_374', 'OCC_376', 'OCC_378', 'OCC_380', 'OCC_382',
    'OCC_384', 'OCC_386', 'OCC_388', 'OCC_390', 'OCC_392', 'OCC_394',
    'OCC_396', 'OCC_398', 'OCC_400', 'OCC_402', 'OCC_404', 'OCC_406',
    'OCC_408', 'OCC_410', 'OCC_412', 'OCC_414', 'OCC_416', 'OCC_418',
    'OCC_420', 'OCC_430', 'OCC_432', 'OCC_434', 'OCC_436', 'OCC_438',
    'OCC_440', 'OCC_442', 'OCC_444', 'OCC_446', 'OCC_448', 'OCC_450',
    'OCC_452', 'OCC_454', 'OCC_456', 'OCC_458', 'OCC_460', 'OCC_462',
    'OCC_464', 'OCC_466', 'OCC_468', 'OCC_470', 'OCC_472', 'OCC_474',
    'OCC_476', 'OCC_478', 'OCC_480', 'OCC_482', 'OCC_484', 'OCC_486',
    'OCC_488', 'OCC_496', 'OCC_500', 'OCC_510', 'OCC_520', 'OCC_600',
    'OCC_602', 'OCC_604', 'OCC_606', 'OCC_608', 'OCC_610', 'OCC_612',
    'OCC_614', 'OCC_700', 'OCC_710', 'OCC_712', 'OCC_714', 'OCC_720',
    'OCC_730', 'OCC_732', 'OCC_740', 'OCC_750', 'OCC_760', 'OCC_770',
    'OCC_780', 'OCC_790', 'OCC_792', 'OCC_794', 'OCC_796', 'OCC_798',
    'OCC_844', 'OCC_866', 'OCC_888', 'OCC_900', 'OCC_902', 'OCC_904',
    'OCC_906', 'OCC_908', 'OCC_910', 'OCC_985', 'OCC_988', 'OCC_995',
    'OCC_996', 'OCC_998',
    'IND_1',   'IND_2',   'IND_3',   'IND_4',   'IND_5',   'IND_6',
    'IND_7',   'IND_8',   'IND_9',   'IND_10',  'IND_11',  'IND_12',
    'IND_13',  'IND_14',  'IND_15',  'IND_16',  'IND_17',  'IND_18',
    'IND_19',  'IND_20',  'IND_21',  'IND_22',  'IND_23',  'IND_24',
    'IND_25',  'IND_26',  'IND_27',  'IND_28',  'IND_29',  'IND_30',
    'IND_31',  'IND_32',  'IND_33',  'IND_34',  'IND_35',  'IND_36',
    'IND_37',  'IND_38',  'IND_39',  'IND_40',  'IND_41',  'IND_42',
    'IND_43',  'IND_44',  'IND_45',  'IND_46',  'IND_47',  'IND_48',
    'IND_49',  'IND_50',  'IND_51',  'IND_52',  'IND_53',  'IND_54',
    'IND_55',  'IND_56',  'IND_57',  'IND_58',  'IND_59',  'IND_60',
    'IND_61',  'IND_62',  'IND_63',  'IND_64',  'IND_65',  'IND_66',
    'IND_67',  'IND_68',  'IND_69',  'IND_70',  'IND_71',  'IND_72',
    'IND_73',  'IND_74',  'IND_75',  'IND_76',  'IND_77',  'IND_78',
    'IND_79',  'IND_80',  'IND_81',  'IND_82',  'IND_83',  'IND_84',
    'IND_85',  'IND_86',  'IND_87',  'IND_88',  'IND_89',  'IND_90',
    'IND_91',  'IND_92',  'IND_93',  'IND_94',  'IND_95',  'IND_96',
    'IND_97',  'IND_98',  'IND_99',  'IND_100', 'IND_101', 'IND_102',
    'IND_103', 'IND_104', 'IND_105', 'IND_106', 'IND_107', 'IND_108',
    'IND_109', 'IND_110', 'IND_111', 'IND_112', 'IND_113', 'IND_114',
    'IND_115', 'IND_116', 'IND_117', 'IND_118', 'IND_119', 'IND_120',
    'IND_121', 'IND_122', 'IND_123', 'IND_124', 'IND_125', 'IND_126',
    'IND_127', 'IND_128', 'IND_129', 'IND_130', 'IND_131', 'IND_995',
    'IND_996', 'IND_998']
# pylint: enable=bad-whitespace


def get_info(
    return_bags_df = False
):
  """Returns US Census dataset info.

  Args:
    return_bags_df: Whether to return the loaded bags dataframe.

  Returns:
    If return_bags_df is True, returns a tuple of dataset info and bags
    dataframe. Otherwise, returns dataset info object.
  """
  features = [
      enum_utils.Feature(key=col, type=enum_utils.FeatureType.REAL)
      for col in FEATURES
  ]
  bags_df = pd.read_feather(TRAIN_BAGS_DATA_PATH)
  n_instances = len(bags_df) * BAG_SIZE

  dataset_info = enum_utils.DatasetInfo(
      bag_id='bag_id',
      instance_id='instance_id',
      bag_id_x_instance_id='bag_id_X_instance_id',
      bag_size=BAG_SIZE,
      n_instances=n_instances,
      features=features,
      label='WKSWORK1',
      memberships=enum_utils.DatasetMembershipInfo(
          instances=dict(), bags=dict()))

  if return_bags_df:
    return dataset_info, bags_df
  return dataset_info
