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

"""Constants file for preprocessing."""

SSCL_COLUMNS = [
    'sale',
    'sales_amount_in_euro',
    'time_delay_for_conversion',
    'click_timestamp',
    'nb_clicks_1week',
    'product_price',
    'product_age_group',
    'device_type',
    'audience_id',
    'product_gender',
    'product_brand',
    'product_category_1',
    'product_category_2',
    'product_category_3',
    'product_category_4',
    'product_category_5',
    'product_category_6',
    'product_category_7',
    'product_country',
    'product_id',
    'product_title',
    'partner_id',
    'user_id',
]
SSCL_CATEGORICAL_COLUMNS = [
    'product_age_group',
    'device_type',
    'audience_id',
    'product_gender',
    'product_brand',
    'product_category_1',
    'product_category_2',
    'product_category_3',
    'product_category_4',
    'product_category_5',
    'product_category_6',
    'product_category_7',
    'product_country',
    'product_id',
    'product_title',
    'partner_id',
    'user_id',
]
SSCL_NUMERICAL_COLUMNS = [
    'time_delay_for_conversion',
    'nb_clicks_1week',
    'product_price',
]
SSCL_TARGET_COLUMN = 'sales_amount_in_euro'
