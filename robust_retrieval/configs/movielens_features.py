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

"""Feature configs for MovieLens Dataset.

The feature_config contains query features and item features. The information
of each feature is a dictionary. Key is the feature name. For each feature:
  * 'ftype' is the feature type, such as `sparse`, `bucketized`, `dense`.
  * 'dtype' is the data type for the feature, such as`string`, 'int64`,
  `float32`.
  * 'emb_dim' is the feature embedding dimension.
  * 'vocab_file_name' is the feature vocabulary file name.
  * 'boundaries_file_name' is the feature boundaries for bucketized feature.
"""

feature_config = {
    # Query features
    'query_features': {
        'user_id': {
            'ftype': 'sparse',
            'dtype': 'string',
            'emb_dim': 128,
            'vocab_file_name': 'user_id'
        },
        'user_rank': {
            'ftype': 'sparse',
            'dtype': 'string',
            'emb_dim': 16,
            'vocab_file_name': 'user_rank'
        },
    },

    # Item features
    'item_features': {
        'item_id': {
            'ftype': 'sparse',
            'dtype': 'string',
            'emb_dim': 128,
            'vocab_file_name': 'item_id'
        },
        'item_rank': {
            'ftype': 'sparse',
            'dtype': 'string',
            'emb_dim': 16,
            'vocab_file_name': 'item_rank'
        },
        'genres': {
            'ftype': 'sparse',
            'dtype': 'string',
            'emb_dim': 16,
            'vocab_file_name': 'genres'
        },
    }
}
