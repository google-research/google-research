# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default configuration parameters."""

CONFIG = {
    'string': {
        'dataset': ('Dataset', 'movielens1m'),
        'model': ('Model', 'DistE'),
        'data_dir':
            ('Path to data directory',
             'data/'),
        'save_dir': ('Path to logs directory',
                     'logs/'),
        'loss_fn': ('Loss function to use', 'PairwiseHinge'),
        'initializer': ('Which initializer to use', 'GlorotUniform'),
        'regularizer': ('Regularizer', 'L2'),
        'optimizer': ('Optimizer', 'Adam'),
        'bias': ('Bias term', 'learn'),
        'dtype': ('Precision to use', 'float64'),
    },
    'float': {
        'lr': ('Learning rate', 1e-3),
        'lr_decay': ('Learning rate decay', 0.96),
        'min_lr': ('Minimum learning rate decay', 1e-5),
        'gamma': ('Margin for distance-based losses', 0),
        'item_reg': ('Regularization weight for item embeddings', 0),
        'user_reg': ('Regularization weight for user embeddings', 0),
        'm': ('Margin for hinge based models', 1)
    },
    'integer': {
        'patience': ('Number of validation steps before early stopping', 30),
        'valid': ('Number of epochs before computing validation metrics', 10),
        'checkpoint': ('Number of epochs before checkpointing the model', 5),
        'max_epochs': ('Maximum number of epochs to train for', 100),
        'rank': ('Embeddings dimension', 32),
        'batch_size': ('Batch size', 1000),
        'neg_sample_size':
            ('Negative sample size, -1 to use loss without negative sampling',
             1),
    },
    'boolean': {
        'train_c': ('Whether to train the hyperbolic curvature or not', False),
        'debug': ('If debug is true, only use 1000 examples for'
                  ' debugging purposes', False),
        'save_logs':
            ('Whether to save the training logs or print to stdout', True),
        'save_model': ('Whether to save the model weights', False),
        'double_neg': ('Whether to use double negative sampling or not', False)
    }
}

