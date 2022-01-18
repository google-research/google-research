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
        'model': ('Model', 'CFTreeModel'),
        'data_dir':
            ('Path to data directory',
             'data/'),
        'save_dir': ('Path to logs directory',
                     'logs/'),
        'loss_fn': ('Loss function to use', 'SeperationHingeTreeLossFn'),
        'initializer': ('Which initializer to use', 'GlorotUniform'),
        'optimizer': ('Optimizer', 'Adam'),
        'dtype': ('Precision to use', 'float64'),
    },
    'float': {
        'lr': ('Learning rate', 1),
        'lr_decay': ('Learning rate decay', 0.96),
        'min_lr': ('Minimum learning rate decay', 1e-5),
        'm': ('Margin for hinge based models', 0.5),
        'sep_w': ('seperation weight for seperation loss', 0.5)
    },
    'integer': {
        'patience': ('Number of validation steps before early stopping', 30),
        'valid': ('Number of epochs before computing validation metrics', 10),
        'checkpoint': ('Number of epochs before checkpointing the model', 10),
        'max_epochs': ('Maximum number of epochs to train for', 100),
        'rank': ('Embeddings dimension', 16),
        'batch_size': ('Batch size', 1000),
        'k_threshold': ('Threshold for top_k eval (see models/base)', 2),
    },
    'boolean': {
        'train_c': ('Whether to train the hyperbolic curvature or not', False),
        'debug': ('If debug is true, only use 1000 examples for'
                  ' debugging purposes', False),
        'save_logs':
            ('Whether to save the training logs or print to stdout', True),
        'save_model': ('Whether to save the model weights', False),
        'stop_grad': ('Whether to stop gradients in node interaction', False),
    },
    'intlist': {
        'nodes_per_level':
            ('How many nodes per level, not including leaves and root',
             [1024]),
        'node_batch_per_level':
            ('Node batch size per level, not including leaves and root', [64]),
        'top_k': ('list of top k values', [1, 10, 50, 100]),
    },
    'floatlist': {
        'radii': ('Radius of each level including leaves, not including root',
                  [5.0, 6.0])
    }
}

