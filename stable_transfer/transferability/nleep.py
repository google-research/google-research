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

"""Implementation of NLEEP (Ranking Neural Checkpoints).

Li, Yandong, et al. "Ranking neural checkpoints." Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 2021.
https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Ranking_Neural_Checkpoints_CVPR_2021_paper.pdf
"""

from sklearn.mixture import GaussianMixture
import tensorflow as tf

from stable_transfer.transferability import leep
from stable_transfer.transferability import transfer_experiment


def get_nleep_score(
    features,
    target_labels,
    num_components_gmm,
    random_state=123):
  """Computes NLEEP (Ranking Neural Checkpoints).

  Args:
    features: matrix [N, D] of source features obtained from the target data,
      where N is the number of datapoints and D their dimensionionality.
    target_labels: ground truth target labels of dimension [N, 1].
    num_components_gmm: gaussian components (5*target class number in the paper)
    random_state: random seed for the GaussianMixture initialization.

  Returns:
    nleep: transferability metric score.

  """
  gmm = GaussianMixture(
      n_components=num_components_gmm, random_state=random_state).fit(features)
  gmm_predictions = gmm.predict_proba(features)
  nleep = leep.get_leep_score(gmm_predictions.astype('float32'), target_labels)
  return nleep


@transfer_experiment.load_or_compute
def get_train_nleep(experiment):
  """Compute NLEEP on the target training data."""
  features, labels = experiment.model_output_on_target_train_dataset('features')
  if 'num_components_gmm' in experiment.config.experiment.nleep:
    num_components_gmm = experiment.config.experiment.nleep.num_components_gmm
  else:  # Set the default value as in the paper.
    num_components_gmm = int((tf.reduce_max(labels) + 1) * 5)
  nleep = get_nleep_score(features, labels, num_components_gmm)
  return dict(nleep=float(nleep))
