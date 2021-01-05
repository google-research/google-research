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

"""This is the code to run the discover concept algorithm in the toy dataset."""
#  lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
import ipca
import numpy as np
import toy_helper


def main(_):
  n_concept = 5
  n_cluster = 5
  n = 60000
  n0 = int(n * 0.8)

  pretrain = True
  # Loads data.
  x, y, concept = toy_helper.load_xyconcept(n, pretrain)
  if not pretrain:
    x_train = x[:n0, :]
    x_val = x[n0:, :]
  y_train = y[:n0, :]
  y_val = y[n0:, :]
  all_feature_dense = np.load('all_feature_dense.npy')
  f_train = all_feature_dense[:n0, :]
  f_val = all_feature_dense[n0:, :]
  # Loads model.
  if not pretrain:
    dense2, predict, _ = toy_helper.load_model(
        x_train, y_train, x_val, y_val, pretrain=pretrain)
  else:
    dense2, predict, _ = toy_helper.load_model(_, _, _, _, pretrain=pretrain)
  # Loads concepts.
  concept_arraynew = np.load('concept_arraynew.npy')
  concept_arraynew2 = np.load('concept_arraynew2.npy')

  for n_concept in range(1, 10):
    print(n_concept)
    # Discovers concept with true cluster.
    finetuned_model_pr = ipca.ipca_model(
        concept_arraynew2,
        dense2,
        predict,
        f_train,
        y_train,
        f_val,
        y_val,
        n_concept,
        comp1=True)
    num_epoch = 5
    for _ in range(num_epoch):
      finetuned_model_pr.fit(
          f_train,
          y_train,
          batch_size=50,
          epochs=10,
          verbose=True,
          validation_data=(f_val, y_val))
    # Evaluates groupacc and get concept_matrix.
    _, _ = ipca.get_groupacc(
        finetuned_model_pr,
        concept_arraynew2,
        f_train,
        f_val,
        concept,
        n_concept,
        n_cluster,
        n0,
        verbose=False)
    # Discovers concepts with self-discovered clusters.
    finetuned_model_pr = ipca.ipca_model(
        concept_arraynew,
        dense2,
        predict,
        f_train,
        y_train,
        f_val,
        y_val,
        n_concept,
        comp1=True)
    num_epoch = 5
    for _ in range(num_epoch):
      finetuned_model_pr.fit(
          f_train,
          y_train,
          batch_size=50,
          epochs=10,
          verbose=True,
          validation_data=(f_val, y_val))
    _, _ = ipca.get_groupacc(
        finetuned_model_pr,
        concept_arraynew,
        f_train,
        f_val,
        concept,
        n_concept,
        n_cluster,
        n0,
        verbose=False)

  for n_concept in range(1, 10):
    print(n_concept)
    concept_matrix_ace = toy_helper.get_ace_concept(concept_arraynew, dense2,
                                                    predict, f_val, n_concept)

    finetuned_model_pr_ace = ipca.ipca_model(
        concept_arraynew,
        dense2,
        predict,
        f_train,
        y_train,
        f_val,
        y_val,
        n_concept,
        verbose=True,
        epochs=0,
        metric='accuracy')

    finetuned_model_pr_ace.layers[-5].set_weights([concept_matrix_ace])

    print(finetuned_model_pr_ace.evaluate(f_val, y_val))

    _, _ = ipca.get_groupacc(
        finetuned_model_pr_ace,
        concept_arraynew,
        f_train,
        f_val,
        concept,
        n_concept,
        n_cluster,
        n0,
        verbose=False)

    concept_matrix_ace2 = toy_helper.get_ace_concept(concept_arraynew2, dense2,
                                                     predict, f_val, n_concept)

    finetuned_model_pr_ace2 = ipca.ipca_model(
        concept_arraynew2,
        dense2,
        predict,
        f_train,
        y_train,
        f_val,
        y_val,
        n_concept,
        verbose=True,
        epochs=0,
        metric='accuracy')

    finetuned_model_pr_ace2.layers[-5].set_weights([concept_matrix_ace2])

    print(finetuned_model_pr_ace2.evaluate(f_val, y_val))

    _, _ = ipca.get_groupacc(
        finetuned_model_pr_ace2,
        concept_arraynew2,
        f_train,
        f_val,
        concept,
        n_concept,
        n_cluster,
        n0,
        verbose=False)

    concept_matrix_pca = toy_helper.get_pca_concept(f_train, n_concept)

    finetuned_model_pr_pca = ipca.ipca_model(
        concept_arraynew,
        dense2,
        predict,
        f_train,
        y_train,
        f_val,
        y_val,
        n_concept,
        verbose=True,
        epochs=0,
        metric='accuracy')

    finetuned_model_pr_pca.layers[-5].set_weights([concept_matrix_pca])

    print(finetuned_model_pr_pca.evaluate(f_val, y_val))

    _, _ = ipca.get_groupacc(
        finetuned_model_pr_pca,
        concept_arraynew,
        f_train,
        f_val,
        concept,
        n_concept,
        n_cluster,
        n0,
        verbose=False)


if __name__ == '__main__':
  app.run(main)
