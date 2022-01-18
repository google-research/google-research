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

# lint as: python3
"""Main file to run AwA experiments."""
import os
import awa_helper
import ipca

DATA_DIR = '/mnt/disks/mndir/Animals_with_Attributes2/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train/')
VALID_DIR = os.path.join(DATA_DIR, 'val/')
SIZE = (299, 299)
BATCH_SIZE = 64
n_concept = 8
pretrained = False

if __name__ == '__main__':

  y_train_logit, y_val_logit, y_train, \
      y_val, f_train, f_val, dense2, predict = awa_helper.load_data(TRAIN_DIR,
                                                                    SIZE,
                                                                    BATCH_SIZE,
                                                                    pretrained,
                                                                    noise=0.0)

  concept_arraynew, concept_arraynew_active, \
      concept_list, active_list = awa_helper.load_conceptarray()

  finetuned_model_pr = ipca.ipca_model(
      concept_arraynew_active,
      dense2,
      predict,
      f_train,
      y_train_logit,
      f_val,
      y_val_logit,
      n_concept,
      verbose=True,
      epochs=150,
      metric='accuracy')

  num_epoch = 50
  for _ in range(num_epoch):
    finetuned_model_pr.fit(
        f_train,
        y_train_logit,
        batch_size=100,
        epochs=10,
        verbose=1,
        validation_data=(f_val, y_val_logit))

  concept_matrix = finetuned_model_pr.layers[-5].get_weights()[0]

  # Plots nearest neighbors in each cluster for each concept.
  awa_helper.plot_nearestneighbors(concept_arraynew_active, concept_matrix,
                                   concept_list, active_list)

  # Calculates conceptSHAP.
  shap_model = ipca.ipca_model_shap(dense2, predict, n_concept, 1024,
                                    concept_matrix)

  print(ipca.get_shap(n_concept, f_val, y_val_logit, shap_model, 0.94, 0.019))
