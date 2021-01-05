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

"""Helper file to run the discover concept algorithm in the AwA dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import pickle
from absl import app
import keras
import keras.backend as K
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import tensorflow.compat.v1 as tf


def load_model(classes):
  """Loads the pretrained model."""
  model = keras.applications.inception_v3.InceptionV3(include_top=True,
                                                      weights='imagenet')
  model.layers.pop()
  for layer in model.layers:
    layer.trainable = False
  last = model.layers[-1].output
  dense1 = Dense(1024, activation='relu', name='concept1')
  dense2 = Dense(1024, activation='relu', name='concept2')
  fc1 = dense1(last)
  fc2 = dense2(fc1)
  predict = Dense(len(classes), name='output')
  logits = predict(fc2)
  def cross_entropy_loss():
    # Returns the cross entropy loss.
    def loss(y_true, y_pred):
      return tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits_v2(
              labels=y_true, logits=y_pred))

    return loss
  finetuned_model = Model(model.input, logits)
  finetuned_model.compile(
      optimizer=SGD(lr=0.01),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])

  finetuned_model.classes = classes
  finetuned_model.load_weights('inception_final.h5')
  feature_dense_model = Model(model.input, fc1)
  fc1_input = Input(shape=(1024,))
  fc2_temp = dense2(fc1_input)
  logits_temp = predict(fc2_temp)
  fc_model = Model(fc1_input, logits_temp)
  fc_model.compile(
      optimizer=SGD(lr=0.01),
      loss=cross_entropy_loss(),
      metrics=['accuracy', 'top_k_categorical_accuracy'])

  for layer in finetuned_model.layers:
    layer.trainable = False
  return finetuned_model, feature_dense_model, fc_model, dense2, predict


def load_data(train_dir, size, batch_size,
              pretrained=True, noise=0.):
  """Loads data and adding noise."""
  def rand_noise(img):
    img_noisy = img + np.random.normal(scale=noise, size=img.shape)
    return img_noisy
  if not pretrained:
    gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.1)
    gen_noisy = keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, preprocessing_function=rand_noise)
    aug = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1)

    batches = aug.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size,
        subset='training')
    batches_fix_train = gen.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='training')
    batches_fix_val = gen_noisy.flow_from_directory(
        train_dir,
        target_size=size,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        subset='validation')

    classes = list(iter(batches.class_indices))
    for c in batches.class_indices:
      classes[batches.class_indices[c]] = c
    num_train_samples = sum([len(files) for _, _, files in os.walk(train_dir)])
    num_train_steps = math.floor(num_train_samples * 0.9 / batch_size)
    num_valid_steps = math.floor(num_train_samples * 0.1 / batch_size)

    y_train = batches_fix_train.classes
    y_val = batches_fix_val.classes
    _, feature_dense_model, _, \
        dense2, predict = load_model(classes)
    f_train = feature_dense_model.predict_generator(
        batches_fix_train,
        steps=num_train_steps,
        workers=40,
        use_multiprocessing=False)
    f_val = feature_dense_model.predict_generator(
        batches_fix_val,
        steps=num_valid_steps,
        workers=40,
        use_multiprocessing=False)
    y_train_logit = tf.keras.utils.to_categorical(
        y_train[:f_train.shape[0]], num_classes=50)
    y_val_logit = tf.keras.utils.to_categorical(
        y_val[:f_val.shape[0]],
        num_classes=50,
    )
    np.save('y_train_logit.npy', y_train_logit)
    np.save('y_val_logit.npy', y_val_logit)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('f_train.npy', f_train)
    np.save('f_val.npy', f_val)
    with open('classes.pickle', 'wb') as handle:
      pickle.dump(classes, handle, pickle.HIGHEST_PROTOCOL)
  else:
    with open('classes.pickle', 'rb') as handle:
      classes = pickle.load(handle)
    _, feature_dense_model, _, \
        dense2, predict = load_model(classes)
    y_train_logit = np.load('y_train_logit.npy')
    y_val_logit = np.load('y_val_logit.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    f_train = np.load('f_train.npy')
    f_val = np.load('f_val.npy')
  return y_train_logit, y_val_logit, y_train, y_val, \
      f_train, f_val, dense2, predict


def target_category_loss(x, category_index, nb_classes):
  return x* K.one_hot([category_index], nb_classes)


def get_ace_concept(concept_arraynew_active, dense2, predict, f_train,
                    concepts_to_select):
  """Calculates the ACE concepts."""
  concept_input = Input(shape=(1024,), name='concept_input')
  fc2_tcav = dense2(concept_input)
  softmax_tcav = predict(fc2_tcav)
  tcav_model = Model(inputs=concept_input, outputs=softmax_tcav)
  tcav_model.layers[-1].activation = None
  tcav_model.layers[-1].trainable = False
  tcav_model.layers[-2].trainable = False
  tcav_model.compile(
      loss='mean_squared_error',
      optimizer=SGD(lr=0.0),
      metrics=['binary_accuracy'])
  tcav_model.summary()

  n_cluster = concept_arraynew_active.shape[0]
  n_percluster = concept_arraynew_active.shape[1]
  print(concept_arraynew_active.shape)
  weight_ace = np.zeros((1024, n_cluster))
  tcav_list_rand = np.zeros((50, 200))
  tcav_list_ace = np.zeros((50, 134))
  for i in range(n_cluster):
    y = np.zeros((n_cluster * n_percluster))
    y[i * n_percluster:(i + 1) * n_percluster] = 1
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 1024)), y)
    weight_ace[:, i] = clf.coef_

  weight_rand = np.zeros((1024, 200))
  for i in range(200):
    y = np.random.randint(2, size=n_cluster * n_percluster)
    clf = LogisticRegression(
        random_state=0,
        solver='lbfgs',
        max_iter=10000,
        C=10.0,
        multi_class='ovr').fit(concept_arraynew_active.reshape((-1, 1024)), y)
    weight_rand[:, i] = clf.coef_

  sig_list = np.zeros(n_cluster)

  for j in range(50):
    grads = (
        K.gradients(target_category_loss(softmax_tcav, j, 50),
                    concept_input)[0])
    gradient_function = K.function([tcav_model.input], [grads])
    grads_val = gradient_function([f_train])[0]
    grad_rand = np.matmul(grads_val, weight_rand)
    grad_ace = np.matmul(grads_val, weight_ace)
    tcav_list_rand[j, :] = np.sum(grad_rand > 0.000, axis=(0))
    tcav_list_ace[j, :] = np.sum(grad_ace > 0.000, axis=(0))
    mean = np.mean(tcav_list_rand[j, :])
    std = np.std(tcav_list_rand[j, :])
    sig_list += (tcav_list_ace[j, :] > mean + std * 2.0).astype(int)
  top_k_index = np.array(sig_list).argsort()[-1 * concepts_to_select:][::-1]
  print(sig_list)
  print(top_k_index)
  return weight_ace[:, top_k_index]


def get_pca_concept(f_train, concepts_to_select):
  pca = PCA()
  pca.fit(f_train)
  weight_pca = np.zeros((1024, concepts_to_select))
  for count, pc in enumerate(pca.components_):
    if count >= concepts_to_select:
      break
    weight_pca[:, count] = pc
  return weight_pca


def load_conceptarray():
  """Loads a preprocessed concept array (by running code of ACE)."""
  concept_arraynew = np.load('concept_arraynew.npy')
  with open('concept_list.pickle', 'rb') as handle:
    concept_list = pickle.load(handle)
  with open('active_list.pickle', 'rb') as handle:
    active_list = pickle.load(handle)
  concept_arraynew_active = np.load('concept_arraynew_active.npy')
  return concept_arraynew, concept_arraynew_active, concept_list, active_list


def plot_nearestneighbors(concept_arraynew_active, concept_matrix, concept_list,
                          active_list, filename='top_concepts_AwA'):
  """Plots nearest neighbors."""
  simarray1 = np.mean(
      np.matmul(concept_arraynew_active, concept_matrix), axis=1)
  simarray1 = simarray1 - np.mean(simarray1, axis=(0))
  simarray_0mean_unitnorm = (simarray1 / np.linalg.norm(simarray1, axis=0))
  top_cluster = (np.argmax(np.abs(simarray_0mean_unitnorm), axis=0))
  for top in top_cluster:
    print(concept_list[active_list[top]])
  neglist = np.abs(np.min(simarray_0mean_unitnorm, axis=0)) > np.max(
      simarray_0mean_unitnorm, axis=0)

  fig = plt.figure(figsize=(18, 28))
  for topc, top in enumerate(top_cluster):
    mypath = './work_dir/concepts/concept_fc1_' + concept_list[
        active_list[top]] + '/'
    onlyfiles = [f for f in os.listdir(mypath) if
                 os.path.isfile(os.path.join(mypath, f))]
    top_image = np.matmul(concept_arraynew_active[top, :, :],
                          concept_matrix[:, topc:topc + 1])
    if not neglist[topc]:
      top_idx = np.argsort(top_image[:, 0])[::-1][:8]
    else:
      top_idx = np.argsort(top_image[:, 0])[:8]
    tempcount = 0
    for count, image in enumerate(onlyfiles):
      if count in top_idx:
        tempcount += 1
        fig.add_subplot(8, 8, 8 * topc + tempcount)
        img = mpimg.imread(os.path.join(mypath, image))
        plt.imshow(img)
        plt.axis('off')
  plt.savefig(filename)


def main(_):
  return


if __name__ == '__main__':
  app.run(main)
