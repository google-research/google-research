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

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import data
import model
# pylint: skip-file

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--real_path', default='../data/resize128')
parser.add_argument('--fake_path', default='../data/fake')
parser.add_argument('--train_label', default='../data/annotations/train_label.txt')
parser.add_argument('--test_label', default='../data/annotations/test_label.txt')
parser.add_argument('--valid_label', default='../data/annotations/val_label.txt')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 20]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--n_class', type=int, default=2, help='Number of class [default: 2]')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate [default: 0.1]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: momentum]')
FLAGS = parser.parse_args()



ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
      'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
      'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
      'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
      'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
      'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
      'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
      'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
      'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
      'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
      'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
      'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
      'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
# tf.set_random_seed(0)# 0 for 512
tf.set_random_seed(100)

(train_images, train_labels, train_att), train_iters = data.data_train(FLAGS.real_path, FLAGS.train_label, 64)
(fake_images, fake_labels, fake_att), fake_iters = data.data_fake(FLAGS.fake_path, FLAGS.train_label, 64)
(valid_images, valid_labels, valid_att), valid_iters = data.data_test(FLAGS.real_path, FLAGS.valid_label, FLAGS.batch_size)
(test_images, test_labels, test_att), test_iters = data.data_test(FLAGS.real_path, FLAGS.test_label, FLAGS.batch_size)

batch_images = tf.placeholder(tf.float32,[None,128,128,3])
batch_labels = tf.placeholder(tf.int32,[None,])
is_training = tf.placeholder(tf.bool)
lr_ph = tf.placeholder(tf.float32)
lr = FLAGS.lr

Y_score = model.vgg(batch_images, FLAGS.n_class, is_training)
Y_hat = tf.nn.softmax(Y_score)
Y_pred = tf.argmax(Y_hat, 1)
Y_label = tf.to_float(tf.one_hot(batch_labels, FLAGS.n_class))

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Y_score, labels = Y_label)
loss_op = tf.reduce_mean(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y_label, 1))
acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
update_op = tf.train.MomentumOptimizer(lr_ph, FLAGS.momentum).minimize(loss_op)
init = tf.global_variables_initializer()

print("================\n\n",train_iters, fake_iters)

with tf.Session() as sess:
  sess.run(init)
  for i in range(FLAGS.max_epoch):
    if i == 30:
      lr *= 0.1
    elif i == 40:
      lr *= 0.1

    for j in range(train_iters):
      co_images, co_labels = sess.run([train_images,train_labels])
      # tr_images, tr_labels = sess.run([train_images,train_labels])
      # fa_images, fa_labels = sess.run([fake_images,fake_labels])
      # co_images = np.concatenate((tr_images,fa_images),axis=0)
      # co_labels = np.concatenate((tr_labels,fa_labels),axis=0)
      loss, acc, _ = sess.run([loss_op, acc_op, update_op], {batch_images:co_images, batch_labels:co_labels, lr_ph:lr, is_training:True})
      if j % 50 == 0:
        print('====epoch_%d====iter_%d: loss=%.4f, train_acc=%.4f' % (i, j, loss, acc))

    valid_acc = 0.0
    y_pred =[]
    y_label = []
    y_att = []
    for k in range(valid_iters):
      va_images, va_labels, va_att = sess.run([valid_images, valid_labels, valid_att])
      batch_acc, batch_pred = sess.run([acc_op,Y_pred], {batch_images:va_images, batch_labels:va_labels, is_training:False})
      valid_acc += batch_acc
      y_pred += batch_pred.tolist()
      y_label += va_labels.tolist()
      y_att += va_att.tolist()
    valid_acc = valid_acc / float(valid_iters)
    valid_eo = data.cal_eo(y_att, y_label, y_pred)
    print('====epoch_%d: valid_acc=%.4f, valid_eo=%.4f' % (i, valid_acc, valid_eo[-1]))
    print('eo: ',valid_eo[0],valid_eo[1])
    print('eo: ',valid_eo[2],valid_eo[3])


