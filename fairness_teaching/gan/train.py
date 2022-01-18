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

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import skimage.io as io
import data
import model
import util
# pylint: skip-file

#####################  parser  #####################

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', default='../data/big', help='Path of all image files')
parser.add_argument('--train_label', default='../data/annotations/train_label.txt', help='Train labels file')
parser.add_argument('--test_label', default='../data/annotations/test_label.txt', help='Test labels file')
parser.add_argument('--valid_label', default='../data/annotations/val_label.txt', help='Valid labels file')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--n_epoch', type=int, default=60, help='Epoch to run [default: 60]')
parser.add_argument('--n_adv', type=int, default=20, help='Epoch to run adv_cls[default: 20]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training [default: 64]')
parser.add_argument('--sample_size', type=int, default=24, help='Sample size during validation [default: 24]')
parser.add_argument('--n_class', type=int, default=2, help='Number of class [default: 2]')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate [default: 0.1]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
FLAGS = parser.parse_args()

#####################  config  #####################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
IMG_PATH = FLAGS.img_path
TRAIN_LABEL = FLAGS.train_label
TEST_LABEL = FLAGS.test_label
VALID_LABEL = FLAGS.valid_label
BATCH_SIZE = FLAGS.batch_size
N_SAMPLE = FLAGS.sample_size
N_EPOCH = FLAGS.n_epoch
N_ADV = FLAGS.n_adv
N_CLASS = FLAGS.n_class
LR = FLAGS.lr
MOMENTUM = FLAGS.momentum
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_PATH = os.path.join(ROOT_PATH, FLAGS.log_dir)
if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
acc_count = 0
while True:
  if os.path.exists(os.path.join(LOG_PATH, 'log_%02d.txt' % acc_count)): acc_count += 1
  else: break
LOG_FNAME = 'log_%02d.txt' % acc_count
LOG_FOUT = open(os.path.join(LOG_PATH, LOG_FNAME), 'w')

(train_images, train_labels), train_iters = data.data_train(IMG_PATH, TRAIN_LABEL, BATCH_SIZE)
(valid_images, valid_labels), valid_iters = data.data_test(IMG_PATH, VALID_LABEL, N_SAMPLE)
Genc = model.Genc()
Gdec = model.Gdec()
D = model.D()
Adv = model.Adv_cls()

####################################################

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)

def G_graph(sess, phg):
  real_labels = train_labels * 2 -1
  fake_labels = -real_labels

  u = Genc.build(train_images, phg['is_training_g'])
  real_images = Gdec.build(u, real_labels, phg['is_training_g'])
  fake_images = Gdec.build(u, fake_labels, phg['is_training_g'])
  fake_gan_logit, fake_cls_logit = D.build(fake_images, phg['is_training_g'])

  gan_loss = -tf.reduce_mean(fake_gan_logit)
  cls_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims((1-train_labels),axis=-1), fake_cls_logit)
  rec_loss = tf.losses.absolute_difference(train_images, real_images)
  reg_loss = tf.reduce_sum(Genc.reg_loss+Gdec.reg_loss)

  final_loss = gan_loss + 10.0*cls_loss + 100.0*rec_loss + reg_loss
  update_op = tf.train.AdamOptimizer(phg['lr_g'], beta1=0.5, name='first_adam').minimize(final_loss, var_list=Genc.vars+Gdec.vars)

  return u, final_loss, update_op

def G_adv_graph(sess, phg):
  real_labels = train_labels * 2 -1
  fake_labels = -real_labels

  u = Genc.build(train_images, phg['is_training_g'])

  Y_score = Adv.build(u, N_CLASS, phg['is_training_g'])
  Y_hat = tf.nn.softmax(Y_score)
  Y_pred = tf.argmax(Y_hat, 1)
  Y_label = tf.to_float(tf.one_hot(train_labels, N_CLASS))
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Y_score, labels = Y_label) #shape [None,2]
  adv_loss = tf.reduce_mean(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y_label, 1))
  acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  real_images = Gdec.build(u, real_labels, phg['is_training_g'])
  fake_images = Gdec.build(u, fake_labels, phg['is_training_g'])
  fake_gan_logit, fake_cls_logit = D.build(fake_images, phg['is_training_g'])

  gan_loss = -tf.reduce_mean(fake_gan_logit)
  cls_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims((1-train_labels),axis=-1), fake_cls_logit)
  rec_loss = tf.losses.absolute_difference(train_images, real_images)
  reg_loss = tf.reduce_sum(Genc.reg_loss+Gdec.reg_loss)

  final_loss = gan_loss + 10.0*cls_loss + 100.0*rec_loss + reg_loss - adv_loss
  update_op = tf.train.AdamOptimizer(phg['lr_g'], beta1=0.5, name='second_adam').minimize(final_loss, var_list=Genc.vars+Gdec.vars)

  return u, final_loss, acc_op, update_op

def D_graph(sess, phd):
  real_labels = train_labels * 2 -1
  fake_labels = -real_labels

  u = Genc.build(train_images, phd['is_training_d'])
  fake_images = Gdec.build(u, fake_labels, phd['is_training_d'])
  train_gan_logit, train_cls_logit = D.build(train_images, phd['is_training_d'])
  fake_gan_logit, fake_cls_logit = D.build(fake_images, phd['is_training_d'])

  train_gan_loss = -tf.reduce_mean(train_gan_logit)
  fake_gan_loss = tf.reduce_mean(fake_gan_logit)
  gradien_p = util.gradient_penalty(lambda x: D.build(x, phd['is_training_d'])[0], train_images, fake_images, '1-gp', 'line')
  cls_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(train_labels,axis=-1), train_cls_logit)
  reg_loss = tf.reduce_sum(D.reg_loss)

  final_loss = train_gan_loss + fake_gan_loss + 10.0*gradien_p + cls_loss + reg_loss
  update_op = tf.train.AdamOptimizer(phd['lr_d'], beta1=0.5).minimize(final_loss, var_list=D.vars)

  return final_loss, update_op

def V_graph(sess, phv):
  real_labels = valid_labels*2 -1
  fake_labels = -real_labels

  u = Genc.build(valid_images, phv['is_training_v'])
  real_images = Gdec.build(u, real_labels, phv['is_training_v'])
  fake_images = Gdec.build(u, fake_labels, phv['is_training_v'])

  return valid_images, real_images, fake_images

def Adv_graph(sess, pha):
  u = Genc.build(train_images, pha['is_training_a'])
  Y_score = Adv.build(u, N_CLASS, pha['is_training_a'])
  Y_hat = tf.nn.softmax(Y_score)
  Y_pred = tf.argmax(Y_hat, 1)
  Y_label = tf.to_float(tf.one_hot(train_labels, N_CLASS))

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Y_score, labels = Y_label) #shape [None,2]
  loss_op = tf.reduce_mean(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y_label, 1))
  acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  update_op = tf.train.MomentumOptimizer(LR, MOMENTUM).minimize(loss_op, var_list=Adv.vars)

  return loss_op, acc_op, update_op, Y_pred

def train():
  lr_g = tf.placeholder(tf.float32, shape=[])
  lr_d = tf.placeholder(tf.float32, shape=[])
  is_training_g = tf.placeholder(tf.bool, shape=[])
  is_training_d = tf.placeholder(tf.bool, shape=[])
  is_training_v = tf.placeholder(tf.bool, shape=[])
  is_training_a = tf.placeholder(tf.bool, shape=[])
  phg = {'lr_g': lr_g,
       'is_training_g': is_training_g}
  phd = {'lr_d': lr_d,
       'is_training_d': is_training_d}
  phv = {'is_training_v': is_training_v}
  pha = {'is_training_a': is_training_a}
  with tf.Session() as sess:
    G_u, G_loss, G_update = G_graph(sess, phg)
    Ga_u, Ga_loss, Ga_acc, Ga_update = G_adv_graph(sess, phg)
    D_loss, D_update  = D_graph(sess, phd)
    tr_images, re_images, fa_images = V_graph(sess, phv)
    adv_loss, adv_acc, adv_update, adv_pred = Adv_graph(sess, pha)
    saver = tf.train.Saver(Genc.vars+Gdec.vars)
    init = tf.global_variables_initializer()
    sess.run(init)
    # saver.restore(sess,LOG_PATH+'/weight.ckpt')

    #####################  pre-train Generator & Discriminator  #####################
    g_dict = {phg['lr_g']: 1e-4,
          phg['is_training_g']: True}
    d_dict = {phd['lr_d']: 1e-4,
          phd['is_training_d']: True}
    for epoch in range(N_EPOCH):
      for i in range(train_iters):
        if i % 6 !=0:
          d_loss, _ = sess.run([D_loss, D_update], feed_dict=d_dict)
        else:
          g_loss, _ = sess.run([G_loss, G_update], feed_dict=g_dict)

        if i % 100 == 0 and i != 0:
          log_string('====pre_epoch_%d====iter_%d: g_loss=%.4f, d_loss=%.4f' % (epoch, i, g_loss, d_loss))
    saver.save(sess,LOG_PATH+'/pretrain.ckpt')

    #####################  train and fix adversarial classifier  #####################
    a_dict = {pha['is_training_a']: True}
    for epoch in range(N_ADV):
      for i in range(train_iters):
        a_loss, a_acc, _ = sess.run([adv_loss, adv_acc, adv_update], feed_dict=a_dict)
        if i % 100 == 0 and i != 0:
          log_string('====adv_epoch_%d====iter_%d: a_loss=%.4f, a_acc=%.4f' % (epoch, i, a_loss, a_acc))

    #####################  fine-tune Generator & Discriminator  #####################
    g_dict = {phg['lr_g']: 5e-6,
          phg['is_training_g']: True}
    d_dict = {phd['lr_d']: 5e-6,
          phd['is_training_d']: True}
    v_dict = {phv['is_training_v']: True}
    for epoch in range(N_EPOCH):
      for i in range(train_iters):
        if i % 6 !=0:
          d_loss, _ = sess.run([D_loss, D_update], feed_dict=d_dict)
        else:
          g_loss, a_acc, _ = sess.run([Ga_loss, Ga_acc, Ga_update], feed_dict=g_dict)

        if i % 100 == 0 and i != 0:
          log_string('====epoch_%d====iter_%d: g_loss=%.4f, d_loss=%.4f, a_acc=%.4f' % (epoch, i, g_loss, d_loss, a_acc))
      a,b,c = sess.run([tr_images, re_images, fa_images], feed_dict = v_dict)
      image_list = [a,b,c]
      sample = np.transpose(image_list, (1, 2, 0, 3, 4))
      sample = np.reshape(sample, (-1, sample.shape[2] * sample.shape[3], sample.shape[4]))
      sample = util.to_uint8(sample)
      io.imsave('%s/Epoch-%d.jpg' % (LOG_PATH, epoch), sample, quality=95)
      saver.save(sess,LOG_PATH+'/weight.ckpt')


if __name__ == "__main__":
  train()
  LOG_FOUT.close()
