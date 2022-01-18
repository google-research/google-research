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

"""Eval and compute metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

from absl import flags
import cv2
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.io.gfile as gfile
import tensorflow_addons as tfa

from comisr.lib.dataloader import inference_data_loader
from comisr.lib.model import fnet
from comisr.lib.model import generator_f
import comisr.lib.ops as ops
import comisr.metrics as metrics


flags.DEFINE_string('input_lr_dir', None,
                    'The directory of the input low-resolution data.')
flags.DEFINE_string('input_hr_dir', None,
                    'The directory of the input high-resolution data.')
flags.DEFINE_integer('input_dir_len', -1,
                     'length of the input for inference mode, -1 means all.')
flags.DEFINE_string('output_dir', None,
                    'The output directory of the checkpoint')
flags.DEFINE_string('output_pre', '',
                    'The name of the subfolder for the images')
flags.DEFINE_string('output_name', 'output', 'The pre name of the outputs')
flags.DEFINE_string(
    'checkpoint_path', None,
    'If provided, the weight will be restored from the provided checkpoint')
flags.DEFINE_integer('num_resblock', 10,
                     'How many residual blocks are there in the generator')
flags.DEFINE_integer('vsr_scale', 4, 'vsr scale.')
flags.DEFINE_string('output_ext', 'png',
                    'The format of the output when evaluating')
flags.DEFINE_string('eval_type_prefix', 'crf25',
                    'The format of the output when evaluating')

flags.DEFINE_string('results', None, 'the list of paths of result directory')
flags.DEFINE_string('targets', None, 'the list of paths of target directory')
flags.DEFINE_boolean('use_ema', True, 'use ema')
flags.DEFINE_boolean('is_vid4_eval', True, 'True is vid4, and false is reds4.')

FLAGS = flags.FLAGS
tf.compat.v1.disable_eager_execution()


def _get_ema_vars():
  """Gets all variables for which we maintain the moving average."""
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain moving average not only for all trainable variables, but also
    # some other non-trainable variables including batch norm moving mean and
    # variance.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  return list(set(ema_vars))


def extract_detail_ops(image, sigma=1.5):
  """extract details from the image tensors."""
  # input image is a 3D or 4D tensor with image range in [0, 1].
  image_blur = tfa.image.gaussian_filter2d(image, sigma=sigma)
  laplacian_image = (image - image_blur)
  return laplacian_image


def inference(
    input_lr_dir,
    input_hr_dir,
    input_dir_len,
    num_resblock,
    vsr_scale,
    checkpoint_path,
    output_dir,
    output_pre,
    output_name,
    output_ext,
):
  """Main inference function."""
  if checkpoint_path is None:
    raise ValueError('The checkpoint file is needed to performing the test.')

  # Declare the test data reader
  inference_data = inference_data_loader(input_lr_dir, input_hr_dir,
                                         input_dir_len)
  input_shape = [
      1,
  ] + list(inference_data.inputs[0].shape)
  output_shape = [1, input_shape[1] * vsr_scale, input_shape[2] * vsr_scale, 3]
  oh = input_shape[1] - input_shape[1] // 8 * 8
  ow = input_shape[2] - input_shape[2] // 8 * 8
  paddings = tf.constant([[0, 0], [0, oh], [0, ow], [0, 0]])
  print('input shape:', input_shape)
  print('output shape:', output_shape)

  # build the graph
  inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')

  pre_inputs = tf.Variable(
      tf.zeros(input_shape), trainable=False, name='pre_inputs')
  pre_gen = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_gen')
  pre_warp = tf.Variable(
      tf.zeros(output_shape), trainable=False, name='pre_warp')

  transpose_pre = tf.space_to_depth(pre_warp, vsr_scale)
  inputs_all = tf.concat((inputs_raw, transpose_pre), axis=-1)
  with tf.variable_scope('generator'):
    gen_output = generator_f(
        inputs_all, 3, num_resblock, vsr_scale, reuse=False)
    # Deprocess the images outputed from the model, and assign things for next
    # frame
    with tf.control_dependencies([tf.assign(pre_inputs, inputs_raw)]):
      outputs = tf.assign(pre_gen, ops.deprocess(gen_output))

  inputs_frames = tf.concat((pre_inputs, inputs_raw), axis=-1)
  with tf.variable_scope('fnet'):
    gen_flow_lr = fnet(inputs_frames, reuse=False)
    gen_flow_lr = tf.pad(gen_flow_lr, paddings, 'SYMMETRIC')

    deconv_flow = gen_flow_lr
    deconv_flow = ops.conv2_tran(
        deconv_flow, 3, 64, 2, scope='deconv_flow_tran1')
    deconv_flow = tf.nn.relu(deconv_flow)
    deconv_flow = ops.conv2_tran(
        deconv_flow, 3, 64, 2, scope='deconv_flow_tran2')
    deconv_flow = tf.nn.relu(deconv_flow)
    deconv_flow = ops.conv2(deconv_flow, 3, 2, 1, scope='deconv_flow_conv')
    gen_flow = ops.upscale_x(gen_flow_lr * 4.0, scale=vsr_scale)
    gen_flow = deconv_flow + gen_flow

    gen_flow.set_shape(output_shape[:-1] + [2])
  pre_warp_hi = tfa.image.dense_image_warp(pre_gen, gen_flow)
  pre_warp_hi = pre_warp_hi + extract_detail_ops(pre_warp_hi)
  before_ops = tf.assign(pre_warp, pre_warp_hi)

  print('Finish building the network')

  if FLAGS.use_ema:
    moving_average_decay = 0.99
    global_step = tf.train.get_or_create_global_step()
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    ema_vars = _get_ema_vars()

  # In inference time, we only need to restore the weight of the generator
  var_list = tf.trainable_variables()

  restore_vars_dict = {}
  if FLAGS.use_ema:
    for v in var_list:
      if re.match(v.name, '.*global_step.*'):
        restore_vars_dict[v.name[:-2]] = v
      else:
        restore_vars_dict[v.name[:-2] + '/ExponentialMovingAverage'] = v
  else:
    restore_vars_dict = var_list

  weight_initiallizer = tf.train.Saver(restore_vars_dict)

  # Define the initialization operation
  init_op = tf.global_variables_initializer()
  local_init_op = tf.local_variables_initializer()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  if not gfile.exists(output_dir):
    gfile.mkdir(output_dir)
  if not output_pre:
    image_dir = output_dir
  else:
    image_dir = os.path.join(output_dir, output_pre)
  if not gfile.exists(image_dir):
    gfile.mkdir(image_dir)

  with tf.Session(config=config) as sess:
    # Load the pretrained model
    sess.run(init_op)
    sess.run(local_init_op)

    print('Loading weights from ckpt model')
    weight_initiallizer.restore(sess, checkpoint_path)
    max_iter = len(inference_data.inputs)

    srtime = 0
    print('Frame evaluation starts!!')
    for i in range(max_iter):
      input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
      feed_dict = {inputs_raw: input_im}
      t0 = time.time()
      if i != 0:
        sess.run(before_ops, feed_dict=feed_dict)
      output_frame = sess.run(outputs, feed_dict=feed_dict)
      srtime += time.time() - t0

      if i >= 5:
        name, _ = os.path.splitext(
            os.path.basename(str(inference_data.paths_LR[i])))
        filename = output_name + '_' + name
        out_path = os.path.join(image_dir, '%s.%s' % (filename, output_ext))
        print('saving image %s' % out_path)
        with tf.gfile.Open(out_path, 'wb') as image_file:
          img = np.clip(output_frame[0] * 255.0, 0, 255).astype(np.uint8)
          _, buff = cv2.imencode('.png', img[:, :, ::-1])
          image_file.write(buff.tostring())

      else:
        print('Warming up %d' % (5 - i))
  tf.reset_default_graph()
  print('total time ' + str(srtime) + ', frame number ' + str(max_iter))


def compute_metrics(input_flags):
  """Compute metrics."""
  if input_flags.is_vid4_eval:
    result_list_all = ['calendar', 'city', 'foliage', 'walk']
  else:
    # Reds4 datasets.
    result_list_all = ['000', '011', '015', '020']
  result_list_all.sort()

  result_list_all = [
      os.path.join(input_flags.output_dir, dir) for dir in result_list_all
  ]

  print('all eval path:')
  print(result_list_all)
  result_list = []
  for dir_name in result_list_all:
    if gfile.isdir(dir_name):
      result_list.append(dir_name)
  target_list = gfile.listdir(input_flags.targets)
  target_list = [
      os.path.join(input_flags.targets, dir_name) for dir_name in target_list
  ]
  target_list.sort()
  folder_n = len(result_list)

  cutfr = 2

  keys = ['PSNR', 'SSIM']
  sum_dict = dict.fromkeys(['FrameAvg_' + _ for _ in keys], 0)
  len_dict = dict.fromkeys(keys, 0)
  avg_dict = dict.fromkeys(['Avg_' + _ for _ in keys], 0)
  folder_dict = dict.fromkeys(['FolderAvg_' + _ for _ in keys], 0)

  for folder_i in range(folder_n):
    print(folder_i)
    result = metrics.list_png_in_dir(result_list[folder_i])
    target = metrics.list_png_in_dir(target_list[folder_i])
    image_no = len(target)

    list_dict = {}
    for key_i in keys:
      list_dict[key_i] = []

    for i in range(cutfr, image_no - cutfr):

      with tf.gfile.Open(result[i], 'rb') as fid:
        raw_im = np.asarray(bytearray(fid.read()), dtype=np.uint8)
        output_img = cv2.imdecode(raw_im, cv2.IMREAD_COLOR).astype(
            np.float32)[:, :, ::-1]
      with tf.gfile.Open(target[i], 'rb') as fid:
        raw_im = np.asarray(bytearray(fid.read()), dtype=np.uint8)
        target_img = cv2.imdecode(raw_im, cv2.IMREAD_COLOR).astype(
            np.float32)[:, :, ::-1]

      msg = 'frame %d, tar %s, out %s, ' % (i, str(
          target_img.shape), str(output_img.shape))
      if (target_img.shape[0] < output_img.shape[0]) or (
          target_img.shape[1] <
          output_img.shape[1]):  # target is not dividable by 4
        output_img = output_img[:target_img.shape[0], :target_img.shape[1]]

      target_img, _, _ = metrics.crop_8x8(target_img)
      output_img, _, _ = metrics.crop_8x8(output_img)

      if input_flags.is_vid4_eval:
        y_channel = True
      else:
        y_channel = False
      if 'PSNR' in keys:  # psnr
        list_dict['PSNR'].append(
            metrics.psnr(target_img, output_img, y_channel))
        msg += 'psnr %02.2f' % (list_dict['PSNR'][-1])

      if 'SSIM' in keys:  # ssim
        list_dict['SSIM'].append(
            metrics.ssim(target_img, output_img, y_channel))
        msg += ', ssim %02.2f' % (list_dict['SSIM'][-1])

      print(msg)
    mode = 'w' if folder_i == 0 else 'a'

    pd_dict = {}
    for cur_num_data in keys:
      num_data = cur_num_data + '_%02d' % folder_i
      cur_list = np.float32(list_dict[cur_num_data])
      pd_dict[num_data] = pd.Series(cur_list)

      num_data_sum = cur_list.sum()
      num_data_len = cur_list.shape[0]

      num_data_mean = num_data_sum / num_data_len
      print('%s, max %02.4f, min %02.4f, avg %02.4f' %
            (num_data, cur_list.max(), cur_list.min(), num_data_mean))

      if folder_i == 0:
        avg_dict['Avg_' + cur_num_data] = [num_data_mean]
      else:
        avg_dict['Avg_' + cur_num_data] += num_data_mean

      sum_dict['FrameAvg_' + cur_num_data] += num_data_sum
      len_dict[cur_num_data] += num_data_len
      folder_dict['FolderAvg_' + cur_num_data] += num_data_mean

    csv_filepath = os.path.join(input_flags.output_dir, 'metrics.csv')
    with tf.gfile.Open(csv_filepath, 'w') as csvf:
      pd.DataFrame(pd_dict).to_csv(csvf, mode=mode)

  for num_data in keys:
    sum_dict['FrameAvg_' + num_data] = pd.Series(
        [sum_dict['FrameAvg_' + num_data] / len_dict[num_data]])
    folder_dict['FolderAvg_' + num_data] = pd.Series(
        [folder_dict['FolderAvg_' + num_data] / folder_n])
    avg_dict['Avg_' + num_data] = pd.Series(
        np.float32(avg_dict['Avg_' + num_data]))
    print('%s, total frame %d, total avg %02.4f, folder avg %02.4f' %
          (num_data, len_dict[num_data], sum_dict['FrameAvg_' + num_data][0],
           folder_dict['FolderAvg_' + num_data][0]))
  cvs_filepath = os.path.join(input_flags.output_dir, 'metrics.csv')
  with tf.gfile.Open(cvs_filepath, 'w') as csvf:
    pd.DataFrame(avg_dict).to_csv(csvf, mode='a')
    pd.DataFrame(folder_dict).to_csv(csvf, mode='a')
    pd.DataFrame(sum_dict).to_csv(csvf, mode='a')
  print('Finished.')


def main(_):
  if FLAGS.checkpoint_path:
    folder_list = gfile.listdir(FLAGS.input_lr_dir)
    for folder_name in folder_list:
      input_lr_dir = os.path.join(FLAGS.input_lr_dir, folder_name)
      output_pre = folder_name
      inference(input_lr_dir, FLAGS.input_hr_dir, FLAGS.input_dir_len,
                FLAGS.num_resblock, FLAGS.vsr_scale, FLAGS.checkpoint_path,
                FLAGS.output_dir, output_pre, FLAGS.output_name,
                FLAGS.output_ext)
    compute_metrics(FLAGS)


if __name__ == '__main__':
  tf.app.run()
