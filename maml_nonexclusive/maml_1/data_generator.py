# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.python.platform import flags
from utils import get_images

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif 'omniglot' in FLAGS.datasource:
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if FLAGS.test_set:
                self.metaval_character_folders = character_folders[num_train+num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif FLAGS.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/train')
            if FLAGS.test_set:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')


    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
            #num_total_batches = int(len(folders)/self.num_classes)
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')

        if FLAGS.task_setting == "ne" and train:
            #all_filenames = []
            #random.shuffle(folders)
            #count = 0
            #while count < num_total_batches:
            #    for i in range(int(len(folders)/self.num_classes)):
            #        count +=1
            #        sampled_character_folders = random.sample(folders[:30], self.num_classes)
            #        random.shuffle(sampled_character_folders)
            #        #sampled_character_folders = folders[i*self.num_classes: (i+1)*self.num_classes]
            #        labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            #        # make sure the above isn't randomized order
            #        labels = [li[0] for li in labels_and_images]
            #        filenames = [li[1] for li in labels_and_images]
            #        all_filenames.extend(filenames)
            """
            folders_tasks = []
            count_tasks = 0
            while count_tasks <= num_total_batches:
                for i in range(int(len(folders)/(self.num_classes))):
                    count_tasks +=1
                    #sampled_character_folders = random.sample(folders, self.num_classes)
                    #random.shuffle(sampled_character_folders)
                    sampled_character_folders = folders[i*self.num_classes: (i+1)*self.num_classes]
                    #random.shuffle(sampled_character_folders)
                    folders_tasks.append(sampled_character_folders)
            random.shuffle(folders_tasks)
            for sampled_character_folders in folders_tasks:
                #print("sampled_character_folders", sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
            """
            all_filenames = []
            for i in range(int(len(folders)/self.num_classes)):
                #sampled_character_folders = random.sample(folders, self.num_classes)
                sampled_character_folders = folders[i*self.num_classes:(i+1)*self.num_classes]
                print(sampled_character_folders)
                #random.shuffle(sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
        else:
            all_filenames = []
            for _ in range(num_total_batches):
                sampled_character_folders = random.sample(folders, self.num_classes)
                random.shuffle(sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource == 'miniimagenet':
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0],self.img_size[1],3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0],self.img_size[1],1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                if FLAGS.datasource == 'omniglot': # and FLAGS.train:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
                        k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        #c = list(zip(all_image_batches, all_label_batches))
        #random.shuffle(c)
        #all_image_batches, all_label_batches = zip(*c)
        #all_image_batches = tf.reshape(all_image_batches, [self.batch_size,int(examples_per_batch),-1])
        #all_label_batches = tf.reshape(all_label_batches, [self.batch_size, int(examples_per_batch)])
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase
