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
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images
from itertools import permutations

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
            if (FLAGS.expt_number == '3' or FLAGS.expt_number == '5') and FLAGS.train:
                print('Inside expt number 3/5, part 1')
                random.seed(1)
            else:
                random.seed(1)
                random.shuffle(character_folders)
            print('number of classes in the dataset', len(character_folders))
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
        elif FLAGS.datasource == 'dclaw':
            self.num_classes = 2
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes

            metatrain_folder = './data/dclaw/train'+FLAGS.dclaw_pn
            if FLAGS.test_set:
                metaval_folder = './data/dclaw/test'+FLAGS.dclaw_pn
            else:
                metaval_folder = './data/dclaw/val'+FLAGS.dclaw_pn

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            metatrain_folders = sorted(metatrain_folders)
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            metaval_folders = sorted(metaval_folders)
            print('metatrain_folders', metatrain_folders)
            print('metaval_folders', metaval_folders)
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')


    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            if FLAGS.expt_number == '6' or FLAGS.expt_number == '8':
                print('Inside expt number 6')
                num_total_batches = 26400
            elif FLAGS.expt_number == '6a' or FLAGS.expt_number == '8a':
                print('Inside expt number 6a')
                num_total_batches = 1440
            elif FLAGS.expt_number == '6b' or FLAGS.expt_number == '8b':
                print('Inside expt number 6b')
                num_total_batches = 12
            else:
                num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        print('expt number: ', FLAGS.expt_number, train)
        if FLAGS.expt_number == '2' and train:
            print('Inside expt number 2')
            all_filenames = []
            """
            go over the folders once, group the adjacent 5 classes together as one task. Non-exclusive
            """
            for task_count in range(int(len(folders)/self.num_classes)):
                #sampled_character_folders = random.sample(folders, self.num_classes)
                sampled_character_folders = folders[task_count*self.num_classes: (task_count+1)*self.num_classes]
                random.shuffle(sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
        elif (FLAGS.expt_number == '9a' or FLAGS.expt_number == '9b' or FLAGS.expt_number == '9c' or FLAGS.expt_number == '11a1' or FLAGS.expt_number == '11a2' or FLAGS.expt_number == '11a3') and train:
            print('Inside expt number 9a/9b/9c/11a1/11a2/11a3')
            all_filenames = []
            """
            go over the folders multiple times, group the adjacent 5 classes together as one task. Non-exclusive
            """
            if FLAGS.expt_number == '9a' or FLAGS.expt_number == '11a1' or FLAGS.expt_number == '11a2' or FLAGS.expt_number == '11a3':
                total_num_tasks = 200000
            elif FLAGS.expt_number =='9b':
                total_num_tasks = 1440
            elif FLAGS.expt_number == '9c':
                total_num_tasks = 26400
            for outer_task_count in range(int(total_num_tasks/(len(folders)/self.num_classes))):
                for task_count in range(int(len(folders)/self.num_classes)):
                    #sampled_character_folders = random.sample(folders, self.num_classes)
                    sampled_character_folders = folders[task_count*self.num_classes: (task_count+1)*self.num_classes]
                    #random.shuffle(sampled_character_folders)
                    labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                    # make sure the above isn't randomized order
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
        elif (FLAGS.expt_number == '7a' or FLAGS.expt_number == '7b' or FLAGS.expt_number == '7c'or FLAGS.expt_number == '11b1' or FLAGS.expt_number == '11b2' or FLAGS.expt_number == '11b3') and train:
            print('Inside expt number 7a/7b/7c/11b1/11b2/11b3')
            all_filenames = []
            """
            go over the folders multiple times, group the adjacent 5 classes together as one task. Non-exclusive
            """
            if FLAGS.expt_number == '7a' or FLAGS.expt_number == '11b1' or FLAGS.expt_number == '11b2' or FLAGS.expt_number == '11b3':
                total_num_tasks = 200000
            elif FLAGS.expt_number =='7b':
                total_num_tasks = 1440
            elif FLAGS.expt_number == '7c':
                total_num_tasks = 26400
            for outer_task_count in range(int(total_num_tasks/(len(folders)/self.num_classes))):
                for task_count in range(int(len(folders)/self.num_classes)):
                    #sampled_character_folders = random.sample(folders, self.num_classes)
                    sampled_character_folders = folders[task_count*self.num_classes: (task_count+1)*self.num_classes]
                    random.shuffle(sampled_character_folders)
                    labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                    # make sure the above isn't randomized order
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
        elif FLAGS.expt_number == '3' and train:
            print('Inside expt number 3')
            all_filenames = []
            """
            removed shuffling of classes in the init function. classes from the same alphabet together now.
            go over the folders once, group the adjacent 5 classes together as one task. Non-exclusive
            """
            for task_count in range(int(len(folders)/self.num_classes)):
                #sampled_character_folders = random.sample(folders, self.num_classes)
                sampled_character_folders = folders[task_count*self.num_classes: (task_count+1)*self.num_classes]
                #random.shuffle(sampled_character_folders)
                #print("Task ", task_count, ": ", sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
        elif (FLAGS.expt_number == '4' or FLAGS.expt_number == '5' or FLAGS.expt_number == '8c') and train:
            print('Inside expt number 4/5')
            all_filenames = []
            """
            go over the folders once, group the adjacent 5 classes together as one task.
            get all permutations of that task. Shuffle these tasks
            """
            task_folders_new = []
            for task_count in range(int(len(folders)/self.num_classes)):
                sampled_character_folders = folders[task_count*self.num_classes: (task_count+1)*self.num_classes]
                task_folders_temp = permutations(sampled_character_folders)
                task_folders_new.extend(task_folders_temp)
            print('total number of tasks: ', len(task_folders_new))
            random.shuffle(task_folders_new)
            for task_count in range(len(task_folders_new)):
                #sampled_character_folders = random.sample(folders, self.num_classes)
                #sampled_character_folders = folders[task_count*self.num_classes: (task_count+1)*self.num_classes]
                sampled_character_folders = task_folders_new[task_count]
                #random.shuffle(sampled_character_folders)
                #print("Task ", task_count, ": ", sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
        elif (FLAGS.expt_number == '12a' or FLAGS.expt_number == '12b' or FLAGS.expt_number == '12c' or FLAGS.expt_number == '12d' or FLAGS.expt_number == '12e' or FLAGS.expt_number == '12f') and train:
            print('Inside expt number 12s')
            all_filenames = []
            """
            Make tasks first.
            Go over the tasks again and again to collect data.
            """
            if FLAGS.expt_number == '12a':
                n_tasks = 12
            elif FLAGS.expt_number == '12b':
                n_tasks = 120
            elif FLAGS.expt_number == '12c':
                n_tasks = 1200
            elif FLAGS.expt_number == '12d':
                n_tasks = 1440
            elif FLAGS.expt_number == '12e':
                n_tasks = 12000
            elif FLAGS.expt_number == '12f':
                n_tasks = 120000
            task_folders_new = []
            for task_count in range(n_tasks):
                sampled_character_folders = random.sample(folders, self.num_classes)
                task_folders_new.append(sampled_character_folders)
            print('total number of tasks: ', len(task_folders_new))
            #random.shuffle(task_folders_new)
            for task_count in range(num_total_batches):
                sampled_character_folders = task_folders_new[task_count%n_tasks]
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
        else:
            if FLAGS.datasource == 'dclaw' and not train:
                all_filenames = []
                print('Inside dclaw validation/testing data tensor creation')
                task_folders_new = []
                for i in range(int(len(folders)/self.num_classes)):
                    sampled_character_folders = folders[i*self.num_classes: (i+1)*self.num_classes]
                    task_folders_new.append(sampled_character_folders)
                for i in range(num_total_batches):
                    sampled_character_folders = task_folders_new[i%len(task_folders_new)]
                    random.shuffle(sampled_character_folders)
                    labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                    # make sure the above isn't randomized order
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)

            else:
                all_filenames = []
                print('Inside expt number 1/6/6a/6b/8/8a/8b/11c1/11c2/11c3')
                for _ in range(num_total_batches):
                    sampled_character_folders = random.sample(folders, self.num_classes)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False, train=train)
                    # make sure the above isn't randomized order
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'dclaw' :
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
