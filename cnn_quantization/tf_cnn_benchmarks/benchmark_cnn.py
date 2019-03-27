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

"""TensorFlow benchmark library.

See the README for more information.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import namedtuple
import contextlib
import math
import multiprocessing
import os
import re
import threading
import time
import traceback

from absl import flags as absl_flags
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from google.protobuf import text_format
from cnn_quantization.tf_cnn_benchmarks import cnn_util
from cnn_quantization.tf_cnn_benchmarks import constants
from cnn_quantization.tf_cnn_benchmarks import datasets
from cnn_quantization.tf_cnn_benchmarks import flags
from cnn_quantization.tf_cnn_benchmarks import mlperf
from cnn_quantization.tf_cnn_benchmarks import variable_mgr
from cnn_quantization.tf_cnn_benchmarks import variable_mgr_util
from cnn_quantization.tf_cnn_benchmarks.cnn_util import log_fn
from cnn_quantization.tf_cnn_benchmarks.models import model_config
from cnn_quantization.tf_cnn_benchmarks.platforms import util as platforms_util
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_util_impl
from tensorflow.python.framework import importer
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest


_DEFAULT_NUM_BATCHES = 100


# GraphInfo encapsulates the tensors/ops that we care about after building a
# graph. We use them to benchmark the graph.
GraphInfo = namedtuple(  # pylint: disable=invalid-name
    'GraphInfo',
    [
        # Ops that produce the input batches (before preprocessing).
        'input_producer_op',
        # Ops that adds the preprocessed images to the staging areas
        'enqueue_ops',
        # Fetches of sess.run()
        'fetches',
        # Op that performs synchronization in distributed mode
        'execution_barrier',
        # The global step variable
        'global_step',
        # Group of ops that perform per-device initialization work
        'local_var_init_op_group',
        # Op to produce summaries
        'summary_op'
    ])


# InputProcessingInfo contains various sources of inputs which will be later fed
# into the model. If synthetic data is used, all four fields are None.
InputProcessingInfo = namedtuple(
    'InputProcessingInfo',
    [
        # The first two fields are non-None iff datasets prefetching is not
        # used.

        # Ops that produce the input batches.
        'input_producer_op',
        # A list of StagingArea for each device.
        'input_producer_stages',

        # Input produced using FunctionBufferingResource. Non-None iff datasets
        # prefetching is used and --use_multi_device_iterator=False
        'function_buffering_resources',

        # Input produced using multi device iterator. Non-None iff datasets
        # prefetching is used and --use_multi_device_iterator=True
        'multi_device_iterator_input'
    ])


# TODO(reedwm): add upper_bound and lower_bound to appropriate integer and
# float flags, and change certain string flags to enum flags.

flags.DEFINE_string('model', 'trivial',
                    'Name of the model to run, the list of supported models '
                    'are defined in models/model.py')
# The code will first check if it's running under benchmarking mode
# or evaluation mode, depending on 'eval':
# Under the evaluation mode, this script will read a saved model,
#   and compute the accuracy of the model against a validation dataset.
#   Additional ops for accuracy and top_k predictors are only used under
#   this mode.
# Under the benchmarking mode, user can specify whether nor not to use
#   the forward-only option, which will only compute the loss function.
#   forward-only cannot be enabled with eval at the same time.
flags.DEFINE_boolean('eval', False, 'whether use eval or benchmarking')
flags.DEFINE_integer('eval_interval_secs', 0,
                     'How often to run eval on saved checkpoints. Usually the '
                     'same as save_model_secs from the corresponding training '
                     'run. Pass 0 to eval only once.')
flags.DEFINE_integer('eval_during_training_every_n_steps', None,
                     'Every n steps during training, pause training, run '
                     'evaluation, then resume training. Must not be used with '
                     '--eval, as unlike --eval, this option causes both '
                     'training and eval to be done. This may take slightly '
                     'more GPU memory than running just training or evaluation '
                     'alone. It also may slightly slow down training, even '
                     'when not taking into account the additional time to '
                     'evaluate.', lower_bound=1)
flags.DEFINE_float('eval_during_training_every_n_epochs', None,
                   'After every n training epochs, pause training, run '
                   'evaluation, then resume training. See '
                   '--eval_during_training_every_n_steps for more information.')
flags.DEFINE_list('eval_during_training_at_specified_steps', [],
                  'Specify a list of training steps, pause training at each of '
                  'these steps, run evaluation, then resume training. See '
                  '--eval_during_training_every_n_steps for more information.')
flags.DEFINE_list('eval_during_training_at_specified_epochs', [],
                  'Specify a list of training epochs, pause training after '
                  'each of these epochs, run evaluation, then resume training. '
                  'See --eval_during_training_every_n_steps for more '
                  'information.')
flags.DEFINE_boolean('forward_only', False,
                     'whether use forward-only or training for benchmarking')
flags.DEFINE_boolean('freeze_when_forward_only', False,
                     'whether to freeze the graph when in forward-only mode.')
flags.DEFINE_boolean('print_training_accuracy', False,
                     'whether to calculate and print training accuracy during '
                     'training')
flags.DEFINE_integer('batch_size', 0, 'batch size per compute device')
flags.DEFINE_integer('eval_batch_size', 0, 'eval batch size per compute device')
flags.DEFINE_integer('batch_group_size', 1,
                     'number of groups of batches processed in the image '
                     'producer.')
flags.DEFINE_integer('num_batches', None, 'number of batches to run, excluding '
                     'warmup. Defaults to %d' % _DEFAULT_NUM_BATCHES)
flags.DEFINE_integer('num_eval_batches', None,
                     'number of eval batches to run, excluding warmup. '
                     'Defaults to --num_batches')
flags.DEFINE_float('num_epochs', None,
                   'number of epochs to run, excluding warmup. '
                   'This and --num_batches cannot both be specified.')
flags.DEFINE_float('num_eval_epochs', None,
                   'number of eval epochs to run, excluding warmup. '
                   'Defaults to --num_epochs')
flags.DEFINE_float('stop_at_top_1_accuracy', None,
                   'If set, stops training after the evaluation accuracy hits '
                   'this number. Can only be used with one of the '
                   '--eval_during_training_* flags.')
flags.DEFINE_boolean('collect_eval_results_async', False,
                     'If True, start a separate process to postprocess eval '
                     'results asynchronously. This currently only works with '
                     'the SSD model.')
flags.DEFINE_integer('num_warmup_batches', None,
                     'number of batches to run before timing')
flags.DEFINE_integer('autotune_threshold', None,
                     'The autotune threshold for the models')
flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs to run on')
flags.DEFINE_string('gpu_indices', '', 'indices of worker GPUs in ring order')
flags.DEFINE_integer('display_every', 10,
                     'Number of local steps after which progress is printed '
                     'out')
flags.DEFINE_float('display_perf_ewma', None,
                   'If set, display numbers of images/sec using exponentially '
                   'weighted moving avearge with the specified weight, which '
                   'defines how much current value contributes to the reported '
                   'average. Increasing weight makes the reported performance '
                   'number reflect more about the real-time speed instead of '
                   'the entire history', lower_bound=0, upper_bound=1)
flags.DEFINE_string('data_dir', None,
                    'Path to dataset in TFRecord format (aka Example '
                    'protobufs). If not specified, synthetic data will be '
                    'used.')
flags.DEFINE_string('data_name', None,
                    'Name of dataset: imagenet or cifar10. If not specified, '
                    'it is automatically guessed based on data_dir.')
flags.DEFINE_string('resize_method', 'bilinear',
                    'Method for resizing input images: crop, nearest, '
                    'bilinear, bicubic, area, or round_robin. The `crop` mode '
                    'requires source images to be at least as large as the '
                    'network input size. The `round_robin` mode applies '
                    'different resize methods based on position in a batch in '
                    'a round-robin fashion. Other modes support any sizes and '
                    'apply random bbox distortions before resizing (even with '
                    'distortions=False).')
flags.DEFINE_boolean('distortions', True,
                     'Enable/disable distortions during image preprocessing. '
                     'These include bbox and color distortions.')
flags.DEFINE_boolean('use_datasets', True,
                     'Enable use of datasets for input pipeline')
flags.DEFINE_string('input_preprocessor', 'default',
                    'Name of input preprocessor. The list of supported input '
                    'preprocessors are defined in preprocessing.py.')
flags.DEFINE_string('gpu_thread_mode', 'gpu_private',
                    'Methods to assign GPU host work to threads. '
                    'global: all GPUs and CPUs share the same global threads; '
                    'gpu_private: a private threadpool for each GPU; '
                    'gpu_shared: all GPUs share the same threadpool.')
flags.DEFINE_integer('per_gpu_thread_count', 0,
                     'The number of threads to use for GPU. Only valid when '
                     'gpu_thread_mode is not global.')
flags.DEFINE_boolean('hierarchical_copy', False,
                     'Use hierarchical copies. Currently only optimized for '
                     'use on a DGX-1 with 8 GPUs and may perform poorly on '
                     'other hardware. Requires --num_gpus > 1, and only '
                     'recommended when --num_gpus=8')
# TODO(hinsu): Support auto-detection of the network topology while still
# retaining the ability to specify a particular topology for debugging.
flags.DEFINE_enum(
    'network_topology', constants.NetworkTopology.DGX1,
    (constants.NetworkTopology.DGX1, constants.NetworkTopology.GCP_V100),
    'Network topology specifies the topology used to connect multiple devices. '
    'Network topology is used to decide the hierarchy to use for the '
    'hierarchical_copy.')
flags.DEFINE_integer('gradient_repacking', 0, 'Use gradient repacking. It'
                     'currently only works with replicated mode. At the end of'
                     'of each step, it repacks the gradients for more efficient'
                     'cross-device transportation. A non-zero value specifies'
                     'the number of split packs that will be formed.',
                     lower_bound=0)
flags.DEFINE_boolean('compact_gradient_transfer', True, 'Compact gradient'
                     'as much as possible for cross-device transfer and '
                     'aggregation.')
flags.DEFINE_enum('variable_consistency', 'strong', ('strong', 'relaxed'),
                  'The data consistency for trainable variables. With strong '
                  'consistency, the variable always have the updates from '
                  'previous step. With relaxed consistency, all the updates '
                  'will eventually show up in the variables. Likely one step '
                  'behind.')
flags.DEFINE_boolean('datasets_repeat_cached_sample', False,
                     'Enable use of a special datasets pipeline that reads a '
                     'single TFRecord into memory and repeats it infinitely '
                     'many times. The purpose of this flag is to make it '
                     'possible to write regression tests that are not '
                     'bottlenecked by CNS throughput. '
                     'Use datasets_use_caching to cache input data.')
flags.DEFINE_enum('local_parameter_device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
                  'Device to use as parameter server: cpu or gpu. For '
                  'distributed training, it can affect where caching of '
                  'variables happens.')
flags.DEFINE_enum('device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
                  'Device to use for computation: cpu or gpu')
flags.DEFINE_enum('data_format', 'NCHW', ('NHWC', 'NCHW'),
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).')
flags.DEFINE_integer('num_intra_threads', None,
                     'Number of threads to use for intra-op parallelism. If '
                     'set to 0, the system will pick an appropriate number.')
flags.DEFINE_integer('num_inter_threads', 0,
                     'Number of threads to use for inter-op parallelism. If '
                     'set to 0, the system will pick an appropriate number.')
flags.DEFINE_string('trace_file', '',
                    'Enable TensorFlow tracing and write trace to this file.')
flags.DEFINE_boolean('use_chrome_trace_format', True,
                     'If True, the trace_file, if specified, will be in a '
                     'Chrome trace format. If False, then it will be a '
                     'StepStats raw proto.')
_NUM_STEPS_TO_PROFILE = 10
_NUM_OPS_TO_PRINT = 20
flags.DEFINE_string('tfprof_file', None,
                    'If specified, write a tfprof ProfileProto to this file. '
                    'The performance and other aspects of the model can then '
                    'be analyzed with tfprof. The first %d steps '
                    'are profiled. Additionally, the top %d most time '
                    'consuming ops will be printed.\n'
                    'Note: profiling with tfprof is very slow, but most of the '
                    'overhead is spent between steps. So, profiling results '
                    'are more accurate than the slowdown would suggest.' %
                    (_NUM_STEPS_TO_PROFILE, _NUM_OPS_TO_PRINT))
flags.DEFINE_string('graph_file', None,
                    'Write the model\'s graph definition to this file. '
                    'Defaults to binary format unless filename ends in "txt".')
flags.DEFINE_string('partitioned_graph_file_prefix', None,
                    'If specified, after the graph has been partitioned and '
                    'optimized, write out each partitioned graph to a file '
                    'with the given prefix.')
flags.DEFINE_enum('optimizer', 'sgd', ('momentum', 'sgd', 'rmsprop', 'adam'),
                  'Optimizer to use')
flags.DEFINE_float('init_learning_rate', None,
                   'Initial learning rate for training.')
flags.DEFINE_string('piecewise_learning_rate_schedule', None,
                    'Specifies a piecewise learning rate schedule based on the '
                    'number of epochs. This is the form LR0;E1;LR1;...;En;LRn, '
                    'where each LRi is a learning rate and each Ei is an epoch '
                    'indexed from 0. The learning rate is LRi if the '
                    'E(i-1) <= current_epoch < Ei. For example, if this '
                    'paramater is 0.3;10;0.2;25;0.1, the learning rate is 0.3 '
                    'for the first 10 epochs, then is 0.2 for the next 15 '
                    'epochs, then is 0.1 until training ends.')
flags.DEFINE_float('num_epochs_per_decay', 0,
                   'Steps after which learning rate decays. If 0, the learning '
                   'rate does not decay.')
flags.DEFINE_float('learning_rate_decay_factor', 0,
                   'Learning rate decay factor. Decay by this factor every '
                   '`num_epochs_per_decay` epochs. If 0, learning rate does '
                   'not decay.')
flags.DEFINE_float('num_learning_rate_warmup_epochs', 0,
                   'Slowly increase to the initial learning rate in the first '
                   'num_learning_rate_warmup_epochs linearly.')
flags.DEFINE_float('minimum_learning_rate', 0,
                   'The minimum learning rate. The learning rate will '
                   'never decay past this value. Requires `learning_rate`, '
                   '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
                   'be set.')
flags.DEFINE_float('resnet_base_lr', None, "Base learning rate at bs=256. Only "
                   "relevant when training ResNet and utilizing the model's "
                   "learning rate heuristic (get_learning_rate).")
flags.DEFINE_float('momentum', 0.9, 'Momentum for training.')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum in RMSProp.')
flags.DEFINE_float('rmsprop_epsilon', 1.0, 'Epsilon term for RMSProp.')
flags.DEFINE_float('adam_beta1', 0.9, 'Beta2 term for the Adam optimizer')
flags.DEFINE_float('adam_beta2', 0.999, 'Beta2 term for the Adam optimizer')
flags.DEFINE_float('adam_epsilon', 1e-8, 'Epsilon term for the Adam optimizer')
flags.DEFINE_float('gradient_clip', None,
                   'Gradient clipping magnitude. Disabled by default.')
flags.DEFINE_float('weight_decay', 0.00004,
                   'Weight decay factor for training.')
flags.DEFINE_float('gpu_memory_frac_for_testing', 0,
                   'If non-zero, the fraction of GPU memory that will be used. '
                   'Useful for testing the benchmark script, as this allows '
                   'distributed mode to be run on a single machine. For '
                   'example, if there are two tasks, each can be allocated '
                   '~40 percent of the memory on a single machine. This is '
                   'also useful for using unified memory, as this can be set '
                   'above 1 to oversubscribe the GPU using unified memory.',
                   lower_bound=0.)
flags.DEFINE_boolean('use_unified_memory', None,
                     'If True, allocate unified memory enabling larger models '
                     'to fit in available device RAM.')
flags.DEFINE_boolean('use_tf_layers', True,
                     'If True, use tf.layers for neural network layers. This '
                     'should not affect performance or accuracy in any way.')
flags.DEFINE_integer('tf_random_seed', 1234,
                     'The TensorFlow random seed. Useful for debugging NaNs, '
                     'as this can be set to various values to see if the NaNs '
                     'depend on the seed.')
flags.DEFINE_string('debugger', None,
                    'If set, use the TensorFlow debugger. If set to "cli", use '
                    'the local CLI debugger. Otherwise, this must be in the '
                    'form hostname:port (e.g., localhost:7007) in which case '
                    'the experimental TensorBoard debugger will be used')
flags.DEFINE_boolean('use_python32_barrier', False,
                     'When on, use threading.Barrier at Python 3.2.')

flags.DEFINE_boolean('ml_perf', False,
                     'When True, change how the Imagenet input pipeline works '
                     'slightly to meet the MLPerf compliance rules. This slows '
                     'down the input pipeline. Without this option, at the end '
                     'of the input pipeline, the image is divided by 127.5, '
                     'then 1.0 is subtracted from it, bringing the image '
                     'values from [0, 255] to [-1.0, 1.0]. With this option, '
                     'each of the three channels (red, green, blue) have the '
                     'average channel value among all image subtracted from '
                     'it, and no division is done.')

flags.DEFINE_boolean('datasets_use_prefetch', True,
                     'Enable use of prefetched datasets for input pipeline. '
                     'This option is meaningless if use_datasets=False.')
flags.DEFINE_integer('datasets_prefetch_buffer_size', 1,
                     'Prefetching op buffer size per compute device.')
flags.DEFINE_integer('datasets_num_private_threads', None,
                     'Number of threads for a private threadpool created for '
                     'all datasets computation. By default, we pick an '
                     'appropriate number. If set to 0, we use the default '
                     'tf-Compute threads for dataset operations.')
flags.DEFINE_boolean('datasets_use_caching', False,
                     'Cache the compressed input data in memory. This improves '
                     'the data input performance, at the cost of additional '
                     'memory.')
flags.DEFINE_integer('datasets_parallel_interleave_cycle_length', None,
                     'Number of parallel file readers interleaving input data.')
flags.DEFINE_boolean('datasets_sloppy_parallel_interleave', False,
                     'Allow parallel interleave to depart from deterministic '
                     'ordering, by temporarily skipping over files whose '
                     'elements are not readily available. This can increase '
                     'througput in particular in the presence of stragglers.')
flags.DEFINE_integer('datasets_parallel_interleave_prefetch', None,
                     'The number of input elements to fetch before they are '
                     'needed for interleaving.')

flags.DEFINE_boolean(
    'use_multi_device_iterator', True,
    'If true, we use the MultiDeviceIterator for prefetching, '
    'which deterministically prefetches the data onto the '
    'various GPUs')
flags.DEFINE_integer(
    'multi_device_iterator_max_buffer_size', 1,
    'Configuration parameter for the MultiDeviceIterator that '
    ' specifies the host side buffer size for each device.')

# Performance tuning parameters.
flags.DEFINE_boolean('winograd_nonfused', True,
                     'Enable/disable using the Winograd non-fused algorithms.')
flags.DEFINE_boolean(
    'batchnorm_persistent', True,
    'Enable/disable using the CUDNN_BATCHNORM_SPATIAL_PERSISTENT '
    'mode for batchnorm.')
flags.DEFINE_boolean('sync_on_finish', False,
                     'Enable/disable whether the devices are synced after each '
                     'step.')
flags.DEFINE_boolean('staged_vars', False,
                     'whether the variables are staged from the main '
                     'computation')
flags.DEFINE_boolean('force_gpu_compatible', False,
                     'whether to enable force_gpu_compatible in GPU_Options')
flags.DEFINE_boolean('allow_growth', None,
                     'whether to enable allow_growth in GPU_Options')
flags.DEFINE_boolean('xla', False, 'whether to enable XLA auto-jit compilation')
flags.DEFINE_boolean('xla_compile', False,
                     'Enable xla to compile the graph. Uncompilable ops will '
                     'result in fatal errors.')
flags.DEFINE_boolean('fuse_decode_and_crop', True,
                     'Fuse decode_and_crop for image preprocessing.')
flags.DEFINE_boolean('distort_color_in_yiq', True,
                     'Distort color of input images in YIQ space.')
flags.DEFINE_boolean('enable_optimizations', True,
                     'Whether to enable grappler and other optimizations.')
flags.DEFINE_string('rewriter_config', None,
                    'Config for graph optimizers, described as a '
                    'RewriterConfig proto buffer.')
flags.DEFINE_enum('loss_type_to_report', 'total_loss',
                  ('base_loss', 'total_loss'),
                  'Which type of loss to output and to write summaries for. '
                  'The total loss includes L2 loss while the base loss does '
                  'not. Note that the total loss is always used while '
                  'computing gradients during training if weight_decay > 0, '
                  'but explicitly computing the total loss, instead of just '
                  'computing its gradients, can have a performance impact.')
flags.DEFINE_boolean('single_l2_loss_op', False,
                     'If True, instead of using an L2 loss op per variable, '
                     'concatenate the variables into a single tensor and do a '
                     'single L2 loss on the concatenated tensor.')
flags.DEFINE_boolean('use_resource_vars', False,
                     'Use resource variables instead of normal variables. '
                     'Resource variables are slower, but this option is useful '
                     'for debugging their performance.')
flags.DEFINE_boolean('compute_lr_on_cpu', False,
                     'If True, do computations related to learning rate on the '
                     'CPU instead of the GPU. This will significantly improve '
                     'XLA performance in some cases.')
flags.DEFINE_boolean('sparse_to_dense_grads', False,
                     'If True, convert all sparse gradients to dense gradients '
                     'before passing them to the optimizer to update '
                     'variables. Only affects models with sparse gradients, '
                     'which currently is only the NCF model.')
# Performance tuning specific to MKL.
flags.DEFINE_boolean('mkl', False, 'If true, set MKL environment variables.')
flags.DEFINE_integer('kmp_blocktime', 0,
                     'The time, in milliseconds, that a thread should wait, '
                     'after completing the execution of a parallel region, '
                     'before sleeping')
flags.DEFINE_string('kmp_affinity', 'granularity=fine,verbose,compact,1,0',
                    'Restricts execution of certain threads (virtual execution '
                    'units) to a subset of the physical processing units in a '
                    'multiprocessor computer.')
flags.DEFINE_integer('kmp_settings', 1,
                     'If set to 1, MKL settings will be printed.')

# fp16 parameters. If use_fp16=False, no other fp16 parameters apply.
flags.DEFINE_boolean('use_fp16', False,
                     'Use 16-bit floats for certain tensors instead of 32-bit '
                     'floats. This is currently experimental.')
# TODO(reedwm): The default loss scale of 128 causes most models to diverge
# on the second step with synthetic data. Changing the tf.set_random_seed
# call to tf.set_random_seed(1235) or most other seed values causes the
# issue not to occur.
flags.DEFINE_float('fp16_loss_scale', None,
                   'If fp16 is enabled, the loss is multiplied by this amount '
                   'right before gradients are computed, then each gradient '
                   'is divided by this amount. Mathematically, this has no '
                   'effect, but it helps avoid fp16 underflow. Set to 1 to '
                   'effectively disable. Ignored during eval.')
flags.DEFINE_boolean('fp16_vars', False,
                     'If fp16 is enabled, also use fp16 for variables. If '
                     'False, the variables are stored in fp32 and casted to '
                     'fp16 when retrieved.  Recommended to leave as False.')
flags.DEFINE_boolean('fp16_enable_auto_loss_scale', False,
                     'If True and use_fp16 is True, automatically adjust the '
                     'loss scale during training.')
flags.DEFINE_integer('fp16_inc_loss_scale_every_n', 1000,
                     'If fp16 is enabled and fp16_enable_auto_loss_scale is '
                     'True, increase the loss scale every n steps.')

# The method for managing variables:
#   parameter_server: variables are stored on a parameter server that holds
#       the master copy of the variable. In local execution, a local device
#       acts as the parameter server for each variable; in distributed
#       execution, the parameter servers are separate processes in the
#       cluster.
#       For each step, each tower gets a copy of the variables from the
#       parameter server, and sends its gradients to the param server.
#   replicated: each GPU has its own copy of the variables. To apply
#       gradients, an all_reduce algorithm or or regular cross-device
#       aggregation is used to replicate the combined gradients to all
#       towers (depending on all_reduce_spec parameter setting).
#   independent: each GPU has its own copy of the variables, and gradients
#       are not shared between towers. This can be used to check performance
#       when no data is moved between GPUs.
#   distributed_replicated: Distributed training only. Each GPU has a copy
#       of the variables, and updates its copy after the parameter servers
#       are all updated with the gradients from all servers. Only works with
#       cross_replica_sync=true. Unlike 'replicated', currently never uses
#       nccl all-reduce for replicating within a server.
#   distributed_all_reduce: Distributed training where all replicas run
#       in a single session, using all-reduce to mutally reduce the
#       gradients.  Uses no parameter servers.  When there is only one
#       worker, this is the same as replicated.
#   collective_all_reduce: Distributed training where all replicas run
#       independepently except for variable initialization and for
#       gradient reduction which is done via collective all-reduce.
#       NOTE: collective_all_reduce in conjunction with use_fp16 can
#       lead to NaNs in some models (resnet50).  TODO(tucker): fix it.
#   horovod: Distributed training using Horovod library. Runs workers using
#       an MPI framework (e.g. Open MPI). Each worker runs training on
#       single GPU, and averages gradients using NCCL or MPI all-reduce.
#       See https://github.com/uber/horovod for more details.
flags.DEFINE_enum('variable_update', 'parameter_server',
                  ('parameter_server', 'replicated', 'distributed_replicated',
                   'independent', 'distributed_all_reduce',
                   'collective_all_reduce', 'horovod'),
                  'The method for managing variables: parameter_server, '
                  'replicated, distributed_replicated, independent, '
                  'distributed_all_reduce, collective_all_reduce, horovod')
flags.DEFINE_string('all_reduce_spec', None,
                    'A specification of the all_reduce algorithm to be used '
                    'for reducing gradients.  For more details, see '
                    'parse_all_reduce_spec in variable_mgr.py.  An '
                    'all_reduce_spec has BNF form:\n'
                    'int ::= positive whole number\n'
                    'g_int ::= int[KkMGT]?\n'
                    'alg_spec ::= alg | alg#int\n'
                    'range_spec ::= alg_spec | alg_spec/alg_spec\n'
                    'spec ::= range_spec | range_spec:g_int:range_spec\n'
                    'NOTE: not all syntactically correct constructs are '
                    'supported.\n\n'
                    'Examples:\n '
                    '"xring" == use one global ring reduction for all '
                    'tensors\n'
                    '"pscpu" == use CPU at worker 0 to reduce all tensors\n'
                    '"nccl" == use NCCL to locally reduce all tensors.  '
                    'Limited to 1 worker.\n'
                    '"nccl/xring" == locally (to one worker) reduce values '
                    'using NCCL then ring reduce across workers.\n'
                    '"pscpu:32k:xring" == use pscpu algorithm for tensors of '
                    'size up to 32kB, then xring for larger tensors.')

# If variable_update==distributed_all_reduce then it may be advantageous
# to aggregate small tensors into one prior to reduction.  These parameters
# control that aggregation.
flags.DEFINE_integer('agg_small_grads_max_bytes', 0,
                     'If > 0, try to aggregate tensors of less than this '
                     'number of bytes prior to all-reduce.')
flags.DEFINE_integer('agg_small_grads_max_group', 10,
                     'When aggregating small tensors for all-reduce do not '
                     'aggregate more than this many into one new tensor.')
flags.DEFINE_integer('allreduce_merge_scope', 1,
                     'Establish a name scope around this many '
                     'gradients prior to creating the all-reduce operations. '
                     'It may affect the ability of the backend to merge '
                     'parallel ops.')

# Distributed training parameters.
flags.DEFINE_enum('job_name', '', ('ps', 'worker', 'controller', ''),
                  'One of "ps", "worker", "controller", "".  Empty for local '
                  'training')
flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('controller_host', None, 'optional controller host')
flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
flags.DEFINE_boolean('cross_replica_sync', True, '')
flags.DEFINE_string('horovod_device', '', 'Device to do Horovod all-reduce on: '
                    'empty (default), cpu or gpu. Default with utilize GPU if '
                    'Horovod was compiled with the HOROVOD_GPU_ALLREDUCE '
                    'option, and CPU otherwise.')

# Summary and Save & load checkpoints.
flags.DEFINE_integer('summary_verbosity', 0, 'Verbosity level for summary ops. '
                     'level 0: disable any summary.\n'
                     'level 1: small and fast ops, e.g.: learning_rate, '
                     'total_loss.\n'
                     'level 2: medium-cost ops, e.g. histogram of all '
                     'gradients.\n'
                     'level 3: expensive ops: images and histogram of each '
                     'gradient.\n')
flags.DEFINE_integer('save_summaries_steps', 0,
                     'How often to save summaries for trained models. Pass 0 '
                     'to disable summaries.')
flags.DEFINE_integer('save_model_secs', 0,
                     'How often to save trained models. Pass 0 to disable '
                     'saving checkpoints every N seconds. A checkpoint is '
                     'saved after training completes regardless of this '
                     'option.')
flags.DEFINE_integer('save_model_steps', None,
                     'How often to save trained models. If specified, '
                     'save_model_secs must not be specified.')
flags.DEFINE_integer('max_ckpts_to_keep', 5,
                     'Max number of checkpoints to keep.')
flags.DEFINE_string('train_dir', None,
                    'Path to session checkpoints. Pass None to disable saving '
                    'checkpoint at the end.')
flags.DEFINE_string('eval_dir', '/tmp/tf_cnn_benchmarks/eval',
                    'Directory where to write eval event logs.')
flags.DEFINE_string('backbone_model_path', None,
                    'Path to pretrained backbone model checkpoint. Pass None '
                    'if not using a backbone model.')
flags.DEFINE_enum('trt_mode', '', ['', 'FP32', 'FP16', 'INT8'],
                  'If this is specified in forward_only mode and '
                  'freeze_when_forward_only is set to True, use TensorRT to '
                  'optimize the graph before execution.')
flags.DEFINE_integer('trt_max_workspace_size_bytes', 4 << 30,
                     'Max workspace size bytes used by the TensorRT optimizer.')

# Benchmark logging for model garden metric
flags.DEFINE_string('benchmark_log_dir', None,
                    'The directory to place the log files containing the '
                    'results of benchmark. The logs are created by '
                    'BenchmarkFileLogger. Requires the root of the Tensorflow '
                    'models repository to be in $PYTHTONPATH.')
flags.DEFINE_string('benchmark_test_id', None,
                    'The unique test ID of the benchmark run. It could be the '
                    'combination of key parameters. It is hardware independent '
                    'and could be used compare the performance between '
                    'different test runs. This flag is designed for human '
                    'consumption, and does not have any impact within the '
                    'system.')

# Activation quantization parameters
flags.DEFINE_boolean('use_relu_x', False,
                     'Replace relu(input) activations by max(min(input,x),0)')
flags.DEFINE_string('last_act_name', 'cg/final',
                    'The name of the last activation')
flags.DEFINE_boolean('relu_x_per_channel', False,
                     'Use a separate x parameter for each channel.')
flags.DEFINE_float('relu_x_decay', 0, 'The strength of L2 regularization on '
                   'the x (upper boundary of relu)')
flags.DEFINE_float('init_relu_x', 6.0, 'Initial value for x (relu boundary)')
flags.DEFINE_enum('relu_x_update', 'fixed',
                  ('fixed', 'gradient_descent', 'moving_average'),
                  'Method to update the x value, should be one of fixed, '
                  'gradient_descent or moving_average')
flags.DEFINE_boolean('quant_act', False, 'Quantize activations.')
flags.DEFINE_integer('quant_act_bits', 4, 'Activation quantization precision')
flags.DEFINE_integer('quant_act_delay', None, 'Delay steps to turn on '
                     'activation fake quantization during training.')

# Weight quantization parameters
flags.DEFINE_boolean('quant_weight', False, 'Quantize weights.')
flags.DEFINE_integer('quant_weight_bits', 4, 'Weight quantization precision')
flags.DEFINE_string('first_weight_name', 'cg/conv0',
                    'The name of the first weight')
flags.DEFINE_string('last_weight_name', 'cg/affine0',
                    'The name of the last weight')
flags.DEFINE_boolean('quant_weight_per_channel', False, 'Whether to quantize.'
                     'weights at per-channel granularity')
flags.DEFINE_boolean('quant_weight_narrow_range', True, 'Whether to use the '
                     'narrow quantization range [1; 2^num_bits - 1] or wide '
                     'range [0; 2^num_bits - 1] for weight quantization.')
flags.DEFINE_float('quant_weight_relative_quantile', 0.0, 'The quantile of '
                   'weights to place boundary for weight quantization.')
flags.DEFINE_boolean('freeze_weight_range', False, 'Whether to freeze the '
                     'boundary of weight quantization during training.')
flags.DEFINE_integer('quant_weight_delay', None, 'Delay steps to turn on '
                     'weight fake quantization during training.')
flags.DEFINE_boolean('tanh_weight_transform', False, 'Whether to pass weights '
                     'through normalized tanh transformation (as described in '
                     'DoReFa quantization paper) before convolution/matmul')

# Weight equalization parameters
flags.DEFINE_float('weight_equalization', 0,
                   'The weight equalization strength.')
flags.DEFINE_enum('weight_equalization_type', 'kurtosis', ('kurtosis', 'tail'),
                  'The type of weight equalization loss to apply, should be '
                  'either kurtosis or tail.')


platforms_util.define_platform_params()


class GlobalStepWatcher(threading.Thread):
  """A helper class for global_step.

  Polls for changes in the global_step of the model, and finishes when the
  number of steps for the global run are done.
  """

  def __init__(self, sess, global_step_op, start_at_global_step,
               end_at_global_step):
    threading.Thread.__init__(self)
    self.sess = sess
    self.global_step_op = global_step_op
    self.start_at_global_step = start_at_global_step
    self.end_at_global_step = end_at_global_step

    self.start_time = 0
    self.start_step = 0
    self.finish_time = 0
    self.finish_step = 0

  def run(self):
    while self.finish_time == 0:
      time.sleep(.25)
      global_step_val, = self.sess.run([self.global_step_op])
      if self.start_time == 0 and global_step_val >= self.start_at_global_step:
        # Use tf.logging.info instead of log_fn, since print (which is log_fn)
        # is not thread safe and may interleave the outputs from two parallel
        # calls to print, which can break tests.
        tf.logging.info('Starting real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.start_time = time.time()
        self.start_step = global_step_val
      if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
        tf.logging.info('Finishing real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.finish_time = time.time()
        self.finish_step = global_step_val

  def done(self):
    return self.finish_time > 0

  def num_steps(self):
    return self.finish_step - self.start_step

  def elapsed_time(self):
    return self.finish_time - self.start_time


class CheckpointNotFoundException(Exception):
  pass


def create_config_proto(params):
  """Returns session config proto.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  if params.num_intra_threads is None:
    if params.device == 'gpu':
      config.intra_op_parallelism_threads = 1
  else:
    config.intra_op_parallelism_threads = params.num_intra_threads
  config.inter_op_parallelism_threads = params.num_inter_threads
  config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
  config.gpu_options.force_gpu_compatible = params.force_gpu_compatible
  if params.allow_growth is not None:
    config.gpu_options.allow_growth = params.allow_growth
  if params.gpu_memory_frac_for_testing > 0:
    config.gpu_options.per_process_gpu_memory_fraction = (
        params.gpu_memory_frac_for_testing)
  if params.use_unified_memory:
    config.gpu_options.experimental.use_unified_memory = (
        params.use_unified_memory)
  if params.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
    # TODO(b/117324590): Re-enable PinToHostOptimizer when b/117324590 is fixed.
    # Currently we have to disable PinToHostOptimizer w/ XLA since it causes
    # OOM/perf cliffs.
    config.graph_options.rewrite_options.pin_to_host_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)
  if params.rewriter_config:
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    text_format.Merge(params.rewriter_config, rewriter_config)
    config.graph_options.rewrite_options.CopyFrom(rewriter_config)
  elif not params.enable_optimizations:
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.layout_optimizer = off
    rewrite_options.constant_folding = off
    rewrite_options.shape_optimization = off
    rewrite_options.remapping = off
    rewrite_options.arithmetic_optimization = off
    rewrite_options.dependency_optimization = off
    rewrite_options.loop_optimization = off
    rewrite_options.function_optimization = off
    rewrite_options.debug_stripper = off
    rewrite_options.disable_model_pruning = True
    rewrite_options.scoped_allocator_optimization = off
    rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
    rewrite_options.pin_to_host_optimization = off
  elif params.variable_update == 'collective_all_reduce':
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')
  if params.variable_update == 'horovod':
    import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
    config.gpu_options.visible_device_list = str(hvd.local_rank())

  return config


def get_mode_from_params(params):
  """Returns the mode in which this script is running.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  Raises:
    ValueError: Unsupported params settings.
  """
  if params.forward_only and params.eval:
    raise ValueError('Only one of forward_only and eval parameters is true')

  if params.eval:
    return constants.BenchmarkMode.EVAL
  elif params.forward_only:
    return constants.BenchmarkMode.FORWARD_ONLY
  elif (params.eval_during_training_every_n_steps or
        params.eval_during_training_every_n_epochs or
        params.eval_during_training_at_specified_steps or
        params.eval_during_training_at_specified_epochs):
    return constants.BenchmarkMode.TRAIN_AND_EVAL
  else:
    return constants.BenchmarkMode.TRAIN


# How many digits to show for the loss and accuracies during training.
LOSS_AND_ACCURACY_DIGITS_TO_SHOW = 3


def benchmark_one_step(sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       partitioned_graph_file_prefix,
                       profiler,
                       image_producer,
                       params,
                       summary_op=None,
                       show_images_per_sec=True,
                       benchmark_logger=None,
                       collective_graph_key=0):
  """Advance one step of benchmarking."""
  should_profile = profiler and 0 <= step < _NUM_STEPS_TO_PROFILE
  need_options_and_metadata = (
      should_profile or collective_graph_key > 0 or
      ((trace_filename or partitioned_graph_file_prefix) and step == -2)
  )
  if need_options_and_metadata:
    run_options = tf.RunOptions()
    if (trace_filename and step == -2) or should_profile:
      run_options.trace_level = tf.RunOptions.FULL_TRACE
    if partitioned_graph_file_prefix and step == -2:
      run_options.output_partition_graphs = True
    if collective_graph_key > 0:
      run_options.experimental.collective_graph_key = collective_graph_key
    run_metadata = tf.RunMetadata()
  else:
    run_options = None
    run_metadata = None
  summary_str = None
  start_time = time.time()
  if summary_op is None:
    results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
  else:
    (results, summary_str) = sess.run(
        [fetches, summary_op], options=run_options, run_metadata=run_metadata)

  if not params.forward_only:
    lossval = results['average_loss']
  else:
    lossval = 0.
  if image_producer is not None:
    image_producer.notify_image_consumption()
  train_time = time.time() - start_time
  step_train_times.append(train_time)
  if (show_images_per_sec and step >= 0 and
      (step == 0 or (step + 1) % params.display_every == 0)):
    speed_mean, speed_uncertainty, speed_jitter = get_perf_timing(
        batch_size, step_train_times, params.display_perf_ewma)
    log_str = '%i\t%s\t%.*f' % (
        step + 1,
        get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter),
        LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
    if 'top_1_accuracy' in results:
      log_str += '\t%.*f\t%.*f' % (
          LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'],
          LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_5_accuracy'])
    log_fn(log_str)
    if benchmark_logger:
      benchmark_logger.log_metric(
          'current_examples_per_sec', speed_mean, global_step=step + 1)
      if 'top_1_accuracy' in results:
        benchmark_logger.log_metric(
            'top_1_accuracy', results['top_1_accuracy'], global_step=step + 1)
        benchmark_logger.log_metric(
            'top_5_accuracy', results['top_5_accuracy'], global_step=step + 1)
  if need_options_and_metadata:
    if should_profile:
      profiler.add_step(step, run_metadata)
    if trace_filename and step == -2:
      log_fn('Dumping trace to %s' % trace_filename)
      trace_dir = os.path.dirname(trace_filename)
      if not gfile.Exists(trace_dir):
        gfile.MakeDirs(trace_dir)
      with gfile.Open(trace_filename, 'w') as trace_file:
        if params.use_chrome_trace_format:
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
        else:
          trace_file.write(str(run_metadata.step_stats))
    if partitioned_graph_file_prefix and step == -2:
      path, filename = os.path.split(partitioned_graph_file_prefix)
      if '.' in filename:
        base_filename, ext = filename.rsplit('.', 1)
        ext = '.' + ext
      else:
        base_filename, ext = filename, ''
      as_text = filename.endswith('txt')
      for graph_def in run_metadata.partition_graphs:
        device = graph_def.node[0].device.replace('/', '_').replace(':', '_')
        graph_filename = '%s%s%s' % (base_filename, device, ext)
        log_fn('Writing partitioned GraphDef as %s to %s' % (
            'text' if as_text else 'binary',
            os.path.join(path, graph_filename)))
        tf.train.write_graph(graph_def, path, graph_filename, as_text)
  return (summary_str, lossval)


def get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter, scale=1):
  if scale == 1:
    # TODO(laigd): rename 'images' to maybe 'inputs', same below.
    return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
            (speed_mean, speed_uncertainty, speed_jitter))
  else:
    return 'images/sec: %.1f' % speed_mean


def get_perf_timing(batch_size, step_train_times, ewma_alpha=None, scale=1):
  """Calculate benchmark processing speed."""
  times = np.array(step_train_times)
  speeds = batch_size / times
  if ewma_alpha:
    weights = np.logspace(len(times)-1, 0, len(times), base=1-ewma_alpha)
    time_mean = np.average(times, weights)
  else:
    time_mean = np.mean(times)
  speed_mean = scale * batch_size / time_mean
  speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
  speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
  return speed_mean, speed_uncertainty, speed_jitter


def load_checkpoint(saver, sess, ckpt_dir):
  """Loads checkpoint from provided directory or full path.

  Args:
    saver: Saver used to restore the checkpoint.
    sess: TensorFlow session.
    ckpt_dir: Path to a folder of checkpoints or full path to a checkpoint.

  Returns:
    Global step.
  """
  model_checkpoint_path = _get_checkpoint_to_load(ckpt_dir)
  global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]
  if not global_step.isdigit():
    global_step = 0
  else:
    global_step = int(global_step)
  saver.restore(sess, model_checkpoint_path)
  log_fn('Successfully loaded model from %s.' % model_checkpoint_path)
  return global_step


def _get_checkpoint_to_load(ckpt_dir):
  """Returns which checkpoint to load.

  Args:
    ckpt_dir: Path to a folder of checkpoints or full path to a checkpoint.

  Returns:
    Full path to checkpoint to load.

  Raises:
    CheckpointNotFoundException: If checkpoint is not found.
  """
  p = re.compile(r'ckpt-\d+$')
  if p.search(ckpt_dir):
    model_checkpoint_path = ckpt_dir
  else:
    # Finds latest checkpoint in directory provided
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      model_checkpoint_path = ckpt.model_checkpoint_path
    else:
      raise CheckpointNotFoundException('No checkpoint file found in dir:{}'.
                                        format(ckpt_dir))
  return model_checkpoint_path


# Params are passed to BenchmarkCNN's constructor. Params is a map from name
# to value, with one field per key in flags.param_specs.
#
# Call make_params() or make_params_from_flags() below to construct a Params
# tuple with default values from flags.param_specs, rather than constructing
# Params directly.
Params = namedtuple('Params', flags.param_specs.keys())  # pylint: disable=invalid-name


def validate_params(params):
  """Validates that the Params tuple had valid values.

  When command-line flags are defined for each ParamSpec by calling
  flags.define_flags(), calling this function is unnecessary because absl
  already does flag validation. Otherwise, this function should be called.

  Args:
     params: A Params tuple.
  Raises:
    ValueError: An element of params had an invalid value.
  """
  for name, value in params._asdict().items():
    param_spec = flags.param_specs[name]
    if param_spec.flag_type in ('integer', 'float'):
      if (value is not None and param_spec.kwargs['lower_bound'] is not None and
          value < param_spec.kwargs['lower_bound']):
        raise ValueError('Param %s value of %s is lower than the lower bound '
                         'of %s' %
                         (name, value, param_spec.kwargs['lower_bound']))
      if (value is not None and param_spec.kwargs['upper_bound'] is not None and
          param_spec.kwargs['upper_bound'] < value):
        raise ValueError('Param %s value of %s is higher than the upper bound '
                         'of %s' %
                         (name, value, param_spec.kwargs['upper_bound']))
    elif (value is not None and param_spec.flag_type == 'enum' and
          value not in param_spec.kwargs['enum_values']):
      raise ValueError('Param %s of value %s is not in %s'%
                       (name, value, param_spec.kwargs['enum_values']))


def make_params(**kwargs):
  """Create a Params tuple for BenchmarkCNN from kwargs.

  Default values are filled in from flags.param_specs.

  Args:
    **kwargs: kwarg values will override the default values.
  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Create a (name: default_value) map from flags.param_specs.
  default_kwargs = {
      name: flags.param_specs[name].default_value
      for name in flags.param_specs
  }
  params = Params(**default_kwargs)._replace(**kwargs)
  validate_params(params)
  return params


def make_params_from_flags():
  """Create a Params tuple for BenchmarkCNN from absl_flags.FLAGS.

  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Collect (name: value) pairs for absl_flags.FLAGS with matching names in
  # flags.param_specs.
  flag_values = {name: getattr(absl_flags.FLAGS, name)
                 for name in flags.param_specs.keys()}
  return Params(**flag_values)


def remove_param_fields(params, fields_to_remove):
  """Remove fields from a Params namedtuple."""
  params_dict = params._asdict()
  for field in fields_to_remove:
    assert field in params_dict, 'Invalid Params field: ' + field
  params_dict = {k: v for k, v in params_dict.items()
                 if k not in fields_to_remove}
  new_params_type = namedtuple('Params', params_dict.keys())
  return new_params_type(**params_dict)


def get_num_batches_and_epochs(params, batch_size, num_examples_per_epoch):
  """Returns the number of batches and epochs to run for.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    batch_size: The number of images per step.
    num_examples_per_epoch: The number of images in a single epoch.

  Returns:
    num_batches: The number of batches to run for.
    num_epochs: The number of epochs to run for. This might be slightly
      smaller than params.num_epochs if specified, because the number of batches
      must be an integer.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  if params.num_batches and params.num_epochs:
    raise ValueError('At most one of --num_batches and --num_epochs may be '
                     'specified.')
  if params.num_epochs:
    num_batches = int(params.num_epochs * num_examples_per_epoch +
                      batch_size - 1) // batch_size
  else:
    num_batches = params.num_batches or _DEFAULT_NUM_BATCHES
  num_epochs = num_batches * batch_size / num_examples_per_epoch
  return (num_batches, num_epochs)


def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
  """Returns a piecewise learning rate tensor.

  Args:
    piecewise_learning_rate_schedule: The --piecewise_learning_rate_schedule
      parameter
    global_step: Scalar tensor representing the global step.
    num_batches_per_epoch: float indicating the number of batches per epoch.

  Returns:
    A scalar float tensor, representing the learning rate.

  Raises:
    ValueError: piecewise_learning_rate_schedule is not formatted correctly.
  """
  pieces = piecewise_learning_rate_schedule.split(';')
  if len(pieces) % 2 == 0:
    raise ValueError('--piecewise_learning_rate_schedule must have an odd '
                     'number of components')
  values = []
  boundaries = []
  for i, piece in enumerate(pieces):
    if i % 2 == 0:
      try:
        values.append(float(piece))
      except ValueError:
        raise ValueError('Invalid learning rate: ' + piece)
    else:
      try:
        boundaries.append(int(int(piece) * num_batches_per_epoch) - 1)
      except ValueError:
        raise ValueError('Invalid epoch: ' + piece)
  return tf.train.piecewise_constant(global_step, boundaries, values,
                                     name='piecewise_learning_rate')


def get_learning_rate(params, global_step, num_examples_per_epoch, model,
                      batch_size):
  """Returns a learning rate tensor based on global_step.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.
    global_step: Scalar tensor representing the global step.
    num_examples_per_epoch: The number of examples per epoch.
    model: The model.Model object to obtain the default learning rate from if no
      learning rate is specified.
    batch_size: Number of examples per step

  Returns:
    A scalar float tensor, representing the learning rate. When evaluated, the
    learning rate depends on the current value of global_step.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  with tf.name_scope('learning_rate'):
    num_batches_per_epoch = num_examples_per_epoch / batch_size

    if params.piecewise_learning_rate_schedule:
      if (params.init_learning_rate is not None or
          params.learning_rate_decay_factor or
          params.minimum_learning_rate or params.num_epochs_per_decay):
        raise ValueError('No other learning rate-related flags can be '
                         'specified if --piecewise_learning_rate_schedule is '
                         'specified')
      learning_rate = get_piecewise_learning_rate(
          params.piecewise_learning_rate_schedule,
          global_step, num_batches_per_epoch)
    elif params.init_learning_rate is not None:
      learning_rate = params.init_learning_rate
      if (params.num_epochs_per_decay > 0 and
          params.learning_rate_decay_factor > 0):
        decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(
            params.init_learning_rate,
            global_step,
            decay_steps,
            params.learning_rate_decay_factor,
            staircase=True)

        if params.minimum_learning_rate != 0.:
          learning_rate = tf.maximum(learning_rate,
                                     params.minimum_learning_rate)
    else:
      learning_rate = model.get_learning_rate(global_step, batch_size)
    if params.num_learning_rate_warmup_epochs > 0 and (
        params.init_learning_rate is not None or
        params.piecewise_learning_rate_schedule):
      warmup_steps = int(num_batches_per_epoch *
                         params.num_learning_rate_warmup_epochs)
      init_lr = params.init_learning_rate
      if init_lr is None:
        init_lr = float(params.piecewise_learning_rate_schedule.split(';')[0])
      warmup_lr = init_lr * tf.cast(global_step, tf.float32) / tf.cast(
          warmup_steps, tf.float32)
      learning_rate = tf.cond(global_step < warmup_steps,
                              lambda: warmup_lr, lambda: learning_rate)

    learning_rate = mlperf.logger.log_deferred_tensor_value(
        mlperf.tags.OPT_LR, learning_rate, global_step, every_n=100)
  return learning_rate


def get_optimizer(params, learning_rate):
  """Returns the optimizer that should be used based on params."""
  if params.optimizer == 'momentum':
    mlperf.logger.log(key=mlperf.tags.OPT_NAME,
                      value=mlperf.tags.SGD_WITH_MOMENTUM)
    mlperf.logger.log(key=mlperf.tags.OPT_MOMENTUM, value=params.momentum)
    opt = tf.train.MomentumOptimizer(
        learning_rate, params.momentum, use_nesterov=True)
  elif params.optimizer == 'sgd':
    mlperf.logger.log(key=mlperf.tags.OPT_NAME, value=mlperf.tags.SGD)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  elif params.optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(
        learning_rate,
        params.rmsprop_decay,
        momentum=params.rmsprop_momentum,
        epsilon=params.rmsprop_epsilon)
  elif params.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate, params.adam_beta1,
                                 params.adam_beta2, params.adam_epsilon)
  else:
    raise ValueError('Optimizer "{}" was not recognized'.
                     format(params.optimizer))
  return opt


def generate_tfprof_profile(profiler, tfprof_file):
  """Generates a tfprof profile, writing it to a file and printing top ops.

  Args:
    profiler: A tf.profiler.Profiler. `profiler.add_step` must have already been
      called.
    tfprof_file: The filename to write the ProfileProto to.
  """
  profile_proto = profiler.serialize_to_string()
  log_fn('Dumping ProfileProto to %s' % tfprof_file)
  with gfile.Open(tfprof_file, 'wb') as f:
    f.write(profile_proto)

  # Print out the execution times of the top operations. Note this
  # information can also be obtained with the dumped ProfileProto, but
  # printing it means tfprof doesn't have to be used if all the user wants
  # is the top ops.
  options = tf.profiler.ProfileOptionBuilder.time_and_memory()
  options['max_depth'] = _NUM_OPS_TO_PRINT
  options['order_by'] = 'accelerator_micros'
  profiler.profile_operations(options)


class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self, params, dataset=None, model=None):
    """Initialize BenchmarkCNN.

    Args:
      params: Params tuple, typically created by make_params or
              make_params_from_flags.
      dataset: If not None, the dataset to use. Otherwise, params is used to
               obtain the dataset.
      model: If not None, the model to use. Otherwise, params is used to obtain
             the model.
    Raises:
      ValueError: Unsupported params settings.
    """
    mlperf.logger.log(key=mlperf.tags.RUN_START)
    self.params = params
    if params.eval:
      self._doing_eval = True
    else:
      # Note self._doing_eval can later switch to True in self._do_eval() if
      # self.params.eval_during_training_* is specified.
      self._doing_eval = False
    self.dataset = dataset or datasets.create_dataset(self.params.data_dir,
                                                      self.params.data_name)
    self.model = model or model_config.get_model_config(
        self.params.model, self.dataset, self.params)
    self.trace_filename = self.params.trace_file
    self.rewriter_config = self.params.rewriter_config
    autotune_threshold = self.params.autotune_threshold if (
        self.params.autotune_threshold) else 1
    min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
    self.num_warmup_batches = self.params.num_warmup_batches if (
        self.params.num_warmup_batches is not None) else max(
            10, min_autotune_warmup)
    self.graph_file = self.params.graph_file
    self.resize_method = self.params.resize_method
    self.sync_queue_counter = 0
    self.num_gpus = self.params.num_gpus
    if self.params.gpu_indices:
      self.gpu_indices = [int(x) for x in self.params.gpu_indices.split(',')]
    else:
      self.gpu_indices = [x for x in range(self.num_gpus)]

    if (self.params.device == 'cpu' and self.params.data_format == 'NCHW' and
        not self.params.mkl):
      raise ValueError('device=cpu requires that data_format=NHWC')

    if ((self.params.num_epochs_per_decay or
         self.params.learning_rate_decay_factor) and
        not (self.params.init_learning_rate is not None and
             self.params.num_epochs_per_decay
             and self.params.learning_rate_decay_factor)):
      raise ValueError('If one of num_epochs_per_decay or '
                       'learning_rate_decay_factor is set, both must be set'
                       'and learning_rate must be set')
    if (self.params.minimum_learning_rate and
        not (self.params.init_learning_rate is not None and
             self.params.num_epochs_per_decay and
             self.params.learning_rate_decay_factor)):
      raise ValueError('minimum_learning_rate requires learning_rate,'
                       'num_epochs_per_decay, and '
                       'learning_rate_decay_factor to be set')

    if (self.params.use_fp16 and self.params.fp16_vars and
        'replicated' in self.params.variable_update and
        self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec):
      raise ValueError('fp16 variables are not supported with NCCL')
    if (self.params.use_fp16 and self.params.fp16_vars and
        self.params.gradient_repacking):
      raise ValueError('--fp16_vars cannot be used with --gradient_repacking')

    if self.params.variable_update == 'horovod' and self.params.num_gpus > 1:
      raise ValueError('Horovod benchmarks require num_gpus=1 on each worker')

    if self.params.variable_update == 'horovod' and self.params.job_name:
      raise ValueError('job_name should not be specified for Horovod.')

    if self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale:
      if self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec:
        raise ValueError('Automatic loss scaling is not supported with NCCL.')
      if self.params.variable_update not in ('parameter_server', 'replicated',
                                             'independent'):
        raise ValueError('Automatic loss scaling is not supported with '
                         'variable_update=%s.' % self.params.variable_update)
      if self.params.staged_vars:
        raise ValueError('Automatic loss scaling is not supported with'
                         'staged_vars.')

    if (self.params.debugger is not None and self.params.debugger != 'cli' and
        ':' not in self.params.debugger):
      raise ValueError('--debugger must be "cli" or in the form '
                       'host:port')

    if self.params.hierarchical_copy and self.params.num_gpus <= 1:
      raise ValueError('--hierarchical_copy requires --num_gpus to be greater '
                       'than 1')

    if params.save_model_secs and params.save_model_steps:
      raise ValueError('At most one of --save_model_secs and '
                       '--save_model_steps can be specified')

    eval_during_training_flags = list(map(bool, [
        params.eval_during_training_every_n_steps,
        params.eval_during_training_every_n_epochs,
        params.eval_during_training_at_specified_steps,
        params.eval_during_training_at_specified_epochs,
    ]))

    if eval_during_training_flags.count(True) > 1:
      raise ValueError('At most one flag with --eval_during_training_* prefix '
                       'must be specified.')

    eval_during_training_enabled = any(eval_during_training_flags)

    if eval_during_training_enabled:
      if params.eval:
        raise ValueError('At most one of --eval and --eval_during_training_* '
                         'must be specified')
      if params.forward_only:
        raise ValueError('At most one of --forward_only and '
                         '--eval_during_training_* must be specified')
      if params.job_name:
        raise ValueError('--eval_during_training_* is not yet supported in '
                         'distributed mode.')
      if params.staged_vars:
        raise ValueError('--eval_during_training_* is not currently compatible '
                         'with --staged_vars')

    if params.stop_at_top_1_accuracy and not eval_during_training_enabled:
      raise ValueError('--stop_at_top_1_accuracy is only supported with '
                       '--eval_during_training_*')
    if params.collect_eval_results_async and params.model != 'ssd300':
      raise ValueError('--collect_eval_results_async only works with ssd300 '
                       'model currently.')
    if self.params.forward_only and self.params.freeze_when_forward_only:
      if self.params.train_dir is not None:
        raise ValueError('In forward_only mode, when --freeze_when_forward_only'
                         ' is True, --train_dir should not be specified')
      if self.params.data_dir and not self.params.datasets_use_prefetch:
        raise ValueError('In forward_only mode, when --freeze_when_forward_only'
                         ' is True and --data_dir is set, '
                         '--datasets_use_prefetch should be set to True')
      if self.params.job_name:
        raise ValueError('In forward_only mode, when --freeze_when_forward_only'
                         ' is True, --job_name should not be specified and '
                         'distributed running is not supported')
      self.forward_only_and_freeze = True
    else:
      self.forward_only_and_freeze = False
      if self.params.trt_mode:
        raise ValueError('--trt_mode should not be specified if one of '
                         '--forward_only and --freeze_when_forward_only is set '
                         'to False')

    self.mode = get_mode_from_params(self.params)

    # Use the batch size from the command line if specified, otherwise use the
    # model's default batch size.  Scale the benchmark's batch size by the
    # number of GPUs.
    if self.params.batch_size > 0:
      self.model.set_batch_size(self.params.batch_size)
    self.batch_size = self.model.get_batch_size() * self.num_gpus
    if self.mode in (constants.BenchmarkMode.TRAIN,
                     constants.BenchmarkMode.TRAIN_AND_EVAL):
      self.train_batch_size = self.batch_size
    else:
      self.train_batch_size = None
    if self.mode in (constants.BenchmarkMode.EVAL,
                     constants.BenchmarkMode.TRAIN_AND_EVAL):
      if self.params.eval_batch_size > 0:
        self.eval_batch_size = self.params.eval_batch_size * self.num_gpus
      else:
        self.eval_batch_size = self.batch_size
    else:
      self.eval_batch_size = None
    self.batch_group_size = self.params.batch_group_size
    self.enable_auto_loss_scale = (
        self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale)
    self.loss_scale = None
    self.loss_scale_normal_steps = None

    self.job_name = self.params.job_name  # "" for local training

    # PS server is used for distributed jobs not using all-reduce.
    use_ps_server = self.job_name and (self.params.variable_update !=
                                       'distributed_all_reduce' and
                                       self.params.variable_update !=
                                       'collective_all_reduce')
    # controller is used for distributed_all_reduce with > 1 worker.
    use_controller = (
        self.params.variable_update == 'distributed_all_reduce' and
        self.job_name)
    if use_controller and not params.controller_host:
      raise ValueError('When variable_update==distributed_all_reduce '
                       'controller_host must also be specified.')
    # collective_all_reduce doesn't need a controller or ps
    self.distributed_collective = (
        self.params.variable_update == 'collective_all_reduce' and
        self.job_name)

    self.local_parameter_device_flag = self.params.local_parameter_device
    if self.job_name:
      self.task_index = self.params.task_index
      self.cluster_manager = platforms_util.get_cluster_manager(
          params, create_config_proto(params))
      assert isinstance(self.cluster_manager, cnn_util.BaseClusterManager)

      worker_prefix = '/job:worker/replica:0/task:%s' % self.task_index
      if use_ps_server:
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=worker_prefix + '/cpu:0',
            cluster=self.cluster_manager.get_cluster_spec())
        # This device on which the queues for managing synchronization between
        # servers should be stored.
        self.sync_queue_devices = [
            '/job:ps/replica:0/task:%s/cpu:0' % i
            for i in range(self.cluster_manager.num_ps())
        ]
      else:
        self.sync_queue_devices = ['/job:worker/replica:0/task:0/cpu:0']
    else:
      self.task_index = 0
      self.cluster_manager = None
      worker_prefix = ''
      self.param_server_device = '/%s:0' % self.params.local_parameter_device
      self.sync_queue_devices = [self.param_server_device]

    if self.cluster_manager:
      self.num_workers = self.cluster_manager.num_workers()
    elif self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      self.num_workers = hvd.size()
    else:
      self.num_workers = 1
    self.num_ps = self.cluster_manager.num_ps() if self.cluster_manager else 0

    if self.num_workers > 1 and self.params.all_reduce_spec == 'nccl':
      raise ValueError('--all_reduce_spec=nccl is invalid in a '
                       'multi-worker job')

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '%s/cpu:0' % worker_prefix

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = [
        '%s/%s:%i' % (worker_prefix, self.params.device, i)
        for i in xrange(self.num_gpus)
    ]

    subset = 'validation' if params.eval else 'train'
    self.num_batches, self.num_epochs = get_num_batches_and_epochs(
        params, self.batch_size * self.num_workers,
        self.dataset.num_examples_per_epoch(subset))
    if self.mode in (constants.BenchmarkMode.EVAL,
                     constants.BenchmarkMode.TRAIN_AND_EVAL):
      # TODO(reedwm): Currently we do extra eval logic for num_eval_batches and
      # the preprocessor. We should encapsulate this logic into a shared
      # function or class.
      if params.num_eval_batches is None and params.num_eval_epochs is None:
        eval_params = self.params
      else:
        eval_params = self.params._replace(
            num_batches=self.params.num_eval_batches,
            num_epochs=self.params.num_eval_epochs)
      self.num_eval_batches, self.num_eval_epochs = get_num_batches_and_epochs(
          eval_params, self.eval_batch_size * self.num_workers,
          self.dataset.num_examples_per_epoch('validation'))
    else:
      self.num_eval_batches, self.num_eval_epochs = None, None

    num_train_examples_per_epoch = self.dataset.num_examples_per_epoch('train')
    if self.params.eval_during_training_every_n_epochs:
      n_epochs = self.params.eval_during_training_every_n_epochs
      self.eval_during_training_at_specified_steps = {
          (int(e * num_train_examples_per_epoch + self.batch_size - 1) //
           self.batch_size)
          for e in np.arange(n_epochs, self.num_epochs, n_epochs)}

    if self.params.eval_during_training_at_specified_steps:
      try:
        self.eval_during_training_at_specified_steps = set(map(
            int, self.params.eval_during_training_at_specified_steps))
      except ValueError:
        raise ValueError('Param eval_during_training_at_specified_steps value '
                         'of %s cannot be converted to a list of integers.' %
                         (self.params.eval_during_training_at_specified_steps))

    if self.params.eval_during_training_at_specified_epochs:
      try:
        n_epochs = list(map(
            float, self.params.eval_during_training_at_specified_epochs))
        offset = n_epochs[0] - 1
        if offset.is_integer():
          offset = int(offset)
        mlperf.logger.log(key=mlperf.tags.EVAL_EPOCH_OFFSET, value=offset)
        self.eval_during_training_at_specified_steps = {
            (int(e * num_train_examples_per_epoch + self.batch_size - 1) //
             self.batch_size)
            for e in n_epochs}
      except ValueError:
        raise ValueError('Param eval_during_training_at_specified_epochs value '
                         'of %s cannot be converted to a list of floats.' %
                         (self.params.eval_during_training_at_specified_epochs))

    if params.eval_during_training_every_n_epochs:
      offset = params.eval_during_training_every_n_epochs - 1
      if offset.is_integer():
        offset = int(offset)
      mlperf.logger.log(key=mlperf.tags.EVAL_EPOCH_OFFSET, value=offset)

    if (self.params.staged_vars and
        self.params.variable_update != 'parameter_server'):
      raise ValueError('staged_vars for now is only supported with '
                       'variable_update=parameter_server')

    if self.params.variable_update == 'parameter_server':
      if self.job_name:
        if not self.params.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
              self)
        else:
          self.variable_mgr = (
              variable_mgr.VariableMgrDistributedFetchFromStagedPS(self))
      else:
        if not self.params.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
        else:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromStagedPS(
              self)
    elif self.params.variable_update == 'replicated':
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, self.params.all_reduce_spec,
          self.params.agg_small_grads_max_bytes,
          self.params.agg_small_grads_max_group,
          self.params.allreduce_merge_scope)
    elif self.params.variable_update == 'distributed_all_reduce':
      assert self.params.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrDistributedAllReduce(
          self, self.params.all_reduce_spec,
          ('worker' if self.num_workers > 1 else 'localhost'),
          self.num_workers, self.params.agg_small_grads_max_bytes,
          self.params.agg_small_grads_max_group,
          self.params.allreduce_merge_scope)
    elif self.params.variable_update == 'collective_all_reduce':
      assert self.params.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrCollectiveAllReduce(
          self, self.params.all_reduce_spec,
          self.num_workers, self.num_gpus, self.task_index,
          self.params.allreduce_merge_scope)
    elif self.params.variable_update == 'distributed_replicated':
      assert self.params.cross_replica_sync
      if not self.job_name:
        raise ValueError('Invalid variable_update in local mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
    elif self.params.variable_update in ('independent', 'horovod'):
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
    else:
      raise ValueError(
          'Invalid variable_update: %s' % self.params.variable_update)

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    if self.job_name:
      if use_ps_server:
        self.global_step_device = self.param_server_device
      elif self.params.variable_update == 'collective_all_reduce':
        self.global_step_device = self.cpu_device
      else:
        self.global_step_device = '/job:worker/replica:0/task:0/cpu:0'
    else:
      self.global_step_device = self.cpu_device

    self.input_preprocessor = None
    self.eval_input_preprocessor = None
    if not self.dataset.use_synthetic_gpu_inputs():
      if not self.params.eval:
        self.input_preprocessor = self.get_input_preprocessor()
      if self.mode in (constants.BenchmarkMode.EVAL,
                       constants.BenchmarkMode.TRAIN_AND_EVAL):
        with self._do_eval():
          self.eval_input_preprocessor = self.get_input_preprocessor()
    self.datasets_use_prefetch = (
        self.params.datasets_use_prefetch and
        # TODO(rohanj): Figure out why --datasets_use_prefetch freezes on the
        # CPU.
        self.params.device.lower() != 'cpu' and
        self.input_preprocessor and
        self.input_preprocessor.supports_datasets())
    self.init_global_step = 0

    self._config_benchmark_logger()

    if self.mode == constants.BenchmarkMode.TRAIN_AND_EVAL:
      # Remove "eval" from params so it is not accidentally used. Since eval can
      # still occur despite params.eval being False, params.eval should never
      # be used. We cannot yet remove this unconditionally, because the SSD
      # model still uses params.eval, and hence does not work properly with
      # --eval_during_training_*.
      # TODO(b/116627045): We should also remove fields that have an eval
      # equivalent, like num_batches and num_eval_batches.
      self.params = remove_param_fields(self.params, {'eval'})

  @contextlib.contextmanager
  def _do_eval(self):
    """Context manager to switches BenchmarkCNN to eval mode.

    Any evaluation code should be put under this context manager. This context
    manager switches self._doing_eval to True. It also switches certain
    attributes, like self.num_batches and self.num_epochs, to be the number of
    batches and epochs for evaluation respectively

    Yields:
      Nothing.
    """
    # TODO(b/116627045): Find a more general way of switching attributes to the
    # eval equivalents.
    old_doing_eval = self._doing_eval
    old_num_batches = self.num_batches
    old_num_epochs = self.num_epochs
    old_batch_size = self.batch_size
    try:
      self._doing_eval = True
      self.num_batches = self.num_eval_batches
      self.num_epochs = self.num_eval_epochs
      self.batch_size = self.eval_batch_size
      self.model.set_batch_size(self.eval_batch_size // self.num_gpus)
      yield
    finally:
      self._doing_eval = old_doing_eval
      self.num_batches = old_num_batches
      self.num_epochs = old_num_epochs
      self.batch_size = old_batch_size
      self.model.set_batch_size(old_batch_size // self.num_gpus)

  def _config_benchmark_logger(self):
    """Config the model garden benchmark logger."""
    model_benchmark_logger = None
    if self.params.benchmark_log_dir is not None:
      try:
        from tensorflow_models.official.utils.logs import logger as models_logger  # pylint: disable=g-import-not-at-top
      except ImportError:
        tf.logging.fatal('Please include tensorflow/models to the PYTHONPATH '
                         'in order to use BenchmarkLogger. Configured '
                         'benchmark_log_dir: %s'
                         % self.params.benchmark_log_dir)
        raise
      model_benchmark_logger = models_logger.BenchmarkFileLogger(
          self.params.benchmark_log_dir)
    self.benchmark_logger = model_benchmark_logger

  # TODO(laigd): this changes the global device list which is used everywhere,
  # consider refactoring it.
  def reset_devices_for_task(self, task_num, is_local=False):
    """Used to imitate another task when building a distributed graph."""
    worker_prefix = ('/job:localhost' if is_local else
                     '/job:worker/replica:0/task:%s' % task_num)
    self.cpu_device = '%s/cpu:0' % worker_prefix
    self.raw_devices = [
        '%s/%s:%i' % (worker_prefix, self.params.device, i)
        for i in xrange(self.num_gpus)
    ]
    self.devices = self.variable_mgr.get_devices()

  def raw_devices_across_tasks(self, is_local=False):
    """Returns list of raw device names across all tasks."""
    if is_local:
      assert self.num_workers == 1
      return self.raw_devices
    else:
      return [
          'job:worker/replica:0/task%s/%s:%i' % (t, self.params.device, i)
          for t in xrange(self.num_workers)
          for i in xrange(self.num_gpus)
      ]

  def print_info(self):
    """Print basic information."""
    benchmark_info = self._get_params_info()
    log_fn('Model:       %s' % self.model.get_model_name())
    log_fn('Dataset:     %s' % benchmark_info['dataset_name'])
    log_fn('Mode:        %s' % self.mode)
    log_fn('SingleSess:  %s' % benchmark_info['single_session'])
    log_fn('Batch size:  %s global' % (self.batch_size * self.num_workers))
    log_fn('             %s per device' % (self.batch_size //
                                           len(self.raw_devices)))
    if self.batch_group_size > 1:
      log_fn('             %d batches per prepocessing group' %
             self.batch_group_size)
    log_fn('Num batches: %d' % self.num_batches)
    log_fn('Num epochs:  %.2f' % self.num_epochs)
    log_fn('Devices:     %s' % benchmark_info['device_list'])
    log_fn('Data format: %s' % self.params.data_format)
    if self.rewriter_config:
      log_fn('RewriterConfig: %s' % self.rewriter_config)
    log_fn('Optimizer:   %s' % self.params.optimizer)
    log_fn('Variables:   %s' % self.params.variable_update)
    if (self.params.variable_update == 'replicated' or
        self.params.variable_update == 'distributed_all_reduce'
        or self.params.variable_update == 'collective_all_reduce'):
      log_fn('AllReduce:   %s' % self.params.all_reduce_spec)
    if self.job_name:
      log_fn('Sync:        %s' % self.params.cross_replica_sync)
    if self.params.staged_vars:
      log_fn('Staged vars: %s' % self.params.staged_vars)
    if self.params.variable_update == 'horovod' and self.params.horovod_device:
      log_fn('Horovod on:  %s' % self.params.horovod_device)
    log_fn('==========')

  def _get_params_info(self):
    """Get the common parameters info for the benchmark run.

    Returns:
      A dict of processed parameters.
    """
    dataset_name = self.dataset.name
    if self.dataset.use_synthetic_gpu_inputs():
      dataset_name += ' (synthetic)'
    single_session = self.params.variable_update == 'distributed_all_reduce'
    if single_session:
      device_list = self.raw_devices_across_tasks()
    elif self.params.variable_update == 'horovod':
      device_list = ['horovod/%s:%d' % (self.params.device, idx)
                     for idx in range(self.num_workers)]
    else:
      device_list = self.raw_devices
    return {
        'dataset_name': dataset_name,
        'single_session': single_session,
        'device_list': device_list,}

  def _log_benchmark_run(self):
    """Log the benchmark info to the logger.

    The info logged here should be similar to print_info(), but in a structured
    JSON format.
    """
    if self.benchmark_logger:
      benchmark_info = self._get_params_info()

      run_param = {
          'model': self.model.get_model_name(),
          'dataset': benchmark_info['dataset_name'],
          'mode': self.mode,
          'single_sess': benchmark_info['single_session'],
          'devices': benchmark_info['device_list'],
          'batch_size': self.batch_size,
          'batch_size_per_device': self.batch_size // len(self.raw_devices),
          'num_batches': self.num_batches,
          'num_epochs': self.num_epochs,
          'data_format': self.params.data_format,
          'rewrite_config': self.rewriter_config,
          'optimizer': self.params.optimizer,
          'session_config': create_config_proto(self.params),
      }
      # TODO(scottzhu): tf_cnn_benchmark might execute several times with
      # different param setting on the same box. This will cause the run file to
      # only contain the latest info. The benchmark_log_dir should be updated
      # for every new run.
      self.benchmark_logger.log_run_info(
          self.model.get_model_name(), benchmark_info['dataset_name'],
          run_param, test_id=self.params.benchmark_test_id)

  def run(self):
    """Run the benchmark task assigned to this process.

    Returns:
      Dictionary of statistics for training or eval.
    Raises:
       ValueError: unrecognized job name.
    """
    if self.params.job_name == 'ps':
      log_fn('Running parameter server %s' % self.task_index)
      self.cluster_manager.join_server()
      return {}

    # For distributed_all_reduce with multiple workers, drive
    # from a separate controller process.
    if self.params.variable_update == 'distributed_all_reduce':
      if self.params.job_name == 'worker':
        log_fn('Starting worker %s' % self.task_index)
        self.cluster_manager.join_server()
        return
      elif self.params.job_name and self.params.job_name != 'controller':
        raise ValueError('unrecognized job name: %s' % self.params.job_name)

    self._log_benchmark_run()
    if self._doing_eval:
      with tf.Graph().as_default():
        # TODO(laigd): freeze the graph in eval mode.
        return self._run_eval()
    else:
      return self._benchmark_train()

  def _run_eval(self):
    """Evaluate a model every self.params.eval_interval_secs.

    Returns:
      Dictionary containing eval statistics. Currently returns an empty
      dictionary.

    Raises:
      ValueError: If self.params.train_dir is unspecified.
    """
    if self.params.train_dir is None:
      raise ValueError('Trained model directory not specified')
    graph_info = self._build_eval_graph()
    saver = tf.train.Saver(self.variable_mgr.savable_variables())
    summary_writer = tf.summary.FileWriter(self.params.eval_dir,
                                           tf.get_default_graph())
    target = ''
    # TODO(huangyp): Check if checkpoints haven't updated for hours and abort.
    while True:
      with tf.Session(
          target=target, config=create_config_proto(self.params)) as sess:
        image_producer = None
        try:
          global_step = load_checkpoint(saver, sess, self.params.train_dir)
          image_producer = self._initialize_eval_graph(
              graph_info.enqueue_ops, graph_info.input_producer_op,
              graph_info.local_var_init_op_group, sess)
        except CheckpointNotFoundException:
          log_fn('Checkpoint not found in %s' % self.params.train_dir)
        else:  # Only executes if an exception was not thrown
          self._eval_once(sess, summary_writer, graph_info.fetches,
                          graph_info.summary_op, image_producer, global_step)
        if image_producer is not None:
          image_producer.done()
        if self.params.eval_interval_secs <= 0:
          break
        time.sleep(self.params.eval_interval_secs)
    return {}

  def _build_eval_graph(self, scope_name=None):
    """Build the evaluation graph.

    Args:
      scope_name: String to filter what summaries are collected. Only summary
        ops whose name contains `scope_name` will be added, which is useful for
        only including evaluation ops.

    Returns:
      A GraphInfo named_tuple containing various useful ops and tensors of the
      evaluation grpah.
    """
    with self._do_eval():
      input_producer_op, enqueue_ops, fetches = self._build_model()
      local_var_init_op = tf.local_variables_initializer()
      table_init_ops = tf.tables_initializer()
      variable_mgr_init_ops = [local_var_init_op]
      if table_init_ops:
        variable_mgr_init_ops.extend([table_init_ops])
      with tf.control_dependencies([local_var_init_op]):
        variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
      local_var_init_op_group = tf.group(*variable_mgr_init_ops)

      summary_op = tf.summary.merge_all(scope=scope_name)
      # The eval graph has no execution barrier because it doesn't run in
      # distributed mode.
      execution_barrier = None
      # We do not use the global step during evaluation.
      global_step = None
      return GraphInfo(input_producer_op, enqueue_ops, fetches,
                       execution_barrier, global_step, local_var_init_op_group,
                       summary_op)

  # TODO(reedwm): For consistency, we should have a similar
  # "_initialize_train_graph" function. They can likely be the same function.
  def _initialize_eval_graph(self, enqueue_ops, input_producer_op,
                             local_var_init_op_group, sess):
    """Initializes the evaluation graph.

    Args:
      enqueue_ops: Ops that adds the preprocessed images to the staging areas.
      input_producer_op: Op that produce the input batches (before
        preprocessing).
      local_var_init_op_group: Group of ops that perform per-device
        initialization work.
      sess: The session to initialize the eval graph with.

    Returns:
      An ImageProducer, or None if an ImageProducer isn't being used.
    """
    with self._do_eval():
      if local_var_init_op_group is not None:
        # We might reinitialize local variables if they were already initialized
        # during training. This is OK.
        sess.run(local_var_init_op_group)
      if self.dataset.queue_runner_required():
        tf.train.start_queue_runners(sess=sess)
      image_producer = None
      if input_producer_op is not None:
        image_producer = cnn_util.ImageProducer(
            sess, input_producer_op, self.batch_group_size,
            self.params.use_python32_barrier)
        image_producer.start()
      if enqueue_ops:
        for i in xrange(len(enqueue_ops)):
          sess.run(enqueue_ops[:(i + 1)])
          if image_producer is not None:
            image_producer.notify_image_consumption()
      return image_producer

  def _eval_once(self, sess, summary_writer, fetches, summary_op,
                 image_producer, global_step):
    """Evaluate the model using the validation dataset."""
    with self._do_eval():
      mlperf.logger.log_eval_epoch(
          mlperf.tags.EVAL_START, global_step, self.batch_size)
      loop_start_time = start_time = time.time()
      # TODO(laigd): refactor the part to compute/report the accuracy. Currently
      # it only works for image models.
      top_1_accuracy_sum = 0.0
      top_5_accuracy_sum = 0.0
      total_eval_count = self.num_batches * self.batch_size
      for step in xrange(self.num_batches):
        if (summary_writer and self.params.save_summaries_steps > 0 and
            (step + 1) % self.params.save_summaries_steps == 0):
          results, summary_str = sess.run([fetches, summary_op])
          summary_writer.add_summary(summary_str)
        else:
          results = sess.run(fetches)
        # Make global_step available in results for postprocessing.
        results['global_step'] = global_step
        results = self.model.postprocess(results)
        top_1_accuracy_sum += results['top_1_accuracy']
        top_5_accuracy_sum += results['top_5_accuracy']
        if (step + 1) % self.params.display_every == 0:
          duration = time.time() - start_time
          examples_per_sec = (
              self.batch_size * self.params.display_every / duration)
          log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
          start_time = time.time()
        if image_producer is not None:
          image_producer.notify_image_consumption()
      loop_end_time = time.time()
      accuracy_at_1 = top_1_accuracy_sum / self.num_batches
      accuracy_at_5 = top_5_accuracy_sum / self.num_batches
      summary = tf.Summary()
      summary.value.add(tag='eval/Accuracy@1', simple_value=accuracy_at_1)
      summary.value.add(tag='eval/Accuracy@5', simple_value=accuracy_at_5)
      for result_key, result_value in results.items():
        if result_key.startswith(constants.SIMPLE_VALUE_RESULT_PREFIX):
          prefix_len = len(constants.SIMPLE_VALUE_RESULT_PREFIX)
          summary.value.add(tag='eval/' + result_key[prefix_len:],
                            simple_value=result_value)
      if summary_writer:
        summary_writer.add_summary(summary, global_step)
      log_fn('Accuracy @ 1 = %.4f Accuracy @ 5 = %.4f [%d examples]' %
             (accuracy_at_1, accuracy_at_5, total_eval_count))
      elapsed_time = loop_end_time - loop_start_time
      images_per_sec = (self.num_batches * self.batch_size / elapsed_time)
      if self.mode != constants.BenchmarkMode.TRAIN_AND_EVAL:
        # Note that we compute the top 1 accuracy and top 5 accuracy for each
        # batch, which will have a slight performance impact.
        log_fn('-' * 64)
        log_fn('total images/sec: %.2f' % images_per_sec)
        log_fn('-' * 64)
      if self.benchmark_logger:
        eval_result = {
            'eval_top_1_accuracy', accuracy_at_1,
            'eval_top_5_accuracy', accuracy_at_5,
            'eval_average_examples_per_sec', images_per_sec,
            tf.GraphKeys.GLOBAL_STEP, global_step,
        }
        self.benchmark_logger.log_evaluation_result(eval_result)
      mlperf.logger.log_eval_epoch(
          mlperf.tags.EVAL_STOP, global_step, self.batch_size)
      mlperf.logger.log(key=mlperf.tags.EVAL_SIZE,
                        value=self.num_batches * self.batch_size)
      if self.params.model != 'ssd300':  # ssd300 logs eval accuracy elsewhere.
        mlperf.logger.log_eval_accuracy(
            accuracy_at_1, global_step, self.train_batch_size,
            examples_per_epoch=self.dataset.num_examples_per_epoch('train'))
      if self.params.stop_at_top_1_accuracy:
        mlperf.logger.log(key=mlperf.tags.EVAL_TARGET,
                          value=self.params.stop_at_top_1_accuracy)
      return accuracy_at_1, accuracy_at_5

  def _benchmark_train(self):
    """Run cnn in benchmark mode. Skip the backward pass if forward_only is on.

    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    graph = tf.Graph()
    with graph.as_default():
      build_result = self._build_graph()
      if self.mode == constants.BenchmarkMode.TRAIN_AND_EVAL:
        with self.variable_mgr.reuse_variables():
          with tf.name_scope('Evaluation') as ns:
            eval_build_results = self._build_eval_graph(ns)
      else:
        eval_build_results = None
    (graph, result_to_benchmark) = self._preprocess_graph(graph, build_result)
    with graph.as_default():
      return self._benchmark_graph(result_to_benchmark, eval_build_results)

  GPU_CACHED_INPUT_VARIABLE_NAME = 'gpu_cached_inputs'

  def _unfreezable_local_variables(self, graph):
    """Get the local variables that we don't want to freeze."""
    return graph.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES,
        # We don't freeze the gpu_cached_images local variable so it won't get
        # constant folded with ops which process the input.
        scope='.*' + BenchmarkCNN.GPU_CACHED_INPUT_VARIABLE_NAME)

  def _build_graph(self):
    """Build the graph.

    Returns:
      A namedtuple containing the ops/tensors that required by
      _benchmark_graph().
    """
    if self.params.variable_update == 'distributed_all_reduce':
      self.single_session = True
      (input_producer_op, enqueue_ops, fetches) = (
          self._build_model_single_session())
    else:
      self.single_session = False
      (input_producer_op, enqueue_ops, fetches) = self._build_model()
    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list, name='main_fetch_group')
    execution_barrier = None
    if (not self.single_session and self.job_name and
        not self.params.cross_replica_sync):
      execution_barrier = self.add_sync_queues_and_barrier(
          'execution_barrier_', [])

    global_step = tf.train.get_global_step()
    with tf.device(self.global_step_device), tf.name_scope('inc_global_step'):
      with tf.control_dependencies([main_fetch_group]):
        fetches['inc_global_step'] = global_step.assign_add(1)

    if ((not self.single_session) and (not self.distributed_collective) and
        self.job_name and self.params.cross_replica_sync):
      # Block all replicas until all replicas are ready for next step.
      fetches['sync_queues'] = self.add_sync_queues_and_barrier(
          'sync_queues_step_end_', [main_fetch_group])

    # Skips the init ops for freezable local variables in forward_only mode so
    # we can remove all the assign ops when converting variables to constants.
    with tf.name_scope('local_variable_initialization'):
      if self.forward_only_and_freeze:
        local_var_init_op = tf.variables_initializer(
            self._unfreezable_local_variables(tf.get_default_graph()))
      else:
        local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()

    variable_manager_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_manager_init_ops.extend([table_init_ops])
    if not self.forward_only_and_freeze:
      with tf.control_dependencies([local_var_init_op]):
        variable_manager_init_ops.extend(self.variable_mgr.get_post_init_ops())
    if ((not self.single_session) and (not self.distributed_collective) and
        self.job_name and self.params.cross_replica_sync):
      # Ensure all workers execute variable_manager_init_ops before they start
      # executing the model.
      variable_manager_init_ops.append(
          self.add_sync_queues_and_barrier('init_ops_end_',
                                           variable_manager_init_ops))
    local_var_init_op_group = tf.group(*variable_manager_init_ops,
                                       name='local_var_init_op_group')
    summary_op = tf.summary.merge_all()

    return GraphInfo(
        input_producer_op=input_producer_op,
        enqueue_ops=enqueue_ops,
        fetches=fetches,
        execution_barrier=execution_barrier,
        global_step=global_step,
        local_var_init_op_group=local_var_init_op_group,
        summary_op=summary_op)

  def _benchmark_graph(self, graph_info, eval_graph_info):
    """Benchmark the training graph.

    Args:
      graph_info: the namedtuple returned by _build_graph() which
        contains all necessary information to benchmark the graph, including
        named tensors/ops list, fetches, etc.
      eval_graph_info: Similar to graph_info but for the eval graph if
        --eval_during_training_* is used. Otherwise, None.
    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    log_fn('Initializing graph')
    if self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      # First worker will be 'chief' - it will write summaries and
      # save checkpoints.
      is_chief = hvd.rank() == 0
    else:
      is_chief = (not self.job_name or self.task_index == 0)

    summary_writer = None
    if (is_chief and self.params.summary_verbosity and self.params.train_dir and
        self.params.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(self.params.train_dir,
                                             tf.get_default_graph())

    # We want to start the benchmark timer right after a image_producer barrier
    # and avoids undesired waiting times on barriers.
    if ((self.num_warmup_batches + len(graph_info.enqueue_ops) - 1) %
        self.batch_group_size) != 0:
      self.num_warmup_batches = int(
          math.ceil(
              (self.num_warmup_batches + len(graph_info.enqueue_ops) - 1.0) /
              (self.batch_group_size)) * self.batch_group_size -
          len(graph_info.enqueue_ops) + 1)
      log_fn('Round up warm up steps to %d to match batch_group_size' %
             self.num_warmup_batches)
      assert ((self.num_warmup_batches + len(graph_info.enqueue_ops) - 1) %
              self.batch_group_size) == 0
    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    if is_chief and not self.forward_only_and_freeze:
      saver = tf.train.Saver(
          self.variable_mgr.savable_variables(),
          save_relative_paths=True,
          max_to_keep=self.params.max_ckpts_to_keep)
    else:
      saver = None
    ready_for_local_init_op = None
    if self.job_name and not (self.single_session or
                              self.distributed_collective):
      # In distributed mode, we don't want to run local_var_init_op_group until
      # the global variables are initialized, because local_var_init_op_group
      # may use global variables (such as in distributed replicated mode). We
      # don't set this in non-distributed mode, because in non-distributed mode,
      # local_var_init_op_group may itself initialize global variables (such as
      # in replicated mode).
      ready_for_local_init_op = tf.report_uninitialized_variables(
          tf.global_variables())
    if self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      bcast_global_variables_op = hvd.broadcast_global_variables(0)
    else:
      bcast_global_variables_op = None

    if self.params.variable_update == 'collective_all_reduce':
      # It doesn't matter what this collective_graph_key value is,
      # so long as it's > 0 and the same at every worker.
      init_run_options = tf.RunOptions()
      init_run_options.experimental.collective_graph_key = 6
    else:
      init_run_options = tf.RunOptions()
    local_var_init_ops = [graph_info.local_var_init_op_group]
    if eval_graph_info:
      # `eval_graph_info.local_var_init_op_group` also includes some of the
      # training initializer ops, since it's difficult to filter them out.
      # Rerunning the training initializer ops is OK, but we add a control
      # dependency since running two sets of training initializer ops at the
      # same time can cause race conditions.
      with tf.control_dependencies(local_var_init_ops):
        local_var_init_ops.append(eval_graph_info.local_var_init_op_group)
    sv = tf.train.Supervisor(
        # For the purpose of Supervisor, all Horovod workers are 'chiefs',
        # since we want session to be initialized symmetrically on all the
        # workers.
        is_chief=is_chief or (self.params.variable_update == 'horovod'
                              or self.distributed_collective),
        # Log dir should be unset on non-chief workers to prevent Horovod
        # workers from corrupting each other's checkpoints.
        logdir=self.params.train_dir if is_chief else None,
        ready_for_local_init_op=ready_for_local_init_op,
        local_init_op=local_var_init_ops,
        saver=saver,
        global_step=graph_info.global_step,
        summary_op=None,
        save_model_secs=self.params.save_model_secs,
        summary_writer=summary_writer,
        local_init_run_options=init_run_options)

    profiler = tf.profiler.Profiler() if self.params.tfprof_file else None
    if self.graph_file is not None:
      path, filename = os.path.split(self.graph_file)
      as_text = filename.endswith('txt')
      log_fn('Writing GraphDef as %s to %s' % (  # pyformat break
          'text' if as_text else 'binary', self.graph_file))
      tf.train.write_graph(tf.get_default_graph().as_graph_def(add_shapes=True),
                           path, filename, as_text)

    start_standard_services = (
        self.params.train_dir or
        self.dataset.queue_runner_required())
    target = self.cluster_manager.get_target() if self.cluster_manager else ''
    with sv.managed_session(
        master=target,
        config=create_config_proto(self.params),
        start_standard_services=start_standard_services) as sess:
      # Anything that can potentially raise an OutOfRangeError with 'sess' MUST
      # be under this try block. The managed_session() context manager silently
      # ignores OutOfRangeError, so we must catch them and wrap them with
      # a different exception type so that they can be propagated up to the
      # caller.
      try:
        stats = self.benchmark_with_session(
            sess, sv, graph_info, eval_graph_info, bcast_global_variables_op,
            is_chief, summary_writer, profiler)
      except tf.errors.OutOfRangeError:
        raise RuntimeError(
            'Received OutOfRangeError. Wrapping in Runtime error to avoid '
            'Supervisor from suppressing the error. Original OutOfRangeError '
            'with traceback:\n' + traceback.format_exc())

    sv.stop()
    if profiler:
      generate_tfprof_profile(profiler, self.params.tfprof_file)
    return stats

  def benchmark_with_session(self, sess, supervisor, graph_info,
                             eval_graph_info, bcast_global_variables_op,
                             is_chief, summary_writer, profiler):
    """Benchmarks the graph with the given session.

    Args:
      sess: The session to benchmark the graph with
      supervisor: The Supervisor that created the session.
      graph_info: the namedtuple returned by _build_graph() which
        contains all necessary information to benchmark the graph, including
        named tensors/ops list, fetches, etc.
      eval_graph_info: Similar to graph_info but for the eval graph if
        --eval_during_training_every_n_steps is used. Otherwise, None.
      bcast_global_variables_op: If Horovod is used, the op to broadcast the
        global variables to all the processes. None if Horovod is not used.
      is_chief: True if this is the chief process.
      summary_writer: The SummaryWriter used to write summaries, or None if
        summaries are not used.
      profiler: The tf.profiler.Profiler, or None if tfprof is not used.

    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    if self.params.backbone_model_path is not None:
      self.model.load_backbone_model(sess, self.params.backbone_model_path)
    if bcast_global_variables_op:
      sess.run(bcast_global_variables_op)
    image_producer = None
    if graph_info.input_producer_op is not None:
      image_producer = cnn_util.ImageProducer(
          sess, graph_info.input_producer_op, self.batch_group_size,
          self.params.use_python32_barrier)
      image_producer.start()
    if graph_info.enqueue_ops:
      for i in xrange(len(graph_info.enqueue_ops)):
        sess.run(graph_info.enqueue_ops[:(i + 1)])
        if image_producer is not None:
          image_producer.notify_image_consumption()
    self.init_global_step, = sess.run([graph_info.global_step])
    if self.job_name and not self.params.cross_replica_sync:
      # TODO(zhengxq): Do we need to use a global step watcher at all?
      global_step_watcher = GlobalStepWatcher(
          sess, graph_info.global_step,
          self.num_workers * self.num_warmup_batches +
          self.init_global_step,
          self.num_workers * (self.num_warmup_batches + self.num_batches) - 1)
      global_step_watcher.start()
    else:
      global_step_watcher = None
    eval_image_producer = None
    if eval_graph_info:
      # We pass local_var_init_op_group=None because the Supervisor already
      # initialized local variables above. We need to have the Supervisor
      # initialize the local variables, because otherwise it throws an error
      # complaining that not all variables were initialized.
      eval_image_producer = self._initialize_eval_graph(
          eval_graph_info.enqueue_ops, eval_graph_info.input_producer_op,
          local_var_init_op_group=None, sess=sess)
    step_train_times = []
    log_fn('Running warm up')
    local_step = -1 * self.num_warmup_batches
    if self.single_session:
      # In single session mode, each step, the global_step is incremented by
      # 1. In non-single session mode, each step, the global_step is
      # incremented once per worker. This means we need to divide
      # init_global_step by num_workers only in non-single session mode.
      end_local_step = self.num_batches - self.init_global_step
    else:
      end_local_step = self.num_batches - (self.init_global_step //
                                           self.num_workers)
    if not global_step_watcher:
      # In cross-replica sync mode, all workers must run the same number of
      # local steps, or else the workers running the extra step will block.
      done_fn = lambda: local_step >= end_local_step
    else:
      done_fn = global_step_watcher.done
    if self.params.debugger is not None:
      if self.params.debugger == 'cli':
        log_fn('The CLI TensorFlow debugger will be used.')
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      else:
        log_fn('The TensorBoard debugger plugin will be used.')
        sess = tf_debug.TensorBoardDebugWrapperSession(sess,
                                                       self.params.debugger)
    mlperf.logger.log(key=mlperf.tags.TRAIN_LOOP)
    skip_final_eval = False
    accuracy_at_1 = None
    last_eval_step = local_step
    loop_start_time = time.time()
    last_average_loss = None
    while not done_fn():
      if local_step == 0:
        log_fn('Done warm up')
        if graph_info.execution_barrier:
          log_fn('Waiting for other replicas to finish warm up')
          sess.run([graph_info.execution_barrier])

        # TODO(laigd): rename 'Img' to maybe 'Input'.
        header_str = ('Step\tImg/sec\t' +
                      self.params.loss_type_to_report.replace('/', ' '))
        if self.params.print_training_accuracy or self.params.forward_only:
          # TODO(laigd): use the actual accuracy op names of the model.
          header_str += '\ttop_1_accuracy\ttop_5_accuracy'
        log_fn(header_str)
        assert len(step_train_times) == self.num_warmup_batches
        # reset times to ignore warm up batch
        step_train_times = []
        loop_start_time = time.time()
      if (summary_writer and
          (local_step + 1) % self.params.save_summaries_steps == 0):
        fetch_summary = graph_info.summary_op
      else:
        fetch_summary = None
      collective_graph_key = 7 if (
          self.params.variable_update == 'collective_all_reduce') else 0
      (summary_str, last_average_loss) = benchmark_one_step(
          sess, graph_info.fetches, local_step,
          self.batch_size * (self.num_workers
                             if self.single_session else 1), step_train_times,
          self.trace_filename, self.params.partitioned_graph_file_prefix,
          profiler, image_producer, self.params, fetch_summary,
          benchmark_logger=self.benchmark_logger,
          collective_graph_key=collective_graph_key)
      if summary_str is not None and is_chief:
        supervisor.summary_computed(sess, summary_str)
      local_step += 1
      if (self.params.save_model_steps and
          local_step % self.params.save_model_steps == 0 and
          local_step > 0 and
          is_chief):
        supervisor.saver.save(sess, supervisor.save_path,
                              supervisor.global_step)
      if (eval_graph_info and local_step > 0 and not done_fn() and
          self._should_eval_during_training(local_step)):
        python_global_step = sess.run(graph_info.global_step)
        num_steps_since_last_eval = local_step - last_eval_step
        # The INPUT_SIZE tag value might not match the
        # PREPROC_NUM_TRAIN_EXAMPLES tag value, because the number of examples
        # run, which is INPUT_SIZE, is rounded up to the nearest multiple of
        # self.batch_size.
        mlperf.logger.log(
            key=mlperf.tags.INPUT_SIZE,
            value=num_steps_since_last_eval * self.batch_size)
        log_fn('Running evaluation at global_step {}'.format(
            python_global_step))
        accuracy_at_1, _ = self._eval_once(
            sess, summary_writer, eval_graph_info.fetches,
            eval_graph_info.summary_op, eval_image_producer,
            python_global_step)
        last_eval_step = local_step
        if (self.params.stop_at_top_1_accuracy and
            accuracy_at_1 >= self.params.stop_at_top_1_accuracy):
          log_fn('Stopping, as eval accuracy at least %s was reached' %
                 self.params.stop_at_top_1_accuracy)
          skip_final_eval = True
          break
        else:
          log_fn('Resuming training')
      if eval_graph_info and self.model.reached_target():
        log_fn('Stopping, as the model indicates its custom goal was reached')
        skip_final_eval = True
        break
    loop_end_time = time.time()
    # Waits for the global step to be done, regardless of done_fn.
    if global_step_watcher:
      while not global_step_watcher.done():
        time.sleep(.25)
    if not global_step_watcher:
      elapsed_time = loop_end_time - loop_start_time
      average_wall_time = elapsed_time / local_step if local_step > 0 else 0
      images_per_sec = (self.num_workers * local_step * self.batch_size /
                        elapsed_time)
      num_steps = local_step * self.num_workers
    else:
      # NOTE: Each worker independently increases the global step. So,
      # num_steps will be the sum of the local_steps from each worker.
      num_steps = global_step_watcher.num_steps()
      elapsed_time = global_step_watcher.elapsed_time()
      average_wall_time = (elapsed_time * self.num_workers / num_steps
                           if num_steps > 0 else 0)
      images_per_sec = num_steps * self.batch_size / elapsed_time

    # We skip printing images/sec if --eval_during_training_* is specified,
    # because we are both processing training and evaluation images, so a
    # singular "images/sec" value is meaningless.
    if self.mode != constants.BenchmarkMode.TRAIN_AND_EVAL:
      log_fn('-' * 64)
      # TODO(laigd): rename 'images' to maybe 'inputs'.
      log_fn('total images/sec: %.2f' % images_per_sec)
      log_fn('-' * 64)
    else:
      log_fn('Done with training')
    num_steps_since_last_eval = local_step - last_eval_step
    mlperf.logger.log(
        key=mlperf.tags.INPUT_SIZE,
        value=num_steps_since_last_eval * self.batch_size)
    python_global_step = sess.run(graph_info.global_step)
    if eval_graph_info and not skip_final_eval:
      log_fn('Running final evaluation at global_step {}'.format(
          python_global_step))
      accuracy_at_1, _ = self._eval_once(
          sess, summary_writer, eval_graph_info.fetches,
          eval_graph_info.summary_op, eval_image_producer, python_global_step)
    num_epochs_ran = (python_global_step * self.batch_size /
                      self.dataset.num_examples_per_epoch('train'))
    mlperf.logger.log_train_epochs(num_epochs_ran)
    if image_producer is not None:
      image_producer.done()
    if eval_image_producer is not None:
      eval_image_producer.done()
    if is_chief:
      if self.benchmark_logger:
        self.benchmark_logger.log_metric(
            'average_examples_per_sec', images_per_sec, global_step=num_steps)

    # Save the model checkpoint.
    if self.params.train_dir is not None and is_chief:
      checkpoint_path = os.path.join(self.params.train_dir, 'model.ckpt')
      if not gfile.Exists(self.params.train_dir):
        gfile.MakeDirs(self.params.train_dir)
      supervisor.saver.save(sess, checkpoint_path, graph_info.global_step)
    if graph_info.execution_barrier:
      # Wait for other workers to reach the end, so this worker doesn't
      # go away underneath them.
      sess.run([graph_info.execution_barrier])
    stats = {
        'num_workers': self.num_workers,
        'num_steps': num_steps,
        'average_wall_time': average_wall_time,
        'images_per_sec': images_per_sec
    }
    if last_average_loss is not None:
      stats['last_average_loss'] = last_average_loss
    success = bool(self.model.reached_target() or
                   (accuracy_at_1 and self.params.stop_at_top_1_accuracy and
                    accuracy_at_1 >= self.params.stop_at_top_1_accuracy))
    mlperf.logger.log(key=mlperf.tags.RUN_STOP, value={'success': success})
    mlperf.logger.log(key=mlperf.tags.RUN_FINAL)
    return stats

  def _should_eval_during_training(self, step):
    """Return True iff should run eval during training at current step."""

    assert self.mode == constants.BenchmarkMode.TRAIN_AND_EVAL

    if self.params.eval_during_training_every_n_steps:
      return step % self.params.eval_during_training_every_n_steps == 0

    # All other --eval_during_training_* flags are converted to step numbers
    # at which the model should run evaluation during training.
    return step in self.eval_during_training_at_specified_steps

  def _preprocess_graph(self, graph, graph_info):
    """Preprocess the graph before executing.

    Depending on the params, it runs various preprocessing on the graph,
    including freezing, TensorRT conversion, etc.

    Args:
      graph: the graph to preprocess.
      graph_info: the namedtuple returned by _build_graph() which
        contains all necessary information to benchmark the graph, including
        named tensors/ops list, fetches, etc.

    Returns:
      The updated graph and graph_info with the ops/tensors/fetches updated
      according to the imported graph.
    """
    assert isinstance(graph_info.fetches, dict)
    assert isinstance(graph_info.global_step, tf.Variable)
    if not self.forward_only_and_freeze:
      return (graph, graph_info)

    # Get the names of the ops that need to keep during conversion.
    flattened_op_names = list(
        set([
            v.name.split(':')[0]
            for v in nest.flatten(graph_info)
            if v is not None
        ]))
    # Get variables that we don't want to freeze.
    # Only keep unfreezable variables in forward_only_and_freeze mode.
    # TODO(laigd): consider making global_step a constant.
    variables_to_keep = {graph_info.global_step: tf.GraphKeys.GLOBAL_VARIABLES}
    variables_to_keep.update({
        local_variable: tf.GraphKeys.LOCAL_VARIABLES
        for local_variable in self._unfreezable_local_variables(graph)
    })

    variable_initializers = [
        variable.initializer.name for variable in variables_to_keep]
    output_node_names = (
        flattened_op_names +
        # Add variable initializer and read ops to the output list, so
        # convert_variables_to_constants() will keep them.
        variable_initializers +
        [variable.value().op.name for variable in variables_to_keep])
    graphdef = graph.as_graph_def(add_shapes=True)

    # Freeze the graph.
    with graph.as_default():
      with tf.Session(config=create_config_proto(self.params)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        graphdef = graph_util.convert_variables_to_constants(
            sess,
            graphdef,
            output_node_names,
            variable_names_blacklist=[
                variable.op.name for variable in variables_to_keep
            ])

    # Run TensorRT conversion.
    if self.params.trt_mode:
      # Import here instead of at top, because this will crash if TensorRT is
      # not installed
      from tensorflow.contrib import tensorrt as trt  # pylint: disable=g-import-not-at-top
      # Avoid TF-TRT bridge from touching all variable initializer ops and their
      # dependencies, since they can directly be fetched by sess.run()s that
      # initialize the variables.
      # pylint: disable=protected-access
      name_to_input_name, _, _ = graph_util_impl._extract_graph_summary(
          graphdef)
      initializer_subgraph_ops = graph_util_impl._bfs_for_reachable_nodes(
          variable_initializers, name_to_input_name)
      # pylint: enable=protected-access

      graphdef = trt.create_inference_graph(
          graphdef,
          outputs=output_node_names + list(initializer_subgraph_ops),
          max_batch_size=self.model.get_batch_size(),
          max_workspace_size_bytes=self.params.trt_max_workspace_size_bytes,
          precision_mode=self.params.trt_mode)

    # Creates a new graph as the default and import the converted graph back.
    updated_graph = tf.Graph()

    def _get_tensors_or_ops(inputs):
      """Gets the updated tensors or ops from 'updated_graph'."""

      def _get_fn(element):
        if element is None:
          return None
        if ':' in element.name:
          return updated_graph.get_tensor_by_name(element.name)
        return updated_graph.get_operation_by_name(element.name)

      if isinstance(inputs, (list, dict, tuple)):
        return nest.map_structure(_get_fn, inputs)
      else:
        return _get_fn(inputs)

    with updated_graph.as_default():
      importer.import_graph_def(graph_def=graphdef, name='')

      # Update the variables
      for variable in variables_to_keep:
        updated_variable = tf.Variable.from_proto(variable.to_proto())
        tf.add_to_collection(variables_to_keep[variable], updated_variable)
        if variable is graph_info.global_step:
          updated_global_step = updated_variable

    updated_graph_info = GraphInfo(
        input_producer_op=_get_tensors_or_ops(graph_info.input_producer_op),
        enqueue_ops=_get_tensors_or_ops(graph_info.enqueue_ops),
        execution_barrier=_get_tensors_or_ops(graph_info.execution_barrier),
        local_var_init_op_group=_get_tensors_or_ops(
            graph_info.local_var_init_op_group),
        fetches=_get_tensors_or_ops(graph_info.fetches),
        global_step=updated_global_step,
        summary_op=None)
    return (updated_graph, updated_graph_info)

  def _build_input_processing(self, shift_ratio=0):
    """"Build the image (pre)processing portion of the model graph.

    Args:
      shift_ratio: shift_ratio for data_flow_ops.RecordInput.

    Returns:
      An InputProcessingInfo containing all the input sources to the model.
    """
    input_processing_info = InputProcessingInfo(
        input_producer_op=None,
        input_producer_stages=None,
        function_buffering_resources=None,
        multi_device_iterator_input=None)

    mlperf.logger.log(key=mlperf.tags.INPUT_ORDER)
    if not self._doing_eval:
      mlperf.logger.log(key=mlperf.tags.INPUT_BATCH_SIZE, value=self.batch_size)

    # If using synthetic gpu inputs, do nothing on the cpu side.
    if self.dataset.use_synthetic_gpu_inputs():
      assert not self.datasets_use_prefetch
      return input_processing_info

    if self._doing_eval:
      input_preprocessor = self.eval_input_preprocessor
      mlperf.logger.log(key=mlperf.tags.PREPROC_NUM_EVAL_EXAMPLES,
                        value=self.dataset.num_examples_per_epoch('validation'))
    else:
      input_preprocessor = self.input_preprocessor
      mlperf.logger.log(key=mlperf.tags.PREPROC_NUM_TRAIN_EXAMPLES,
                        value=self.dataset.num_examples_per_epoch('train'))

    # Use prefetching mechanism provided by dataset input pipeline.
    if self.datasets_use_prefetch:
      if self.params.use_multi_device_iterator:
        multi_device_iterator = (
            input_preprocessor.build_multi_device_iterator(
                self.batch_size, len(self.devices), self.cpu_device,
                self.params, self.raw_devices, self.dataset, self._doing_eval))
        return input_processing_info._replace(
            multi_device_iterator_input=multi_device_iterator.get_next())

      subset = 'validation' if self._doing_eval else 'train'
      function_buffering_resources = (
          input_preprocessor.build_prefetch_input_processing(
              self.batch_size, self.model.get_input_shapes(subset),
              len(self.devices), self.cpu_device, self.params, self.devices,
              self.model.get_input_data_types(subset), self.dataset,
              self._doing_eval))
      return input_processing_info._replace(
          function_buffering_resources=function_buffering_resources)

    # Not using dataset prefetching. Use a staging area to mimic the prefetching
    # behavior instead.
    with tf.device(self.cpu_device):
      if self._doing_eval:
        subset = 'validation'
      else:
        subset = 'train'
      input_list = input_preprocessor.minibatch(
          self.dataset,
          subset=subset,
          params=self.params,
          shift_ratio=shift_ratio)

      input_producer_op = []
      input_producer_stages = []
      for device_num in range(len(self.devices)):
        staging_area = data_flow_ops.StagingArea(
            [parts[0].dtype for parts in input_list],
            shapes=[parts[0].get_shape() for parts in input_list],
            shared_name='input_producer_staging_area_%d_eval_%s' %
            (device_num, self._doing_eval))
        input_producer_stages.append(staging_area)
        for group_index in xrange(self.batch_group_size):
          batch_index = group_index + device_num * self.batch_group_size
          put_op = staging_area.put(
              [parts[batch_index] for parts in input_list])
          input_producer_op.append(put_op)
      assert input_producer_op

    return input_processing_info._replace(
        input_producer_op=input_producer_op,
        input_producer_stages=input_producer_stages)

  def _maybe_initialize_fp16(self):
    """Initialize fp16 settings."""
    if self.params.use_fp16 and not self._doing_eval:
      init_loss_scale_val = float(self.params.fp16_loss_scale or
                                  self.model.get_fp16_loss_scale())
      self.loss_scale = None
      self.loss_scale_normal_steps = None
      if self.enable_auto_loss_scale or init_loss_scale_val != 1:
        self.loss_scale = tf.get_variable(
            name='loss_scale',
            initializer=init_loss_scale_val,
            dtype=tf.float32,
            trainable=False)
      if self.enable_auto_loss_scale:
        self.loss_scale_normal_steps = tf.get_variable(
            name='loss_scale_normal_steps', initializer=0, trainable=False)

  def _build_model(self):
    """Build the TensorFlow graph."""
    if self.datasets_use_prefetch:
      assert not self.params.staged_vars
      assert not self.variable_mgr.supports_staged_vars()

    # Adjust seed so different workers start read different input files.
    if self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      seed_adjustment = hvd.rank()
    else:
      seed_adjustment = 0
    mlperf.logger.log(key=mlperf.tags.RUN_SET_RANDOM_SEED,
                      value=self.params.tf_random_seed + seed_adjustment)
    tf.set_random_seed(self.params.tf_random_seed + seed_adjustment)
    mlperf.logger.log(key=mlperf.tags.RUN_SET_RANDOM_SEED,
                      value=4321 + seed_adjustment)
    np.random.seed(4321 + seed_adjustment)
    phase_train = not (self._doing_eval or self.params.forward_only)

    if self._doing_eval:
      mode_string = 'evaluation'
    else:
      mode_string = 'training'

    log_fn('Generating {} model'.format(mode_string))
    losses = []
    device_grads = []
    all_logits = []
    all_accuracy_ops = {}
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()
      self._maybe_initialize_fp16()

    # Build the processing and model for the worker.
    input_producer_op = None
    with tf.name_scope('input_processing'):
      input_processing_info = self._build_input_processing(shift_ratio=0)
      if input_processing_info.input_producer_op is not None:
        input_producer_op = tf.group(*input_processing_info.input_producer_op)
    update_ops = None
    staging_delta_ops = []

    for device_num in range(len(self.devices)):
      with tf.name_scope('tower_%i' % device_num) as name_scope, (
          self.variable_mgr.create_outer_variable_scope(device_num)):
        results = self.add_forward_pass_and_gradients(
            phase_train, device_num, device_num, input_processing_info,
            gpu_compute_stage_ops, gpu_grad_stage_ops)

        if self.params.backbone_model_path:
          self.model.add_backbone_saver()

        if phase_train:
          losses.append(results['loss'])
          device_grads.append(results['gradvars'])
        else:
          all_logits.append(results['logits'])
        if not phase_train or self.params.print_training_accuracy:
          for name, op in results.items():
            if name.startswith('accuracy:'):
              key = name[9:]
              if key not in all_accuracy_ops:
                all_accuracy_ops[key] = []
              all_accuracy_ops[key].append(op)

        if device_num == 0:
          # Retain the Batch Normalization updates operations only from the
          # first tower. These operations update the moving mean and moving
          # variance variables, which are updated (but not used) during
          # training, and used during evaluation. The moving mean and variance
          # approximate the true mean and variance across all images in the
          # dataset. Therefore, in replicated mode, these moving averages would
          # be almost identical for each tower, and so we only update and save
          # the moving averages for one tower. In parameter server mode, all
          # towers share a copy of the variables so we also only need to update
          # and save the moving averages once.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          if self.datasets_use_prefetch:
            assert not self.variable_mgr.staging_delta_ops
          else:
            staging_delta_ops = list(self.variable_mgr.staging_delta_ops)

    enqueue_ops = []
    if not self.datasets_use_prefetch:
      if self.variable_mgr.supports_staged_vars():
        for staging_ops in self.variable_mgr.staging_vars_on_devices:
          gpu_compute_stage_ops.extend(
              [put_op for _, (put_op, _) in six.iteritems(staging_ops)])
      enqueue_ops.append(tf.group(*gpu_compute_stage_ops,
                                  name='gpu_compute_stage_ops_group'))
      if gpu_grad_stage_ops:
        staging_delta_ops += gpu_grad_stage_ops
      if staging_delta_ops:
        enqueue_ops.append(tf.group(*(staging_delta_ops)))

    if (self.mode == constants.BenchmarkMode.TRAIN_AND_EVAL and
        self.params.variable_update == 'replicated'):
      # We need to get all the update ops instead of only those for the first
      # tower. This is because during evaluation, each tower will read from its
      # own tower's moving averages instead of the first tower's moving
      # averages.
      # TODO(reedwm): Have each tower read from the first tower's moving
      # averages for a slight performance gain.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      mlperf.logger.log(key=mlperf.tags.INPUT_BN_SPAN,
                        value=self.batch_size // len(self.raw_devices))

    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  enqueue_ops, update_ops, all_accuracy_ops,
                                  phase_train)
    return (input_producer_op, enqueue_ops, fetches)

  def _build_fetches(self, global_step, all_logits, losses, device_grads,
                     enqueue_ops, update_ops, all_accuracy_ops, phase_train):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {}
    if enqueue_ops:
      fetches['enqueue_ops'] = enqueue_ops
    for name, ops in all_accuracy_ops.items():
      # For fetches that starts with 'tensor:', keep dimension and skip reducing
      # them to scalars.
      if name.startswith(constants.UNREDUCED_ACCURACY_OP_PREFIX):
        key = name[len(constants.UNREDUCED_ACCURACY_OP_PREFIX):]
        fetches[key] = tf.concat(ops, 0)
      else:
        fetches[name] = tf.reduce_sum(ops) / self.batch_size
        if self.task_index == 0 and self.params.summary_verbosity >= 1:
          tf.summary.scalar(name, fetches[name])

    if not phase_train:
      if self.params.forward_only:
        fetches['all_logits'] = tf.concat(all_logits, 0)
      return fetches
    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    # TODO(reedwm): Greatly simplify the learning rate code.
    if (self.params.variable_update == 'horovod' or
        self.params.variable_update == 'collective_all_reduce'):
      # Each worker independently increments global_step.
      examples_per_step = self.batch_size * self.num_workers
    else:
      # global_step is shared by all workers, and so every iteration
      # global_step is incremented by num_workers.
      examples_per_step = self.batch_size
    if self.params.compute_lr_on_cpu:
      with tf.device(self.cpu_device):
        learning_rate = get_learning_rate(self.params, global_step,
                                          self.dataset.num_examples_per_epoch(),
                                          self.model, examples_per_step)

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        with tf.name_scope('average_loss'):
          average_loss = tf.reduce_mean(losses)
        with tf.name_scope('get_gradients_to_apply'):
          avg_grads = self.variable_mgr.get_gradients_to_apply(d,
                                                               gradient_state)

        if not self.params.compute_lr_on_cpu:
          # We compute the learning rate once for each device in
          # `apply_gradient_devices`.
          learning_rate = get_learning_rate(
              self.params, global_step, self.dataset.num_examples_per_epoch(),
              self.model, examples_per_step)
        gradient_clip = self.params.gradient_clip
        if gradient_clip is not None:
          with tf.name_scope('clip_gradients'):
            clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                                               +gradient_clip), var)
                             for grad, var in avg_grads]
        else:
          clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate_tensor')
        opt = get_optimizer(self.params, learning_rate)
        loss_scale_params = variable_mgr_util.AutoLossScaleParams(
            enable_auto_loss_scale=self.enable_auto_loss_scale,
            loss_scale=self.loss_scale,
            loss_scale_normal_steps=self.loss_scale_normal_steps,
            inc_loss_scale_every_n=self.params.fp16_inc_loss_scale_every_n,
            is_chief=not self.job_name or self.task_index == 0)

        with tf.name_scope('append_apply_gradient_ops'):
          self.variable_mgr.append_apply_gradients_ops(
              gradient_state, opt, clipped_grads, training_ops,
              loss_scale_params)
    train_op = tf.group(*(training_ops + update_ops), name='train_ops_group')

    with tf.device(self.cpu_device):
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar(self.params.loss_type_to_report, average_loss)
        if self.loss_scale is not None:
          tf.summary.scalar('loss_scale', self.loss_scale)
        if self.loss_scale_normal_steps:
          tf.summary.scalar('loss_scale_normal_steps',
                            self.loss_scale_normal_steps)

        if self.params.summary_verbosity >= 2:
          self.gradient_histogram_summary(avg_grads)

        if self.params.summary_verbosity >= 3:
          for grad, var in avg_grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)
          for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        if self.params.use_relu_x and not self.params.relu_x_per_channel:
          with tf.name_scope('relu_var_values'):
            for var in tf.global_variables():
              if ('relu_x/x' in var.name
                  and ('v0/cg' in var.name or 'v/cg' in var.name)
                  and 'Momentum' not in var.name):
                tf.summary.scalar(var.name, var)

        weight_equalization = self.params.weight_equalization
        if weight_equalization is not None and weight_equalization != 0:
          with tf.name_scope('weight_distribution'):
            for var in tf.global_variables():
              if ('kernel' in var.name
                  and ('v0/cg' in var.name or 'v/cg' in var.name)
                  and 'Momentum' not in var.name):
                tf.summary.histogram(var.name, var)

    fetches['train_op'] = train_op
    fetches['average_loss'] = average_loss
    return fetches

  def gradient_histogram_summary(self, avg_grads):
    """Create histogram of log values of all non-zero gradients."""
    with tf.name_scope('log_gradients_summary'):
      all_grads = []
      for grad, _ in avg_grads:
        all_grads.append(tf.reshape(grad, [-1]))
      grads = tf.abs(tf.concat(all_grads, 0))
      # exclude grads with zero values.
      indices_for_non_zero_grads = tf.where(tf.not_equal(grads, 0))
      log_grads = tf.reshape(
          tf.log(tf.gather(grads, indices_for_non_zero_grads)), [-1])
      tf.summary.histogram('log_gradients', log_grads)

  def _build_model_single_session(self):
    """Build the TensorFlow graph for multiple replicas in a single_session.

    Returns:
      input_producer_op:
      enqueue_ops:
      fetches:

    Raises:
       ValueError: optimizer not recognized.

    Single session runs multiple model replicas as part of one large
    distributed graph, whose global execution is always step-synchronized.
    """
    # verify assumptions
    assert self.params.task_index == 0
    assert not self._doing_eval
    assert not self.params.forward_only
    assert not self.params.staged_vars

    tf.set_random_seed(self.params.tf_random_seed)
    np.random.seed(4321)
    phase_train = True

    log_fn('Generating training model')
    losses = []
    device_grads = []
    all_logits = []
    all_accuracy_ops = {}
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()

    update_ops = []
    global_input_producer_op = []

    is_local = not self.job_name
    if is_local:
      assert self.num_workers == 1
    for task_num in range(self.num_workers):
      # Reset the devices that self.variable_mgr knows about to those
      # belonging to the next worker (task).
      self.reset_devices_for_task(task_num, is_local)
      # Build the per-worker image processing
      with tf.name_scope('input_processing'):
        input_processing_info = self._build_input_processing(
            shift_ratio=(task_num / self.num_workers))
      if input_processing_info.input_producer_op is not None:
        global_input_producer_op.extend(input_processing_info.input_producer_op)
      # Build the per-worker model replica.
      for rel_device_num in range(len(self.devices)):
        abs_device_num = task_num * len(self.devices) + rel_device_num
        with self.variable_mgr.create_outer_variable_scope(
            abs_device_num), tf.name_scope(
                'task_%i_tower_%i' % (task_num, rel_device_num)) as name_scope:
          task_results = self.add_forward_pass_and_gradients(
              phase_train, rel_device_num, abs_device_num,
              input_processing_info, gpu_compute_stage_ops, gpu_grad_stage_ops)

          if self.params.backbone_model_path:
            self.model.add_backbone_saver()

          if phase_train:
            losses.append(task_results['loss'])
            device_grads.append(task_results['gradvars'])
          else:
            all_logits.append(task_results['logits'])
          if not phase_train or self.params.print_training_accuracy:
            for name, op in task_results.items():
              if name.startswith('accuracy:'):
                key = name[9:]
                if key not in all_accuracy_ops:
                  all_accuracy_ops[key] = []
                all_accuracy_ops[key].append(op)

          if rel_device_num == 0:
            # Retain the Batch Normalization updates operations only
            # from the first tower. These operations update the moving
            # mean and moving variance variables, which are updated
            # (but not used) during training, and used during
            # evaluation. The moving mean and variance approximate the
            # true mean and variance across all images in the
            # dataset. Therefore, in replicated mode, these moving
            # averages would be almost identical for each tower, and
            # so we only update and save the moving averages for one
            # tower. In parameter server mode, all towers share a copy
            # of the variables so we also only need to update and save
            # the moving averages once.
            update_ops.extend(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope))
            assert not self.variable_mgr.staging_delta_ops

    enqueue_ops = []
    if gpu_compute_stage_ops:
      enqueue_ops.append(tf.group(*gpu_compute_stage_ops,
                                  name='gpu_compute_stage_ops'))
    assert not self.variable_mgr.supports_staged_vars()
    assert not gpu_grad_stage_ops

    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  enqueue_ops, update_ops, all_accuracy_ops,
                                  phase_train)
    if global_input_producer_op:
      global_input_producer_op = tf.group(*global_input_producer_op)
    else:
      global_input_producer_op = None
    return (global_input_producer_op, enqueue_ops, fetches)

  def add_forward_pass_and_gradients(self,
                                     phase_train,
                                     rel_device_num,
                                     abs_device_num,
                                     input_processing_info,
                                     gpu_compute_stage_ops,
                                     gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes
    if self.datasets_use_prefetch:
      function_buffering_resource = None
      if input_processing_info.function_buffering_resources:
        function_buffering_resource = (
            input_processing_info.function_buffering_resources[rel_device_num])

      input_data = None
      if input_processing_info.multi_device_iterator_input:
        input_data = (
            input_processing_info.multi_device_iterator_input[rel_device_num])

      # Exactly one of function_buffering_resource or input_data is not None.
      if function_buffering_resource is None and input_data is None:
        raise ValueError('Both function_buffering_resource and input_data '
                         'cannot be null if datasets_use_prefetch=True')
      if function_buffering_resource is not None and input_data is not None:
        raise ValueError('Both function_buffering_resource and input_data '
                         'cannot be specified. Only one should be.')
      with tf.device(self.raw_devices[rel_device_num]):
        if function_buffering_resource is not None:
          subset = 'validation' if self._doing_eval else 'train'
          input_list = prefetching_ops.function_buffering_resource_get_next(
              function_buffering_resource,
              output_types=self.model.get_input_data_types(subset))
        else:
          input_list = input_data
    else:
      if not self.dataset.use_synthetic_gpu_inputs():
        input_producer_stage = input_processing_info.input_producer_stages[
            rel_device_num]
        with tf.device(self.cpu_device):
          host_input_list = input_producer_stage.get()
        with tf.device(self.raw_devices[rel_device_num]):
          gpu_compute_stage = data_flow_ops.StagingArea(
              [inp.dtype for inp in host_input_list],
              shapes=[inp.get_shape() for inp in host_input_list])
          # The CPU-to-GPU copy is triggered here.
          gpu_compute_stage_op = gpu_compute_stage.put(host_input_list)
          input_list = gpu_compute_stage.get()
          gpu_compute_stage_ops.append(gpu_compute_stage_op)
      else:
        with tf.device(self.raw_devices[rel_device_num]):
          # Minor hack to avoid H2D copy when using synthetic data
          input_list = self.model.get_synthetic_inputs(
              BenchmarkCNN.GPU_CACHED_INPUT_VARIABLE_NAME, nclass)

    # Labels reshaping happens all on gpu:0. Reshaping synthetic labels on
    # multiple devices slows down XLA computation for an unknown reason.
    # TODO(b/116875203): Find/address root cause of XLA slow down.
    labels_device_placement_hack = (
        self.dataset.use_synthetic_gpu_inputs() and self.params.xla_compile)

    def device_aware_reshape(tensor, shape):
      device = self.devices[rel_device_num]
      # Labels are int32, place reshapes on gpu:0 (no device placement) when the
      # hack is enabled.
      if labels_device_placement_hack and tensor.dtype == tf.int32:
        device = ''
      with tf.device(device):
        return tf.reshape(tensor, shape=shape)

    subset = 'validation' if self._doing_eval else 'train'
    input_shapes = self.model.get_input_shapes(subset)
    input_list = [
        device_aware_reshape(input_list[i], shape=input_shapes[i])
        for i in range(len(input_list))
    ]

    def forward_pass_and_gradients():
      """Builds forward pass and gradient computation network.

      When phase_train=True and print_training_accuracy=False:
        return [loss] + grads

      When phase_train=True and print_training_accuracy=True:
        return [logits, loss] + grads

      When phase_train=False,
        return [logits]

      Its output can always be unpacked by

      ```
        outputs = forward_pass_and_gradients()
        logits, loss, grads = unpack_forward_pass_and_gradients_output(outputs)
      ```

      Returns:
        outputs: A list of tensors depending on different modes.
      """

      build_network_result = self.model.build_network(
          input_list, phase_train, nclass)
      logits = build_network_result.logits

      if not phase_train:
        return [logits]

      base_loss = self.model.loss_function(input_list, build_network_result)
      params = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num)
      l2_loss = None
      we_loss = None
      relu_x_loss = None
      total_loss = base_loss
      with tf.name_scope('l2_loss'):
        fp32_params = params
        if self.model.data_type == tf.float16 and self.params.fp16_vars:
          # fp16 reductions are very slow on GPUs, so cast to fp32 before
          # calling tf.nn.l2_loss and tf.add_n.
          # TODO(b/36217816): Once the bug is fixed, investigate if we should do
          # this reduction in fp16.
          fp32_params = (tf.cast(p, tf.float32) for p in params)
        filtered_params = self.model.filter_l2_loss_vars(fp32_params)
        if rel_device_num == len(self.devices) - 1:
          # We compute the L2 loss for only one device instead of all of them,
          # because the L2 loss for each device is the same. To adjust for this,
          # we multiply the L2 loss by the number of devices. We choose the
          # last device because for some reason, on a Volta DGX1, the first four
          # GPUs take slightly longer to complete a step than the last four.
          # TODO(reedwm): Shard the L2 loss computations across GPUs.
          if self.params.single_l2_loss_op:
            # TODO(reedwm): If faster, create a fused op that does the L2 loss
            # on multiple tensors, and use that instead of concatenating
            # tensors.
            reshaped_params = [tf.reshape(p, (-1,)) for p in filtered_params]
            l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
          else:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in filtered_params])
      weight_decay = self.params.weight_decay
      mlperf.logger.log(key=mlperf.tags.OPT_WEIGHT_DECAY, value=weight_decay)
      if (weight_decay is not None and weight_decay != 0. and
          l2_loss is not None):
        mlperf.logger.log(key=mlperf.tags.MODEL_L2_REGULARIZATION,
                          value=weight_decay)
        total_loss += len(self.devices) * weight_decay * l2_loss

      if (self.params.use_relu_x and
          self.params.relu_x_update == 'gradient_descent'):
        with tf.name_scope('relu_x_loss'):
          relu_params = self.model.filter_relu_loss_vars(fp32_params)
          relu_x_decay = self.params.relu_x_decay
          if (rel_device_num == len(self.devices) - 1 and
              relu_x_decay is not None and relu_x_decay != 0):
            relu_x_loss = tf.add_n([tf.nn.l2_loss(v) for v in relu_params])
            total_loss += len(self.devices) * relu_x_decay * relu_x_loss

      weight_equalization = self.params.weight_equalization
      if (weight_equalization is not None and weight_equalization != 0):
        with tf.name_scope('weight_equal_loss'):
          we_params = self.model.filter_we_loss_vars(fp32_params)
          if rel_device_num == len(self.devices) - 1:
            if self.params.weight_equalization_type == 'kurtosis':
              we_loss_fn = self.model.kurtosis_loss
            elif self.params.weight_equalization_type == 'tail':
              we_loss_fn = self.model.tail_loss
            we_loss = tf.add_n([we_loss_fn(v) for v in we_params])
            total_loss += len(self.devices) * weight_equalization * we_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      scaled_loss = (total_loss if self.loss_scale is None
                     else total_loss * self.loss_scale)
      grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
      if self.params.sparse_to_dense_grads:
        # Passing a sparse gradient to convert_to_tensor turns it into a dense
        # gradient. A sparse gradient is an instance of tf.IndexedSlices.
        # convert_to_tensor does not modify dense tensors.
        grads = [tf.convert_to_tensor(g) for g in grads]
      if self.loss_scale is not None:
        # TODO(reedwm): If automatic loss scaling is not used, we could avoid
        # these multiplications by directly modifying the learning rate instead.
        # If this is done, care must be taken to ensure that this scaling method
        # is correct, as some optimizers square gradients and do other
        # operations which might not be compatible with modifying both the
        # gradients and the learning rate.

        grads = [
            grad * tf.cast(1. / self.loss_scale, grad.dtype) for grad in grads
        ]

      if self.params.variable_update == 'horovod':
        import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
        if self.params.horovod_device:
          horovod_device = '/%s:0' % self.params.horovod_device
        else:
          horovod_device = ''
        # All-reduce gradients using Horovod.
        grads = [hvd.allreduce(grad, average=False, device_dense=horovod_device)
                 for grad in grads]

      if self.params.staged_vars:
        grad_dtypes = [grad.dtype for grad in grads]
        grad_shapes = [grad.shape for grad in grads]
        grad_stage = data_flow_ops.StagingArea(grad_dtypes, grad_shapes)
        grad_stage_op = grad_stage.put(grads)
        # In general, this decouples the computation of the gradients and
        # the updates of the weights.
        # During the pipeline warm up, this runs enough training to produce
        # the first set of gradients.
        gpu_grad_stage_ops.append(grad_stage_op)
        grads = grad_stage.get()

      if self.params.loss_type_to_report == 'total_loss':
        loss = total_loss
      else:
        loss = base_loss

      if self.params.print_training_accuracy:
        return [logits, loss] + grads
      else:
        return [loss] + grads

    def unpack_forward_pass_and_gradients_output(forward_pass_and_grad_outputs):
      """Unpacks outputs from forward_pass_and_gradients.

      Args:
        forward_pass_and_grad_outputs: Output from forward_pass_and_gradients.

      Returns:
        logits: Unscaled probability distribution from forward pass.
          If unavailable, None is returned.
        loss: Loss function result from logits.
          If unavailable, None is returned.
        grads: Gradients for all trainable variables.
          If unavailable, None is returned.
      """
      logits = None
      # logits is only fetched in non-train mode or when
      # print_training_accuracy is set.
      if not phase_train or self.params.print_training_accuracy:
        logits = forward_pass_and_grad_outputs.pop(0)

      loss = (
          forward_pass_and_grad_outputs[0]
          if forward_pass_and_grad_outputs else None)
      grads = (
          forward_pass_and_grad_outputs[1:]
          if forward_pass_and_grad_outputs else None)

      return logits, loss, grads

    def make_results(logits, loss, grads):
      """Generate results based on logits, loss and grads."""
      results = {}  # The return value

      if logits is not None:
        results['logits'] = logits
        accuracy_ops = self.model.accuracy_function(input_list, logits)
        for name, op in accuracy_ops.items():
          results['accuracy:' + name] = op

      if loss is not None:
        results['loss'] = loss

      if grads is not None:
        param_refs = self.variable_mgr.trainable_variables_on_device(
            rel_device_num, abs_device_num, writable=True)
        results['gradvars'] = list(zip(grads, param_refs))

      return results

    with tf.device(self.devices[rel_device_num]):
      outputs = maybe_compile(forward_pass_and_gradients, self.params)
      logits, loss, grads = unpack_forward_pass_and_gradients_output(outputs)
      return make_results(logits, loss, grads)

  def get_input_preprocessor(self):
    """Returns the image preprocessor to used, based on the model.

    Returns:
      The image preprocessor, or None if synthetic data should be used.
    """
    shift_ratio = 0
    if self.job_name:
      # shift_ratio prevents multiple workers from processing the same batch
      # during a step
      shift_ratio = self.task_index / self.num_workers

    processor_class = self.dataset.get_input_preprocessor(
        self.params.input_preprocessor)
    assert processor_class
    subset = 'validation' if self._doing_eval else 'train'
    return processor_class(
        self.batch_size * self.batch_group_size,
        self.model.get_input_shapes(subset),
        len(self.devices) * self.batch_group_size,
        dtype=self.model.data_type,
        train=(not self._doing_eval),
        # TODO(laigd): refactor away image model specific parameters.
        distortions=self.params.distortions,
        resize_method=self.resize_method,
        shift_ratio=shift_ratio,
        summary_verbosity=self.params.summary_verbosity,
        distort_color_in_yiq=self.params.distort_color_in_yiq,
        fuse_decode_and_crop=self.params.fuse_decode_and_crop,
        match_mlperf=self.params.ml_perf)

  def add_sync_queues_and_barrier(self, name_prefix, enqueue_after_list):
    """Adds ops to enqueue on all worker queues.

    Args:
      name_prefix: prefixed for the shared_name of ops.
      enqueue_after_list: control dependency from ops.

    Returns:
      An op that should be used as control dependency before starting next step.
    """
    self.sync_queue_counter += 1
    with tf.device(self.sync_queue_devices[(
        self.sync_queue_counter % len(self.sync_queue_devices))]):
      sync_queues = [
          tf.FIFOQueue(self.num_workers, [tf.bool], shapes=[[]],
                       shared_name='%s%s' % (name_prefix, i))
          for i in range(self.num_workers)]
      queue_ops = []
      # For each other worker, add an entry in a queue, signaling that it can
      # finish this step.
      token = tf.constant(False)
      with tf.control_dependencies(enqueue_after_list):
        for i, q in enumerate(sync_queues):
          if i == self.task_index:
            queue_ops.append(tf.no_op())
          else:
            queue_ops.append(q.enqueue(token))

      # Drain tokens off queue for this worker, one for each other worker.
      queue_ops.append(
          sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

      return tf.group(*queue_ops)


def _is_mkl_flag_absent(mkl_flag):
  return not (absl_flags.FLAGS.is_parsed() and mkl_flag in absl_flags.FLAGS
              and absl_flags.FLAGS[mkl_flag].present)


def _print_os_env_ignored_warning(mkl_flag, flag_default_val, os_env_var):
  tf.logging.warn(
      ('OS ENV variable %s=%s is ignored and script default: '
       '%s is used. Use --%s to override.') %
      (os_env_var, os.environ[os_env_var], flag_default_val, mkl_flag))


def _set_environ_vars(params):
  """Sets up the environment variables that BenchmarkCNN should use."""
  if params.batchnorm_persistent:
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  else:
    os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)
  if params.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  if params.autotune_threshold:
    os.environ['TF_AUTOTUNE_THRESHOLD'] = str(params.autotune_threshold)
  os.environ['TF_SYNC_ON_FINISH'] = str(int(params.sync_on_finish))
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Sets environment variables for MKL
  # If OS ENV vars are overridden by script defaults, a warning msg is printed.
  if params.mkl:
    mkl_flags = ['kmp_blocktime', 'kmp_settings', 'kmp_affinity',
                 'num_intra_threads']
    for mkl_flag in mkl_flags:
      os_env_var = mkl_flag.upper()
      if mkl_flag == 'num_intra_threads':
        os_env_var = 'OMP_NUM_THREADS'
      flag_val = str(getattr(params, mkl_flag))
      if _is_mkl_flag_absent(mkl_flag) and os_env_var in os.environ:
        _print_os_env_ignored_warning(mkl_flag, flag_val, os_env_var)
      os.environ[os_env_var] = flag_val
      if mkl_flag == 'num_intra_threads' and not params.num_intra_threads:
        os.environ.pop(os_env_var, None)

  # Sets GPU thread settings
  if params.device.lower() == 'gpu':
    params = params._replace(gpu_thread_mode=params.gpu_thread_mode.lower())
    if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
      raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)
    os.environ['TF_GPU_THREAD_MODE'] = params.gpu_thread_mode

    if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
      raise ValueError(
          'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
          params.per_gpu_thread_count)
    # Default to two threads. One for the device compute and the other for
    # memory copies.
    per_gpu_thread_count = params.per_gpu_thread_count or 2
    total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

    if params.gpu_thread_mode == 'gpu_private':
      os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    elif params.gpu_thread_mode == 'gpu_shared':
      os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

    cpu_count = multiprocessing.cpu_count()
    if not params.num_inter_threads and params.gpu_thread_mode in [
        'gpu_private', 'gpu_shared'
    ]:
      main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
      params = params._replace(num_inter_threads=main_thread_count)

    if (params.datasets_use_prefetch and
        params.datasets_num_private_threads is None):
      # From the total cpu thread count, subtract the total_gpu_thread_count,
      # and then 2 threads per GPU device for event monitoring and sending /
      # receiving tensors
      num_monitoring_threads = 2 * params.num_gpus
      num_private_threads = max(
          cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
      params = params._replace(datasets_num_private_threads=num_private_threads)
  return params


def setup(params):
  """Sets up the environment that BenchmarkCNN should run in.

  Args:
    params: Params tuple, typically created by make_params or
      make_params_from_flags.

  Returns:
    A potentially modified params.
  Raises:
    ValueError: invalid parames combinations.
  """
  # Set up environment variables before doing any other global initialization to
  # make sure it uses the appropriate environment variables.
  params = _set_environ_vars(params)

  # horovod needs to be initialized before create_config_proto() call since
  # it will be used in config generation if enabled.
  if params.variable_update == 'horovod':
    import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
    hvd.init()

  platforms_util.initialize(params, create_config_proto(params))

  if not params.job_name:
    # Create a dummy session to initialize TF global variables using the input
    # params. Otherwise, ListDevices function may create global devices using
    # the default config instead of using the user provided config.
    #
    # TODO(hinsu): Find a way to achieve the same for distributed benchmark. It
    # is not legal to create distributed session after local session. It is also
    # not possible to create distributed session here as that results in
    # multiple creation of ClusterManager and Server.
    with tf.Session(config=create_config_proto(params)) as sess:
      del sess

  return params


def maybe_compile(computation, params):
  if params and params.xla_compile:
    return tf.xla.experimental.compile(computation)
  else:
    return computation()
