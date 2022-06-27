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

"""train!

train estimators

"""
import os


from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options
import pandas as pd
from tensorflow.io import gfile

import icetea.beam_utils as beam_utils
from icetea.config import MakeParameters
from icetea.utils import DataSimulation
from icetea.utils import experiments


FLAGS = flags.FLAGS
flags.DEFINE_integer('b_quick', 2, 'number of simulations to run - quick')
flags.DEFINE_integer('b', 10, 'number of simulations to run')
flags.DEFINE_string(
    'path_output',
    '',
    'The configs file path')

flags.DEFINE_string(
    'path_datasets',
    '',
    'The data file path')

flags.DEFINE_enum(
    'setting',
    default='ukb',
    case_sensitive=False,
    enum_values=['quick', 'samples', 'covariates', 'synthetic', 'ukb'],
    help='Type of experiment: quick, samples, covariates, synthetic, ukb')

flags.DEFINE_list('data_names', ['ukb'], 'Data: Names')
flags.DEFINE_list('data_ncol', [1, 10, 100, 500, 1000, 10000],
                  'Data: Number of columns')
flags.DEFINE_list('data_n', [500, 1000, 5000, 10000, 50000],
                  'Data: Sample Sizes')

flags.DEFINE_list('data_noise', [True], 'simple_linear: Noise')
flags.DEFINE_list('data_linear', [True, False], 'simple_linear: Linear')
flags.DEFINE_bool('data_low_dimension', True, 'ACIC: dimension')

flags.DEFINE_list('method_estimator', ['oaxaca', 'aipw'], 'Method: Estimator')

flags.DEFINE_list('method_base_model', [
    'ElasticNet', 'Lasso', 'MeanDiff', 'RandomForest', 'LinearRegression',
    'NN_regression'
], 'Method: Model Control')

flags.DEFINE_list('method_metric', ['mse'], 'Method: Metric')
flags.DEFINE_list('method_prop_score', ['prop'],
                  'Method: Prop Score')


flags.DEFINE_bool('pipe', False, 'Test the model without making pipeline')


def main(_):
  print('Running...')

  # Making config files.
  configs_data = {'data_name': FLAGS.data_names}
  if FLAGS.data_names[0] == 'simple_linear':
    configs_data['data_noise'] = FLAGS.data_noise
    configs_data['data_linear'] = FLAGS.data_linear
    configs_data['setting'] = [FLAGS.setting]
    if FLAGS.setting == 'covariates':
      print('Setting: covariates')
      configs_data['data_n'] = [10000]
      configs_data['data_num_covariates'] = FLAGS.data_ncol
    elif FLAGS.setting == 'samples':
      print('Setting: samples')
      configs_data['data_n'] = FLAGS.data_n
      configs_data['data_num_covariates'] = [10]
    else:
      print('Setting: quick')
      configs_data['data_n'] = [10000]
      configs_data['data_num_covariates'] = [10, 15]
      configs_data['data_linear'] = [True]
  else:
    configs_data['data_path'] = [FLAGS.path_datasets]
    configs_data['data_low_dimension'] = [FLAGS.data_low_dimension]

  configs_methods = {
      'name_estimator': FLAGS.method_estimator,
      'name_base_model': FLAGS.method_base_model,
      'name_metric': FLAGS.method_metric,
      'name_prop_score': FLAGS.method_prop_score
  }
  config_files = MakeParameters(configs_data, configs_methods)
  param_method = config_files.parameters_method
  param_data = config_files.parameters_data

  # Output folder.
  if not gfile.exists(FLAGS.path_output):
    raise ValueError("Invalid choice of path '{}'.".format(FLAGS.path_output))

  print('Output Path:', FLAGS.path_output)

  if FLAGS.setting == 'quick':
    pipe_input = param_data * int(FLAGS.b_quick)
  elif FLAGS.setting == 'synthetic' and not FLAGS.pipe:
    b_count_path = FLAGS.path_datasets + FLAGS.data_names+'/'
    if FLAGS.data_names == 'ACIC':
      if FLAGS.data_low_dimension:
        b_count_path = b_count_path + 'low_dimensional_datasets/'
      else:
        b_count_path = b_count_path + 'high_dimensional_datasets/'
      list_files = gfile.listdir(b_count_path)
      pipe_input = param_data * int(len(list_files))
    else:
      list_files = gfile.listdir(b_count_path)
      pipe_input = param_data * (int(len(list_files)) * 30)
  else:
    pipe_input = param_data * int(FLAGS.b)

  # For quick exploration
  if FLAGS.pipe:
    for d in pipe_input[0:10]:
      for method in param_method:
        table, _ = experiments(
            data=beam_utils.data(1, d), seed=1, param_method=method)
        print(table)
  else:
    beam_options = pipeline_options.PipelineOptions()
    print(param_data, param_method)
    with beam.Pipeline(options=beam_options) as pipe:
      _ = (
          pipe
          | 'Create Data' >> beam.Create(enumerate(pipe_input))
          | 'Run Methods' >> beam.FlatMap(beam_utils.organize_param_methods,
                                          param_method)
          | 'Shuffle 1' >> beam.Reshuffle()
          | beam.Map(beam_utils.methods)
          | 'Full data' >> beam.Map(beam_utils.convert_dict_to_csv_record)
          | 'Shuffle 2' >> beam.Reshuffle()
          | 'data save' >>
          beam.io.WriteToText(FLAGS.path_output + 'experiments_data')
          )

  if FLAGS.setting != 'ukb':
    # Running once to save colnames.
    data = DataSimulation(seed=0, param_data=param_data[0])
    _, table_names = experiments(data=data, seed=0,
                                 param_method=param_method[0])
    experiments_colnames = pd.DataFrame(table_names)
    path = os.path.join(FLAGS.path_output + 'experiments_colnames.txt')
    with gfile.GFile(path, 'wt') as table_names:
      table_names.write(experiments_colnames.to_csv(index=False))

  print('Done!')

if __name__ == '__main__':
  app.run(main)
