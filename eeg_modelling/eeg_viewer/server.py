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

"""Serves web UI requests for Waveforms Viewer.

Serves requests for various data queries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import traceback

from absl import app
from absl import flags
from absl import logging

import flask
from flask import wrappers as flask_wrappers
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from werkzeug.contrib import wrappers as werkzeug_wrappers

from eeg_modelling.eeg_viewer import data_source
from eeg_modelling.eeg_viewer import prediction_data_service
from eeg_modelling.eeg_viewer import waveform_data_service
from eeg_modelling.pyprotos import data_pb2
from eeg_modelling.pyprotos import prediction_output_pb2

FLAGS = flags.FLAGS

flags.DEFINE_boolean('flask_debug',
                     False,
                     'Run in debug mode.')
flags.DEFINE_string('flask_address',
                    '127.0.0.1',
                    'Debug server bind address (for local runs only)')
flags.DEFINE_integer('flask_port',
                     5000,
                     'Port for server to listen on',
                     lower_bound=1024)
flags.DEFINE_string('file_type',
                    'EEG',
                    'Waveform file type (e.g. ECG, EEG, etc)')

flask_app = flask.Flask(__name__)

_MAX_SAMPLES = 2500


def TfExDataSourceConstructor(*args):
  """Select a EEG or ECG/EKG TfExampleDataSource instance, given by FLAGS.file_type.

  Args:
    *args: arguments passed to the constructor
  Returns:
    TfExampleEegDataSource or TfExampleEkgDataSource instance
  """
  if FLAGS.file_type == 'EEG':
    return data_source.TfExampleEegDataSource(*args)
  elif FLAGS.file_type == 'ECG' or FLAGS.file_type == 'EKG':
    return data_source.TfExampleEkgDataSource(*args)
  else:
    logging.warning('Unknown file type %s, using EEG', FLAGS.file_type)
    return data_source.TfExampleEegDataSource(*args)


def FetchTfExFromFile(filepath):
  """Fetches TF Example from a file and wraps it in a WaveformsExample.

  Args:
    filepath: local file to load the TF example proto
  Returns:
    TF Example
  """
  tf_example = tf.train.Example()
  with open(filepath, 'rb') as f:
    tf_example.ParseFromString(f.read())
  return tf_example


def FetchPredictionsFromFile(filepath):
  """Fetch PredictionOutputs proto from file.

  Args:
    filepath: local file to load the PredictionOutputs proto
  Returns:
    PredictionOutputs
  """
  pred_outputs = prediction_output_pb2.PredictionOutputs()
  with open(filepath, 'rb') as f:
    pred_outputs.ParseFromString(f.read())

  return pred_outputs


def MakeErrorHandler(get_message):
  """Decorator for Error handler functions.

  Args:
    get_message: function that returns the error message for a specific case
  Returns:
    ErrorHandler function
  """
  def ErrorHandler(e):
    tb = traceback.format_exc(sys.exc_info()[2])
    detail = str(e)
    response = flask.jsonify(message=get_message(detail),
                             detail=detail,
                             traceback=tb)
    response.status_code = 404
    return response
  return ErrorHandler


@flask_app.errorhandler(ValueError)
@MakeErrorHandler
def ValueErrorMessage(detail):
  return 'Wrong value provided: %s' % detail


@flask_app.errorhandler(KeyError)
@MakeErrorHandler
def KeyErrorMessage(unused_detail):
  return 'Key not found'


@flask_app.errorhandler(NotImplementedError)
@MakeErrorHandler
def NotImplementedErrorMessage(detail):
  return '%s not implemented yet' % detail


@flask_app.errorhandler(IOError)
@MakeErrorHandler
def IOErrorMessage(unused_detail):
  return 'Path not found'


@flask_app.route('/', methods=['GET'])
def IndexPage():
  """Serves home page of the Waveforms Viewer."""
  return flask.render_template('index.html', file_type=FLAGS.file_type)


@flask_app.route('/waveform_data/chunk', methods=['POST'])
def RequestData():
  """Returns a data response from a DataRequest.

  Contains a slice of waveform data and accompanying metadata and optionally
  associated prediction data.
  Returns:
    A Flask response containing a serialized DataResponse.
  """
  flask.request.environ['CONTENT_TYPE'] = 'application/x-protobuf'
  data_request = flask.request.parse_protobuf(data_pb2.DataRequest)

  data_response = CreateDataResponse(data_request)

  response = flask.make_response(data_response.SerializeToString())

  return response


def CreateDataResponse(request):
  """Creates a DataResponse from a DataRequest.

  Args:
    request: A DataRequest instance received from client.
  Returns:
    A DataResponse instance.
  Raises:
    NotImplementedError: Try to load from SSTable or EDF
    IOError: No tf path provided
  """
  data_response = data_pb2.DataResponse()

  pred_outputs = None

  if request.tf_ex_sstable_path:
    raise NotImplementedError('Loading SSTables')

  elif request.edf_path:
    raise NotImplementedError('Loading EDF')

  elif request.tf_ex_file_path:
    tf_example = FetchTfExFromFile(request.tf_ex_file_path)
    waveform_data_source = TfExDataSourceConstructor(tf_example, '')

    if request.prediction_file_path:
      pred_outputs = FetchPredictionsFromFile(request.prediction_file_path)

  else:
    raise IOError('No path provided')

  data_response.waveform_metadata.CopyFrom(waveform_data_service.GetMetadata(
      waveform_data_source, _MAX_SAMPLES))
  data_response.waveform_chunk.CopyFrom(waveform_data_service.GetChunk(
      waveform_data_source, request, _MAX_SAMPLES))

  if pred_outputs:
    waveforms_pred = prediction_data_service.PredictionDataService(
        pred_outputs, waveform_data_source, _MAX_SAMPLES)

    data_response.prediction_metadata.CopyFrom(waveforms_pred.GetMetadata())
    data_response.prediction_chunk.CopyFrom(waveforms_pred.GetChunk(request))

  return data_response


class Request(werkzeug_wrappers.ProtobufRequestMixin, flask_wrappers.Request):
  """Request class with protobuf mixin."""


def main(unused_argv):
  flask_app.request_class = Request

  print('Serving in port ' + str(FLAGS.flask_port))
  server = WSGIServer(('', FLAGS.flask_port), flask_app)
  server.serve_forever()


if __name__ == '__main__':
  app.run(main)
