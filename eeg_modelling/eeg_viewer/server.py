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

from absl import app as absl_app
from absl import flags
from absl import logging

import flask
from flask import wrappers as flask_wrappers
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from werkzeug.contrib import wrappers as werkzeug_wrappers

from eeg_modelling.eeg_viewer import data_source
from eeg_modelling.eeg_viewer import prediction_data_service
from eeg_modelling.eeg_viewer import similarity
from eeg_modelling.eeg_viewer import waveform_data_service
from eeg_modelling.pyprotos import data_pb2
from eeg_modelling.pyprotos import prediction_output_pb2
from eeg_modelling.pyprotos import similarity_pb2


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

# Maximum amount of EEG samples returned to the client
# This prevents overloading the response if the time window requested is too big
_MAX_SAMPLES_CLIENT = 2500


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


def FetchEdfDataSource(unused_edf_path):
  raise NotImplementedError('Load EDF')


def FetchTfExFromSSTable(unused_sstable, unused_key):
  raise NotImplementedError('Load TF Example from SSTable')


def FetchPredictionsFromSSTable(unused_sstable, unused_key):
  raise NotImplementedError('Load predictions from SSTable')


def GetWaveformDataSource(file_params):
  """Get the waveform DataSource instance from file parameters.

  Args:
    file_params: Request holding the file parameters.
  Returns:
    DataSource instance.
  Raises:
    IOError: No tf path provided
  """
  if file_params.tf_ex_sstable_path:
    tf_example, sstable_key = FetchTfExFromSSTable(
        file_params.tf_ex_sstable_path, file_params.sstable_key)
    waveform_data_source = TfExDataSourceConstructor(
        tf_example, sstable_key)

  elif file_params.edf_path:
    waveform_data_source = FetchEdfDataSource(file_params.edf_path)

  elif file_params.tf_ex_file_path:
    tf_example = FetchTfExFromFile(file_params.tf_ex_file_path)
    waveform_data_source = TfExDataSourceConstructor(tf_example, '')

  else:
    raise IOError('No path provided')

  return waveform_data_source


def GetPredictionsOutputs(file_params):
  """Get the PredictionOutputs instance from file parameters.

  Args:
    file_params: Request holding the file parameters.
  Returns:
    PredictionOutputs instance.
  """
  pred_outputs = None

  if file_params.tf_ex_sstable_path and file_params.prediction_sstable_path:
    pred_outputs = FetchPredictionsFromSSTable(
        file_params.prediction_sstable_path, file_params.sstable_key)

  elif file_params.tf_ex_file_path and file_params.prediction_file_path:
    pred_outputs = FetchPredictionsFromFile(file_params.prediction_file_path)

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


@MakeErrorHandler
def ValueErrorMessage(detail):
  return 'Wrong value provided: %s' % detail


@MakeErrorHandler
def KeyErrorMessage(unused_detail):
  return 'Key not found'


@MakeErrorHandler
def NotImplementedErrorMessage(detail):
  return '%s not implemented yet' % detail


@MakeErrorHandler
def IOErrorMessage(unused_detail):
  return 'Path not found'


def RegisterErrorHandlers(app):
  """Registers the error handlers in the flask app.

  Args:
    app: Flask application to register the handlers.
  """
  app.register_error_handler(ValueError, ValueErrorMessage)
  app.register_error_handler(KeyError, KeyErrorMessage)
  app.register_error_handler(NotImplementedError, NotImplementedErrorMessage)
  app.register_error_handler(IOError, IOErrorMessage)


def IndexPage():
  """Serves home page of the Waveforms Viewer."""
  return flask.render_template('index.html', file_type=FLAGS.file_type)


def MakeProtoRequestHandler(request_proto):
  """Decorator to create a request handler for a specific type of request.

  Args:
    request_proto: Constructor of the request proto type.
  Returns:
    Decorator to create a request handler.
  """
  def RequestHandler(create_data_response):
    """Actual decorator to create a request handler.

    Args:
      create_data_response: function that receives a proto request and
        returns a proto response and optional extra headers.
    Returns:
      Decorated function that can handle a request.
    """
    def Handler():
      """Handler of the request."""
      flask.request.environ['CONTENT_TYPE'] = 'application/x-protobuf'
      data_request = flask.request.parse_protobuf(request_proto)

      data_response, extra_headers = create_data_response(data_request)

      response = flask.make_response(data_response.SerializeToString())

      if extra_headers:
        for header in extra_headers:
          response.headers[header] = extra_headers[header]

      return response
    return Handler
  return RequestHandler


@MakeProtoRequestHandler(data_pb2.DataRequest)
def CreateDataResponse(request):
  """Creates a DataResponse from a DataRequest.

  Args:
    request: A DataRequest instance received from client.
  Returns:
    A DataResponse instance and extra headers.
  """
  waveform_data_source = GetWaveformDataSource(request)
  pred_outputs = GetPredictionsOutputs(request)

  data_response = data_pb2.DataResponse()
  data_response.waveform_metadata.CopyFrom(waveform_data_service.GetMetadata(
      waveform_data_source, _MAX_SAMPLES_CLIENT))
  data_response.waveform_chunk.CopyFrom(waveform_data_service.GetChunk(
      waveform_data_source, request, _MAX_SAMPLES_CLIENT))

  if pred_outputs:
    waveforms_pred = prediction_data_service.PredictionDataService(
        pred_outputs, waveform_data_source, _MAX_SAMPLES_CLIENT)

    data_response.prediction_metadata.CopyFrom(waveforms_pred.GetMetadata())
    data_response.prediction_chunk.CopyFrom(waveforms_pred.GetChunk(request))

  # When only an SSTable file pattern is provided, the cache will return the TF
  # Example under the first iterated key.  Since the order of the keys is not
  # guaranteed, this response will not be cached as it is not idempotent.
  extra_headers = dict()
  no_cache = request.tf_ex_sstable_path and not request.sstable_key
  extra_headers['Cache-Control'] = 'no-cache' if no_cache else 'public'

  return data_response, extra_headers


@MakeProtoRequestHandler(similarity_pb2.SimilarPatternsRequest)
def SearchSimilarPatterns(similar_patterns_request):
  """Searches through a file for similar patterns.

  Args:
    similar_patterns_request: SimilarPatternsRequest instance defining the
      search to perform.
  Returns:
    SimilarPatternsResponse instance with the found similar patterns, and no
      extra headers.
  Raises:
    ValueError: No channel selected.
  """
  requested_channels = similar_patterns_request.channel_data_ids

  if not requested_channels:
    raise ValueError('Must select at least one channel')

  file_params = similar_patterns_request.file_params
  waveform_data_source = GetWaveformDataSource(file_params)

  filter_params = similar_patterns_request.filter_params

  data = waveform_data_service.GetChunkDataAsNumpy(
      waveform_data_source, requested_channels,
      filter_params.low_cut, filter_params.high_cut, filter_params.notch)

  sampling_freq = waveform_data_service.GetSamplingFrequency(
      waveform_data_source, requested_channels)

  seen_events = similar_patterns_request.seen_events

  similar_patterns_response = similarity.CreateSimilarPatternsResponse(
      data, similar_patterns_request.start_time,
      similar_patterns_request.duration, seen_events, sampling_freq)

  return similar_patterns_response, {}


def RegisterRoutes(app):
  """Registers routes in the flask app.

  Args:
    app: Flask app to register the routes to.
  """
  app.add_url_rule('/', 'index', IndexPage, methods=['GET'])
  app.add_url_rule(
      '/waveform_data/chunk',
      'chunk_data',
      CreateDataResponse,
      methods=['POST'])
  app.add_url_rule(
      '/similarity',
      'similarity',
      SearchSimilarPatterns,
      methods=['POST'])


class Request(werkzeug_wrappers.ProtobufRequestMixin, flask_wrappers.Request):
  """Request class with protobuf mixin."""


def main(unused_argv):
  flask_app.request_class = Request

  RegisterErrorHandlers(flask_app)
  RegisterRoutes(flask_app)

  print('Serving in port ' + str(FLAGS.flask_port))
  server = WSGIServer(('', FLAGS.flask_port), flask_app)
  server.serve_forever()


if __name__ == '__main__':
  absl_app.run(main)
