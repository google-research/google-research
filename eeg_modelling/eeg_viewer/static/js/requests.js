// Copyright 2019 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

goog.module('eeg_modelling.eeg_viewer.Requests');

const BipolarChannel = goog.require('proto.eeg_modelling.protos.ChannelDataId.BipolarChannel');
const ChannelDataId = goog.require('proto.eeg_modelling.protos.ChannelDataId');
const DataRequest = goog.require('proto.eeg_modelling.protos.DataRequest');
const DataResponse = goog.require('proto.eeg_modelling.protos.DataResponse');
const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const EventType = goog.require('goog.net.EventType');
const ResponseType = goog.require('goog.net.XhrIo.ResponseType');
const SingleChannel = goog.require('proto.eeg_modelling.protos.ChannelDataId.SingleChannel');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const XhrIo = goog.require('goog.net.XhrIo');
const closureString = goog.require('goog.string');
const events = goog.require('goog.events');
const log = goog.require('goog.log');


/**
 * @typedef {{
 *   message: ?string,
 *   detail: ?string,
 *   traceback: ?string,
 * }}
 */
let ErrorResponse;


class Requests {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will make a new HTTP request when one of the
    // filter setting or chunk setting params change.
    store.registerListener(
        [
          Store.Property.CHUNK_START, Store.Property.CHUNK_DURATION,
          Store.Property.LOW_CUT, Store.Property.HIGH_CUT, Store.Property.NOTCH,
          Store.Property.CHANNEL_IDS
        ],
        'Requests', (store) => this.handleRequestParameters(false, store));
    // This listener callback will make a new HTTP request when one of the
    // file params change.
    store.registerListener(
        [
          Store.Property.TFEX_SSTABLE_PATH,
          Store.Property.PREDICTION_SSTABLE_PATH, Store.Property.EDF_PATH,
          Store.Property.SSTABLE_KEY, Store.Property.TFEX_FILE_PATH,
          Store.Property.PREDICTION_FILE_PATH
        ],
        'Requests', (store) => this.handleRequestParameters(true, store));

    this.logger_ = log.getLogger('eeg_modelling.eeg_viewer.Requests');
  }

  /**
   * Creates a new XhrIO object (used for testing HTTP requests).
   * @return {!XhrIo} An XhrIo object.
   */
  getXhrIo() {
    return new XhrIo();
  }

  /**
   * Creates a request for a chunk of waveform data.
   * @param {string} url The path to send the request to.
   * @param {!DataRequest} requestContent A request protobuf.
   * @param {function(!Object): !Object} formatResponse Formats response data.
   * @return {!Promise} A promise returning an XhrIo response.
   */
  createXhrIoPromise(url, requestContent, formatResponse) {
    const parseJsonResponse = function(response) {
      return JSON.parse(String.fromCharCode.apply(null, new Uint8Array(response)));
    };

    const xhrIo = this.getXhrIo();
    xhrIo.setResponseType(ResponseType.ARRAY_BUFFER);
    return new Promise((resolve, reject) => {
      events.listen(xhrIo, EventType.COMPLETE, function(e) {
        if (e.target.getStatus() != 200) {
          const response = (e.target.getResponseHeader('Content-Type') === 'application/json')
              ? parseJsonResponse(e.target.getResponse())
              : {};
          reject(response);
        } else {
          resolve(formatResponse(e.target.getResponse()));
        }
      });
      log.info(this.logger_, `POST request to ${url}, params: ${requestContent}`);
      xhrIo.send(url, 'POST', requestContent.serializeBinary());
    });
  }

  /**
   * Creates a data request Uri from the ChunkStore.
   * @param {!Store.StoreData} store Snapshot of chunk data store.
   * @return {!Promise} A promise that makes an Xhr request.
   */
  createDataResponsePromise(store) {
    const url = '/waveform_data/chunk';

    let channelDataIds = [];
    if (store.channelIds) {
      channelDataIds =
          store.channelIds
              .map((channelId) =>
                  this.convertIndexStrToChannelDataId(channelId))
              .filter(channelDataId => channelDataId);
    }

    const requestContent = new DataRequest();
    requestContent.setTfExSstablePath(store.tfExSSTablePath);
    requestContent.setSstableKey(store.sstableKey);
    requestContent.setPredictionSstablePath(store.predictionSSTablePath);
    requestContent.setEdfPath(store.edfPath);
    requestContent.setTfExFilePath(store.tfExFilePath);
    requestContent.setPredictionFilePath(store.predictionFilePath);
    requestContent.setChunkDurationSecs(store.chunkDuration);
    requestContent.setChunkStart(store.chunkStart);
    requestContent.setChannelDataIdsList(channelDataIds);
    requestContent.setLowCut(store.lowCut);
    requestContent.setHighCut(store.highCut);
    requestContent.setNotch(store.notch);

    const formatResponse = (response) => {
      return DataResponse.deserializeBinary(response);
    };

    return this.createXhrIoPromise(url, requestContent, formatResponse);
  }

  /**
   * Handles HTTP requests for chunk data.
   * @param {boolean} fileParamDirty Indicates if the call was triggered
   *     by a change in a file parameter.
   * @param {!Store.StoreData} store Snapshot of chunk data store.
   */
  handleRequestParameters(fileParamDirty, store) {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.REQUEST_START,
      data: {
        fileParamDirty,
      },
    });
    this.createDataResponsePromise(store)
        .then((value) => {
          Dispatcher.getInstance().sendAction({
            actionType: Dispatcher.ActionType.REQUEST_RESPONSE_OK,
            data: value,
          });
        })
        .catch((error) => {
          const errorResponse = /** @type {!ErrorResponse} */ (error);
          log.error(
              this.logger_,
              errorResponse.message + ', ' + errorResponse.detail);
          Dispatcher.getInstance().sendAction({
            actionType: Dispatcher.ActionType.REQUEST_RESPONSE_ERROR,
            data: {
              message: errorResponse.message || 'Unknown Request Error',
            },
          });
        });
  }

  /**
   * Converts channel ID's expressed in string form to proto format.
   * @param {string} channelStr The channel ID in string format.
   * @return {?ChannelDataId} Converted string channel ID.
   */
  convertIndexStrToChannelDataId(channelStr) {
    const channelIndices = channelStr.split('-')
        .map(x => closureString.parseInt(x))
        .filter(x => !isNaN(x));
    const channelDataId = new ChannelDataId();

    if (channelIndices.length == 2) {
      const bipolarChannel = new BipolarChannel();
      bipolarChannel.setIndex(channelIndices[0]);
      bipolarChannel.setReferentialIndex((channelIndices[1]));

      channelDataId.setBipolarChannel(bipolarChannel);
    } else if (channelIndices.length == 1) {
      const singleChannel = new SingleChannel();
      singleChannel.setIndex((channelIndices[0]));

      channelDataId.setSingleChannel(singleChannel);
    } else {
      log.error(this.logger_, ('Malformed channel string ' + channelStr));
      return null;
    }
    return channelDataId;
  }
}

goog.addSingletonGetter(Requests);

exports = Requests;
