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
const FileParams = goog.require('proto.eeg_modelling.protos.FileParams');
const FilterParams = goog.require('proto.eeg_modelling.protos.FilterParams');
const ResponseType = goog.require('goog.net.XhrIo.ResponseType');
const SimilarPatternsRequest = goog.require('proto.eeg_modelling.protos.SimilarPatternsRequest');
const SimilarPatternsResponse = goog.require('proto.eeg_modelling.protos.SimilarPatternsResponse');
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

/**
 * @typedef {!DataRequest|!SimilarPatternsRequest}
 */
let ProtoRequest;


class Requests {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will make a new HTTP request when a data request
    // property changes.
    store.registerListener(
        Store.RequestProperties, 'Requests',
        (store, changedProperties) =>
            this.handleRequestParameters(store, changedProperties));
    // This listener callback will make a new HTTP request to search similar
    // patterns.
    store.registerListener(
        [Store.Property.SIMILAR_PATTERN_TARGET], 'Requests',
        (store) => this.handleSearchSimilarPattern(store));

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
   * Creates a request with XhrIo and wraps it in a promise.
   * @param {string} url The path to send the request to.
   * @param {!ProtoRequest} requestContent Proto instance
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
   * Sends a request and dispatches an action once the request ends.
   * @param {string} url The path to send the request to.
   * @param {!ProtoRequest} requestContent Proto request instance.
   * @param {function(!Object): !Object} formatResponse Formats response data.
   * @param {!Dispatcher.ActionType} actionOk Action to dispatch if the
   *     request succeeds.
   * @param {!Dispatcher.ActionType} actionError Action to dispatch if the
   *     request fails.
   */
  sendRequest(url, requestContent, formatResponse, actionOk, actionError) {
    this.createXhrIoPromise(url, requestContent, formatResponse)
      .then((data) => {
        Dispatcher.getInstance().sendAction({
          actionType: actionOk,
          data,
        });
      })
      .catch((error) => {
        const errorResponse = /** @type {!ErrorResponse} */ (error);
        log.error(
            this.logger_,
            errorResponse.message + ', ' + errorResponse.detail);
        Dispatcher.getInstance().sendAction({
          actionType: actionError,
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
   * @private
   */
  convertIndexStrToChannelDataId_(channelStr) {
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

  /**
   * Sets the channelDataIds param from a list into a request protobuf.
   * @param {!ProtoRequest} requestContent Proto request to set the param to.
   * @param {?Array<string>} channelIds Array of channel ids.
   * @private
   */
  setChannelDataIdsParam_(requestContent, channelIds) {
    let channelDataIds = [];
    if (channelIds) {
      channelDataIds =
          channelIds
              .map(
                  (channelId) =>
                      this.convertIndexStrToChannelDataId_(channelId))
              .filter(channelDataId => channelDataId);
    }

    requestContent.setChannelDataIdsList(channelDataIds);
  }

  /**
   * Sets the file parameters from the store into a request protobuf.
   * @param {!DataRequest|!FileParams} requestContent Proto instance to set the
   *     params to.
   * @param {!Store.StoreData} store Data from the store.
   * @private
   */
  setFileParams_(requestContent, store) {
    requestContent.setTfExSstablePath(store.tfExSSTablePath);
    requestContent.setSstableKey(store.sstableKey);
    requestContent.setPredictionSstablePath(store.predictionSSTablePath);
    requestContent.setEdfPath(store.edfPath);
    requestContent.setTfExFilePath(store.tfExFilePath);
    requestContent.setPredictionFilePath(store.predictionFilePath);
  }

  /**
   * Sets the filter parameters from the store into a request protobuf
   * @param {!DataRequest|!FilterParams} requestContent Proto instance to set
   *     the params to.
   * @param {!Store.StoreData} store Data from the store.
   * @private
   */
  setFilterParams_(requestContent, store) {
    requestContent.setLowCut(store.lowCut);
    requestContent.setHighCut(store.highCut);
    requestContent.setNotch(store.notch);
  }

  /**
   * Handles HTTP requests for chunk data.
   * @param {!Store.StoreData} store Snapshot of chunk data store.
   * @param {!Array<!Store.Property>} changedProperties List of the properties
   *     that changed because of the last action.
   */
  handleRequestParameters(store, changedProperties) {
    const fileParamDirty = Store.FileRequestProperties.some(
        (param) => changedProperties.includes(param));
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.REQUEST_START,
      data: {
        fileParamDirty,
      },
    });
    const url = '/waveform_data/chunk';
    const requestContent = new DataRequest();
    this.setFileParams_(requestContent, store);
    this.setFilterParams_(requestContent, store);
    this.setChannelDataIdsParam_(requestContent, store.channelIds);

    requestContent.setChunkDurationSecs(store.chunkDuration);
    requestContent.setChunkStart(store.chunkStart);

    const formatResponse = (response) => {
      return DataResponse.deserializeBinary(response);
    };

    this.sendRequest(
      url,
      requestContent,
      formatResponse,
      Dispatcher.ActionType.REQUEST_RESPONSE_OK,
      Dispatcher.ActionType.REQUEST_RESPONSE_ERROR,
    );
  }

  /**
   * Creates a SimilarPatternsRequest proto from the data saved in the store.
   * @param {!Store.StoreData} store Snapshot of chunk data store.
   * @return {!SimilarPatternsRequest} A SimilarPatternsRequest proto object.
   * @private
   */
  createSimilarPatternsRequest_(store) {
    const fileParams = new FileParams();
    const filterParams = new FilterParams();
    this.setFileParams_(fileParams, store);
    this.setFilterParams_(filterParams, store);

    const requestContent = new SimilarPatternsRequest();
    requestContent.setFileParams(fileParams);
    requestContent.setFilterParams(filterParams);
    requestContent.setStartTime(store.similarPatternTarget.startTime);
    requestContent.setDuration(store.similarPatternTarget.duration);

    this.setChannelDataIdsParam_(requestContent, store.channelIds);

    return requestContent;
  }

  /**
   * Sends a request to search for a similar pattern.
   * @param {!Store.StoreData} store Data from the store.
   */
  handleSearchSimilarPattern(store) {
    if (!store.similarPatternTarget) {
      return;
    }
    const url = '/similarity';
    const requestContent = this.createSimilarPatternsRequest_(store);
    const formatResponse = (response) => {
      return SimilarPatternsResponse.deserializeBinary(response);
    };

    this.sendRequest(
      url,
      requestContent,
      formatResponse,
      Dispatcher.ActionType.SEARCH_SIMILAR_RESPONSE_OK,
      Dispatcher.ActionType.SEARCH_SIMILAR_RESPONSE_ERROR,
    );
  }
}

goog.addSingletonGetter(Requests);

exports = Requests;
