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

goog.module('eeg_modelling.eeg_viewer.Store');

const AttributionMap = goog.require('proto.eeg_modelling.protos.PredictionChunk.AttributionMap');
const ChannelDataId = goog.require('proto.eeg_modelling.protos.ChannelDataId');
const ChunkScoreData = goog.require('proto.eeg_modelling.protos.PredictionMetadata.ChunkScoreData');
const DataResponse = goog.require('proto.eeg_modelling.protos.DataResponse');
const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const JspbMap = goog.require('jspb.Map');
const log = goog.require('goog.log');
const {assert, assertArray, assertInstanceof, assertNumber, assertString} = goog.require('goog.asserts');

/**
 * Mode for the prediction data to be displayed.
 * @enum {string}
 */
const PredictionMode = {
  NONE: 'None',
  CHUNK_SCORES: 'Chunk Scores',
  ATTRIBUTION_MAPS: 'Attribution Maps',
};

/**
 * Properties of the store data.
 * @enum {string}
 */
const Property = {
  ABS_START: 'absStart',
  ANNOTATIONS: 'annotations',
  ATTRIBUTION_MAPS: 'attributionMaps',
  CHANNEL_IDS: 'channelIds',
  CHUNK_DURATION: 'chunkDuration',
  CHUNK_GRAPH_DATA: 'chunkGraphData',
  CHUNK_SCORES: 'chunkScores',
  CHUNK_START: 'chunkStart',
  EDF_PATH: 'edfPath',
  ERROR: 'error',
  FILE_TYPE: 'fileType',
  HIGH_CUT: 'highCut',
  INDEX_CHANNEL_MAP: 'indexChannelMap',
  IS_TYPING: 'isTyping',
  LABEL: 'label',
  LOADING_STATUS: 'loadingStatus',
  LOW_CUT: 'lowCut',
  NOTCH: 'notch',
  NUM_SECS: 'numSecs',
  PATIENT_ID: 'patientId',
  PREDICTION_CHUNK_SIZE: 'predictionChunkSize',
  PREDICTION_CHUNK_START: 'predictionChunkStart',
  PREDICTION_FILE_PATH: 'predictionFilePath',
  PREDICTION_MODE: 'predictionMode',
  PREDICTION_SSTABLE_PATH: 'predictionSSTablePath',
  SAMPLING_FREQ: 'samplingFreq',
  SERIES_HEIGHT: 'seriesHeight',
  SENSITIVITY: 'sensitivity',
  SSTABLE_KEY: 'sstableKey',
  TIMESCALE: 'timeScale',
  TFEX_FILE_PATH: 'tfExFilePath',
  TFEX_SSTABLE_PATH: 'tfExSSTablePath',
};

/** @const {!Array<!Property>} */
const responseProperties = [
  Property.ABS_START,
  Property.ANNOTATIONS,
  Property.ATTRIBUTION_MAPS,
  Property.CHUNK_GRAPH_DATA,
  Property.CHUNK_SCORES,
  Property.FILE_TYPE,
  Property.INDEX_CHANNEL_MAP,
  Property.NUM_SECS,
  Property.PATIENT_ID,
  Property.PREDICTION_CHUNK_SIZE,
  Property.PREDICTION_CHUNK_START,
  Property.SAMPLING_FREQ,
];

/**
 * @typedef {{
 *   properties: !Array<!Property>,
 *   id: string,
 *   callback: !Function,
 * }}
 */
let Listener;

/**
 * @typedef {{
 *   labelText: ?string,
 *   startTime: ?number,
 *   id: ?string,
 * }}
 */
let Annotation;

/**
 * @typedef {{
 *   cols: !Array<{id: string, label: string, type: string}>,
 *   rows: !Array<{c: !Array<{v: (boolean|number|string)}>}>,
 * }}
 */
let DataTableInput;

/**
 * @typedef {{
 *   message: string,
 *   timestamp: number,
 * }}
 */
let ErrorInfo;

/**
 * Possible status when loading data.
 * @enum {number}
 */
const LoadingStatus = {
  NO_DATA: 0,   /* No data loaded */
  LOADING: 1,   /* First request in progress */
  LOADED: 2,    /* Finished first request successfully */
  RELOADING: 3, /* Subsequent request in progress (e.g. move chunks, etc) */
  RELOADED: 4,  /* Finished any subsequent request (either success or not) */
};

/**
 * @typedef {{
 *   absStart: ?number,
 *   annotations: ?Array<!Annotation>,
 *   attributionMaps: ?JspbMap<string, !AttributionMap>,
 *   channelIds: ?Array<string>,
 *   chunkDuration: number,
 *   chunkGraphData: ?DataTableInput,
 *   chunkScores: ?Array<!ChunkScoreData>,
 *   chunkStart: number,
 *   edfPath: ?string,
 *   error: ?ErrorInfo,
 *   fileType: ?string,
 *   highCut: number,
 *   indexChannelMap: ?JspbMap<string, string>,
 *   isTyping: boolean,
 *   label: string,
 *   loadingStatus: !LoadingStatus,
 *   lowCut: number,
 *   notch: number,
 *   numSecs: ?number,
 *   patientId: ?string,
 *   predictionChunkSize: ?number,
 *   predictionChunkStart: ?number,
 *   predictionFilePath: ?string,
 *   predictionMode: !PredictionMode,
 *   predictionSSTablePath: ?string,
 *   samplingFreq: ?number,
 *   seriesHeight: number,
 *   sensitivity: number,
 *   sstableKey: ?string,
 *   timeScale: number,
 *   tfExSSTablePath: ?string,
 *   tfExFilePath: ?string,
 * }}
 */
let StoreData;


/**
 * Contains the state of the application in data stores.
 */
class Store {

  constructor() {
    /** @public {!StoreData} */
    this.storeData = {
      absStart: null,
      annotations: null,
      attributionMaps: null,
      channelIds: null,
      chunkDuration: 10,
      chunkGraphData: null,
      chunkScores: null,
      chunkStart: 0,
      edfPath: null,
      error: null,
      fileType: null,
      highCut: 70,
      indexChannelMap: null,
      isTyping: true,
      label: 'SZ',
      loadingStatus: LoadingStatus.NO_DATA,
      lowCut: 1.6,
      notch: 0,
      numSecs: null,
      patientId: null,
      predictionChunkSize: null,
      predictionChunkStart: null,
      predictionFilePath: null,
      predictionMode: PredictionMode.NONE,
      predictionSSTablePath: null,
      samplingFreq: null,
      seriesHeight: 100,
      sensitivity: 5,
      sstableKey: null,
      timeScale: 1,
      tfExSSTablePath: null,
      tfExFilePath: null,
    };

    /** @public {!Array<!Listener>} */
    this.registeredListeners = [];

    const dispatcher = Dispatcher.getInstance();

    const registerCallback = (actionType, callback) => {
      dispatcher.registerCallback(
        actionType,
        actionData => this.callbackWrapper(
          callback.bind(this),
          actionData,
        ),
      );
    };

    registerCallback(Dispatcher.ActionType.ANNOTATION_SELECTION,
                     this.handleAnnotationSelection);
    registerCallback(Dispatcher.ActionType.CHANGE_TYPING_STATUS,
                     this.handleChangeTypingStatus);
    registerCallback(Dispatcher.ActionType.ERROR,
                     this.handleError);
    registerCallback(Dispatcher.ActionType.MENU_FILE_LOAD,
                     this.handleMenuFileLoad);
    registerCallback(Dispatcher.ActionType.NAV_BAR_CHUNK_REQUEST,
                     this.handleNavBarRequest);
    registerCallback(Dispatcher.ActionType.REQUEST_RESPONSE_ERROR,
                     this.handleRequestResponseError);
    registerCallback(Dispatcher.ActionType.REQUEST_RESPONSE_OK,
                     this.handleRequestResponseOk);
    registerCallback(Dispatcher.ActionType.REQUEST_START,
                     this.handleRequestStart);
    registerCallback(Dispatcher.ActionType.PREDICTION_CHUNK_REQUEST,
                     this.handlePredictionChunkRequest);
    registerCallback(Dispatcher.ActionType.PREDICTION_MODE_SELECTION,
                     this.handlePredictionModeSelection);
    registerCallback(Dispatcher.ActionType.PREDICTION_LABEL_SELECTION,
                     this.handlePredictionLabelSelection);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_GRIDLINES,
                     this.handleToolBarGridlines);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_HIGH_CUT,
                     this.handleToolBarHighCut);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_LOW_CUT,
                     this.handleToolBarLowCut);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_MONTAGE,
                     this.handleToolBarMontage);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_NEXT_CHUNK,
                     this.handleToolBarNextChunk);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_NEXT_SEC,
                     this.handleToolBarNextSec);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_NOTCH,
                     this.handleToolBarNotch);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_PREV_CHUNK,
                     this.handleToolBarPrevChunk);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_PREV_SEC,
                     this.handleToolBarPrevSec);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_SENSITIVITY,
                     this.handleToolBarSensitivity);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_ZOOM,
                     this.handleToolBarZoom);
    registerCallback(Dispatcher.ActionType.WARNING,
                     this.handleError);
    registerCallback(Dispatcher.ActionType.WINDOW_LOCATION_PENDING_REQUEST,
                     this.handleWindowLocationPendingRequest);

    this.logger_ = log.getLogger('eeg_modelling.eeg_viewer.Store');
  }

  /**
   * Registers a listener that triggers a callback when any of the registered
   * properties changes.
   * @param {!Array<!Property>} properties List of store properties.
   * @param {string} id The ID of the view registering.
   * @param {!Function} callback The function to call when the properties
   * change.
   */
  registerListener(properties, id, callback) {
    this.registeredListeners.push({
      properties: properties,
      id: id,
      callback: callback,
    });
  }

  /**
   * Emits snapshot of the Store to all registered views if it has changed.
   * @param {!StoreData} oldStoreData Copy of the store data before callback.
   */
  emitChange(oldStoreData) {
    const changedProperties = [];
    for (let prop in oldStoreData) {
      if (JSON.stringify(this.storeData[prop]) !=
          JSON.stringify(oldStoreData[prop])) {
        changedProperties.push(prop);
      }
    }

    if (!changedProperties || changedProperties.length === 0) {
      log.info(this.logger_, 'No property changed');
      return;
    }

    const changeId = `[${Date.now() % 10000}]`;

    log.info(this.logger_,
        `${changeId} Emitting chunk data store change for properties... ${changedProperties.toString()}`);

    this.registeredListeners.forEach((listener) => {
      const propertyTriggers = listener.properties.filter(
          prop => changedProperties.includes(prop));
      if (propertyTriggers.length > 0) {
        log.info(this.logger_, `${changeId} ... to ${listener.id} view (${propertyTriggers.toString()})`);
        listener.callback(Object.assign({}, this.storeData));
      }
    });
  }

  /**
   * Update the error property.
   * Add a timestamp so every new message received is different from the
   * previous one, so the error listeners are called every time.
   * @param {string} message New error message
   */
  updateError(message) {
    this.storeData.error = {
      message,
      timestamp: Date.now(),
    };
  }

  /**
   * Handles data from a REQUEST_RESPONSE_OK action.
   * @param {!DataResponse} data The data payload from the action.
   */
  handleRequestResponseOk(data) {
    const waveformChunk = data.getWaveformChunk();
    this.storeData.chunkGraphData = /** @type {?DataTableInput} */ (JSON.parse(
        assertString(waveformChunk.getWaveformDatatable())));
    this.storeData.channelIds =
        waveformChunk.getChannelDataIdsList()
            .map(
                channelDataId =>
                    this.convertChannelDataIdToIndexStr(channelDataId))
            .filter(channelStr => channelStr);
    this.storeData.samplingFreq = waveformChunk.getSamplingFreq();
    const waveformMeta = data.getWaveformMetadata();
    this.storeData.absStart = waveformMeta.getAbsStart();
    this.storeData.annotations = waveformMeta.getLabelsList().map((label) => {
      return {
        labelText: label.getLabelText(),
        startTime: label.getStartTime(),
        id: null,
      };
    });
    this.storeData.fileType = waveformMeta.getFileType();
    this.storeData.indexChannelMap = assertInstanceof(
        waveformMeta.getChannelDictMap(), JspbMap);
    this.storeData.numSecs = waveformMeta.getNumSecs();
    this.storeData.patientId = waveformMeta.getPatientId();
    this.storeData.sstableKey = waveformMeta.getSstableKey() || null;
    if (data.hasPredictionChunk() &&
        data.getPredictionChunk().getChunkStart() != null &&
        data.getPredictionChunk().getChunkDuration() != null) {
      const predictionChunk = data.getPredictionChunk();
      this.storeData.attributionMaps = predictionChunk.getAttributionDataMap();
      this.storeData.predictionChunkSize = assertNumber(
          predictionChunk.getChunkDuration());
      this.storeData.predictionChunkStart = assertNumber(
          predictionChunk.getChunkStart());
    } else {
      this.storeData.attributionMaps = null;
      this.storeData.predictionChunkSize = null;
      this.storeData.predictionChunkStart = null;
    }
    if (data.hasPredictionMetadata()) {
      const predictionMeta = data.getPredictionMetadata();
      this.storeData.chunkScores = predictionMeta.getChunkScoresList();
    } else {
      this.storeData.chunkScores = null;
    }

    const wasFirstLoad = this.storeData.loadingStatus === LoadingStatus.LOADING;
    this.storeData.loadingStatus =
        wasFirstLoad ? LoadingStatus.LOADED : LoadingStatus.RELOADED;
  }

  /**
   * Handles data from a WINDOW_LOCATION_PENDING_REQUEST action.
   * @param {!Dispatcher.FragmentData} data The data payload from the action.
   */
  handleWindowLocationPendingRequest(data) {
    responseProperties.forEach((prop) => {
      this.storeData[prop] = null;
    });

    /**
     * Parse a string to number.
     * If is not a number returns 0
     */
    const numberParser = (string) => Number(string) || 0;

    /**
     * Split a string separated by commas into an array.
     */
    const stringToArray = (str) => str ? str.split(',') : [];

    /**
     * Update a key from storeData with the value from incoming data.
     */
    const updateKey = (storeKey, parser = undefined) => {
      const dataKey = storeKey.toLowerCase();
      const newValue = data[dataKey];
      if (newValue && this.storeData[storeKey] != newValue) {
        this.storeData[storeKey] = parser ? parser(newValue) : newValue;
      }
    };

    updateKey('tfExSSTablePath');
    updateKey('predictionSSTablePath');
    updateKey('sstableKey');
    updateKey('edfPath');
    updateKey('tfExFilePath');
    updateKey('predictionFilePath');
    updateKey('chunkStart', numberParser);
    updateKey('chunkDuration', numberParser);
    updateKey('channelIds', stringToArray);
    updateKey('lowCut', numberParser);
    updateKey('highCut', numberParser);
    updateKey('notch', numberParser);
  }

  /**
   * Handles data from an ERROR action or WARNING action, that will modify
   * the error message. Use these actions for any common error or warning
   * (except special types, such as request response error).
   * @param {!Dispatcher.ErrorData} data The data payload from the action.
   */
  handleError(data) {
    this.updateError(data.message);
  }

  /**
   * Handles data from a REQUEST_RESPONSE_ERROR action that will modify the
   * error message and the loading status.
   * @param {!Dispatcher.ErrorData} data The data payload from the action.
   */
  handleRequestResponseError(data) {
    this.updateError(data.message);
    const wasFirstLoad = this.storeData.loadingStatus === LoadingStatus.LOADING;
    this.storeData.loadingStatus =
        wasFirstLoad ? LoadingStatus.NO_DATA : LoadingStatus.RELOADED;
  }

  /**
   * Handles data from a REQUEST_START action that will modify the loading
   * status.
   * @param {!Dispatcher.RequestStartData} data The data payload from the
   *     action.
   */
  handleRequestStart(data) {
    const isFirstLoad = this.storeData.loadingStatus === LoadingStatus.NO_DATA;
    this.storeData.loadingStatus = (isFirstLoad || data.fileParamDirty) ?
        LoadingStatus.LOADING :
        LoadingStatus.RELOADING;
  }

  /**
   * Handles data from a CHANGE_TYPING_STATUS action, that will update the
   * typing status in the store.
   */
  handleChangeTypingStatus(data) {
    this.storeData.isTyping = data.isTyping;
  }

  /**
   * Handles data from a TOOL_BAR_GRIDLINES action that will modify the time
   * scale.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handleToolBarGridlines(data) {
    assertNumber(data.selectedValue);
    this.storeData.timeScale = data.selectedValue;
  }

  /**
   * Handles data from a TOOL_BAR_HIGH_CUT action which will modify the high cut
   * filter parameter.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handleToolBarHighCut(data) {
    assertNumber(data.selectedValue);
    this.storeData.highCut = data.selectedValue;
  }

  /**
   * Handles data from a TOOL_BAR_LOW_CUT action which will modify the low cut
   * filter parameter.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handleToolBarLowCut(data) {
    assertNumber(data.selectedValue);
    this.storeData.lowCut = data.selectedValue;
  }

  /**
   * Handles data from a TOOL_BAR_MONTAGE action.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handleToolBarMontage(data) {
    assertArray(data.selectedValue);
    this.storeData.channelIds = data.selectedValue;
  }

  /**
   * Handles data from a TOOL_BAR_NEXT_CHUNK action which will modify the chunk
   * start.
   */
  handleToolBarNextChunk() {
    this.storeData.chunkStart = (this.storeData.chunkStart +
        this.storeData.chunkDuration);
  }

  /**
   * Handles data from a TOOL_BAR_NEXT_SEC action which will modify the chunk
   * start.
   */
  handleToolBarNextSec() {
    this.storeData.chunkStart = this.storeData.chunkStart + 1;
  }

  /**
   * Handles data from a TOOL_BAR_NOTCH action.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handleToolBarNotch(data) {
    assertNumber(data.selectedValue);
    this.storeData.notch = data.selectedValue;
  }

  /**
   * Handles data from a TOOL_BAR_PREV_CHUNK action which will modify the chunk
   * start.
   */
  handleToolBarPrevChunk() {
    this.storeData.chunkStart = (this.storeData.chunkStart -
        this.storeData.chunkDuration);
  }

  /**
   * Handles data from a TOOL_BAR_PREV_SEC action which will modify the chunk
   * start.
   */
  handleToolBarPrevSec() {
    this.storeData.chunkStart = this.storeData.chunkStart - 1;
  }

  /**
   * Handles data from a TOOL_BAR_SENSITIVITY action which will modify the
   * sensitivity.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handleToolBarSensitivity(data) {
    assertNumber(data.selectedValue);
    this.storeData.sensitivity = data.selectedValue;
  }

  /**
   * Handles data from a TOOL_BAR_ZOOM action which will modify the chunk
   * duration.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handleToolBarZoom(data) {
    assertNumber(data.selectedValue);
    this.storeData.chunkDuration = data.selectedValue;
  }

  /**
   * Handles data from a PREDICTION_CHUNK_REQUEST action which will modify the
   * chunk start.
   * @param {!Dispatcher.TimeData} data The data payload from the action.
   */
  handlePredictionChunkRequest(data) {
    this.storeData.chunkStart = Math.round(data.time);
  }

  /**
   * Handles data from a PREDICTION_MODE_SELECTION action which will modify the
   * prediction viewing mode.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handlePredictionModeSelection(data) {
    const mode = assertString(data.selectedValue);
    assert(Object.values(PredictionMode).includes(mode));
    this.storeData.predictionMode = /** @type {!PredictionMode} */(mode);
  }

  /**
   * Handles data from a PREDICTION_LABEL_SELECTION action which will modify the
   * label.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   */
  handlePredictionLabelSelection(data) {
    this.storeData.label = assertString(data.selectedValue);
  }

  /**
   * Handles data from an ANNOTATION_SELECTION action which will update the
   * chunk start.
   * @param {!Dispatcher.TimeData} data The data payload from the action.
   */
  handleAnnotationSelection(data) {
    this.storeData.chunkStart = Math.round(data.time -
        this.storeData.chunkDuration / 2);
  }

  /**
   * Handles data from a MENU_FILE_LOAD action which may update the file input
   * options.
   * @param {!Dispatcher.FileParamData} data The data payload from the action.
   */
  handleMenuFileLoad(data) {
    this.storeData.tfExSSTablePath = data.tfExSSTablePath || null;
    this.storeData.predictionSSTablePath = data.predictionSSTablePath || null;
    this.storeData.sstableKey = data.sstableKey || null;
    this.storeData.edfPath = data.edfPath || null;
    this.storeData.tfExFilePath = data.tfExFilePath || null;
    this.storeData.predictionFilePath = data.predictionFilePath || null;
  }

  /**
   * Handles data from a NAV_BAR_CHUNK_REQUEST action which will update the
   * chunk start.
   * @param {!Dispatcher.TimeData} data The data payload from the action.
   */
  handleNavBarRequest(data) {
    if (data.time || data.time === 0) {
      this.storeData.chunkStart = this.storeData.chunkDuration * Math.floor(
        data.time / this.storeData.chunkDuration);
    }
  }

  /**
   * Converts a ChannelDataId instance to a string format where multiple
   * channels are separated by a '-' character.
   * @param {!ChannelDataId} channelDataId ChannelDataId instance.
   * @return {?string} String format ChannelDataId.
   */
  convertChannelDataIdToIndexStr(channelDataId) {
    if (channelDataId.hasSingleChannel()) {
      return channelDataId.getSingleChannel().getIndex().toString();
    } else if (channelDataId.hasBipolarChannel()) {
      return [
        channelDataId.getBipolarChannel().getIndex().toString(),
        channelDataId.getBipolarChannel().getReferentialIndex().toString()
      ].join('-');
    } else {
      log.error(this.logger_, 'Empty ChannelDataId');
      return null;
    }
  }

  /**
   * Clips chunk time within [0, numSecs].
   */
  clipTime() {
    if (this.storeData.numSecs) {
      this.storeData.chunkStart = Math.min(this.storeData.chunkStart,
          this.storeData.numSecs - this.storeData.chunkDuration);
      this.storeData.chunkStart = Math.max(this.storeData.chunkStart, 0);
    }
  }

  /**
   * Creates a copy of the current StoreData state.
   * @return {!StoreData} Copy of the current state.
   */
  getStoreDataState() {
    return /** @type {!StoreData} */(JSON.parse(
        JSON.stringify(this.storeData)));
  }

  /**
   * Handles an action event dispatched to Stores.
   * @param {!Function} handler The function to handle the event data.
   * @param {!Object} data The data accompanying the action event.
   */
  callbackWrapper(handler, data) {
    const oldStoreData = this.getStoreDataState();
    handler(data);
    this.clipTime();
    this.emitChange(oldStoreData);
  }
}

goog.addSingletonGetter(Store);

exports = Store;
exports.StoreData = StoreData;
exports.ErrorInfo = ErrorInfo;
exports.Property = Property;
exports.PredictionMode = PredictionMode;
exports.LoadingStatus = LoadingStatus;
