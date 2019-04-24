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
const SimilarPatternsResponse = goog.require('proto.eeg_modelling.protos.SimilarPatternsResponse');
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
  GRAPH_POINT_CLICK: 'graphPointClick',
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
  SIMILAR_PATTERN_EDIT: 'similarPatternEdit',
  SIMILAR_PATTERN_ERROR: 'similarPatternError',
  SIMILAR_PATTERN_RESULT: 'similarPatternResult',
  SIMILAR_PATTERN_TARGET: 'similarPatternTarget',
  SSTABLE_KEY: 'sstableKey',
  TIMESCALE: 'timeScale',
  TFEX_FILE_PATH: 'tfExFilePath',
  TFEX_SSTABLE_PATH: 'tfExSSTablePath',
  WAVE_EVENTS: 'waveEvents',
  WAVE_EVENT_DRAFT: 'waveEventDraft',
};


/** @const {!Array<!Property>} */
const FileRequestProperties = [
  Property.TFEX_SSTABLE_PATH,
  Property.PREDICTION_SSTABLE_PATH,
  Property.SSTABLE_KEY,
  Property.EDF_PATH,
  Property.TFEX_FILE_PATH,
  Property.PREDICTION_FILE_PATH,
];

/** @const {!Array<!Property>} */
const NumberRequestProperties = [
  Property.CHUNK_START,
  Property.CHUNK_DURATION,
  Property.LOW_CUT,
  Property.HIGH_CUT,
  Property.NOTCH,
];

/** @const {!Array<!Property>} */
const ListRequestProperties = [
  Property.CHANNEL_IDS,
];

/** @const {!Array<!Property>} */
const RequestProperties = [
  ...NumberRequestProperties,
  ...FileRequestProperties,
  ...ListRequestProperties,
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
 *   id: (number|undefined),
 *   labelText: string,
 *   startTime: number,
 *   duration: number,
 *   channelList: !Array<string>,
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
 * @typedef {{
 *   timeValue: number,
 *   channelName: string,
 *   xPos: number,
 *   yPos: number,
 * }}
 */
let GraphDataPoint;

/**
 * @typedef {{
 *   score: number,
 *   startTime: number,
 *   duration: number,
 *   channelList: !Array<string>,
 * }}
 */
let SimilarPattern;

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
 *   graphPointClick: ?GraphDataPoint,
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
 *   similarPatternEdit: ?Annotation,
 *   similarPatternError: ?ErrorInfo,
 *   similarPatternResult: ?Array<!SimilarPattern>,
 *   similarPatternTarget: ?Annotation,
 *   seriesHeight: number,
 *   sensitivity: number,
 *   sstableKey: ?string,
 *   timeScale: number,
 *   tfExSSTablePath: ?string,
 *   tfExFilePath: ?string,
 *   waveEvents: !Array<!Annotation>,
 *   waveEventDraft: ?Annotation,
 * }}
 */
let StoreData;

/**
 * @typedef {{
 *   absStart: (?number|undefined),
 *   annotations: (?Array<!Annotation>|undefined),
 *   attributionMaps: (?JspbMap<string, !AttributionMap>|undefined),
 *   channelIds: (?Array<string>|undefined),
 *   chunkDuration: (number|undefined),
 *   chunkGraphData: (?DataTableInput|undefined),
 *   chunkScores: (?Array<!ChunkScoreData>|undefined),
 *   chunkStart: (number|undefined),
 *   edfPath: (?string|undefined),
 *   error: (?ErrorInfo|undefined),
 *   fileType: (?string|undefined),
 *   graphPointClick: (?GraphDataPoint|undefined),
 *   highCut: (number|undefined),
 *   indexChannelMap: (?JspbMap<string, string>|undefined),
 *   isTyping: (boolean|undefined),
 *   label: (string|undefined),
 *   loadingStatus: (!LoadingStatus|undefined),
 *   lowCut: (number|undefined),
 *   notch: (number|undefined),
 *   numSecs: (?number|undefined),
 *   patientId: (?string|undefined),
 *   predictionChunkSize: (?number|undefined),
 *   predictionChunkStart: (?number|undefined),
 *   predictionFilePath: (?string|undefined),
 *   predictionMode: (!PredictionMode|undefined),
 *   predictionSSTablePath: (?string|undefined),
 *   samplingFreq: (?number|undefined),
 *   similarPatternEdit: (?Annotation|undefined),
 *   similarPatternError: (?ErrorInfo|undefined),
 *   similarPatternResult: (?Array<!SimilarPattern>|undefined),
 *   similarPatternTarget: (?Annotation|undefined),
 *   seriesHeight: (number|undefined),
 *   sensitivity: (number|undefined),
 *   sstableKey: (?string|undefined),
 *   timeScale: (number|undefined),
 *   tfExSSTablePath: (?string|undefined),
 *   tfExFilePath: (?string|undefined),
 *   waveEvents: (!Array<!Annotation>|undefined),
 *   waveEventDraft: (?Annotation|undefined),
 * }}
 */
let PartialStoreData;

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
      graphPointClick: null,
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
      similarPatternEdit: null,
      similarPatternError: null,
      similarPatternResult: null,
      similarPatternTarget: null,
      seriesHeight: 100,
      sensitivity: 5,
      sstableKey: null,
      timeScale: 1,
      tfExSSTablePath: null,
      tfExFilePath: null,
      waveEvents: [],
      waveEventDraft: null,
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

    registerCallback(Dispatcher.ActionType.ADD_WAVE_EVENT,
                     this.handleAddWaveEvent);
    registerCallback(Dispatcher.ActionType.ANNOTATION_SELECTION,
                     this.handleAnnotationSelection);
    registerCallback(Dispatcher.ActionType.CHANGE_TYPING_STATUS,
                     this.handleChangeTypingStatus);
    registerCallback(Dispatcher.ActionType.CLICK_GRAPH,
                     this.handleClickGraph);
    registerCallback(Dispatcher.ActionType.DELETE_WAVE_EVENT,
                     this.handleDeleteWaveEvent);
    registerCallback(Dispatcher.ActionType.ERROR,
                     this.handleError);
    registerCallback(Dispatcher.ActionType.MENU_FILE_LOAD,
                     this.handleMenuFileLoad);
    registerCallback(Dispatcher.ActionType.NAVIGATE_TO_SPAN,
                     this.handleNavigateToSpan);
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
    registerCallback(Dispatcher.ActionType.SEARCH_SIMILAR_REQUEST,
                     this.handleSimilarPatternsRequest);
    registerCallback(Dispatcher.ActionType.SEARCH_SIMILAR_RESPONSE_OK,
                     this.handleSimilarPatternsResponseOk);
    registerCallback(Dispatcher.ActionType.SEARCH_SIMILAR_RESPONSE_ERROR,
                     this.handleSimilarPatternsResponseError);
    registerCallback(Dispatcher.ActionType.SIMILAR_PATTERN_ACCEPT,
                     this.handleSimilarPatternAccept);
    registerCallback(Dispatcher.ActionType.SIMILAR_PATTERN_EDIT,
                     this.handleSimilarPatternEdit);
    registerCallback(Dispatcher.ActionType.SIMILAR_PATTERN_REJECT,
                     this.handleSimilarPatternReject);
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
    registerCallback(Dispatcher.ActionType.TOOL_BAR_NOTCH,
                     this.handleToolBarNotch);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_PREV_CHUNK,
                     this.handleToolBarPrevChunk);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_SENSITIVITY,
                     this.handleToolBarSensitivity);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_SHIFT_SECS,
                     this.handleToolBarShiftSecs);
    registerCallback(Dispatcher.ActionType.TOOL_BAR_ZOOM,
                     this.handleToolBarZoom);
    registerCallback(Dispatcher.ActionType.UPDATE_WAVE_EVENT_DRAFT,
                     this.handleUpdateWaveEventDraft);
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
   * @param {!PartialStoreData} newStoreData Object with store data to update.
   */
  emitChange(newStoreData) {
    const changedProperties = [];
    for (let prop in newStoreData) {
      const oldValue = this.storeData[prop];
      const newValue = newStoreData[prop];
      if (JSON.stringify(oldValue) !== JSON.stringify(newValue)) {
        changedProperties.push(prop);
        this.storeData[prop] = newValue;
      }
    }

    if (!changedProperties || changedProperties.length === 0) {
      log.info(this.logger_, 'No property changed');
      return;
    }

    const changeId = `[${Date.now() % 10000}]`;

    log.info(
        this.logger_,
        `${changeId} Emitting chunk data store change for properties... ${
            changedProperties.toString()}`);

    this.registeredListeners.forEach((listener) => {
      const propertyTriggers = listener.properties.filter(
          prop => changedProperties.includes(prop));
      if (propertyTriggers.length > 0) {
        log.info(
            this.logger_,
            `${changeId} ... to ${listener.id} view (${
                propertyTriggers.toString()})`);
        listener.callback(Object.assign({}, this.storeData), changedProperties);
      }
    });
  }

  /**
   * Update the error property.
   * Add a timestamp so every new message received is different from the
   * previous one, so the error listeners are called every time.
   * @param {string} message New error message
   * @return {!ErrorInfo} New error info.
   */
  newError(message) {
    return {
      message,
      timestamp: Date.now(),
    };
  }

  /**
   * Adds a wave event to the waveEvents property of the store.
   * Sets the id of the wave event, and returns a copy of the array (does not
   * modify in place).
   * @param {!Annotation} waveEvent New wave event to add.
   * @return {!Array<!Annotation>} Copy of the array with the new item added.
   * @private
   */
  addWaveEvent_(waveEvent) {
    const length = this.storeData.waveEvents.length;
    const lastId = length > 0 ? this.storeData.waveEvents[length - 1].id : 0;
    return [
      ...this.storeData.waveEvents,
      Object.assign({}, waveEvent, {
        id: lastId + 1,
      }),
    ];
  }

  /**
   * Handles data from an ADD_WAVE_EVENT action, which will add a new wave
   * event to the list.
   * @param {!Annotation} waveEvent The new wave event.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleAddWaveEvent(waveEvent) {
    return {
      waveEvents: this.addWaveEvent_(waveEvent),
      waveEventDraft: null,
    };
  }

  /**
   * Handles data from an UPDATE_WAVE_EVENT_DRAFT action.
   * @param {?Annotation} waveEventDraft The draft wave event.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleUpdateWaveEventDraft(waveEventDraft) {
    return {
      waveEventDraft: waveEventDraft &&
          /** @type {!Annotation} */ (Object.assign({}, waveEventDraft)),
    };
  }

  /**
   * Handles data from an DELETE_WAVE_EVENT action, which will delete an event.
   * @param {!Dispatcher.IdData} data The wave event to delete.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleDeleteWaveEvent(data) {
    return {
      waveEvents: this.storeData.waveEvents.filter(wave => wave.id !== data.id),
    };
  }

  /**
   * Handles data from an CLICK_GRAPH action, which will save the click data.
   * @param {!GraphDataPoint} data The click information.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleClickGraph(data) {
    return {
      graphPointClick: data,
    };
  }

  /**
   * Handles data from a REQUEST_RESPONSE_OK action.
   * @param {!DataResponse} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleRequestResponseOk(data) {
    const /** !PartialStoreData */ newStoreData = {};

    const waveformChunk = data.getWaveformChunk();
    newStoreData.chunkGraphData = /** @type {!DataTableInput} */ (JSON.parse(
        assertString(waveformChunk.getWaveformDatatable())));
    newStoreData.channelIds =
        waveformChunk.getChannelDataIdsList()
            .map(
                channelDataId =>
                    this.convertChannelDataIdToIndexStr(channelDataId))
            .filter(channelStr => channelStr);
    newStoreData.samplingFreq = waveformChunk.getSamplingFreq();
    const waveformMeta = data.getWaveformMetadata();
    newStoreData.absStart = waveformMeta.getAbsStart();
    newStoreData.annotations = waveformMeta.getLabelsList().map((label) => {
      return {
        labelText: label.getLabelText(),
        startTime: label.getStartTime(),
        duration: 0,
        channelList: [],
      };
    });
    newStoreData.fileType = waveformMeta.getFileType();
    newStoreData.indexChannelMap = assertInstanceof(
        waveformMeta.getChannelDictMap(), JspbMap);
    newStoreData.numSecs = waveformMeta.getNumSecs();
    newStoreData.patientId = waveformMeta.getPatientId();
    newStoreData.sstableKey = waveformMeta.getSstableKey() || null;
    if (data.hasPredictionChunk() &&
        data.getPredictionChunk().getChunkStart() != null &&
        data.getPredictionChunk().getChunkDuration() != null) {
      const predictionChunk = data.getPredictionChunk();
      newStoreData.attributionMaps = predictionChunk.getAttributionDataMap();
      newStoreData.predictionChunkSize = assertNumber(
          predictionChunk.getChunkDuration());
      newStoreData.predictionChunkStart = assertNumber(
          predictionChunk.getChunkStart());
    } else {
      newStoreData.attributionMaps = null;
      newStoreData.predictionChunkSize = null;
      newStoreData.predictionChunkStart = null;
    }
    if (data.hasPredictionMetadata()) {
      const predictionMeta = data.getPredictionMetadata();
      newStoreData.chunkScores = predictionMeta.getChunkScoresList();
    } else {
      newStoreData.chunkScores = null;
    }

    const wasFirstLoad = this.storeData.loadingStatus === LoadingStatus.LOADING;
    newStoreData.loadingStatus =
        wasFirstLoad ? LoadingStatus.LOADED : LoadingStatus.RELOADED;

    return newStoreData;
  }

  /**
   * Handles data from a WINDOW_LOCATION_PENDING_REQUEST action.
   * @param {!Dispatcher.FragmentData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleWindowLocationPendingRequest(data) {
    const /** !PartialStoreData */ newStoreData = {};

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
     * Update a key from newStoreData with the value from incoming data.
     */
    const updateKey = (storeKey, parser = undefined) => {
      const dataKey = storeKey.toLowerCase();
      const newValue = data[dataKey];
      if (newValue && this.storeData[storeKey] != newValue) {
        newStoreData[storeKey] = parser ? parser(newValue) : newValue;
      }
    };

    updateKey(Property.TFEX_SSTABLE_PATH);
    updateKey(Property.PREDICTION_SSTABLE_PATH);
    updateKey(Property.SSTABLE_KEY);
    updateKey(Property.EDF_PATH);
    updateKey(Property.TFEX_FILE_PATH);
    updateKey(Property.PREDICTION_FILE_PATH);
    updateKey(Property.CHUNK_START, numberParser);
    updateKey(Property.CHUNK_DURATION, numberParser);
    updateKey(Property.CHANNEL_IDS, stringToArray);
    updateKey(Property.LOW_CUT, numberParser);
    updateKey(Property.HIGH_CUT, numberParser);
    updateKey(Property.NOTCH, numberParser);

    return newStoreData;
  }

  /**
   * Handles data from an ERROR action or WARNING action, that will modify
   * the error message. Use these actions for any common error or warning
   * (except special types, such as request response error).
   * @param {!Dispatcher.ErrorData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleError(data) {
    return {
      error: this.newError(data.message),
    };
  }

  /**
   * Handles data from a REQUEST_RESPONSE_ERROR action that will modify the
   * error message and the loading status.
   * @param {!Dispatcher.ErrorData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleRequestResponseError(data) {
    const error = this.newError(data.message);
    const wasFirstLoad = this.storeData.loadingStatus === LoadingStatus.LOADING;
    const loadingStatus =
        wasFirstLoad ? LoadingStatus.NO_DATA : LoadingStatus.RELOADED;
    return {
      error,
      loadingStatus,
    };
  }

  /**
   * Handles data from a REQUEST_START action that will modify the loading
   * status.
   * @param {!Dispatcher.RequestStartData} data The data payload from the
   *     action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleRequestStart(data) {
    const isFirstLoad = this.storeData.loadingStatus === LoadingStatus.NO_DATA;
    const loadingStatus = (isFirstLoad || data.fileParamDirty) ?
        LoadingStatus.LOADING :
        LoadingStatus.RELOADING;
    return {
      loadingStatus,
    };
  }

  /**
   * Handles data from a CHANGE_TYPING_STATUS action, that will update the
   * typing status in the store.
   * @param {!Dispatcher.IsTypingData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleChangeTypingStatus(data) {
    return {
      isTyping: data.isTyping,
    };
  }

  /**
   * Handles data from a SEARCH_SIMILAR_REQUEST action, which will save the
   * similar pattern target and trigger a request.
   * @param {!Annotation} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleSimilarPatternsRequest(data) {
    return {
      similarPatternTarget: data,
    };
  }

  /**
   * Handles data from a SEARCH_SIMILAR_RESPONSE_OK action, which will save
   * the similar patterns found.
   * @param {!SimilarPatternsResponse} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleSimilarPatternsResponseOk(data) {
    const targetPattern = this.storeData.similarPatternTarget;
    const similarPatternResult = data.getSimilarPatternsList().map(
        (similarPattern) => /** @type {!SimilarPattern} */ (Object.assign(
            {},
            similarPattern.toObject(),
            {
              channelList: [...targetPattern.channelList],
            },
            )));
    return {
      similarPatternError: null,
      similarPatternResult,
    };
  }

  /**
   * Handles data from a SEARCH_SIMILAR_RESPONSE_ERROR action, which will update
   * the error and result of the request.
   * @param {!Dispatcher.ErrorData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleSimilarPatternsResponseError(data) {
    return {
      similarPatternError: this.newError(data.message),
      similarPatternResult: null,
    };
  }

  /**
   * Removes a similar pattern from the similarPatternResult property.
   * Returns a copy of the array, does not modify in place.
   * @param {!SimilarPattern} removePattern Similar pattern to remove.
   * @return {!Array<!SimilarPattern>} Copy of the similarPatternResult, with
   *     the pattern removed.
   * @private
   */
  removeSimilarPattern_(removePattern) {
    return this.storeData.similarPatternResult.filter(
        (pattern) => pattern.startTime !== removePattern.startTime);
  }

  /**
   * Handles data from a SIMILAR_PATTERN_ACCEPT action, which will move the
   * accepted pattern to the wave events list, and remove it from the similar
   * patterns list.
   * @param {!SimilarPattern} data The similar pattern to accept.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleSimilarPatternAccept(data) {
    const targetPattern = this.storeData.similarPatternTarget;
    const similarPatternResult = this.removeSimilarPattern_(data);
    const waveEvents = this.addWaveEvent_({
      labelText: targetPattern.labelText,
      startTime: data.startTime,
      duration: data.duration,
      channelList: data.channelList,
    });
    return {
      waveEvents,
      similarPatternResult,
    };
  }

  /**
   * Handles data from a SIMILAR_PATTERN_EDIT action, which will save the
   * pattern and remove it from the similar patterns list.
   * @param {!SimilarPattern} data The similar pattern to edit.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleSimilarPatternEdit(data) {
    return {
      similarPatternEdit: {
        labelText: this.storeData.similarPatternTarget.labelText,
        startTime: data.startTime,
        duration: data.duration,
        channelList: data.channelList,
      },
      similarPatternResult: this.removeSimilarPattern_(data),
    };
  }

  /**
   * Handles data from a SIMILAR_PATTERN_REJECT action, which will remove the
   * similar pattern from the list.
   * @param {!SimilarPattern} data The similar pattern to reject.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleSimilarPatternReject(data) {
    return {
      similarPatternResult: this.removeSimilarPattern_(data),
    };
  }

  /**
   * Handles data from a TOOL_BAR_GRIDLINES action that will modify the time
   * scale.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarGridlines(data) {
    assertNumber(data.selectedValue);
    return {
      timeScale: data.selectedValue,
    };
  }

  /**
   * Handles data from a TOOL_BAR_HIGH_CUT action which will modify the high cut
   * filter parameter.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarHighCut(data) {
    assertNumber(data.selectedValue);
    return {
      highCut: data.selectedValue,
    };
  }

  /**
   * Handles data from a TOOL_BAR_LOW_CUT action which will modify the low cut
   * filter parameter.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarLowCut(data) {
    assertNumber(data.selectedValue);
    return {
      lowCut: data.selectedValue,
    };
  }

  /**
   * Handles data from a TOOL_BAR_MONTAGE action.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarMontage(data) {
    assertArray(data.selectedValue);
    return {
      channelIds: data.selectedValue,
    };
  }

  /**
   * Handles data from a TOOL_BAR_NEXT_CHUNK action which will modify the chunk
   * start.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarNextChunk() {
    return {
      chunkStart: this.storeData.chunkStart + this.storeData.chunkDuration,
    };
  }

  /**
   * Handles data from a TOOL_BAR_SHIFT_SECS action which will move the chunk
   * start an amount of seconds, either forward or backwards.
   * @param {!Dispatcher.TimeData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarShiftSecs(data) {
    return {
      chunkStart: this.storeData.chunkStart + data.time,
    };
  }

  /**
   * Handles data from a TOOL_BAR_NOTCH action.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarNotch(data) {
    assertNumber(data.selectedValue);
    return {
      notch: data.selectedValue,
    };
  }

  /**
   * Handles data from a TOOL_BAR_PREV_CHUNK action which will modify the chunk
   * start.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarPrevChunk() {
    return {
      chunkStart: this.storeData.chunkStart - this.storeData.chunkDuration,
    };
  }

  /**
   * Handles data from a TOOL_BAR_SENSITIVITY action which will modify the
   * sensitivity.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarSensitivity(data) {
    assertNumber(data.selectedValue);
    return {
      sensitivity: data.selectedValue,
    };
  }

  /**
   * Handles data from a TOOL_BAR_ZOOM action which will modify the chunk
   * duration.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handleToolBarZoom(data) {
    assertNumber(data.selectedValue);
    return {
      chunkDuration: data.selectedValue,
    };
  }

  /**
   * Handles data from a PREDICTION_CHUNK_REQUEST action which will modify the
   * chunk start.
   * @param {!Dispatcher.TimeData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handlePredictionChunkRequest(data) {
    return {
      chunkStart: Math.round(data.time),
    };
  }

  /**
   * Handles data from a PREDICTION_MODE_SELECTION action which will modify the
   * prediction viewing mode.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handlePredictionModeSelection(data) {
    const mode = assertString(data.selectedValue);
    assert(Object.values(PredictionMode).includes(mode));
    return {
      predictionMode: /** @type {!PredictionMode} */(mode),
    };
  }

  /**
   * Handles data from a PREDICTION_LABEL_SELECTION action which will modify the
   * label.
   * @param {!Dispatcher.SelectionData} data The data payload from the action.
   * @return {!PartialStoreData} store data with changed properties.
   */
  handlePredictionLabelSelection(data) {
    return {
      label: assertString(data.selectedValue),
    };
  }

  /**
   * Handles data from an ANNOTATION_SELECTION action which will update the
   * chunk start.
   * @param {!Dispatcher.TimeData} data The data payload from the action.
   * @return {?PartialStoreData} store data with changed properties.
   */
  handleAnnotationSelection(data) {
    return {
      chunkStart: Math.round(data.time - this.storeData.chunkDuration / 2),
    };
  }

  /**
   * Handles data from a MENU_FILE_LOAD action which may update the file input
   * options.
   * @param {!Dispatcher.FileParamData} data The data payload from the action.
   * @return {?PartialStoreData} store data with changed properties.
   */
  handleMenuFileLoad(data) {
    return {
      tfExSSTablePath: data.tfExSSTablePath || null,
      predictionSSTablePath: data.predictionSSTablePath || null,
      sstableKey: data.sstableKey || null,
      edfPath: data.edfPath || null,
      tfExFilePath: data.tfExFilePath || null,
      predictionFilePath: data.predictionFilePath || null,
    };
  }

  /**
   * Handles data from a NAV_BAR_CHUNK_REQUEST action which will update the
   * chunk start.
   * @param {!Dispatcher.TimeData} data The data payload from the action.
   * @return {?PartialStoreData} store data with changed properties.
   */
  handleNavBarRequest(data) {
    if (!goog.isNumber(data.time)) {
      return null;
    }
    return {
      chunkStart: Math.round(data.time - this.storeData.chunkDuration / 2),
    };
  }

  /**
   * Handles data from a NAVIGATE_TO_SPAN action which will update the
   * chunk start.
   * If the span is shorter than chunkDuration, it will try to leave the span
   * in the middle of the viewport.
   * If not, it will try to leave the start of the span in the middle of the
   * viewport.
   * @param {!Dispatcher.TimeSpanData} data The data payload from the action.
   * @return {?PartialStoreData} store data with changed properties.
   */
  handleNavigateToSpan(data) {
    let viewportCenter;
    if (data.duration < this.storeData.chunkDuration) {
      viewportCenter = data.startTime + data.duration / 2;
    } else {
      viewportCenter = data.startTime;
    }
    const chunkStart = viewportCenter - this.storeData.chunkDuration / 2;
    return {
      chunkStart: Math.round(chunkStart),
    };
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
   * Clips chunk start within [0, numSecs].
   * @param {!PartialStoreData} newStoreData New store data.
   */
  clipChunkStart(newStoreData) {
    if (!newStoreData || newStoreData.chunkStart == null) {
      return;
    }

    const numSecs = newStoreData.numSecs != null ?
        newStoreData.numSecs : this.storeData.numSecs;
    const chunkDuration = newStoreData.chunkDuration != null ?
        newStoreData.chunkDuration : this.storeData.chunkDuration;

    if (numSecs) {
      newStoreData.chunkStart =
          Math.min(newStoreData.chunkStart, numSecs - chunkDuration);
      newStoreData.chunkStart = Math.max(newStoreData.chunkStart, 0);
    }
  }

  /**
   * Handles an action event dispatched to Stores.
   * @param {!Function} handler The function to handle the event data.
   * @param {!Object} data The data accompanying the action event.
   */
  callbackWrapper(handler, data) {
    const newStoreData = handler(data);
    this.clipChunkStart(newStoreData);
    this.emitChange(newStoreData);
  }
}

goog.addSingletonGetter(Store);

exports = Store;
exports.StoreData = StoreData;
exports.Annotation = Annotation;
exports.SimilarPattern = SimilarPattern;
exports.ErrorInfo = ErrorInfo;
exports.Property = Property;
exports.PredictionMode = PredictionMode;
exports.LoadingStatus = LoadingStatus;
exports.FileRequestProperties = FileRequestProperties;
exports.NumberRequestProperties = NumberRequestProperties;
exports.ListRequestProperties = ListRequestProperties;
exports.RequestProperties = RequestProperties;
