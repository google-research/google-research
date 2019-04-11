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

goog.module('eeg_modelling.eeg_viewer.Dispatcher');

const log = goog.require('goog.log');

/**
 * Action types supported by the viewer.
 * @enum {string}
 */
const ActionType = {
  ADD_WAVE_EVENT: 'addWaveEvent',
  ANNOTATION_SELECTION: 'annotationSelection',
  CHANGE_TYPING_STATUS: 'changeTypingStatus',
  ERROR: 'error',
  MENU_FILE_LOAD: 'menuFileLoad',
  NAV_BAR_CHUNK_REQUEST: 'navBarChunkRequest',
  PREDICTION_CHUNK_REQUEST: 'predictionChunkRequest',
  PREDICTION_MODE_SELECTION: 'predictionModeSelection',
  PREDICTION_LABEL_SELECTION: 'predictionLabelSelection',
  REQUEST_RESPONSE_ERROR: 'requestResponseError',
  REQUEST_RESPONSE_OK: 'requestResponseOk',
  REQUEST_START: 'requestStart',
  TOOL_BAR_GRIDLINES: 'toolBarGridlines',
  TOOL_BAR_HIGH_CUT: 'toolBarHighCut',
  TOOL_BAR_LOW_CUT: 'toolBarLowCut',
  TOOL_BAR_MONTAGE: 'toolBarMontage',
  TOOL_BAR_NEXT_CHUNK: 'toolBarNextChunk',
  TOOL_BAR_NEXT_SEC: 'toolBarNextSec',
  TOOL_BAR_NOTCH: 'toolBarNotch',
  TOOL_BAR_PREV_CHUNK: 'toolBarPrevChunk',
  TOOL_BAR_PREV_SEC: 'toolBarPrevSec',
  TOOL_BAR_SENSITIVITY: 'toolBarSensitivity',
  TOOL_BAR_ZOOM: 'toolBarZoom',
  WARNING: 'warning',
  WINDOW_LOCATION_PENDING_REQUEST: 'windowLocationPendingRequest',
};

/**
 * @typedef {{
 *   actionType: !ActionType,
 *   data: ?Object,
 * }}
 */
let ActionEvent;

/**
 * @typedef {{
 *   time: number
 * }}
 */
let TimeData;

/**
 * @typedef {{
 *   tfExSSTablePath: ?string,
 *   predictionSSTablePath: ?string,
 *   sstableKey: ?string,
 *   edfPath: ?string,
 *   edfSegmentIndex: ?number,
 *   tfExFilePath: ?string,
 *   predictionFilePath: ?string,
 * }}
 */
let FileParamData;

/**
 * @typedef {{
 *   chunkduration: ?number,
 *   chunkstart: ?number,
 *   edfpath: ?string,
 *   edfsegmentindex: ?number,
 *   tfexsstablepath: ?string,
 *   predictionsstablepath: ?string,
 *   sstablekey: ?string,
 *   tfexfilepath: ?string,
 *   predictionfilepath: ?string,
 *   channelIds: ?Array<string>,
 *   highcut: ?number,
 *   lowcut: ?number,
 *   notch: ?number,
 * }}
 */
let FragmentData;

/**
 * @typedef {{
 *   message: string,
 * }}
 */
let ErrorData;

/**
 * @typedef {{
 *   selectedValue: (string|number|!Array<string>),
 * }}
 */
let SelectionData;

/**
 * @typedef {{
 *   fileParamDirty: boolean,
 * }}
 */
let RequestStartData;

/**
 * @typedef {{
 *   isTyping: boolean,
 * }}
 */
let IsTypingData;


class Dispatcher {

  constructor() {
    this.callbacks = {};
    Object.values(ActionType).forEach((actionType) => {
      this.callbacks[actionType] = [];
    });
    this.logger_ = log.getLogger('eeg_modelling.eeg_viewer.Dispatcher');
  }

  /**
   * Adds a Store to the list of Stores the Dispatcher will send actions to.
   * @param {!ActionType} actionType The action type under which to register the
   * callback.
   * @param {!Function} callback A callback to run when an event of the given
   * action type is triggered.
   */
  registerCallback(actionType, callback) {
    this.callbacks[actionType].push(callback);
  }

  /**
   * Sends an action event to all registered Stores.
   * @param {!ActionEvent} actionEvent An event with a type and a payload of
   * data that may affect Store state.
   */
  sendAction(actionEvent) {
    log.info(this.logger_, 'Sending action type: ' + actionEvent.actionType);
    this.callbacks[actionEvent.actionType].forEach((callback) => {
      callback(actionEvent.data);
    });
  }
}

goog.addSingletonGetter(Dispatcher);

exports = Dispatcher;
exports.ActionType = ActionType;
exports.FragmentData = FragmentData;
exports.TimeData = TimeData;
exports.FileParamData = FileParamData;
exports.ErrorData = ErrorData;
exports.SelectionData = SelectionData;
exports.RequestStartData = RequestStartData;
exports.IsTypingData = IsTypingData;
