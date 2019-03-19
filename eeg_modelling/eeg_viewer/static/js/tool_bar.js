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

goog.module('eeg_modelling.eeg_viewer.ToolBar');

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const EventType = goog.require('goog.events.EventType');
const Keys = goog.require('goog.events.Keys');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const dom = goog.require('goog.dom');
const log = goog.require('goog.log');
const {assertInstanceof, assertNumber} = goog.require('goog.asserts');

// Montages are a standard format for clinicians to view EEG signals in.
// Each list element is a standard (A-B), and each standard consists of the
// difference between the signals from two leads (A, B) placed on the scalp.
// Some leads are referred to by multiple codes (C|D), but only one will exist
// in one file.
//
// See this website for more info about EEG montages.
// http://eegatlas-online.com/index.php/en/montages
const MONTAGES = {};
MONTAGES['longitudinal bipolar'] = [
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP1-F7', 'F7-T3|T7', 'T3|T7-T5|P7', 'T5|P7-O1',
    'FP2-F8', 'F8-T4|T8', 'T4|T8-T|T8', 'T6|P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'EKG1-EKG2'
];
MONTAGES['transverse bipolar'] = [
    'F7-FP1', 'FP1-FP2', 'FP2-F8',
    'F7-F3', 'F3-FZ', 'FZ-F4', 'F4-F8',
    'A1-T3|T7', 'T3|T7-C3', 'C3-CZ', 'CZ-C4', 'C4-T4|T8', 'T4|T8-A2',
    'T5|P7-P3', 'P3-PZ', 'PZ-P4', 'P4-T6|P8',
    'T5|P7-O1', 'O1-O2', 'O2-T6|P8',
    'EKG1-EKG2'
];
MONTAGES['referential'] = [
    'FP1-CZ', 'F3-CZ', 'C3-CZ', 'P3-CZ', 'O1-CZ',
    'FP2-CZ', 'F4-CZ', 'C4-CZ', 'P4-CZ', 'O2-CZ',
    'F7-CZ', 'T3|T7-CZ', 'T5|P7-CZ',
    'F8-CZ', 'T4|T8-CZ', 'T6|P8-CZ',
    'FZ-CZ', 'PZ-CZ',
    'A1-CZ', 'A2-CZ',
    'EKG1-EKG2'
];
MONTAGES['circumferential'] = [
    'O1-T5|P7', 'T5|P7-T3|T7', 'T3|T7-F7', 'F7-FP1', 'FP1-FP2', 'FP2-F8',
    'F8-T4|T8', 'T4|T8-T6|P8', 'T6|P8-O2',
    'FP1-F7', 'F7-A1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FZ-CZ', 'CZ-PZ',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-A2',
    'EKG1-EKG2'
];
MONTAGES['reverse circumferential'] = [
    'FP1-F7', 'F7-T3|T7', 'T3|T7-T5|P7', 'T5|P7-O1', 'O1-O2', 'O2-T6|P8',
    'T6|P8-T4|T8', 'T4|T8-F8', 'F8-FP2',
    'FP1-F7', 'F7-A1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FZ-CZ', 'CZ-PZ',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-A2',
    'EKG1-EKG2'
];

/**
 * @typedef {{
 *   indexStrList: !Array<string>,
 *   missingChannelList: ?Array<string>,
 * }}
 */
let MontageInfo;

/**
 * Asserts that an element is an HTMLButtonElement.
 * @param {?Element} element HTML element.
 * @return {!HTMLButtonElement} The HTML button.
 */
function assertButton(element) {
  return assertInstanceof(element, HTMLButtonElement);
}


class ToolBar {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will enable/disable the buttons in the tool bar
    // based on the HTTP request status and the position of the viewport
    // timespan in the full file.
    store.registerListener([Store.Property.CHUNK_START,
        Store.Property.CHUNK_DURATION, Store.Property.NUM_SECS,
        Store.Property.CHUNK_GRAPH_DATA],
        'ToolBar', (store) => this.handleChunkNavigation(store));
    // This listener callback will initialize the handlers for the montage drop
    // down menu options with the feature list to request.
    store.registerListener(
        [Store.Property.INDEX_CHANNEL_MAP], 'ToolBar',
        (store) => this.handleIndexChannelMap(store));
    // This listener callback will subscribe a handler for the
    // window.onkeydown event, only when the user is not typing
    store.registerListener(
        [Store.Property.IS_TYPING], 'ToolBar',
        (store) => this.handleChangeTypingStatus(store));

    /** @private {?string} */
    this.openedMenuPrefix_ = null;

    /** @private {?function(!Event):*} */
    this.keyPressHandler_ = null;
  }

  getMontages() {
    return MONTAGES;
  }

  /**
   * @private
   * Opens or closes a menu, by changing the button color and toggling the menu
   * panel.
   * @param {string} menuPrefix name of the menu to toggle.
   * @param {boolean} open whether or not to open the menu.
   */
  toggleMenuButtonAndPanel_(menuPrefix, open) {
    document.getElementById(`${menuPrefix}-button`).classList.toggle('mdl-button--accent', open);
    document.getElementById(`${menuPrefix}-panel`).classList.toggle('hidden', !open);
  }

  /**
   * Handles click on one of the side menus.
   */
  toggleMenu(menuPrefix) {
    const prevOpenedMenuPrefix = this.openedMenuPrefix_;

    if (prevOpenedMenuPrefix) {
      this.toggleMenuButtonAndPanel_(prevOpenedMenuPrefix, false);
      this.openedMenuPrefix_ = null;
    }

    if (prevOpenedMenuPrefix !== menuPrefix) {
      this.toggleMenuButtonAndPanel_(menuPrefix, true);
      this.openedMenuPrefix_ = menuPrefix;
    }
  }

  /**
   * Subscribes or unsubscribes a onkeydown Event handler.
   * @param {!Store.StoreData} store Store object.
   */
  handleChangeTypingStatus(store) {
    if (this.keyPressHandler_) {
      window.removeEventListener(EventType.KEYDOWN, this.keyPressHandler_);
      this.keyPressHandler_ = null;
    }

    const isDataPresent = store.loadingStatus !== Store.LoadingStatus.NO_DATA;
    if (!store.isTyping && isDataPresent) {
      this.keyPressHandler_ = (event) => {
        if (event.key === Keys.LEFT || event.key === 'h') {
          this.prevChunk();
        } else if (event.key === Keys.RIGHT || event.key === 'l') {
          this.nextChunk();
        }
      };

      window.addEventListener(EventType.KEYDOWN, this.keyPressHandler_);
    }
  }

  /**
   * Select a dropdown element.
   * @param {string} id HTML ID for the dropdown value location.
   * @param {string} value Value to put in the element.
   * @param {?Object} eventValue Value to update the store with.
   */
  selectDropdown(id, value, eventValue) {
    const element = document.querySelector(`#${id} > div`);
    if (value != null) {
      dom.setTextContent(element, value);
    }
    if (eventValue == null) {
      return;
    }
    let actionType = null;
    switch (id) {
      case 'gridline-dropdown':
        actionType = Dispatcher.ActionType.TOOL_BAR_GRIDLINES;
        break;
      case 'seconds-dropdown':
        actionType = Dispatcher.ActionType.TOOL_BAR_ZOOM;
        break;
      case 'montage-dropdown':
        actionType = Dispatcher.ActionType.TOOL_BAR_MONTAGE;
        break;
      case 'sensitivity-dropdown':
        actionType = Dispatcher.ActionType.TOOL_BAR_SENSITIVITY;
        break;
      case 'low-cut-dropdown':
        actionType = Dispatcher.ActionType.TOOL_BAR_LOW_CUT;
        break;
      case 'high-cut-dropdown':
        actionType = Dispatcher.ActionType.TOOL_BAR_HIGH_CUT;
        break;
      case 'notch-dropdown':
        actionType = Dispatcher.ActionType.TOOL_BAR_NOTCH;
        break;
      default:
        log.error(
            log.getLogger('eeg_modelling.eeg_viewer.ToolBar'),
            `Unexpected dropdown ${id}`);
    }
    if (actionType) {
      Dispatcher.getInstance().sendAction({
        actionType: actionType,
        data: {
          selectedValue: eventValue,
        },
      });
    }
  }

  /**
   * Handles the selection of a montage in the settings.  This includes
   * displaying a warning if the montage requested is incomplete and setting the
   * dropdown value (like the other dropdowns function).
   * @param {string} montageName The name of the montage.
   * @param {!Object<string, !MontageInfo>} montageMap The map of montage names
   * to their corresponding channel index information.
   */
  montageSelectionCallback(montageName, montageMap) {
    if (montageMap[montageName].missingChannelList) {
      Dispatcher.getInstance().sendAction({
        actionType: Dispatcher.ActionType.WARNING,
        data: {
          message:
              ('Incomplete montage. Could not find channels ' +
               montageMap[montageName].missingChannelList),
        },
      });
    }
    this.selectDropdown('montage-dropdown', montageName,
        montageMap[montageName].indexStrList);
  }

  /**
   * Creates the index-centric info for each pre-programmed montage.  This
   * includes a list of available index strings for the montage that will be
   * used in a data request and an optional list of channel names for which
   * there were no associated indices.
   * @param {!Store.StoreData} store Store object with chunk data.
   * @param {!Array<string>} montage The list of channel names.
   * @return {!MontageInfo} The index info for a pre-programmed montage.
   */
  createMontageInfo(store, montage) {
    const /** !Map<string, number> */ channelIndexMap = new Map(
        store.indexChannelMap.getEntryList().map(x => [x[1], x[0]]));

    const missingChannelList = [];

    // Given a singular channel string that may have multiple pseudonyns like A
    // or B, 'A|B', return the index associated with either A or B if one exists
    // or return null.
    const getChannelIndexFromNameOptions = (nameWithOptions) => {
      const validNames = nameWithOptions.split('|')
          .filter(name => channelIndexMap.has(name));
      const validIndices = validNames.map(
          (name) => channelIndexMap.get(name));
      if (validIndices.length > 0) {
        return validIndices[0];
      } else {
        missingChannelList.push(nameWithOptions);
        return null;
      }
    };

    // Given a channel that may be bipolar (consist of the difference between
    // two singular channels) A-B|C, return the index string that results from
    // joining the results of getting the index of each set of channel options
    // (A, B|C) if all results are non-null, otherwise return null.
    const getBipolarChannelIndexString = (bipolarName) => {
      const singularChannels = bipolarName.split('-');
      const indices = singularChannels
          .map((channelName) => getChannelIndexFromNameOptions(channelName))
          .filter(index => index != null);
      if (indices.length == singularChannels.length) {
        return indices.join('-');
      } else {
        return null;
      }
    };

    const indexStrList = montage
        .map((channelStr) => getBipolarChannelIndexString(channelStr))
        .filter(channel => channel != null);

    return {
      indexStrList: indexStrList,
      missingChannelList: missingChannelList,
    };
  }

  /**
   * Initializes the montage drop down menu with the index values for the
   * corresponding lead standards to use for an API request.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleIndexChannelMap(store) {
    if (!store.indexChannelMap) {
      return;
    }

    const montageMap = {};
    Object.entries(this.getMontages()).forEach(([montageName, montage]) => {
      montageMap[montageName] = this.createMontageInfo(store, montage);
    });

    const dropdownOptions = document.querySelectorAll(
        'ul[for=montage-dropdown] > li');
    dropdownOptions.forEach((option) => {
      option.onclick = () => this.montageSelectionCallback(option.id,
          montageMap);
    });

    let currentMontage = null;
    Object.entries(montageMap).forEach(([montageName, montageInfo]) => {
      if (store.channelIds == montageInfo.indexStrList) {
        currentMontage = montageName;
      }
    });
    this.selectDropdown('montage-dropdown', currentMontage, null);
  }

  /**
   * Enables and disables and sets tool bar buttons appropriately.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleChunkNavigation(store) {
    if (!store.chunkGraphData) {
      document.querySelectorAll('#tool-bar button').forEach((button) => {
        assertButton(button).disabled = true;
      });
      return;
    }

    document.querySelectorAll('#tool-bar button').forEach((button) => {
      assertButton(button).disabled = false;
    });

    const chunkStart = store.chunkStart;
    const chunkDuration = store.chunkDuration;
    const numSecs = assertNumber(store.numSecs);

    const prevButton =
        assertButton(document.getElementById('prev-button'));
    const nextButton =
        assertButton(document.getElementById('next-button'));
    const prevSecButton =
        assertButton(document.getElementById('prev-sec-button'));
    const nextSecButton =
        assertButton(document.getElementById('next-sec-button'));
    prevButton.disabled = (chunkStart <= 0);
    nextButton.disabled = (chunkStart + chunkDuration >= numSecs);
    prevSecButton.disabled = (chunkStart <= 0);
    nextSecButton.disabled = (chunkStart + chunkDuration >= numSecs);

    this.selectDropdown('low-cut-dropdown', `${store.lowCut} Hz`, null);
    this.selectDropdown('high-cut-dropdown', `${store.highCut} Hz`, null);
    this.selectDropdown('notch-dropdown', `${store.notch} Hz`, null);
    this.selectDropdown('sensitivity-dropdown',
        `${store.sensitivity} ${String.fromCharCode(956)}V`, null);
    this.selectDropdown('time-frame-dropdown', `${store.chunkDuration} sec`,
        null);
    this.selectDropdown('grid-dropdown', `${1/store.timeScale} / sec`, null);
  }

  /**
   * Calculates the epoch of the current data in the file.
   * @param {!Store.StoreData} store Store object with chunk data.
   * @return {string} The epoch number out of total epochs.
   */
  calculateEpoch(store) {
    // The rounding process could end up only partially reflecting the current
    // epoch in view if the time span is split between epochs.
    const epoch = String(Math.round(store.chunkStart / store.chunkDuration));
    const total = String(Math.ceil(
        assertNumber(store.numSecs) / store.chunkDuration));
    return `${epoch}/${total}`;
  }

  /**
   * Requests next chunk in the file.
   */
  nextChunk() {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.TOOL_BAR_NEXT_CHUNK,
      data: {},
    });
  }

  /**
   * Requests chunk starting 1 second later.
   */
  nextSec() {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.TOOL_BAR_NEXT_SEC,
      data: {},
    });
  }

  /**
   * Requests previous chunk in the file.
   */
  prevChunk() {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.TOOL_BAR_PREV_CHUNK,
      data: {},
    });
  }

  /**
   * Requests chunk starting 1 second earlier.
   */
  prevSec() {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.TOOL_BAR_PREV_SEC,
      data: {},
    });
  }
}

goog.addSingletonGetter(ToolBar);

exports = ToolBar;
