// Copyright 2021 The Google Research Authors.
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
const montages = goog.require('eeg_modelling.eeg_viewer.montages');
const {assertInstanceof, assertNumber} = goog.require('goog.asserts');

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
        } else if (event.key === 'a') {
          this.shiftSecs_(-5);
        } else if (event.key === 'd') {
          this.shiftSecs_(5);
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
   * @param {!Object<string, !montages.MontageInfo>} montageMap The map of
   *     montage names to their corresponding channel index information.
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
   * Initializes the montage drop down menu with the index values for the
   * corresponding lead standards to use for an API request.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleIndexChannelMap(store) {
    if (!store.indexChannelMap) {
      return;
    }

    const montageMap = {};
    Object.entries(montages.getMontages()).forEach(([montageName, montage]) => {
      montageMap[montageName] = montages.createMontageInfo(
          store.indexChannelMap, /** @type {!Array<string>} */ (montage));
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
   * Sends an action to shift the chunk start an amount of seconds.
   * @param {number} seconds Amount of seconds to move.
   * @private
   */
  shiftSecs_(seconds) {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.TOOL_BAR_SHIFT_SECS,
      data: {
        time: seconds,
      },
    });
  }

  /**
   * Requests chunk starting 1 second later.
   */
  nextSec() {
    this.shiftSecs_(1);
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
    this.shiftSecs_(-1);
  }
}

goog.addSingletonGetter(ToolBar);

exports = ToolBar;
