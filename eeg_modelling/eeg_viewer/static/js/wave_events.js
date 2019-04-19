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

/**
 * @fileoverview Support a panel with a list of events added by the user, named
 * 'Wave Events', such as SZ, ED, etc.
 * Note that for the user, these are just named 'Events'. WaveEvents is the
 * name used in the code, to avoid confusion with programming events.
 */

goog.module('eeg_modelling.eeg_viewer.WaveEvents');

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const dom = goog.require('goog.dom');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');
const utils = goog.require('eeg_modelling.eeg_viewer.utils');

class WaveEvents {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will display the events list.
    store.registerListener([Store.Property.WAVE_EVENTS], 'WaveEvents',
        (store) => this.handleWaveEvents(store));
    // This listener will show the similarity response in the UI (either success
    // or error).
    store.registerListener(
        [
          Store.Property.SIMILAR_PATTERN_RESULT,
          Store.Property.SIMILAR_PATTERN_ERROR,
        ],
        'WaveEvents', (store) => this.handleSearchSimilarResponse(store));

    /** @private {string} */
    this.tableId_ = 'wave-events-table';
    /** @private {string} */
    this.emptyTextId_ = 'no-events-text';

    /** @private @const {string} */
    this.similarContainerId_ = 'similar-patterns-container';
    /** @private @const {string} */
    this.similarTableId_ = 'similar-patterns-table';
    /** @private @const {string} */
    this.loadingSpinnerId_ = 'similar-patterns-spinner';
    /** @private @const {string} */
    this.errorTextId_ = 'similarity-error';

    /** @private @const {string} */
    this.actionMenuContainer_ = 'event-actions-container';

    /** @private @const {string} */
    this.similarPatternActions_ = 'pattern-actions-container';

    /** @private {?Store.Annotation} */
    this.clickedWaveEvent_ = null;

    /** @private {?Store.SimilarPattern} */
    this.clickedSimilarPattern_ = null;
  }

  /**
   * Display the list of events.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleWaveEvents(store) {
    if (!store.waveEvents || store.waveEvents.length === 0) {
      utils.hideElement(this.tableId_);
      utils.showElement(this.emptyTextId_);
      return;
    }

    utils.showElement(this.tableId_);
    utils.hideElement(this.emptyTextId_);
    let tableBody = document.querySelector(`#${this.tableId_} tbody`);
    if (tableBody) {
      tableBody.remove();
    }
    tableBody = document.createElement('tbody');
    document.getElementById(this.tableId_).appendChild(tableBody);

    store.waveEvents.forEach((waveEvent) => {
      const row = /** @type {!HTMLElement} */ (document.createElement('tr'));
      const addTextElementToRow = (text) => {
        const element = document.createElement('td');
        element.classList.add('mdl-data-table__cell--non-numeric');
        dom.setTextContent(element, text);
        row.appendChild(element);
      };

      addTextElementToRow(waveEvent.labelText);

      const absStartTime = store.absStart + waveEvent.startTime;
      addTextElementToRow(formatter.formatTime(absStartTime, true));
      addTextElementToRow(formatter.formatDuration(waveEvent.duration));

      row.onclick = (event) => {
        const top = event.y - event.offsetY + row.offsetHeight;
        this.handleWaveEventClick(waveEvent, top);
      };

      tableBody.appendChild(row);
    });
  }

  /**
   * Handles a click in a wave event in the list, which will open the actions
   * menu.
   * @param {!Store.Annotation} waveEvent Wave event clicked.
   * @param {number} top Top coordinate to position the wave event actions card.
   */
  handleWaveEventClick(waveEvent, top) {
    if (this.clickedWaveEvent_ && this.clickedWaveEvent_.id === waveEvent.id) {
      this.closeWaveEventMenu();
      return;
    }

    this.clickedWaveEvent_ = waveEvent;

    const menuContainer = document.getElementById(this.actionMenuContainer_);
    menuContainer.style.top = `${top}px`;
    menuContainer.classList.remove('hidden');
  }

  /**
   * Navigates to the previously clicked Wave Event,
   * and closes the wave event actions menu.
   */
  navigateToWaveEvent() {
    const { startTime, duration } = this.clickedWaveEvent_;
    this.closeWaveEventMenu();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.NAVIGATE_TO_SPAN,
      data: {
        startTime,
        duration,
      },
    });
  }

  /**
   * Searches similar patterns to the previously clicked wave event,
   * and closes the wave event actions menu.
   */
  searchSimilarPatterns() {
    const selectedWave = Object.assign({}, this.clickedWaveEvent_);
    this.closeWaveEventMenu();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SEARCH_SIMILAR_REQUEST,
      data: selectedWave,
    });

    utils.showElement(this.similarContainerId_);
    utils.hideElement(this.similarTableId_);
    utils.hideElement(this.errorTextId_);
    utils.showMDLSpinner(this.loadingSpinnerId_);
  }

  /**
   * Deletes the previously clicked wave event, and closes the wave event
   * actions menu.
   */
  deleteWaveEvent() {
    const waveEventId = this.clickedWaveEvent_.id;
    this.closeWaveEventMenu();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.DELETE_WAVE_EVENT,
      data: {
        id: waveEventId,
      },
    });
  }

  /**
   * Closes the wave event actions menu and clears the selection.
   */
  closeWaveEventMenu() {
    utils.hideElement(this.actionMenuContainer_);
    this.clickedWaveEvent_ = null;
  }

  /**
   * Handles a response to the Similar Patterns request, and displays the result
   * in the panel.
   * @param {!Store.StoreData} store Data from the store.
   */
  handleSearchSimilarResponse(store) {
    utils.hideMDLSpinner(this.loadingSpinnerId_);
    if (store.similarPatternError) {
      utils.showElement(this.errorTextId_);

      const { message } = store.similarPatternError;
      document.getElementById(this.errorTextId_).textContent = message;
      return;
    }
    utils.showElement(this.similarTableId_);
    this.createSimilarPatternTable(store);
  }

  /**
   * Creates and populates a table with the similar patterns found.
   * @param {!Store.StoreData} store Data from the store.
   */
  createSimilarPatternTable(store) {
    let tableBody = document.querySelector(`#${this.similarTableId_} tbody`);
    if (tableBody) {
      tableBody.remove();
    }
    tableBody = document.createElement('tbody');
    document.getElementById(this.similarTableId_).appendChild(tableBody);

    store.similarPatternResult.forEach((similarPattern) => {
      const row = /** @type {!HTMLElement} */ (document.createElement('tr'));
      const addTextElementToRow = (text) => {
        const element = document.createElement('td');
        element.classList.add('mdl-data-table__cell--non-numeric');
        dom.setTextContent(element, text);
        row.appendChild(element);
      };

      const absStartTime = store.absStart + similarPattern.startTime;
      addTextElementToRow(formatter.formatTime(absStartTime, true));

      addTextElementToRow(similarPattern.score.toFixed(2));

      row.onclick = (event) => {
        const top = event.y - event.offsetY + row.offsetHeight;
        this.handleSimilarPatternClick(similarPattern, top);
      };

      tableBody.appendChild(row);
    });
  }

  /**
   * Handles a click in a similar pattern, which will display the similar
   * pattern actions menu.
   * @param {!Store.SimilarPattern} similarPattern Similar pattern clicked.
   * @param {number} top Top coordinate to position the menu.
   */
  handleSimilarPatternClick(similarPattern, top) {
    if (this.clickedSimilarPattern_ &&
        this.clickedSimilarPattern_.startTime === similarPattern.startTime) {
      this.closeSimilarPatternMenu();
      return;
    }

    this.clickedSimilarPattern_ = similarPattern;

    const menuContainer = document.getElementById(this.similarPatternActions_);
    menuContainer.style.top = `${top}px`;
    menuContainer.classList.remove('hidden');
  }

  /**
   * Navigates to the previously selected similar pattern, and closes the
   * similar pattern actions menu.
   */
  navigateToPattern() {
    const { startTime, duration } = this.clickedSimilarPattern_;
    this.closeSimilarPatternMenu();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.NAVIGATE_TO_SPAN,
      data: {
        startTime,
        duration,
      },
    });
  }

  /**
   * Accepts a similar pattern, and closes the similar pattern actions menu.
   */
  acceptSimilarPattern() {
    const selectedPattern = Object.assign({}, this.clickedSimilarPattern_);
    this.closeSimilarPatternMenu();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SIMILAR_PATTERN_ACCEPT,
      data: selectedPattern,
    });
  }

  /**
   * Rejects a similar pattern, and closes the similar pattern actions menu.
   */
  rejectSimilarPattern() {
    const selectedPattern = Object.assign({}, this.clickedSimilarPattern_);
    this.closeSimilarPatternMenu();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SIMILAR_PATTERN_REJECT,
      data: selectedPattern,
    });
  }

  /**
   * Closes the similar pattern actions menu, and clears the selection.
   */
  closeSimilarPatternMenu() {
    utils.hideElement(this.similarPatternActions_);
    this.clickedSimilarPattern_ = null;
  }

}

goog.addSingletonGetter(WaveEvents);

exports = WaveEvents;

