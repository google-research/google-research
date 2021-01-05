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
const log = goog.require('goog.log');
const utils = goog.require('eeg_modelling.eeg_viewer.utils');


/** @const {number} default height of the similar patterns action menu. */
const defaultSimilarPatternsActionHeight = 167;

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
          Store.Property.SIMILAR_PATTERNS_UNSEEN,
          Store.Property.SIMILAR_PATTERN_ERROR,
        ],
        'WaveEvents', (store) => this.handleSearchSimilarResponse(store));
    // This listener will highlight the target on the wave events table.
    store.registerListener(
        [Store.Property.SIMILAR_PATTERN_TEMPLATE], 'WaveEvents',
        (store) => this.handleSimilarityTarget(store, false));
    // This listener will highlight the curve target on the wave events table.
    store.registerListener(
        [Store.Property.SIMILARITY_CURVE_TEMPLATE], 'WaveEvents',
        (store) => this.handleSimilarityTarget(store, true));
    // This listener will update the metrics displayed in the UI.
    store.registerListener(
        [Store.Property.SIMILAR_PATTERNS_SEEN],
        'WaveEvents', (store) => this.handleSimilarityMetrics(store));
    // This listener callback will update the loading spinner
    store.registerListener(
        [Store.Property.SIMILARITY_CURVE_RESULT], 'WaveEvents',
        (store) => this.handleIncomingCurve(store));
    this.logger_ = log.getLogger('eeg_modelling.eeg_viewer.WaveEvents');

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
    this.curveSpinnerId_ = 'curve-spinner';
    /** @private @const {string} */
    this.clearCurveButtonId_ = 'clear-curve-button';
    /** @private @const {string} */
    this.errorTextId_ = 'similarity-error';
    /** @private @const {string} */
    this.searchMoreButtonId_ = 'similar-patterns-search-more';

    /** @private @const {string} */
    this.actionMenuContainer_ = 'event-actions-container';

    /** @private @const {string} */
    this.similarPatternActions_ = 'pattern-actions-container';

    /** @private {?Store.Annotation} */
    this.clickedWaveEvent_ = null;

    /** @private {?Store.SimilarPattern} */
    this.clickedSimilarPattern_ = null;

    /** @private {?number} */
    this.prevTemplateId_ = null;

    /** @private {?number} */
    this.prevCurveTemplateId_ = null;

    /** @private @const {string} */
    this.templateEventClass_ = 'template-event';

    /** @private @const {string} */
    this.curveTemplateEventClass_ = 'template-event-curve';
  }

  /**
   * Gets the Y coordinate of the end of an element that fired an event.
   * i.e The top + height coordinate of the element.
   * @param {!Event} event Event fired by the element.
   * @return {number} Y coordinate of the end of the element.
   * @private
   */
  getPositionBelow_(event) {
    const target = /** @type {!HTMLElement} */ (event.target);
    return event.y - event.offsetY + target.offsetHeight;
  }

  /**
   * Gets the top coordinate of an element that fired an event.
   * @param {!Event} event Event fired by the element.
   * @return {number} Y coordinate of the end of the element.
   * @private
   */
  getPositionAbove_(event) {
    return event.y - event.offsetY;
  }

  /**
   * Returns the id of a wave events table row, given a wave event id.
   * @param {number} waveEventId Id of the wave event to select from the table.
   * @return {string} HTML id of the row.
   * @private
   */
  getRowId_(waveEventId) {
    return `wave-event-row-${waveEventId}`;
  }

  /**
   * Highlights the similarity target on the wave events list.
   * @param {!Store.StoreData} store store data.
   * @param {boolean} isForCurve indicates if selected event is for the
   *     similarity curve or for the search-similar-pattern.
   */
  handleSimilarityTarget(store, isForCurve) {
    const selectedClass =
        isForCurve ? this.curveTemplateEventClass_ : this.templateEventClass_;

    const prevTemplateId =
        isForCurve ? this.prevCurveTemplateId_ : this.prevTemplateId_;

    if (prevTemplateId) {
      const prevSelected =
          document.getElementById(this.getRowId_(prevTemplateId));
      if (prevSelected) {
        prevSelected.classList.remove(selectedClass);
      }
    }

    const template = isForCurve ? store.similarityCurveTemplate :
                                  store.similarPatternTemplate;

    if (!template || template.id == null) {
      if (isForCurve) {
        this.prevCurveTemplateId_ = null;
      } else {
        this.prevTemplateId_ = null;
        utils.hideElement(this.searchMoreButtonId_);
      }
      return;
    }

    const row = document.getElementById(this.getRowId_(template.id));
    row.classList.add(selectedClass);

    if (isForCurve) {
      this.prevCurveTemplateId_ = template.id;
    } else {
      this.prevTemplateId_ = template.id;
    }
  }

  /**
   * Updates the metrics UI with the amount of accepted similar patterns.
   * @param {!Store.StoreData} store Data from the store.
   */
  handleSimilarityMetrics(store) {
    let acceptedAmount = 0;
    let rejectedAmount = 0;
    const attemptsAmount = store.similarPatternsSeen.length;
    store.similarPatternsSeen.forEach((similarPattern) => {
      switch (similarPattern.status) {
        case Store.SimilarPatternStatus.ACCEPTED:
          acceptedAmount += 1;
          break;
        case Store.SimilarPatternStatus.REJECTED:
          rejectedAmount += 1;
          break;
        default:
          break;
      }
    });

    document.getElementById('metrics-attempts').textContent = attemptsAmount;
    document.getElementById('metrics-accepted').textContent = acceptedAmount;
    document.getElementById('metrics-rejected').textContent = rejectedAmount;

    const precision =
        attemptsAmount === 0 ? 0 : acceptedAmount / attemptsAmount;

    document.getElementById('metrics-precision').textContent =
        precision.toFixed(2);

    document.querySelectorAll('.at-k').forEach((span) => {
      span.textContent = attemptsAmount;
    });
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
    const table = document.getElementById(this.tableId_);
    let tableBody = document.querySelector(`#${this.tableId_} tbody`);
    if (tableBody) {
      tableBody.remove();
    }
    tableBody = document.createElement('tbody');
    table.appendChild(tableBody);

    const templateId =
        store.similarPatternTemplate && store.similarPatternTemplate.id;
    const curveTemplateId =
        store.similarityCurveTemplate && store.similarityCurveTemplate.id;

    store.waveEvents.forEach((waveEvent, index) => {
      if (waveEvent.id == null) {
        log.error(this.logger_, `Event with no id: index ${index}`);
        return;
      }

      const row = document.createElement('tr');
      row.id = this.getRowId_(/** @type {number} */ (waveEvent.id));
      if (waveEvent.id === templateId) {
        row.classList.add(this.templateEventClass_);
      }
      if (waveEvent.id === curveTemplateId) {
        row.classList.add(this.curveTemplateEventClass_);
      }
      tableBody.appendChild(row);

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

      const channelTooltip = waveEvent.channelList.length > 0 ?
          waveEvent.channelList.join('<br>') :
          'No channels';
      utils.addMDLTooltip(row, row.id, channelTooltip);

      row.onclick = (event) => {
        this.handleWaveEventClick(event, waveEvent);
      };
    });

    const scrollableTable = table.parentElement;
    scrollableTable.scrollTop = scrollableTable.scrollHeight;
  }

  /**
   * Handles a click in a wave event in the list, which will open the actions
   * menu.
   * @param {!Event} event Click event.
   * @param {!Store.Annotation} waveEvent Wave event clicked.
   */
  handleWaveEventClick(event, waveEvent) {
    if (this.clickedWaveEvent_ && this.clickedWaveEvent_.id === waveEvent.id) {
      this.closeWaveEventMenu();
      return;
    }

    this.clickedWaveEvent_ = waveEvent;

    const top = this.getPositionBelow_(event);
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
   * @param {boolean=} closeMenu whether or not to close the menu.
   */
  searchSimilarPatterns(closeMenu = true) {
    const selectedWave = Object.assign({}, this.clickedWaveEvent_);
    if (closeMenu) {
      this.closeWaveEventMenu();
    }
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SEARCH_SIMILAR_REQUEST,
      data: selectedWave,
    });

    utils.hideElement(this.errorTextId_);
    utils.hideElement(this.searchMoreButtonId_);
    utils.showMDLSpinner(this.loadingSpinnerId_);
  }

  /**
   * Updates the spinner and clear-curve button, according to the new result.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleIncomingCurve(store) {
    if (store.similarityCurveResult) {
      utils.hideMDLSpinner(this.curveSpinnerId_);
      utils.showElement(this.clearCurveButtonId_);
    } else {
      utils.hideElement(this.clearCurveButtonId_);
    }
  }

  /**
   * Requests a similarity curve for the previously clicked wave event,
   * and closes the wave event actions menu.
   * @param {boolean=} closeMenu whether or not to close the menu. The menu will
   *     be closed only if a request is actually sent.
   */
  getSimilarityCurve(closeMenu = true) {
    const selectedWave = Object.assign({}, this.clickedWaveEvent_);
    if (this.prevCurveTemplateId_ === selectedWave['id']) {
      return;
    }

    if (closeMenu) {
      this.closeWaveEventMenu();
    }

    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SIMILARITY_CURVE_REQUEST,
      data: selectedWave,
    });

    utils.hideElement(this.clearCurveButtonId_);
    utils.showMDLSpinner(this.curveSpinnerId_);
  }

  /**
   * Sends an action to clear the similarity curve.
   */
  clearCurve() {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SIMILARITY_CURVE_CLEAR,
      data: {},
    });
    this.prevCurveTemplateId_ = null;
  }

  /**
   * Requests a similarity curve and searches similar patterns.
   */
  searchAndGetCurve() {
    this.searchSimilarPatterns(false);
    this.getSimilarityCurve(false);

    this.closeWaveEventMenu();
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
    if (store.similarPatternTemplate) {
      utils.showElement(this.searchMoreButtonId_);
    }
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

    if (!store.similarPatternsUnseen ||
        store.similarPatternsUnseen.length === 0) {
      return;
    }

    store.similarPatternsUnseen.forEach((similarPattern) => {
      const row = document.createElement('tr');
      const addTextElementToRow = (text) => {
        const element = document.createElement('td');
        element.classList.add('mdl-data-table__cell--non-numeric');
        dom.setTextContent(element, text);
        row.appendChild(element);
      };

      addTextElementToRow(similarPattern.score.toFixed(2));

      const absStartTime = store.absStart + similarPattern.startTime;
      addTextElementToRow(formatter.formatTime(absStartTime, true));
      addTextElementToRow(formatter.formatDuration(similarPattern.duration));

      row.onclick = (event) => {
        this.handleSimilarPatternClick(event, similarPattern);
      };

      tableBody.appendChild(row);
    });
  }

  /**
   * Handles a click in a similar pattern, which will display the similar
   * pattern actions menu.
   * @param {!Event} event Click event.
   * @param {!Store.SimilarPattern} similarPattern Similar pattern clicked.
   */
  handleSimilarPatternClick(event, similarPattern) {
    if (this.clickedSimilarPattern_ &&
        this.clickedSimilarPattern_.startTime === similarPattern.startTime) {
      this.closeSimilarPatternMenu();
      return;
    }

    this.clickedSimilarPattern_ = similarPattern;

    let top = this.getPositionBelow_(event);

    const menuContainer = /** @type {!HTMLElement} */ (
        document.getElementById(this.similarPatternActions_));
    const menuHeight =
        menuContainer.offsetHeight || defaultSimilarPatternsActionHeight;

    if (top + menuHeight > window.innerHeight) {
      top = this.getPositionAbove_(event) - menuHeight;
    }

    menuContainer.style.top = `${top}px`;
    menuContainer.classList.remove('hidden');
  }

  /**
   * Navigates to the previously selected similar pattern.
   */
  navigateToPattern() {
    const { startTime, duration } = this.clickedSimilarPattern_;
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.NAVIGATE_TO_SPAN,
      data: {
        startTime,
        duration,
      },
    });
  }

  /**
   * Confirms a similar pattern for accepting, rejecting or editing; and closes
   * the similar pattern menu.
   * @param {!Dispatcher.ActionType} actionType Action to perform on the similar
   *     pattern, either accept, reject or edit.
   * @private
   */
  confirmSimilarPattern_(actionType) {
    const selectedPattern = Object.assign({}, this.clickedSimilarPattern_);
    this.closeSimilarPatternMenu();
    Dispatcher.getInstance().sendAction({
      actionType,
      data: selectedPattern,
    });
  }

  /**
   * Accepts a similar pattern.
   */
  acceptSimilarPattern() {
    this.confirmSimilarPattern_(Dispatcher.ActionType.SIMILAR_PATTERN_ACCEPT);
  }

  /**
   * Set a similar pattern for edition.
   */
  editSimilarPattern() {
    this.confirmSimilarPattern_(Dispatcher.ActionType.SIMILAR_PATTERN_EDIT);
  }

  /**
   * Rejects a similar pattern.
   */
  rejectSimilarPattern() {
    this.confirmSimilarPattern_(Dispatcher.ActionType.SIMILAR_PATTERN_REJECT);
  }

  /**
   * Closes the similar pattern actions menu, and clears the selection.
   */
  closeSimilarPatternMenu() {
    utils.hideElement(this.similarPatternActions_);
    this.clickedSimilarPattern_ = null;
  }

  /**
   * Sends an action to reject all the similar patterns found, and closes the
   * similar pattern menu in case it was open.
   */
  rejectAll() {
    this.closeSimilarPatternMenu();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SIMILAR_PATTERN_REJECT_ALL,
      data: {},
    });
  }

  /**
   * Searches more similar patterns.
   */
  searchMore() {
    utils.hideElement(this.errorTextId_);
    utils.hideElement(this.searchMoreButtonId_);
    utils.showMDLSpinner(this.loadingSpinnerId_);
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SEARCH_SIMILAR_REQUEST_MORE,
      data: {},
    });
  }

  /**
   * Toggles the similarity settings modal.
   */
  toggleSimilaritySettings() {
    const hidden = document.getElementById('similarity-settings-modal')
                       .classList.toggle('hidden');
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.CHANGE_TYPING_STATUS,
      data: {
        isTyping: !hidden,
      },
    });
  }

  /**
   * Save the similarity settings selected in the UI.
   */
  saveSimilaritySettings() {
    const topN = Number(utils.getInputElement('similarity-top-n').value);
    const mergeCloseResults = utils.getInputElement('similarity-merge').checked;
    const mergeThreshold =
        Number(utils.getInputElement('similarity-merge-threshold').value);

    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.SEARCH_SIMILAR_UPDATE_SETTINGS,
      data: {
        topN,
        mergeCloseResults,
        mergeThreshold,
      },
    });
    this.toggleSimilaritySettings();
  }

  /**
   * Downloads all events.
   */
  downloadEvents() {
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.DOWNLOAD_DATA,
      data: {
        properties: [
          Store.Property.SIMILAR_PATTERN_PAST_TRIALS,
          Store.Property.SIMILAR_PATTERN_TEMPLATE,
          Store.Property.SIMILAR_PATTERNS_SEEN,
          Store.Property.SIMILAR_PATTERNS_UNSEEN,
          Store.Property.WAVE_EVENTS,
        ],
        name: 'events',
      },
    });
  }
}

goog.addSingletonGetter(WaveEvents);

exports = WaveEvents;
