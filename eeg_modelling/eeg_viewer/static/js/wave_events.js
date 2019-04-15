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

    /** @private {string} */
    this.tableId_ = 'wave-events-table';
    /** @private {string} */
    this.emptyTextId_ = 'no-events-text';

    /** @private @const {string} */
    this.actionMenuContainer_ = 'event-actions-container';

    /** @private {?Store.Annotation} */
    this.clickedWaveEvent_ = null;
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
}

goog.addSingletonGetter(WaveEvents);

exports = WaveEvents;

