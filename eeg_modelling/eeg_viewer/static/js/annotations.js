// Copyright 2022 The Google Research Authors.
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

goog.module('eeg_modelling.eeg_viewer.Annotations');

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const dom = goog.require('goog.dom');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');


class Annotations {

  constructor() {
    const store = Store.getInstance();
    // This listener callback will initialize the annotation menu with a list of
    // annotations.
    store.registerListener([Store.Property.ANNOTATIONS], 'Annotations',
        (store) => this.handleAnnotations(store));
    // This listener callback will highlight the annotations in the menu if they
    // are in the current chunk.
    store.registerListener([Store.Property.CHUNK_START,
        Store.Property.CHUNK_DURATION], 'Annotations',
        (store) => this.handleChunkNavigation(store));
  }
  /**
   * Highlights any annotation menu rows that appear in the viewport.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleChunkNavigation(store) {
    const chunkStart = store.absStart + store.chunkStart;
    const chunkEnd = chunkStart + store.chunkDuration;
    const annotationRows = document.querySelectorAll(
        'table.annotation tbody tr');
    annotationRows.forEach((row) => {
      const annotationTime = Number(row.getAttribute('data-start'));
      if (annotationTime < chunkEnd && annotationTime >= chunkStart) {
        row.classList.add('in-viewport');
      } else {
        row.classList.remove('in-viewport');
      }
    });
  }

  /**
   * Initialize an annotation navigation menu.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleAnnotations(store) {
    const annotationTable = document.querySelector('table.annotation tbody');
    annotationTable.textContent = '';
    store.annotations.forEach((annotation) => {
      if (annotation.startTime != null
          && annotation.labelText != null) {
        const row = document.createElement('tr');
        const label = document.createElement('td');
        label.classList.add('mdl-data-table__cell--non-numeric');
        dom.setTextContent(label,
            /** @type {string} */ (annotation.labelText));
        const time = document.createElement('td');
        dom.setTextContent(time,
            formatter.formatTime(store.absStart + annotation.startTime));
        row.onclick = () => {
          Dispatcher.getInstance().sendAction({
            actionType: Dispatcher.ActionType.ANNOTATION_SELECTION,
            data: {
              time: annotation.startTime,
            },
          });
        };
        row.setAttribute('data-start',
            /** @type {number} */ (annotation.startTime));
        row.appendChild(label);
        row.appendChild(time);
        annotationTable.appendChild(row);
      }
    });
    dom.setTextContent(document.querySelector(
        '#labels-panel .mdl-card__title-text'),
        `Labels (${store.annotations.length})`);
    this.handleChunkNavigation(store);
  }
}

goog.addSingletonGetter(Annotations);

exports = Annotations;
