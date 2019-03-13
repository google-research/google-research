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

goog.module('eeg_modelling.eeg_viewer.Predictions');

const ChunkScoreData = goog.require('proto.eeg_modelling.protos.PredictionMetadata.ChunkScoreData');
const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const TableSorter = goog.require('goog.ui.TableSorter');
const dom = goog.require('goog.dom');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');
const {assertString} = goog.require('goog.asserts');


class Predictions {

  constructor() {
    /** @public {!TableSorter} */
    this.tableSorter = new TableSorter();

    const store = Store.getInstance();
    // This listener callback will initialize the prediction menu with chunks
    // that are ranked by their score for the current label.
    store.registerListener([Store.Property.CHUNK_SCORES],
        'Predictions', (store) => this.handleChunkScores(store));
    // This listener callback will highlight the chunks that intersect with the
    // timespan in the viewport.
    store.registerListener([Store.Property.CHUNK_START,
        Store.Property.CHUNK_DURATION], 'Predictions',
        (store) => this.handleChunkNavigation(store));
  }

  /**
   * Filters prediction rows by value of the true positive and true negative
   * buttons.
   * @param {?string} buttonId HTML ID of the button clicked.
   */
  filterPredictions(buttonId) {
    if (buttonId) {
      const filterButton = document.getElementById(buttonId);
      filterButton.classList.toggle('mdl-button--accent');
      const filterIcon = document.querySelector(`#${buttonId} i`);
      dom.setTextContent(filterIcon,
          filterIcon.innerHTML == 'toggle_on' ? 'toggle_off' : 'toggle_on');
    }
    const truePos = document.querySelectorAll(
        'table.prediction tr.ir-true-pos');
    const truePosSelected =
        document.querySelector('#true-pos i').innerHTML == 'toggle_on';
    truePos.forEach((truePosRow) => {
      truePosRow.style.display = truePosSelected ? 'table-row' : 'none';
    });
    const trueNeg = document.querySelectorAll(
        'table.prediction tr.ir-true-neg');
    const trueNegSelected =
        document.querySelector('#true-neg i').innerHTML == 'toggle_on';
    trueNeg.forEach((trueNegRow) => {
      trueNegRow.style.display = trueNegSelected ? 'table-row' : 'none';
    });
  }

  /**
   * Sorts predictions highest to lowest prediction score or lowest to highest.
   */
  sortPredictions() {
    const reverse = document.querySelector('td.rank').innerHTML == 1;
    const label = document.querySelector('#label-dropdown > div').innerHTML;
    let col = -1;
    document.querySelectorAll('.prediction th').forEach((header, index) => {
      if (header.classList.contains('predicted') &&
          header.classList.contains(label)) {
        col = index;
      }
    });
    this.tableSorter.sort(col, reverse);
  }

  /**
   * Highlight the prediction menu rows that appear in the viewport.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleChunkNavigation(store) {
    if (!store.chunkScores) {
      return;
    }
    const chunk_start = store.chunkStart;
    const chunk_end = chunk_start + store.chunkDuration;
    const predictionRows = document.querySelectorAll(
        'table.prediction tbody tr');
    predictionRows.forEach((row) => {
      const prediction_start = Number(row.getAttribute('data-start'));
      const prediction_end = (prediction_start +
                              Number(row.getAttribute('data-duration')));
      if (Math.max(chunk_start, prediction_start) <
          Math.min(chunk_end, prediction_end)) {
        row.classList.add('in-viewport');
      } else {
        row.classList.remove('in-viewport'); }
    });
  }

  /**
   * Set the rank values of the rows after sorting.
   */
  initializeRanks() {
    const rankCells = document.querySelectorAll('td.rank');
    rankCells.forEach((rankCell, i) => {
      dom.setTextContent(rankCell, i + 1);
    });
  }

  /**
   * Handles click on a prediction label.
   * @param {string} dropdownValue The label value to set as the dropdown value.
   */
  handlePredictionLabelSelection(dropdownValue) {
    dom.setTextContent(document.querySelector('#label-dropdown > div'),
        dropdownValue);
    document.querySelectorAll('table.prediction tbody tr').forEach((row) => {
      const score = Number(
          row.querySelector('.actual.' + dropdownValue).innerHTML);
      if (score > 0) {
        row.classList.add('ir-true-pos');
        row.classList.remove('ir-true-neg');
      } else {
        row.classList.add('ir-true-neg');
        row.classList.remove('ir-true-pos');
      }
    });
    this.sortPredictions();
    this.initializeRanks();
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.PREDICTION_LABEL_SELECTION,
      data: {
        selectedValue: dropdownValue,
      },
    });
  }

  /**
   * Handles selection of a prediction mode from a dropdown menu.
   * @param {string} mode The mode selected.
   */
  handlePredictionModeSelection(mode) {
    dom.setTextContent(document.querySelector('#mode-dropdown > div'),
        mode);
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.PREDICTION_MODE_SELECTION,
      data: {
        selectedValue: mode,
      },
    });
  }

  /**
   * Creates a header row for the predictions menu with given labels.
   * @param {!ChunkScoreData} chunkScoreData A chunk score data instance.
   * @returns {!Array<string>} Ordered list of labels to be displayed.
   */
  createPredictionHeaderRow(chunkScoreData) {
    let header = document.querySelector('table.prediction thead');
    if (header) {
      header.remove();
    }
    header = document.createElement('thead');
    document.querySelector('table.prediction').appendChild(header);
    const headerRow = document.createElement('tr');
    document.querySelector('table.prediction thead').appendChild(headerRow);
    const rankHeader = document.createElement('th');
    rankHeader.classList.add('mdl-data-table__cell--non-numeric', 'rank');
    dom.setTextContent(rankHeader, 'Rank');
    rankHeader.onclick = () => this.sortPredictions();
    headerRow.append(rankHeader);
    const labelHeader = document.createElement('th');
    labelHeader.classList.add('mdl-data-table__cell--non-numeric', 'time');
    dom.setTextContent(labelHeader, 'Start Time');
    headerRow.append(labelHeader);
    const orderedHeaders = [];
    const keyIter = chunkScoreData.getScoreDataMap().keys();
    let key = keyIter.next();
    while (!key.done) {
      const value = assertString(key.value);
      orderedHeaders.push(value);
      const predictionHeader = document.createElement('th');
      predictionHeader.classList.add(value, 'predicted');
      dom.setTextContent(predictionHeader, value);
      headerRow.append(predictionHeader);

      const actualHeader = document.createElement('th');
      actualHeader.classList.add(value, 'actual');
      dom.setTextContent(actualHeader, value);
      headerRow.append(actualHeader);

      key = keyIter.next();
    }
    return orderedHeaders;
  }

  /**
   * Initialize prediction navigation menu.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleChunkScores(store) {
    if (!store.chunkScores) {
      return;
    }
    const chunkScoreDataList = store.chunkScores;
    const headers = this.createPredictionHeaderRow(chunkScoreDataList[0]);
    let predictionTable = document.querySelector('table.prediction tbody');
    if (predictionTable) {
      predictionTable.remove();
    }
    predictionTable = document.createElement('tbody');
    document.querySelector('table.prediction').appendChild(predictionTable);
    chunkScoreDataList.forEach((chunkScoreData) => {
      if (chunkScoreData.getStartTime() != null
          && chunkScoreData.getDuration() != null) {
        const row = document.createElement('tr');
        const rank = document.createElement('td');
        rank.classList.add('mdl-data-table__cell--non-numeric', 'rank');
        row.appendChild(rank);
        const label = document.createElement('td');
        label.classList.add('mdl-data-table__cell--non-numeric', 'time');
        dom.setTextContent(label,
            formatter.formatTime(chunkScoreData.getStartTime() +
              store.absStart));
        row.appendChild(label);
        headers.forEach((labelName) => {
          const labelData = chunkScoreData.getScoreDataMap().get(labelName);
          if (labelData.getPredictedValue() != null
              && labelData.getActualValue() != null) {
            const predictedScore = document.createElement('td');
            predictedScore.classList.add(labelName, 'predicted');
            dom.setTextContent(predictedScore,
                /** @type {number} */ (labelData.getPredictedValue()));
            row.appendChild(predictedScore);

            const actualScore = document.createElement('td');
            actualScore.classList.add(labelName, 'actual');
            dom.setTextContent(actualScore,
                /** @type {number} */ (labelData.getActualValue()));
            row.appendChild(actualScore);
          }
        });
        row.onclick = () => {
          Dispatcher.getInstance().sendAction({
            actionType: Dispatcher.ActionType.PREDICTION_CHUNK_REQUEST,
            data: {
              time: chunkScoreData.getStartTime(),
            },
          });
        };
        row.setAttribute('data-start',
            /** @type {number} */ (chunkScoreData.getStartTime()));
        row.setAttribute('data-duration',
            /** @type {number} */ (chunkScoreData.getDuration()));
        row.id = 'prediction-row-' + chunkScoreData.getStartTime();
        predictionTable.appendChild(row);
      }
    });
    this.tableSorter.dispose();
    this.tableSorter = new TableSorter();
    this.tableSorter.decorate(document.querySelector('table.prediction'));
    this.handlePredictionLabelSelection(store.label);
    this.handlePredictionModeSelection(store.predictionMode);
    this.handleChunkNavigation(store);
  }
}

goog.addSingletonGetter(Predictions);

exports = Predictions;
