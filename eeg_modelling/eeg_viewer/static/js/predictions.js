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

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const JspbMap = goog.require('jspb.Map');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const TableSorter = goog.require('goog.ui.TableSorter');
const dom = goog.require('goog.dom');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');


// TODO(pdpino): move these functions to an utils.js file
/**
 * @typedef {{
 *   componentHandler: {
 *     upgradeElement: function(!Element):void,
 *   },
 * }}
 */
let MDLEnhancedWindow;

/**
 * @typedef {{
 *   checked: boolean,
 * }}
 */
let HTMLCheckboxElement;

/**
 * @typedef {{
 *   x: number,
 *   y: number,
 * }}
 */
let CartesianPoint;

/**
 * Return the keys of a proto map.
 * @param {!JspbMap} protoMap map to extract the keys from.
 * @return {!Array<string>} Array with Map keys
 */
function getProtoMapKeys(protoMap) {
  const keys = [];
  const keyIter = protoMap.keys();
  let key = keyIter.next();
  while (!key.done) {
    keys.push(key.value);
    key = keyIter.next();
  }
  return keys;
}

/**
 * Transforms polar to cartesian coordinates.
 * The angle=0 is in the 12 o'clock position, and increases counter-clockwise.
 * @param {number} angle Angle in radians.
 * @param {!CartesianPoint} center Center of the frame of reference.
 * @param {number} radius Radius of the circle.
 * @return {!CartesianPoint} Cartesian coordinates of the point.
 */
function polarToCartesian(angle, center, radius) {
  return {
    x: Math.floor(center.x - radius * Math.sin(angle)),
    y: Math.floor(center.y - radius * Math.cos(angle)),
  };
}

/**
 * Returns a SVG element with a ring filled with a certain fraction.
 * @param {number} height Height of the SVG element in pixels.
 * @param {number} width Width of the SVG element in pixels.
 * @param {number} radius Radius of the ring in pixels.
 * @param {!CartesianPoint} center Coordinates of the center of the ring in
 * pixels, relative to the start of the SVG.
 * @param {number} filledFraction Number between 0 and 1 that indicates the
 * fraction of the ring to fill with a solid color.
 * @param {string} filledColor Color to use in the filled side.
 * @param {string=} emptyColor Color to use in the empty side of the ring.
 * Defaults to filledColor.
 * @return {!Element} SVG element containing the ring.
 */
function createSVGRing(
    height, width, radius, center, filledFraction, filledColor,
    emptyColor = undefined) {
  const svgNamespace = 'http://www.w3.org/2000/svg';

  const svg = document.createElementNS(svgNamespace, 'svg');
  svg.setAttribute('height', height);
  svg.setAttribute('width', width);
  const start = polarToCartesian(0, center, radius);
  const end = polarToCartesian(filledFraction * Math.PI * 2, center, radius);
  const strokeWidth = 5;
  const filledOpacity = 1;
  const emptyOpacity = 0.2;
  emptyColor = emptyColor || filledColor;

  const isMoreFilled = filledFraction >= 0.5;

  if (start.x !== end.x || start.y !== end.y) {
    const filledPath = document.createElementNS(svgNamespace, 'path');
    filledPath.setAttribute('fill', 'transparent');
    filledPath.setAttribute('stroke', filledColor);
    filledPath.setAttribute('stroke-width', strokeWidth);
    filledPath.setAttribute('opacity', filledOpacity);
    const filledIsLargeArc = isMoreFilled ? 1 : 0;
    const counterClockwise = 0;
    filledPath.setAttribute('d', [
      "M", start.x, start.y,
      "A", radius, radius, 0, filledIsLargeArc, counterClockwise, end.x, end.y,
    ].join(' '));

    const emptyPath = document.createElementNS(svgNamespace, 'path');
    emptyPath.setAttribute('fill', 'transparent');
    emptyPath.setAttribute('stroke', emptyColor);
    emptyPath.setAttribute('stroke-width', strokeWidth);
    emptyPath.setAttribute('opacity', emptyOpacity);
    const emptyIsLargeArc = isMoreFilled ? 0 : 1;
    const clockwise = 1;
    emptyPath.setAttribute('d', [
      "M", start.x, start.y,
      "A", radius, radius, 0, emptyIsLargeArc, clockwise, end.x, end.y,
    ].join(' '));

    svg.appendChild(filledPath);
    svg.appendChild(emptyPath);
  } else {
    const color = isMoreFilled ? filledColor : emptyColor;
    const opacity = isMoreFilled ? filledOpacity : emptyOpacity;
    const circle = document.createElementNS(svgNamespace, 'circle');
    circle.setAttribute('cx', center.x);
    circle.setAttribute('cy', center.y);
    circle.setAttribute('r', radius);
    circle.setAttribute('stroke-width', strokeWidth);
    circle.setAttribute('stroke', color);
    circle.setAttribute('opacity', opacity);
    circle.setAttribute('fill', 'transparent');
    svg.appendChild(circle);
  }

  return svg;
}

/**
 * Add a MDL tooltip element to display on hover of a target element.
 * @param {!Element} parentElement HTML element to add the tooltip element.
 * Must be inserted in the DOM before calling this function
 * (for the upgradeElement function to work).
 * @param {string} targetId HTML id of the element targeted by the tooltip.
 * @param {string} tooltipText Text to display on the tooltip.
 */
function addTooltipElement(parentElement, targetId, tooltipText) {
  const tooltip = document.createElement('div');
  tooltip.setAttribute('for', targetId);
  tooltip.className = 'mdl-tooltip mdl-tooltip--large';
  dom.setTextContent(tooltip, tooltipText);

  parentElement.appendChild(tooltip);

  const mdlCastWindow = /** @type {!MDLEnhancedWindow} */ (window);
  mdlCastWindow.componentHandler.upgradeElement(tooltip);
}


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


    // For now the threshold is set with any number. In the future it will be
    // set with a verified value (i.e. a number that predicts well),
    // and possibly there will be support to change it from the UI.
    /** @private {number} threshold to classify as positive or negative */
    this.threshold_ = 0.5;

    /** @private {!Object} Currently active filters */
    this.activeFilters_ = {
      label: {
        ED: true,
        SZ: true,
      },
      prediction: {
        pos: true,
        neg: false,
      },
    };

    /** @private {string} */
    this.noPredictionsId_ = 'no-predictions-text';
  }

  /**
   * Toggles a filter and refilters the predictions list with the new filter
   * settings.
   * @param {!Event} event
   * @param {string} parameter
   * @param {string} filterValue
   */
  toggleFilter(event, parameter, filterValue) {
    const target = /** @type {!HTMLCheckboxElement} */ (event.target);
    this.activeFilters_[parameter][filterValue] = target.checked;
    this.filterPredictions();
  }

  /**
   * Filter predictions according to the currently active filters.
   */
  filterPredictions() {
    const rows = document.querySelectorAll('table.prediction tbody tr');
    let rowCounter = 0;
    rows.forEach((row) => {
      const displayRow = Object.keys(this.activeFilters_).every((parameter) => {
        const rowValue = row.getAttribute(parameter);
        return rowValue && this.activeFilters_[parameter][rowValue];
      });
      row.style.display = displayRow ? 'table-row' : 'none';
      rowCounter += displayRow;
    });
    this.setPredictionsListEmpty_(!rowCounter);
  }

  /**
   * Highlight the prediction menu rows that appear in the viewport.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleChunkNavigation(store) {
    if (!store.chunkScores) {
      return;
    }
    const chunkStart = store.chunkStart;
    const chunkEnd = chunkStart + store.chunkDuration;
    const predictionRows = document.querySelectorAll(
        'table.prediction tbody tr');
    predictionRows.forEach((row) => {
      const predictionStart = Number(row.getAttribute('data-start'));
      const predictionEnd = (predictionStart +
                              Number(row.getAttribute('data-duration')));
      if (Math.max(chunkStart, predictionStart) <
          Math.min(chunkEnd, predictionEnd)) {
        row.classList.add('in-viewport');
      } else {
        row.classList.remove('in-viewport'); }
    });
  }

  /**
   * Set the label selected in the dropdown text.
   * @param {string} label Label to set in the UI.
   * @private
   */
  setLabelDropdownText_(label) {
    dom.setTextContent(document.querySelector('#label-dropdown > div'),
        label);
  }

  /**
   * Set the prediction mode selected in the dropdown text.
   * @param {string} mode Name of the mode to set in the UI.
   * @private
   */
  setModeDropdownText_(mode) {
    dom.setTextContent(document.querySelector('#mode-dropdown > div'),
        mode);
  }

  /**
   * Handles click on a prediction label.
   * @param {string} dropdownValue The label value to set as the dropdown value.
   */
  handlePredictionLabelSelection(dropdownValue) {
    this.setLabelDropdownText_(dropdownValue);
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
    this.setModeDropdownText_(mode);
    Dispatcher.getInstance().sendAction({
      actionType: Dispatcher.ActionType.PREDICTION_MODE_SELECTION,
      data: {
        selectedValue: mode,
      },
    });
  }

  /**
   * Sets the predictions table as empty or not empty.
   * If empty shows a 'empty' message.
   * @param {boolean} empty Indicates if the table is empty or not.
   * @private
   */
  setPredictionsListEmpty_(empty) {
    const noPredictionsText = document.getElementById(this.noPredictionsId_);
    dom.setTextContent(noPredictionsText, 'No predictions found');
    noPredictionsText.classList.toggle('hidden', !empty);
  }

  /**
   * Creates the table body of the predictions list.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  createPredictionTableBody(store) {
    let tableBody = document.querySelector('table.prediction tbody');
    if (tableBody) {
      tableBody.remove();
    }
    tableBody = document.createElement('tbody');
    document.querySelector('table.prediction').appendChild(tableBody);

    let rowCounter = 0;
    store.chunkScores.forEach((chunkScoreData) => {
      const startTime = /** @type {number} */ (chunkScoreData.getStartTime());
      const duration = /** @type {number} */ (chunkScoreData.getDuration());
      if (startTime != null && duration != null) {
        const prettyTime = formatter.formatTime(startTime + store.absStart);

        const scoreDataMap = chunkScoreData.getScoreDataMap();

        getProtoMapKeys(scoreDataMap).forEach((labelName) => {
          const labelData = scoreDataMap.get(labelName);
          const predictedValue = labelData.getPredictedValue();
          const predictionProba = labelData.getPredictionProbability() || 0;

          if (predictedValue == null) {
            return;
          }

          const isPositive = predictedValue > this.threshold_;

          const row = document.createElement('tr');
          row.setAttribute('label', labelName);
          row.setAttribute('prediction', isPositive ? 'pos' : 'neg');
          tableBody.appendChild(row);

          const addTextElementToRow = (text) => {
            const element = document.createElement('td');
            element.classList.add('mdl-data-table__cell--non-numeric');
            dom.setTextContent(element, text);
            row.appendChild(element);
          };

          const addCircleElementToRow = (probability) => {
            // An auxiliary hidden td is added with the actual probability
            // score, so the TableSorter uses these values to sort when sorting
            // by the Pred column
            const scoreElement = document.createElement('td');
            scoreElement.textContent = probability.toFixed(8);
            scoreElement.style.display = 'none';
            row.appendChild(scoreElement);

            const circleElement = document.createElement('td');
            circleElement.classList.add('prediction-circle');
            row.appendChild(circleElement);

            const height = 48; // height of a row
            const width = 87;
            const radius = 13;
            const center = { x: 23, y: height/2 };
            const svgCircle = createSVGRing(
                height, width, radius, center, probability, 'rgb(255, 99, 71)');
            svgCircle.id = `conf-${rowCounter}`;
            circleElement.appendChild(svgCircle);

            const confidence = Math.floor(probability * 100);
            addTooltipElement(
                circleElement, svgCircle.id, `${confidence}% confidence`);
          };

          addTextElementToRow(prettyTime);
          addTextElementToRow(isPositive ? labelName : `no ${labelName}`);
          addCircleElementToRow(predictionProba);

          row.onclick = () => {
            Dispatcher.getInstance().sendAction({
              actionType: Dispatcher.ActionType.PREDICTION_CHUNK_REQUEST,
              data: {
                time: startTime,
              },
            });
          };

          row.setAttribute('data-start', startTime);
          row.setAttribute('data-duration', duration);

          rowCounter += 1;
        });
      }
    });

    this.setPredictionsListEmpty_(!rowCounter);
  }

  /**
   * Sets the predictions panel UI as empty or not empty, depending if
   * predictions were loaded or not.
   * If not loaded, hides the table and dropdowns, shows an 'empty' message and
   * disables the filter button.
   * @param {boolean} loaded Indicates if predictions were loaded or not.
   * @private
   */
  setPredictionsLoadedUI_(loaded) {
    const noPredictionsText = document.getElementById(this.noPredictionsId_);
    dom.setTextContent(noPredictionsText, 'No predictions loaded');
    noPredictionsText.classList.toggle('hidden', loaded);

    const table = document.querySelector('table.prediction');
    table.classList.toggle('hidden', !loaded);

    document.querySelectorAll('#predictions-panel > .dropdown')
        .forEach(
            dropdownDiv => dropdownDiv.classList.toggle('hidden', !loaded));

    const filterButton = document.getElementById('predictions-filter-button');
    if (loaded) {
      filterButton.removeAttribute('disabled');
    } else {
      filterButton.setAttribute('disabled', '');
    }
  }

  /**
   * Initialize prediction navigation menu.
   * @param {!Store.StoreData} store Store object with chunk data.
   */
  handleChunkScores(store) {
    if (!store.chunkScores) {
      this.setPredictionsLoadedUI_(false);
      return;
    }

    this.setPredictionsLoadedUI_(true);

    this.createPredictionTableBody(store);

    this.tableSorter.dispose();
    this.tableSorter = new TableSorter();
    this.tableSorter.decorate(document.querySelector('table.prediction'));

    /**
     * Comparator function for numbers or strings.
     */
    const sortFunction = (a, b) => a > b ? 1 : -1;

    this.tableSorter.setDefaultSortFunction(sortFunction);
    const timeColumn = 0;
    this.tableSorter.sort(timeColumn);

    this.filterPredictions();
    this.setLabelDropdownText_(store.label);
    this.setModeDropdownText_(store.predictionMode);

  }
}

goog.addSingletonGetter(Predictions);

exports = Predictions;
