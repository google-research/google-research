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

/**
 * @fileoverview Defines a controller to manage the two charts being rendered
 * with complementary data and the interactions between those two charts.
 */
goog.module('eeg_modelling.eeg_viewer.ChartController');

const Graph = goog.require('eeg_modelling.eeg_viewer.Graph');
const PredictionsChart = goog.require('eeg_modelling.eeg_viewer.PredictionsChart');
const {assert, assertNumber} = goog.require('goog.asserts');

/**
 * @typedef {{
 *   row: number,
 *   column: number,
 * }}
 */
let MouseOverEvent;


class ChartController {

  constructor() {
    // The main chart contains the channels of waveform data.
    this.mainChart = Graph.getInstance();
    // The auxiliary chart contains timeline and prediction data derived from
    // the waveform data file.
    this.auxChart = PredictionsChart.getInstance();

    this.mainChart.registerChartEventListener('onmouseover',
        (e) => this.crosshairHandler_(e));
    this.mainChart.registerChartEventListener('onmouseout',
        (e) => this.crosshairRemovalHandler_());
  }

  /**
   * Asserts that an event is a Google Charts mouseover event.
   * @param {?Event} e The event received.
   * @return {!MouseOverEvent} A Google Charts mouseover event.
   */
  assertMouseOverEvent(e) {
    assert(e);
    assert('row' in e);
    assertNumber(e.row);
    assert('column' in e);
    assertNumber(e.column);
    return /** @type {!MouseOverEvent} */ (e);
  }

  /**
   * This crosshair handler will trigger a crosshair on the auxiliary chart at
   * the equivalent x- and y- axis value when the crosshair is triggered on
   * the main chart.
   * @param {?Event} e The mouse over event object.
   * @private
   */
  crosshairHandler_(e) {
    if (!e || !this.auxChart.isVisible()) {
      return;
    }
    let mouseoverEvent = this.assertMouseOverEvent(e);
    // Determine the corresponding row and column in the auxiliary chart from
    // the mouse over event.
    const xAxisValue = this.mainChart.getDataTable().getValue(
        mouseoverEvent.row, 0);
    // The navigation bar x-axis is instantiated with rounded second values.
    const auxChartXAxisValue = Math.round(xAxisValue);
    let auxChartRow = -1;
    const auxChartDataTable = this.auxChart.getDataTable();
    // Scan the auxiliary chart x-axis values to find the column of the value
    // that matches the rounded mouse over value.
    for (let i = 0; i < auxChartDataTable.getNumberOfRows(); i++) {
      if (auxChartDataTable.getValue(i, 0) == auxChartXAxisValue) {
        auxChartRow = i;
        break;
      }
    }
    // Adjust for the annotation column in the main chart data.
    let auxChartColumn = mouseoverEvent.column - 1;
    // If the navigation bar does not display all column values, select the
    // column in view.
    if (this.auxChart.getOption('vAxis.viewWindow.max') == 0.5) {
      auxChartColumn = auxChartDataTable.getNumberOfColumns() - 1;
    }
    // Select the column and row to trigger the crosshair in the auxiliary
    // chart.
    if (auxChartRow != -1) {
      this.auxChart.getChart().setSelection(
          // Adjust for the annotation row in the graph chart.
          [{row: auxChartRow, column: auxChartColumn}]);
    }
  }

  /**
   * Removes any selected data points in the auxiliary chart in order to trigger
   * crosshair removal.
   * @private
   */
  crosshairRemovalHandler_() {
    if (this.auxChart.isVisible()) {
      this.auxChart.getChart().setSelection([]);
    }
  }
}

goog.addSingletonGetter(ChartController);

exports = ChartController;
