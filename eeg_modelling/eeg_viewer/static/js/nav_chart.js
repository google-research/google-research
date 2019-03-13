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

goog.module('eeg_modelling.eeg_viewer.NavChart');

const ChartBase = goog.require('eeg_modelling.eeg_viewer.ChartBase');
const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const array = goog.require('goog.array');
const {assertNumber} = goog.require('goog.asserts');


/**
 * Handles displaying the full time span of a file as well as annotations and
 * navigational capabilities.
 */
class NavChart extends ChartBase {

  constructor() {
    super();

    /** @public {string} */
    this.containerId = 'nav-line-chart-container';

    /** @public {string} */
    this.parentId = 'tool-bar';

    /** @public {string} */
    this.overlayId = 'nav-overlay';

    this.chartOptions.chartArea.backgroundColor = 'lightgrey';
    this.chartOptions.chartArea.height = '30%';
    this.chartOptions.colors = this.generateColors(2, '#fff');
    this.chartOptions.height = 64;
    this.chartOptions.hAxis.baselineColor = 'white';
    this.chartOptions.hAxis.gridlines.color = 'white';
    this.chartOptions.vAxis.viewWindow = {
      min: 0,
      max: 1,
    };

    this.chartListeners = [
      {
        type: 'click',
        handler: (e) => {
          const cli = this.getChartLayoutInterface();
          const chartArea = cli.getChartAreaBoundingBox();
          const x = cli.getHAxisValue(e.offsetX + chartArea.left);
          Dispatcher.getInstance().sendAction({
            actionType: Dispatcher.ActionType.NAV_BAR_CHUNK_REQUEST,
            data: {
              time: x,
            }
          });
        },
      },
    ];

    const store = Store.getInstance();
    // This listener callback will update the underlay of the graph which draws
    // a heatmap with the prediction data specified by the mode and shades the
    // timespan in the viewport.
    store.registerListener([Store.Property.CHUNK_START,
        Store.Property.CHUNK_DURATION, Store.Property.NUM_SECS], 'NavChart',
        (store) => this.handleChunkNavigationAndPredictionData(store));
  }

  /**
   * @override
   */
  getVTickDisplayValues(store) {
    return [];
  }

  /**
   * @override
   */
  getHTickValues(store) {
    const numTicks = this.getNumSecs(store);
    return array.range(0, numTicks).map((x) => x + this.getStart(store));
  }

  /**
   * @override
   */
  getStart(store) {
    return 0;
  }

  /**
   * @override
   */
  getNumSecs(store) {
    return assertNumber(store.numSecs);
  }

  /**
   * @override
   */
  updateChartOptions(store) {
    const parentWidth = this.getParent().clientWidth;
    const buttonWidth = 64 * document.querySelectorAll('#tool-bar > button').length;
    const containerWidth = parentWidth - buttonWidth;
    this.setOption('width', containerWidth);
    this.setOption('hAxis.viewWindow', {
      min: this.getStart(store),
      max: (this.getStart(store) + this.getNumSecs(store)),
    });
    this.setOption('hAxis.ticks', this.createHTicks(store));
  }

  /**
   * @override
   */
  createOverlay(store) {
    super.createOverlay(store);
    this.highlightViewport(store);
  }

  /**
   * Update underlay with new chunk start or duration and any new prediction if
   * the mode specifies to display prediction data.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   */
  handleChunkNavigationAndPredictionData(store) {
    if (store.numSecs) {
      this.handleChartData(store);
    }
  }
}

goog.addSingletonGetter(NavChart);

exports = NavChart;
