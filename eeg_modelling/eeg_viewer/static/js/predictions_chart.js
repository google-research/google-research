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

goog.module('eeg_modelling.eeg_viewer.PredictionsChart');

const ChartBase = goog.require('eeg_modelling.eeg_viewer.ChartBase');
const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const array = goog.require('goog.array');

/**
 * @typedef {{
 *   getStart: function(!Store.StoreData):number,
 *   getNumSecs: function(!Store.StoreData):number,
 *   drawOverlay: ?function(!Store.StoreData):void,
 *   getVTickDisplayValues: ?function(!Store.StoreData):!Array<string>,
 *   getVAxisMin: function(!Store.StoreData):number,
 *   getVAxisMax: function(!Store.StoreData):number,
 * }}
 */
let Mode;


/**
 * A class that handles rendering a bar consisting of a Google Chart and an
 * optional heat map overlay that provides navigational functionality by
 * requesting the data slice at the location of the user's click.
 */
class PredictionsChart extends ChartBase {

  constructor() {
    super();

    /** @public {string} */
    this.overlayId = 'prediction-overlay';

    /** @public {string} */
    this.containerId = 'prediction-line-chart-container';

    this.chartOptions.crosshair.orientation = 'both';
    this.chartOptions.dataOpacity = 0;

    /** @public {!Object<!Store.PredictionMode, !Mode>} */
    this.modes = {
      [Store.PredictionMode.NONE]: {
        getStart: (store) => 0,
        getNumSecs: (store) => store.numSecs,
        drawOverlay: null,
        getVTickDisplayValues: () => [],
        getVAxisMin: (store) => -0.5,
        getVAxisMax: (store) => 0.5,
      },
      [Store.PredictionMode.CHUNK_SCORES]: {
        getStart: (store) => 0,
        getNumSecs: (store) => store.numSecs,
        drawOverlay: (store) => this.drawChunkScores(store),
        getVTickDisplayValues: () => [],
        getVAxisMin: (store) => -0.5,
        getVAxisMax: (store) => 0.5,
      },
      [Store.PredictionMode.ATTRIBUTION_MAPS]: {
        getStart: (store) => store.predictionChunkStart,
        getNumSecs: (store) => store.predictionChunkSize,
        drawOverlay: (store) => this.drawAttributionMap(store),
        getVTickDisplayValues: (store) => store.chunkGraphData.cols.slice(1)
            .map((x) => x.id),
        getVAxisMin: (store) => -store.seriesHeight / 2,
        getVAxisMax: (store) => store.seriesHeight * (
            store.chunkGraphData.cols.length - 1.5),
      },
    };

    /** @public {!Object<!Store.PredictionMode, number>} */
    this.height = {
      [Store.PredictionMode.NONE]: 0,
      [Store.PredictionMode.CHUNK_SCORES]: 0.15,
      [Store.PredictionMode.ATTRIBUTION_MAPS]: 0.4,
    };

    this.chartListeners = [
      {
        type: 'click',
        handler: (event) => {
          if (!event.targetID.startsWith('point')) {
            return;
          }
          const cli = this.getChartLayoutInterface();
          const x = cli.getHAxisValue(event.x);
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
    store.registerListener([Store.Property.ATTRIBUTION_MAPS,
        Store.Property.LABEL, Store.Property.CHUNK_SCORES,
        Store.Property.PREDICTION_MODE, Store.Property.NUM_SECS],
        'PredictionsChart',
        (store) => this.handleChartData(store));
  }

  /**
   * @override
   */
  getVTickDisplayValues(store) {
    return this.modes[store.predictionMode].getVTickDisplayValues(store);
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
    return this.modes[store.predictionMode].getStart(store);
  }

  /**
   * @override
   */
  getNumSecs(store) {
    return this.modes[store.predictionMode].getNumSecs(store);
  }

  /**
   * @override
   */
  updateChartOptions(store) {
    this.setOption('hAxis.viewWindow', {
      min: this.getStart(store),
      max: (this.getStart(store) + this.getNumSecs(store)),
    });
    this.setOption('vAxis.viewWindow', {
       min: this.modes[store.predictionMode].getVAxisMin(store),
       max: this.modes[store.predictionMode].getVAxisMax(store),
    });
    this.setOption('colors',
        this.generateColors(store.chunkGraphData.cols.length, '#fff'));
    super.updateChartOptions(store);
  }

  /**
   * @override
   */
  createOverlay(store) {
    super.createOverlay(store);
    if (this.modes[store.predictionMode].drawOverlay) {
      this.modes[store.predictionMode].drawOverlay(store);
    }
    this.highlightViewport(store);
  }
}

goog.addSingletonGetter(PredictionsChart);

exports = PredictionsChart;
