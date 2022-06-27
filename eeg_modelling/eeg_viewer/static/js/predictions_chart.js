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

goog.module('eeg_modelling.eeg_viewer.PredictionsChart');

const ChartBase = goog.require('eeg_modelling.eeg_viewer.ChartBase');
const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const array = goog.require('goog.array');

/**
 * @typedef {{
 *   getStart: function(!Store.StoreData):number,
 *   getNumSecs: function(!Store.StoreData):number,
 *   getElementsToDraw: !ChartBase.GetElementsToDraw,
 *   getVTickDisplayValues: ?function(!Store.StoreData):!Array<string>,
 *   getVAxisMin: function(!Store.StoreData):number,
 *   getVAxisMax: function(!Store.StoreData):number,
 *   updateDataProperties: !Array<!Store.Property>,
 *   updateOverlayProperties: !Array<!Store.Property>,
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

    this.highlightViewportStyle = {
      color: 'rgba(83, 109, 254, 0.2)',
      fill: true,
    };

    /** @public {string} */
    this.containerId = 'prediction-line-chart-container';

    this.chartOptions.crosshair.orientation = 'both';
    this.chartOptions.dataOpacity = 0;

    /** @public {!Object<!Store.PredictionMode, !Mode>} */
    this.modes = {
      [Store.PredictionMode.NONE]: {
        getStart: (store) => 0,
        getNumSecs: (store) => store.numSecs,
        getElementsToDraw: (store) => [],
        getVTickDisplayValues: () => [],
        getVAxisMin: (store) => -0.5,
        getVAxisMax: (store) => 0.5,
        updateDataProperties: [],
        updateOverlayProperties: [],
      },
      [Store.PredictionMode.CHUNK_SCORES]: {
        getStart: (store) => 0,
        getNumSecs: (store) => store.numSecs,
        getElementsToDraw: (store) => this.drawChunkScores(store),
        getVTickDisplayValues: () => [],
        getVAxisMin: (store) => -0.5,
        getVAxisMax: (store) => 0.5,
        updateDataProperties: [Store.Property.NUM_SECS],
        updateOverlayProperties: [Store.Property.CHUNK_SCORES],
      },
      [Store.PredictionMode.ATTRIBUTION_MAPS]: {
        getStart: (store) => store.predictionChunkStart,
        getNumSecs: (store) => store.predictionChunkSize,
        getElementsToDraw: (store, chartArea) =>
            this.drawAttributionMap(store, chartArea),
        getVTickDisplayValues: (store) =>
            store.chunkGraphData.cols.slice(1).map((x) => x.id),
        getVAxisMin: (store) => -store.seriesHeight / 2,
        getVAxisMax: (store) =>
            store.seriesHeight * (store.chunkGraphData.cols.length - 1.5),
        updateDataProperties: [
          Store.Property.PREDICTION_CHUNK_START,
          Store.Property.PREDICTION_CHUNK_SIZE,
        ],
        updateOverlayProperties: [Store.Property.ATTRIBUTION_MAPS],
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

    this.overlayLayers = [{
      name: 'predictions',
      getElementsToDraw: (store, chartArea) => {
        const modeOptions = this.modes[store.predictionMode];
        return modeOptions.getElementsToDraw(store, chartArea);
      },
    }];

    const store = Store.getInstance();
    // This listener callback will update the underlay of the graph which draws
    // a heatmap with the prediction data specified by the mode and shades the
    // timespan in the viewport.
    store.registerListener(
        [
          Store.Property.ATTRIBUTION_MAPS, Store.Property.LABEL,
          Store.Property.CHUNK_SCORES, Store.Property.PREDICTION_MODE,
          Store.Property.NUM_SECS,
        ],
        'PredictionsChart',
        (store, changedProperties) =>
            this.handleChartData(store, changedProperties));
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
   * Returns the attribution maps to draw in the predictions chart canvas.
   * @param {!Store.StoreData} store Store data.
   * @param {!ChartBase.ChartArea} chartArea Chart area dimensions.
   * @return {!Array<!ChartBase.OverlayElement>} Elements to draw in the canvas.
   */
  drawAttributionMap(store, chartArea) {
    if (!store.attributionMaps) {
      return [];
    }
    const map = store.attributionMaps.get(store.label);
    if (!map) {
      return [];
    }
    const nChannels = store.channelIds.length;
    const channelHeight = chartArea.height / nChannels;
    const predictionChunkStart = Number(store.predictionChunkStart);
    const predictionChunkSize = Number(store.predictionChunkSize);
    const elements = [];
    store.channelIds.forEach((channelId, seriesIndex) => {
      const attrValues = map.getAttributionMapMap().get(channelId)
          .getAttributionList();
      const width = predictionChunkSize / attrValues.length;
      const top = seriesIndex * channelHeight;
      attrValues.forEach((opacity, rowIndex) => {
        const startX = predictionChunkStart + rowIndex * width;
        elements.push({
          fill: true,
          color: `rgba(255,110,64,${opacity})`,
          top,
          startX,
          endX: startX + width,
          height: channelHeight,
          minWidth: 0,
        });
      });
    });
    return elements;
  }

  /**
   * Returns the chunk scores to draw in the predictions chart canvas.
   * @param {!Store.StoreData} store Store data.
   * @return {!Array<!ChartBase.OverlayElement>} Elements to draw in the canvas.
   */
  drawChunkScores(store) {
    if (!store.chunkScores) {
      return [];
    }
    return store.chunkScores.reduce((drawElements, chunkScoreData) => {
      const scoreData = chunkScoreData.getScoreDataMap().get(store.label);
      if (!scoreData) {
        return drawElements;
      }
      const predictedValue = scoreData.getPredictedValue();
      const opacity = this.getOpacity(predictedValue ? predictedValue : 0);
      const chunkStartTime = Number(chunkScoreData.getStartTime());
      const chunkScoreDuration = Number(chunkScoreData.getDuration());
      drawElements.push({
        fill: true,
        color: `rgba(255,110,64,${opacity})`,
        startX: chunkStartTime,
        endX: chunkStartTime + chunkScoreDuration,
      });
      return drawElements;
    }, []);
  }

  /**
   * @override
   */
  shouldUpdateData(store, changedProperties) {
    return ChartBase.changedPropertiesIncludeAny(
        changedProperties,
        this.modes[store.predictionMode].updateDataProperties,
    );
  }

  /**
   * @override
   */
  shouldRedrawContent(store, changedProperties) {
    return ChartBase.changedPropertiesIncludeAny(changedProperties, [
      Store.Property.PREDICTION_MODE,
    ]);
  }

  /**
   * @override
   */
  shouldRedrawOverlay(store, changedProperties) {
    return ChartBase.changedPropertiesIncludeAny(changedProperties, [
      Store.Property.LABEL,
      ...this.modes[store.predictionMode].updateOverlayProperties,
    ]);
  }
}

goog.addSingletonGetter(PredictionsChart);

exports = PredictionsChart;
