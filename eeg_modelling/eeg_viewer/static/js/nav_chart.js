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
const DataTable = goog.require('google.visualization.DataTable');
const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const array = goog.require('goog.array');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');
const utils = goog.require('eeg_modelling.eeg_viewer.utils');

/**
 * Handles displaying the full time span of a file as well as annotations and
 * navigational capabilities.
 */
class NavChart extends ChartBase {
  /**
   * Creates a function to scale a value in the Y axis.
   * The scale goes from (min, max) to (0, 1), where min and max are the
   * minimum and maximum available values.
   * @param {!Array<number>} values array of existing values.
   * @return {!Function} scaler function.
   */
  static getYScale(values) {
    if (!values || values.length === 0) {
      return () => 0;
    }

    const minValue = utils.getArrayMin(values);
    const maxValue = utils.getArrayMax(values);

    const bottomPadding = 0.15;
    const topPadding = 0.05;

    const minTarget = 0 + bottomPadding;
    const maxTarget = 1 - topPadding;
    const slope = (maxTarget - minTarget) / (maxValue - minValue);
    const intersection = minTarget - slope * minValue;
    const yScale = (value) => slope * value + intersection;

    return yScale;
  }

  constructor() {
    super();

    /** @public {string} */
    this.containerId = 'nav-line-chart-container';

    /** @public {string} */
    this.parentId = 'tool-bar';

    /** @public {string} */
    this.overlayId = 'nav-overlay';

    this.highlightViewportStyle = {
      color: '#1b66fe',
      fill: true,
      height: 5,
    };

    /** @private {number} */
    this.overlayMarkerPercentage_ = 0.75;

    this.chartOptions.chartArea.backgroundColor = 'lightgrey';
    this.chartOptions.chartArea.height = '40%';
    this.chartOptions.chartArea.top = 10;
    this.chartOptions.chartArea.left = 35;
    this.chartOptions.chartArea.width = '94%';
    this.chartOptions.crosshair.color = 'transparent';
    this.chartOptions.crosshair.selected.color = 'transparent';
    this.chartOptions.crosshair.trigger = 'focus';
    this.chartOptions.colors = ['transparent'];
    this.chartOptions.lineWidth = 2;
    this.chartOptions.height = 64;
    this.chartOptions.hAxis.baselineColor = 'white';
    this.chartOptions.hAxis.gridlines.color = 'white';
    this.chartOptions.vAxis.viewWindow = {
      min: 0,
      max: 1,
    };
    this.chartOptions.tooltip.isHtml = true;
    this.chartOptions.tooltip.trigger = 'focus';

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

    this.overlayLayers = [
      {
        name: 'waveEvents',
        getElementsToDraw: (store, chartArea) =>
            this.drawWaveEvents(store, chartArea),
      },
      {
        name: 'similarPatterns',
        getElementsToDraw: (store, chartArea) =>
            this.drawSimilarPatterns(store, chartArea),
      },
      {
        name: 'waveEventDraft',
        getElementsToDraw: (store, chartArea) =>
            this.drawWaveEventDraft(store, chartArea),
      },
    ];

    const store = Store.getInstance();
    // This listener callback will update the underlay of the graph which draws
    // a heatmap with the prediction data specified by the mode and shades the
    // timespan in the viewport.
    store.registerListener(
        [
          Store.Property.CHUNK_START, Store.Property.CHUNK_DURATION,
          Store.Property.NUM_SECS, Store.Property.WAVE_EVENTS,
          Store.Property.SIMILAR_PATTERNS_UNSEEN,
          Store.Property.WAVE_EVENT_DRAFT,
          Store.Property.SIMILARITY_CURVE_RESULT,
        ],
        'NavChart',
        (store, changedProperties) =>
            this.handleChartData(store, changedProperties));
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
    return Number(store.numSecs);
  }

  /**
   * @override
   */
  createDataTable(store) {
    const dataTable = new DataTable();
    dataTable.addColumn('number', 'seconds');
    dataTable.addColumn('number', 'placeholder');
    dataTable.addColumn({
      type: 'string',
      role: 'tooltip',
      p: {
        html: true,
      },
    });

    const yScale = NavChart.getYScale(store.similarityCurveResult);

    const numSecs = this.getNumSecs(store);
    const rowData = array.range(0, numSecs + 1).map((seconds) => {
      const prettyTime = formatter.formatTime(store.absStart + seconds);
      let tooltip = `<p>${prettyTime}</p>`;
      let value = 0.5;
      if (store.similarityCurveResult) {
        const index = store.samplingFreq * seconds;
        const score = store.similarityCurveResult[index] || 0;
        tooltip = `${tooltip} <p>Sim: ${score.toFixed(2)}</p>`;
        value = yScale(score);
      }
      return [seconds, value, tooltip];
    });

    dataTable.addRows(rowData);
    return dataTable;
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

    const lineColor = store.similarityCurveResult ? '#9d90f4' : 'transparent';
    this.setOption('colors', [lineColor]);
  }

  /**
   * Returns an array of elements that represent the wave events to draw in
   * the nav chart canvas.
   * @param {!Store.StoreData} store Store data.
   * @return {!Array<!ChartBase.OverlayElement>} Elements to draw in the canvas.
   */
  drawWaveEvents(store, chartArea) {
    return store.waveEvents.map((waveEvent) => ({
      fill: true,
      color: 'rgb(34, 139, 34)', // green
      startX: waveEvent.startTime,
      endX: waveEvent.startTime + waveEvent.duration,
      top: chartArea.height * this.overlayMarkerPercentage_,
    }));
  }

  /**
   * Returns an array of elements that represent the similar patterns to draw in
   * the nav chart canvas.
   * @param {!Store.StoreData} store Store data.
   * @return {!Array<!ChartBase.OverlayElement>} Elements to draw in the canvas.
   */
  drawSimilarPatterns(store, chartArea) {
    if (!store.similarPatternsUnseen) {
      return [];
    }
    return store.similarPatternsUnseen.map((similarPattern) => ({
      fill: true,
      color: 'rgb(255, 140, 0)', // orange
      startX: similarPattern.startTime,
      endX: similarPattern.startTime + similarPattern.duration,
      top: chartArea.height * this.overlayMarkerPercentage_,
    }));
  }

  /**
   * Returns an array of elements that represent the similar patterns to draw in
   * the nav chart canvas.
   * @param {!Store.StoreData} store Store data.
   * @return {!Array<!ChartBase.OverlayElement>} Elements to draw in the canvas.
   */
  drawWaveEventDraft(store, chartArea) {
    if (!store.waveEventDraft) {
      return [];
    }
    return [{
      fill: true,
      color: 'rgba(255, 70, 71)', // red
      startX: store.waveEventDraft.startTime,
      endX: store.waveEventDraft.startTime + store.waveEventDraft.duration,
      top: chartArea.height * this.overlayMarkerPercentage_,
    }];
  }

  /**
   * @override
   */
  shouldUpdateData(store, changedProperties) {
    return ChartBase.changedPropertiesIncludeAny(changedProperties, [
      Store.Property.NUM_SECS,
      Store.Property.SIMILARITY_CURVE_RESULT,
    ]);
  }

  /**
   * @override
   */
  shouldRedrawContent(store, changedProperties) {
    return false;
  }

  /**
   * @override
   */
  shouldRedrawOverlay(store, changedProperties) {
    return ChartBase.changedPropertiesIncludeAny(changedProperties, [
      Store.Property.CHUNK_START,
      Store.Property.CHUNK_DURATION,
      Store.Property.WAVE_EVENTS,
      Store.Property.SIMILAR_PATTERNS_UNSEEN,
      Store.Property.WAVE_EVENT_DRAFT,
    ]);
  }
}

goog.addSingletonGetter(NavChart);

exports = NavChart;
