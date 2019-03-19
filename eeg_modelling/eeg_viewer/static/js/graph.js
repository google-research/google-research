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

goog.module('eeg_modelling.eeg_viewer.Graph');

const ChartBase = goog.require('eeg_modelling.eeg_viewer.ChartBase');
const DataTable = goog.require('google.visualization.DataTable');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const array = goog.require('goog.array');
const {assert, assertNumber, assertString} = goog.require('goog.asserts');

/**
 * @typedef {{
 *   matcher: !RegExp,
 *   getTransformation: function(!Store.StoreData):number,
 * }}
 */
let NameMatcher;

/**
 * Regular expressions to categorize channel types within each file type.  They
 * are used to determine the relative sensitiviy applied to the channel.
 * @type {!Object<string, !Array<!NameMatcher>>}
 */
const channelNameMatchers = {
  'EEG': [
    {
      matcher: new RegExp('EKG'),
      getTransformation: (store) => 7 / (2 * store.sensitivity),
    },
    {
      matcher: new RegExp('^SZ_BIN$'),
      getTransformation: (store) => store.seriesHeight / 2,
    },
    {
      matcher: new RegExp('.*'),
      getTransformation: (store) => 7 / store.sensitivity,
    },
  ],
  'EKG': [
    {
      matcher: new RegExp('.*'),
      getTransformation: () => 20,
    },
  ],
  'ECG': [
    {
      matcher: new RegExp('.*'),
      getTransformation: () => 20,
    },
  ],
};


class Graph extends ChartBase {

  constructor() {
    super();

    this.containerId = 'line-chart-container';

    this.chartOptions.chartArea.backgroundColor = {};

    this.height = {
      [Store.PredictionMode.NONE]: 1.0,
      [Store.PredictionMode.CHUNK_SCORES]: 0.85,
      [Store.PredictionMode.ATTRIBUTION_MAPS]: 0.6,
    };

    /** @public {!Map<string, number>} */
    this.channelTransformations = new Map([]);

    const store = Store.getInstance();
    // This listener callback will initialize a graph with the annotations and
    // DataTable.
    store.registerListener([Store.Property.ANNOTATIONS,
        Store.Property.CHUNK_GRAPH_DATA, Store.Property.TIMESCALE,
        Store.Property.SENSITIVITY], 'Graph',
        (store) => this.handleChartData(store));
  }

  /**
   * @override
   */
  getHTickValues(store) {
    return array.range(store.chunkStart, store.chunkStart + store.chunkDuration,
        store.timeScale);
  }

  /**
   * @override
   */
  getVTickDisplayValues(store) {
    return store.chunkGraphData.cols.slice(1).map((x) => x.id);
  }

  /**
   * Derives render transformation coefficient from series ID.
   * @param {string} seriesName Name of the series of data.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {number} Coefficient to multiply data series by.
   */
  getRenderTransformation(seriesName, store) {
    if (this.channelTransformations.has(seriesName)) {
      return this.channelTransformations.get(seriesName);
    }
    assert(store.sensitivity != 0);
    // Default transformation for any file or channel type.
    let transformation = 1 / store.sensitivity;
    const nameMatchers = channelNameMatchers[assertString(store.fileType)];
    if (!nameMatchers) {
      return transformation;
    }
    for (const nameMatcher of nameMatchers) {
      if (nameMatcher.matcher.test(seriesName)) {
        transformation = nameMatcher.getTransformation(store);
        break;
      }
    }
    this.channelTransformations.set(seriesName, transformation);
    return transformation;
  }

  /**
   * Staggers data series vertically by given series height.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @param {!Object} data DataTable object in object literal JSON format.
   * @return {!DataTable} A DataTable object with values adjusted for
   *     sensitivity and series offset, with the display values formatted.
   */
  formatDataForRendering(store, data) {
    const dataTable = new DataTable(data);
    // Skips over the first column of data that becomes the axis values.
    for (let c = 1; c < dataTable.getNumberOfColumns(); c++) {
      const offset = this.getRenderOffset(c, store);
      const transform = this.getRenderTransformation(dataTable.getColumnId(c),
                                                     store);
      for (let r = 0; r < dataTable.getNumberOfRows(); r++) {
        if (dataTable.getValue(r, c) != null) {
          const value = assertNumber(dataTable.getValue(r, c));
          const transformedValue = value * transform + offset;
          dataTable.setValue(r, c, transformedValue);
          dataTable.setFormattedValue(r, c, '' + value);
        }
      }
    }
    return dataTable;
  }

  /**
   * @override
   */
  getStart(store) {
    return store.chunkStart;
  }

  /**
   * @override
   */
  getNumSecs(store) {
    return store.chunkDuration;
  }

  /**
   * @override
   */
  createDataTable(store) {
    const chunkGraphData = /** @type {!Object} */ (JSON.parse(JSON.stringify(
        store.chunkGraphData)));
    const dataTable = this.formatDataForRendering(store, chunkGraphData);
    this.addAnnotations(store, dataTable);
    return dataTable;
  }

  /**
   * @override
   */
  updateChartOptions(store) {
    const numSeries = store.chunkGraphData.cols.length;
    this.setOption('tooltip.trigger', 'selection');
    this.setOption('vAxis.viewWindow', {
       min: -store.seriesHeight * 2,
       max: store.seriesHeight * numSeries,
    });
    this.setOption('colors',
        this.generateColors(store.chunkGraphData.cols.length, '#696969'));
    super.updateChartOptions(store);
  }

  /**
   * Increases the sensitivity of the selected channel.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @param {number} modifier An additive modifer for the channel
   * transformations.
   */
  modifyChannelSensitivity(store, modifier) {
    const channelColumn = this.getChart().getSelection()[0].column;
    const channelId = this.getDataTable().getColumnId(channelColumn);
    this.channelTransformations.set(channelId,
        this.channelTransformations.get(channelId) + modifier);
    this.handleChartData(store);
  }

  /**
   * @override
   */
  addChartActions(store) {
    const sensitivityModifier = 0.5;
    const chart = this.getChart();
    if (chart) {
      chart.setAction({
        id: 'increase',
        text: 'Increase Sensitivity',
        action: () => this.modifyChannelSensitivity(store, sensitivityModifier),
      });
      chart.setAction({
        id: 'decrease',
        text: 'Decrease Sensitivity',
        action: () => this.modifyChannelSensitivity(store,
            -sensitivityModifier),
      });
    }
  }

  /**
   * @override
   */
  handleChartData(store) {
    if (!store.chunkGraphData) {
      this.channelTransformations = new Map([]);
      return;
    }
    super.handleChartData(store);
  }
}

goog.addSingletonGetter(Graph);

exports = Graph;
