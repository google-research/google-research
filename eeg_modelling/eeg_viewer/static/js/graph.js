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
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');
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

/**
 * Creates the tooltip to display on a chart point, in HTML format.
 * @param {string} timestamp Formatted time to display.
 * @param {string} columnName Name of the column of the point.
 * @param {string|number} value Y value in the point, in uV.
 * @return {string} HTML to display in tooltip.
 */
function createHTMLTooltip(timestamp, columnName, value) {
  return `<p>${timestamp}</p><p>${columnName}</p><p>${
      Number(value).toFixed(2)} ${String.fromCharCode(956)}V</p>`;
}

class Graph extends ChartBase {

  constructor() {
    super();

    this.containerId = 'line-chart-container';

    this.chartOptions.chartArea.backgroundColor = {};

    this.chartOptions.annotations = {
      boxStyle: {
        stroke: 'black',
        strokeWidth: 1,
        rx: 5,
        ry: 5,
        gradient: {
          color1: 'rgb(83, 109, 254)',
          color2: 'rgb(83, 109, 254)',
          x1: '0%', y1: '0%',
          x2: '100%', y2: '100%',
          useObjectBoundingBoxUnits: true,
        },
      },
      textStyle: {
        fontSize: 15,
        bold: false,
      },
    };
    this.chartOptions.crosshair.color = 'rgb(83, 109, 254)';
    this.chartOptions.crosshair.selected.color = 'rgb(34, 139, 34)';
    this.chartOptions.tooltip.isHtml = true;
    this.chartOptions.tooltip.trigger = 'focus';

    this.height = {
      [Store.PredictionMode.NONE]: 1.0,
      [Store.PredictionMode.CHUNK_SCORES]: 0.85,
      [Store.PredictionMode.ATTRIBUTION_MAPS]: 0.6,
    };

    /** @public {!Map<string, number>} */
    this.channelTransformations = new Map([]);

    this.chartListeners = [
      {
        type: 'click',
        handler: (event) => {
          if (!event.targetID || !event.targetID.startsWith('vAxis')) {
            return;
          }

          const cli = this.getChartLayoutInterface();
          const chartArea = cli.getChartAreaBoundingBox();

          const seriesIndexReversed = Number(event.targetID.split('#')[3]);
          const columnIndex =
              this.seriesIndexToColumnIndex_(seriesIndexReversed, true);
          this.handleChannelNameClick(columnIndex, event.y, chartArea.left);
        },
      },
    ];

    /** @private {?Function} */
    this.changeChannelSensitivity_ = null;

    /** @private {?string} */
    this.clickedChannelName_ = null;

    /** @private @const {string} */
    this.channelActionsId_ = 'channel-actions-container';

    /** @private @const {string} */
    this.channelActionsTitleId_ = 'channel-actions-title';

    const store = Store.getInstance();
    // This listener callback will initialize a graph with the annotations and
    // DataTable.
    store.registerListener([Store.Property.ANNOTATIONS,
        Store.Property.CHUNK_GRAPH_DATA, Store.Property.TIMESCALE,
        Store.Property.SENSITIVITY], 'Graph',
        (store) => this.handleChartData(store));
    // This listener callback will resize the graph considering if the
    // predictions chart is in display.
    store.registerListener(
        [Store.Property.PREDICTION_MODE],
        'Graph', (store) => this.handleDraw(store));
  }

  /**
   * Transforms the series index to the column index of a channel.
   * The series index is the correlative order of the channels as displayed in
   * the chart, starting at the bottom of the chart, and starting at 0.
   * The column index is the index used directly in the data table.
   *
   * This function considers the following columns in the data table:
   *   0: time
   *   1: annotation
   *   2: annotationText (HTML)
   *   3: first channel
   *   4: first channel tooltip
   *   5: second channel
   *   6: second channel tooltip
   *   7: ...etc
   * E.g., if the seriesIndex is 1, the columnIndex returned should be 5.
   *
   * @param {number} seriesIndex Position of the channel in the chart.
   * @param {boolean=} reversed Indicates if the series index is reversed.
   * @return {number} Column index as it appears in the data table.
   * @private
   */
  seriesIndexToColumnIndex_(seriesIndex, reversed = false) {
    if (reversed) {
      const nCols = this.dataTable.getNumberOfColumns();
      const nChannels = (nCols - 3) / 2;
      seriesIndex = nChannels - 1 - seriesIndex;
    }

    return 3 + 2 * seriesIndex;
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
   * Staggers data series vertically, considering sensitivity and series offset.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @param {!DataTable} dataTable instance.
   */
  formatDataForRendering(store, dataTable) {
    // Skips over the first column of data that becomes the axis values.
    for (let col = 1; col < dataTable.getNumberOfColumns(); col++) {
      const offset = this.getRenderOffset(col, store);
      const transform = this.getRenderTransformation(dataTable.getColumnId(col),
                                                     store);
      for (let row = 0; row < dataTable.getNumberOfRows(); row++) {
        if (dataTable.getValue(row, col) != null) {
          const value = Number(dataTable.getFormattedValue(row, col));
          const transformedValue = value * transform + offset;

          // The formatted value on each cell holds the actual voltage value.
          // The value holds the value with the transformation applied.
          dataTable.setValue(row, col, transformedValue);
          dataTable.setFormattedValue(row, col, value);
        }
      }
    }
  }

  /**
   * Sets formatted time in the domain column to use when rendering.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @param {!DataTable} dataTable instance.
   */
  formatDomainForRendering(store, dataTable) {
    for (let row = 0; row < dataTable.getNumberOfRows(); row++) {
      const timeValue = dataTable.getValue(row, 0);
      const formattedTime =
          formatter.formatTime(store.absStart + timeValue, true);
      dataTable.setFormattedValue(row, 0, formattedTime);
    }
  }

  /**
   * Formats the annotations for DataTable.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @param {!DataTable} dataTable DataTable object to add the annotations to.
   */
  addAnnotations(store, dataTable) {
    dataTable.insertColumn(1, 'string');
    dataTable.setColumnProperty(1, 'role', 'annotation');
    dataTable.insertColumn(2, 'string');
    dataTable.setColumnProperty(2, 'role', 'annotationText');
    dataTable.setColumnProperty(2, 'html', true);

    store.annotations.forEach((annotation, index) => {
      const labelText = `<p>${annotation.labelText}</p>`;
      const samplingFreq = assertNumber(store.samplingFreq);
      const startTime = assertNumber(annotation.startTime);
      // Find the closest 'x' to the actual start time of the annotation, where
      // 'x' is a point on the x-axis.  Note that the x-axis points are
      // 1/samplingFreq apart from each other.
      const x = (Math.round(startTime * samplingFreq) / samplingFreq);
      for (let row = 0; row < dataTable.getNumberOfRows(); row++) {
        if (dataTable.getValue(row, 0) == x) {
          dataTable.setValue(row, 1, 'Label');
          dataTable.setValue(row, 2, labelText);
        }
      }
    });
  }

  /**
   * Adds columns in the data table with HTML tooltips for each point in the
   * graph.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @param {!DataTable} dataTable DataTable object to add the annotations to.
   */
  addTooltips(store, dataTable) {
    // The first data column has index 3:
    // Columns: [time, annotation, annotationText, data, ...]
    const firstDataCol = 3;

    const nRows = dataTable.getNumberOfRows();
    const nCols = dataTable.getNumberOfColumns();

    // TODO(pdpino): make the column insertion more efficient
    for (let dataCol = nCols - 1; dataCol >= firstDataCol; dataCol--) {
      const tooltipCol = dataCol + 1;
      dataTable.insertColumn(tooltipCol, 'string');
      dataTable.setColumnProperty(tooltipCol, 'role', 'tooltip');
      dataTable.setColumnProperty(tooltipCol, 'html', true);
      const channelName = dataTable.getColumnLabel(dataCol);

      for (let row = 0; row < nRows; row++) {
        const prettyTime = dataTable.getFormattedValue(row, 0);
        const value = dataTable.getFormattedValue(row, dataCol);
        const tooltipHtml = createHTMLTooltip(prettyTime, channelName, value);
        dataTable.setValue(row, tooltipCol, tooltipHtml);
      }
    }
  }

  /**
   * @override
   */
  createDataTable(store) {
    const chunkGraphData = /** @type {!Object} */ (JSON.parse(JSON.stringify(
        store.chunkGraphData)));
    const dataTable = new DataTable(chunkGraphData);
    this.formatDataForRendering(store, dataTable);
    this.formatDomainForRendering(store, dataTable);
    this.addAnnotations(store, dataTable);
    this.addTooltips(store, dataTable);
    return dataTable;
  }

  /**
   * @override
   */
  updateChartOptions(store) {
    const numSeries = store.chunkGraphData.cols.length;
    this.setOption('vAxis.viewWindow', {
       min: -store.seriesHeight * 2,
       max: store.seriesHeight * numSeries,
    });
    this.setOption('colors',
        this.generateColors(store.chunkGraphData.cols.length, '#696969'));
    super.updateChartOptions(store);
  }

  /**
   * Handles a click in a channel name, which will enable the sensitivity menu.
   * @param {number} columnIndex Column index of the channel.
   * @param {number} yPos Position of the click in the y axis.
   * @param {number} chartAreaLeft Left position of the chart area.
   */
  handleChannelNameClick(columnIndex, yPos, chartAreaLeft) {
    const channelName = this.getDataTable().getColumnId(columnIndex);

    if (channelName === this.clickedChannelName_) {
      this.closeSensitivityMenu();
      return;
    }

    this.clickedChannelName_ = channelName;

    const channelActionsContainer =
        document.getElementById(this.channelActionsId_);
    const channelActionsTitle =
        document.getElementById(this.channelActionsTitleId_);

    channelActionsContainer.style.left = `${chartAreaLeft}px`;
    channelActionsContainer.style.top = `${yPos - 20}px`;
    channelActionsTitle.textContent = this.clickedChannelName_;

    channelActionsContainer.classList.remove('hidden');
  }

  /**
   * Closes the sensitivity menu and clears the channel click information.
   */
  closeSensitivityMenu() {
    document.getElementById(this.channelActionsId_).classList.add('hidden');
    document.getElementById(this.channelActionsTitleId_).textContent = '';
    this.clickedChannelName_ = null;
  }

  /**
   * Changes the sensitivity of the clicked channel.
   * @param {number} modifier Sensitivity modifier.
   * @private
   */
  changeClickedChannelSensitivity_(modifier) {
    if (this.changeChannelSensitivity_ && this.clickedChannelName_) {
      this.changeChannelSensitivity_(this.clickedChannelName_, modifier);
    }
  }

  /**
   * Increases the sensitivity of the clicked channel.
   */
  increaseSensitivity() {
    this.changeClickedChannelSensitivity_(2);
  }

  /**
   * Decreases the sensitivity of the clicked channel.
   */
  decreaseSensitivity() {
    this.changeClickedChannelSensitivity_(0.5);
  }

  /**
   * @override
   */
  shouldBeVisible(store) {
    const shouldBeVisible = !!store.chunkGraphData;
    if (!shouldBeVisible) {
      this.channelTransformations = new Map([]);
    }
    return shouldBeVisible;
  }

  /**
   * @override
   */
  handleChartData(store) {
    this.changeChannelSensitivity_ = (channelName, modifier) => {
      this.channelTransformations.set(
          channelName, this.channelTransformations.get(channelName) * modifier);
      this.handleChartData(store);
    };

    super.handleChartData(store);
  }
}

goog.addSingletonGetter(Graph);

exports = Graph;

