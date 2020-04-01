// Copyright 2020 The Google Research Authors.
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

goog.module('eeg_modelling.eeg_viewer.ChartBase');

const DataTable = goog.require('google.visualization.DataTable');
const EventType = goog.require('goog.events.EventType');
const LineChart = goog.require('google.visualization.LineChart');
const Store = goog.require('eeg_modelling.eeg_viewer.Store');
const array = goog.require('goog.array');
const events = goog.require('goog.events');
const formatter = goog.require('eeg_modelling.eeg_viewer.formatter');
const gvizEvents = goog.require('google.visualization.events');
const object = goog.require('goog.object');
const {assert, assertInstanceof, assertObject, assertString} = goog.require('goog.asserts');

/**
 * @typedef {{
 *   annotations: (undefined|{
 *     boxStyle: {
 *       stroke: (string|undefined),
 *       strokeWidth: (number|undefined),
 *       rx: (number|undefined),
 *       ry: (number|undefined),
 *       gradient: {
 *         color1: (string|undefined),
 *         color2: (string|undefined),
 *         x1: (string|undefined),
 *         y1: (string|undefined),
 *         x2: (string|undefined),
 *         y2: (string|undefined),
 *         useObjectBoundingBoxUnits: (boolean|undefined),
 *       },
 *     },
 *     textStyle: {
 *       fontSize: (number|undefined),
 *       bold: (boolean|undefined),
 *     },
 *   }),
 *   chartArea: {
 *     backgroundColor: (string|{
 *       stroke: (string|undefined),
 *       strokeWidth: (number|undefined),
 *     }),
 *     height: (string|undefined),
 *     width: (string|undefined),
 *     top: (number|undefined),
 *     left: (number|undefined),
 *   },
 *   colors: ?Array<string>,
 *   crosshair: {
 *     color: string,
 *     orientation: string,
 *     selected: {
 *       color: string,
 *     },
 *     trigger: string,
 *   },
 *   dataOpacity: (number|undefined),
 *   hAxis: {
 *     baselineColor: string,
 *     gridlines: {
 *       color: string
 *     },
 *     textStyle: {
 *       color: string,
 *       fontSize: number,
 *     },
 *     ticks: !Array<!Tick>,
 *   },
 *   height: (number|undefined),
 *   legend: {
 *     position: string,
 *   },
 *   lineWidth: number,
 *   selectionMode: (string|undefined),
 *   tooltip: {
 *     trigger: string,
 *     isHtml: (boolean|undefined),
 *   },
 *   vAxis: {
 *     baselineColor: string,
 *     gridlines: {
 *       color: string
 *     },
 *     ticks: !Array<!Tick>,
 *     textStyle: {
 *       fontSize: number,
 *     },
 *     viewWindow: {
 *       min: number,
 *       max: number,
 *     },
 *   },
 *   width: (number|undefined),
 * }}
 */
let ChartOptions;

/**
 * @typedef {{
 *   left: number,
 *   top: number,
 *   width: number,
 *   height: number,
 * }}
 */
let ChartArea;

/**
 * @typedef {{
 *   type: string,
 *   handler: function(?Event):void,
 * }}
 */
let ChartListener;

/**
 * @typedef {{
 *   v: number,
 *   f: string,
 * }}
 */
let Tick;

/**
 * @typedef {{
 *   color: string,
 *   fill: boolean,
 * }}
 */
let HighlightViewportStyle;

/**
 * @typedef {{
 *   startX: (number|undefined),
 *   endX: (number|undefined),
 *   color: (string|undefined),
 *   fill: (boolean|undefined),
 *   height: (number|undefined),
 *   top: (number|undefined),
 *   minWidth: (number|undefined),
 * }}
 */
let OverlayElement;

/**
 * @typedef {
 *   function(!Store.StoreData, !ChartArea=):!Array<!OverlayElement>
 * }
 */
let GetElementsToDraw;

/**
 * @typedef {{
 *   name: string,
 *   getElementsToDraw: !GetElementsToDraw,
 * }}
 */
let OverlayLayer;


const /** @type {{LineChart}} */ chartDep = {LineChart};
const /** @type {number} */ marginHeight = 20;
const /** @type {number} */ axisLabelHeight = 50;


/**
 * Manages a Google Charts Wrapper instance and an optional overlay.
 * @abstract
 */
class ChartBase {

  static getChartDep() {
    return chartDep;
  }

  /**
   * Returns a boolean indicating if one of properties is in changedProperties.
   * Utility function used to determine if the charts should update.
   * @param {!Array<?Store.Property>} changedProperties
   * @param {!Array<?Store.Property>} properties
   * @return {boolean} Indicate if there is a property present in both arrays.
   */
  static changedPropertiesIncludeAny(changedProperties, properties) {
    return properties.some((prop) => changedProperties.includes(prop));
  }

  constructor() {
    /** @public {?LineChart} */
    this.chart = null;

    /** @public {?DataTable} */
    this.dataTable = null;

    // These options are defined in the Google Charts Line Chart API.
    // https://developers.google.com/chart/interactive/docs/gallery/linechart#configuration-options
    /** @public {!ChartOptions} */
    this.chartOptions = {
      // Does not include border for axis labels.
      chartArea: {
        backgroundColor: {
          stroke: '#696969',
          strokeWidth: 1,
        },
        width: '90%',
      },
      colors: ['#696969', '#696969'],
      crosshair: {
        color: '#696969',
        orientation: 'vertical',
        selected: {
          color: '#696969',
        },
        trigger: 'both',
      },
      hAxis: {
        baselineColor: 'black',
        gridlines: {
          color: 'black',
        },
        textStyle: {
          color: '#000',
          fontSize: 14,
        },
        ticks: [],
      },
      legend: {
        position: 'none',
      },
      lineWidth: 1,
      tooltip: {
        trigger: 'none',
      },
      vAxis: {
        baselineColor: '#fff',
        gridlines: {
          color: '#fff',
        },
        ticks: [],
        textStyle: {
          fontSize: 14,
        },
        viewWindow: {
          min: 0,
          max: 1,
        },
      },
    };

    /** @public {!Object<!Store.PredictionMode, number>} */
    this.height = {
      // The values represent the percentage of the height of the parent
      // container in decimal format.
      [Store.PredictionMode.NONE]: 1.0,
      [Store.PredictionMode.CHUNK_SCORES]: 1.0,
      [Store.PredictionMode.ATTRIBUTION_MAPS]: 1.0,
    };

    /** @type {?string} */
    this.overlayId = null;

    /** @protected {?HighlightViewportStyle} */
    this.highlightViewportStyle = null;

    /** @protected {!Array<!OverlayLayer>} */
    this.overlayLayers = [];

    /** @type {?string} */
    this.containerId = null;

    /** @type {?string} */
    this.parentId = 'parent-chart-container';

    /**
     * Handler called when an immediate redraw is required (e.g. when resizing
     * the screen).
     * @protected {?function(this:ChartBase):void}
     */
    this.redrawHandler = null;

    /**
     * Handler called when a data update and redraw are required (e.g. when
     * updating the sensitivity of a channel).
     * @protected {?function(this:ChartBase):void}
     */
    this.updateDataAndRedrawHandler = null;

    /** @type {!Array<!ChartListener>} */
    this.chartListeners = [];

    /** @private {boolean} */
    this.listenersRegistered_ = false;

    /** @private {boolean} */
    this.visible_ = false;
  }

  generateColors(num, color) {
    return array.range(0, num).map(x => color);
  }

  /**
   * Returns the option value given a key string where nested keys are separated
   * by periods.
   * @param {string} keyString The key(s) to look up the value for.
   * @return {?(string|number|!Array|!Object)} The value.
   */
  getOption(keyString) {
    const keys = keyString ? keyString.split('.') : [];
    const value = object.getValueByKeys(this.chartOptions, ...keys);
    return /** @type {?(string|number|!Array|!Object)} */ (value);
  }

  /**
   * Sets the chart option value given a key string where nested keys are
   * separated by periods and a value.
   * @param {string} keyString The key(s) to replace the value of.
   * @param {!Object|number|string} value The value to set the option to.
   */
  setOption(keyString, value) {
    const keys = keyString.split('.');
    // Given a key string 'A.B.C', we want the object at chartOptions[A][B],
    // which we will call the 'proximal object' so that we can set the value at
    // the key C, which we will call the 'proximal key',
    // proximalObject[proximalKey] = value.
    const proximalObjectKeys = keys.slice(0, keys.length - 1);
    const proximalKey = keys[keys.length - 1];
    const proximalObject = this.getOption(proximalObjectKeys.join('.'));
    assertObject(proximalObject);
    proximalObject[proximalKey] = value;
  }

  /**
   * Derives vertical offset for series data.
   * @param {number} index Index of the series in the data.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {number} Y-axis offset for rendering the series at the given index.
   */
  getRenderOffset(index, store) {
    const numSeries = store.chunkGraphData.cols.length;
    return store.seriesHeight * (numSeries - index - 1);
  }

  /**
   * Formats tick for Google Chart consumption.
   * @param {number} value The horizontal axis value of the tick.
   * @param {string} displayValue The value to display with the tick.
   * @return {!Tick} A tick formatted for Google Charts.
   */
  formatTick(value, displayValue) {
    return {v: value, f: displayValue};
  }

  /**
   * Returns the horizontal tick values in relative time.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {!Array<number>} A list of tick values corresponding to the axis.
   */
  getHTickValues(store) {
    return [];
  }

  /**
   * Creates ticks for the horizontal axis that display values in absolute time.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {!Array<!Tick>} A list of ticks.
   */
  createHTicks(store) {
    const tickValues = this.getHTickValues(store);
    // Limits labels on the ticks so they fit in the view.
    const labelFreq = Math.ceil(tickValues.length / 10);
    return tickValues.filter((val, i) => i % labelFreq == 0)
        .map((val) => this.formatTick(val, formatter.formatTime(
            val + store.absStart)));
  }

  /**
   * Returns the vertical tick display values.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {!Array<string>} A list of tick display values.
   */
  getVTickDisplayValues(store) {
    return [];
  }

  /**
   * Creates a ticker that generates data series labels on the y-axis.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {!Array<!Tick>} A list of ticks.
   */
  createVTicks(store) {
    const tickDisplayValues = this.getVTickDisplayValues(store);
    return tickDisplayValues.map((val, i) => this.formatTick(
        this.getRenderOffset(i + 1, store), val));
  }

  /**
   * Returns the start of the underlying data relative to waveform file start.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {number} The second offset from the start of the file.
   * @abstract
   */
  getStart(store) {}

  /**
   * Returns the number of seconds of data represented in the chart.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {number} Number of seconds of data.
   * @abstract
   */
  getNumSecs(store) {}

  /**
   * Creates a DataTable to load the Line Chart with.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @return {!DataTable} A DataTable with an initialized x-axis.
   */
  createDataTable(store) {
    const numSeries = store.chunkGraphData.cols.length - 1;
    const dataTable = new DataTable();
    dataTable.addColumn('number', 'seconds');
    for (let i = 0; i < numSeries; i++) {
      dataTable.addColumn('number', 'placeholder');
    }
    const axisData = array.range(0, (this.getNumSecs(store) + 1))
        .map((i) => [i + this.getStart(store)]);
    const colData = array.range(0, numSeries)
        .map((x, i) => (numSeries - i - 1) * store.seriesHeight);
    const rowData = axisData.map(x => x.concat(colData));

    dataTable.addRows(rowData);
    return dataTable;
  }

  /**
   * Updates the chart options object with the store data and returns it.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   */
  updateChartOptions(store) {
    const percentage = this.height[store.predictionMode];
    const parentHeight = this.getParent().clientHeight;
    const containerHeight = Math.ceil(
        percentage * parentHeight) - 2 * marginHeight;
    this.setOption('height', containerHeight);
    const chartPercentage = 1 - axisLabelHeight / containerHeight;
    this.setOption('chartArea.height',
        `${Math.floor(100 * chartPercentage)}%`);
    this.setOption('hAxis.ticks', this.createHTicks(store));
    this.setOption('vAxis.ticks', this.createVTicks(store));
  }

  /**
   * @protected
   * Returns a boolean indicating if the chart should be displayed in the
   * screen.
   * @param {!Store.StoreData} store Store object.
   * @return {boolean}
   */
  shouldBeVisible(store) {
    const numSecs = this.getNumSecs(store);
    const height = this.height[store.predictionMode];
    return numSecs > 0 && height > 0;
  }

  /**
   * @private
   * Shows or hides the chart container and overlay in the screen, and set the
   * this.visible_ property.
   * @param {boolean} visible
   */
  setVisibility_(visible) {
    this.getContainer().classList.toggle('hidden', !visible);
    if (this.overlayId) {
      document.getElementById(this.overlayId)
          .classList.toggle('hidden', !visible);
    }
    this.visible_ = visible;
  }

  /**
   * Returns a boolean indicating if the chart is currently drawn in the screen.
   * @return {boolean}
   */
  isVisible() {
    return this.visible_;
  }

  /**
   * Initializes a Line Chart.
   */
  initChart() {
    if (!this.chart) {
      this.chart = new
          chartDep.LineChart(document.getElementById(
              assertString(this.containerId)));
    }
  }

  /**
   * @private
   * Clears the chart and releases its resources.
   */
  clearChart_() {
    if (this.chart) {
      this.chart.clearChart();
    }
  }

  /**
   * Add a chart event listener to the list that will be added on initialization
   * of the chart to listen to an event emitted by the underlying Google Chart.
   * @param {string} eventType The event type (e.g. click or select).
   * @param {function(?Event):void} eventHandler Function called in response.
   */
  registerChartEventListener(eventType, eventHandler) {
    this.chartListeners.push({
      type: eventType,
      handler: eventHandler,
    });
  }

  /**
   * Removes all chart event listeners.
   */
  removeChartEventListeners() {
    if (this.chart) {
      gvizEvents.removeAllListeners(this.getChart());
    }
  }

  /**
   * Adds event listeners through the Google Charts API.
   */
  addChartEventListeners() {
    if (this.chart && !this.listenersRegistered_) {
      this.chartListeners.forEach((chartListener) => {
        gvizEvents.addListener(this.getChart(), chartListener.type,
            chartListener.handler);
      });
      this.listenersRegistered_ = true;
    }
  };

  /**
   * Draws the chart content, with an updated configuration given the most
   * recent store state.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   */
  drawContent(store) {
    this.updateChartOptions(store);
    this.getContainer().style.height = `${this.getOption('height')}px`;
    this.chart.draw(this.dataTable, this.chartOptions);
  }

  /**
   * Registers a callback to redraw the chart when the window resizes, bound to
   * the most recent store state.
   * @param {!Store.StoreData} store Store data.
   * @private
   */
  registerResizeHandler_(store) {
    if (this.redrawHandler) {
      events.unlisten(window, EventType.RESIZE, this.redrawHandler);
    }
    this.redrawHandler = () => {
      this.drawContent(store);
      this.drawOverlay(store);
    };
    events.listen(window, EventType.RESIZE, this.redrawHandler);
  }

  /**
   * Creates a handler to update the data and redraw, bound to the most recent
   * store state.
   * Use this handler when the chart must be redrawn (also updating the
   * dataTable) from a direct call, not in a change in the store.
   * E.g., when the user changes the sensitivity of a channel.
   * @param {!Store.StoreData} store Store data.
   * @private
   */
  createUpdateDataAndRedrawHandler_(store) {
    this.updateDataAndRedrawHandler = () => {
      this.clearChart_();
      this.dataTable = this.createDataTable(store);
      this.drawContent(store);
      this.drawOverlay(store);
    };
  }

  /**
   * Returns a boolean indicating if the data table of the chart should be
   * updated.
   * @param {!Store.StoreData} store Data from the store.
   * @param {!Array<!Store.Property>} changedProperties Properties that changed
   *     in the last action dispatched.
   * @return {boolean}
   * @abstract
   */
  shouldUpdateData(store, changedProperties) {}

  /**
   * Returns a boolean indicating if the chart content should be updated, but
   * updating the data table is not needed (e.g. resize the window).
   * @param {!Store.StoreData} store Data from the store.
   * @param {!Array<!Store.Property>} changedProperties Properties that changed
   *     in the last action dispatched.
   * @return {boolean}
   * @abstract
   */
  shouldRedrawContent(store, changedProperties) {}

  /**
   * Returns a boolean indicating if the chart overlay should be updated.
   * @param {!Store.StoreData} store Data from the store.
   * @param {!Array<!Store.Property>} changedProperties Properties that changed
   *     in the last action dispatched.
   * @return {boolean}
   * @abstract
   */
  shouldRedrawOverlay(store, changedProperties) {}

  /**
   * Updates or initializes the chart with the newest data.
   * Always use this method as callback when a property in the store changes,
   * and override the shouldSomething() methods to decide what should be
   * updated/redrawn or not.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @param {!Array<!Store.Property>} changedProperties Store properties that
   *     changed in the last action.
   */
  handleChartData(store, changedProperties) {
    const visible = this.shouldBeVisible(store);
    this.setVisibility_(visible);
    if (!visible) {
      return;
    }
    this.initChart();
    this.registerResizeHandler_(store);
    this.createUpdateDataAndRedrawHandler_(store);

    const shouldUpdateData =
        !this.dataTable || this.shouldUpdateData(store, changedProperties);
    const shouldRedrawContent =
        shouldUpdateData || this.shouldRedrawContent(store, changedProperties);
    const shouldRedrawOverlay = shouldRedrawContent ||
        this.shouldRedrawOverlay(store, changedProperties);

    if (shouldUpdateData) {
      this.clearChart_();
      this.dataTable = this.createDataTable(store);
    }

    if (shouldRedrawContent) {
      this.drawContent(store);
    }

    if (shouldRedrawOverlay) {
      this.drawOverlay(store);
    }

    this.addChartEventListeners();
  }

  getDataTable() {
    return this.dataTable;
  }

  getContainer() {
    return document.querySelector(`#${this.containerId}`);
  }

  getCanvas() {
    return document.querySelector(`#${this.overlayId}`);
  }

  getParent() {
    return document.querySelector(`#${this.parentId}`);
  }

  getContext() {
    return this.getCanvas().getContext('2d');
  }

  /**
   * Returns the Line Chart inside the Chart Wrapper.
   * @return {?LineChart} The Line Chart inside the wrapper.
   */
  getChart() {
    return this.chart;
  }

  /**
   * Returns the Chart Layout Interface that provides translation functions
   * between the DOM and the Chart.
   * @return {!google.visualization.ChartLayoutInterface} The chart's layout interface.
   */
  getChartLayoutInterface() {
    assert(this.getChart());
    const cli = this.getChart().getChartLayoutInterface();
    assert('getXLocation' in cli);
    assertInstanceof(cli.getXLocation, Function);
    assert('getChartAreaBoundingBox' in cli);
    assertInstanceof(cli.getChartAreaBoundingBox, Function);

    return /** @type {!google.visualization.ChartLayoutInterface} */(
        this.getChart().getChartLayoutInterface());
  }

  /**
   * Creates the overlay.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   */
  drawOverlay(store) {
    if (!this.overlayId) {
      return;
    }
    this.sizeAndPositionOverlay();
    this.drawOverlayLayers(store);
  }

  /**
   * Positions and sizes the canvas overlay to the underlying Line Chart.
   */
  sizeAndPositionOverlay() {
    const canvas = this.getCanvas();
    const chartArea = this.getChartLayoutInterface().getChartAreaBoundingBox();
    // Position canvas over chart area
    const container = this.getContainer();
    canvas.style.top = `${chartArea.top + container.offsetTop}px`;
    canvas.style.left = `${chartArea.left + container.offsetLeft}px`;
    canvas.width = chartArea.width;
    canvas.height = chartArea.height;
  }

  /**
   * Draws the overlay layers and highlights the viewport if needed.
   * The different layers are defined in the overlayLayers field, and are used
   * to represent different type of data being drawn in the overlay. The
   * viewport highlighting is considered a special type of layer.
   * The subclass should override the overlayLayers field and the
   * highlightViewportStyle to customize what is drawn.
   * @param {!Store.StoreData} store Store object containing request chunk data.
   * @protected
   */
  drawOverlayLayers(store) {
    const canvas = this.getCanvas();
    const context = this.getContext();
    context.clearRect(0, 0, canvas.width, canvas.height);
    const cli = this.getChartLayoutInterface();
    const chartArea = cli.getChartAreaBoundingBox();

    /** @type {!OverlayElement} */
    const defaultConfig = {
      height: chartArea.height,
      top: 0,
      minWidth: 5,
    };

    const drawElement = (element) => {
      /** @type {!OverlayElement} */
      const drawConfig = Object.assign({}, defaultConfig, element);

      const startX = cli.getXLocation(drawConfig.startX) - chartArea.left;
      const endX = cli.getXLocation(drawConfig.endX) - chartArea.left;
      const width = Math.max(endX - startX, drawConfig.minWidth);

      if (drawConfig.fill) {
        context.fillStyle = drawConfig.color;
        context.fillRect(startX, drawConfig.top, width, drawConfig.height);
      } else {
        context.strokeStyle = drawConfig.color;
        context.lineWidth = 8;
        context.strokeRect(startX, drawConfig.top, width, drawConfig.height);
      }
    };

    this.overlayLayers.forEach((layer) => {
      const layerElements = layer.getElementsToDraw(store, chartArea);
      layerElements.forEach(drawElement);
    });

    if (this.highlightViewportStyle) {
      drawElement(Object.assign({}, this.highlightViewportStyle, {
        startX: store.chunkStart,
        endX: store.chunkStart + store.chunkDuration,
      }));
    }
  }

  /**
   * Creates an opacity [0, 1] from a number (-inf, inf).
   * @param {number} score Prediction score ranging from (-inf, inf)
   * @return {number} An opacity.
   */
  getOpacity(score) {
    const scale = 0.5;
    return 1.0 / (1.0 + Math.exp(-scale * score));
  }
}

goog.addSingletonGetter(ChartBase);

exports = ChartBase;
exports.Tick = Tick;
exports.OverlayElement = OverlayElement;
exports.ChartArea = ChartArea;
exports.GetElementsToDraw = GetElementsToDraw;
