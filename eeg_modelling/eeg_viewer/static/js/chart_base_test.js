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

goog.module('eeg_modelling.eeg_viewer.ChartBase.tests');
goog.setTestOnly();

const AttributionMap = goog.require('proto.eeg_modelling.protos.PredictionChunk.AttributionMap');
const AttributionValues = goog.require('proto.eeg_modelling.protos.PredictionChunk.AttributionMap.AttributionValues');
const ChartBase = goog.require('eeg_modelling.eeg_viewer.ChartBase');
const ChunkScoreData = goog.require('proto.eeg_modelling.protos.PredictionMetadata.ChunkScoreData');
const DataTable = goog.require('google.visualization.DataTable');
const EventType = goog.require('goog.events.EventType');
const LineChart = goog.require('google.visualization.LineChart');
const MockControl = goog.require('goog.testing.MockControl');
const PredictionChunk = goog.require('proto.eeg_modelling.protos.PredictionChunk');
const ScoreData = goog.require('proto.eeg_modelling.protos.PredictionMetadata.ChunkScoreData.ScoreData');
const dom = goog.require('goog.dom');
const events = goog.require('goog.events');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;

let chartBase;
let storeData;
let overlayCanvas;
let chartContainer;


testSuite({

  setUp() {
    mockControl = new MockControl();

    chartBase = ChartBase.getInstance();

    // Define things to resemble a subclass, which would have these overridden
    chartBase.overlayId = 'overlay';
    chartBase.containerId = 'chart-container';
    chartBase.parentId = 'parent-container';
    chartBase.getNumSecs = (store) => Number(store.numSecs);

    storeData = {
      absStart: 2,
      annotations: [
        {startTime: 0, labelText: 'should not be added', id: null},
        {startTime: 42, labelText: 'do not trust the mice', id: null},
      ],
      chunkStart: 4,
      chunkDuration: 3,
      chunkGraphData: {
        cols: [
          {id: 'axis', label: 'axis', type: 'number'},
          {id: 'series 1', label: 'series 1', type: 'number'},
          {id: 'series 2', label: 'series 2', type: 'number'},
        ],
        rows: [
          {c: [{v: 41}, {v: 0}, {v: 1}]},
          {c: [{v: 42}, {v: 0}, {v: 1}]},
          {c: [{v: 43}, {v: 0}, {v: 1}]},
        ],
      },
      numSecs: 10,
      predictionChunkSize: 3,
      predictionChunkStart: 4,
      predictionMode: 'None',
      samplingFreq: 1,
      sensitivity: 4,
      seriesHeight: 14,
      timeScale: 1,
    };

    const parentContainer = document.createElement('div');
    parentContainer.id = 'parent-container';
    chartContainer = document.createElement('div');
    chartContainer.id = 'chart-container';
    chartContainer.style.position = 'absolute';
    chartContainer.style.top = '5px';
    chartContainer.style.left = '11px';
    parentContainer.appendChild(chartContainer);
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.id = 'overlay';
    parentContainer.appendChild(overlayCanvas);
    document.body.appendChild(parentContainer);
  },

  testAddAnnotations() {
    const dataTable = new DataTable(storeData.chunkGraphData);
    chartBase.addAnnotations(storeData, dataTable);
    assertEquals('do not trust the mice', dataTable.getValue(1, 1));
  },

  testGetRenderOffset() {
    assertEquals(14, chartBase.getRenderOffset(1, storeData));
    assertEquals(0, chartBase.getRenderOffset(2, storeData));
  },

  testSizeAndPosition() {
    const boundBox = {top: 5, left: 3, width: 42, height: 77};
    const mockGetCli = mockControl.createMethodMock(chartBase,
        'getChartLayoutInterface');
    mockGetCli().$returns({
      getChartAreaBoundingBox: () => boundBox,
    });

    mockControl.$replayAll();

    chartBase.sizeAndPositionOverlay();

    mockControl.$verifyAll();
    assertEquals('10px', overlayCanvas.style.top);
    assertEquals('14px', overlayCanvas.style.left);
    assertEquals(42, overlayCanvas.width);
    assertEquals(77, overlayCanvas.height);
  },

  testCreateHTicks() {
    const mockGetHTickValues = mockControl.createMethodMock(chartBase,
        'getHTickValues');
    mockGetHTickValues(storeData).$returns([...Array(20).keys()]);
    mockControl.$replayAll();

    const hTicks = chartBase.createHTicks(storeData);

    mockControl.$verifyAll();

    const expectedTicks = [
      {v: 0, f: '00:00:02'},
      {v: 2, f: '00:00:04'},
      {v: 4, f: '00:00:06'},
      {v: 6, f: '00:00:08'},
      {v: 8, f: '00:00:10'},
      {v: 10, f: '00:00:12'},
      {v: 12, f: '00:00:14'},
      {v: 14, f: '00:00:16'},
      {v: 16, f: '00:00:18'},
      {v: 18, f: '00:00:20'},
    ];
    assertObjectEquals(expectedTicks, hTicks);
  },

  testCreateVTicks() {
    const mockGetVTickDisplayValues = mockControl.createMethodMock(chartBase,
        'getVTickDisplayValues');
    mockGetVTickDisplayValues(storeData).$returns(['series 1', 'series 2']);
    mockControl.$replayAll();

    const vTicks = chartBase.createVTicks(storeData);

    mockControl.$verifyAll();

    const expectedTicks = [
      {v: 14, f: 'series 1'},
      {v: 0, f: 'series 2'},
    ];
    assertObjectEquals(expectedTicks, vTicks);
  },

  testCreateDataTable() {
    const mockGetNumSecs = mockControl.createMethodMock(chartBase,
        'getNumSecs');
    mockGetNumSecs(storeData).$returns(1);
    const mockGetStart = mockControl.createMethodMock(chartBase,
        'getStart');
    mockGetStart(storeData).$returns(0).$times(2);
    mockControl.$replayAll();

    const expectedTable = new DataTable({
      cols: [
        {id: '', pattern: '', label: 'seconds', type: 'number'},
        {id: '', pattern: '', label: 'placeholder', type: 'number'},
        {id: '', pattern: '', label: 'placeholder', type: 'number'},
      ],
      rows: [
        {c: [{v: 0}, {v: 14}, {v: 0}]},
        {c: [{v: 1}, {v: 14}, {v: 0}]},
      ],
    });

    const dataTable = chartBase.createDataTable(storeData);

    mockControl.$replayAll();

    assertObjectEquals(expectedTable, dataTable);
  },

  testUpdateChartOptions() {
    const mockParent = mockControl.createMethodMock(chartBase, 'getParent');
    mockParent().$returns({clientHeight: 500});

    const mockCreateHTicks = mockControl.createMethodMock(chartBase,
        'createHTicks');
    mockCreateHTicks(storeData).$returns([0, 1, 1, 2]);

    const mockCreateVTicks = mockControl.createMethodMock(chartBase,
        'createVTicks');
    mockCreateVTicks(storeData).$returns(['a', 'b', 'c']);

    const mockSet = mockControl.createMethodMock(chartBase, 'setOption');
    mockSet('height', 460).$once();
    mockSet('chartArea.height', '89%').$once();
    mockSet('hAxis.ticks', [0, 1, 1, 2]).$once();
    mockSet('vAxis.ticks', ['a', 'b', 'c']).$once();
    mockControl.$replayAll();

    chartBase.updateChartOptions(storeData);

    mockControl.$verifyAll();
  },

  testInitChart() {
    chartBase.initChart();

    const lineChartInstance = (new LineChart(document.createElement('div')));
    assertObjectEquals(chartBase.chart.prototype, lineChartInstance.prototype);
  },

  testHandleDraw() {
    chartBase.dataTable = new DataTable({
      cols: [
        {id: '', pattern: '', label: 'seconds', type: 'number'},
        {id: '', pattern: '', label: 'placeholder', type: 'number'},
      ],
      rows: [
        {c: [{v: 0}, {v: 0}]},
        {c: [{v: 1}, {v: 0}]},
      ],
    });
    chartBase.chartOptions = {
      height: '90%',
      width: '90%',
    };

    const mockUpdateOptions = mockControl.createMethodMock(chartBase,
        'updateChartOptions');
    mockUpdateOptions(storeData).$once();

    const mockContainer = {style: {height: ''}};
    const mockGetContainer = mockControl.createMethodMock(chartBase,
        'getContainer');
    mockGetContainer().$returns(mockContainer);

    const mockGet = mockControl.createMethodMock(chartBase, 'getOption');
    mockGet('height').$returns(42);

    const mockChart = mockControl.createStrictMock(LineChart);
    mockChart.draw(chartBase.dataTable, chartBase.chartOptions).$once();
    chartBase.chart = mockChart;

    const mockOverlay = mockControl.createMethodMock(chartBase,
        'createOverlay');
    mockOverlay(storeData).$once();

    const mockResize = (storeData) => 42;
    chartBase.resizeHandler = mockResize;
    const mockUnlisten = mockControl.createMethodMock(events, 'unlisten');
    mockUnlisten(window, EventType.RESIZE, mockResize).$once();

    const mockDraw = () => 42;
    const mockCreateResizeHandler = mockControl.createMethodMock(chartBase,
        'createResizeHandler');
    mockCreateResizeHandler(storeData).$returns(mockDraw);

    const mockListen = mockControl.createMethodMock(events, 'listen');
    mockListen(window, EventType.RESIZE, mockDraw).$once();

    mockControl.$replayAll();

    chartBase.handleDraw(storeData);

    mockControl.$verifyAll();

    assertEquals('42px', mockContainer.style.height);
  },

  testHandleChartData() {
    const fakeDatatable = new DataTable({
      cols: [
        {id: '', pattern: '', label: 'seconds', type: 'number'},
        {id: '', pattern: '', label: 'placeholder', type: 'number'},
      ],
      rows: [
        {c: [{v: 0}, {v: 0}]},
        {c: [{v: 1}, {v: 0}]},
      ],
    });
    const mockCreateDatatable = mockControl.createMethodMock(chartBase,
        'createDataTable');
    mockCreateDatatable(storeData).$returns(fakeDatatable);

    const mockHandleDraw = mockControl.createMethodMock(chartBase,
        'handleDraw');
    mockHandleDraw(storeData).$once();

    mockControl.$replayAll();

    chartBase.handleChartData(storeData);

    mockControl.$verifyAll();

    assertObjectEquals(fakeDatatable, chartBase.getDataTable());
  },

  testDrawChunkScores() {
    const scoreData = new ScoreData();
    scoreData.setPredictedValue(0.42);

    const chunkScoreData = new ChunkScoreData();
    chunkScoreData.setDuration(1);
    chunkScoreData.setStartTime(1);
    chunkScoreData.getScoreDataMap().set('test label', scoreData);
    storeData.chunkScores = [chunkScoreData];
    storeData.label = 'test label';

    const boundBox = {top: 5, left: 3, width: 42, height: 77};
    const mockGetCli = mockControl.createMethodMock(chartBase,
        'getChartLayoutInterface');
    mockGetCli().$returns({
      getChartAreaBoundingBox: () => boundBox,
      getXLocation: (x) => x,
    });

    const mockCanvas = mockControl.createMethodMock(chartBase, 'getCanvas');
    mockCanvas().$returns({
      width: 3,
      height: 42,
    });

    const mockFill = mockControl.createFunctionMock('fillRect');
    mockFill(-2, 0, 1, 77).$once();
    const mockClear = mockControl.createFunctionMock('clearRect');
    mockClear(0, 0, 3, 42).$once();
    const mockGetContext = mockControl.createMethodMock(chartBase,
        'getContext');
    mockGetContext().$returns({
      fillRect: mockFill,
      clearRect: mockClear,
    });
    mockControl.$replayAll();

    chartBase.drawChunkScores(storeData);

    mockControl.$verifyAll();
  },

  testDrawAttributionMap() {
    const attributionValuesCh1 = new AttributionValues();
    attributionValuesCh1.setAttributionList([0, 0.2]);
    const attributionValuesCh2 = new AttributionValues();
    attributionValuesCh2.setAttributionList([0.3, 0.7]);

    const attributionMap = new AttributionMap();
    attributionMap.getAttributionMapMap().set('CHAN1', attributionValuesCh1);
    attributionMap.getAttributionMapMap().set('CHAN2', attributionValuesCh2);
    const predChunk = new PredictionChunk();
    predChunk.getAttributionDataMap().set('test label', attributionMap);
    storeData.attributionMaps = predChunk.getAttributionDataMap();
    storeData.label = 'test label';
    storeData.channelIds = ['CHAN1', 'CHAN2'];

    const boundBox = {top: 5, left: 3, width: 42, height: 76};
    const mockGetCli = mockControl.createMethodMock(chartBase,
        'getChartLayoutInterface');
    mockGetCli().$returns({
      getChartAreaBoundingBox: () => boundBox,
      getXLocation: (x) => x,
    });

    const mockCanvas = mockControl.createMethodMock(chartBase, 'getCanvas');
    mockCanvas().$returns({
      width: 3,
      height: 42,
    });

    const mockFill = mockControl.createFunctionMock('fillRect');
    mockFill(1, 0, 1.5, 38).$once();
    mockFill(2.5, 0, 1.5, 38).$once();
    mockFill(1, 38, 1.5, 38).$once();
    mockFill(2.5, 38, 1.5, 38).$once();
    const mockClear = mockControl.createFunctionMock('clearRect');
    mockClear(0, 0, 3, 42).$once();
    const mockGetContext = mockControl.createMethodMock(chartBase,
        'getContext');
    mockGetContext().$returns({
      fillRect: mockFill,
      clearRect: mockClear,
    });
    mockControl.$replayAll();

    chartBase.drawAttributionMap(storeData);

    mockControl.$verifyAll();
  },

  tearDown() {
    dom.removeChildren(dom.getDocument().body);
    mockControl.$tearDown();
  },
});
