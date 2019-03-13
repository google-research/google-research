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

goog.module('eeg_modelling.eeg_viewer.Graph.tests');
goog.setTestOnly();

const DataTable = goog.require('google.visualization.DataTable');
const Graph = goog.require('eeg_modelling.eeg_viewer.Graph');
const LineChart = goog.require('google.visualization.LineChart');
const MockControl = goog.require('goog.testing.MockControl');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;

let graph;
let storeData;
let expectedFormattedData;
let annotatedData;


testSuite({

  setUp() {
    mockControl = new MockControl();

    graph = new Graph();

    storeData = {
      absStart: 2,
      chunkGraphData: {
        cols: [
          {id: 'axis', label: 'axis', type: 'number'},
          {id: 'series 1', label: 'series 1', type: 'number'},
          {id: 'series 2', label: 'series 2', type: 'number'},
        ],
        rows: [
          {c: [{v: 40}, {v: 0}, {v: 1}]},
          {c: [{v: 41}, {v: 0}, {v: 1}]},
          {c: [{v: 42}, {v: 0}, {v: 1}]},
          {c: [{v: 43}, {v: 0}, {v: 1}]},
          {c: [{v: 44}, {v: 0}, {v: 1}]},
          {c: [{v: 45}, {v: 0}, {v: 1}]},
          {c: [{v: 46}, {v: 0}, {v: 1}]},
          {c: [{v: 47}, {v: 0}, {v: 1}]},
          {c: [{v: 48}, {v: 0}, {v: 1}]},
          {c: [{v: 49}, {v: 0}, {v: 1}]},
        ],
      },
      annotations: [
        {startTime: 0, labelText: 'should not be added'},
        {startTime: 42, labelText: 'do not trust the mice'},
      ],
      chunkStart: 40,
      chunkDuration: 10,
      fileType: 'EEG',
      predictionMode: 'None',
      samplingFreq: 1,
      seriesHeight: 42,
      sensitivity: 1,
      timeScale: 1,
    };

    expectedFormattedData = {
      cols: [
        {id: 'axis', label: 'axis', type: 'number'},
        {id: 'series 1', label: 'series 1', type: 'number'},
        {id: 'series 2', label: 'series 2', type: 'number'}
      ],
      rows: [
        {c: [{v: 40}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 41}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 42}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 43}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 44}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 45}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 46}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 47}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 48}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 49}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
      ],
    };

    annotatedData = {
      cols: [
        {id: 'axis', label: 'axis', type: 'number'},
        {id: '', label: '', pattern: '', type: 'string',
         p: {role: 'annotation'}},
        {id: 'series 1', label: 'series 1', type: 'number'},
        {id: 'series 2', label: 'series 2', type: 'number'},
      ],
      rows: [
        {c: [{v: 40}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 41}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {
          c: [
            {v: 42}, {v: 'do not trust the mice'}, {v: 42, f: '0'},
            {v: 7, f: '1'}
          ]
        },
        {c: [{v: 43}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 44}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 45}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 46}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 47}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 48}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
        {c: [{v: 49}, {v: null}, {v: 42, f: '0'}, {v: 7, f: '1'}]},
      ],
    };

    graph.parentId = 'parent-chart-container';
    graph.containerId = 'line-chart-container';

    const parentContainer = document.createElement('div');
    parentContainer.id = graph.parentId;
    const chunk = document.createElement('div');
    chunk.id = graph.containerId;
    parentContainer.appendChild(chunk);
    document.body.appendChild(parentContainer);
  },

  testGetRenderTransformation_Default() {
    graph.channelTransformations = new Map([]);
    storeData.fileType = 'DEFAULT';
    assertEquals(1, graph.getRenderTransformation(
        'DEFAULT', storeData));
  },

  testGetRenderTransformation_SavedTransformation() {
    graph.channelTransformations = new Map([['TEST', 7]]);
    assertEquals(7, graph.getRenderTransformation('TEST', storeData));
  },

  testGetRenderTransformation_Eeg() {
    graph.channelTransformations = new Map([]);
    storeData.fileType = 'EEG';
    assertEquals(21, graph.getRenderTransformation('SZ_BIN', storeData));
    assertEquals(3.5, graph.getRenderTransformation('EKG', storeData));
    assertEquals(7, graph.getRenderTransformation('DEFAULT', storeData));
  },

  testGetRenderTransformation_Ekg() {
    graph.channelTransformations = new Map([]);
    storeData.fileType = 'EKG';
    assertEquals(20, graph.getRenderTransformation('DEFAULT', storeData));
  },

  testGetHTickValues() {
    assertArrayEquals([40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        graph.getHTickValues(storeData));
  },

  testGetVTickDisplayValues() {
    assertArrayEquals(['series 1', 'series 2'],
        graph.getVTickDisplayValues(storeData));
  },

  testFormatData() {
    const formattedData =
        graph.formatDataForRendering(storeData, storeData.chunkGraphData);
    assertObjectEquals(new DataTable(expectedFormattedData), formattedData);
  },

  testCreateDataTable() {
    const datatable = graph.createDataTable(storeData);
    assertObjectEquals(new DataTable(annotatedData), datatable);
  },

  testUpdateChartOptions() {
    const mockParent = mockControl.createMethodMock(graph, 'getParent');
    mockParent().$returns({clientHeight: 500});

    const mockSet = mockControl.createMethodMock(graph, 'setOption');
    mockSet('tooltip.trigger', 'selection').$once();
    mockSet('vAxis.viewWindow', {min: -84, max: 126}).$once();
    mockSet('colors', ['#696969', '#696969', '#696969']).$once();
    mockSet('height', 460).$once();
    mockSet('chartArea.height', '89%').$once();
    mockSet('hAxis.ticks',
        [{v: 40, f: '00:00:42'}, {v: 41, f: '00:00:43'}, {v: 42, f: '00:00:44'},
         {v: 43, f: '00:00:45'}, {v: 44, f: '00:00:46'}, {v: 45, f: '00:00:47'},
         {v: 46, f: '00:00:48'}, {v: 47, f: '00:00:49'}, {v: 48, f: '00:00:50'},
         {v: 49, f: '00:00:51'}]).$once();
    mockSet('vAxis.ticks',
        [{v: 42, f: 'series 1'}, {v: 0, f: 'series 2'}]).$once();

    mockControl.$replayAll();

    graph.updateChartOptions(storeData);

    mockControl.$verifyAll();
  },

  testModifyChannelSensitivity() {
    const mockChart = mockControl.createStrictMock(LineChart);
    mockChart.getSelection().$returns([{row: 42, column: 66}]);
    const mockGetChart = mockControl.createMethodMock(graph, 'getChart');
    mockGetChart().$returns(mockChart);

    const mockDataTable = mockControl.createStrictMock(DataTable);
    mockDataTable.getColumnId(66).$returns('test col');
    const mockGetDataTable = mockControl.createMethodMock(graph,
        'getDataTable');
    mockGetDataTable().$returns(mockDataTable);

    const mockHandleDraw = mockControl.createMethodMock(graph,
        'handleChartData');
    mockHandleDraw(storeData).$once();

    mockControl.$replayAll();

    graph.channelTransformations.set('test col', 0);
    graph.modifyChannelSensitivity(storeData, 0.49);

    mockControl.$verifyAll();

    assert(graph.channelTransformations.has('test col'));
    assertEquals(0.49, graph.channelTransformations.get('test col'));
  },

  tearDown() {
    mockControl.$tearDown();
  },
});
