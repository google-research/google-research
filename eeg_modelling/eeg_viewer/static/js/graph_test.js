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

goog.module('eeg_modelling.eeg_viewer.Graph.tests');
goog.setTestOnly();

const DataTable = goog.require('google.visualization.DataTable');
const Graph = goog.require('eeg_modelling.eeg_viewer.Graph');
const MockControl = goog.require('goog.testing.MockControl');
const gvizEvents = goog.require('google.visualization.events');
const singleton = goog.require('goog.testing.singleton');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;

let graph;
let storeData;
let annotatedData;


testSuite({

  setUp() {
    mockControl = new MockControl();

    singleton.resetAll();
    graph = Graph.getInstance();

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
      channelIds: [],
      chunkStart: 40,
      chunkDuration: 10,
      fileType: 'EEG',
      predictionMode: 'None',
      samplingFreq: 1,
      seriesHeight: 42,
      sensitivity: 1,
      timeScale: 1,
      waveEvents: [],
    };

    annotatedData = {
      cols: [
        {id: 'axis', label: 'axis', type: 'number'},
        {id: '', label: '', pattern: '', type: 'string',
         p: {role: 'annotation'}},
        {id: '', label: '', pattern: '', type: 'string',
         p: {role: 'annotationText', html: true}},
        {id: 'series 1', label: 'series 1', type: 'number'},
        {id: '', label: '', type: 'string',
         p: {role: 'tooltip', html: true}},
        {id: 'series 2', label: 'series 2', type: 'number'},
        {id: '', label: '', type: 'string',
         p: {role: 'tooltip', html: true}},
      ],
      rows: [
        {c: [{v: 40}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'},]},
        {c: [{v: 41}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
        {
          c: [
            {v: 42}, {v: 'label'}, {v: 'do not trust the mice'},
            {v: 42, f: 0}, {v: 'html tooltip'},
            {v: 7, f: 1}, {v: 'html tooltip'},
          ]
        },
        {c: [{v: 43}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
        {c: [{v: 44}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
        {c: [{v: 45}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
        {c: [{v: 46}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
        {c: [{v: 47}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
        {c: [{v: 48}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
        {c: [{v: 49}, {v: null}, {v: null}, {v: 42, f: 0}, {v: 'html tooltip'},
             {v: 7, f: 1}, {v: 'html tooltip'}]},
      ],
    };

    graph.parentId = 'parent-chart-container';
    graph.containerId = 'line-chart-container';

    const parentContainer = document.createElement('div');
    parentContainer.id = graph.parentId;

    const overlay = document.createElement('canvas');
    overlay.id = graph.overlayId;
    parentContainer.appendChild(overlay);

    const chunk = document.createElement('div');
    chunk.id = graph.containerId;
    parentContainer.appendChild(chunk);

    document.body.appendChild(parentContainer);

    const channelActionsContainer = document.createElement('div');
    channelActionsContainer.id = 'channel-actions-container';

    const channelActionsTitle = document.createElement('div');
    channelActionsTitle.id = 'channel-actions-title';
    channelActionsContainer.appendChild(channelActionsTitle);

    document.body.appendChild(channelActionsContainer);
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

  testAddAnnotations() {
    const dataTable = new DataTable(storeData.chunkGraphData);
    graph.addAnnotations(storeData, dataTable);
    assertContains('do not trust the mice', dataTable.getValue(2, 2));
  },

  testCreateDataTable() {
    const dataTable = graph.createDataTable(storeData);

    const expectedDataTable = new DataTable(annotatedData);
    assertEquals(
        expectedDataTable.getNumberOfRows(), dataTable.getNumberOfRows());
    assertEquals(
        expectedDataTable.getNumberOfColumns(), dataTable.getNumberOfColumns());
  },

  testUpdateChartOptions() {
    const mockParent = mockControl.createMethodMock(graph, 'getParent');
    mockParent().$returns({clientHeight: 500});

    const mockSet = mockControl.createMethodMock(graph, 'setOption');
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
    const selectedSeries = 'series 2';
    const seriesReversedIndex = 0;

    graph.channelTransformations.set(selectedSeries, 2);
    graph.handleChartData(storeData, ['chunkGraphData']);

    gvizEvents.trigger(graph.getChart(), 'click', {
      targetID: `vAxis#0#label#${seriesReversedIndex}`,
      x: 6.4,
      y: 0,
    });
    graph.increaseSensitivity();

    assert(graph.channelTransformations.has(selectedSeries));
    assertEquals(4, graph.channelTransformations.get(selectedSeries));
  },

  tearDown() {
    mockControl.$tearDown();
  },
});
