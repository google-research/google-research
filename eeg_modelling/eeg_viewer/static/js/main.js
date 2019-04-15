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

goog.module('eeg_modelling.eeg_viewer.views');

const Annotations = goog.require('eeg_modelling.eeg_viewer.Annotations');
const ChartController = goog.require('eeg_modelling.eeg_viewer.ChartController');
const Console = goog.require('goog.debug.Console');
const Error = goog.require('eeg_modelling.eeg_viewer.Error');
const Graph = goog.require('eeg_modelling.eeg_viewer.Graph');
const Menus = goog.require('eeg_modelling.eeg_viewer.Menus');
const NavChart = goog.require('eeg_modelling.eeg_viewer.NavChart');
const Predictions = goog.require('eeg_modelling.eeg_viewer.Predictions');
const Requests = goog.require('eeg_modelling.eeg_viewer.Requests');
const ToolBar = goog.require('eeg_modelling.eeg_viewer.ToolBar');
const WaveEvents = goog.require('eeg_modelling.eeg_viewer.WaveEvents');
const WindowLocation = goog.require('eeg_modelling.eeg_viewer.WindowLocation');

/** @const {!Console} */
const console = new Console();
console.setCapturing(true);

Annotations.getInstance();
ChartController.getInstance();
Error.getInstance();

/** @const {!Graph} */
const graph = Graph.getInstance();
goog.exportSymbol(
    'graph.closeSensitivityMenu',
    goog.bind(graph.closeSensitivityMenu, graph));
goog.exportSymbol(
    'graph.increaseSensitivity',
    goog.bind(graph.increaseSensitivity, graph));
goog.exportSymbol(
    'graph.decreaseSensitivity',
    goog.bind(graph.decreaseSensitivity, graph));
goog.exportSymbol(
    'graph.selectWaveEventType',
    goog.bind(graph.selectWaveEventType, graph));
goog.exportSymbol(
    'graph.closeWaveEventForm',
    goog.bind(graph.closeWaveEventForm, graph));
goog.exportSymbol(
    'graph.saveWaveEvent',
    goog.bind(graph.saveWaveEvent, graph));

/** @const {!Menus} */
const menus = Menus.getInstance();
goog.exportSymbol('menus.loadFile', goog.bind(menus.loadFile, menus));
goog.exportSymbol(
    'menus.toggleFileInfo', goog.bind(menus.toggleFileInfo, menus));
goog.exportSymbol('menus.toggleMenu', goog.bind(menus.handleMenuToggle, menus));
goog.exportSymbol(
    'menus.selectFileType', goog.bind(menus.selectFileType, menus));

NavChart.getInstance();

/** @const {!Predictions} */
const predictions = Predictions.getInstance();
goog.exportSymbol(
    'predictions.selectPredictionMode',
    goog.bind(predictions.handlePredictionModeSelection, predictions));
goog.exportSymbol(
    'predictions.toggleFilter',
    goog.bind(predictions.toggleFilter, predictions));
goog.exportSymbol(
    'predictions.selectPredictionLabel',
    goog.bind(predictions.handlePredictionLabelSelection, predictions));

Requests.getInstance();

/** @const {!ToolBar} */
const toolBar = ToolBar.getInstance();
goog.exportSymbol(
    'toolBar.toggleMenu', goog.bind(toolBar.toggleMenu, toolBar));
goog.exportSymbol(
    'toolBar.selectDropdown', goog.bind(toolBar.selectDropdown, toolBar));
goog.exportSymbol('toolBar.nextChunk', goog.bind(toolBar.nextChunk, toolBar));
goog.exportSymbol('toolBar.nextSec', goog.bind(toolBar.nextSec, toolBar));
goog.exportSymbol('toolBar.prevChunk', goog.bind(toolBar.prevChunk, toolBar));
goog.exportSymbol('toolBar.prevSec', goog.bind(toolBar.prevSec, toolBar));

/** @const {!WindowLocation} */
const windowLocation = WindowLocation.getInstance();

/** @const {!WaveEvents} */
const waveEvents = WaveEvents.getInstance();
goog.exportSymbol(
    'waveEvents.closeWaveEventMenu',
    goog.bind(waveEvents.closeWaveEventMenu, waveEvents));
goog.exportSymbol(
    'waveEvents.deleteWaveEvent',
    goog.bind(waveEvents.deleteWaveEvent, waveEvents));
goog.exportSymbol(
    'waveEvents.navigateToWaveEvent',
    goog.bind(waveEvents.navigateToWaveEvent, waveEvents));


/**
 * Make a request to the server when the URL changes.
 */
window.onhashchange = () => windowLocation.makeDataRequest();

windowLocation.makeDataRequest();
