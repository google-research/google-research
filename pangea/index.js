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

/**
 * @fileoverview Description of this file.
 */

class Graph {

  constructor() {
    this.nodes = {};
    this.edges = {};
  }

  /**
   * @param {string} node
   * @param {!Object=} data
   */
  addNode(node, data = {}) {
    this.nodes[node] = data;
    this.edges[node] = {};
  }

  /**
   * @param {string} node
   */
  removeNode(node) {
    delete this.nodes[node];
    delete this.edges[node];
    Object.values(this.edges).map(function(neighbors) {
      delete neighbors[node];
    });
  }

  /**
   * @param {string} source
   * @param {string} target
   * @param {!Object=} data
   */
  addEdge(source, target, data = {}) {
    this.edges[source][target] = data;
  }

  /**
   * @param {string} source
   * @param {string} target
   */
  removeEdge(source, target) {
    delete this.edges[source][target];
  }

  /**
   * @param {string} node
   * @return {!Array<string>}
   */
  neighbors(node) {
    return Object.keys(this.edges[node]);
  }

}

class Environment {

  /**
   * @param {!Graph} graph
   * @param {!THREE.TextureLoader} loader
   */
  constructor(graph, loader) {
    this.graph = graph;
    this.loader = loader;
    this.camera = new THREE.PerspectiveCamera();
    this.scene = new THREE.Scene();
    this.node = null;
    this.cache = {};
  }

  /**
   * @param {string} node
   * @param {!Array<!THREE.Object3D>=} objects
   */
  set(node, objects = []) {
    var matrix = (this.node ?
      this.graph.nodes[this.node].matrix : new THREE.Matrix4());
    var inverse = this.graph.nodes[node].inverse;

    var direction = new THREE.Vector3();
    this.camera.getWorldDirection(direction);
    direction.applyQuaternion(
        new THREE.Quaternion().setFromRotationMatrix(matrix));
    direction.applyQuaternion(
      new THREE.Quaternion().setFromRotationMatrix(inverse));
    this.camera.lookAt(direction);

    objects.map(function(object) {
      object.applyMatrix(matrix);
      object.applyMatrix(inverse);
    });

    var scope = this;

    this.graph.neighbors(node).concat([node]).map(function(node) {
      if (!Object.keys(scope.cache).includes(node)) {
        var url = scope.graph.nodes[node].url;
        scope.cache[node] = scope.loader.load(url);
      }
    });

    this.scene.background = this.cache[node];
    this.node = node;
  }

  /**
   * @param {string} node
   * @param {!Array<!THREE.Object3D>=} objects
   * @param {?Function=} onStart
   * @param {?Function=} onComplete
   * @param {number=} duration
   * @return {!TWEEN.Tween}
   */
  tween(node, objects = [], onStart = null, onComplete = null, duration = 1e3) {
    var pos1 = new THREE.Vector3();
    var rot0, rot1, fov0, fov1;

    var scope = this;

    var coords = {alpha: 0};
    return new TWEEN.Tween(coords).to({alpha: 1}, duration)
      .easing(TWEEN.Easing.Cubic.InOut)
      .onStart(function() {
        pos1.setFromMatrixPosition(scope.graph.nodes[node].matrix);
        pos1.applyMatrix4(scope.graph.nodes[scope.node].inverse);

        rot0 = scope.camera.quaternion.clone();
        scope.camera.lookAt(pos1);
        rot1 = scope.camera.quaternion.clone();
        scope.camera.quaternion.copy(rot0);

        fov0 = scope.camera.fov;
        var distance = pos1.length();
        fov1 = fov0 * Math.min(distance, 1 / distance);

        if (onStart) onStart();
      })
      .onUpdate(function() {
        var alpha = this.alpha || coords.alpha;

        var rot = scope.camera.quaternion;
        THREE.Quaternion.slerp(rot0, rot1, rot, alpha);

        scope.camera.fov = (1 - alpha) * fov0 + alpha * fov1;
        scope.camera.updateProjectionMatrix();
      })
      .onComplete(function() {
        scope.camera.fov = fov0;
        scope.camera.updateProjectionMatrix();
        scope.set(node, objects);

        if (onComplete) onComplete();
      });
  }

  /**
   * @param {number=} digits
   * @return {!Object}
   */
  snapshot(digits = 2) {
    var direction = new THREE.Vector3();
    this.camera.getWorldDirection(direction);

    function round(number) {
      return Number(number.toFixed(digits));
    }

    return {
      node: this.node,
      aspect: round(this.camera.aspect),
      fov: round(this.camera.fov),
      direction: direction.toArray().map(round),
    };
  }

  /**
   * @param {!Object} snapshot
   * @param {!Array<!THREE.Object3D>=} objects
   */
  setFromSnapshot(snapshot, objects = []) {
    this.set(snapshot.node, objects);

    var direction = new THREE.Vector3();
    direction.fromArray(snapshot.direction);
    this.camera.lookAt(direction);

    this.camera.aspect = snapshot.aspect;
    this.camera.fov = snapshot.fov;
    this.camera.updateProjectionMatrix();
  }

}

class Logger {

  /**
   * @param {!Function} equals
   * @param {number=} minTimeDelta
   */
  constructor(equals, minTimeDelta = 0) {
    this.equals = equals || function() {
      return false;
    };
    this.minTimeDelta = minTimeDelta;
    this.entries = [];
  }

  /**
   * @param {!Object} entry
   * @return {boolean}
   */
  log(entry) {
    var previous = this.entries[this.entries.length - 1];
    var time = new Date().getTime();

    var timeDelta = previous ? time - previous[0] : Infinity;
    var redundant = previous ? this.equals(entry, previous[1]) : false;

    if (timeDelta > this.minTimeDelta && !redundant) {
      this.entries.push([time, entry]);
      return true;
    }

    return false;
  }

}

/**
 * @param {!Object} global
 * @param {!Function} factory
 */
function universalModuleDefinition(global, factory) {
  if (typeof exports == 'object' && typeof module != 'undefined') {
    factory(exports);
  } else if (typeof define == 'function' && define.amd) {
    define(['exports'], factory);
  } else {
    factory(global || self);
  }
}

universalModuleDefinition(this, function(exports) {
  exports.Graph = Graph;
  exports.Environment = Environment;
  exports.Logger = Logger;
});
