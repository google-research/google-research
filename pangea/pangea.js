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
 * @fileoverview Pangea codebase.
 */


class Base {

  /**
   * Constructor for the Base class.
   * @param {!Object} graph - The navigation graph.
   * @param {!Function} texturizer - The node texturizer.
   */
  constructor(graph, texturizer) {

    var scope = this;

    // Dependencies.

    if (typeof THREE == 'undefined') {
      console.error('ERROR: pangea requires THREE');
    }

    if (typeof TWEEN == 'undefined') {
      console.error('ERROR: pangea requires TWEEN');
    }

    if (THREE.OrbitControls == undefined) {
      console.error('ERROR: pangea requires THREE.OrbitControls');
    }

    scope.graph = graph;
    scope.texturizer = texturizer;
    scope.node = null;

    scope.renderer = new THREE.WebGLRenderer();
    scope.canvas = scope.renderer.domElement;

    scope.scene = new THREE.Scene();
    scope.up = new THREE.Vector3(0, 1, 0);

    var fov = 75;
    var aspect = scope.canvas.width / scope.canvas.height;
    scope.camera = new THREE.PerspectiveCamera(fov, aspect);

    scope.controls = new THREE.OrbitControls(scope.camera, scope.canvas);
    scope.controls.enableZoom = false;
    scope.controls.enabled = true;

    scope.objects = new THREE.Group();
    scope.scene.add(scope.objects);

  }


  /**
   * Initializes the scene at the given node.
   * @param {string} node - The node to initialize.
   */
  initialize(node) {

    var scope = this;

    scope.node = node;
    scope.scene.background = scope.texturizer(scope.node);

  }

  /**
   * Adds an object to scene with local visibility.
   * @param {!THREE.Object3D} object - The object to be added.
   */
  add(object) {

    var scope = this;

    if (!scope.node) {
      console.error('ERROR: the initialize method has not been called');
      return;
    }

    // Apply the pose of the current node before adding it to the scene.

    object.applyMatrix(scope.graph[scope.node].inverse);
    scope.objects.add(object);

  }

  /**
   * Sets the scene to the given node.
   * @param {string} node - The node to set.
   * @param {?THREE.Texture} texture - The texture to set.
   */
  set(node, texture) {

    var scope = this;

    // revert relative pose of the current node
    // apply relative pose of the node being set

    scope.objects.children.forEach(function(object) {

      object.applyMatrix(scope.graph[scope.node].matrix);
      object.applyMatrix(scope.graph[node].inverse);

    });

    scope.node = node;
    scope.scene.background = texture || scope.texturizer(scope.node);

  }

  /**
   * Updates the camera and controls.
   */
  update() {

    var scope = this;

    if (!scope.node) {
      console.error('ERROR: the initialize method has not been called');
      return;
    }

    // The visibility of objects added with the add method is determined
    // by the visibility of the closest node in the graph relative to the
    // current node.

    scope.objects.children.forEach(function(object) {

      var closest = null;
      var minimum = Infinity;

      for (var node in scope.graph) {

        if (scope.graph[node].included) {

          var point = new THREE.Vector3();
          point.setFromMatrixPosition(scope.graph[node].matrix);
          point.applyMatrix4(scope.graph[scope.node].inverse);

          var distance = point.distanceTo(object.position);

          if (distance < minimum) {

            minimum = distance;
            closest = node;

          }

        }

      }

      object.visible = (
          scope.graph[scope.node].visibility.includes(closest)
          || closest == scope.node
      );

    });

    // Simulate first person perspective by positioning the target attribute
    // of THREE.OrbitControls to be an epsilon distance in front of the camera.

    var epsilon = 1e-5;
    var look = new THREE.Vector3();
    scope.camera.getWorldDirection(look);
    look.multiplyScalar(epsilon);

    scope.controls.target.copy(scope.camera.position);
    scope.controls.target.add(look);
    scope.controls.update();

    scope.camera.updateProjectionMatrix();

  }

  /**
   * Renders the scene.
   */
  render() {

    var scope = this;

    scope.update();

    scope.renderer.render(scope.scene, scope.camera);

  }

}


class Environment extends Base {

  /**
   * Constructor for the Environment class.
   * @param {!THREE.Object3D} object - The navigation cursor.
   * @param {!Object} graph - The navigation graph.
   * @param {!Function} texturizer - The node texturizer.
   */
  constructor(object, graph, texturizer) {

    super(graph, texturizer);

    var scope = this;

    scope.controls.enabled = true;

    // Enable logging to use with the Playback class.

    scope.logging = true;
    scope.events = [];

    // The navigation cursor object renders above the node closest to the mouse.
    // It serves as the primary navigation interface for the user.

    scope.object = object;
    scope.scene.add(scope.object);

    scope.duration = 1500;
    scope.ease = TWEEN.Easing.Cubic.InOut;

    var log = scope.log.bind(scope);

    scope.raycaster = new THREE.Raycaster();
    scope.mouse = new THREE.Vector2();

    scope.canvas.addEventListener('mousemove', function(e) {

      e.preventDefault();

      var rect = scope.canvas.getBoundingClientRect();

      scope.mouse.x = ((e.clientX - rect.left) / scope.canvas.width) * 2 - 1;
      scope.mouse.y = -((e.clientY - rect.top) / scope.canvas.height) * 2 + 1;

      scope.raycaster.setFromCamera(scope.mouse, scope.camera);

      log();

    });

    scope.canvas.addEventListener('contextmenu', function() {

      if (!scope.node) {
        console.error('ERROR: the initialize method has not been called');
        return;
      }

      if (scope.raycaster.intersectObject(scope.object).length == 0) {
        return;
      }

      var node = scope.object.node;
      var texture = scope.texturizer(node);

      // Let s be the position of the source node (global coordinates).
      // Let t be the position of the target node (global coordinates).

      var s = new THREE.Vector3();
      s.setFromMatrixPosition(scope.graph[scope.node].matrix);
      var t = new THREE.Vector3();
      t.setFromMatrixPosition(scope.graph[node].matrix);
      var st = t.clone().sub(s);
      var distance = st.length();

      // Let u be the source camera rotation (source coordinates).
      // Let v be the target camera rotation (source coordinates).

      var u = scope.camera.quaternion.clone();
      var look = s.clone().add(st);
      look.applyMatrix4(scope.graph[scope.node].inverse);
      scope.camera.lookAt(look);
      var v = scope.camera.quaternion.clone();
      scope.camera.quaternion.copy(u);

      // Rotate the camera from u to v (source coordinates).

      var first = new TWEEN.Tween({t: 0})
      .to({t: 1}, scope.duration)
      .easing(scope.ease)
      .onUpdate(function(obj) {

        THREE.Quaternion.slerp(u, v, scope.camera.quaternion, obj.t);

      });

      // Zoom the camera in the direction of v (source coordinates).

      var zoom = Math.min(1 / distance, distance);

      var second = new TWEEN.Tween(scope.camera)
        .to({fov: scope.camera.fov * zoom}, scope.duration)
        .easing(scope.ease)
        .onUpdate(log)
        .onComplete(function() {

          // Rotate camera in the direction of v (target coordinates).

          var look = t.clone().add(st);
          look.applyMatrix4(scope.graph[node].inverse);
          scope.camera.lookAt(look);
          scope.camera.fov /= zoom;

          // Move from the source to the target node in the graph.

          scope.set(node, texture);
          scope.controls.enabled = true;

        });

      scope.controls.enabled = false;
      first.chain(second).start();

    });

  }

  /**
   * Logs the state of the environment.
   */
  log() {

    var scope = this;

    if (!scope.node) {
      console.error('ERROR: the initialize method has not been called');
      return;
    }

    if (!scope.logging) {
      return;
    }

    scope.events.push({

      camera: scope.camera.toJSON(),
      controlsEnabled: scope.controls.enabled,
      mouse: {

        vector: scope.mouse.toArray(),
        ray: {

          origin: scope.raycaster.ray.origin.toArray(),
          direction: scope.raycaster.ray.direction.toArray(),

        },

      },
      node: scope.node,
      time: Date.now(),

    });

  }

  /**
   * Updates the camera and controls.
   */
  update() {

    var scope = this;

    if (!scope.node) {
      console.error('ERROR: the initialize method has not been called');
      return;
    }

    // Find the node closest to the ray from the camera to the mouse.

    var closest = null;
    var minimum = Infinity;

    for (var node in scope.graph) {

      if (scope.graph[scope.node].visibility.includes(node)
          && scope.graph[node].included) {

        var offset = scope.up.clone();
        offset.multiplyScalar(scope.graph[node].height);

        var point = new THREE.Vector3();
        point.setFromMatrixPosition(scope.graph[node].matrix);
        point.sub(offset);
        point.applyMatrix4(scope.graph[scope.node].inverse);

        var distance = scope.raycaster.ray.distanceToPoint(point);

        if (distance < minimum) {
          minimum = distance;
          closest = node;
        }

      }

    }

    // Move the navigation cursor object to the node closest to the mouse.

    scope.object.position.set(0, 0, 0);
    scope.object.rotation.set(0, 0, 0);
    scope.object.scale.set(1, 1, 1);

    var offset = scope.up.clone();
    offset.multiplyScalar(scope.graph[closest].height);

    // Apply the pose of the closest node and drop the object to ground level
    // (global coordinates). The pose is always positioned at eye level.

    scope.object.applyMatrix(scope.graph[closest].matrix);
    scope.object.position.sub(offset);

    // Convert to the coordinate system of the current node.

    scope.object.applyMatrix(scope.graph[scope.node].inverse);

    scope.object.visible = true;
    scope.object.node = closest;

    TWEEN.update();
    super.update();

  }

}


class Playback extends Base {

  /**
   * Constructor for the Playback class.
   * @param {!Object} graph - The navigation graph.
   * @param {!Function} texturizer - The node texturizer.
   */
  constructor(graph, texturizer) {

    super(graph, texturizer);

    var scope = this;

    scope.controls.enabled = false;

    scope.events  = null;
    scope.textures = {};

    scope.time = null;
    scope.alive = null;
    scope.i = null;

    // Let speed be the playback time multiplier.

    scope.speed = 1;

  }

  /**
   * Initializes the Playback.
   * @param {!Array} events - The events from an Environment to replay.
   */
  initialize(events) {

    var scope = this;

    if (events.length < 1) {
      console.error('ERROR: playback requires > 1 events');
      return;
    }

    scope.events = events;

    // Buffer every playback referenced in the events.

    scope.events.forEach(function(event) {

      if (!scope.textures[event.node]) {
        scope.textures[event.node] = scope.texturizer(event.node);
      }

    });

    // Let time be the timestamp of the first call to the render method.
    // Let i be the index of the current event frame.

    scope.time = null;
    scope.alive = true;
    scope.i = 0;

    scope.node = scope.events[0].node;
    scope.scene.background = scope.textures[scope.node];

  }

  /**
   * Renders the scene.
   * @param {number} time - The timestamp of the animation frame.
   */
  render(time) {

    var scope = this;

    time *= scope.speed;
    scope.time = scope.time || time;

    // Increment the index of the current event frame until the event time is
    // greater than the timestamp of the animation frame.

    while ((scope.i < scope.events.length)
           && ((time - scope.time)
               >= (scope.events[scope.i].time - scope.events[0].time))) {
      scope.i++;
    }

    if (scope.i < scope.events.length) {

      // The animation is made smooth by interpolating between camera rotations
      // and field-of-view.

      var start = scope.events[scope.i - 1];
      var end = scope.events[scope.i];

      console.log(start);

      // Let p be timestamp of previous event frame.
      // Let q be timestamp of current event frame.
      // Let r be timestamp of the animation frame.
      // Let t be progression of r between p and q (between 0 and 1).

      var p = start.time - scope.events[0].time;
      var q = end.time - scope.events[0].time;
      var r = time - scope.time;
      var t = (r - p) / (q - p);

      // Let u be camera rotation of previous event frame.
      // Let v be camera rotation of current event frame.
      // Let w be the slerp between u and v.

      var u = new THREE.Matrix4().fromArray(start.camera.object.matrix);
      u = new THREE.Quaternion().setFromRotationMatrix(u);

      var v = new THREE.Matrix4().fromArray(end.camera.object.matrix);
      v = new THREE.Quaternion().setFromRotationMatrix(v);

      var w = new THREE.Quaternion();
      THREE.Quaternion.slerp(u, v, w, t);

      scope.camera.rotation.setFromQuaternion(w);
      scope.camera.fov = (
        t * start.camera.object.fov) + ((1 - t) * end.camera.object.fov);

      if (end.node != scope.node) {
        scope.set(end.node, scope.textures[end.node]);
      }

      super.render();

    } else {

      scope.alive = false;

    }

  }

}


var PANGEA = {

  Environment,
  Playback,

};


(function(root) {

  if (typeof define === 'function' && define.amd) {

    define([], function() {

      return PANGEA;

    });

  } else if (typeof module !== 'undefined'
             && typeof exports === 'object') {

    module.exports = PANGEA;

  } else if (root !== undefined) {

    root.PANGEA = PANGEA;

  }

})(this);
