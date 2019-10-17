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
 * @fileoverview Navigation graph environment using three.js.
 */

{
  // Constants used for three.js rendering.
  const EPS = 0.000001;
  const FAR = 1000000;

  /**
   * Navigation graph environment using three.js.
   */
  var Environment = class {
    /**
     * Constructs a navigation environment.
     * @param {!Graph} graph A navigation graph.
     */
    constructor(graph) {
      console.assert(graph.isGraph);

      // General three.js scene rendering things.
      var camera = new THREE.PerspectiveCamera(75, 2, EPS, FAR);
      var scene = new THREE.Scene();
      var renderer = new THREE.WebGLRenderer();
      var canvas = renderer.domElement;
      var controls = new THREE.OrbitControls(camera, canvas);
      var raycaster = new THREE.Raycaster();
      var mouse = new THREE.Vector2();

      // graphObjects: Objects in graph coordinates.
      var graphObjects = new THREE.Group();
      scene.add(graphObjects);
      graph.localToWorld(graphObjects);

      // dynamicGraphObjects: Objects in graph coordinates whose
      // visibility are determined by the closest node in the graph.
      var dynamicGraphObjects = new THREE.Group();
      graphObjects.add(dynamicGraphObjects);

      // staticGraphObjects: Objects in graph coordinates whose visibility
      // are determined by the visible property.
      var staticGraphObjects = new THREE.Group();
      graphObjects.add(staticGraphObjects);

      // Preload environment texture mappings.
      var cameraObjects = new THREE.Group();
      graphObjects.add(cameraObjects);

      graph.getNodes().forEach(node => {
        var cameraObject = graph.getCameraObject(node);
        cameraObjects.add(cameraObject);

        // Only one cameraObject is visible at any given time.
        cameraObject.visible = false;
        cameraObject.name = node;
      });

      // The point object tracks the state of the environment. It
      // indicates the current node of the graph and the neighbor
      // closest to the cursor.
      var geometry = new THREE.Geometry();
      geometry.vertices.push(new THREE.Vector3());
      var material = new THREE.PointsMaterial({

        color: 0xffffff,
        size: 10,
        sizeAttenuation: false,

      });

      // The point is in three.js coordinates; NOT graph coordinates.
      var point = new THREE.Points(geometry, material);
      scene.add(point);

      point.node = null;
      point.neighbor = null;

      // Adding event listeners.
      var scope = this;

      // Update the mouse position and raycaster for navigation.
      canvas.addEventListener('mousemove', e => {
        e.preventDefault();
        var r = canvas.getBoundingClientRect();
        mouse.x = ((e.clientX - r.left) / canvas.width) * 2 - 1;
        mouse.y = -((e.clientY - r.top) / canvas.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
      });

      // Right-click to move between nodes. The destination is determined
      // by the closest node to the mouse.
      canvas.addEventListener('contextmenu', e => {
        e.preventDefault();
        if (point.neighbor && controls.enabled) {
          scope.transition(point.neighbor);
        }
      });

      // Public attributes.
      this.camera = camera;
      this.canvas = canvas;
      this.controls = controls;
      this.raycaster = raycaster;
      this.renderer = renderer;
      this.scene = scene;

      // Private attributes.
      this._cameraObjects = cameraObjects;
      this._dynamicGraphObjects = dynamicGraphObjects;
      this._graph = graph;
      this._graphObjects = graphObjects;
      this._point = point;
      this._staticGraphObjects = staticGraphObjects;

      // Initialize to the first node in the graph.
      this.initialize(graph.getNodes()[0]);
    }

    // Private methods.

    /**
     * Positions the orbit control target to be a small distance in front
     * of the camera. This simulates a first-person control scheme.
     */
    _updateControls() {
      var direction = new THREE.Vector3();
      this.camera.getWorldDirection(direction);
      var offset = direction.multiplyScalar(EPS);
      this.controls.target.copy(this.camera.position).add(offset);
    }

    // Graph navigation methods.

    /**
     * Initializes the scene associated with a node in the graph. This
     * method teleports the user to a node in the graph, but does not
     * modify the direction the camera is facing. This method is called on
     * two major occasions: (1) initializing the scene and (2) at the end of
     * a transition animation.
     * @param {string} node A node in the graph.
     */
    initialize(node) {
      var scope = this;

      // Set the scene visibility and camera position.
      this._cameraObjects.children.forEach(cameraObject => {
        cameraObject.visible = false;
        if (cameraObject.name == node) {
          cameraObject.visible = true;
          cameraObject.getWorldPosition(scope.camera.position);
        }
      });

      // Set the visibility of dynamic graph objects.
      this._dynamicGraphObjects.children.forEach(object => {
        var neighbors = scope._graph.getNeighbors(node);
        object.visible = (

            neighbors.includes(object.node) || (object.node == node)

        );
      });

      this._point.node = node;
      this._point.neighbor = null;

      // Propagate graph coordinate correction to children objects.
      this._graphObjects.updateMatrixWorld();
      this._updateControls();
    }

    /**
     * Locks the controls and kicks off a transition animation from the
     * current node to the one given.
     * @param {string} node A node in the graph. Does NOT need to be
     * adjacent to the current node, however it is strongly encouraged.
     * @param {number=} duration The animation duration in milliseconds.
     * @param {!Function=} ease The easing function.
     */
    transition(node, duration = 1000, ease = TWEEN.Easing.Cubic.InOut) {
      var scope = this;

      // Variables that are initialized onStart.
      var rotation0, rotation1, fov0, fov1;

      var cameraObject = this._cameraObjects.getObjectByName(node);
      var position1 = new THREE.Vector3();
      cameraObject.getWorldPosition(position1);

      var animation =
          new TWEEN.Tween({alpha: 0})
              .to({alpha: 1}, duration)
              .easing(ease)
              .onStart(() => {
                // Disable controls during animation.
                scope.controls.enabled = false;

                rotation0 = scope.camera.quaternion.clone();
                scope.camera.lookAt(position1);
                rotation1 = scope.camera.quaternion.clone();
                scope.camera.quaternion.copy(rotation0);

                // The field of view dimishes inversely with distance.
                var distance = scope.camera.position.distanceTo(position1);
                fov0 = scope.camera.fov;
                fov1 = fov0 * Math.min(distance, 1 / distance);
              })
              .onUpdate((tmp) => {
                var alpha = tmp.alpha;
                var beta = 1 - alpha;

                // https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
                THREE.Quaternion.slerp(

                    rotation0, rotation1, scope.camera.quaternion, alpha

                );

                scope.camera.fov = beta * fov0 + alpha * fov1;
                scope.camera.updateProjectionMatrix();
              })
              .onComplete(() => {
                // Re-enable controls.
                scope.controls.enabled = true;

                // Reset the field of view and initialize the next scene.
                scope.camera.fov = fov0;
                scope.camera.updateProjectionMatrix();
                scope.initialize(node);
              });

      animation.start();
    }

    /**
     * Update method to be called before the render pass.
     */
    update() {
      var scope = this;

      // Assign point.neighbor to the neighbor of the current node that is
      // closest to the cursor if rendered at floor-level.
      var minimumDistance = Infinity;
      this._graph.getNeighbors(this._point.node).forEach(neighbor => {
        var cameraObjects = scope._cameraObjects;
        var cameraObject = cameraObjects.getObjectByName(neighbor);
        var position = new THREE.Vector3();
        cameraObject.getWorldPosition(position);

        // Trace a ray to floor-level by subtracting the camera height.
        position.y -= scope._graph.getCameraHeight(neighbor);
        var distance = scope.raycaster.ray.distanceToPoint(position);

        if (distance < minimumDistance) {
          minimumDistance = distance;
          scope._point.neighbor = neighbor;
          scope._point.position.copy(position);
        }
      });

      this._updateControls();
      TWEEN.update();
    }

    // Methods for adding and removing scene objects.

    /**
     * Adds a dynamic graph object to the scene. The visibility of this
     * object is dynamically computed based on the current node and the
     * closest node to this object.
     * These are objects that can be occluded. For example a chair.
     * @param {!THREE.Object3D} object An object in graph coordinates.
     */
    addDynamicGraphObject(object) {
      // Bind the object to its closest node.

      // Add the object and apply the coordinate corrections.
      this._dynamicGraphObjects.add(object);
      this._graphObjects.updateMatrixWorld();
      var position0 = object.getWorldPosition(new THREE.Vector3());

      // Assign a node property to this object. This property indicates
      // the closest node in the graph.
      var minimumDistance = Infinity;
      this._cameraObjects.children.forEach((cameraObject) => {
        var position1 = new THREE.Vector3();
        cameraObject.getWorldPosition(position1);
        var distance = position0.distanceTo(position1);

        if (distance < minimumDistance) {
          minimumDistance = distance;
          object.node = cameraObject.name;
        }
      });

      // The object is visible if its node is visible from the current
      // node.
      var neighbors = this._graph.getNeighbors(this._point.node);
      object.visible = (

          neighbors.includes(object.node) || (object.node == this._point.node)

      );
    }

    /**
     * Adds a static graph object to the scene. The visibility of this
     * object is NOT determined by graph connectivity.
     * This is generally used for adding lighting and fog to the scene.
     * @param {!THREE.Object3D} object An object in graph coordinates.
     */
    addStaticGraphObject(object) {
      // Add the object and apply the coordinate corrections.
      this._staticGraphObjects.add(object);
      this._graphObjects.updateMatrixWorld();
    }

    /**
     * Removes a dynamic graph object.
     * @param {!THREE.Object3D} object An object to be removed.
     */
    removeDynamicGraphObject(object) {
      this._dynamicGraphObjects.remove(object);
    }

    /**
     * Removes a static graph object.
     * @param {!THREE.Object3D} object An object to be removed.
     */
    removeStaticGraphObject(object) {
      this._staticGraphObjects.remove(object);
    }

    /**
     * Sets the visibility property of the point. By default the point is
     * rendered as a white point at floor-level. This point indicates the
     * position one would move to upon right-clicking.
     * @param {boolean} bool Whether the point is visible.
     */
    setPointVisibility(bool) {
      this._point.visible = bool;
    }

    // Environment state methods.

    /**
     * Exports the environment state as a JSON object.
     * @return { !Object } The state of the environment.
     */
    getState() {
      return {

        node: this._point.node,
        aspect: this.camera.aspect,
        fov: this.camera.fov,
        matrix: this.camera.matrix.toArray(),

      };
    }

    /**
     * Checks if two states are the same.
     * @param {!Object} state0 An environment state.
     * @param {?Object} state1 An optional environment state. If this is
     * not provided, the current state with be used.
     * @param {number=} tolerance The numerical tolerance of the check.
     * @return {boolean} Whether the two states are the same.
     */
    isSameState(state0, state1, tolerance = EPS) {
      if (state1 == undefined) state1 = this.getState();
      if (state0.node != state1.node) return false;
      if (state0.aspect != state1.aspect) return false;
      if (state0.fov != state1.fov) return false;

      for (var i = 0; i < 16; i++) {
        var absdif = Math.abs(state0.matrix[i] - state1.matrix[i]);
        if (absdif > tolerance) return false;
      }

      return true;
    }

    /**
     * Sets the environment state.
     * @param {!Object} state An environment state.
     */
    setState(state) {
      this.initialize(state.node);

      this.camera.aspect = state.aspect;
      this.camera.fov = state.fov;
      this.camera.updateProjectionMatrix();

      new THREE.Matrix4()
          .fromArray(state.matrix)
          .decompose(

              this.camera.position, this.camera.rotation, this.camera.scale

          );
    }

    // Rendering methods.

    /**
     * Renders the scene to the renderer canvas.
     */
    render() {
      this.renderer.render(this.scene, this.camera);
    }

    /**
     * Sets the size of the renderer canvas and adjusts the camera aspect
     * ratio.
     * @param {number} width The width of the canvas.
     * @param {number} height The height of the canvas.
     */
    setRendererSize(width, height) {
      this.renderer.setSize(width, height);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }

    /**
     * Updates and renders the scene.
     */
    updateAndRender() {
      this.update();
      this.render();
    }
  };
}

(global => {
  if (typeof define === 'function' && define.amd) {
    define([], () => Environment);

  } else if (typeof module !== 'undefined' && typeof exports === 'object') {
    module.exports = Environment;

  } else if (global !== undefined) {
    global.Environment = Environment;
  }
})(this);
