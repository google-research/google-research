# Pangea

Pangea is an web simulator for panoramic navigation.

## Example

An example for the Matterport3D dataset is provided in `index.html`.

Download the [panoramas](https://niessner.github.io/Matterport/) and
[graphs](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity)
and put them in a `directory` structured in the following way.

*   `{directory}/{scan}_connectivity.json` - where `scan` identifies a graph.
*   `{directory}/{pano}_skybox{i}_sami.` - where `pano` identifies a node and
    `i` is a skybox side.

Start a server by calling `python -m SimpleHTTPServer`.

## Documentation

Pangea's core functionality is contained within `index.js`.

### `Environment`

An environment for panoramic navigation.

#### Constructor

##### `Environment(graph: Graph, loader: THREE.TextureLoader)`

*   `graph` - A graph where nodes are panoramas and edges indicate navigability.
*   `loader` - An panorama loader.

Note that the graph must contain the following node properties.

*   `matrix: THREE.Matrix4` - The camera pose matrix.
*   `inverse: THREE.Matrix4` - The inverse of the camera pose matrix.
*   `url: string` or `Array<string>` - The panorama url for the loader.
*   `height: number` - The height of the camera from ground level.

#### Properties

##### `.graph: Graph`

See constructor.

##### `.loader: THREE.TextureLoader`

See constructor.

##### `.camera: THREE.PerspectiveCamera`

The environment camera.

##### `.scene: THREE.Scene`

The environment scene.

##### `.node: string`

The node in the graph the environment is set to. Default is `null`.

##### `.cache: Object(string, THREE.Texture)`

A cache that maps nodes to panoramas.

#### Methods

##### `.set(node: string, objects: Array(THREE.Object3D)): null`

*   `node` - A node in the graph.
*   `objects` - The objects in the scene.

Transforms the pose the camera and given objects to be relative to the pose of
the given node, then sets the background of the scene to the panorama of the
given node.

##### `.setFromSnapshot(snapshot: Object, objects: Array(THREE.Object3D))`

*   `snapshot` - An output of `.snapshot` from an environment with the same
    graph.
*   `objects` - The objects in the scene.

Sets the environment from a snapshot.

##### `.snapshot(digits: number): Object`

*   `digit` - The number of digits to round a number to.

Returns a snapshot of the environment containing the following properties.

*   `node: string` - The node in the graph the environment is set to.
*   `aspect: number` - The aspect ratio of the camera.
*   `fov: number` - The horizontal field of view of the camera.
*   `direction: Array(number)` - The direction the camera is facing relative to
    the pose of the current node.

##### `.tween(node: string, objects: Array(THREE.Object3D), onStart: Function, onComplete: Function, duration: number): TWEEN.Tween`

*   `node` - A node in the graph.
*   `objects` - The objects in the scene.
*   `onStart` - A function which takes no arguments that is called upon starting
    the tween.
*   `onComplete` - A function which takes no arguments that is called upon
    completing the tween.
*   `duration` - The duration of the animation in milliseconds.

Similar to `.set`, but returns an animation tween that rotates and zooms the
camera towards the given node.

### `Graph`

A general purpose graph implementation.

#### Constructor

##### `Graph()`

Creates an empty graph.

#### Properties

##### `.edges: Object(string, Object(string, Object))`

A nested object mapping nodes to edge objects that map neighbors to data
objects.

##### `.nodes Object(string, Object)`

A nested object mapping nodes to data objects.

#### Methods

##### `.addEdge(source: string, target: string, data: Object): null`

*   `source` - A node to add an outgoing edge.
*   `target` - A node to add an incoming edge.
*   `data` - An object containing data about the edge.

Adds an edge to the graph.

##### `.addNode(node: string, data: Object): null`

*   `node` - A node to add to the graph.
*   `data` - An object containing data about the node.

Adds a node to the graph.

##### `.removeEdge(source: string, target: string): null`

*   `source` - The node to remove an outgoing edge.
*   `target` - The node to remove an incoming edge.

Removes an edge from the graph.

##### `.removeNode(node: string): null`

*   `node` - The node to remove from the graph.

Removes a node from the graph.

### Logger

A general purpose time series logger.

#### Constructor

##### `Logger(equals: Function, minTimeDelta: number)`

*   `equals` - A function that compares two entries. Returns true if two entries
    are equal and false otherwise. By default this is a function that always
    returns false.
*   `minTimeDelta` - The minimum time that must elapse between two consecutive
    entries.

#### Properties

##### `.equals: Function`

See constructor.

##### `.minTimeDelta: number`

See constructor.

##### `.entries: Array(Array)`

An array of timestamp-entry pairs.

#### Properties

##### `.log(entry: Object): boolean`

*   `entry` - An entry to be logged.

Logs an entry if the minimum time has elapsed and the entry is not redundant.
