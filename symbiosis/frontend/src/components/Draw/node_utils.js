// Copyright 2025 The Google Research Authors.
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

// node_utils.js
import { Node, NodeOrigin, Rect, Box } from '@xyflow/react';

import {
  boxToRect,
  getNodePositionWithOrigin,
  rectToBox,
} from '@xyflow/system';

/**
 * Compares two nodes based on their type for sorting.
 * Group nodes are placed before other node types.
 *
 * @param {Node} a - The first node.
 * @param {Node} b - The second node.
 * @returns {number} -1 if a should come before b, 1 if a should come after b, 0 if they have the same order.
 */
export const sortNodes = (a, b) => {
  if (a.type === b.type) {
    return 0; // No order change if the types are the same
  }
  return a.type === 'group' && b.type !== 'group' ? -1 : 1; // Group nodes come before other types
};

/**
 * Generates a unique ID with an optional prefix.
 *
 * @param {string} [prefix='variablenode'] - The prefix for the ID.
 * @returns {string} The generated unique ID.
 */
export const getId = (prefix = 'variablenode') => `${prefix}_${Math.random() * 10000}`;

/**
 * Calculates the position of a node inside its parent group node.
 *
 * @param {Node} node - The child node.
 * @param {Node} groupNode - The parent group node.
 * @returns {{ x: number, y: number }} The position of the child node relative to the parent.
 */
export const getNodePositionInsideParent = (
    node,
    groupNode
) => {
  const position = node.position ?? { x: 0, y: 0 }; // Use the node's position or default to { x: 0, y: 0 }
  const nodeWidth = node.measured?.width ?? 0; // Get the node's width or default to 0
  const nodeHeight = node.measured?.height ?? 0; // Get the node's height or default to 0
  const groupWidth = groupNode.measured?.width ?? 0; // Get the group node's width or default to 0
  const groupHeight = groupNode.measured?.height ?? 0; // Get the group node's height or default to 0

  // Adjust the x position to keep the node within the parent's bounds
  if (position.x < groupNode.position.x) {
    position.x = 0;
  } else if (position.x + nodeWidth > groupNode.position.x + groupWidth) {
    position.x = groupWidth - nodeWidth;
  } else {
    position.x = position.x - groupNode.position.x;
  }

  // Adjust the y position to keep the node within the parent's bounds
  if (position.y < groupNode.position.y) {
    position.y = 0;
  } else if (position.y + nodeHeight > groupNode.position.y + groupHeight) {
    position.y = groupHeight - nodeHeight;
  } else {
    position.y = position.y - groupNode.position.y;
  }

  return position; // Return the calculated position
};

/**
 * Calculates the combined bounds of two boxes.
 *
 * @param {Box} box1 - The first box.
 * @param {Box} box2 - The second box.
 * @returns {Box} The combined bounds of the two boxes.
 */
export const getBoundsOfBoxes = (box1, box2) => ({
  x: Math.min(box1.x, box2.x), // Minimum x coordinate
  y: Math.min(box1.y, box2.y), // Minimum y coordinate
  x2: Math.max(box1.x2, box2.x2), // Maximum x2 coordinate
  y2: Math.max(box1.y2, box2.y2), // Maximum y2 coordinate
});

/**
 * Calculates the bounding rectangle of a set of nodes.
 *
 * @param {Node[]} nodes - The array of nodes.
 * @param {NodeOrigin} [nodeOrigin=[0, 0]] - The origin of the nodes.
 * @returns {Rect} The bounding rectangle of the nodes.
 */
export const getRelativeNodesBounds = (
    nodes,
    nodeOrigin = [0, 0] // Default node origin
) => {
  if (nodes.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 }; // Return an empty rectangle if there are no nodes
  }

  // Reduce the nodes array to a single box that encompasses all nodes
  const box = nodes.reduce(
      (currBox, node) => {
        const { x, y } = getNodePositionWithOrigin(node, nodeOrigin); // Get the node's position with the specified origin
        return getBoundsOfBoxes(
            currBox,
            rectToBox({ // Convert the node's dimensions to a box
              x,
              y,
              width: node.width || 0,
              height: node.height || 0,
            })
        );
      },
      { x: Infinity, y: Infinity, x2: -Infinity, y2: -Infinity } // Initial box with infinite dimensions
  );

  return boxToRect(box); // Convert the final box to a rectangle
};