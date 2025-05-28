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

// edge_utils.js
import { Position, MarkerType } from '@xyflow/react';

/**
 * Calculates the intersection point of an edge with a node's border.
 *
 * @param {Node} intersectionNode - The node whose border the edge intersects.
 * @param {Node} targetNode - The node the edge is connected to.
 * @returns {{ x: number, y: number }} The coordinates of the intersection point.
 */
function getNodeIntersection(intersectionNode, targetNode) {
  const { width: intersectionNodeWidth, height: intersectionNodeHeight } =
      intersectionNode.measured; // Get the dimensions of the intersection node
  const intersectionNodePosition = intersectionNode.internals.positionAbsolute; // Get the absolute position of the intersection node
  const targetPosition = targetNode.internals.positionAbsolute; // Get the absolute position of the target node

  // Calculate the center coordinates of the intersection node
  const w = intersectionNodeWidth / 2;
  const h = intersectionNodeHeight / 2;

  // Calculate the coordinates of the center of the intersection node
  const x2 = intersectionNodePosition.x + w;
  const y2 = intersectionNodePosition.y + h;

  // Calculate the coordinates of the center of the target node
  const x1 = targetPosition.x + targetNode.measured.width / 2;
  const y1 = targetPosition.y + targetNode.measured.height / 2;

  // Calculate intermediate values for finding the intersection point
  const xx1 = (x1 - x2) / (2 * w) - (y1 - y2) / (2 * h);
  const yy1 = (x1 - x2) / (2 * w) + (y1 - y2) / (2 * h);
  const a = 1 / (Math.abs(xx1) + Math.abs(yy1));
  const xx3 = a * xx1;
  const yy3 = a * yy1;

  // Calculate the x and y coordinates of the intersection point
  const x = w * (xx3 + yy3) + x2;
  const y = h * (-xx3 + yy3) + y2;

  return { x, y }; // Return the intersection point coordinates
}

/**
 * Determines the position of the edge handle relative to the node.
 *
 * @param {Node} node - The node the edge is connected to.
 * @param {{ x: number, y: number }} intersectionPoint - The intersection point of the edge with the node's border.
 * @returns {Position} The position of the edge handle (top, right, bottom, or left).
 */
function getEdgePosition(node, intersectionPoint) {
  const n = { ...node.internals.positionAbsolute, ...node }; // Combine node position and dimensions
  const nx = Math.round(n.x); // Round the x coordinate of the node
  const ny = Math.round(n.y); // Round the y coordinate of the node
  const px = Math.round(intersectionPoint.x); // Round the x coordinate of the intersection point
  const py = Math.round(intersectionPoint.y); // Round the y coordinate of the intersection point

  // Determine the edge position based on the intersection point relative to the node's center
  if (px <= nx + 1) {
    return Position.Left;
  }
  if (px >= nx + n.measured.width - 1) {
    return Position.Right;
  }
  if (py <= ny + 1) {
    return Position.Top;
  }
  if (py >= n.y + n.measured.height - 1) {
    return Position.Bottom;
  }

  return Position.Top; // Default to Position.Top if no other condition is met
}

/**
 * Calculates the parameters needed to create an edge between two nodes.
 *
 * @param {Node} source - The source node of the edge.
 * @param {Node} target - The target node of the edge.
 * @returns {{ sx: number, sy: number, tx: number, ty: number, sourcePos: Position, targetPos: Position }} The edge parameters.
 */
export function getEdgeParams(source, target) {
  const sourceIntersectionPoint = getNodeIntersection(source, target); // Get the intersection point on the source node
  const targetIntersectionPoint = getNodeIntersection(target, source); // Get the intersection point on the target node

  const sourcePos = getEdgePosition(source, sourceIntersectionPoint); // Get the edge position on the source node
  const targetPos = getEdgePosition(target, targetIntersectionPoint); // Get the edge position on the target node

  // Return the parameters needed to create the edge
  return {
    sx: sourceIntersectionPoint.x, // Source x coordinate
    sy: sourceIntersectionPoint.y, // Source y coordinate
    tx: targetIntersectionPoint.x, // Target x coordinate
    ty: targetIntersectionPoint.y, // Target y coordinate
    sourcePos, // Source position
    targetPos, // Target position
  };
}

/**
 * Creates an example set of nodes and edges for a circular layout.
 *
 * @returns {{ nodes: Node[], edges: Edge[] }} An object containing the nodes and edges.
 */
export function createNodesAndEdges() {
  const nodes = [];
  const edges = [];
  const center = { x: window.innerWidth / 2, y: window.innerHeight / 2 }; // Calculate the center of the window

  // Add a target node at the center
  nodes.push({ id: 'target', data: { label: 'Target' }, position: center });

  // Add source nodes in a circle around the target node
  for (let i = 0; i < 8; i++) {
    const degrees = i * (360 / 8); // Calculate the angle in degrees
    const radians = degrees * (Math.PI / 180); // Convert degrees to radians
    const x = 250 * Math.cos(radians) + center.x; // Calculate the x coordinate of the source node
    const y = 250 * Math.sin(radians) + center.y; // Calculate the y coordinate of the source node

    nodes.push({ id: `${i}`, data: { label: 'Source' }, position: { x, y } }); // Add the source node

    edges.push({
      id: `edge-${i}`, // Generate a unique ID for the edge
      target: 'target', // Connect the edge to the target node
      source: `${i}`, // Connect the edge to the source node
      type: 'floating',
      markerEnd: {
        type: MarkerType.Arrow, // Add an arrow marker to the end of the edge
      },
    });
  }

  return { nodes, edges }; // Return the nodes and edges
}