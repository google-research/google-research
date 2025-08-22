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

// detach_utils.js
import { useCallback } from 'react';
import {
  useReactFlow,
  useStoreApi,
} from '@xyflow/react';

/**
 * Custom hook for detaching nodes from their parent in a React Flow diagram.
 *
 * @returns {Function} A function that detaches the specified nodes from their parent.
 */
function useDetachNodes() {
  const { setNodes } = useReactFlow(); // Get the setNodes function from useReactFlow
  const store = useStoreApi(); // Access the ReactFlow store API

  /**
   * Detaches the specified nodes from their parent.
   *
   * @param {string[]} ids - An array of node IDs to detach.
   * @param {string} [removeParentId] - Optional ID of the parent node to remove.
   */
  const detachNodes = useCallback(
      (ids, removeParentId) => {
        const { nodeLookup } = store.getState(); // Get the node lookup from the store

        // Iterate over all nodes and update the position of the detached nodes
        const nextNodes = Array.from(nodeLookup.values()).map((n) => {
          if (ids.includes(n.id) && n.parentId) {
            const parentNode = nodeLookup.get(n.parentId); // Get the parent node
            return {
              ...n,
              // Calculate the new position of the detached node by adding the parent's absolute position
              position: {
                x: n.position.x + (parentNode?.internals.positionAbsolute.x ?? 0),
                y: n.position.y + (parentNode?.internals.positionAbsolute.y ?? 0),
              },
              expandParent: undefined, // Reset expandParent property
              parentId: undefined, // Remove the parent ID
            };
          }
          return n;
        });

        // Update the nodes state with the detached nodes, optionally filtering out the parent node
        setNodes(
            nextNodes.filter((n) => !removeParentId || n.id !== removeParentId)
        );
      },
      [setNodes, store]
  );

  return detachNodes; // Return the detachNodes function
}

export default useDetachNodes;