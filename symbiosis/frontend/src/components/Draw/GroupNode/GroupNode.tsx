import { memo } from 'react';
import {
  NodeProps,
  NodeToolbar,
  useReactFlow,
  useStore,
  useStoreApi,
  NodeResizer,
} from '@xyflow/react';

import useDetachNodes from '../detach_utils';
import { getRelativeNodesBounds } from '../node_utils';

// Style for the resizer lines
const lineStyle = { borderColor: 'white' };

// GroupNode component for rendering a group node in the flow diagram
function GroupNode({ id }: NodeProps) {
  const store = useStoreApi(); // Access the ReactFlow store API
  const { deleteElements } = useReactFlow(); // Get the deleteElements function from useReactFlow
  const detachNodes = useDetachNodes(); // Custom hook for detaching nodes

  // Calculate the minimum width and height of the group node based on its child nodes
  const { minWidth, minHeight, hasChildNodes } = useStore(
      (store) => {
        const childNodes = Array.from(store.nodeLookup.values()).filter(
            (n) => n.parentId === id
        );
        const rect = getRelativeNodesBounds(childNodes); // Calculate the bounds of the child nodes

        return {
          minWidth: rect.x + rect.width + 10,
          minHeight: rect.y + rect.height + 10,
          hasChildNodes: childNodes.length > 0,
        };
      },
      isEqual // Custom comparison function for memoization
  );

  // Callback function to delete the group node
  const onDelete = () => {
    deleteElements({ nodes: [{ id }] });
  };

  // Callback function to detach all child nodes from the group node
  const onDetach = () => {
    const childNodeIds = Array.from(store.getState().nodeLookup.values())
        .filter((n) => n.parentId === id)
        .map((n) => n.id);

    detachNodes(childNodeIds, id);
  };

  return (
      <div>
        {/* NodeResizer for resizing the group node */}
        <NodeResizer
            lineStyle={lineStyle}
            minHeight={minHeight}
            minWidth={minWidth}
        />

        {/* NodeToolbar for the delete and ungroup buttons */}
        <NodeToolbar className="nodrag">
          <button onClick={onDelete}>Delete</button> {/* Delete button */}
          {hasChildNodes && <button onClick={onDetach}>Ungroup</button>} {/* Ungroup button (only if the group has child nodes) */}
        </NodeToolbar>
      </div>
  );
}

// Type for the object used in the isEqual comparison function
type IsEqualCompareObj = {
  minWidth: number;
  minHeight: number;
  hasChildNodes: boolean;
};

// Custom comparison function for memoizing the useStore hook
function isEqual(prev: IsEqualCompareObj, next: IsEqualCompareObj): boolean {
  return (
      prev.minWidth === next.minWidth &&
      prev.minHeight == next.minHeight &&
      prev.hasChildNodes === next.hasChildNodes
  );
}

export default memo(GroupNode);