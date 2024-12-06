import {
  useNodes,
  Node,
  NodeToolbar,
  useStoreApi,
  useReactFlow,
  getNodesBounds,
} from '@xyflow/react';

import { getId } from '../node_utils';

const padding = 25; // Padding to add around the grouped nodes

// SelectedNodesToolbar component for grouping selected nodes in the flow diagram
export default function SelectedNodesToolbar() {
  const nodes = useNodes(); // Access all nodes in the ReactFlow instance
  const { setNodes } = useReactFlow(); // Get the setNodes function from useReactFlow
  const store = useStoreApi(); // Access the ReactFlow store API

  // Filter the selected nodes and exclude any nodes that have a parent (already grouped)
  const selectedNodes = nodes.filter((node) => node.selected && !node.parentId);
  const selectedNodeIds = selectedNodes.map((node) => node.id); // Get the IDs of the selected nodes
  const isVisible = selectedNodeIds.length > 1; // Show the toolbar only if more than one node is selected

  // Callback function to group the selected nodes
  const onGroup = () => {
    const rectOfNodes = getNodesBounds(selectedNodes); // Get the bounding rectangle of the selected nodes
    const groupId = getId('group'); // Generate a unique ID for the group node

    // Calculate the position of the group node
    const parentPosition = {
      x: rectOfNodes.x,
      y: rectOfNodes.y,
    };

    // Create the group node object
    const groupNode = {
      id: groupId,
      type: 'group',
      position: parentPosition,
      style: {
        width: rectOfNodes.width + padding * 2,
        height: rectOfNodes.height + padding * 2,
      },
      data: {},
    };

    // Update the positions of the selected nodes to be relative to the group node
    const nextNodes: Node[] = nodes.map((node) => {
      if (selectedNodeIds.includes(node.id)) {
        return {
          ...node,
          position: {
            x: node.position.x - parentPosition.x + padding,
            y: node.position.y - parentPosition.y + padding,
          },
          extent: 'parent',
          parentId: groupId,
        };
      }

      return node;
    });

    // Reset the selected elements and update the nodes state with the group node and the updated child nodes
    store.getState().resetSelectedElements();
    store.setState({ nodesSelectionActive: false });
    setNodes([groupNode, ...nextNodes]);
  };

  return (
      // Render the NodeToolbar with the "Group selected nodes" button
      <NodeToolbar nodeId={selectedNodeIds} isVisible={isVisible}>
        <button onClick={onGroup}>Group selected nodes</button>
      </NodeToolbar>
  );
}