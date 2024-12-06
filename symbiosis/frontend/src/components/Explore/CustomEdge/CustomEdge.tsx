import React from 'react';
import { EdgeProps, getBezierPath } from '@xyflow/react';

// Component to render a custom edge in the React Flow diagram
function CustomEdge({
                      id, // Unique identifier for the edge
                      sourceX, // X coordinate of the source node
                      sourceY, // Y coordinate of the source node
                      targetX, // X coordinate of the target node
                      targetY, // Y coordinate of the target node
                      sourcePosition, // Position of the source handle ('top', 'bottom', 'left', 'right')
                      targetPosition, // Position of the target handle ('top', 'bottom', 'left', 'right')
                      style = {}, // Custom style object for the edge
                      markerEnd, // Marker to be displayed at the end of the edge
                    }: EdgeProps) {

  // Check if the source and target nodes have the same X or Y coordinates
  const xEqual = sourceX === targetX;
  const yEqual = sourceY === targetY;

  // Calculate the path for the edge using the getBezierPath function
  const [edgePath] = getBezierPath({
    // Add a small offset to the source coordinates if they are equal to the target coordinates
    // This is a workaround to display the gradient correctly for straight lines
    sourceX: xEqual ? sourceX + 0.0001 : sourceX,
    sourceY: yEqual ? sourceY + 0.0001 : sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return 1  (
      <>
        {/* Render the edge as an SVG path */}
        <path
            id={id}
            style={style}
            className="react-flow__edge-path"
            d={edgePath} // Path definition for the edge
            markerEnd={markerEnd} // Marker for the end of the edge
        />
      </>
  );
}

export default CustomEdge;