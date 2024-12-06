import React, { memo, ReactNode, useEffect, useRef } from 'react';
import { Handle, NodeProps, NodeToolbar, Position } from '@xyflow/react';
import Chip from '@mui/material/Chip';
import { useTheme } from '@mui/material/styles';

// Define the type for the data passed to the GraphNode component
export type GraphNodeData = {
  title: string; // Title of the node
  label: string; // Label of the node
  icon?: ReactNode; // Optional icon for the node
  subline?: string; // Optional subline for the node
  sdgId?: string; // Optional SDG ID for the node
  toolbarEnabled?: boolean; // Flag to control the visibility of the toolbar
};

// Style object for the NodeToolbar component
const toolbarStyle = {
  backgroundColor: '#b39ddb',
  color: 'black',
  padding: '8px',
  border: '2px solid #673ab7',
  borderRadius: '8px',
  boxShadow: '10px 0 15px rgba(42, 138, 246, 0.3)',
  width: '350px',
  height: 'auto',
  position: 'relative',
  fontFamily: 'Monospace',
  fontWeight: 500,
  letterSpacing: '-0.2px',
  fontSize: 12,
};

// Style object for the title of the node
const titleStyle = {
  fontFamily: 'Monospace',
  fontWeight: 500,
  position: 'absolute',
  color: 'white', // Default color, will be overridden by theme
  left: '100%',
  marginLeft: '10px',
  fontSize: 12,
  textWrap: 'break-word',
  lineHeight: 1,
  textOverflow: 'ellipsis',
  width: '250px',
  display: 'flex',
  justifyContent: 'space-between',
};

// GraphNode component, memoized to improve performance
const GraphNode = memo(function GraphNode({ data }: NodeProps<GraphNodeData>) {
  const theme = useTheme(); // Access the theme to get dynamic colors

  // Compute the title style by merging the default titleStyle with the color from the theme
  const computedTitleStyle = { ...titleStyle, color: theme.palette.text.primary };

  return (
      <>
        {/* Wrapper div for the node with a gradient */}
        <div className="graphwrapper graphgradient">
          {/* NodeToolbar component to display additional information on hover */}
          <NodeToolbar
              isVisible={data.toolbarEnabled || false} // Show toolbar if toolbarEnabled is true
              style={toolbarStyle}
              position={Position.Left}
              offset={30}
          >
            <div style={{ position: 'relative', zIndex: 1 }}>
              {data.id} (Target) : {data.subline} {/* Display the node ID and subline */}
            </div>
          </NodeToolbar>

          {/* Inner div for the node content */}
          <div className="inner">
            <div className="body">
              {/* Handles for connecting edges */}
              <Handle type="target" position={Position.Left} />
              <Handle type="source" position={Position.Right} />
            </div>
          </div>
        </div>

        {/* Div to display the title of the node with dynamic color */}
        <div style={computedTitleStyle}>
          {data.title ? data.title : data.label} {/* Display title or label if title is not available */}

          {/* Chip to display "SDG Target" */}
          <Chip
              label="SDG Target"
              color="primary"
              sx={{
                width: '70px',
                height: '15px',
                textAlign: 'center',
                fontSize: '8px',
              }}
          />
        </div>
      </>
  );
});

export default GraphNode;