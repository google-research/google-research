import React, { memo, ReactNode, useEffect, useRef } from 'react';
import { Handle, NodeProps, NodeToolbar, Position } from '@xyflow/react';
import { SiUnitednations } from 'react-icons/si';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Chip from '@mui/material/Chip';

// Define the type for the data passed to the FocusNode component
export type FocusNodeData = {
  title: string; // Title of the node
  icon?: ReactNode; // Optional icon for the node
  subline?: string; // Optional subline for the node
  id?: string; // Optional ID for the node
};

// FocusNode component, memoized to improve performance
const FocusNode = memo(function FocusNode({ data }: NodeProps<FocusNodeData>) {
  const [expanded, setExpanded] = React.useState(false); // State to control the expansion of the node

  return (
      <>
        {/* Outer div with a cloud shape and gradient */}
        <div className="cloud gradient">
          <div>
            <SiUnitednations /> {/* Icon for the United Nations */}
          </div>
        </div>

        {/* Wrapper div with a gradient */}
        <div className="wrapper gradient">
          {/* Inner div with conditional class for expansion */}
          <div className={`inner ${expanded ? 'expanded' : ''}`}>
            <div className="body">
              {data.icon && <div className="icon">{data.icon}</div>} {/* Render the icon if provided */}
              {/* Render the title with word wrap */}
              <div className="title" style={{ wordWrap: 'break-word' }}>{data.title}</div>
            </div>

            {/* SVG element to define a linear gradient */}
            <svg width={0} height={0}>
              <linearGradient id="linearColors" x1={0} y1={0} x2={1} y2={1}>
                <stop offset={0} stopColor="rgba(78, 91, 135, 1)" />
                <stop offset={1} stopColor="rgba(174, 190, 241, 1)" />
              </linearGradient>
            </svg>

            {/* Div to hold the expand icon, chip, and subline */}
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              {/* Expand icon with conditional rotation and transition */}
              <ExpandMoreIcon
                  sx={{
                    fontSize: 20,
                    fill: "url(#linearColors)", // Use the defined linear gradient
                    mt: "10px",
                    transform: expanded ? 'rotate(180deg)' : 'none', // Rotate icon if expanded
                    transition: 'transform 0.3s ease', // Add a transition effect
                  }}
                  onClick={() => setExpanded(!expanded)} // Toggle the expanded state on click
              />

              {/* Chip to display "SDG Goal" */}
              <Chip
                  label="SDG Goal"
                  color="primary"
                  sx={{
                    width: '60px',
                    height: '15px',
                    textAlign: 'center',
                    fontSize: '8px',
                  }}
              />

              {/* Conditionally render the subline if expanded */}
              {expanded &&
                  <div className="subline expanded-content">
                    {data.id} (Goal) : {data.subline}
                  </div>
              }
            </div>

            {/* Handles for connecting edges */}
            <Handle type="target" position={Position.Left} />
            <Handle type="source" position={Position.Right} />
          </div>
        </div>
      </>
  );
});

export default FocusNode;